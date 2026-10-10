"""Scene-referred tone stage of process version 2 (1.6.0).

Process version 1 (1.0–1.5) applied the camera develop curve first and then exposure, the
four tone sliders and contrast to the result, clipping every channel at 1: highlights the
curve had already compressed could not be brought back by lowering exposure, and saturated
colours were clipped channel by channel.  Process version 2 works on the unclipped linear
decode instead:

1. white-balance fine tuning and exposure in scene-linear light;
2. shadows / highlights as a *local* tone mapping: the gain follows an edge-aware smoothing
   of log luminance (a fast guided filter computed once per photo at low resolution), so
   texture inside the shadows or the sky keeps its contrast and strong edges get no halo;
   blacks / whites stay a per-pixel (global) tone curve;
3. the develop curve (camera reference or linear) as the tone map, applied to luminance so
   hue and saturation are kept; luminance above white is limited to white;
4. contrast, then gamut mapping into sRGB at constant OkLab lightness and hue (``color``).

The low-resolution guided-filter coefficients (``Context``) are computed from the whole frame
and resampled for any block, so a zoomed tile matches the whole-frame render exactly.
"""
from __future__ import annotations

import threading
from functools import lru_cache
from typing import NamedTuple

import cv2
import numpy as np

from . import color
from .curves import IDENTITY, evaluate

TABLE = 65537                     # samples of the tone map over scene luminance 0…HEADROOM
HEADROOM = 4.                     # two stops above white: the shoulder's range
SHOULDER = .92                    # display level where the camera curve hands over to the shoulder
CONTEXT_EDGE = 512                # long edge of the local tone-mapping grid
CONTEXT_EPS = .2                  # guided-filter ε in stops²: steps above ~0.45 stop are kept
FLOOR = 2. ** -16                 # luminance floor for log2


class Context(NamedTuple):
    """Guided-filter coefficients over the whole frame: base(log2 Y) = a · log2 Y + b."""
    a: np.ndarray
    b: np.ndarray


class Spec(NamedTuple):
    """How ``tonal`` maps a block: tone table, local context and the block's place in the frame."""
    table: np.ndarray
    context: Context | None = None
    area: object = None           # engine.Area of the block, None for a whole frame


def _curve_key(develop):
    if develop.get('mode') != 'camera' or develop.get('curve', IDENTITY) == IDENTITY:
        return None
    return tuple(tuple(map(float, p)) for p in develop['curve'])


@lru_cache(maxsize=16)
def _table(curve):
    """Display-linear luminance over scene luminance 0…HEADROOM.

    Linear develop (also every JPEG / TIFF source) is the identity, limited to white.  The camera
    reference curve hands over to a filmic shoulder where it reaches ``SHOULDER``: same value and
    slope, then an exponential approach to white, so luminance pushed past white (exposure,
    whites, highlights recovered from the sensor) rolls off instead of clipping flat."""
    axis = np.linspace(0, HEADROOM, TABLE)
    if curve is None:
        values = np.minimum(axis, 1)
    else:
        values = evaluate([list(p) for p in curve], np.minimum(axis, 1))
        fine = np.linspace(0, 1, 16385)
        curve_values = evaluate([list(p) for p in curve], fine)
        start = int(np.searchsorted(curve_values, SHOULDER))
        if 0 < start < len(fine) - 2:
            knee, level = fine[start], curve_values[start]
            slope = max((curve_values[start + 1] - curve_values[start - 1]) / (fine[start + 1] - fine[start - 1]), 1e-3)
            beyond = axis > knee
            values[beyond] = 1 - (1 - level) * np.exp(-slope * (axis[beyond] - knee) / (1 - level))
    table = np.asarray(values, np.float32)
    table.setflags(write=False)
    return table


def table(develop):
    """Display-linear luminance of scene luminance 0…HEADROOM for the develop settings."""
    return _table(_curve_key(develop or {}))


def identity():
    return _table(None)


def spec(edits, context=None, area=None):
    return Spec(table(edits.get('develop', {})), context, area)


def mask_spec():
    """Masks adjust the already tone-mapped picture: no tone curve, no local context."""
    return Spec(identity())


def lookup(table_, x):
    """Linear interpolation of ``table_`` over 0…HEADROOM (as the GPU graphs gather it)."""
    position = np.clip(x, 0, HEADROOM) * np.float32((TABLE - 1) / HEADROOM)
    index = np.minimum(position.astype(np.int32), TABLE - 2)
    fraction = position - index
    left = table_[index]
    return left + (table_[index + 1] - left) * fraction


def white_balance_gains(a):
    temp, tint = a['temperature'] / 100, a['tint'] / 100
    return np.array([2 ** (.4 * temp + .15 * tint), 2 ** (-.15 * tint), 2 ** (-.4 * temp + .15 * tint)], np.float32)


def needs_context(a):
    return bool(a.get('shadows', 0) or a.get('highlights', 0))


def context(balanced, a):
    """Local tone-mapping coefficients of a whole frame (``balanced``: scene-linear, after
    lens, retouching and white balance), or None when shadows and highlights are zero."""
    if not needs_context(a):
        return None
    h, w = balanced.shape[:2]
    if max(h, w) > CONTEXT_EDGE:
        ratio = CONTEXT_EDGE / max(h, w)
        small = cv2.resize(np.asarray(balanced, np.float32), (max(1, round(w * ratio)), max(1, round(h * ratio))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = np.asarray(balanced, np.float32)
    y = color.luminance(small * white_balance_gains(a))
    log = np.log2(np.maximum(y, FLOOR)).astype(np.float32)
    radius = max(2, round(min(log.shape) / 20))

    def box(v):
        return cv2.boxFilter(v, -1, (2 * radius + 1, 2 * radius + 1), borderType=cv2.BORDER_REFLECT)
    mean = box(log)
    variance = np.maximum(box(log * log) - mean * mean, 0)
    gain = variance / (variance + CONTEXT_EPS)
    offset = mean - gain * mean
    return Context(box(gain).astype(np.float32), box(offset).astype(np.float32))


_maps = {}
_maps_lock = threading.Lock()


def _resampled(ctx, shape, area):
    """(H, W, 2) of the context's a and b for a block; the same arithmetic for a whole frame
    and for any block of it, so tiles match the whole-frame render bit for bit."""
    from .engine import Area, resize_region
    h, w = shape[:2]
    area = area or Area(w, h, 0, 0, w, h)
    key = (id(ctx), tuple(area))
    with _maps_lock:
        entry = _maps.get(key)
    if entry is None or entry[0] is not ctx:
        size = (area.width, area.height)
        values = np.dstack([resize_region(ctx.a, size, area), resize_region(ctx.b, size, area)]).astype(np.float32)
        values.setflags(write=False)
        entry = (ctx, values)
        with _maps_lock:
            while len(_maps) >= 6:
                _maps.pop(next(iter(_maps)))
            _maps[key] = entry
    return entry[1]


def maps(spec_, shape, exposure):
    """(H, W, 2) coefficients for a block of ``shape``; exposure shifts log2 Y by a constant,
    which moves the filtered base by the same amount: b' = b + EV · (1 − a)."""
    values = _resampled(spec_.context, shape, spec_.area)
    if not exposure:
        return values
    shifted = values.copy()
    shifted[..., 1] += np.float32(exposure) * (1 - values[..., 0])
    return shifted


def _signed_srgb(x):
    return np.sign(x) * color.linear_to_srgb(np.abs(x))


def _signed_linear(x):
    return np.sign(x) * color.srgb_to_linear(np.abs(x))


def tonal(image, a, spec_):
    """NumPy reference of the process-2 tone stage: scene-linear in, display sRGB 0–1 out."""
    x = np.asarray(image, np.float32) * (white_balance_gains(a) * np.float32(2 ** a['exposure']))
    t = spec_.table
    y = color.luminance(x)
    stops = None
    if a['shadows'] or a['highlights']:
        if spec_.context is not None:
            coefficients = maps(spec_, x.shape, a['exposure'])
            base = np.exp2(coefficients[..., 0] * np.log2(np.maximum(y, FLOOR)) + coefficients[..., 1])
        else:
            base = np.maximum(y, FLOOR)
        p = np.clip(lookup(t, base), 0, 1) ** .45
        stops = (a['shadows'] * (1 - p) ** 2 + a['highlights'] * p ** 3) / 65
    if a['blacks'] or a['whites']:
        p = np.clip(lookup(t, y), 0, 1) ** .45
        ends = (a['blacks'] * (1 - p) ** 6 + a['whites'] * p ** 6) / 85
        stops = ends if stops is None else stops + ends
    if stops is not None:
        x *= np.exp2(stops)[..., None]
        y = color.luminance(x)
    x *= (lookup(t, y) / np.maximum(y, 1e-7))[..., None]
    if a['contrast']:
        x = _signed_linear((_signed_srgb(x) - .5) * np.float32(1 + a['contrast'] / 125) + .5)
    return color.linear_to_srgb(color.gamut_map(x))


def develop_view(source, develop):
    """The developed starting point (before / after comparison, masks, thumbnails)."""
    x = np.asarray(source, np.float32)
    y = color.luminance(x)
    return color.linear_to_srgb(color.gamut_map(x * (lookup(table(develop), y) / np.maximum(y, 1e-7))[..., None]))


def auto_tone(source, develop, gains=None):
    """Exposure and tone sliders for process 2: the median lands near display mid grey."""
    from .engine import resize_limit
    sample = resize_limit(source, 500)
    if gains is not None:
        sample = sample * np.asarray(gains, np.float32)
    y = color.luminance(sample)
    dark, mid, light = np.percentile(y, [5, 50, 98])
    if light < 1e-5:
        return dict(exposure=0., shadows=0., highlights=0., blacks=0., whites=0., contrast=0.)
    t = table(develop)
    low, high = -2.5, 2.5
    for _ in range(24):
        middle = (low + high) / 2
        low, high = (middle, high) if lookup(t, np.float32(max(mid, 1e-6) * 2 ** middle)) < .18 else (low, middle)
    exposure = min((low + high) / 2, float(np.log2(1.6 / max(light, .01))))
    high_after = float(lookup(t, np.float32(light * 2 ** exposure))) + max(0., light * 2 ** exposure - 1)
    low_after = float(lookup(t, np.float32(dark * 2 ** exposure)))
    return dict(exposure=round(float(exposure), 2), shadows=round(float(np.clip((.05 - low_after) * 550, 0, 40))),
                highlights=round(float(np.clip((.85 - high_after) * 90, -65, 0))),
                blacks=-5., whites=0., contrast=5.)
