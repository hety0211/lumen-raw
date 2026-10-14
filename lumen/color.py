"""Perceptual colour for process version 2 (1.6.0): OkLab / OkLCh and gamut mapping.

Process version 2 keeps colours outside sRGB as negative or above-one linear values until the
end of the tone stage, then maps them into sRGB by lowering chroma at constant OkLab lightness
and hue instead of clipping each channel (which shifts hue: a saturated sunset turns yellow,
a deep blue sky turns cyan-purple).  Saturation, vibrance, the 8-colour mixer and colour
grading work in OkLCh (Björn Ottosson's OkLab, 2020), where raising the chroma of a blue sky
does not drift towards purple and does not change its lightness.

Every function here has a matching ONNX graph (``gpu_graphs``) and Metal kernel (``metal``);
this NumPy code is the reference both are tested against.
"""
from __future__ import annotations

import colorsys
from functools import lru_cache
from typing import NamedTuple

import numpy as np

from .model import COLORS

#: Linear sRGB → LMS and LMS' (cube root) → OkLab, and their inverses (Ottosson 2020).
RGB_TO_LMS = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                       [0.2119034982, 0.6806995451, 0.1073969566],
                       [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
LMS_TO_LAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                       [1.9779984951, -2.4285922050, 0.4505937099],
                       [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)
LAB_TO_LMS = np.array([[1., 0.3963377774, 0.2158037573],
                       [1., -0.1055613458, -0.0638541728],
                       [1., -0.0894841775, -1.2914855480]], np.float32)
LMS_TO_RGB = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                       [-1.2684380046, 2.6097574011, -0.3413193965],
                       [-0.0041960863, -0.7034186147, 1.7076147010]], np.float32)
LUMA = np.array([.2126, .7152, .0722], np.float32)

#: Bisection steps of the gamut mapping: chroma within 2^-10 of the sRGB boundary.
GAMUT_STEPS = 10
#: A colour counts as inside sRGB with this tolerance (float32 round trips).
GAMUT_EPS = 1e-6
#: Gamut mapping trades lightness for chroma by this much (Ottosson's adaptive L0, α = 0.4):
#: too-bright saturated colours get slightly darker instead of collapsing to near white.
GAMUT_ALPHA = .4
#: OkLab chroma of the most saturated sRGB colours is 0.26–0.32.
CHROMA_FULL = .3
#: Below this chroma hue-dependent lightness changes fade out (greys have no hue).
CHROMA_HUE = .04
#: Mixer lightness: ±100 scales OkLab L by 2^±LIGHTNESS (≈ the 1.x HSV value change).
LIGHTNESS = .66
#: Grading strength 100 adds this much OkLab chroma (process 1 averages 0.098 on mid grey).
GRADE_CHROMA = .1
HUE_SAMPLES = 361


def _rgb_to_xyz(primaries, white=(.3127, .3290)):
    """3 × 3 linear RGB → XYZ of an RGB space from its xy primaries and white point."""
    xy = np.asarray(primaries, np.float64)
    columns = np.stack([xy[:, 0] / xy[:, 1], np.ones(3), (1 - xy.sum(axis=1)) / xy[:, 1]])
    w = np.array([white[0] / white[1], 1, (1 - white[0] - white[1]) / white[1]])
    return columns * np.linalg.solve(columns, w)


class Space(NamedTuple):
    """An output RGB space: linear sRGB ↔ linear target matrices and its transfer curve.

    The pipeline keeps working in linear sRGB primaries (unclipped in process 2); only gamut
    mapping and the display encoding use the output space."""
    key: str
    label: str
    to_srgb: np.ndarray            # linear target → linear sRGB
    from_srgb: np.ndarray          # linear sRGB → linear target
    to_xyz: np.ndarray             # linear target → XYZ (D65)
    gamma: float = 0.              # 0: the sRGB curve; otherwise a pure power law


def _space(key, label, primaries, gamma=0.):
    srgb = _rgb_to_xyz([(.64, .33), (.30, .60), (.15, .06)])
    target = _rgb_to_xyz(primaries)
    to_srgb = np.linalg.solve(srgb, target)
    return Space(key, label, to_srgb.astype(np.float32), np.linalg.inv(to_srgb).astype(np.float32),
                 target.astype(np.float32), gamma)


#: Export colour spaces: Display P3 uses the sRGB curve, Adobe RGB (1998) a 563/256 gamma.
SPACES = {
    'srgb': _space('srgb', 'sRGB', [(.64, .33), (.30, .60), (.15, .06)]),
    'p3': _space('p3', 'Display P3', [(.680, .320), (.265, .690), (.150, .060)]),
    'adobe-rgb': _space('adobe-rgb', 'Adobe RGB (1998)', [(.64, .33), (.21, .71), (.15, .06)], 563 / 256),
}


def space(key):
    """The output space for a key (None: sRGB)."""
    return SPACES[key or 'srgb'] if not isinstance(key, Space) else key


def wide(value):
    """True for an output space other than sRGB."""
    return value is not None and space(value).key != 'srgb'


def encode(linear, value=None):
    """Linear values of the output space → its display encoding (negatives clamp to 0)."""
    if not wide(value) or not space(value).gamma:
        return linear_to_srgb(linear)
    return (np.maximum(np.asarray(linear, np.float32), 0) ** np.float32(1 / space(value).gamma)).astype(np.float32)


def decode(encoded, value=None):
    """Display encoding of the output space → its linear values."""
    if not wide(value) or not space(value).gamma:
        return srgb_to_linear(encoded)
    return (np.maximum(np.asarray(encoded, np.float32), 0) ** np.float32(space(value).gamma)).astype(np.float32)


def to_linear_srgb(encoded, value=None):
    """Display encoding of the output space → linear sRGB primaries (unclipped)."""
    linear = decode(encoded, value)
    return linear if not wide(value) else (linear @ space(value).to_srgb.T).astype(np.float32)


def reencode(srgb, value=None):
    """Display sRGB → display values of the output space (sRGB fits inside P3 and Adobe RGB)."""
    if not wide(value):
        return srgb
    linear = srgb_to_linear(np.clip(srgb, 0, 1)) @ space(value).from_srgb.T
    return encode(np.clip(linear, 0, 1), value)


def srgb_to_linear(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= .04045, x / 12.92, np.maximum((x + .055) / 1.055, 0) ** 2.4).astype(np.float32)


def linear_to_srgb(x):
    x = np.maximum(np.asarray(x, np.float32), 0)
    return np.where(x <= .0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - .055).astype(np.float32)


def to_oklab(rgb):
    """Linear sRGB (any range) → OkLab, channel last."""
    lms = np.asarray(rgb, np.float32) @ RGB_TO_LMS.T
    return np.cbrt(lms) @ LMS_TO_LAB.T


def from_oklab(lab):
    """OkLab → linear sRGB (unclipped)."""
    lms = np.asarray(lab, np.float32) @ LAB_TO_LMS.T
    return (lms * lms * lms) @ LMS_TO_RGB.T


def _inside(rgb):
    return (rgb.min(axis=-1) >= -GAMUT_EPS) & (rgb.max(axis=-1) <= 1 + GAMUT_EPS)


def _excess(rgb):
    """How far a colour lies outside the sRGB cube (≤ 0 inside)."""
    return np.maximum(rgb.max(axis=-1) - 1, -rgb.min(axis=-1))


def anchor(lightness, chroma):
    """Grey lightness a colour is pulled towards (Ottosson's adaptive L0 around 0.5): its own
    lightness for greys, nearer mid grey the more chroma it has; 1 for any grey above white."""
    offset = lightness - .5
    e1 = .5 + np.abs(offset) + GAMUT_ALPHA * chroma
    return .5 * (1 + np.sign(offset) * (e1 - np.sqrt(np.maximum(e1 * e1 - 2 * np.abs(offset), 0))))


def gamut_map(rgb, value=None):
    """Linear sRGB → [0, 1]³ of the output space ``value`` (default sRGB) in that space's linear
    values: colours outside keep their OkLab hue and move in a straight line
    towards a grey of similar lightness (``anchor``) until they fit, so they lose chroma and,
    when too bright or too dark for their chroma, a little lightness.  Colours inside are
    returned unchanged.

    The position on the line is bracketed by bisection and then interpolated linearly between
    the bracket ends (one secant step), so the result is continuous in its input: a block render
    and the whole-frame render agree although float rounding may flip a bisection step."""
    rgb = np.asarray(rgb, np.float32)
    flat = rgb.reshape(-1, 3)
    if wide(value):
        matrix = space(value).from_srgb.T
        target = lambda v: v @ matrix
    else:
        target = lambda v: v
    outside = ~_inside(target(flat))
    if not outside.any():
        return np.clip(target(rgb), 0, 1)
    result = flat.copy()
    lab = to_oklab(flat[outside])
    lightness, ab = lab[:, :1], lab[:, 1:]
    grey = anchor(lightness, np.sqrt((ab * ab).sum(axis=1, keepdims=True))).astype(np.float32)
    span = lightness - grey

    def at(t):
        return from_oklab(np.concatenate([grey + span * t, ab * t], axis=1))
    low = np.zeros((len(lab), 1), np.float32)
    high = np.ones((len(lab), 1), np.float32)
    excess_low, excess_high = _excess(target(at(low)))[:, None], _excess(target(at(high)))[:, None]
    for _ in range(GAMUT_STEPS):
        middle = (low + high) * np.float32(.5)
        excess = _excess(target(at(middle)))[:, None]
        fits = excess <= GAMUT_EPS
        low, excess_low = np.where(fits, middle, low), np.where(fits, excess, excess_low)
        high, excess_high = np.where(fits, high, middle), np.where(fits, excess_high, excess)
    step = np.clip(-excess_low / np.maximum(excess_high - excess_low, np.float32(1e-12)), 0, 1)
    result[outside] = at(low + (high - low) * step)
    return np.clip(target(result).reshape(rgb.shape), 0, 1)


def luminance(rgb):
    return rgb[..., 0] * LUMA[0] + rgb[..., 1] * LUMA[1] + rgb[..., 2] * LUMA[2]


# --------------------------------------------------------------------- hue correspondence

@lru_cache(maxsize=None)
def ok_hue(hsv_hue):
    """OkLCh hue (degrees) of the fully saturated sRGB colour at an HSV hue."""
    rgb = srgb_to_linear(np.array(colorsys.hsv_to_rgb((hsv_hue % 360) / 360, 1, 1), np.float32))
    _, a, b = to_oklab(rgb[None])[0]
    return float(np.degrees(np.arctan2(b, a)) % 360)


def _centers():
    """HSV centres of the mixer colours and the OkLCh hues of the same colours, unwrapped."""
    hsv = [float(c[1]) for c in COLORS]
    ok = [ok_hue(h) for h in hsv]
    for i in range(1, len(ok)):
        while ok[i] <= ok[i - 1]:
            ok[i] += 360
    # Close the circle: the first centre again, one turn later.
    return np.array(hsv + [hsv[0] + 360]), np.array(ok + [ok[0] + 360])


def hsv_from_ok(h):
    """Map OkLCh hue to the HSV hue scale of the mixer (piecewise linear between centres)."""
    hsv, ok = _centers()
    h = (np.asarray(h, np.float64) - ok[0]) % 360 + ok[0]
    return np.interp(h, ok, hsv) % 360


def ok_from_hsv(h):
    hsv, ok = _centers()
    h = (np.asarray(h, np.float64) - hsv[0]) % 360 + hsv[0]
    return np.interp(h, hsv, ok) % 360


def mixer_tables(hsl):
    """Per OkLCh degree (0…360): hue shift in degrees, chroma factor − 1, lightness stops.

    The controls keep their 1.x meaning: each colour's sliders act on that colour and fade
    smoothly into its neighbours, and hue +100 moves a colour 45° along the HSV hue circle
    (red → orange → yellow …), now measured between the same colours in OkLCh."""
    axis = np.arange(HUE_SAMPLES, dtype=np.float64)
    centers = [c[1] for c in COLORS] + [360]
    controls = np.asarray(list(hsl) + [hsl[0]], dtype=np.float64)
    virtual = hsv_from_ok(axis)
    shifted = ok_from_hsv(virtual + np.interp(virtual, centers, controls[:, 0]) * .45)
    shift = (shifted - axis + 180) % 360 - 180
    chroma = np.interp(virtual, centers, controls[:, 1]) / 100
    light = np.interp(virtual, centers, controls[:, 2]) / 100 * LIGHTNESS
    return shift.astype(np.float32), chroma.astype(np.float32), light.astype(np.float32)


@lru_cache(maxsize=1)
def vibrance_weights():
    """Vibrance acts less on skin hues (orange, OkLCh ≈ 40–75°)."""
    axis = np.arange(HUE_SAMPLES, dtype=np.float64)
    distance = (axis - 58 + 180) % 360 - 180
    return (1 - .5 * np.exp(-(distance / 22) ** 2)).astype(np.float32)


def _lookup(table, hue):
    """Linear interpolation into a 1° table, as the GPU graphs do."""
    position = np.clip(hue, 0, 360)
    index = np.clip(np.floor(position), 0, HUE_SAMPLES - 2).astype(np.int64)
    fraction = (position - index).astype(np.float32)
    return table[index] + (table[index + 1] - table[index]) * fraction


# --------------------------------------------------------------------- display stage

def grade_vectors(grading):
    """OkLab (a, b) tint of each zone at full weight."""
    vectors = []
    for zone in ('shadows', 'midtones', 'highlights'):
        hue, strength = grading.get(zone, [0, 0])
        angle = np.radians(ok_hue(float(hue)))
        vectors.append(np.array([np.cos(angle), np.sin(angle)], np.float32) * np.float32(strength / 100 * GRADE_CHROMA))
    return vectors


def graded(grading):
    return any(grading.get(z, [0, 0])[1] for z in ('shadows', 'midtones', 'highlights'))


def adjust(x, saturation=0., vibrance=0., hsl=(), mono=False, grading=None, value=None):
    """Saturation, vibrance, the 8-colour mixer, monochrome and colour grading on display values
    (0–1, gamma encoded in the output space ``value``, default sRGB), in one OkLCh pass followed
    by one gamut mapping into that space.

    Monochrome keeps OkLab lightness (a perceptual grey), grading adds its tint in OkLab (a, b)
    after it, so toned black and white works and lightness never changes."""
    grading = grading or {}
    mixer = bool(np.any(hsl))
    tint = graded(grading)
    if not (saturation or vibrance or mixer or mono or tint):
        return x
    lab = to_oklab(to_linear_srgb(x, value))
    lightness, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    if saturation or vibrance or mixer:
        chroma = np.sqrt(a * a + b * b)
        hue = np.degrees(np.arctan2(b, a)) % 360
        scale = np.full(chroma.shape, 1 + saturation / 100, np.float32)
        if vibrance:
            scale += vibrance / 100 * (1 - np.minimum(chroma / CHROMA_FULL, 1)) * _lookup(vibrance_weights(), hue)
        if mixer:
            shift, factor, light = mixer_tables(hsl)
            weight = np.minimum(chroma / CHROMA_HUE, 1)
            scale *= 1 + _lookup(factor, hue)
            lightness = lightness * np.exp2(_lookup(light, hue) * weight)
            hue = hue + _lookup(shift, hue)
        chroma = chroma * np.maximum(scale, 0)
        radians = np.radians(hue)
        a, b = chroma * np.cos(radians), chroma * np.sin(radians)
    if mono:
        a, b = np.zeros_like(a), np.zeros_like(b)
    if tint:
        level = np.clip(lightness + grading.get('balance', 0) / 300, 0, 1)
        weights = [(1 - level) ** 2, 2 * level * (1 - level), level ** 2]
        protection = .2 + .8 * np.sin(np.pi * np.clip(lightness, 0, 1))
        offset = sum((w * protection)[..., None] * v for w, v in zip(weights, grade_vectors(grading)))
        a, b = a + offset[..., 0], b + offset[..., 1]
    lab = np.stack([lightness, a, b], axis=-1).astype(np.float32)
    return encode(gamut_map(from_oklab(lab), value), value)
