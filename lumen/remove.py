"""AI removal (2.1.0): a painted or picked region filled by LaMa, stored as a patch.

The fill is computed once, when the region is made, from the original photograph around it
(lens-corrected, with the retouching before it applied) and kept in the recipe as a 16-bit
patch: a generative fill computed again at another size would come out different, so the
preview, every zoom level and the export all show the same pixels.  The patch is in the linear
values of the retouch stage (before white balance, see ``engine._base``).

LaMa (big-lama, Suvorov et al., Apache-2.0) sees the region and as much picture again on each
side, brought down to at most ``WORK_EDGE`` pixels.  Where the patch is enlarged on the
original (a large region), it lacks the sensor grain of its surroundings; ``place`` adds the
missing part, measured around the region when it was made.
"""
from __future__ import annotations

import base64
import random
import threading
from functools import lru_cache

import cv2
import numpy as np

from .i18n import tr

MODEL = 'inpaint-lama.onnx'
#: Longest side of the image LaMa fills (it holds up to about 2000 pixels; 1024 is ~1–2 s on a CPU).
WORK_EDGE = 1024
#: Context on each side of the region, as a fraction of its size and at least this many work pixels.
CONTEXT, MIN_CONTEXT = 1., 96
#: The region LaMa fills is the painted one grown by this many work pixels: object edges and
#: soft brush ends would otherwise be copied into the fill.
GROW = 3
GAMMA = 2.2
#: Stored patches: the work image is at most ``WORK_EDGE`` pixels; base64 text at most this long.
PATCH_EDGE = 2048
PATCH_CHARS = 32_000_000
DECODED_BYTES = 256 * 2 ** 20
_decoded = {}
_lock = threading.Lock()


@lru_cache(maxsize=1)
def session():
    # LaMa's Reshape nodes fail on DirectML; it runs on the CPU (0.3 s for a small region).  Without
    # ONNX Runtime's arena its 2–3 GB of working memory is returned after each fill.
    from . import ai_models, compute
    return compute.session(ai_models.path(MODEL), False, arena=False)


def available():
    from . import ai_models
    return ai_models.available(MODEL)


def _noise_factor(sigma=1.):
    """Standard deviation of ``x - GaussianBlur(x, sigma)`` for unit white noise ``x``."""
    k = cv2.getGaussianKernel(int(2 * np.ceil(3 * sigma) + 1), sigma)
    g = k @ k.T
    centre = g[g.shape[0] // 2, g.shape[1] // 2]
    return float(np.sqrt(1 - 2 * centre + (g * g).sum()))


NOISE_FACTOR = _noise_factor()


def grain(image, known):
    """Per channel: noise standard deviation ÷ √level of ``image`` over the ``known`` pixels
    (robust, from the residual of a 1-pixel blur), the sensor grain a fill should carry."""
    if known.sum() < 64:
        return [0., 0., 0.]
    x = np.ascontiguousarray(image, np.float32)
    residual = x - cv2.GaussianBlur(x, (0, 0), 1.)
    level = np.sqrt(np.maximum(cv2.GaussianBlur(x, (0, 0), 2.), 1e-4))
    z = (residual / level)[known]
    mad = np.median(np.abs(z - np.median(z, axis=0)), axis=0)
    return [float(v) for v in np.clip(mad / .6745 / NOISE_FACTOR, 0, .5)]


def _encode(image, alpha, scale):
    """16-bit PNG of the premultiplied, gamma-encoded patch (zero where nothing is filled)."""
    colour = np.clip(image / scale, 0, 1) ** (1 / GAMMA) * alpha[..., None]
    rgba = np.dstack([colour[..., ::-1], alpha])
    ok, data = cv2.imencode('.png', np.round(rgba * 65535).astype(np.uint16))
    if not ok:
        raise ValueError(tr('无法保存移除结果。'))
    return base64.b64encode(data).decode('ascii')


def decode(encoded, scale):
    """``(premultiplied linear colour, alpha)`` of a stored patch."""
    key = (encoded, scale) if isinstance(encoded, str) else None
    with _lock:
        hit = _decoded.pop(key, None) if key else None
        if hit is not None:
            _decoded[key] = hit
            return hit
    if not isinstance(encoded, str) or len(encoded) > PATCH_CHARS:
        raise ValueError(tr('移除结果数据过大。'))
    data = base64.b64decode(encoded, validate=True)
    import struct
    if data[:8] != b'\x89PNG\r\n\x1a\n' or len(data) < 24:
        raise ValueError(tr('移除结果不是有效 PNG。'))
    w, h = struct.unpack('>II', data[16:24])
    if not 1 <= w <= PATCH_EDGE or not 1 <= h <= PATCH_EDGE:
        raise ValueError(tr('移除结果分辨率超出限制。'))
    rgba = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.dtype != np.uint16 or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(tr('移除结果像素损坏。'))
    rgba = rgba.astype(np.float32) / 65535
    alpha = np.ascontiguousarray(rgba[..., 3])
    colour = rgba[..., 2::-1] / np.maximum(alpha, 1e-6)[..., None]
    linear = np.clip(colour, 0, 1) ** GAMMA * (scale * alpha[..., None])
    result = (np.ascontiguousarray(linear, np.float32), alpha)
    for array in result:
        array.setflags(write=False)
    if key:
        with _lock:
            _decoded[key] = result
            while len(_decoded) > 1 and sum(c.nbytes + a.nbytes for c, a in _decoded.values()) > DECODED_BYTES:
                _decoded.pop(next(iter(_decoded)))
    return result


def validate(patch):
    """A checked copy of a stored patch (raises ValueError)."""
    import math
    if not isinstance(patch, dict):
        raise ValueError(tr('无效修复笔划。'))
    def number(value, low, high):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(tr('无效修复笔划。'))
        return float(value)
    box = patch.get('box')
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(tr('无效修复笔划。'))
    box = [number(v, 0, 1) for v in box]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(tr('无效修复笔划。'))
    frame = patch.get('frame')
    if not isinstance(frame, list) or len(frame) != 2:
        raise ValueError(tr('无效修复笔划。'))
    grains = patch.get('grain', [0., 0., 0.])
    if not isinstance(grains, list) or len(grains) != 3:
        raise ValueError(tr('无效修复笔划。'))
    result = dict(box=box, scale=number(patch.get('scale'), 1e-6, 1e6), frame=[int(number(v, 1, 1e6)) for v in frame],
                  grain=[number(v, 0, .5) for v in grains], kept=number(patch.get('kept', 0), 0, 1),
                  seed=int(number(patch.get('seed', 0), 0, 2 ** 31 - 1)), png=patch.get('png'))
    decode(result['png'], result['scale'])
    return result


def window(patch, h, w):
    """Pixels ``(x0, y0, x1, y1)`` of a frame ``h × w`` that a patch covers."""
    x0, y0, x1, y1 = patch['box']
    return (max(0, int(np.floor(x0 * w))), max(0, int(np.floor(y0 * h))),
            min(w, int(np.ceil(x1 * w))), min(h, int(np.ceil(y1 * h))))


def place(patch, h, w):
    """``(premultiplied colour, alpha, grain std)`` of a patch resampled onto its window of a
    frame ``h × w``; the same for a whole frame and for any block of it."""
    colour, alpha = decode(patch['png'], patch['scale'])
    ph, pw = alpha.shape
    x0, y0, x1, y1 = window(patch, h, w)
    bx0, by0, bx1, by1 = patch['box'][0] * w, patch['box'][1] * h, patch['box'][2] * w, patch['box'][3] * h
    sx, sy = pw / (bx1 - bx0), ph / (by1 - by0)
    if sx > 1.5 or sy > 1.5:
        # Smaller than stored: average down first, so the patch does not alias.
        size = (max(1, round(pw / sx)), max(1, round(ph / sy)))
        colour = cv2.resize(colour, size, interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
        sx, sy = size[0] / (bx1 - bx0), size[1] / (by1 - by0)
    matrix = np.float32([[sx, 0, (x0 + .5 - bx0) * sx - .5], [0, sy, (y0 + .5 - by0) * sy - .5]])
    size = (x1 - x0, y1 - y0)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    colour = cv2.warpAffine(colour, matrix, size, flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = cv2.warpAffine(alpha, matrix, size, flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Grain: the photograph at this size has native noise × density d; the enlarged patch keeps
    # what LaMa reproduced at its own density q.  The difference is added as white noise.
    density = min(1., w / patch['frame'][0])
    stored = pw / max(1e-6, (patch['box'][2] - patch['box'][0]) * patch['frame'][0])
    missing = density ** 2 - stored ** 2 * patch.get('kept', 0)
    std = np.asarray(patch.get('grain', [0, 0, 0]), np.float32) * np.sqrt(max(0., missing))
    return colour, alpha, std


def composite(current, patch, opacity, h, w):
    """``current`` (the patch's window of a frame ``h × w``) with the patch over it."""
    colour, alpha, std = place(patch, h, w)
    weight = opacity / 100
    result = current * (1 - alpha * weight)[..., None] + colour * weight
    if std.max() > 1e-5:
        level = np.sqrt(np.maximum(colour / np.maximum(alpha, 1e-6)[..., None], 0))
        noise = np.random.default_rng([patch['seed'], h, w]).standard_normal(colour.shape, np.float32)
        result += noise * std * level * (alpha * weight)[..., None]
    return result


def fill(frame, before, coverage, area, gains=(1., 1., 1.)):
    """The patch removing ``coverage`` (0–1 over the pixels ``area = (x0, y0, x1, y1)`` of the
    lens-corrected ``frame``, at its own resolution) after the retouch operations ``before``.

    ``gains`` (the white balance) only make the colours LaMa sees natural; the patch stays in
    the frame's linear values."""
    from . import retouch
    h, w = frame.shape[:2]
    ax0, ay0, ax1, ay1 = area
    coverage = np.asarray(coverage, np.float32)
    ys, xs = np.nonzero(coverage > 0)
    if not len(xs):
        raise ValueError(tr('没有需要移除的区域。'))
    hx0, hy0, hx1, hy1 = ax0 + xs.min(), ay0 + ys.min(), ax0 + xs.max() + 1, ay0 + ys.max() + 1
    size = max(hx1 - hx0, hy1 - hy0)
    # Context around the hole; its scale decides how many pixels of it LaMa sees.
    margin = int(np.ceil(max(size * CONTEXT, MIN_CONTEXT * max(1., (size * (1 + 2 * CONTEXT)) / WORK_EDGE))))
    cx0, cy0, cx1, cy1 = max(0, hx0 - margin), max(0, hy0 - margin), min(w, hx1 + margin), min(h, hy1 + margin)
    region = np.asarray(retouch.apply(frame, before, (cx0, cy0, cx1, cy1)), np.float32)
    hole = np.zeros(region.shape[:2], np.float32)
    hole[ay0 - cy0:ay1 - cy0, ax0 - cx0:ax1 - cx0] = coverage
    scale = min(1., WORK_EDGE / max(region.shape[:2]))
    work = (max(8, round(region.shape[1] * scale)), max(8, round(region.shape[0] * scale)))
    gains = np.asarray(gains, np.float32)
    balanced = np.maximum(region, 0) * gains
    peak = max(float(np.percentile(balanced.max(axis=2), 99.7)), 1e-4)
    encoded = np.clip(balanced / peak, 0, 1) ** (1 / GAMMA)
    small = cv2.resize(encoded, work, interpolation=cv2.INTER_AREA) if scale < 1 else encoded
    small_hole = cv2.resize(hole, work, interpolation=cv2.INTER_AREA) if scale < 1 else hole
    known = small_hole <= 0
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GROW + 1, 2 * GROW + 1))
    holes = cv2.dilate((small_hole > 0).astype(np.uint8), grow).astype(np.float32)
    ph, pw = (-small.shape[0]) % 8, (-small.shape[1]) % 8
    padded = np.pad(small, ((0, ph), (0, pw), (0, 0)), mode='symmetric')
    padded_hole = np.pad(holes, ((0, ph), (0, pw)), mode='symmetric')
    result = session().run(None, {'rgb': np.ascontiguousarray(padded.transpose(2, 0, 1)[None], np.float32),
                                  'mask': np.ascontiguousarray(padded_hole[None, None], np.float32)})[0]
    filled = np.clip(result[0].transpose(1, 2, 0)[:small.shape[0], :small.shape[1]], 0, 1)
    if not np.isfinite(filled).all():
        raise ValueError(tr('移除模型返回无效结果。'))
    linear = filled ** GAMMA * peak / np.maximum(gains, 1e-6)
    # The patch: the hole's bounding box (work pixels) with its coverage as alpha.
    alpha = np.clip(small_hole, 0, 1)
    ys, xs = np.nonzero(alpha > 0)
    px0, py0 = max(0, xs.min() - 1), max(0, ys.min() - 1)
    px1, py1 = min(alpha.shape[1], xs.max() + 2), min(alpha.shape[0], ys.max() + 2)
    sx, sy = work[0] / region.shape[1], work[1] / region.shape[0]
    box = [(cx0 + px0 / sx) / w, (cy0 + py0 / sy) / h, (cx0 + px1 / sx) / w, (cy0 + py1 / sy) / h]
    patch_linear, patch_alpha = linear[py0:py1, px0:px1], alpha[py0:py1, px0:px1]
    stored_scale = max(float(patch_linear.max()), 1e-6)
    # Grain of the photograph around the hole (native pixels) and how much of it LaMa's fill
    # already carries.
    gx0, gy0 = max(0, hx0 - cx0 - 64), max(0, hy0 - cy0 - 64)
    gx1, gy1 = min(region.shape[1], hx1 - cx0 + 64), min(region.shape[0], hy1 - cy0 + 64)
    noise = grain(region[gy0:gy1, gx0:gx1], hole[gy0:gy1, gx0:gx1] <= 0)
    near = cv2.dilate(holes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
    ring = np.mean(grain(small, near & known))
    inside = np.mean(grain(filled, holes > 0))
    kept = float(np.clip((inside / ring) ** 2, 0, 1)) if ring > 1e-6 else 1.
    return dict(box=[float(np.clip(v, 0, 1)) for v in box], scale=stored_scale, frame=[w, h], grain=noise,
                kept=kept, seed=random.randrange(2 ** 31), png=_encode(patch_linear, patch_alpha, stored_scale))


def brush_coverage(op, h, w):
    """Coverage of a painted removal over a frame ``h × w``, as ``(coverage, area)`` for ``fill``:
    all of the stroke, softened outward by its feather.  (The heal tool feathers inward; a removal
    must replace everything painted, or the object shows through the soft edge.)"""
    from . import retouch
    _, mask, radius, area = retouch.stroke(op, h, w)
    soft = radius * op['feather'] / 100
    if soft < 1:
        return (mask > 0).astype(np.float32), area
    distance = cv2.distanceTransform(np.where(mask > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 5)
    t = np.clip(1 - distance / soft, 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32), area


def object_coverage(alpha, h, w, grow=.004):
    """Coverage over a frame ``h × w`` of a picked object (``alpha`` at any size): grown a little,
    since object edges, fringes and soft shadows would be copied into the fill, with soft ends.
    Returns ``(coverage, area)`` for ``fill``."""
    big = cv2.resize(np.asarray(alpha, np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > .5
    ys, xs = np.nonzero(big)
    if not len(xs):
        raise ValueError(tr('没有需要移除的区域。'))
    radius = max(2, int(round(grow * min(h, w))))
    pad = 2 * radius + 2
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1, y1 = min(w, xs.max() + pad + 1), min(h, ys.max() + pad + 1)
    local = big[y0:y1, x0:x1].astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    grown = cv2.dilate(local, kernel)
    soft = cv2.GaussianBlur(grown.astype(np.float32), (0, 0), max(1., radius / 2))
    coverage = np.where(grown > 0, np.maximum(soft, .5) * 2 - 1, 0) ** .5
    return np.clip(coverage, 0, 1).astype(np.float32), (x0, y0, x1, y1)
