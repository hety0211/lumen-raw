"""Per-photo, monotonic baseline rendering derived from the camera JPEG."""
import io
import numpy as np
import cv2
from PIL import Image, ImageOps
from .i18n import N_


IDENTITY = [[0., 0.], [1., 1.]]


def luminance(rgb):
    return rgb[..., 0]*.2126 + rgb[..., 1]*.7152 + rgb[..., 2]*.0722


#: Limits of the camera colour reference (``chroma``): how much more or less colourful than a
#: plain colour-matrix development the camera's own picture may be before it is not believed.
CHROMA_RANGE = (.75, 2.)
#: Lightness band (OkLab) the colourfulness is compared in: the camera's picture loses colour
#: towards white and black for reasons of its own (highlight roll-off, noise reduction).
CHROMA_BAND = (.2, .95)
#: Long edge both pictures are averaged down to first: sensor noise reads as colour otherwise
#: (the camera's picture is denoised, the fresh decode is not).
CHROMA_EDGE = 96
#: Mean OkLab chroma below which a picture has too little colour to compare (fog, snow, night).
CHROMA_FLOOR = .012
#: Labels kept in recipes (``develop['source']``) and translated where shown; projects of
#: earlier versions carry the first ones.
SOURCES = (N_('相机预览亮度参考'), N_('相机 HDR 预览亮度参考（PQ 转 SDR）'), N_('标准亮度（相机预览不可用）'),
           N_('相机预览亮度与色彩参考'))


def chroma(settings):
    """Chroma gain of the develop starting point: the camera colour reference (2.3.1), 1 for the
    linear starting point and for recipes made before it existed."""
    if not settings or settings.get('mode') != 'camera':
        return 1.
    value = settings.get('chroma', 1.)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else 1.


def _colourfulness(srgb):
    """Mean OkLab chroma of a display sRGB picture over its mid lightness, noise averaged out;
    None when too little of it lies there."""
    from . import color
    h, w = srgb.shape[:2]
    ratio = CHROMA_EDGE / max(h, w)
    small = cv2.resize(np.ascontiguousarray(srgb, np.float32), (max(1, round(w * ratio)), max(1, round(h * ratio))),
                       interpolation=cv2.INTER_AREA) if ratio < 1 else np.asarray(srgb, np.float32)
    lab = color.to_oklab(color.srgb_to_linear(np.clip(small, 0, 1)))
    band = (lab[..., 0] > CHROMA_BAND[0]) & (lab[..., 0] < CHROMA_BAND[1])
    if band.sum() < .1 * band.size:
        return None
    return float(np.hypot(lab[..., 1], lab[..., 2])[band].mean())


def camera_chroma(developed, preview):
    """How much more colourful the camera's own picture (``preview``, display sRGB) is than
    LUMEN's development of the same photo with the camera's brightness (``developed``): the chroma
    gain that makes the starting point look like the picture the camera showed (2.3.1).

    The tone curve is matched to the camera's brightness on luminance, which keeps every colour's
    saturation as the sensor recorded it; a camera (and Lightroom's profiles) applies its curve in
    a way that also deepens colours, and adds a picture style on top.  Like the curve, the gain
    compares the two pictures as wholes, so a preview with other proportions or lens corrections
    still fits.  1 when the photo has too little colour to tell, or the camera's picture is
    black-and-white (a monochrome style does not make the RAW monochrome)."""
    ours, camera = _colourfulness(developed), _colourfulness(preview)
    if ours is None or camera is None or ours < CHROMA_FLOOR or camera < .25 * ours:
        return 1.
    return round(float(np.clip(camera / ours, *CHROMA_RANGE)), 3)


def _preview(raw, path):
    """``(camera picture as display sRGB 0–1, at most 800 pixels; label of the reference; whether
    its colours are the camera's)`` — raises without one."""
    from . import previews
    label, own = SOURCES[0], True
    try:
        thumb = raw.extract_thumb()
    except Exception:
        # 1.4.1: Canon HDR PQ files embed HEVC previews that LibRaw cannot return.
        hevc = previews.hevc_preview(path) if path else None
        if hevc is None:
            raise
        # The SDR rendering of a PQ preview is LUMEN's own: its brightness is used, not its colour.
        return hevc.astype(np.float32)/255, SOURCES[1], False
    if isinstance(thumb.data, bytes):
        with Image.open(io.BytesIO(thumb.data)) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            im.thumbnail((800, 800))
            return np.asarray(im, np.float32)/255, label, own
    return np.asarray(thumb.data, np.float32)/255, label, own


def _curve(linear, preview):
    from .engine import to_linear, resize_limit
    a = luminance(resize_limit(linear, 800))
    b = luminance(to_linear(resize_limit(preview, 800)))
    quantiles = [0.5, 2, 5, 10, 20, 35, 50, 65, 80, 90, 95, 98, 99.5]
    xs, ys = np.percentile(a, quantiles), np.percentile(b, quantiles)
    points = [[0., 0.]]
    for x, y in zip(xs, ys):
        x = float(x)
        # Guard pathological thumbnails and preserve strictly ordered anchors.
        if x > points[-1][0] + .0001 and x < .999:
            y = float(np.clip(y, max(points[-1][1], x/8), min(1., x*16)))
            points.append([x,y])
    points.append([1.,1.])
    return points


def _fallback(linear):
    from .engine import resize_limit
    # Fallback only when no embedded JPEG can be decoded; do not invent metadata.
    high = float(np.percentile(luminance(resize_limit(linear, 800)), 99.5))
    gain = float(np.clip(.8/max(high, .02), 1, 8))
    return [[0.,0.],[.8/gain,.8],[1.,1.]], SOURCES[2]


def camera_curve(raw, linear, path=None):
    """``(curve, label)``: the tone curve that gives ``linear`` (0–1) the camera picture's brightness."""
    try:
        preview, label, _ = _preview(raw, path)
        return _curve(linear, preview), label
    except Exception:
        return _fallback(linear)


def camera_reference(raw, linear, path=None):
    """The develop starting point of a RAW photo (``edits['develop']``): the camera picture's
    brightness as a tone curve and, since 2.3.1, its colourfulness as a chroma gain.

    ``linear`` is the decoded photo in linear sRGB primaries at preview size, unclipped."""
    from . import tone
    clipped = np.clip(linear, 0, 1)
    try:
        preview, label, own = _preview(raw, path)
        curve = _curve(clipped, preview)
    except Exception:
        curve, label = _fallback(clipped)
        return dict(mode='camera', curve=curve, source=label, chroma=1.)
    gain = 1.
    if own:
        try:
            gain = camera_chroma(tone.develop_view(linear, dict(mode='camera', curve=curve)), preview)
        except Exception:
            gain = 1.
    return dict(mode='camera', curve=curve, source=SOURCES[3] if gain != 1. else label, chroma=gain)


def apply(source, settings):
    if settings.get('mode') != 'camera' or settings.get('curve', IDENTITY) == IDENTITY:
        return source
    from .curves import evaluate
    lum = luminance(source)
    axis = np.linspace(0,1,16385)
    mapped = np.interp(lum,axis,evaluate(settings['curve'],axis)).astype(np.float32)
    ratio = np.clip(mapped/np.maximum(lum,1e-7),0,16)
    return source * ratio[...,None]
