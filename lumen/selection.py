"""Offline open-source semantic / saliency masks, picked objects and contiguous color selection."""
import base64
import threading
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from .i18n import tr


def encode(alpha):
    ok, data = cv2.imencode('.png', np.round(np.clip(alpha,0,1)*255).astype(np.uint8))
    if not ok:
        raise ValueError(tr('无法保存蒙版像素。'))
    return base64.b64encode(data).decode('ascii')


#: Stored masks: largest side and pixel count (1.7.0: fine masks up to ``FINE_EDGE``; 2048 before).
MASK_EDGE = 8192
MASK_PIXELS = 40_000_000
MASK_CHARS = 48_000_000
#: Decoded masks kept in memory (bytes); a fine 4096-pixel mask is about 45 MB as float.
DECODED_BYTES = 768 * 2 ** 20
_decoded = {}
_decoded_lock = __import__('threading').Lock()


def decode(encoded):
    with _decoded_lock:
        hit = _decoded.pop(encoded, None) if isinstance(encoded, str) else None
        if hit is not None:
            _decoded[encoded] = hit
            return hit
    result = _decode(encoded)
    with _decoded_lock:
        _decoded[encoded] = result
        while len(_decoded) > 1 and sum(v.nbytes for v in _decoded.values()) > DECODED_BYTES:
            _decoded.pop(next(iter(_decoded)))
    return result


def _decode(encoded):
    if not isinstance(encoded,str) or len(encoded) > MASK_CHARS:
        raise ValueError(tr('蒙版数据过大。'))
    data = base64.b64decode(encoded,validate=True)
    # Validate dimensions before handing compressed bytes to the decoder.
    import struct
    if data[:8] != b'\x89PNG\r\n\x1a\n' or len(data) < 24:
        raise ValueError(tr('蒙版不是有效 PNG。'))
    w,h = struct.unpack('>II',data[16:24])
    if not 1 <= w <= MASK_EDGE or not 1 <= h <= MASK_EDGE or w * h > MASK_PIXELS:
        raise ValueError(tr('蒙版分辨率超出限制。'))
    result = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_GRAYSCALE)
    if result is None:
        raise ValueError(tr('蒙版像素损坏。'))
    result = result.astype(np.float32)/255
    result.setflags(write=False)
    return result


@lru_cache(maxsize=8)
def _shrunk(encoded, width, height):
    """A stored mask larger than the frame, averaged down to exactly the frame's size: whole
    frames and any block of them read the same pixels (1.7.0)."""
    result = cv2.resize(decode(encoded), (width, height), interpolation=cv2.INTER_AREA)
    result.setflags(write=False)
    return result


def stored(mask, shape):
    """The stored raster of ``mask`` for a frame of ``shape``: as stored, or averaged down to the
    frame when it is larger (a fine mask on the preview)."""
    h, w = shape[:2]
    raster = decode(mask['raster'])
    if raster.shape[0] >= h and raster.shape[1] >= w and raster.shape != (h, w):
        return _shrunk(mask['raster'], w, h)
    return raster


#: 2.1.0: landscape groups in the order of the network's output (tools/convert_landscape.py).
LANDSCAPE = ('water', 'vegetation', 'mountain', 'building', 'ground')
MODELS = {'sky':'skyseg.onnx','person':'person-deeplab.onnx','background':'u2netp.onnx',
          'subject':'u2netp.onnx','foreground':'midas-small.onnx','landscape':'landscape-seg.onnx',
          'object-encoder':'object-encoder.onnx','object-decoder':'object-decoder.onnx'}
#: The 2.1.0 networks run on the CPU: about 0.2–0.5 s, faster than DirectML on the test machine
#: and no first-use GPU compilation for a single image; their working memory is returned after
#: each run (no ONNX Runtime arena, see ``compute.Session``).
CPU_ONLY = ('landscape', 'object-encoder', 'object-decoder')


def session(kind, cuda=False):
    """The ONNX Runtime session of a mask kind; the five landscape kinds share one (2.1.0: the cache
    key was taken before they were mapped to it, so each made its own copy of the network)."""
    kind = 'landscape' if kind in LANDSCAPE else kind
    return _session(kind, bool(cuda) and kind not in CPU_ONLY)


@lru_cache(maxsize=8)
def _session(kind, cuda):
    from . import compute
    path = Path(__file__).resolve().parents[1]/'assets/models'/MODELS[kind]
    if kind in CPU_ONLY:
        return compute.session(path,False,arena=False)
    return compute.session(path,cuda)


session.cache_clear = _session.cache_clear


def refine(alpha, rgb, sharpen=True):
    """Align a coarse network mask (e.g. 320 × 320) with the edges of ``rgb``.

    Upscaled to a 1600-pixel preview the network edge is a ramp of about ten pixels.  Up to
    1.6.0 a 9 × 9 grey guided filter left most of that ramp outside the subject, so darkening
    the sky or background left a light outline around people, trees and buildings.  The ramp
    is now first narrowed around 0.5 (``sharpen``; not for depth, whose ramp is deliberate),
    since a guided filter keeps the mask's average on each side of an edge; then a colour
    guided filter whose window spans the ramp (two network pixels) moves the edge onto the
    colour edge of the photograph (1.6.1)."""
    h,w = rgb.shape[:2]
    p = cv2.resize(np.asarray(alpha,np.float32),(w,h),interpolation=cv2.INTER_LINEAR)
    if sharpen:
        t = np.clip((p-.5)/.12+.5,0,1)
        p = t*t*(3-2*t)
    ratio = max(h/alpha.shape[0],w/alpha.shape[1])
    radius = max(4,int(np.ceil(2*ratio)))
    guide = np.ascontiguousarray(rgb,dtype=np.float32)
    box = lambda a: cv2.boxFilter(a,-1,(2*radius+1,2*radius+1),normalize=True,borderType=cv2.BORDER_REFLECT)
    mean_i,mean_p = box(guide),box(p)
    cov = box(guide*p[...,None])-mean_i*mean_p[...,None]
    s = {(i,j): box(guide[...,i]*guide[...,j])-mean_i[...,i]*mean_i[...,j]+(1e-3 if i==j else 0)
         for i in range(3) for j in range(i,3)}
    # a = Σ⁻¹ · cov per pixel, with the symmetric 3 × 3 inverse written out.
    c00,c01,c02 = s[1,1]*s[2,2]-s[1,2]**2, s[0,2]*s[1,2]-s[0,1]*s[2,2], s[0,1]*s[1,2]-s[0,2]*s[1,1]
    c11,c12,c22 = s[0,0]*s[2,2]-s[0,2]**2, s[0,1]*s[0,2]-s[0,0]*s[1,2], s[0,0]*s[1,1]-s[0,1]**2
    det = s[0,0]*c00+s[0,1]*c01+s[0,2]*c02
    x,y,z = cov[...,0],cov[...,1],cov[...,2]
    a = np.dstack([c00*x+c01*y+c02*z, c01*x+c11*y+c12*z, c02*x+c12*y+c22*z])/det[...,None]
    b = mean_p-(a*mean_i).sum(axis=2)
    return np.clip((box(a)*guide).sum(axis=2)+box(b),0,1)


#: Long edge of a fine mask (1.7.0): the original is refined at most at this size.
FINE_EDGE = 4096
#: Network input size per kind (the models have fixed inputs); for the landscape network the
#: long side of its output (1/8 of a 512-pixel short side), for objects that of the decoder's mask.
NETWORK = dict(sky=320, person=384, background=320, subject=320, foreground=256,
               **{kind: 96 for kind in LANDSCAPE}, object=256)


def _blur(x, sigma):
    """Gaussian blur of ``x`` with a large ``sigma`` through a reduced copy (cheap at any sigma)."""
    h, w = x.shape[:2]
    factor = max(1., sigma / 3)
    if factor <= 1.5:
        return cv2.GaussianBlur(x, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    small = cv2.resize(x, (max(1, round(w / factor)), max(1, round(h / factor))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / factor, borderType=cv2.BORDER_REFLECT)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _spread(image, known, scales):
    """Colours of the ``known`` pixels carried into their surroundings: normalised convolution at
    increasing scales, each filling only where the finer ones had too little support."""
    result = np.zeros_like(image)
    filled = np.zeros(known.shape, np.float32)
    weight = known.astype(np.float32)
    weighted = image * weight[..., None]
    for sigma in scales:
        support = _blur(weight, sigma)
        colour = _blur(weighted, sigma) / np.maximum(support, 1e-6)[..., None]
        take = np.clip(support / .08, 0, 1) * (1 - filled)
        result += colour * take[..., None]
        filled += take
    rest = 1 - filled
    if rest.max() > 0 and weight.sum() > 0:
        result += (weighted.reshape(-1, 3).sum(axis=0) / weight.sum()) * rest[..., None]
    return result


def _guided(guide, p, radius, eps):
    """Grey guided filter (He et al.) of ``p`` with the luminance of ``guide``."""
    i = cv2.cvtColor(np.ascontiguousarray(guide, np.float32), cv2.COLOR_RGB2GRAY)
    box = lambda a: cv2.boxFilter(a, -1, (2 * radius + 1, 2 * radius + 1), borderType=cv2.BORDER_REFLECT)
    mean_i, mean_p = box(i), box(p)
    a = (box(i * p) - mean_i * mean_p) / (box(i * i) - mean_i * mean_i + eps)
    b = mean_p - a * mean_i
    return box(a) * i + box(b)


def fine(alpha, rgb, network=320, grow=.05, shrink=.05):
    """A mask refined on a larger developed image ``rgb`` (display values 0–1), for hair, fur,
    branches and other fine edges (1.7.0).

    ``alpha`` is the mask of the quick pass (the network's output aligned to a preview by
    ``refine``).  Its edge is only as exact as the network's pixel (a 320-pixel network on a
    4000-pixel image: 12 pixels), so the band where it is neither clearly in nor out becomes
    unknown.  There each pixel's colour is explained as a mix of the nearest foreground and
    background colours (estimated by normalised convolution from the known pixels around it):
    the mixing ratio is the alpha, as in sampling-based matting.  Where the two colours are
    alike or the mix does not explain the pixel, the guided-filter alpha of the quick pass is
    kept; a small guided filter removes single-pixel noise.  No learned matting model is used."""
    h, w = rgb.shape[:2]
    guide = np.ascontiguousarray(np.clip(rgb, 0, 1), np.float32)
    coarse = cv2.resize(np.asarray(alpha, np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    pixel = max(h, w) / network                       # one network pixel at this size
    band = max(3, int(round(pixel * .75)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    # Only what the quick pass is sure of is known: a twig it half saw (0.9) is refined too.
    inside = cv2.erode((coarse >= .99).astype(np.uint8), kernel)
    outside = cv2.erode((coarse <= .01).astype(np.uint8), kernel)
    unknown = (inside == 0) & (outside == 0)
    if not unknown.any() or not inside.any() or not outside.any():
        return coarse
    scales = (band * .5, band * 2., band * 8., band * 32.)
    fg = _spread(guide, inside, scales)
    bg = _spread(guide, outside, scales)
    d = fg - bg
    separation = (d * d).sum(axis=2)
    projected = np.clip(((guide - bg) * d).sum(axis=2) / (separation + 1e-4), 0, 1)
    residual = np.sqrt(((guide - (bg + d * projected[..., None])) ** 2).sum(axis=2))
    confidence = separation / (separation + .08 ** 2) * np.exp(-(residual / (.25 * np.sqrt(separation) + .02)) ** 2)
    # Only near an edge the network drew, or on fine detail of the photo (twigs, wires, hair):
    # where the mask is a broad ramp over smooth tones (haze, depth, an object the network was
    # unsure of) the colours do not say what belongs to the mask, so the quick pass stays.
    gy, gx = np.gradient(_blur(coarse, band * .5))
    steep = cv2.dilate(np.hypot(gx, gy) * (4 * band), kernel)
    detail = np.sqrt(((guide - cv2.GaussianBlur(guide, (0, 0), band * .5)) ** 2).sum(axis=2))
    detail /= .15 * np.sqrt(separation) + .01
    confidence *= np.clip(np.maximum(steep, detail), 0, 1) ** 2
    smooth = np.clip(_guided(guide, coarse, max(2, band // 2), 1e-3), 0, 1)
    mixed = np.clip(_guided(guide, (confidence * projected + (1 - confidence) * smooth).astype(np.float32), 1, 1e-4), 0, 1)
    # Into the known parts over one band, so the estimate meets them without a step.
    distance = cv2.distanceTransform(np.where(unknown, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
    t = np.clip(distance / band, 0, 1)
    weight = t * t * (3 - 2 * t)
    result = np.clip(mixed * (1 - weight) + (inside > 0) * weight, 0, 1)
    # How far the colours may overrule the network: raising the mask needs it to have given the
    # pixel at least ``grow``, lowering at most 1 − ``shrink`` (see ``refine_fine``).
    up = np.clip(coarse / max(grow, 1e-3), 0, 1)
    down = np.clip((1 - coarse) / max(shrink, 1e-3), 0, 1)
    result = np.where(result > coarse, coarse + (result - coarse) * up, coarse - (coarse - result) * down)
    return np.clip(result, 0, 1).astype(np.float32)


#: Kinds the fine pass refines: depth ("foreground") keeps its deliberate ramp, colour picks are hard.
FINE_KINDS = ('sky', 'person', 'background', 'subject') + LANDSCAPE + ('object',)


def fine_size(width, height):
    """Size of the image a fine mask is refined on: the original, at most ``FINE_EDGE`` long."""
    ratio = min(1., FINE_EDGE / max(width, height))
    return max(1, round(width * ratio)), max(1, round(height * ratio))


def fine_guide(source, edits):
    """The developed, lens-corrected and retouched original at ``fine_size`` (display values 0–1),
    the image the fine pass looks at; ``source`` is the decoded original (any size larger than the
    preview)."""
    from . import engine
    h, w = source.shape[:2]
    size = fine_size(w, h)
    small = cv2.resize(np.asarray(source, np.float32), size, interpolation=cv2.INTER_AREA) if size != (w, h) else source
    return engine.selection_view(small, edits)


def refine_fine(kind, alpha, guide):
    """The fine mask of ``kind`` from its quick ``alpha`` (preview size), or None when the kind is
    not refined.  Backgrounds are refined as the subject they invert."""
    if kind not in FINE_KINDS:
        return None
    if kind == 'background':
        return 1 - fine(1 - np.asarray(alpha, np.float32), guide, NETWORK['subject'])
    if kind == 'sky':
        # Glass, water and pale walls can look like sky; dark twigs and wires in a sky cannot.
        return fine(alpha, guide, NETWORK[kind], grow=.3, shrink=.02)
    return fine(alpha, guide, NETWORK[kind])


def landscape(rgb):
    """Probability of each ``LANDSCAPE`` group, shape (5, h, w) at 1/8 of the network input: the
    image with a 512-pixel short side (ADE20K evaluation), the long side a multiple of 32."""
    h, w = rgb.shape[:2]
    short, long = LANDSCAPE_SHORT, min(LANDSCAPE_LONG, int(np.ceil(max(h, w) / min(h, w) * LANDSCAPE_SHORT / 32)) * 32)
    size = (long, short) if w >= h else (short, long)
    x = cv2.resize(np.asarray(rgb, np.float32), size, interpolation=cv2.INTER_AREA)
    x = (x - np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    sess = session('landscape')
    prediction = sess.run(None, {'image': np.ascontiguousarray(x.transpose(2,0,1)[None], np.float32)})[0][0]
    if not np.isfinite(prediction).all():
        raise ValueError(tr('识别模型返回无效结果。'))
    return prediction


LANDSCAPE_SHORT, LANDSCAPE_LONG = 512, 1536


def automatic(rgb, kind, cuda=False):
    if kind in LANDSCAPE:
        prediction = landscape(rgb)[LANDSCAPE.index(kind)]
        return refine(np.clip(prediction, 0, 1), rgb), session(kind).get_providers()[0]
    sess = session('background' if kind=='subject' else kind,cuda)
    size = 256 if kind == 'foreground' else (384 if kind == 'person' else 320)
    x = cv2.resize(rgb,(size,size),interpolation=cv2.INTER_AREA).astype(np.float32)
    if kind not in ('person','foreground'):
        x /= max(float(x.max()),1e-5)
    x = (x - np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    prediction = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name:x.transpose(2,0,1)[None].copy()})[0].squeeze()
    if not np.isfinite(prediction).all():
        raise ValueError(tr('识别模型返回无效结果。'))
    if kind=='foreground':
        # MiDaS predicts relative inverse depth: larger values are nearer.
        low,high=np.percentile(prediction,[10,95])
        relative=np.clip((prediction-low)/max(float(high-low),1e-6),0,1)
        prediction=np.clip((relative-.48)/.22,0,1)
        prediction=prediction*prediction*(3-2*prediction)
    alpha = refine(np.clip(prediction,0,1),rgb,sharpen=kind!='foreground')
    if kind == 'background':
        alpha = 1-alpha
    return alpha, sess.get_providers()[0]


#: 2.1.0: picked objects (EfficientViT-SAM L0): the encoder sees the image with its long side at
#: ``OBJECT_INPUT`` pixels; the decoder answers each click in about 15 ms from its embedding.
OBJECT_INPUT, OBJECT_FRAME = 512, 1024
_embedding = {}
_embedding_lock = threading.Lock()


def object_embedding(rgb, key=None):
    """The encoder's embedding of ``rgb`` (display sRGB 0–1); the last one is kept under ``key``."""
    with _embedding_lock:
        if key is not None and _embedding.get('key') == key:
            return _embedding['value']
    h, w = rgb.shape[:2]
    scale = OBJECT_INPUT / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    x = cv2.resize(np.clip(np.asarray(rgb, np.float32), 0, 1) * 255, (nw, nh), interpolation=cv2.INTER_AREA)
    x = (x - np.array([123.675, 116.28, 103.53], np.float32)) / np.array([58.395, 57.12, 57.375], np.float32)
    padded = np.zeros((OBJECT_INPUT, OBJECT_INPUT, 3), np.float32)
    padded[:nh, :nw] = x
    value = session('object-encoder').run(None, {'input_image': np.ascontiguousarray(padded.transpose(2, 0, 1)[None])})[0]
    if not np.isfinite(value).all():
        raise ValueError(tr('识别模型返回无效结果。'))
    if key is not None:
        with _embedding_lock:
            _embedding.update(key=key, value=value)
    return value


def _object_probability(embedding, points, h, w):
    """The decoder's probability of the object at ``points`` over an ``h × w`` photo, at the
    photo's size in the 1024 frame."""
    scale = OBJECT_FRAME / max(h, w)
    coords = np.array([[[x * w * scale, y * h * scale] for x, y, _ in points]], np.float32)
    labels = np.array([[label for *_, label in points]], np.float32)
    logits = session('object-decoder').run(None, {'image_embeddings': embedding, 'point_coords': coords,
                                                  'point_labels': labels})[0][0, 0]
    if not np.isfinite(logits).all():
        raise ValueError(tr('识别模型返回无效结果。'))
    # The decoder's 256-pixel mask spans the padded 1024 frame; the photo fills its top left.
    frame = cv2.resize(logits, (OBJECT_FRAME, OBJECT_FRAME), interpolation=cv2.INTER_LINEAR)
    frame = frame[:max(1, round(h * scale)), :max(1, round(w * scale))]
    return (1 / (1 + np.exp(-np.clip(frame, -30, 30)))).astype(np.float32)


def object_mask(rgb, points, key=None):
    """Mask of the object at ``points`` (``[x, y, label]``, normalised; label 1 adds the object
    under the click, 0 takes it away), aligned with the edges of ``rgb``.

    Given only points, the decoder can leave an island of the object under a subtracting click
    (a part of the third of three loaves), so that object, as the click alone selects it, is taken
    away as well; not when it contains an adding click: then the click marks a part of the
    selected object (a paw of the cat) and the decoder's answer stands."""
    if not points:
        raise ValueError(tr('请在画面上点击要选择的物体。'))
    embedding = object_embedding(rgb, key)
    h, w = rgb.shape[:2]
    probability = _object_probability(embedding, points, h, w)
    added = [(x, y) for x, y, label in points if label]
    for x, y, label in points:
        if label:
            continue
        own = _object_probability(embedding, [[x, y, 1]], h, w)
        oh, ow = own.shape
        if any(own[min(oh - 1, int(py * oh)), min(ow - 1, int(px * ow))] > .5 for px, py in added):
            continue
        probability = np.minimum(probability, 1 - own)
    return refine(probability, rgb)


def color_region(rgb, point, tolerance=18):
    """Fixed seed Lab distance plus 8-connected flood fill; no remote inference."""
    h,w = rgb.shape[:2]
    x,y = round(point[0]*(w-1)),round(point[1]*(h-1))
    lab = cv2.cvtColor(rgb.astype(np.float32),cv2.COLOR_RGB2LAB)
    distance = np.linalg.norm(lab-lab[y,x],axis=2)
    allowed = (distance <= tolerance).astype(np.uint8)
    # Flood-fill only the component containing the seed, not disconnected colors.
    flooded = allowed.copy()
    cv2.floodFill(flooded,None,(x,y),2,loDiff=0,upDiff=0,flags=8|cv2.FLOODFILL_FIXED_RANGE)
    return (flooded==2).astype(np.float32)


def raster_alpha(mask, shape, area=None):
    """Stored mask resized to a frame of ``shape``; with ``area`` only that block of it (1.4.0)."""
    h,w = shape[:2]
    feather = mask.get('feather',0)/100*min(h,w)*.008
    raster = stored(mask, shape)
    if area is None or area.shape == (h,w):
        alpha = cv2.resize(raster,(w,h),interpolation=cv2.INTER_LINEAR)
        if feather > .2:
            alpha = cv2.GaussianBlur(alpha,(0,0),feather)
        return alpha
    from .engine import resize_region
    # OpenCV's float Gaussian kernel reaches about 4 sigma; read that much context.
    outer = area.grow(int(np.ceil(feather*4))+2 if feather > .2 else 0)
    alpha = resize_region(raster,(w,h),outer)
    if feather > .2:
        alpha = cv2.GaussianBlur(alpha,(0,0),feather)
    return np.ascontiguousarray(alpha[outer.inner(area)])
