"""Portrait retouching (2.3.0, the 人像 panel): skin, eyes, teeth and face shape per face.

Faces are found with YuNet and placed with MediaPipe Face Mesh's 478 landmarks (the models the
culling hints already use, ``culling.Faces``).  The landmarks are stored in the recipe, relative
to the lens-corrected frame like masks and retouching, so rendering and export need no model and
give the same result at every resolution: every radius below is a share of the face's size.

Each face has seven settings 0–100, applied in the scene-linear frame right after retouching
(``engine._base``), on a gamma-2.2 copy of the face's window:

* **祛痘 acne** — small round spots darker or redder than the skin around them (a morphological
  closing at the scale of blemishes, against the spread of the face's own skin) are filled from
  their surroundings by normalised convolution, with fine skin texture from beside them laid back
  in.  Only the open skin of the face is searched: creases and shadows that belong to the face
  (beside the nose, smile lines, under the eyes) and long or hollow shapes are never spots.
* **磨皮 smooth** — frequency separation: an edge-preserving guided filter at a working scale
  of about 500 pixels across the face removes uneven tone and blemish texture; the finer pores
  stay (a little softened at high settings), so skin does not turn to plastic.
* **去黑眼圈 circles** — under each eye, the low-frequency difference to the cheek just below
  is added, brightening only.
* **美白 whiten** — skin lifted by a screen-like curve that leaves highlights alone and slightly
  desaturated; it reaches the neck through the skin-colour model so the face does not float.
* **白牙 teeth** — inside the inner lips, bright and not red pixels (teeth, not tongue or gums)
  lose their yellow and brighten.
* **大眼 eyes** — a smooth local magnification around each iris (radius 1.1 eye widths).
* **瘦脸 slim** — Gustafsson's local translation warps pull six points of the jaw towards the
  face's axis.

The skin mask is the face oval and the neck below it, minus eyes, brows, lips, nostrils and thin
lines at eye height (glasses frames), times a skin-colour likelihood learnt from the cheeks and
forehead of that face.
"""
from __future__ import annotations

import math
import threading
import zlib
from collections import OrderedDict

import cv2
import numpy as np

#: Settings of one face, in the order of the panel.
KEYS = ('smooth', 'whiten', 'acne', 'circles', 'teeth', 'eyes', 'slim')
COLOUR_KEYS = ('smooth', 'whiten', 'acne', 'circles', 'teeth')
POINTS = 478
MAX_FACES = 24
#: Smallest face (pixels across, on the detection image) given landmarks.
MIN_FACE_PIXELS = 24

# MediaPipe Face Mesh indices.  "Right" is the person's right (the left of a frontal photo).
FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
             148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
LIPS_OUTER = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146)
LIPS_INNER = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95)
RIGHT_EYE = (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7)
LEFT_EYE = (263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249)
RIGHT_LOWER_LID = (33, 7, 163, 144, 145, 153, 154, 155, 133)
LEFT_LOWER_LID = (263, 249, 390, 373, 374, 380, 381, 382, 362)
RIGHT_BROW = (70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
LEFT_BROW = (300, 293, 334, 296, 336, 285, 295, 282, 283, 276)
NOSTRILS = (64, 219, 218, 237, 44, 1, 274, 457, 438, 439, 294, 327, 326, 2, 97, 98)
RIGHT_IRIS, LEFT_IRIS = 468, 473
#: Cheeks and forehead: where the face's skin colour is learnt.
SKIN_SAMPLES = (50, 280, 151, 205, 425, 108, 337)
#: Jaw points pulled in by 瘦脸 (person's right, then left) and how strongly.
JAW = ((132, .55), (58, .9), (172, 1.), (136, .8), (361, .55), (288, .9), (397, 1.), (365, .8))
CHEEK_RIGHT, CHEEK_LEFT, BROW_MIDDLE, CHIN, FOREHEAD = 234, 454, 168, 152, 10
#: Creases and shadows every face has, which 祛痘 leaves alone: beside the nose, then the smile lines.
NOSE_SIDES = ((193, 196, 198, 129, 64), (417, 419, 420, 358, 294))
SMILE_LINES = ((64, 203, 206, 216, 212), (294, 423, 426, 436, 432))
#: The crease under the lower lip and the dimple of the chin.
CHIN_CREASE = (201, 200, 421)
CHIN_DIMPLE = (200, 199, 175)

#: Long edge of a face (pixels) the tonal work is computed at; finer detail is kept as it is.
WORK_SIZE = 520
#: Strongest effect of each warp: magnification at the iris, jaw shift as a share of face width.
EYE_SCALE = .25
SLIM_SHIFT = .05
#: How far above the skin's texture a spot must stand out for 祛痘 at 0 and at 100.
LIGHTEST_SPOT, FAINTEST_SPOT = 7., 3.5
#: A mark darker than the skin by this share of its brightness is a mole or tattoo, not a blemish.
MARK_CONTRAST = .33


def settings():
    return dict.fromkeys(KEYS, 0.)


def defaults():
    return dict(faces=[])


def validate(data):
    """A checked copy of a recipe's ``portrait`` entry; raises ValueError when malformed."""
    if data is None:
        return defaults()
    if not isinstance(data, dict) or not isinstance(data.get('faces', []), list) or len(data.get('faces', [])) > MAX_FACES:
        raise ValueError('portrait')

    def number(value, low, high):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('portrait')
        return float(value)
    faces = []
    for face in data.get('faces', []):
        if not isinstance(face, dict):
            raise ValueError('portrait')
        box, points, values = face.get('box'), face.get('points'), face.get('settings', {})
        if not isinstance(box, list) or len(box) != 4 or not isinstance(points, list) or len(points) != POINTS \
                or not isinstance(values, dict):
            raise ValueError('portrait')
        box = [number(v, -1, 2) for v in box]
        if box[2] <= 0 or box[3] <= 0:
            raise ValueError('portrait')
        checked = []
        for p in points:
            if not isinstance(p, list) or len(p) != 2:
                raise ValueError('portrait')
            checked.append([number(p[0], -1, 2), number(p[1], -1, 2)])
        faces.append(dict(box=box, points=checked, settings={k: number(values.get(k, 0), 0, 100) for k in KEYS}))
    return dict(faces=faces)


def active(data):
    """Faces of a ``portrait`` entry with any setting on."""
    return [f for f in (data or {}).get('faces', []) if any(f['settings'].get(k, 0) for k in KEYS)]


# ----------------------------------------------------------------------------------- detection

def detect(rgb, analyser=None):
    """Faces of a display RGB image (uint8, or float 0–1), left to right: ``[{box, points,
    settings}]`` relative to the image.  None when the face models are unavailable."""
    from . import culling
    analyser = analyser or culling.faces()
    if analyser is None:
        return None
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255 + .5, 0, 255).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb[..., :3])
    h, w = rgb.shape[:2]
    faces = []
    for found in _candidates(analyser, rgb):
        if max(found[2], found[3]) < MIN_FACE_PIXELS:
            continue
        try:
            points, presence = analyser.landmarks(rgb, found)
        except Exception:
            continue
        if presence < .5:
            continue
        x, y, fw, fh = found[:4]
        faces.append(dict(box=[round(x / w, 5), round(y / h, 5), round(fw / w, 5), round(fh / h, 5)],
                          points=[[round(float(px) / max(1, w - 1), 5), round(float(py) / max(1, h - 1), 5)]
                                  for px, py in points],
                          settings=settings()))
    faces.sort(key=lambda f: (f['box'][0], f['box'][1]))
    return faces[:MAX_FACES]


#: Long edges of the extra, coarser looks: YuNet misses faces that fill most of the frame.
COARSE_EDGES = (640, 320)


def _candidates(analyser, rgb):
    """YuNet's faces at its usual size and at coarser ones, overlaps merged (best score kept)."""
    found = list(analyser.detect(rgb))
    h, w = rgb.shape[:2]
    for edge in COARSE_EDGES:
        scale = edge / max(h, w)
        if scale >= .9:
            continue
        small = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        for face in analyser.detect(small):
            x, y, fw, fh, score, (rx, ry), (lx, ly) = face
            k = 1 / scale
            found.append((x * k, y * k, fw * k, fh * k, score, (rx * k, ry * k), (lx * k, ly * k)))
    found.sort(key=lambda f: -f[4])
    kept = []
    for face in found:
        if all(iou(face[:4], other[:4]) < .35 for other in kept):
            kept.append(face)
    return kept


def iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    w = max(0., min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    h = max(0., min(ay0 + ah, by0 + bh) - max(ay0, by0))
    union = aw * ah + bw * bh - w * h
    return w * h / union if union > 0 else 0.


def merge(old, new):
    """Faces found again keep their settings: a new face takes those of the old face its box
    overlaps most (intersection over union above 0.3).  Faces are matched by box, never by
    number, so a face found or lost elsewhere does not move settings to someone else."""
    taken = set()
    result = []
    for face in new:
        best, score = None, .3
        for index, previous in enumerate(old):
            overlap = iou(face['box'], previous['box'])
            if index not in taken and overlap > score:
                best, score = index, overlap
        if best is not None:
            taken.add(best)
            face = dict(face, settings=dict(old[best]['settings']))
        result.append(face)
    return result


def face_at(faces, point):
    """Index of the face whose box holds ``point`` (frame coordinates 0–1), the smallest such, or None."""
    found = [(f['box'][2] * f['box'][3], i) for i, f in enumerate(faces)
             if f['box'][0] <= point[0] <= f['box'][0] + f['box'][2] and f['box'][1] <= point[1] <= f['box'][1] + f['box'][3]]
    return min(found)[1] if found else None


# ------------------------------------------------------------------------------------ geometry

def pixels(face, h, w):
    return np.asarray(face['points'], np.float64) * [w - 1, h - 1]


def face_size(points):
    """Width (cheek to cheek) and height (hairline to chin) of a face in pixels."""
    return (float(np.linalg.norm(points[CHEEK_LEFT] - points[CHEEK_RIGHT])),
            float(np.linalg.norm(points[CHIN] - points[FOREHEAD])))


def window(face, h, w):
    """Pixels ``(x0, y0, x1, y1)`` a face's retouching reads and writes, in a ``h × w`` frame."""
    points = pixels(face, h, w)
    size = max(face_size(points))
    low = points.min(axis=0) - .45 * size
    high = points.max(axis=0) + [.45 * size, .75 * size]       # the neck below the chin
    x0, y0 = np.clip(np.floor(low), 0, [w, h]).astype(int)
    x1, y1 = np.clip(np.ceil(high) + 1, 0, [w, h]).astype(int)
    return int(x0), int(y0), int(max(x0, x1)), int(max(y0, y1))


def _overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def needed(faces, rect, h, w):
    """The block of the frame to retouch so block ``rect`` matches the whole-frame result: it
    grows by the window of every face that writes what it, or a later face, reads."""
    area = tuple(rect)
    for face in reversed(faces):
        box = window(face, h, w)
        if box[0] < box[2] and box[1] < box[3] and _overlaps(box, area):
            area = (min(area[0], box[0]), min(area[1], box[1]), max(area[2], box[2]), max(area[3], box[3]))
    return area


def apply(block, faces, shape, area=None, keep=False):
    """Retouch ``faces`` (with settings on) in ``block``, which is the block ``area = (x0, y0, x1,
    y1)`` of a frame of ``shape = (h, w)`` (the whole frame when None).  Returns ``block`` itself
    when nothing applies, else a retouched copy.  Exact for any block from ``needed``.  ``keep``:
    the call renders one block of a view (the 100 % view's tiles), so retouched faces are kept for
    the next block (``_retouched``)."""
    h, w = shape[:2]
    ax0, ay0, ax1, ay1 = area or (0, 0, w, h)
    out = None
    for face in faces:
        x0, y0, x1, y1 = window(face, h, w)
        x0, y0, x1, y1 = max(x0, ax0), max(y0, ay0), min(x1, ax1), min(y1, ay1)
        if x1 - x0 < 8 or y1 - y0 < 8 or not any(face['settings'].get(k, 0) for k in KEYS):
            continue
        if out is None:
            out = np.array(block, np.float32, copy=True)
        patch = out[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0]
        patch[:] = _retouched(patch, face, h, w, (x0, y0, x1, y1), keep)
    return block if out is None else out


#: 2.3.0: retouched faces kept for the blocks of the 100 % view: every block a face reaches needs
#: the whole face (3.8 s for one 15 MP portrait), so it is computed once and copied from here.
CACHE_BYTES = 512 * 2 ** 20
_cache = OrderedDict()
_cache_lock = threading.Lock()
_computing = {}


def _retouched(patch, face, h, w, box, blocks=True):
    """``retouch_face`` of a face's window ``patch``, from the cache when the same face with the same
    settings was retouched on the same pixels (a checksum of the window) before."""
    points = pixels(face, h, w) - [box[0], box[1]]
    if not blocks or patch.size < 1 << 20:      # whole frames (exports, previews) are not kept
        return retouch_face(patch, points, face['settings'])
    key = (h, w, box, zlib.crc32(np.ascontiguousarray(patch).data), zlib.crc32(repr((face['points'], sorted(face['settings'].items()))).encode()))
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
        lock = _computing.setdefault(key, threading.Lock())
    with lock:                                   # one block computes a face, the others wait for it
        with _cache_lock:
            if key in _cache:
                return _cache[key]
        result = retouch_face(patch, points, face['settings'])
        result.flags.writeable = False
        with _cache_lock:
            _cache[key] = result
            _computing.pop(key, None)
            while sum(v.nbytes for v in _cache.values()) > CACHE_BYTES and len(_cache) > 1:
                _cache.popitem(last=False)
        return result


# ------------------------------------------------------------------------------------ retouching

def _blur(x, sigma):
    if sigma < .3:
        return x
    return cv2.GaussianBlur(x, (0, 0), float(sigma), borderType=cv2.BORDER_REFLECT)


def _fill(shape, points, indices, scale=1.):
    mask = np.zeros(shape[:2], np.float32)
    polygon = np.round(np.asarray([points[i] for i in indices]) * scale * 16).astype(np.int32)
    cv2.fillPoly(mask, [polygon], 1., lineType=cv2.LINE_AA, shift=4)
    return mask


def _luma(p):
    return p[..., 0] * .2126 + p[..., 1] * .7152 + p[..., 2] * .0722


def _smoothstep(x, low, high):
    t = np.clip((x - low) / max(high - low, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


def _resize(x, size):
    return cv2.resize(x, size, interpolation=cv2.INTER_AREA if size[0] < x.shape[1] else cv2.INTER_LINEAR)


class Face:
    """Landmarks of one face in its window, the working scale and the masks the tools share."""

    def __init__(self, p, points):
        self.p = p
        self.points = points
        self.width, self.height = face_size(points)
        self.size = max(self.width, self.height, 8.)
        h, w = p.shape[:2]
        self.scale = min(1., WORK_SIZE / self.size)
        self.small_size = (max(8, round(w * self.scale)), max(8, round(h * self.scale)))
        self._skin = None

    def small(self, x):
        return x if self.scale >= 1 else _resize(x, self.small_size)

    def large(self, x):
        h, w = self.p.shape[:2]
        return x if self.scale >= 1 else cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)

    def skin(self):
        """Soft skin mask over the window, computed at the working scale."""
        if self._skin is None:
            self._skin = self._skin_mask()
        return self._skin

    def _skin_mask(self):
        s, pts = self.scale, self.points
        small = self.small(self.p)
        shape = small.shape
        size = self.size * s
        oval = _blur(_fill(shape, pts, FACE_OVAL, s), .02 * size)
        # The neck: from the jaw down by most of a face height, through the skin-colour model below.
        jaw = [pts[i] * s for i in (132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361)]
        down = (pts[CHIN] - pts[BROW_MIDDLE]) * s
        down /= max(np.linalg.norm(down), 1e-6)
        neck = np.zeros(shape[:2], np.float32)
        polygon = np.asarray(jaw + [jaw[-1] + down * .75 * self.height * s, jaw[0] + down * .75 * self.height * s])
        cv2.fillPoly(neck, [np.round(polygon * 16).astype(np.int32)], 1., lineType=cv2.LINE_AA, shift=4)
        neck = _blur(neck, .08 * size) * _smoothstep(_blur(neck, .08 * size), .2, .6)
        region = np.maximum(oval, .9 * neck)
        excluded = np.zeros(shape[:2], np.float32)
        for indices, grow in ((RIGHT_EYE, .035), (LEFT_EYE, .035), (RIGHT_BROW, .02), (LEFT_BROW, .02),
                              (LIPS_OUTER, .015), (NOSTRILS, .0)):
            part = _fill(shape, pts, indices, s)
            radius = max(1, round(grow * size))
            if grow:
                part = cv2.dilate(part, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
            excluded = np.maximum(excluded, part)
        excluded = _blur(excluded, .012 * size)
        return region * (1 - excluded) * self._likelihood(small, size) * (1 - self._frames(small, size))

    def _frames(self, small, size):
        """Thin long lines across the skin at eye height: glasses frames, which the skin tools would
        blur away.  Thin bright and thin dark lines (top-hat and black-hat), well above the skin's
        own texture, in one piece at least a tenth of the face long."""
        s, pts = self.scale, self.points
        shape = small.shape
        luma = _luma(small)
        radius = max(1, round(.012 * size))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        ridge = np.maximum(cv2.morphologyEx(luma, cv2.MORPH_TOPHAT, kernel), cv2.morphologyEx(luma, cv2.MORPH_BLACKHAT, kernel))
        h, w = shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        down = self.down()
        along = (xx - pts[BROW_MIDDLE][0] * s) * float(down[0]) + (yy - pts[BROW_MIDDLE][1] * s) * float(down[1])
        band = _fill(shape, pts, FACE_OVAL, s) * ((along > -.12 * self.height * s) & (along < .24 * self.height * s))
        eyes = np.zeros(shape[:2], np.float32)
        for indices in (RIGHT_EYE, LEFT_EYE, RIGHT_BROW, LEFT_BROW):
            eyes = np.maximum(eyes, _fill(shape, pts, indices, s))
        grow = max(1, round(.012 * size))
        eyes = cv2.dilate(eyes, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
        search = (band > .5) & (eyes < .5)
        if search.sum() < 50:
            return np.zeros(shape[:2], np.float32)
        values = ridge[search]
        middle = float(np.median(values))
        limit = max(middle + 4.5 * 1.4826 * float(np.median(np.abs(values - middle))), .05 * float(np.median(luma[search])))
        found = ((ridge > limit) & search).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(found, 8)
        if count < 2:
            return np.zeros(shape[:2], np.float32)
        depth = np.zeros(count, np.float32)
        np.maximum.at(depth, labels.ravel(), cv2.distanceTransform(found, cv2.DIST_L2, 3).ravel())
        long = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) >= .1 * size
        keep = long & (2 * depth <= .035 * size)
        keep[0] = False
        if not keep.any():
            return np.zeros(shape[:2], np.float32)
        lines = keep[labels].astype(np.float32)
        grow = max(1, round(.006 * size))
        lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
        return np.clip(_blur(lines, max(.5, .003 * size)) * 1.3, 0, 1)

    def open_skin(self):
        """Where 祛痘 looks for spots (bool, working scale): well inside the face oval (the neck and
        whatever lies against the face are not its skin), away from the features and from the
        creases and shadows every face has."""
        s, pts = self.scale, self.points
        shape = self.small(self.p).shape
        size = self.size * s

        def disc(radius):
            radius = max(1, round(radius * size))
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        oval = cv2.erode(_fill(shape, pts, FACE_OVAL, s), disc(.03))
        closed = np.zeros(shape[:2], np.float32)
        for indices, grow in ((RIGHT_EYE, .05), (LEFT_EYE, .05), (RIGHT_BROW, .03), (LEFT_BROW, .03), (LIPS_OUTER, .03), (NOSTRILS, .02)):
            closed = np.maximum(closed, cv2.dilate(_fill(shape, pts, indices, s), disc(grow)))
        lines = np.zeros(shape[:2], np.uint8)

        def stroke(indices, radius):
            path = np.round(np.asarray([pts[i] for i in indices]) * s).astype(np.int32)
            cv2.polylines(lines, [path], False, 1, max(1, round(2 * radius * size)), cv2.LINE_8)
        for path in NOSE_SIDES:
            stroke(path, .032)
        for path in SMILE_LINES:
            stroke(path, .04)
        stroke(CHIN_CREASE, .025)
        stroke(CHIN_DIMPLE, .03)
        # Under the eyes: the bags and the shadow of the lid.
        down = self.down()
        for right in (True, False):
            lid, width, _ = self.eye(right)
            lid = np.asarray([pts[i] for i in lid])
            below = np.concatenate([lid, (lid + down * width * .75)[::-1]]) * s
            cv2.fillPoly(lines, [np.round(below).astype(np.int32)], 1)
        return (oval > .5) & (closed < .5) & (lines == 0)

    def _likelihood(self, small, size):
        """How skin-like each pixel's chromaticity is, learnt from the cheeks and forehead."""
        # Chroma noise (JPEG blocks, high ISO) would speckle the mask: compare smoothed colour.
        smooth = _blur(small, max(.7, .006 * size))
        total = np.maximum(smooth.sum(axis=2), 1e-4)
        chroma = np.stack([smooth[..., 0] / total, smooth[..., 1] / total], axis=2)
        luma = _luma(smooth)
        samples = np.zeros(small.shape[:2], np.uint8)
        for index in SKIN_SAMPLES:
            x, y = self.points[index] * self.scale
            cv2.circle(samples, (round(x), round(y)), max(2, round(.045 * size)), 1, -1)
        chosen = (samples > 0) & (luma > .05)
        if chosen.sum() < 20:
            return np.ones(small.shape[:2], np.float32)
        values = chroma[chosen]
        mean = np.median(values, axis=0)
        covariance = np.cov((values - mean).T) + np.eye(2) * .006 ** 2
        inverse = np.linalg.inv(covariance)
        d = chroma - mean
        distance = d[..., 0] ** 2 * inverse[0, 0] + 2 * d[..., 0] * d[..., 1] * inverse[0, 1] + d[..., 1] ** 2 * inverse[1, 1]
        colour = np.exp(-distance / (2 * 3. ** 2))
        bright = _smoothstep(luma / max(float(np.median(luma[chosen])), 1e-4), .25, .5)
        return _blur((colour * bright).astype(np.float32), max(.7, .008 * size))

    def eye(self, right):
        lid = RIGHT_LOWER_LID if right else LEFT_LOWER_LID
        corners = self.points[lid[0]], self.points[lid[-1]]
        width = float(np.linalg.norm(corners[0] - corners[1]))
        iris = self.points[RIGHT_IRIS if right else LEFT_IRIS]
        centre = .6 * iris + .4 * (corners[0] + corners[1]) / 2
        return lid, width, centre

    def down(self):
        d = self.points[CHIN] - self.points[BROW_MIDDLE]
        return d / max(np.linalg.norm(d), 1e-6)


def retouch_face(patch, points, values):
    """One face's retouching of ``patch`` (scene-linear, its window); ``points`` in patch pixels."""
    colour = any(values.get(k, 0) for k in COLOUR_KEYS)
    out = patch
    if colour:
        x = np.maximum(patch, 0)
        p = x ** (1 / 2.2)
        face = Face(p, points)
        if values.get('acne'):
            p = _acne(p, face, values['acne'] / 100)
        if values.get('smooth'):
            p = _smooth(p, face, values['smooth'] / 100)
        if values.get('circles'):
            p = _circles(p, face, values['circles'] / 100)
        if values.get('whiten'):
            p = _whiten(p, face, values['whiten'] / 100)
        if values.get('teeth'):
            p = _teeth(p, face, values['teeth'] / 100)
        out = (patch + (np.maximum(p, 0) ** 2.2 - x)).astype(np.float32)
    if values.get('eyes') or values.get('slim'):
        out = _warp(out, points, values.get('eyes', 0) / 100, values.get('slim', 0) / 100)
    return out


def _guided(guide, src, radius, eps):
    size = (2 * radius + 1, 2 * radius + 1)
    box = lambda v: cv2.boxFilter(v, -1, size, borderType=cv2.BORDER_REFLECT)
    mean_i, mean_p = box(guide), box(src)
    a = (box(guide * src) - mean_i * mean_p) / (box(guide * guide) - mean_i * mean_i + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


def _smooth(p, face, amount):
    small = face.small(p)
    size = face.size * face.scale
    radius = max(2, round(.028 * size))
    eps = (.012 + .03 * amount) ** 2
    base = np.stack([_guided(small[..., c], small[..., c], radius, eps) for c in range(3)], axis=2)
    # A second, wider pass evens out larger blotches at high settings.
    if amount > .5:
        wide = np.stack([_guided(base[..., c], base[..., c], radius * 2, eps * 1.5) for c in range(3)], axis=2)
        base = base + (wide - base) * ((amount - .5) * 2)
    skin = face.skin()
    change = face.large((base - small) * skin[..., None])
    result = p + change * min(1., amount * 1.1)
    if amount > .3:
        # Pores stay, slightly softened at high settings.
        sigma = max(.5, .0025 * face.size)
        fine = p - _blur(p, sigma)
        result -= fine * (face.large(skin)[..., None] * (amount - .3) * .45)
    return result


def _acne(p, face, amount):
    small = face.small(p)
    size = face.size * face.scale
    skin = face.skin()
    luma = _luma(small)
    red = small[..., 0] - small[..., 1]
    core, around = max(.7, .006 * size), max(2., .03 * size)
    # Morphological closing against the lightly smoothed skin: only what is narrower than a large
    # spot stands out, and the shading of the face, or hair and eyes nearby, adds nothing.
    reach = max(1, round(.028 * size))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    dark = cv2.morphologyEx(_blur(luma, core * .6), cv2.MORPH_BLACKHAT, kernel)
    redder = cv2.morphologyEx(_blur(red, core * .6), cv2.MORPH_TOPHAT, kernel)
    # Only well inside the skin: beside eyes, brows, lips, nostrils or hair the dark feature
    # itself would look like a spot.
    edge = max(1, round(around * .7))
    border = cv2.dilate((skin < .15).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge + 1, 2 * edge + 1)))
    inside = (skin > .35) & (border == 0) & face.open_skin()
    if inside.sum() < 50:
        return p
    level = float(np.median(luma[inside]))

    def standing_out(v, floor):
        # How far above this face's own skin texture.  Clear skin has almost no spread: the floor is
        # the contrast a spot needs to be seen at all.
        values = v[inside]
        middle = float(np.median(values))
        return (v - middle) / max(float(np.median(np.abs(values - middle))) * 1.4826, floor * level, 1e-4)
    score = np.maximum(standing_out(dark, .012), standing_out(redder, .007))
    threshold = LIGHTEST_SPOT - (LIGHTEST_SPOT - FAINTEST_SPOT) * amount
    # Far darker than the skin, or colder (blemishes redden, they do not turn blue), is not a
    # blemish: moles, tattoos and painted marks stay.
    warmth = _blur(small[..., 0] - small[..., 2], core * .6)
    usual = float(np.median(warmth[inside]))
    marks = (dark > MARK_CONTRAST * level) | (warmth < usual - max(.5 * abs(usual), .03))
    spots = _round_spots(((score > threshold) & inside).astype(np.uint8), size, marks)
    if not spots.any():
        return p
    # A spot fades out: the fill reaches nearby pixels that still stand out, and no further.
    reach = max(1, round(.014 * size))
    near = cv2.dilate(spots, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1)))
    spots = np.maximum(spots, ((score > threshold * .55) & inside & (near > 0)).astype(np.float32))
    radius = max(1, round(core * 1.6))
    spots = cv2.dilate(spots, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    mask = np.clip(_blur(spots, core) * 1.6, 0, 1) * np.clip(skin * 2, 0, 1)
    # Filled from the clear skin around the spot only (normalised convolution).
    clear = (1 - np.clip(_blur(spots, core * .5) * 2, 0, 1)) * skin
    sigma = max(1.5, .025 * size)
    fill = _blur(small * clear[..., None], sigma) / np.maximum(_blur(clear, sigma), 1e-4)[..., None]
    mask, fill = face.large(mask), face.large(fill - small) + p
    # Fine skin texture from beside the spot, so the filled place is not flat.
    sigma = max(.5, .003 * face.size)
    fine = np.clip(p - _blur(p, sigma), -.03, .03) * face.large(skin)[..., None]
    shift = max(2, round(.035 * face.size))
    fill += .7 * np.roll(fine, shift, axis=1)
    strength = min(1., .6 + amount)
    return p + (fill - p) * (mask * strength)[..., None]


def _round_spots(found, size, marks=None):
    """The parts of ``found`` (uint8) shaped like blemishes: small, round and solid.  Creases, hair
    across the skin and the edges of shadows come out long or hollow.  A part that touches ``marks``
    (bool) is left out."""
    count, labels, _, _ = cv2.connectedComponentsWithStats(found, 8)
    if count < 2:
        return np.zeros(found.shape, np.float32)
    ys, xs = np.nonzero(labels)
    which = labels[ys, xs]
    marked = np.zeros(count, bool) if marks is None else np.bincount(which, marks[ys, xs], count) > 0
    area = np.bincount(which, minlength=count).astype(np.float64)
    area[0] = 1
    mx, my = np.bincount(which, xs, count) / area, np.bincount(which, ys, count) / area
    dx, dy = xs - mx[which], ys - my[which]
    xx = np.bincount(which, dx * dx, count) / area + 1 / 12
    yy = np.bincount(which, dy * dy, count) / area + 1 / 12
    xy = np.bincount(which, dx * dy, count) / area
    middle, reach = (xx + yy) / 2, np.sqrt(((xx - yy) / 2) ** 2 + xy ** 2)
    # An ellipse's axes are four standard deviations long.
    major, minor = 4 * np.sqrt(middle + reach), 4 * np.sqrt(np.maximum(middle - reach, 1e-6))
    solid = area >= .55 * np.pi / 4 * major * minor
    keep = (major <= .06 * size) & (major <= 2.2 * minor) & solid & (area >= max(4., (.006 * size) ** 2)) & ~marked
    keep[0] = False
    return keep[labels].astype(np.float32)


def _circles(p, face, amount):
    small = face.small(p)
    s = face.scale
    total = np.zeros(small.shape[:2], np.float32)
    change = np.zeros_like(small)
    down = face.down()
    for right in (True, False):
        lid, width, _ = face.eye(right)
        width *= s
        if width < 4:
            continue
        lid_points = np.asarray([face.points[i] * s for i in lid])
        # Deepest in the middle of the eye, tapering to the corners.
        t = np.linspace(0, 1, len(lid_points))[:, None]
        bulge = np.sin(np.pi * t) ** .7
        top = lid_points + down * width * .1
        bottom = lid_points + down * width * (.12 + .45 * bulge)
        region = np.zeros(small.shape[:2], np.float32)
        cv2.fillPoly(region, [np.round(np.concatenate([top, bottom[::-1]]) * 16).astype(np.int32)], 1.,
                     lineType=cv2.LINE_AA, shift=4)
        region = _blur(region, .1 * width)
        band = np.zeros(small.shape[:2], np.uint8)
        reference = np.concatenate([lid_points[1:-1] + down * width * .75, (lid_points[1:-1] + down * width * 1.05)[::-1]])
        cv2.fillPoly(band, [np.round(reference).astype(np.int32)], 1)
        weights = band.astype(np.float32) * face.skin()
        if weights.sum() < 4:
            continue
        target = (small * weights[..., None]).sum(axis=(0, 1)) / weights.sum()
        low = _blur(small, .15 * width)
        lift = target - low
        brighter = _luma(lift)
        lift *= np.where(brighter > 0, 1., .25)[..., None]
        change += lift * region[..., None]
        total = np.maximum(total, region)
    if not total.any():
        return p
    return p + face.large(change) * min(1., amount * 1.25)


def _whiten(p, face, amount):
    skin = face.large(face.skin())[..., None]
    clipped = np.clip(p, 0, 1)
    lifted = 1 - (1 - clipped) ** (1 + .55 * amount)
    luma = _luma(lifted)[..., None]
    lifted = luma + (lifted - luma) * (1 - .12 * amount)
    lifted = np.where(p > 1, p, lifted)
    return p + (lifted - p) * skin


def _teeth(p, face, amount):
    s = face.scale
    small = face.small(p)
    size = face.size * s
    mouth = _fill(small.shape, face.points, LIPS_INNER, s)
    radius = max(1, round(.006 * size))
    mouth = cv2.erode(mouth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    inside = mouth > .5
    if inside.sum() < max(12, (.015 * size) ** 2):
        return p
    luma = _luma(small)
    level = float(np.percentile(luma[inside], 80))
    red = small[..., 0] - (small[..., 1] + small[..., 2]) / 2
    teeth = _smoothstep(luma, .55 * level, .85 * level) * (1 - _smoothstep(red, .07, .16))
    mask = face.large(_blur(mouth * teeth, max(.6, .004 * size)))[..., None]
    clipped = np.clip(p, 0, 1)
    luma = _luma(clipped)[..., None]
    yellow = np.maximum((clipped[..., :1] + clipped[..., 1:2]) / 2 - clipped[..., 2:], 0)
    white = clipped.copy()
    white[..., 2:] += yellow * .85
    white[..., :2] -= yellow * .15
    white = luma + (white - luma) * .55
    white = 1 - (1 - np.clip(white, 0, 1)) ** 1.5
    white = np.where(p > 1, p, white)
    return p + (white - p) * mask * min(1., amount)


def _warp(image, points, eyes, slim):
    """大眼 and 瘦脸 as one inverse map over the window (no displacement at its edges)."""
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    width, height = face_size(points)
    if eyes:
        for right in (True, False):
            lid = RIGHT_LOWER_LID if right else LEFT_LOWER_LID
            corners = points[lid[0]], points[lid[-1]]
            eye_width = float(np.linalg.norm(corners[0] - corners[1]))
            centre = .6 * points[RIGHT_IRIS if right else LEFT_IRIS] + .4 * (corners[0] + corners[1]) / 2
            radius = 1.1 * eye_width
            if radius < 3:
                continue
            x0, y0 = np.maximum(0, np.floor(centre - radius).astype(int))
            x1, y1 = np.minimum([w, h], np.ceil(centre + radius).astype(int) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            ox, oy = xx[y0:y1, x0:x1] - centre[0], yy[y0:y1, x0:x1] - centre[1]
            r = np.sqrt(ox * ox + oy * oy) / radius
            pull = np.where(r < 1, EYE_SCALE * eyes * (1 - r * r) ** 2, 0).astype(np.float32)
            # Destination q samples centre + (q - centre)(1 - pull): magnified at the iris.
            dx[y0:y1, x0:x1] -= ox * pull
            dy[y0:y1, x0:x1] -= oy * pull
    if slim:
        # Each jaw point moves towards the face's axis; between them the shift is blended with
        # weights that vanish smoothly within a radius, normalised where they overlap, so the
        # field never exceeds the largest shift and stays far too gentle to fold.
        top, chin = points[BROW_MIDDLE], points[CHIN]
        axis = (chin - top) / max(np.linalg.norm(chin - top), 1e-6)
        radius = .34 * width
        weights = np.zeros((h, w), np.float32)
        sx = np.zeros((h, w), np.float32)
        sy = np.zeros((h, w), np.float32)
        for index, strength in JAW:
            c = points[index]
            foot = top + axis * float(np.dot(c - top, axis))
            direction = foot - c
            length = float(np.linalg.norm(direction))
            if length < 1 or radius < 3:
                continue
            m = direction / length * SLIM_SHIFT * slim * strength * width
            x0, y0 = np.maximum(0, np.floor(c - radius).astype(int))
            x1, y1 = np.minimum([w, h], np.ceil(c + radius).astype(int) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            ox, oy = xx[y0:y1, x0:x1] - c[0], yy[y0:y1, x0:x1] - c[1]
            r = np.minimum(np.sqrt(ox * ox + oy * oy) / radius, 1)
            weight = ((1 - r * r) ** 2).astype(np.float32)
            weights[y0:y1, x0:x1] += weight
            sx[y0:y1, x0:x1] += weight * np.float32(m[0])
            sy[y0:y1, x0:x1] += weight * np.float32(m[1])
        scale = 1 / np.maximum(weights, 1)
        # Destination x samples x - shift: the content just outside the jaw moves in.
        dx -= sx * scale
        dy -= sy * scale
    if not dx.any() and not dy.any():
        return image
    return cv2.remap(image, xx + dx, yy + dy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
