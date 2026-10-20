"""Culling hints (2.0.0): how sharp a photo is, its faces and whether their eyes are closed.

Everything is computed from the camera view the library shows (``previews.camera_view``, at most
2560 pixels), so it costs no RAW decode:

* **Sharpness** (0–100) compares, in 32-pixel tiles of a 1600-pixel greyscale copy, the energy of
  the second derivative with that of the first along four directions.  The ratio falls as detail
  is blurred and does not depend on contrast, so a misty landscape is not called soft.  The
  sharpest tenth of the tiles with the most structure counts (a sharp subject before a blurred
  background is sharp), in the direction where it is lowest (motion blur smears one direction).
  On a 1600-pixel view a 1.2-pixel Gaussian blur scores about 55, 2 pixels about 30.
* **Faces** come from YuNet (OpenCV's ``FaceDetectorYN``); faces at least 3 % of the long edge are
  counted.  For each, Google's MediaPipe Face Mesh places 478 landmarks in a crop rolled so the
  eyes are level (refined once from its own points), and an eye's opening is the distance between
  its lids over its width.  A face whose wider-open eye is below 0.16 has its eyes closed.  This
  is a hint: faces in profile, heavily tilted or partly covered are often missed.
* A 16 × 16 greyscale **signature** tells frames of one scene from the next (burst grouping).
"""
from __future__ import annotations

import logging
import math
import threading
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

MODELS = Path(__file__).resolve().parents[1] / 'assets' / 'models'
DETECTOR = MODELS / 'face-yunet.onnx'
LANDMARKS = MODELS / 'face-landmarks.onnx'
#: Long edge of the greyscale copy sharpness is measured on.
SHARPNESS_EDGE = 1600
#: Long edge faces are detected on (YuNet finds faces from about 10 pixels).
DETECT_EDGE = 1280
#: Faces smaller than this share of the long edge are not counted.
MIN_FACE = .03
#: Eye opening (lid distance / eye width) below which an eye is closed.
CLOSED = .16
#: Face Mesh indices: outer and inner corner, then upper / lower lid pairs.
RIGHT_EYE = (33, 133, (159, 145), (158, 153), (160, 144))
LEFT_EYE = (263, 362, (386, 374), (385, 380), (387, 373))

_FIRST = {0: np.array([[-.5, 0, .5]], np.float32), 90: np.array([[-.5], [0], [.5]], np.float32),
          45: np.array([[0, 0, .5], [0, 0, 0], [-.5, 0, 0]], np.float32) / np.sqrt(2),
          135: np.array([[-.5, 0, 0], [0, 0, 0], [0, 0, .5]], np.float32) / np.sqrt(2)}
_SECOND = {0: np.array([[1, -2, 1]], np.float32), 90: np.array([[1], [-2], [1]], np.float32),
           45: np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]], np.float32) / 2,
           135: np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]], np.float32) / 2}


def _resize(image, long_edge):
    scale = long_edge / max(image.shape[:2])
    if scale >= 1:
        return image
    return cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA)


def _tiles(values, size=32):
    h, w = values.shape
    h, w = h // size * size, w // size * size
    return values[:h, :w].reshape(h // size, size, w // size, size).mean(axis=(1, 3))


def sharpness(gray):
    """Sharpness 0–100 of a greyscale (or RGB) image, float 0–1 or uint8."""
    gray = np.asarray(gray)
    if gray.ndim == 3:
        gray = cv2.cvtColor(np.ascontiguousarray(gray[..., :3]), cv2.COLOR_RGB2GRAY)
    if gray.dtype == np.uint8:
        gray = gray.astype(np.float32) / 255
    gray = _resize(gray.astype(np.float32, copy=False), SHARPNESS_EDGE)
    if min(gray.shape) < 64:
        return None
    gray = cv2.GaussianBlur(gray, (0, 0), .5)          # JPEG blocks and noise are not detail
    levels = []
    for direction in (0, 45, 90, 135):
        first = _tiles(cv2.filter2D(gray, cv2.CV_32F, _FIRST[direction]) ** 2)
        second = _tiles(cv2.filter2D(gray, cv2.CV_32F, _SECOND[direction]) ** 2)
        strong = (first >= np.percentile(first, 75)) & (first > 2e-5)
        if not strong.any():
            return 0.
        levels.append(np.log2(np.percentile(second[strong] / first[strong], 90)))
    return float(np.clip((min(levels) + 4) / 5.5 * 100, 0, 100))


def signature(gray):
    small = cv2.resize(np.asarray(gray), (16, 16), interpolation=cv2.INTER_AREA)
    if small.dtype != np.uint8:
        small = np.clip(small * 255, 0, 255).astype(np.uint8)
    return small.tobytes()


class Faces:
    """Face detection and eye openness; one per process (``faces()``), safe across threads."""

    def __init__(self, detector=DETECTOR, landmarks=LANDMARKS):
        import onnxruntime as ort
        self.detector_path = str(detector)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.log_severity_level = 3
        self.mesh = ort.InferenceSession(str(landmarks), options, providers=['CPUExecutionProvider'])
        self.input = self.mesh.get_inputs()[0].name
        self._local = threading.local()

    def _detector(self, size):
        detector = getattr(self._local, 'detector', None)
        if detector is None:
            detector = cv2.FaceDetectorYN.create(self.detector_path, '', size, .7, .3, 50)
            self._local.detector = detector
        detector.setInputSize(size)
        return detector

    def detect(self, rgb):
        """Faces of an RGB uint8 image: ``[(x, y, w, h, score, right_eye, left_eye)]`` in its pixels."""
        small = _resize(rgb, DETECT_EDGE)
        scale = rgb.shape[1] / small.shape[1]
        bgr = np.ascontiguousarray(small[..., ::-1])
        _, found = self._detector((bgr.shape[1], bgr.shape[0])).detect(bgr)
        faces = []
        for face in (found if found is not None else []):
            x, y, w, h = (float(v) * scale for v in face[:4])
            faces.append((x, y, w, h, float(face[14]), (float(face[4]) * scale, float(face[5]) * scale),
                          (float(face[6]) * scale, float(face[7]) * scale)))
        return faces

    def _landmarks(self, rgb, cx, cy, size, angle):
        matrix = cv2.getRotationMatrix2D((cx, cy), angle, 256 / size)
        matrix[:, 2] += (128 - cx, 128 - cy)
        patch = cv2.warpAffine(rgb, matrix, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        output = self.mesh.run(None, {self.input: patch[None].astype(np.float32) / 255})
        points = output[0].reshape(-1, 3)[:, :2]
        presence = 1 / (1 + math.exp(-float(np.ravel(output[1])[0])))
        inverse = cv2.invertAffineTransform(matrix)
        return points @ inverse[:, :2].T + inverse[:, 2], presence

    @staticmethod
    def opening(points, eye):
        outer, inner, *pairs = eye
        width = float(np.linalg.norm(points[outer] - points[inner]))
        return float(np.mean([np.linalg.norm(points[a] - points[b]) for a, b in pairs])) / max(width, 1e-6)

    def eyes(self, rgb, face):
        """``(presence, right opening, left opening)`` of a detected face."""
        x, y, w, h, _, (rx, ry), (lx, ly) = face
        angle = math.degrees(math.atan2(ly - ry, lx - rx))
        points, presence = self._landmarks(rgb, x + w / 2, y + h / 2, 1.5 * max(w, h), angle)
        low, high = points.min(axis=0), points.max(axis=0)
        (cx, cy), size = (low + high) / 2, 1.5 * float(max(high - low))
        roll = math.degrees(math.atan2(*(points[LEFT_EYE[0]] - points[RIGHT_EYE[0]])[::-1]))
        if size > 8:
            points, presence = self._landmarks(rgb, float(cx), float(cy), size, roll)
        return presence, self.opening(points, RIGHT_EYE), self.opening(points, LEFT_EYE)

    def analyse(self, rgb):
        """``(faces, closed, boxes)``: counted faces, how many have closed eyes, and per face
        ``[x, y, w, h, closed]`` relative to the image."""
        height, width = rgb.shape[:2]
        smallest = MIN_FACE * max(height, width)
        boxes, closed = [], 0
        for face in self.detect(rgb):
            if max(face[2], face[3]) < smallest:
                continue
            try:
                presence, right, left = self.eyes(rgb, face)
            except Exception:
                log.debug('face landmarks failed', exc_info=True)
                continue
            if presence < .5:
                continue
            shut = max(right, left) < CLOSED
            closed += shut
            boxes.append([round(face[0] / width, 4), round(face[1] / height, 4), round(face[2] / width, 4),
                          round(face[3] / height, 4), int(shut)])
        return len(boxes), closed, boxes


_faces = None
_faces_lock = threading.Lock()


def faces():
    """The shared face analyser, or None when its models are missing or fail to load."""
    global _faces
    with _faces_lock:
        if _faces is None:
            try:
                _faces = Faces() if DETECTOR.is_file() and LANDMARKS.is_file() else False
            except Exception:
                log.warning('face models unavailable', exc_info=True)
                _faces = False
        return _faces or None


def analyse(rgb):
    """Culling hints of an upright RGB uint8 view: ``{sharpness, faces, eyes_closed, signature,
    boxes}`` (faces / eyes_closed are None when the face models are unavailable)."""
    gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    result = dict(sharpness=sharpness(gray), signature=signature(gray), faces=None, eyes_closed=None, boxes=[])
    analyser = faces()
    if analyser is not None:
        try:
            result['faces'], result['eyes_closed'], result['boxes'] = analyser.analyse(np.ascontiguousarray(rgb))
        except Exception:
            log.warning('face analysis failed', exc_info=True)
    return result
