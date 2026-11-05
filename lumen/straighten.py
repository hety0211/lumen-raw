"""Automatic levelling (2.3.0, 自动水平 in the crop toolbar): the straighten angle that makes a
photo's horizon and verticals straight.

Written for LUMEN from the general method of line-based upright adjustment (e.g. Lee et al.,
"Automatic Upright Adjustment of Photographs", CVPR 2012), without its perspective part:

1. Line segments are found with OpenCV's LSD detector on a grey copy (long edge ``EDGE``).
2. Segments within ``SEARCH`` degrees of horizontal vote for the angle that levels them, those
   within ``SEARCH`` degrees of vertical for the angle that makes them upright.  A vote weighs
   the segment's length; long horizontals (a sea or lake horizon) count more, and verticals
   less the farther they are from the middle of the frame, where converging lines (a camera
   tilted up at a building) lean the most.
3. The votes are smoothed (Gaussian, ``SPREAD`` degrees) and the strongest peak wins, refined
   by the mean of the votes near it.  Without enough agreeing line length there is no answer.

The angle follows ``engine.crop_rotate``: positive turns the picture counter-clockwise.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

EDGE = 1200
#: Largest correction looked for, in degrees.
SEARCH = 20.
#: Width of a vote, degrees.
SPREAD = .35
#: Agreeing line length needed, as a share of the long edge.
SUPPORT = .3


def segments(rgb):
    """Line segments ``[x0, y0, x1, y1]`` of an RGB (or grey) image at long edge ``EDGE``, and
    that image's size."""
    image = np.asarray(rgb)
    if image.dtype != np.uint8:
        image = np.clip(image * 255 + .5, 0, 255).astype(np.uint8)
    grey = cv2.cvtColor(np.ascontiguousarray(image[..., :3]), cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    h, w = grey.shape
    scale = EDGE / max(h, w)
    if scale < 1:
        grey = cv2.resize(grey, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    detector = cv2.createLineSegmentDetector()
    lines = detector.detect(grey)[0]
    return (np.zeros((0, 4), np.float32) if lines is None else lines.reshape(-1, 4)), grey.shape[:2]


def votes(lines, shape):
    """``(angles, weights)``: what each segment says the straighten angle is."""
    h, w = shape
    long_edge = max(h, w)
    angles, weights = [], []
    for x0, y0, x1, y1 in lines:
        dx, dy = float(x1 - x0), float(y1 - y0)
        length = math.hypot(dx, dy)
        if length < .04 * long_edge:
            continue
        if dx < 0:
            dx, dy = -dx, -dy
        slope = math.degrees(math.atan2(dy, dx))            # (-90, 90]
        if abs(slope) <= SEARCH:
            # Rotating counter-clockwise by the slope levels the segment.
            angles.append(slope)
            weights.append(length * (2. if length > .25 * w else 1.))
        elif abs(slope) >= 90 - SEARCH:
            if dy < 0:
                dx, dy = -dx, -dy
            angles.append(math.degrees(math.atan2(-dx, dy)))
            middle = abs((x0 + x1) / 2 - w / 2) / (w / 2)
            weights.append(length * .8 * (1 - .6 * min(1., middle)))
    return np.asarray(angles, np.float64), np.asarray(weights, np.float64)


def estimate(rgb):
    """``(angle, confidence)`` for a display RGB image of the frame, or ``(None, 0)`` when its
    lines do not agree on one."""
    lines, shape = segments(rgb)
    angles, weights = votes(lines, shape)
    if not len(angles):
        return None, 0.
    grid = np.arange(-SEARCH, SEARCH + 1e-9, .05)
    score = (weights[None, :] * np.exp(-(grid[:, None] - angles[None, :]) ** 2 / (2 * SPREAD ** 2))).sum(axis=1)
    peak = float(grid[int(np.argmax(score))])
    near = np.abs(angles - peak) <= 2 * SPREAD
    support = float(weights[near].sum())
    if support < SUPPORT * max(shape):
        return None, support / max(shape)
    angle = float(np.average(angles[near], weights=weights[near]))
    confidence = support / float(weights.sum())
    return round(angle, 2), round(min(1., confidence), 3)
