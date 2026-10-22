"""Ordered, non-destructive inpainting, aligned cloning and AI removal in source coordinates."""
import cv2
import numpy as np


def _window(op, h, w):
    if op['kind'] == 'remove':
        # 2.1.0: the stored patch decides what an AI removal covers (see lumen/remove.py).
        from . import remove
        return None, None, remove.window(op['patch'], h, w)
    pts = np.asarray(op['points']) * [w - 1, h - 1]
    radius = max(1., op['radius'] * min(h, w))
    margin = int(np.ceil(radius * 2 + 8))
    x0, y0 = np.maximum(0, np.floor(pts.min(axis=0)).astype(int) - margin)
    x1, y1 = np.minimum([w, h], np.ceil(pts.max(axis=0)).astype(int) + margin + 1)
    return pts, radius, (int(x0), int(y0), int(x1), int(y1))


def _reads(op, window, h, w):
    """Pixels an operation reads: its window, plus the shifted window a clone samples from."""
    x0, y0, x1, y1 = window
    if op['kind'] != 'clone':
        return [window]
    dx, dy = np.asarray(op['offset']) * [w - 1, h - 1]
    sx0, sy0 = max(0, int(np.floor(x0 + dx)) - 1), max(0, int(np.floor(y0 + dy)) - 1)
    sx1, sy1 = min(w, int(np.ceil(x1 - 1 + dx)) + 2), min(h, int(np.ceil(y1 - 1 + dy)) + 2)
    return [window] + ([(sx0, sy0, sx1, sy1)] if sx0 < sx1 and sy0 < sy1 else [])


def stroke(op, h, w):
    """Coverage of a brush operation over its window ``(x0, y0, x1, y1)`` of a frame ``h × w``:
    the painted stroke, feathered inward so pixels outside it stay exactly intact."""
    pts, radius, window = _window(dict(op, kind='heal'), h, w)
    x0, y0, x1, y1 = window
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    local = np.round(pts - [x0, y0]).astype(np.int32)
    r = max(1, round(radius))
    cv2.polylines(mask, [local], False, 255, r * 2, cv2.LINE_8)
    for p in (local[0], local[-1]):
        cv2.circle(mask, tuple(p), r, 255, -1)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    return np.minimum(1, distance / max(1, radius * op['feather'] / 100)), mask, radius, window


def _union(a, b):
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def apply(source, operations, rect=None):
    """Apply the enabled operations in order.

    With ``rect = (x0, y0, x1, y1)`` only that block is returned (1.4.0): it reads the pixels the
    relevant operations depend on, including clone sources, and matches the whole-frame result.
    """
    active = [o for o in operations if o['enabled'] and o['opacity']]
    h, w = source.shape[:2]
    if not active:
        return source if rect is None else source[rect[1]:rect[3], rect[0]:rect[2]]
    if rect is None:
        needed, ops = (0, 0, w, h), active
    else:
        # Walk backwards: an operation matters when it writes pixels the block, or a later
        # relevant operation, reads.  The block to process grows by what it reads in turn.
        needed, ops = tuple(rect), []
        for op in reversed(active):
            _, _, window = _window(op, h, w)
            if _overlaps(window, needed):
                ops.insert(0, op)
                for area in _reads(op, window, h, w):
                    needed = _union(needed, area)
    nx0, ny0, nx1, ny1 = needed
    out = source[ny0:ny1, nx0:nx1].copy()
    for op in ops:
        if op['kind'] == 'remove':
            from . import remove
            x0, y0, x1, y1 = remove.window(op['patch'], h, w)
            if x0 < x1 and y0 < y1:
                current = out[y0 - ny0:y1 - ny0, x0 - nx0:x1 - nx0]
                current[:] = remove.composite(current, op['patch'], op['opacity'], h, w)
            continue
        coverage, mask, radius, (x0, y0, x1, y1) = stroke(op, h, w)
        alpha = coverage * op['opacity'] / 100
        current = out[y0 - ny0:y1 - ny0, x0 - nx0:x1 - nx0]
        if op['kind'] == 'clone':
            dx, dy = np.asarray(op['offset']) * [w - 1, h - 1]
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            mx, my = xx + dx, yy + dy
            alpha *= (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
            # Whole-pixel shifts keep the sub-pixel positions of the full-frame maps exact.
            replacement = cv2.remap(out, mx.astype(np.float32) - np.float32(nx0), my.astype(np.float32) - np.float32(ny0),
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            # NS supports single-channel float32, preserving high precision and HDR
            # values. Small sensor-dust regions work best; this is not generative AI.
            replacement = np.stack([cv2.inpaint(np.ascontiguousarray(current[..., c]), mask,
                max(2, min(12, radius / 3)), cv2.INPAINT_NS) for c in range(3)], axis=2)
        out[y0 - ny0:y1 - ny0, x0 - nx0:x1 - nx0] = current * (1 - alpha[..., None]) + replacement * alpha[..., None]
    if rect is None:
        return out
    return out[rect[1] - ny0:rect[3] - ny0, rect[0] - nx0:rect[2] - nx0]
