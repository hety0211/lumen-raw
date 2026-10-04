"""Visible-area detail rendering for zoomed-in views (1.4.0).

Up to 1.3.1, zooming past the 1600-pixel preview rendered the whole photograph at
original resolution, and the GUI thread then converted that frame — plus the
original for comparison, clipping warnings and the crop view — into screen
images.  On a 61-megapixel Sony A7R V each of those steps produced several
700 MB arrays, so the window stopped responding for seconds after every edit.

The editor now renders only what is on screen, at the coarsest resolution level
(1, 1/2, 1/4 … of the original) that still has an image pixel for every screen
pixel.  Rendered blocks are cut into 512-pixel tiles and converted to QImage in
the worker; the GUI thread only draws finished tiles.  Tiles are cached, so
panning renders just the newly exposed tiles, and edits never reuse tiles of an
earlier state.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np
from PySide6.QtGui import QImage

from . import engine, lens
from .i18n import tr

log = logging.getLogger(__name__)

TILE = 512
CACHE_BYTES = 256 * 2**20
#: Blocks without spatial tools are cut into chunks of at most this many tiles, so the
#: first part of a large screen appears early; tools that need context render in one block.
CHUNK_TILES = 8
SHADOW_COLOR = (75, 134, 255, 210)
HIGHLIGHT_COLOR = (255, 75, 90, 210)


def level_size(width, height, level):
    """Frame size at resolution level ``level`` (1/2^level of the original)."""
    if not level:
        return width, height
    return max(1, round(width / 2 ** level)), max(1, round(height / 2 ** level))


def level_for(density):
    """Coarsest level that still shows at least one image pixel per device pixel."""
    if density >= 1:
        return 0
    return max(0, int(math.floor(math.log2(1 / density) + 1e-9)))


def visible_tiles(image_rect, viewport, size):
    """Tiles of a ``size`` frame drawn into ``image_rect`` that intersect ``viewport``.

    Both rectangles are ``(x, y, width, height)`` in the same (widget) coordinates.
    """
    width, height = size
    rx, ry, rw, rh = image_rect
    vx, vy, vw, vh = viewport
    left, right = max(rx, vx), min(rx + rw, vx + vw)
    top, bottom = max(ry, vy), min(ry + rh, vy + vh)
    if right <= left or bottom <= top or rw <= 0 or rh <= 0:
        return []
    x0, x1 = (left - rx) / rw * width, (right - rx) / rw * width
    y0, y1 = (top - ry) / rh * height, (bottom - ry) / rh * height
    columns, rows = math.ceil(width / TILE), math.ceil(height / TILE)
    tx0, tx1 = max(0, int(x0 // TILE)), min(columns, math.ceil(x1 / TILE))
    ty0, ty1 = max(0, int(y0 // TILE)), min(rows, math.ceil(y1 / TILE))
    return [(tx, ty) for ty in range(ty0, ty1) for tx in range(tx0, tx1)]


def group(tiles, chunk=None):
    """Cover tiles with few rectangles ``(tx0, ty0, tx1, ty1)`` of whole tiles.

    Rows of contiguous tiles merge with the run above when they span the same
    columns; with ``chunk``, rectangles are cut into bands of at most that many tiles.
    """
    rows = {}
    for tx, ty in tiles:
        rows.setdefault(ty, []).append(tx)
    rects = []
    for ty in sorted(rows):
        columns = sorted(rows[ty])
        start = previous = columns[0]
        runs = []
        for tx in columns[1:]:
            if tx != previous + 1:
                runs.append((start, previous + 1))
                start = tx
            previous = tx
        runs.append((start, previous + 1))
        for tx0, tx1 in runs:
            for rect in rects:
                if rect[0] == tx0 and rect[2] == tx1 and rect[3] == ty:
                    rect[3] = ty + 1
                    break
            else:
                rects.append([tx0, ty, tx1, ty + 1])
    if chunk:
        bands = []
        for tx0, ty0, tx1, ty1 in rects:
            step = max(1, chunk // (tx1 - tx0))
            bands += [[tx0, y, tx1, min(ty1, y + step)] for y in range(ty0, ty1, step)]
        rects = bands
    return [tuple(rect) for rect in rects]


class Tile:
    """Ready-to-draw images of one tile; ``image`` is None outside the frame."""
    __slots__ = ('x0', 'y0', 'width', 'height', 'image', 'shadows', 'highlights', 'nbytes')

    def __init__(self, x0, y0, width=0, height=0, image=None, shadows=None, highlights=None):
        self.x0, self.y0, self.width, self.height = x0, y0, width, height
        self.image, self.shadows, self.highlights = image, shadows, highlights
        self.nbytes = sum(i.sizeInBytes() for i in (image, shadows, highlights) if i is not None)


def _warning(hit, color):
    if not hit.any():
        return None
    data = np.zeros((*hit.shape, 4), np.uint8)
    data[hit] = color
    image = QImage(data.data, data.shape[1], data.shape[0], data.strides[0], QImage.Format.Format_RGBA8888)
    return image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)


def make_tile(rgb, x0, y0):
    """QImages of a display-ready sRGB block; QImage may be built outside the GUI thread."""
    h, w = rgb.shape[:2]
    # Same 8-bit truncation as widgets.qimage, stored as 0xffRRGGBB for fast painting.
    rgb8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    data = np.empty((h, w, 4), np.uint8)
    data[..., 0], data[..., 1], data[..., 2], data[..., 3] = rgb8[..., 2], rgb8[..., 1], rgb8[..., 0], 255
    image = QImage(data.data, w, h, data.strides[0], QImage.Format.Format_RGB32).copy()
    peak = rgb.max(axis=2)
    return Tile(x0, y0, w, h, image, _warning(peak <= .005, SHADOW_COLOR), _warning(peak >= .995, HIGHLIGHT_COLOR))


def tile_array(tile):
    """8-bit RGB pixels of a tile, for diagnostics and tests."""
    image = tile.image.convertToFormat(QImage.Format.Format_RGB888)
    rows = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())
    return rows[:, :image.width() * 3].reshape(image.height(), image.width(), 3).copy()


class DetailLayer:
    """What the canvas draws over the preview: tiles of a ``size`` frame (live cache dicts)."""

    def __init__(self, size, main, before=None):
        self.size, self.main, self.before = size, main, before


class TileStore:
    """Least-recently-used tile cache by layer ``(kind, key, level)``."""

    def __init__(self, max_bytes=CACHE_BYTES):
        self.max_bytes = max_bytes
        self.layers = {}
        self.order = OrderedDict()
        self.bytes = 0

    def layer(self, kind, key, level):
        return self.layers.setdefault((kind, key, level), {})

    def missing(self, kind, key, level, tiles):
        layer = self.layers.get((kind, key, level), {})
        for index in tiles:
            name = (kind, key, level) + index
            if name in self.order:
                self.order.move_to_end(name)
        return [index for index in tiles if index not in layer]

    def add(self, kind, key, level, tiles):
        layer = self.layer(kind, key, level)
        fresh = set()
        for index, tile in tiles.items():
            name = (kind, key, level) + index
            self.bytes -= self.order.pop(name, 0)
            layer[index] = tile
            self.order[name] = tile.nbytes
            self.bytes += tile.nbytes
            fresh.add(name)
        while self.bytes > self.max_bytes and self.order:
            name = next(iter(self.order))
            if name in fresh:
                break
            self.bytes -= self.order.pop(name)
            self.layers.get(name[:3], {}).pop(name[3:], None)

    def retain(self, keys):
        """Drop every layer whose ``(kind, key)`` is not in ``keys``."""
        for name in [n for n in self.layers if n[:2] not in keys]:
            for index in self.layers.pop(name):
                self.bytes -= self.order.pop(name + index, 0)

    def clear(self):
        self.layers.clear()
        self.order.clear()
        self.bytes = 0


class DetailSource:
    """Original-resolution linear pixels of one photograph and their 1/2^k reductions.

    Filled lazily by detail jobs (one at a time through the scheduler); the window
    replaces the whole object when another photograph opens.
    """

    def __init__(self, path):
        self.path = path
        self.full = None
        self.levels = {}
        self.contexts = {}
        self.lock = threading.Lock()

    def level(self, level):
        with self.lock:
            if self.full is None:
                started = time.perf_counter()
                self.full = engine.load_image(self.path, None)[0]
                log.info('decoded %s at %d × %d in %.0f ms', self.path, self.full.shape[1], self.full.shape[0],
                         (time.perf_counter() - started) * 1000)
            image = self.full
            height, width = image.shape[:2]
            for k in range(1, level + 1):
                if k not in self.levels:
                    self.levels[k] = cv2.resize(image, level_size(width, height, k), interpolation=cv2.INTER_AREA)
                image = self.levels[k]
            return image

    def dehaze_context(self, edits, backend, proxy, proxy_scale):
        """Dehaze statistics of the preview, shared by every level so all zooms match it."""
        a = edits['adjustments']
        key = engine._key(edits.get('wb_gain'), edits.get('white_balance'), edits.get('retouch'),
                          edits.get('develop'), {k: a.get(k, 0) for k in engine.TONAL_KEYS},
                          a.get('denoise', 0), a.get('color_noise', 0), a.get('dehaze', 0), backend.gpu_pointwise,
                          lens.key(edits.get('lens')))
        with self.lock:
            if key not in self.contexts:
                self.contexts.clear()
                self.contexts[key] = engine.dehaze_context_for(proxy, edits, backend, proxy_scale)
            return self.contexts[key]


def render(holder, request, cancel):
    """Worker: render blocks of tiles; returns ``[(kind, key, level, {index: Tile})]``."""
    def check():
        if cancel.is_set():
            raise InterruptedError(tr('细节渲染已过期'))
    started = time.perf_counter()
    edits, backend, level, final = request['edits'], request['backend'], request['level'], request['final']
    source = holder.level(level)
    check()
    height, width = source.shape[:2]
    frame_width, frame_height = engine.display_size(edits, (width, height), final)
    detail_scale = width / holder.full.shape[1]
    context = None
    if any(kind == 'edited' for kind, _, _ in request['blocks']) and edits['adjustments'].get('dehaze', 0) > 0:
        context = holder.dehaze_context(edits, backend, request['proxy'], request['proxy_scale'])
    results, pixels = [], 0
    for kind, key, (tx0, ty0, tx1, ty1) in request['blocks']:
        check()
        x0, y0 = tx0 * TILE, ty0 * TILE
        x1, y1 = min(frame_width, tx1 * TILE), min(frame_height, ty1 * TILE)
        block = None
        if x0 < x1 and y0 < y1:
            block = engine.render_display(source, edits, backend, (x0, y0, x1, y1), final, detail_scale,
                                          kind, context, check)
            pixels += block.shape[0] * block.shape[1]
        tiles = {}
        for ty in range(ty0, ty1):
            for tx in range(tx0, tx1):
                left, top = tx * TILE, ty * TILE
                part = block[top - y0:top - y0 + TILE, left - x0:left - x0 + TILE] if block is not None else None
                tiles[(tx, ty)] = make_tile(part, left, top) if part is not None and part.size else Tile(left, top)
        results.append((kind, key, level, tiles))
    log.debug('detail level %d: %d blocks, %.1f MP in %.0f ms', level, len(results), pixels / 1e6,
              (time.perf_counter() - started) * 1000)
    return results
