"""Original-resolution detail for zoomed-in views (1.4.0: visible tiles only).

The window keeps the 1600-pixel preview for the whole frame.  Zoomed past it,
``viewport`` renders the visible tiles at the matching resolution level in the
background; this mixin decides which tiles the view needs, starts one job at a
time and hands finished tiles to the canvas.  No original-resolution array is
touched on the GUI thread.

* While a mouse button is held on a slider, curve, color wheel or the canvas,
  no detail is rendered; it follows once the button is released.
* A running detail job is cancelled as soon as an edit, a zoom or a pan makes
  its result useless, so it never delays the next preview.
"""
import copy
import logging
import threading
import time
from typing import NamedTuple
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication
from . import develop, engine, lens, viewport
from .scheduler import Activity as A
from .i18n import tr

log = logging.getLogger(__name__)


class DetailView(NamedTuple):
    level: int
    size: tuple
    tiles: list
    final: bool


class ResolutionMixin:
    def init_resolution(self):
        self.detail_source = None
        self.tiles = viewport.TileStore()
        self.detail_request = None
        self.document_token = 0
        self._original = (None, None)
        self._reference = (None, None)
        self.detail_timer = QTimer(self)
        self.detail_timer.setSingleShot(True)
        self.detail_timer.setInterval(220)
        self.detail_timer.timeout.connect(self.request_detail)

    @property
    def full_source(self):
        """Decoded original-resolution pixels of the open photograph, once a zoom needed them."""
        return self.detail_source.full if self.detail_source is not None else None

    def reset_resolution(self):
        self.document_token += 1
        self.cancel_detail()
        self.detail_source = None
        self.tiles.clear()
        self._original = (None, None)
        self._reference = (None, None)
        self.detail_timer.stop()
        if hasattr(self, 'canvas'):
            self.canvas.set_detail(None)

    def preview_original(self):
        """Developed original at preview size for comparison and luminance masks, cached per photo
        and develop settings instead of being recomputed on every display update (1.4.0)."""
        key = (self.document_token, engine._key(self.edits.get('develop', {}), engine.process_version(self.edits)))
        if self._original[0] != key:
            self._original = (key, engine.develop_view(self.source, self.edits))
        return self._original[1]

    def preview_reference(self):
        """Developed preview after lens corrections: what luminance masks select from (1.5.1)."""
        lens_key = lens.key(self.edits.get('lens'))
        if lens_key is None:
            return self.preview_original()
        key = (self.document_token, engine._key(self.edits.get('develop', {}), lens_key, engine.process_version(self.edits)))
        if self._reference[0] != key:
            self._reference = (key, engine.develop_view(lens.correct(self.source, self.edits), self.edits))
        return self._reference[1]

    def detail_keys(self):
        final = self.final_view.isChecked()
        geometry = [self.edits.get('crop'), self.edits.get('straighten', 0), self.edits.get('rotation', 0)] if final else None
        return {'edited': (self.document_token, self.generation, final),
                'original': (self.document_token, engine._key(self.edits.get('develop', {}), geometry,
                                                              engine.process_version(self.edits)), final)}

    def wanted_kinds(self):
        if self.comparing:
            return ['original']
        return ['edited', 'original'] if self.split_check.isChecked() else ['edited']

    def detail_view(self):
        """Resolution level and visible tiles the zoom needs, or None while the preview suffices."""
        if self.source is None or self.canvas.image is None or not self.info:
            return None
        width, height = self.info.get('width', 0), self.info.get('height', 0)
        if width <= self.source.shape[1] and height <= self.source.shape[0]:
            return None
        if self.full_source is not None:
            height, width = self.full_source.shape[:2]
        final = self.final_view.isChecked()
        full_width = engine.display_size(self.edits, (width, height), final)[0]
        rect = self.canvas.image_rect()
        # Device pixels, not Qt logical pixels, determine source-detail demand.
        density = rect.width() * self.canvas.devicePixelRatioF() / full_width
        if density <= self.canvas.image.width() / full_width * 1.03:
            return None
        level = viewport.level_for(density)
        size = engine.display_size(self.edits, viewport.level_size(width, height, level), final)
        if size[0] <= self.canvas.image.width() * 1.03:
            return None
        tiles = viewport.visible_tiles((rect.x(), rect.y(), rect.width(), rect.height()),
                                       (0, 0, self.canvas.width(), self.canvas.height()), size)
        return DetailView(level, size, tiles, final)

    def detail_needed(self):
        return self.detail_view() is not None

    def missing_detail(self, view=None, keys=None):
        view = view or self.detail_view()
        if view is None:
            return []
        keys = keys or self.detail_keys()
        missing = []
        for kind in self.wanted_kinds():
            tiles = self.tiles.missing(kind, keys[kind], view.level, view.tiles)
            if tiles:
                missing.append((kind, keys[kind], tiles))
        return missing

    def detail_ready(self):
        """True when every visible tile the view needs is on screen (or the preview suffices)."""
        return not self.missing_detail()

    def refresh_detail(self, restart=False):
        """Show cached tiles for the current view and schedule rendering of missing ones."""
        if not hasattr(self, 'canvas'):
            return
        self.cancel_stale_detail()
        if self.source is None:
            self.canvas.set_detail(None)
            return
        keys = self.detail_keys()
        self.tiles.retain(set(keys.items()))
        view = self.detail_view()
        if view is None:
            self.canvas.set_detail(None)
            return
        main = 'original' if self.comparing else 'edited'
        before = self.tiles.layer('original', keys['original'], view.level) \
            if self.split_check.isChecked() and not self.comparing else None
        self.canvas.set_detail(viewport.DetailLayer(view.size, self.tiles.layer(main, keys[main], view.level), before))
        if self.missing_detail(view, keys) and (restart or not self.detail_timer.isActive()):
            self.detail_timer.start(220)

    def viewport_changed(self):
        if self.source is not None:
            self.refresh_detail(restart=True)

    def cancel_detail(self):
        if self.detail_request is not None:
            self.detail_request['cancel'].set()

    def cancel_stale_detail(self):
        """Abandon a running render whose tiles can no longer be shown."""
        request = self.detail_request
        if request is None or request['cancel'].is_set():
            return
        view = self.detail_view() if self.source is not None else None
        keys = self.detail_keys() if self.source is not None else {}
        stale = view is None or request['level'] != view.level or \
            any(keys.get(kind) != key for kind, key, _ in request['blocks'])
        if not stale:
            visible = set(view.tiles)
            stale = not any((tx, ty) in visible for _, _, (tx0, ty0, tx1, ty1) in request['blocks']
                            for ty in range(ty0, ty1) for tx in range(tx0, tx1))
        if stale:
            log.debug('cancel stale detail render')
            request['cancel'].set()

    def request_detail(self):
        if self.source is None or self.work.busy(A.LOADING, A.EXPORTING, A.AI) or self.closing:
            return
        if QApplication.mouseButtons() != Qt.MouseButton.NoButton and not self.compare.isDown():
            # Dragging a slider, curve, wheel or the image: render after the button is released.
            self.detail_timer.start(150)
            return
        if self.work.active(A.DETAIL):
            return  # the running job requests the next block when it ends
        view = self.detail_view()
        keys = self.detail_keys()
        missing = self.missing_detail(view, keys)
        if not missing:
            return
        kind, key, tiles = missing[0]
        a = self.edits['adjustments']
        shape = viewport.level_size(*(self.full_source.shape[1::-1] if self.full_source is not None else
                                      (self.info['width'], self.info['height'])), view.level)[::-1]
        margin = engine.spatial_margin(a, shape) + max([engine.spatial_margin(m['adjustments'], shape)
                                                        for m in self.edits['masks'] if m['enabled']], default=0)
        rects = viewport.group(tiles, viewport.CHUNK_TILES if margin <= 32 else None)
        cx = np.mean([t[0] for t in view.tiles]) + .5
        cy = np.mean([t[1] for t in view.tiles]) + .5
        rect = min(rects, key=lambda r: ((r[0] + r[2]) / 2 - cx) ** 2 + ((r[1] + r[3]) / 2 - cy) ** 2)
        if self.detail_source is None:
            self.detail_source = viewport.DetailSource(self.source_path)
        holder, token, cancel = self.detail_source, self.document_token, threading.Event()
        request = dict(edits=copy.deepcopy(self.edits), backend=self.backend, level=view.level, final=view.final,
                       blocks=[(kind, key, rect)], proxy=self.source,
                       proxy_scale=max(.1, self.source.shape[1] / self.info.get('width', self.source.shape[1])))
        self.work.begin(A.DETAIL)
        self.detail_request = dict(cancel=cancel, level=view.level, blocks=request['blocks'])
        if holder.full is None:
            self.statusBar().showMessage(tr('正在读取原图细节… 完成后自动替换当前画面'))
        started = time.perf_counter()

        def finished():
            self.work.end(A.DETAIL)
            if self.detail_request is not None and self.detail_request['cancel'] is cancel:
                self.detail_request = None

        def success(result):
            finished()
            if token != self.document_token:
                return
            for kind, key, level, tiles in result:
                self.tiles.add(kind, key, level, tiles)
            self.refresh_detail()
            if self.detail_ready() and self.detail_view() is not None:
                level = self.detail_view().level
                self.statusBar().showMessage((tr('原图细节已就绪 · 100% 对应原片像素') if level == 0 else
                                              tr('细节已就绪 · 原图 1/{v} 分辨率', v=2 ** level)) +
                                             f' · {(time.perf_counter() - started) * 1000:.0f} ms')
            self.request_detail()

        def fail(text):
            finished()
            if token != self.document_token:
                return
            if cancel.is_set():
                self.refresh_detail()
            else:
                log.warning('detail render failed: %s', text)
                self.statusBar().showMessage(tr('原图细节加载失败，可重试缩放：') + text)
        self.job(lambda: viewport.render(holder, request, cancel), success, fail, priority=-1)
