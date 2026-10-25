"""Background work of the library (2.0.0): EXIF index, thumbnails and culling hints.

Three kinds of task, most urgent first: reading capture data with ExifTool (sorting, dates and
bursts depend on it), thumbnails the grid is showing (the last requested first, so scrolling
fetches what is on screen), and the culling analysis of photos not analysed yet.  A few low
priority threads do the work; they stop while ``paused()`` is true (AI processing, exports) and
are started again with ``kick()``.  Results reach the interface through Qt signals.
"""
from __future__ import annotations

import collections
import logging
import threading
import time

from PySide6.QtCore import QObject, QRunnable, QThread, QThreadPool, Signal
from PySide6.QtGui import QImage

from . import library_db

log = logging.getLogger(__name__)


def possible(parts):
    """``parts`` without the closed eyes while the face models are not on this computer (2.1.1:
    a program-only installation before its models are downloaded)."""
    from . import ai_models, culling
    if library_db.EYES in parts and not (ai_models.available(culling.DETECTOR) and ai_models.available(culling.LANDMARKS)):
        return set(parts) - {library_db.EYES}
    return set(parts)


class _Signals(QObject):
    thumbnail = Signal(str, QImage)
    changed = Signal(str, list)
    progress = Signal(int, int)


class _Runner(QRunnable):
    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self.setAutoDelete(True)

    def run(self):
        try:
            QThread.currentThread().setPriority(QThread.Priority.LowPriority)
        except Exception:
            pass
        try:
            self.worker._loop()
        finally:
            self.worker._finished()


class LibraryWorker(QObject):
    def __init__(self, db, cache, ratings=None, paused=lambda: False, threads=3, parent=None):
        super().__init__(parent)
        self.db, self.cache, self.ratings, self.paused = db, cache, ratings, paused
        self.signals = _Signals()
        self.thumbnail_ready = self.signals.thumbnail
        self.changed = self.signals.changed
        self.progress = self.signals.progress
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(max(1, threads))
        self._lock = threading.Lock()
        self._thumbs = collections.OrderedDict()
        self._analyse = collections.OrderedDict()
        self._index = False
        self._indexing = False
        self._running = 0
        self._closing = False
        #: 2.1.1: the parts analysed in the background (sharpness and closed eyes, each switchable).
        self.parts = set(library_db.BOTH)
        self.done = self.total = 0
        # A window destroyed without closing: the thread pool's destructor waits for its runners,
        # so they must stop after their current task instead of emptying the queues.
        stop = self._stop = threading.Event()
        self.destroyed.connect(lambda *_: stop.set())

    # -- requests (GUI thread)

    def want_thumbnail(self, path):
        with self._lock:
            self._thumbs[path] = True
            self._thumbs.move_to_end(path, last=False)
            while len(self._thumbs) > 600:
                self._thumbs.popitem(last=True)
        self.kick()

    def forget_thumbnails(self):
        with self._lock:
            self._thumbs.clear()

    def want_index(self):
        with self._lock:
            self._index = True
        self.kick()

    @property
    def analysis_enabled(self):
        return bool(self.parts)

    def want_analysis(self, rows, first=False):
        """Analyse ``rows`` (``[{id, path}]``, with ``analysis`` when known); ``first`` puts them
        before earlier requests.  2.1.1: the parts a row already has travel with it, so the worker
        reads nothing from the database before it writes the result."""
        with self._lock:
            for row in rows:
                self._analyse[row['id']] = (row['path'], library_db.analysed_parts(row))
                if first:
                    self._analyse.move_to_end(row['id'], last=False)
            self.total = self.done + len(self._analyse)
        self.kick()

    def pending(self):
        with self._lock:
            return bool(self._index or self._thumbs or self._analyse) or self._running > 0

    def kick(self):
        if self._closing or self.paused():
            return
        with self._lock:
            work = len(self._thumbs) + len(self._analyse) + bool(self._index)
            start = max(0, min(self.pool.maxThreadCount() - self._running, work))
            self._running += start
        for _ in range(start):
            self.pool.start(_Runner(self))

    def shutdown(self, timeout=5000):
        self._closing = True
        self._stop.set()
        with self._lock:
            self._thumbs.clear()
            self._analyse.clear()
            self._index = False
        return self.pool.waitForDone(timeout)

    # -- workers

    def _finished(self):
        with self._lock:
            self._running -= 1

    def _take(self):
        with self._lock:
            if self._index and not self._indexing:
                self._index = False
                self._indexing = True
                return 'index', None
            if self._thumbs:
                path, _ = self._thumbs.popitem(last=False)
                return 'thumb', path
            if self._analyse and self.parts:
                photo, (path, done) = self._analyse.popitem(last=False)
                return 'analyse', (photo, path, done)
        return None, None

    def _loop(self):
        while not self._closing and not self._stop.is_set():
            try:
                if self.paused():
                    return
            except RuntimeError:            # the window is gone
                return
            kind, item = self._take()
            if kind is None:
                return
            try:
                if kind == 'index':
                    self._run_index()
                elif kind == 'thumb':
                    self._run_thumbnail(item)
                else:
                    self._run_analysis(*item)
            except Exception:
                log.warning('library task %s failed for %s', kind, item, exc_info=True)
                if kind == 'analyse':
                    try:
                        self.db.set_analysis(item[0], {}, self.parts)      # unreadable: not tried again and again
                    except Exception:
                        pass

    def _run_index(self):
        try:
            while not self._closing and not self._stop.is_set() and not self.paused():
                rows = self.db.unindexed()
                if not rows:
                    break
                self.db.index(rows, self.ratings)       # the database tells its listeners
        finally:
            with self._lock:
                self._indexing = False
                again = bool(self.db.unindexed(1)) and not self._closing
                if again:
                    self._index = True

    def _run_thumbnail(self, path):
        data = self.cache.thumbnail(path)
        image = QImage.fromData(data, 'JPG')
        if not image.isNull():
            self.thumbnail_ready.emit(path, image)

    def _run_analysis(self, photo, path, done=frozenset()):
        from . import culling
        parts = frozenset(self.parts) - frozenset(done)
        if library_db.EYES in parts and culling.faces() is None:
            # 2.1.1: without the face models the eyes are not marked done; they wait, without
            # decoding the photo again and again, until the window queues them when the models
            # arrive (update_dialog.UpdateMixin.models_changed).
            parts -= {library_db.EYES}
        if not parts:
            self._count()
            return
        started = time.perf_counter()
        view = self.cache.view(path)
        if not self.cache.has_thumbnail(path):
            image = QImage.fromData(self.cache.thumbnail(path, view), 'JPG')
            if not image.isNull():
                self.thumbnail_ready.emit(path, image)
        result = culling.analyse(view, library_db.SHARPNESS in parts, library_db.EYES in parts)
        self.db.set_analysis(photo, result, parts)
        self._count()
        log.debug('analysed %s in %.0f ms', path, (time.perf_counter() - started) * 1000)

    def _count(self):
        with self._lock:
            self.done += 1
            done, total = self.done, max(self.total, self.done)
        self.progress.emit(done, total)
