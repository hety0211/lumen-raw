"""Explicit work state and single-owner image job scheduling (1.3.0).

Before 1.3.0 the main window tracked its work with independent booleans
(``loading``, ``exporting``, ``ai_busy``, ...) spread across six mixins, and each
entry point repeated its own combination of checks.  ``WorkState`` now owns
those flags and one conflict table decides which activity may start while
another runs.  ``JobScheduler`` owns the background queue: exactly one image
computation holds the native thread budget at a time, the highest priority job
runs next, and shutting down drops queued work while the active job drains.

The legacy attribute names remain available on the window as properties so
dialogs, diagnostics and tests keep working unchanged.
"""
from __future__ import annotations

import enum
import logging
import time
from collections import deque

from PySide6.QtCore import QObject, QRunnable, Signal
from .i18n import tr

log = logging.getLogger(__name__)


class Activity(enum.Enum):
    LOADING = 'loading'          # decoding the current photograph
    EXPORTING = 'exporting'      # full-size render and file encoding
    AI = 'ai'                    # AI enhancement / denoise / merge dialogs
    SELECTION = 'selection'      # automatic mask inference
    RENDER = 'render'            # preview render in flight
    DETAIL = 'detail'            # original-resolution detail render
    THUMBNAILS = 'thumbnails'    # filmstrip thumbnail batch


A = Activity
#: Foreground work that blocks closing the window.
EXCLUSIVE = frozenset({A.LOADING, A.EXPORTING, A.AI, A.SELECTION})
#: Activities that must be idle before the key may start.
BLOCKED_BY = {
    A.LOADING: frozenset({A.LOADING, A.EXPORTING, A.AI}),
    A.EXPORTING: frozenset({A.LOADING, A.EXPORTING, A.AI, A.SELECTION}),
    A.AI: frozenset({A.LOADING, A.EXPORTING, A.AI, A.SELECTION}),
    A.SELECTION: frozenset({A.LOADING, A.AI, A.SELECTION}),
    A.RENDER: frozenset({A.LOADING, A.AI, A.RENDER}),
    A.DETAIL: frozenset({A.LOADING, A.EXPORTING, A.AI, A.DETAIL}),
    A.THUMBNAILS: frozenset({A.AI, A.THUMBNAILS}),
}
LABELS = {A.LOADING: tr('正在读取'), A.EXPORTING: tr('正在导出'), A.AI: tr('AI / 合成处理中'),
          A.SELECTION: tr('正在识别蒙版'), A.RENDER: tr('正在更新预览'), A.DETAIL: tr('正在读取原图细节'),
          A.THUMBNAILS: tr('正在生成缩略图')}
ORDER = (A.AI, A.EXPORTING, A.LOADING, A.SELECTION, A.DETAIL, A.RENDER, A.THUMBNAILS)


class WorkState(QObject):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active = set()
        self.pending_render = False
        self.closing = False
        self.transitions = deque(maxlen=200)

    def active(self, activity):
        return activity in self._active

    def busy(self, *activities):
        """Any of the given activities running; with no arguments, any foreground work."""
        return bool(self._active & set(activities or EXCLUSIVE))

    def blockers(self, activity):
        return BLOCKED_BY[activity] & self._active

    def can_start(self, activity):
        return not self.closing and not self.blockers(activity)

    def begin(self, activity, force=False):
        if activity in self._active and force:
            return True
        if not force and not self.can_start(activity):
            log.debug('refused %s while %s', activity.value, sorted(a.value for a in self.blockers(activity)))
            return False
        self._record('begin', activity)
        self._active.add(activity)
        self.changed.emit()
        return True

    def end(self, activity):
        if activity in self._active:
            self._record('end', activity)
            self._active.discard(activity)
            self.changed.emit()

    def set(self, activity, flag):
        """Boolean assignment used by the legacy window attributes."""
        if flag:
            self.begin(activity, force=True)
        else:
            self.end(activity)

    @property
    def phase(self):
        if self.closing:
            return 'closing'
        for activity in ORDER:
            if activity in self._active:
                return activity.value
        return 'idle'

    def describe(self):
        labels = [LABELS[a] for a in ORDER if a in self._active]
        return ' · '.join(labels) if labels else tr('空闲')

    def _record(self, event, activity):
        self.transitions.append((time.monotonic(), event, activity.value))
        log.debug('%s %s', event, activity.value)


class JobSignals(QObject):
    success = Signal(object)
    failed = Signal(str)
    done = Signal()


class Job(QRunnable):
    def __init__(self, fn, name=''):
        super().__init__()
        self.fn = fn
        self.name = name or getattr(fn, '__qualname__', 'job')
        self.signals = JobSignals()

    def run(self):
        started = time.perf_counter()
        try:
            self.signals.success.emit(self.fn())
        except Exception as exc:
            if isinstance(exc, InterruptedError):
                log.info('job %s cancelled: %s', self.name, exc)
            else:
                log.exception('job %s failed', self.name)
            self.signals.failed.emit(str(exc) or type(exc).__name__)
        finally:
            log.debug('job %s finished in %.0f ms', self.name, (time.perf_counter() - started) * 1000)
            self.signals.done.emit()


class JobScheduler(QObject):
    """One active image job; highest priority first, FIFO within a priority."""
    drained = Signal()

    def __init__(self, pool, parent=None):
        super().__init__(parent)
        self.pool = pool
        self.jobs = set()
        self.queue = []
        self.active = None
        self.closing = False

    def submit(self, fn, success, fail, priority=0, name=''):
        if self.closing:
            return None
        job = Job(fn, name)
        self.jobs.add(job)
        job.signals.success.connect(lambda result: success(result) if not self.closing else None)
        job.signals.failed.connect(lambda text: fail(text) if not self.closing else None)
        job.signals.done.connect(lambda: self._finish(job))
        self.queue.append((priority, job))
        self._next()
        return job

    def _next(self):
        # Queue on the GUI thread instead of blocking Qt workers on Python locks.
        if self.closing or self.active is not None or not self.queue:
            return
        index = max(range(len(self.queue)), key=lambda i: (self.queue[i][0], -i))
        priority, job = self.queue.pop(index)
        self.active = job
        self.pool.start(job, priority)

    def _finish(self, job):
        self.jobs.discard(job)
        if self.active is job:
            self.active = None
        if self.closing:
            self.drained.emit()
        else:
            self._next()

    def shutdown(self):
        """Drop queued work; True when nothing is running any more."""
        self.closing = True
        for _, job in self.queue:
            self.jobs.discard(job)
        self.queue.clear()
        return self.active is None


def legacy_flag(activity):
    return property(lambda self: self.work.active(activity),
                    lambda self, value: self.work.set(activity, bool(value)))


class WorkStateAccess:
    """Window attributes kept for dialogs, diagnostics and the regression tests."""
    loading = legacy_flag(A.LOADING)
    exporting = legacy_flag(A.EXPORTING)
    ai_busy = legacy_flag(A.AI)
    selection_busy = legacy_flag(A.SELECTION)
    render_running = legacy_flag(A.RENDER)
    full_busy = legacy_flag(A.DETAIL)
    film_loading = legacy_flag(A.THUMBNAILS)

    @property
    def pending(self):
        return self.work.pending_render

    @pending.setter
    def pending(self, value):
        self.work.pending_render = bool(value)

    @property
    def closing(self):
        return self.scheduler.closing

    @closing.setter
    def closing(self, value):
        self.scheduler.closing = self.work.closing = bool(value)

    @property
    def jobs(self):
        return self.scheduler.jobs

    @property
    def queued_jobs(self):
        return self.scheduler.queue

    @property
    def active_job(self):
        return self.scheduler.active
