"""Smart previews on disk (2.0.0).

Opening a photo used to decode it again every time: a 45–61 MP RAW takes 1–1.8 s (LibRaw, the
camera's preview for the develop reference, ExifTool), most of it for the same 1600-pixel preview
as last time.  This cache keeps three things per photo, keyed by its path, size and modification
time (and the LUMEN version and interface language, which shape the decoded metadata):

* the decoded preview ``engine.load_image`` returns — linear float pixels stored as float16 with
  their bytes shuffled into planes and zlib-compressed (about 7 MB for 1600 × 1067), plus its
  ``info`` as JSON.  Reading one takes about 40 ms.  Previews are rounded to float16 also when they
  were just decoded, so a photo looks the same whether it came from the cache or not;
* the camera view (``previews.camera_view``) the library and the culling view show, as JPEG;
* the library thumbnail, as a small JPEG in the index database.

The cache is rebuildable: everything in it can be thrown away.  It lives in the per-user cache
folder (``LUMEN_PREVIEW_CACHE`` overrides it), is limited to ``settings.ini`` ``preview_cache_mb``
(default 4 GB; 0 turns it off) and drops the least recently used entries beyond that.  The editor,
the command line and the MCP server share it; the index is SQLite in WAL mode and every file is
written to a temporary name and renamed.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json
import logging
import os
import sqlite3
import struct
import threading
import time
import zlib
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

FORMAT = 1
MAGIC = b'LPV1'
SETTING = 'preview_cache_mb'
DEFAULT_MB = 4096
#: Choices offered in the library menu (MB; 0 = off).
CHOICES = (0, 1024, 2048, 4096, 8192, 16384, 32768)
THUMB_EDGE = 320
#: Decoded previews kept in memory (each about 20 MB as float32).
MEMORY_ITEMS = 6


def folder():
    override = os.environ.get('LUMEN_PREVIEW_CACHE')
    if override:
        return Path(override)
    from .host import cache_folder
    return cache_folder() / 'previews'


def limit_mb():
    from .i18n import setting
    try:
        value = int(setting(SETTING, str(DEFAULT_MB)))
    except (TypeError, ValueError):
        value = DEFAULT_MB
    return max(0, value)


def stamp(path):
    """``(key, resolved path)`` of a photo as it is on disk now; changes when the file does."""
    from . import __version__
    from .i18n import language
    resolved = Path(path).resolve()
    status = resolved.stat()
    text = f'{os.path.normcase(str(resolved))}|{status.st_size}|{status.st_mtime_ns}|{FORMAT}|{__version__}|{language()}'
    return hashlib.sha1(text.encode('utf-8')).hexdigest(), str(resolved)


def quantize(rgb):
    """The float16 precision previews are kept at (relative error below 0.05 %)."""
    return rgb.astype(np.float16).astype(np.float32)


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def pack(rgb, info):
    """Bytes of a decoded preview (header, info JSON, shuffled float16 planes, zlib)."""
    half = np.ascontiguousarray(rgb, np.float16)
    planes = np.frombuffer(half.tobytes(), np.uint8).reshape(-1, 2).T.tobytes()
    header = json.dumps(dict(shape=list(half.shape), info=_plain(info)), ensure_ascii=False).encode('utf-8')
    return MAGIC + struct.pack('<I', len(header)) + header + zlib.compress(planes, 1)


def unpack(data):
    if data[:4] != MAGIC:
        raise ValueError('not a LUMEN preview')
    size, = struct.unpack('<I', data[4:8])
    header = json.loads(data[8:8 + size].decode('utf-8'))
    shape = tuple(header['shape'])
    planes = np.frombuffer(zlib.decompress(data[8 + size:]), np.uint8)
    if planes.size != 2 * int(np.prod(shape)):
        raise ValueError('truncated preview')
    rgb = planes.reshape(2, -1).T.copy().view(np.float16).reshape(shape).astype(np.float32)
    return rgb, header['info']


class PreviewCache:
    def __init__(self, root=None, limit=None):
        self.root = Path(root) if root else folder()
        self.limit = (limit_mb() if limit is None else limit) * 1024 * 1024
        self._lock = threading.RLock()
        self._db = None
        self._memory = collections.OrderedDict()
        self._inflight = {}
        self._writer = ThreadPoolExecutor(1, thread_name_prefix='preview-cache-write')
        self._prefetch = ThreadPoolExecutor(1, thread_name_prefix='preview-prefetch')
        self._pending = 0
        self.paused = lambda: False
        self.hits = self.misses = 0

    @property
    def enabled(self):
        return self.limit > 0

    # -- index

    def _index(self):
        if self._db is None:
            self.root.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(str(self.root / 'index.db'), timeout=10, check_same_thread=False, isolation_level=None)
            try:
                db.execute('PRAGMA journal_mode=WAL')
            except sqlite3.DatabaseError:
                # A damaged index: the cache is rebuildable, start a new one.
                db.close()
                log.warning('preview cache index damaged; starting a new one')
                for name in ('index.db', 'index.db-wal', 'index.db-shm'):
                    (self.root / name).unlink(missing_ok=True)
                db = sqlite3.connect(str(self.root / 'index.db'), timeout=10, check_same_thread=False, isolation_level=None)
                db.execute('PRAGMA journal_mode=WAL')
            db.execute('PRAGMA synchronous=NORMAL')
            db.execute('CREATE TABLE IF NOT EXISTS entries(key TEXT NOT NULL, kind TEXT NOT NULL, bytes INTEGER NOT NULL, '
                       'used REAL NOT NULL, PRIMARY KEY(key, kind))')
            db.execute('CREATE TABLE IF NOT EXISTS thumbs(key TEXT PRIMARY KEY, data BLOB NOT NULL)')
            db.execute('CREATE INDEX IF NOT EXISTS entries_used ON entries(used)')
            self._db = db
        return self._db

    def _file(self, key, kind):
        return self.root / kind / key[:2] / f'{key}.{ "lpv" if kind == "d" else "jpg"}'

    def _touch(self, key, kind):
        try:
            with self._lock:
                self._index().execute('UPDATE entries SET used=? WHERE key=? AND kind=?', (time.time(), key, kind))
        except sqlite3.Error:
            log.debug('cache index busy', exc_info=True)

    def _record(self, key, kind, size):
        with self._lock:
            self._index().execute('INSERT OR REPLACE INTO entries(key, kind, bytes, used) VALUES (?, ?, ?, ?)',
                                  (key, kind, int(size), time.time()))

    def _forget(self, key, kind):
        try:
            self._file(key, kind).unlink(missing_ok=True)
        except OSError:
            pass
        with self._lock:
            self._index().execute('DELETE FROM entries WHERE key=? AND kind=?', (key, kind))
            if kind == 't':
                self._index().execute('DELETE FROM thumbs WHERE key=?', (key,))

    def _read(self, key, kind):
        path = self._file(key, kind)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError:
            log.info('preview cache unreadable: %s', path, exc_info=True)
            return None
        self._touch(key, kind)
        return data

    def _write(self, key, kind, data):
        path = self._file(key, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f'{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
        temp.write_bytes(data)
        os.replace(temp, path)
        self._record(key, kind, len(data))

    def total_bytes(self):
        try:
            with self._lock:
                return int(self._index().execute('SELECT COALESCE(SUM(bytes), 0) FROM entries').fetchone()[0])
        except Exception:
            return 0

    def trim(self, target=None):
        """Drop least recently used entries until the cache is below ``target`` bytes (90 % of the
        limit by default)."""
        target = int(self.limit * .9) if target is None else target
        total = self.total_bytes()
        if total <= max(target, 0) and self.limit:
            return 0
        removed = 0
        with self._lock:
            rows = self._index().execute('SELECT key, kind, bytes FROM entries ORDER BY used').fetchall()
        for key, kind, size in rows:
            if total <= target:
                break
            self._forget(key, kind)
            total -= size
            removed += 1
        return removed

    def clear(self):
        """Empty the cache (memory and disk)."""
        with self._lock:
            self._memory.clear()
            removed = self.trim(-1)
            try:
                self._index().execute('DELETE FROM thumbs')
                self._index().execute('VACUUM')
            except sqlite3.Error:
                pass
        return removed

    def set_limit(self, megabytes):
        self.limit = max(0, int(megabytes)) * 1024 * 1024
        if self.limit:
            self._writer.submit(self.trim)
        else:
            self._writer.submit(self.clear)

    def _later(self, function, *args):
        def run():
            try:
                function(*args)
                self._pending += 1
                if self._pending >= 16:
                    self._pending = 0
                    self.trim()
            except Exception:
                log.warning('preview cache write failed', exc_info=True)
        return self._writer.submit(run)

    def flush(self):
        """Wait for queued writes (tests, shutting down)."""
        self._writer.submit(lambda: None).result()

    def close(self):
        self._prefetch.shutdown(wait=False, cancel_futures=True)
        self._writer.shutdown(wait=True)
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    # -- decoded previews

    def _remember(self, key, value):
        with self._lock:
            self._memory[key] = value
            self._memory.move_to_end(key)
            while len(self._memory) > MEMORY_ITEMS:
                self._memory.popitem(last=False)

    def cached(self, path):
        """The decoded preview of ``path`` from memory or disk, or None.  ``(rgb, info)`` are copies."""
        try:
            key, _ = stamp(path)
        except OSError:
            return None
        with self._lock:
            found = self._memory.get(key)
            if found is not None:
                self._memory.move_to_end(key)
        if found is None and self.enabled:
            data = self._read(key, 'd')
            if data is not None:
                try:
                    found = unpack(data)
                except Exception:
                    log.info('dropped a damaged preview of %s', path, exc_info=True)
                    self._forget(key, 'd')
                    found = None
                if found is not None:
                    self._remember(key, found)
        if found is None:
            return None
        rgb, info = found
        return rgb.copy(), copy.deepcopy(info)

    def load(self, path):
        """``engine.load_image(path)`` through the cache: ``(rgb, info)``.  A cache that cannot be
        read or written (permissions, a full disk) never keeps a photo from opening."""
        from . import engine
        if not self.enabled:
            return engine.load_image(path)
        try:
            found = self.cached(path)
            key, _ = stamp(path)
        except Exception:
            log.warning('preview cache unavailable', exc_info=True)
            return engine.load_image(path)
        if found is not None:
            self.hits += 1
            return found[0], engine.located(found[1], path)       # 2.1.2: the sidecar's position, now
        with self._lock:
            future = self._inflight.get(key)
            owner = future is None
            if owner:
                future = self._inflight[key] = Future()
        if not owner:
            rgb, info = future.result()
            return rgb.copy(), engine.located(copy.deepcopy(info), path)
        try:
            self.misses += 1
            rgb, info = engine._load_image(path)
            rgb = np.ascontiguousarray(quantize(rgb))
            self._remember(key, (rgb, info))
            self._later(self._write, key, 'd', pack(rgb, info))
            future.set_result((rgb, info))
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)
        return rgb.copy(), engine.located(copy.deepcopy(info), path)

    def prefetch(self, paths):
        """Decode ``paths`` in the background (one at a time, skipped while ``paused()``)."""
        if not self.enabled:
            return

        def run(path):
            if self.paused():
                return
            try:
                key, _ = stamp(path)
                with self._lock:
                    if key in self._memory or key in self._inflight:
                        return
                if self._file(key, 'd').is_file():
                    self.cached(path)       # warm memory
                else:
                    self.load(path)
            except Exception:
                log.debug('prefetch of %s failed', path, exc_info=True)
        for path in paths:
            self._prefetch.submit(run, path)

    # -- camera views and thumbnails

    def view(self, path, long_edge=None):
        """``previews.camera_view`` of ``path`` as uint8 RGB, through the cache."""
        from . import previews
        long_edge = long_edge or previews.VIEW_EDGE
        quick = Path(path).suffix.lower() in previews.QUICK
        if not self.enabled or quick or long_edge != previews.VIEW_EDGE:
            return previews.camera_view(path, long_edge)
        key, _ = stamp(path)
        try:
            data = self._read(key, 'v')
        except Exception:
            log.warning('preview cache unavailable', exc_info=True)
            data = None
        if data is not None:
            try:
                return previews.jpeg_view(data, None)
            except Exception:
                self._forget(key, 'v')
        array = previews.camera_view(path, long_edge)
        encoded = previews.encode_jpeg(array, 90)
        self._later(self._write, key, 'v', encoded)
        return array

    def thumbnail(self, path, view=None):
        """A JPEG thumbnail (≤ 320 pixels) of ``path`` as bytes, through the cache; ``view`` is a
        camera view already at hand."""
        from . import previews
        key, _ = stamp(path)
        if self.enabled:
            try:
                with self._lock:
                    row = self._index().execute('SELECT data FROM thumbs WHERE key=?', (key,)).fetchone()
            except Exception:
                log.warning('preview cache unavailable', exc_info=True)
                row = None
            if row is not None:
                self._touch(key, 't')
                return bytes(row[0])
        if view is None:
            view = previews.camera_view(path, THUMB_EDGE * 2)
        data = previews.encode_jpeg(previews.thumbnail_of(view, THUMB_EDGE), 85)
        if self.enabled:
            def store():
                with self._lock:
                    self._index().execute('INSERT OR REPLACE INTO thumbs(key, data) VALUES (?, ?)', (key, data))
                self._record(key, 't', len(data))
            self._later(store)
        return data

    def has_thumbnail(self, path):
        try:
            key, _ = stamp(path)
            with self._lock:
                return self._index().execute('SELECT 1 FROM thumbs WHERE key=?', (key,)).fetchone() is not None
        except Exception:
            return False


_default = None
_default_lock = threading.Lock()


def default():
    """The process-wide cache (created on first use)."""
    global _default
    with _default_lock:
        if _default is None:
            _default = PreviewCache()
        return _default


def load(path):
    return default().load(path)
