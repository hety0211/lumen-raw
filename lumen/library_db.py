"""The photo library (2.0.0): a SQLite catalog of folders, photos, keywords and smart albums.

Before 2.0 LUMEN only knew the photos of the open 选片集 (album).  The library remembers every
photo imported, by folder, with the capture data the library shows and filters by (camera, lens,
exposure, date with sub-seconds for burst grouping), its keywords and the culling hints
(``culling``: sharpness, faces, closed eyes).  It lives in ``library.db`` in the data folder
(``LUMEN_LIBRARY`` overrides it; next to ``LUMEN_CATALOG`` when only that is set).

Star ratings, flags and colour labels stay in ``catalog.Catalog`` (the append-only log the
command line and MCP agents already share); queries join the two by path.  Photos are never
written to: keywords, like ratings, go to XMP sidecars only when the user turned that on.

The database runs in WAL mode with one connection per thread, so the interface reads while the
background indexer writes, and the command line can use it at the same time.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

from . import catalog
from .i18n import N_

log = logging.getLogger(__name__)

SCHEMA = 1
#: Supported photo files (as ``commands.SUPPORTED``).
EXTENSIONS = None
#: Version of the culling analysis stored with each photo; older results are computed again.
ANALYSIS = 1
#: Folders never descended into when a folder is imported.
SKIP_FOLDERS = {'$recycle.bin', 'system volume information', '.lumen-cache', '__macosx', '.trash', '.trashes',
                '.spotlight-v100', '.fseventsd', '@eadir'}
#: Photos in one batch of the ExifTool index.
INDEX_BATCH = 64
#: Most photos one import adds.
IMPORT_LIMIT = 100000
#: Burst grouping: at most this many seconds between frames of the same camera …
BURST_GAP = 2.0
#: … whose 16 × 16 signatures (each normalised to zero mean, unit deviation) differ by at most
#: this much (mean absolute difference; the same scene stays well below, another one above).
BURST_DIFFERENCE = .6
#: Built-in smart albums: (id, title, query).
BUILTIN = (('picked', N_('已留用'), 'flag:pick'), ('stars3', N_('★ 3 以上'), 'rating>=3'),
           ('rejected', N_('已排除'), 'flag:reject'), ('blurry', N_('可能模糊'), 'blurry'),
           ('closed', N_('可能闭眼'), 'eyes:closed'), ('bursts', N_('连拍组'), 'burst'),
           ('best', N_('连拍中最佳'), 'burst:best'), ('untagged', N_('没有关键词'), 'keyword:none'))

COLUMNS = ('id', 'path', 'folder_id', 'name', 'ext', 'size', 'mtime', 'added', 'indexed', 'captured', 'captured_ts',
           'make', 'model', 'lens', 'iso', 'aperture', 'shutter', 'focal', 'width', 'height', 'orientation',
           'gps_lat', 'gps_lon', 'serial', 'analysis', 'sharpness', 'faces', 'eyes_closed', 'signature', 'face_data',
           'missing')


def extensions():
    global EXTENSIONS
    if EXTENSIONS is None:
        from .engine import RAW_EXTENSIONS
        EXTENSIONS = frozenset(RAW_EXTENSIONS | {'.jpg', '.jpeg', '.png', '.tif', '.tiff'})
    return EXTENSIONS


def default_path():
    override = os.environ.get('LUMEN_LIBRARY')
    if override:
        return Path(override)
    ratings = os.environ.get('LUMEN_CATALOG')
    if ratings:
        return Path(ratings).with_name('library.db')
    from .host import data_folder
    return data_folder() / 'library.db'


def key(path):
    return os.path.normcase(str(Path(path).resolve()))


def capture_time(tags):
    """``(text, seconds)`` of a photo's capture time with sub-seconds, or ``('', None)``.  The
    seconds only order frames of one camera, so the time zone is ignored."""
    for name in ('SubSecDateTimeOriginal', 'DateTimeOriginal', 'CreateDate'):
        value = str(tags.get(name) or '').strip()
        match = re.match(r'^(\d{4}):(\d\d):(\d\d)[ T](\d\d):(\d\d):(\d\d)(\.\d+)?', value)
        if not match or match.group(1) == '0000':
            continue
        y, mo, d, h, mi, s, fraction = match.groups()
        try:
            moment = _dt.datetime(int(y), int(mo), int(d), int(h), int(mi), int(s), tzinfo=_dt.timezone.utc)
        except ValueError:
            continue
        seconds = moment.timestamp() + (float(fraction) if fraction else 0.)
        text = f'{y}-{mo}-{d} {h}:{mi}:{s}' + (fraction[:4] if fraction else '')
        return text, seconds
    return '', None


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float('inf') else None


def _keywords(tags):
    found = []
    for name in ('Subject', 'Keywords'):
        value = tags.get(name)
        values = value if isinstance(value, list) else [value] if value not in (None, '') else []
        for item in values:
            for part in str(item).split(',') if name == 'Keywords' and isinstance(value, str) else [str(item)]:
                part = part.strip()
                if part and part.lower() not in {k.lower() for k in found}:
                    found.append(part[:100])
    return found


def fields_from_tags(tags):
    """Catalog columns from ExifTool JSON (``read_tags``)."""
    captured, seconds = capture_time(tags)
    size = str(tags.get('ImageSize') or '')
    match = re.match(r'^(\d+)\D+(\d+)$', size)
    lens = str(tags.get('LensModel') or tags.get('LensID') or tags.get('Lens') or '').strip()
    if lens.lower().startswith('unknown'):
        lens = ''
    return dict(captured=captured, captured_ts=seconds, make=str(tags.get('Make') or '').strip()[:80],
                model=str(tags.get('Model') or '').strip()[:80], lens=lens[:120], iso=_number(tags.get('ISO')),
                aperture=_number(tags.get('FNumber')), shutter=_number(tags.get('ExposureTime')),
                focal=_number(tags.get('FocalLength')), width=int(match.group(1)) if match else None,
                height=int(match.group(2)) if match else None, orientation=int(_number(tags.get('Orientation')) or 1),
                gps_lat=_number(tags.get('GPSLatitude')), gps_lon=_number(tags.get('GPSLongitude')),
                serial=str(tags.get('SerialNumber') or '')[:40])


TAGS = ('-Make', '-Model', '-LensModel', '-Lens', '-LensID', '-DateTimeOriginal', '-SubSecDateTimeOriginal', '-CreateDate',
        '-ISO#', '-FNumber#', '-ExposureTime#', '-FocalLength#', '-Orientation#', '-Composite:GPSLatitude#',
        '-Composite:GPSLongitude#', '-ImageSize#', '-SerialNumber', '-XMP-xmp:Rating#', '-XMP-xmp:Label',
        '-XMP-dc:Subject', '-IPTC:Keywords')


def read_tags(paths):
    """``{path: tags}`` for photos (ExifTool, one process per batch); sidecar XMP keywords,
    ratings and labels win over the photo's own."""
    from .catalog import _run, _sidecars
    result = {}
    names = [str(Path(p).resolve()) for p in paths]
    if not names:
        return result
    sidecars = {}
    for name in names:
        found = _sidecars(name)
        if found:
            sidecars[str(found[0])] = name
    files = names + list(sidecars)
    with tempfile.NamedTemporaryFile('w', suffix='.args', delete=False, encoding='utf-8') as handle:
        handle.write('\n'.join(['-j', '-fast', '-charset', 'filename=UTF8', *TAGS, '--', *files]))
        arguments = handle.name
    try:
        output = _run(['-charset', 'filename=UTF8', '-@', arguments], timeout=max(60, len(files)))
    finally:
        os.unlink(arguments)
    try:
        records = json.loads(output.stdout.decode('utf-8') or '[]')
    except ValueError:
        records = []
    by_key = {os.path.normcase(str(Path(n))): n for n in names}
    side_keys = {os.path.normcase(str(Path(s))): p for s, p in sidecars.items()}
    for record in records:
        source = os.path.normcase(str(Path(record.get('SourceFile', ''))))
        if source in by_key:
            result.setdefault(by_key[source], {}).update({k: v for k, v in record.items() if k != 'SourceFile'})
    for record in records:
        source = os.path.normcase(str(Path(record.get('SourceFile', ''))))
        photo = side_keys.get(source)
        if photo:
            tags = result.setdefault(photo, {})
            for name in ('Rating', 'Label', 'Subject', 'Keywords'):
                if name in record:
                    tags[name] = record[name]
    return result


class LibraryDB:
    """The catalog database.  Every method may be called from any thread."""

    def __init__(self, path=None):
        self.path = Path(path) if path else default_path()
        self._local = threading.local()
        self._lock = threading.RLock()
        self._connections = []
        self.listeners = []
        with self._lock:
            try:
                self._setup(self._db())
            except sqlite3.DatabaseError:
                # A damaged file (a crash of the disk, a copy cut short): keep it aside, start anew.
                self.close()
                aside = self.path.with_name(f'{self.path.stem}.damaged-{time.strftime("%Y%m%d-%H%M%S")}{self.path.suffix}')
                log.error('library database %s is damaged; moved to %s', self.path, aside)
                os.replace(self.path, aside)
                for suffix in ('-wal', '-shm'):
                    Path(str(self.path) + suffix).unlink(missing_ok=True)
                self._setup(self._db())

    # -- connections

    def _db(self):
        connection = getattr(self._local, 'connection', None)
        if connection is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(str(self.path), timeout=15, isolation_level=None, check_same_thread=False)
            connection.row_factory = sqlite3.Row
            try:
                connection.execute('PRAGMA journal_mode=WAL')
                connection.execute('PRAGMA synchronous=NORMAL')
                connection.execute('PRAGMA foreign_keys=ON')
            except sqlite3.Error:
                connection.close()
                raise
            self._local.connection = connection
            with self._lock:
                self._connections.append(connection)
        return connection

    def close(self):
        with self._lock:
            for connection in self._connections:
                try:
                    connection.close()
                except sqlite3.Error:
                    pass
            self._connections.clear()
        self._local = threading.local()

    def _setup(self, db):
        db.executescript('''
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS folders(id INTEGER PRIMARY KEY, path TEXT NOT NULL, key TEXT NOT NULL UNIQUE,
                root INTEGER NOT NULL DEFAULT 0, added REAL);
            CREATE TABLE IF NOT EXISTS photos(id INTEGER PRIMARY KEY, path TEXT NOT NULL, key TEXT NOT NULL UNIQUE,
                folder_id INTEGER REFERENCES folders(id) ON DELETE CASCADE, name TEXT, ext TEXT, size INTEGER,
                mtime REAL, added REAL, indexed INTEGER NOT NULL DEFAULT 0, captured TEXT, captured_ts REAL,
                make TEXT, model TEXT, lens TEXT, iso REAL, aperture REAL, shutter REAL, focal REAL,
                width INTEGER, height INTEGER, orientation INTEGER, gps_lat REAL, gps_lon REAL, serial TEXT,
                analysis INTEGER NOT NULL DEFAULT 0, sharpness REAL, faces INTEGER, eyes_closed INTEGER,
                signature BLOB, face_data TEXT, missing INTEGER NOT NULL DEFAULT 0);
            CREATE INDEX IF NOT EXISTS photos_folder ON photos(folder_id);
            CREATE INDEX IF NOT EXISTS photos_captured ON photos(captured_ts);
            CREATE TABLE IF NOT EXISTS keywords(id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE COLLATE NOCASE);
            CREATE TABLE IF NOT EXISTS photo_keywords(photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
                keyword_id INTEGER NOT NULL REFERENCES keywords(id) ON DELETE CASCADE, PRIMARY KEY(photo_id, keyword_id));
            CREATE INDEX IF NOT EXISTS photo_keywords_keyword ON photo_keywords(keyword_id);
            CREATE TABLE IF NOT EXISTS collections(id INTEGER PRIMARY KEY, name TEXT NOT NULL, query TEXT NOT NULL,
                created REAL, position INTEGER NOT NULL DEFAULT 0);
        ''')
        db.execute('INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)', ('schema', str(SCHEMA)))

    def _changed(self, kind, ids=()):
        for listener in list(self.listeners):
            try:
                listener(kind, list(ids))
            except Exception:
                log.exception('library listener failed')

    # -- folders and photos

    def _folder(self, db, folder, root=False):
        name = key(folder)
        row = db.execute('SELECT id, root FROM folders WHERE key=?', (name,)).fetchone()
        if row is not None:
            if root and not row['root']:
                db.execute('UPDATE folders SET root=1 WHERE id=?', (row['id'],))
            return row['id']
        return db.execute('INSERT INTO folders(path, key, root, added) VALUES (?, ?, ?, ?)',
                          (str(Path(folder).resolve()), name, int(root), time.time())).lastrowid

    @staticmethod
    def scan(folder, recursive=True, limit=IMPORT_LIMIT):
        """Supported photos in ``folder`` (sorted), skipping system and hidden folders."""
        found = []
        root = Path(folder)
        stack = [root]
        while stack and len(found) < limit:
            current = stack.pop()
            try:
                entries = sorted(os.scandir(current), key=lambda e: e.name.lower())
            except OSError:
                continue
            folders = []
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if recursive and entry.name.lower() not in SKIP_FOLDERS and not entry.name.startswith('.'):
                            folders.append(Path(entry.path))
                    elif entry.is_file() and Path(entry.name).suffix.lower() in extensions() and not entry.name.startswith('._'):
                        found.append(str(Path(entry.path).resolve()))
                except OSError:
                    continue
            stack.extend(reversed(folders))
        return found[:limit]

    def add(self, paths, roots=()):
        """Add photos (and mark ``roots`` as imported folders); returns ``(added ids, all ids)``."""
        added, ids = [], []
        now = time.time()
        with self._lock:
            db = self._db()
            db.execute('BEGIN IMMEDIATE')
            try:
                for root in roots:
                    self._folder(db, root, root=True)
                folders = {}
                for path in paths:
                    resolved = Path(path).resolve()
                    name = key(resolved)
                    row = db.execute('SELECT id, missing FROM photos WHERE key=?', (name,)).fetchone()
                    if row is not None:
                        ids.append(row['id'])
                        if row['missing']:
                            db.execute('UPDATE photos SET missing=0 WHERE id=?', (row['id'],))
                        continue
                    try:
                        status = resolved.stat()
                    except OSError:
                        continue
                    parent = str(resolved.parent)
                    if parent not in folders:
                        folders[parent] = self._folder(db, parent)
                    photo = db.execute('INSERT INTO photos(path, key, folder_id, name, ext, size, mtime, added) '
                                       'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                                       (str(resolved), name, folders[parent], resolved.name, resolved.suffix.lower(),
                                        status.st_size, status.st_mtime, now)).lastrowid
                    added.append(photo)
                    ids.append(photo)
                db.execute('COMMIT')
            except BaseException:
                db.execute('ROLLBACK')
                raise
        if added or roots:
            self._changed('photos', added)
        return added, ids

    def import_folder(self, folder, recursive=True):
        """Add every supported photo of ``folder``; returns ``(added ids, all ids)``."""
        return self.add(self.scan(folder, recursive), roots=[folder])

    def rescan(self, folder_id, recursive=True):
        """Add new files of a folder and mark files that are gone as missing; ``(added, missing)``."""
        folder = self.folder(folder_id)
        if folder is None:
            return [], []
        added, ids = self.add(self.scan(folder['path'], recursive)) if Path(folder['path']).is_dir() else ([], [])
        missing = []
        for row in self.rows(('folder', folder_id, recursive)):
            gone = not Path(row['path']).is_file()
            if gone != bool(row['missing']):
                self._db().execute('UPDATE photos SET missing=? WHERE id=?', (int(gone), row['id']))
            if gone:
                missing.append(row['id'])
        self._changed('photos', added + missing)
        return added, missing

    def remove(self, ids):
        """Forget photos (files on disk are not touched)."""
        ids = list(ids)
        with self._lock:
            db = self._db()
            for start in range(0, len(ids), 500):
                chunk = ids[start:start + 500]
                db.execute(f'DELETE FROM photos WHERE id IN ({",".join("?" * len(chunk))})', chunk)
            db.execute('DELETE FROM folders WHERE root=0 AND id NOT IN (SELECT DISTINCT folder_id FROM photos '
                       'WHERE folder_id IS NOT NULL)')
        self._changed('removed', ids)

    def remove_folder(self, folder_id):
        """Forget a folder, its sub-folders and their photos."""
        folder = self.folder(folder_id)
        if folder is None:
            return 0
        ids = [row['id'] for row in self.rows(('folder', folder_id, True))]
        prefix = folder['key'].rstrip('\\/') + os.sep
        with self._lock:
            db = self._db()
            self.remove(ids)
            db.execute('DELETE FROM folders WHERE id=? OR substr(key, 1, ?)=?', (folder_id, len(prefix), prefix))
        self._changed('folders')
        return len(ids)

    def folder(self, folder_id):
        row = self._db().execute('SELECT * FROM folders WHERE id=?', (folder_id,)).fetchone()
        return dict(row) if row else None

    def folders(self):
        """Every folder with its photo count: ``[{id, path, key, root, count}]``."""
        rows = self._db().execute('SELECT f.id, f.path, f.key, f.root, COUNT(p.id) AS count FROM folders f '
                                  'LEFT JOIN photos p ON p.folder_id=f.id GROUP BY f.id ORDER BY f.key').fetchall()
        return [dict(r) for r in rows]

    def count(self):
        return int(self._db().execute('SELECT COUNT(*) FROM photos').fetchone()[0])

    def id_of(self, path):
        row = self._db().execute('SELECT id FROM photos WHERE key=?', (key(path),)).fetchone()
        return row['id'] if row else None

    def ids_of(self, paths):
        return [i for i in (self.id_of(p) for p in paths) if i is not None]

    def record(self, path):
        rows = self.rows(('paths', [path]))
        return rows[0] if rows else None

    # -- reading rows

    def rows(self, source=('all',)):
        """Photos of a source as dicts (catalog columns plus ``keywords``):

        ``('all',)``, ``('recent',)`` (the last import), ``('folder', id, include_sub_folders)``,
        ``('keyword', id)``, ``('paths', [paths])`` (in that order) or ``('ids', [ids])``."""
        db = self._db()
        kind = source[0]
        query = 'SELECT * FROM photos'
        args = []
        if kind == 'recent':
            row = db.execute('SELECT MAX(added) FROM photos').fetchone()
            if row[0] is None:
                return []
            query += ' WHERE added>=?'
            args.append(row[0] - 1)
        elif kind == 'folder':
            folder = self.folder(source[1])
            if folder is None:
                return []
            if len(source) > 2 and source[2]:
                prefix = folder['key'].rstrip('\\/') + os.sep
                query += ' WHERE folder_id IN (SELECT id FROM folders WHERE id=? OR substr(key, 1, ?)=?)'
                args += [folder['id'], len(prefix), prefix]
            else:
                query += ' WHERE folder_id=?'
                args.append(folder['id'])
        elif kind == 'keyword':
            query += ' WHERE id IN (SELECT photo_id FROM photo_keywords WHERE keyword_id=?)'
            args.append(source[1])
        elif kind in ('paths', 'ids'):
            wanted = [key(p) for p in source[1]] if kind == 'paths' else list(source[1])
            rows = {}
            for start in range(0, len(wanted), 500):
                chunk = wanted[start:start + 500]
                column = 'key' if kind == 'paths' else 'id'
                for row in db.execute(f'SELECT * FROM photos WHERE {column} IN ({",".join("?" * len(chunk))})', chunk):
                    rows[row[column]] = dict(row)
            result = [rows[w] for w in wanted if w in rows]
            self._attach_keywords(result)
            return result
        result = [dict(r) for r in db.execute(query, args).fetchall()]
        self._attach_keywords(result)
        return result

    def _attach_keywords(self, rows):
        by_id = {r['id']: r for r in rows}
        for r in rows:
            r['keywords'] = []
        if not rows:
            return rows
        db = self._db()
        if len(rows) > 400:
            pairs = db.execute('SELECT pk.photo_id, k.name FROM photo_keywords pk JOIN keywords k ON k.id=pk.keyword_id '
                               'ORDER BY k.name COLLATE NOCASE').fetchall()
        else:
            ids = list(by_id)
            pairs = db.execute(f'SELECT pk.photo_id, k.name FROM photo_keywords pk JOIN keywords k ON k.id=pk.keyword_id '
                               f'WHERE pk.photo_id IN ({",".join("?" * len(ids))}) ORDER BY k.name COLLATE NOCASE', ids).fetchall()
        for photo_id, name in pairs:
            if photo_id in by_id:
                by_id[photo_id]['keywords'].append(name)
        return rows

    # -- EXIF index

    def unindexed(self, limit=INDEX_BATCH):
        return [dict(r) for r in self._db().execute('SELECT id, path FROM photos WHERE indexed=0 AND missing=0 '
                                                     'ORDER BY id LIMIT ?', (limit,)).fetchall()]

    def index(self, rows, ratings=None):
        """Read capture data, XMP ratings and keywords of ``rows`` (``[{id, path}]``) with
        ExifTool.  Ratings found in XMP fill photos ``ratings`` (a ``catalog.Catalog``) does not
        know yet.  Returns the ids indexed."""
        if not rows:
            return []
        try:
            tags = read_tags([r['path'] for r in rows])
        except Exception:
            log.warning('ExifTool index failed', exc_info=True)
            tags = {}
        imported = {}
        done = []
        with self._lock:
            db = self._db()
            db.execute('BEGIN IMMEDIATE')
            try:
                for row in rows:
                    found = tags.get(str(Path(row['path']).resolve()), {})
                    fields = fields_from_tags(found)
                    missing = 0 if Path(row['path']).is_file() else 1
                    db.execute(f'UPDATE photos SET indexed=1, missing=?, {", ".join(f"{k}=?" for k in fields)} WHERE id=?',
                               (missing, *fields.values(), row['id']))
                    words = _keywords(found)
                    if words:
                        self._add_keywords(db, [row['id']], words)
                    rating = catalog._fields_from_tags(found)
                    if rating and not (found.get('Rating') == 0 and set(rating) == {'rating'}):
                        imported[row['path']] = rating
                    done.append(row['id'])
                db.execute('COMMIT')
            except BaseException:
                db.execute('ROLLBACK')
                raise
        if ratings is not None and imported:
            try:
                ratings.merge(imported)
            except Exception:
                log.warning('XMP ratings not merged', exc_info=True)
        self._changed('indexed', done)
        return done

    def index_all(self, ratings=None, cancelled=lambda: False):
        """Index every photo that is not yet (the command line, tests)."""
        total = 0
        while not cancelled():
            rows = self.unindexed()
            if not rows:
                break
            total += len(self.index(rows, ratings))
        return total

    # -- culling analysis

    def unanalysed(self, ids=None, limit=200):
        db = self._db()
        if ids is None:
            rows = db.execute('SELECT id, path FROM photos WHERE analysis<? AND missing=0 ORDER BY id LIMIT ?',
                              (ANALYSIS, limit)).fetchall()
        else:
            ids = list(ids)
            rows = []
            for start in range(0, len(ids), 500):
                chunk = ids[start:start + 500]
                rows += db.execute(f'SELECT id, path FROM photos WHERE analysis<? AND missing=0 AND id IN '
                                   f'({",".join("?" * len(chunk))})', (ANALYSIS, *chunk)).fetchall()
        return [dict(r) for r in rows]

    def set_analysis(self, photo_id, result):
        with self._lock:
            self._db().execute('UPDATE photos SET analysis=?, sharpness=?, faces=?, eyes_closed=?, signature=?, '
                               'face_data=? WHERE id=?',
                               (ANALYSIS, result.get('sharpness'), result.get('faces'), result.get('eyes_closed'),
                                result.get('signature'), json.dumps(result.get('boxes') or []), photo_id))
        self._changed('analysed', [photo_id])

    # -- keywords

    def _add_keywords(self, db, ids, names):
        for name in names:
            name = str(name).strip()[:100]
            if not name:
                continue
            db.execute('INSERT OR IGNORE INTO keywords(name) VALUES (?)', (name,))
            keyword = db.execute('SELECT id FROM keywords WHERE name=?', (name,)).fetchone()['id']
            db.executemany('INSERT OR IGNORE INTO photo_keywords(photo_id, keyword_id) VALUES (?, ?)',
                           [(i, keyword) for i in ids])

    def add_keywords(self, ids, names):
        names = clean_keywords(names)
        ids = list(ids)
        if not ids or not names:
            return
        with self._lock:
            db = self._db()
            db.execute('BEGIN IMMEDIATE')
            try:
                self._add_keywords(db, ids, names)
                db.execute('COMMIT')
            except BaseException:
                db.execute('ROLLBACK')
                raise
        self._changed('keywords', ids)

    def remove_keywords(self, ids, names):
        ids = list(ids)
        with self._lock:
            db = self._db()
            for name in clean_keywords(names):
                row = db.execute('SELECT id FROM keywords WHERE name=?', (name,)).fetchone()
                if row is None:
                    continue
                db.executemany('DELETE FROM photo_keywords WHERE photo_id=? AND keyword_id=?', [(i, row['id']) for i in ids])
            db.execute('DELETE FROM keywords WHERE id NOT IN (SELECT DISTINCT keyword_id FROM photo_keywords)')
        self._changed('keywords', ids)

    def set_keywords(self, ids, names):
        ids = list(ids)
        with self._lock:
            db = self._db()
            db.executemany('DELETE FROM photo_keywords WHERE photo_id=?', [(i,) for i in ids])
            self._add_keywords(db, ids, clean_keywords(names))
            db.execute('DELETE FROM keywords WHERE id NOT IN (SELECT DISTINCT keyword_id FROM photo_keywords)')
        self._changed('keywords', ids)

    def keywords(self):
        """``[{id, name, count}]`` by name."""
        rows = self._db().execute('SELECT k.id, k.name, COUNT(pk.photo_id) AS count FROM keywords k '
                                  'LEFT JOIN photo_keywords pk ON pk.keyword_id=k.id GROUP BY k.id '
                                  'ORDER BY k.name COLLATE NOCASE').fetchall()
        return [dict(r) for r in rows]

    # -- smart albums

    def collections(self):
        return [dict(r) for r in self._db().execute('SELECT * FROM collections ORDER BY position, id').fetchall()]

    def add_collection(self, name, query):
        catalog.parse_query(query)          # raises ValueError
        with self._lock:
            row = self._db().execute('INSERT INTO collections(name, query, created, position) VALUES (?, ?, ?, '
                                     '(SELECT COALESCE(MAX(position), 0) + 1 FROM collections))',
                                     (str(name).strip()[:80] or query, query, time.time())).lastrowid
        self._changed('collections')
        return row

    def update_collection(self, collection_id, name=None, query=None):
        if query is not None:
            catalog.parse_query(query)
        with self._lock:
            if name is not None:
                self._db().execute('UPDATE collections SET name=? WHERE id=?', (str(name).strip()[:80], collection_id))
            if query is not None:
                self._db().execute('UPDATE collections SET query=? WHERE id=?', (query, collection_id))
        self._changed('collections')

    def remove_collection(self, collection_id):
        with self._lock:
            self._db().execute('DELETE FROM collections WHERE id=?', (collection_id,))
        self._changed('collections')

    def collection(self, collection_id):
        row = self._db().execute('SELECT * FROM collections WHERE id=?', (collection_id,)).fetchone()
        return dict(row) if row else None

    # -- queries

    def search(self, ratings, query='', source=('all',), edited=None, sort='captured'):
        """Rows of ``source`` matching ``query`` (``catalog.parse_query`` syntax, with the 2.0
        fields), each with ``rating``, ``flag``, ``label`` from ``ratings`` and burst fields;
        sorted by ``sort`` (captured, name, rating, added).  Raises ValueError for a bad query."""
        terms = catalog.parse_query(query)
        rows = self.rows(source)
        annotate_bursts(rows)
        annotate_hints(rows)
        result = []
        for row in rows:
            meta = ratings.get(row['path']) if ratings is not None else catalog.empty()
            row.update(meta)
            if catalog.matches(terms, row['path'], meta, edited(row['path']) if edited else False, row):
                result.append(row)
        return sort_rows(result, sort)


def clean_keywords(names):
    if isinstance(names, str):
        names = re.split(r'[,，;；\n]', names)
    result = []
    for name in names:
        name = str(name).strip().strip('"').strip()[:100]
        if name and name.lower() not in {n.lower() for n in result}:
            result.append(name)
    return result


def sort_rows(rows, sort='captured'):
    if sort == 'name':
        return sorted(rows, key=lambda r: (r['name'] or '').lower())
    if sort == 'rating':
        return sorted(rows, key=lambda r: (-int(r.get('rating') or 0), r.get('captured_ts') or 0, (r['name'] or '').lower()))
    if sort == 'added':
        return sorted(rows, key=lambda r: (-(r.get('added') or 0), (r['name'] or '').lower()))
    return sorted(rows, key=lambda r: (r.get('captured_ts') is None, r.get('captured_ts') or 0, (r['name'] or '').lower()))


def _similar(a, b):
    if not a or not b or len(a) != len(b):
        return True
    return signature_difference(a, b) <= BURST_DIFFERENCE


def signature_difference(a, b):
    """How different two 16 × 16 signatures look, brightness and contrast aside."""
    import numpy as np
    x = np.frombuffer(a, np.uint8).astype(np.float32)
    y = np.frombuffer(b, np.uint8).astype(np.float32)
    x = (x - x.mean()) / (x.std() + 4)
    y = (y - y.mean()) / (y.std() + 4)
    return float(np.mean(np.abs(x - y)))


def _stem(row):
    return Path(row.get('name') or row.get('path') or '').stem.lower()


def _copy(a, b):
    """Same moment and the same file name stem (a copy in another folder, an export such as
    ``name-Lumen.jpg``, or the JPEG of a RAW + JPEG pair): not another frame of a burst."""
    if abs((a.get('captured_ts') or 0) - (b.get('captured_ts') or 0)) > 1e-3:
        return False
    x, y = _stem(a), _stem(b)
    return bool(x and y) and (x == y or x.startswith(y) or y.startswith(x))


def annotate_bursts(rows, gap=BURST_GAP):
    """Group frames shot in quick succession by one camera: sets ``burst`` (group number, 0 when
    alone), ``burst_index`` / ``burst_size`` and ``best`` (the frame to keep: eyes open first,
    then the sharpest) on every row."""
    for row in rows:
        row.update(burst=0, burst_index=0, burst_size=1, best=False)
    timed = sorted((r for r in rows if r.get('captured_ts') is not None),
                   key=lambda r: ((r.get('make') or '', r.get('model') or '', r.get('serial') or ''), r['captured_ts'], r['name'] or ''))
    groups, current = [], []
    for row in timed:
        if current:
            if any(_copy(other, row) for other in current[-3:]):
                continue                    # a copy, an export or the JPEG of a RAW + JPEG pair
            last = current[-1]
            same = (last.get('make'), last.get('model'), last.get('serial')) == (row.get('make'), row.get('model'), row.get('serial'))
            if same and 0 <= row['captured_ts'] - last['captured_ts'] <= gap and _similar(last.get('signature'), row.get('signature')):
                current.append(row)
                continue
            if len(current) > 1:
                groups.append(current)
        current = [row]
    if len(current) > 1:
        groups.append(current)
    groups.sort(key=lambda g: g[0]['captured_ts'])
    for number, group in enumerate(groups, 1):
        best = max(group, key=lambda r: (not r.get('eyes_closed'), r.get('sharpness') or 0, -group.index(r)))
        for index, row in enumerate(group):
            row.update(burst=number, burst_index=index + 1, burst_size=len(group), best=row is best)
    return groups


#: Sharpness below this (0–100) is called blurry outright …
BLURRY = 22.
#: … and so is a frame this far below the sharpest of its burst.
BURST_BLURRY = 14.


def annotate_hints(rows):
    """``blurry`` on analysed rows: low sharpness, or much softer than the best frame of its burst."""
    best = {}
    for row in rows:
        if row.get('burst') and row.get('sharpness') is not None:
            best[row['burst']] = max(best.get(row['burst'], 0), row['sharpness'])
    for row in rows:
        value = row.get('sharpness')
        if value is None:
            row['blurry'] = False
            continue
        row['blurry'] = bool(value < BLURRY or (row.get('burst') and best.get(row['burst'], 0) - value > BURST_BLURRY))
    return rows


def write_keywords_xmp(entries):
    """Write ``{photo: [keywords]}`` into XMP sidecars (dc:subject, as Lightroom); photos are never
    modified.  Returns ``(written, errors)``."""
    from .catalog import _run, sidecar
    if not entries:
        return [], []
    lines = []
    for photo, words in entries.items():
        lines += ['-charset', 'filename=UTF8', '-overwrite_original', '-XMP-dc:Subject=']
        lines += [f'-XMP-dc:Subject={w}' for w in words]
        lines += [str(sidecar(photo).resolve()), '-execute']
    with tempfile.NamedTemporaryFile('w', suffix='.args', delete=False, encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
        arguments = handle.name
    try:
        result = _run(['-@', arguments], timeout=max(60, 2 * len(entries)))
    finally:
        os.unlink(arguments)
    written = [p for p in entries if sidecar(p).is_file()]
    errors = [line for line in result.stderr.decode('utf-8', 'replace').splitlines() if line.strip()
              and not line.startswith('Warning')]
    return written, errors


def write_keywords_xmp_later(entries):
    def run():
        try:
            write_keywords_xmp(entries)
        except Exception:
            log.warning('keyword XMP write failed', exc_info=True)
    threading.Thread(target=run, name='xmp-keywords', daemon=True).start()
