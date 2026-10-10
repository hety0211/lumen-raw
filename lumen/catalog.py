"""Library metadata: star ratings, pick / reject flags and colour labels (1.6.0).

The catalog is an append-only operation log, ``catalog.jsonl`` in the data folder: every change
is one JSON line, flushed and synced to disk at once, so a crash never loses a rating and there
is nothing to save.  Replaying the log rebuilds the state; a torn last line (a crash during the
write) is skipped.  The log is rewritten compactly when it has grown well past the number of
photos it describes.  Other processes (the command line, the MCP server) append to the same
file; ``refresh`` reads what they added.

Ratings and labels interoperate with XMP (``xmp:Rating``, ``xmp:Label``) through the bundled
ExifTool.  Lumen never writes into a photograph: it reads embedded XMP and sidecars, and writes
sidecars only (``<name>.xmp``, the Lightroom / Bridge / Capture One convention).  A rejected photo
is ``xmp:Rating = -1`` as Lightroom writes it.  Lightroom Classic catalogs (``.lrcat``) can be
imported read-only.
"""
from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

FLAGS = ('pick', 'reject')
LABELS = ('red', 'yellow', 'green', 'blue', 'purple')
#: Colours of the labels in the interface.
LABEL_COLORS = dict(red='#e0585b', yellow='#e6c84f', green='#69b46c', blue='#5b8fe0', purple='#a874d8')
#: xmp:Label values written (Lightroom's English names) and the names read in several languages.
XMP_LABELS = dict(red='Red', yellow='Yellow', green='Green', blue='Blue', purple='Purple')
_LABEL_NAMES = {
    'red': ('red', '红色', '紅色', '红', '紅', 'rot', 'rouge', 'rojo', 'rosso', '赤', '빨강', 'красный'),
    'yellow': ('yellow', '黄色', '黃色', '黄', '黃', 'gelb', 'jaune', 'amarillo', 'giallo', '노랑', 'жёлтый', 'желтый'),
    'green': ('green', '绿色', '綠色', '绿', '綠', 'grün', 'grun', 'vert', 'verde', '緑', '초록', 'зелёный', 'зеленый'),
    'blue': ('blue', '蓝色', '藍色', '蓝', '藍', 'blau', 'bleu', 'azul', 'blu', '青', '파랑', 'синий'),
    'purple': ('purple', '紫色', '紫', 'lila', 'violett', 'violet', 'morado', 'púrpura', 'viola', '보라', 'фиолетовый'),
}
COMPACT_RATIO = 4          # rewrite when the log has 4× more lines than photos (and ≥ 2000)


XMP_SETTING = 'library_xmp'


def xmp_sync():
    # A catalog chosen with LUMEN_CATALOG (tests, scripts) never writes sidecars.
    from .i18n import setting
    return os.environ.get('LUMEN_CATALOG') is None and setting(XMP_SETTING, '0') == '1'


def label_from_text(text):
    """A label key from an XMP / Lightroom label name in any common language, or ''."""
    name = str(text or '').strip().lower()
    for key, names in _LABEL_NAMES.items():
        if name in names:
            return key
    return ''


def default_path():
    override = os.environ.get('LUMEN_CATALOG')
    if override:
        return Path(override)
    from .host import data_folder
    return data_folder() / 'catalog.jsonl'


def key(path):
    """Catalog key of a photo: absolute, and case-folded where the file system is."""
    return os.path.normcase(str(Path(path).resolve()))


def empty():
    return dict(rating=0, flag='', label='')


def validate(fields):
    """Checked subset of ``rating`` / ``flag`` / ``label``; raises ValueError."""
    result = {}
    if 'rating' in fields:
        rating = fields['rating']
        if isinstance(rating, bool) or not isinstance(rating, (int, float)) or rating != int(rating) or not 0 <= rating <= 5:
            raise ValueError('rating must be an integer 0–5')
        result['rating'] = int(rating)
    if 'flag' in fields:
        flag = fields['flag'] or ''
        flag = '' if flag in ('none', 'unflagged') else flag
        if flag not in ('',) + FLAGS:
            raise ValueError('flag must be pick, reject or none')
        result['flag'] = flag
    if 'label' in fields:
        label = fields['label'] or ''
        label = '' if label == 'none' else label_from_text(label) or label
        if label not in ('',) + LABELS:
            raise ValueError('label must be red, yellow, green, blue, purple or none')
        result['label'] = label
    return result


class Catalog:
    """Ratings, flags and labels of photos, persisted as an append-only log."""

    def __init__(self, path=None):
        self.path = Path(path) if path else default_path()
        self._lock = threading.RLock()
        self._entries = {}
        self._names = {}
        self._offset = 0
        self._lines = 0
        self.listeners = []
        #: Mirror every change into XMP sidecars (setting in the 图库 menu; off by default).
        self.write_xmp_on_change = xmp_sync()
        self._load()

    # -- persistence

    def _apply(self, record):
        path = record.get('path')
        if not isinstance(path, str):
            return
        try:
            fields = validate({k: record[k] for k in ('rating', 'flag', 'label') if k in record})
        except ValueError:
            return
        name = key(path)
        entry = self._entries.setdefault(name, empty())
        entry.update(fields)
        self._names[name] = path

    def _read(self, start):
        try:
            with open(self.path, 'rb') as handle:
                handle.seek(start)
                data = handle.read()
        except FileNotFoundError:
            return 0
        complete = data.rfind(b'\n') + 1   # a torn last line waits for its end (or is skipped)
        count = 0
        for line in data[:complete].splitlines():
            if not line.strip():
                continue
            try:
                self._apply(json.loads(line.decode('utf-8')))
                count += 1
            except (ValueError, UnicodeDecodeError, AttributeError):
                log.warning('catalog: skipped a damaged line')
        self._offset = start + complete
        return count

    def _load(self):
        with self._lock:
            self._entries.clear()
            self._names.clear()
            self._lines = self._read(0)
            if self._lines >= 2000 and self._lines > COMPACT_RATIO * max(1, len(self._entries)):
                self.compact()

    def refresh(self):
        """Read lines other processes appended; True when anything changed."""
        with self._lock:
            try:
                size = self.path.stat().st_size
            except FileNotFoundError:
                return False
            if size < self._offset:          # compacted by another process
                self._load()
                return True
            if size == self._offset:
                return False
            added = self._read(self._offset)
            self._lines += added
            return bool(added)

    def _append(self, records):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records).encode('utf-8')
        with open(self.path, 'ab+') as handle:
            # After a torn write (a crash mid-line) start on a new line, so only that line is lost.
            if handle.seek(0, os.SEEK_END):
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b'\n':
                    data = b'\n' + data
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        self._lines += len(records)

    def compact(self):
        """Rewrite the log with one line per photo (atomically)."""
        with self._lock:
            records = [dict(path=self._names[k], **{f: v for f, v in e.items() if v}) for k, e in self._entries.items()
                       if any(e.values())]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix('.jsonl.tmp')
            with open(temp, 'w', encoding='utf-8') as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.path)
            self._offset = self.path.stat().st_size
            self._lines = len(records)

    # -- access

    def get(self, path):
        with self._lock:
            return dict(self._entries.get(key(path), empty()))

    def known(self, path):
        with self._lock:
            return key(path) in self._entries

    def set(self, paths, **fields):
        """Set fields of every photo in ``paths``; returns the paths that changed."""
        fields = validate(fields)
        if not fields:
            return []
        changed, records = [], []
        with self._lock:
            self.refresh()
            for path in paths:
                name = key(path)
                current = self._entries.get(name, empty())
                if all(current.get(k) == v for k, v in fields.items()):
                    continue
                record = dict(path=str(Path(path).resolve()), t=round(time.time(), 3), **fields)
                records.append(record)
                self._apply(record)
                changed.append(str(path))
            if records:
                start = self._offset
                self._append(records)
                # Re-read from before our lines: picks up anything another process appended meanwhile.
                self._lines += self._read(start) - len(records)
        if changed:
            if self.write_xmp_on_change:
                write_xmp_later({p: self.get(p) for p in changed})
            for listener in list(self.listeners):
                try:
                    listener(changed)
                except Exception:
                    log.exception('catalog listener failed')
        return changed

    def merge(self, values, overwrite=False):
        """Apply ``{path: fields}`` (an import); only empty entries unless ``overwrite``."""
        changed = []
        for path, fields in values.items():
            if not overwrite and self.known(path) and any(self.get(path).values()):
                continue
            changed += self.set([path], **fields)
        return changed

    def items(self):
        with self._lock:
            return [(self._names[k], dict(v)) for k, v in self._entries.items()]


# --------------------------------------------------------------------------- queries

_FIELD = re.compile(r'^(-?)(rating|stars|star|flag|label|color|colour|name|ext|folder|edited)\s*(>=|<=|=|:|>|<)\s*(.*)$', re.I)


def parse_query(text):
    """``rating>=3 label:red flag:pick name:DSC0* -flag:reject edited`` → list of terms.

    Fields: rating / stars (= : > >= < <=), flag (pick, reject, none), label / color (red …,
    none, any), name (substring or wildcard), ext, folder (substring), edited (yes / no).  A bare
    word matches the file name; ``picked``, ``rejected``, ``unflagged``, ``edited`` and colour
    names are shorthands; a leading ``-`` negates a term.  All terms must match."""
    terms = []
    for token in str(text or '').split():
        negate = token.startswith('-') and len(token) > 1
        word = token[1:] if negate else token
        lowered = word.lower()
        match = _FIELD.match(word)
        if match:
            _, field, op, value = match.groups()
            field = {'stars': 'rating', 'star': 'rating', 'color': 'label', 'colour': 'label'}.get(field.lower(), field.lower())
            op = '=' if op == ':' else op
            if field == 'rating':
                try:
                    value = int(value)
                except ValueError:
                    raise ValueError(f'rating needs a number: {token}')
            terms.append((negate, field, op, str(value).lower() if field != 'rating' else value))
        elif lowered in ('picked', 'pick', 'rejected', 'reject', 'unflagged'):
            terms.append((negate, 'flag', '=', {'picked': 'pick', 'rejected': 'reject', 'unflagged': 'none'}.get(lowered, lowered)))
        elif lowered in ('edited', 'unedited'):
            terms.append((negate != (lowered == 'unedited'), 'edited', '=', 'yes'))
        elif label_from_text(lowered):
            terms.append((negate, 'label', '=', label_from_text(lowered)))
        elif re.fullmatch(r'★{1,5}\+?', word):
            terms.append((negate, 'rating', '>=', word.count('★')))
        else:
            terms.append((negate, 'name', '=', lowered))
    return terms


def _compare(value, op, target):
    return {'=': value == target, '>': value > target, '>=': value >= target,
            '<': value < target, '<=': value <= target}[op]


def matches(terms, path, meta, edited=False):
    """True when a photo (``meta`` from ``Catalog.get``) satisfies every term."""
    p = Path(path)
    for negate, field, op, value in terms:
        if field == 'rating':
            ok = _compare(int(meta.get('rating', 0)), op, value)
        elif field == 'flag':
            ok = (meta.get('flag') or 'none') == value
        elif field == 'label':
            ok = bool(meta.get('label')) if value == 'any' else (meta.get('label') or 'none') == (label_from_text(value) or value)
        elif field == 'name':
            name = p.name.lower()
            ok = fnmatch.fnmatch(name, value) if any(c in value for c in '*?[') else value in name
        elif field == 'ext':
            ok = p.suffix.lower().lstrip('.') == value.lstrip('.')
        elif field == 'folder':
            ok = value in str(p.parent).lower()
        elif field == 'edited':
            ok = edited == (value in ('yes', 'true', '1', ''))
        else:
            ok = True
        if ok == negate:
            return False
    return True


# --------------------------------------------------------------------------- XMP

def sidecar(path):
    """``<name>.xmp`` next to the photo (Lightroom's convention)."""
    return Path(path).with_suffix('.xmp')


def _sidecars(path):
    path = Path(path)
    return [p for p in (path.with_suffix('.xmp'), path.with_name(path.name + '.xmp')) if p.is_file()]


def _exiftool():
    from .white_balance import exiftool
    command = exiftool()
    if command is None:
        raise RuntimeError('ExifTool is not available')
    return command


def _run(arguments, timeout=120):
    env = dict(os.environ, LC_ALL='C', LANG='C', LC_CTYPE='C')
    result = subprocess.run([*_exiftool(), '-config', '', *arguments], capture_output=True, timeout=timeout,
                            env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    return result


def _fields_from_tags(tags):
    fields = {}
    rating = tags.get('Rating')
    if isinstance(rating, (int, float)):
        if rating < 0:
            fields['flag'] = 'reject'
        else:
            fields['rating'] = int(max(0, min(5, round(rating))))
    label = label_from_text(tags.get('Label'))
    if label:
        fields['label'] = label
    return fields


def read_xmp(paths):
    """``{photo: fields}`` from XMP sidecars, or embedded XMP when a photo has no sidecar."""
    sources = {}
    for path in paths:
        found = _sidecars(path)
        sources[str(found[0] if found else path)] = str(path)
    if not sources:
        return {}
    result = {}
    names = list(sources)
    for start in range(0, len(names), 200):
        batch = names[start:start + 200]
        with tempfile.NamedTemporaryFile('w', suffix='.args', delete=False, encoding='utf-8') as handle:
            handle.write('\n'.join(['-j', '-charset', 'filename=UTF8', '-XMP-xmp:Rating#', '-XMP-xmp:Label', '--'] + batch))
            arguments = handle.name
        try:
            output = _run(['-charset', 'filename=UTF8', '-@', arguments])
        finally:
            os.unlink(arguments)
        try:
            records = json.loads(output.stdout.decode('utf-8') or '[]')
        except ValueError:
            records = []
        for record in records:
            source = record.get('SourceFile', '')
            photo = sources.get(source) or sources.get(str(Path(source))) or \
                next((p for s, p in sources.items() if Path(s).resolve() == Path(source).resolve()), None)
            fields = _fields_from_tags(record)
            if photo and fields:
                result[photo] = fields
    return result


def write_xmp_later(entries):
    threading.Thread(target=lambda: write_xmp(entries), name='xmp-write', daemon=True).start()


def write_xmp(entries):
    """Write ``{photo: fields}`` into sidecars (created when missing; other tags kept).
    Returns ``(written, errors)``."""
    if not entries:
        return [], []
    lines = []
    for photo, fields in entries.items():
        target = sidecar(photo)
        rating = -1 if fields.get('flag') == 'reject' else int(fields.get('rating', 0))
        label = XMP_LABELS.get(fields.get('label', ''), '')
        # Absolute paths never start with '-'; '--' would turn the next '-execute' into a file name.
        lines += ['-charset', 'filename=UTF8', '-overwrite_original', f'-XMP-xmp:Rating={rating}',
                  f'-XMP-xmp:Label={label}', str(target.resolve()), '-execute']
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


# --------------------------------------------------------------------------- Lightroom

def read_lightroom(path):
    """``{photo: fields}`` from a Lightroom Classic catalog (read-only; a copy is read when
    Lightroom holds it open).  Virtual copies are skipped."""
    path = Path(path)
    temp = None
    try:
        try:
            connection = sqlite3.connect(f'file:{path.as_posix()}?mode=ro', uri=True)
            connection.execute('SELECT 1 FROM Adobe_images LIMIT 1')
        except sqlite3.Error:
            temp = Path(tempfile.mkdtemp(prefix='lumen-lrcat-'))
            shutil.copy2(path, temp / 'catalog.lrcat')
            connection = sqlite3.connect(f'file:{(temp / "catalog.lrcat").as_posix()}?mode=ro', uri=True)
        query = '''SELECT root.absolutePath, folder.pathFromRoot, file.baseName, file.extension,
                          image.rating, image.pick, image.colorLabels
                   FROM Adobe_images image
                   JOIN AgLibraryFile file ON image.rootFile = file.id_local
                   JOIN AgLibraryFolder folder ON file.folder = folder.id_local
                   JOIN AgLibraryRootFolder root ON folder.rootFolder = root.id_local'''
        try:
            rows = connection.execute(query + ' WHERE image.masterImage IS NULL').fetchall()
        except sqlite3.Error:
            rows = connection.execute(query).fetchall()
        connection.close()
    finally:
        if temp is not None:
            shutil.rmtree(temp, ignore_errors=True)
    result = {}
    for root, folder, base, extension, rating, pick, labels in rows:
        photo = str(Path(str(root or '') + str(folder or '')) / (f'{base}.{extension}' if extension else str(base)))
        fields = {}
        if rating:
            fields['rating'] = int(max(0, min(5, round(float(rating)))))
        if pick in (1, 1.0):
            fields['flag'] = 'pick'
        elif pick in (-1, -1.0):
            fields['flag'] = 'reject'
        label = label_from_text(labels)
        if label:
            fields['label'] = label
        if fields:
            result[photo] = fields
    return result
