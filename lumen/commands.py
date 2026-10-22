"""One command layer for the window, the command line, the control channel and MCP (1.6.0).

Every operation an agent or script can perform is a ``Command`` with a stable id (``edit.apply``,
``library.rate`` …), a JSON-schema parameter description and one handler.  Handlers act on a
``Context``: the open editor window (``control.WindowContext``, every edit is one undo step the
user can take back) or a headless session (``session.Session``, used by ``lumen-cli`` and by the
MCP server when the editor is not running).  The command line, the JSON-lines control channel
and the MCP server all dispatch through ``run``.

Edit values use the same vocabulary as natural-language editing (``nl_edit.plan``): final values,
checked and clipped to the ranges of the recipe, unknown keys ignored.
"""
from __future__ import annotations

import base64
import copy
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__, catalog, engine, model, nl_edit
from .i18n import tr

SUPPORTED = engine.RAW_EXTENSIONS | {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


class CommandError(Exception):
    """A command could not run; the message is meant for the caller (often an agent)."""


@dataclass(frozen=True)
class Command:
    id: str
    summary: str
    handler: Callable
    params: dict = field(default_factory=dict)
    required: tuple = ()
    read_only: bool = False
    tool: str | None = None           # MCP tool name

    def schema(self):
        return dict(type='object', properties=copy.deepcopy(self.params), required=list(self.required),
                    additionalProperties=False)


REGISTRY: dict[str, Command] = {}


def command(id, summary, params=None, required=(), read_only=False, tool=None):
    def register(handler):
        REGISTRY[id] = Command(id, summary, handler, params or {}, tuple(required), read_only, tool)
        return handler
    return register


class Context:
    """What commands act on.  ``path=None`` always means the current photo."""
    mode = 'headless'
    catalog: catalog.Catalog
    #: 2.0.0: the photo library catalog (``library_db.LibraryDB``).
    library_db = None

    def photos(self): raise NotImplementedError
    def current(self): raise NotImplementedError
    def selected(self): raise NotImplementedError
    def select(self, paths): raise NotImplementedError
    def add(self, paths): raise NotImplementedError
    def open(self, path): raise NotImplementedError
    def edits(self, path): raise NotImplementedError
    def apply(self, path, edits, label): raise NotImplementedError
    def step(self, path, delta): raise NotImplementedError
    def source(self, path): raise NotImplementedError
    def initialized(self, path): raise NotImplementedError
    def save_project(self, path, target): raise NotImplementedError
    def backend(self): raise NotImplementedError

    def notify(self, text):
        pass

    def presets_changed(self):
        """The preset library changed (1.7.0): the window reloads its list."""

    def library_changed(self):
        """The library catalog changed (2.0.0): the window shows it."""


# --------------------------------------------------------------------------- arguments

_TYPES = dict(string=str, integer=int, number=(int, float), boolean=bool, array=list, object=dict)


def _check(name, value, spec):
    kind = spec.get('type')
    if kind:
        kinds = kind if isinstance(kind, list) else [kind]
        if value is None and 'null' in kinds:
            return value
        ok = any(isinstance(value, _TYPES[k]) and not (k in ('integer', 'number') and isinstance(value, bool))
                 for k in kinds if k in _TYPES)
        if not ok and 'integer' in kinds and isinstance(value, float) and value.is_integer():
            value, ok = int(value), True
        if not ok:
            raise CommandError(f'{name} must be {" or ".join(kinds)}')
    if 'enum' in spec and value not in spec['enum']:
        raise CommandError(f'{name} must be one of {", ".join(map(str, spec["enum"]))}')
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise CommandError(f'{name} must be finite')
        if 'minimum' in spec and value < spec['minimum'] or 'maximum' in spec and value > spec['maximum']:
            raise CommandError(f'{name} must be between {spec.get("minimum")} and {spec.get("maximum")}')
    if isinstance(value, list) and 'items' in spec:
        value = [_check(f'{name}[{i}]', v, spec['items']) for i, v in enumerate(value)]
    return value


def arguments(cmd, args):
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise CommandError('arguments must be a JSON object')
    unknown = set(args) - set(cmd.params)
    if unknown:
        raise CommandError(f'unknown argument(s) for {cmd.id}: {", ".join(sorted(unknown))}')
    # null for an optional parameter means "not given" (lumen-cli passes unset options as None).
    args = {k: v for k, v in args.items() if v is not None or k in cmd.required}
    missing = [k for k in cmd.required if k not in args]
    if missing:
        raise CommandError(f'missing argument(s) for {cmd.id}: {", ".join(missing)}')
    return {k: _check(k, v, cmd.params[k]) for k, v in args.items()}


def run(context, command_id, args=None):
    """Run a command; returns a JSON-serialisable dict.  Raises CommandError."""
    cmd = REGISTRY.get(command_id)
    if cmd is None:
        raise CommandError(f'unknown command {command_id!r}; see commands.list')
    result = cmd.handler(context, **arguments(cmd, args))
    return result if isinstance(result, dict) else dict(result=result)


def describe():
    return [dict(id=c.id, summary=c.summary, read_only=c.read_only, tool=c.tool, parameters=c.schema())
            for c in sorted(REGISTRY.values(), key=lambda c: c.id)]


# --------------------------------------------------------------------------- helpers

PATH = dict(type='string', description='Photo path; omit for the current photo.')
PATHS = dict(type='array', items=dict(type='string'),
             description='Photo paths; omit for the selected photos (or the current photo).')


def _resolve(path):
    return str(Path(path).expanduser().resolve())


def _target(context, path):
    """The photo an edit command acts on, opened if needed."""
    if path:
        path = _resolve(path)
        if path != context.current():
            if not Path(path).is_file():
                raise CommandError(f'file not found: {path}')
            context.open(path)
        return path
    current = context.current()
    if not current:
        raise CommandError('no photo is open; call photo.open (open_photo) first')
    return current


def _targets(context, paths):
    if paths:
        return [_resolve(p) for p in paths]
    selected = context.selected()
    if selected:
        return selected
    current = context.current()
    if current:
        return [current]
    raise CommandError('no photos given, selected or open')


def _fresh(edits):
    """The recipe the photo had when first opened (its own white balance, develop reference, lens)."""
    fresh = model.recipe()
    for key in ('white_balance', 'develop', 'lens', 'process'):
        fresh[key] = copy.deepcopy(edits.get(key, fresh[key]))
    return fresh


def _summary(path, edits):
    return dict(path=path, process=edits.get('process', 1), state=nl_edit.current_state(edits),
                changes=nl_edit.changes(_fresh(edits), edits))


def _meta(context, path):
    return dict(path=path, name=Path(path).name, **context.catalog.get(path))


# --------------------------------------------------------------------------- app and library

@command('app.status', 'Editor state: mode (gui or headless), version, current and selected photos.',
         read_only=True, tool='lumen_status')
def app_status(context):
    current = context.current()
    result = dict(application='LUMEN RAW', version=__version__, mode=context.mode, photos=len(context.photos()),
                  current=current, selected=context.selected())
    if current:
        result['process'] = context.edits(current).get('process', 1)
    return result


@command('commands.list', 'Every command id with its parameters (JSON schema).', read_only=True, tool='list_commands')
def commands_list(context):
    return dict(commands=describe())


@command('library.list', 'Photos in the library with rating, flag and colour label, optionally filtered.',
         dict(query=dict(type='string', description='Filter, e.g. "rating>=3 -flag:reject label:red name:DSC0*". '
                                                    'Fields: rating, flag (pick/reject/none), label (red/yellow/green/blue/purple/none/any), '
                                                    'name, ext, folder, edited.'),
              limit=dict(type='integer', minimum=1, maximum=5000, description='Default 500.')),
         read_only=True, tool='list_photos')
def library_list(context, query='', limit=500):
    try:
        terms = catalog.parse_query(query)
    except ValueError as exc:
        raise CommandError(str(exc))
    current, selected = context.current(), set(context.selected())
    plain = ('rating', 'flag', 'label', 'name', 'ext', 'folder', 'edited')
    records = _records(context, context.photos()) if any(t[1] not in plain for t in terms) else {}
    photos = []
    for path in context.photos():
        meta = context.catalog.get(path)
        edited = any(t[1] == 'edited' for t in terms) and context.edits(path) != _fresh(context.edits(path))
        if catalog.matches(terms, path, meta, edited, records.get(path)):
            photos.append(dict(_meta(context, path), current=path == current, selected=path in selected))
    return dict(count=len(photos), photos=photos[:limit])


def _records(context, paths):
    """Library catalog rows (2.0.0 fields) of ``paths`` by path, with bursts and hints."""
    from . import library_db
    try:
        rows = context.library_db.rows(('paths', paths))
    except Exception:
        return {}
    library_db.annotate_bursts(rows)
    library_db.annotate_hints(rows)
    return {row['path']: row for row in rows}


@command('library.add', 'Import photos (RAW, JPEG, PNG, TIFF) or every supported photo of a folder.',
         dict(paths=dict(type='array', items=dict(type='string'), description='Files or folders.'),
              recursive=dict(type='boolean', description='Also look in sub-folders of folders.')),
         required=('paths',), tool='add_photos')
def library_add(context, paths, recursive=False):
    files = []
    for name in paths:
        p = Path(name).expanduser()
        if p.is_dir():
            pattern = '**/*' if recursive else '*'
            files += sorted(str(f.resolve()) for f in p.glob(pattern) if f.is_file() and f.suffix.lower() in SUPPORTED)
        elif p.is_file() and p.suffix.lower() in SUPPORTED:
            files.append(str(p.resolve()))
        else:
            raise CommandError(f'not a supported photo or folder: {name}')
    if len(context.photos()) + len(files) > 2000:
        raise CommandError('a library holds at most 2000 photos')
    added = context.add(files)
    return dict(added=added, count=len(context.photos()))


@command('library.select', 'Select photos (for rating, copying edits or batch export).',
         dict(paths=dict(type='array', items=dict(type='string'))), required=('paths',), tool='select_photos')
def library_select(context, paths):
    paths = [_resolve(p) for p in paths]
    missing = [p for p in paths if p not in context.photos()]
    if missing:
        raise CommandError(f'not in the library (add_photos first): {missing[0]}')
    context.select(paths)
    return dict(selected=paths)


def _rate(context, paths, **fields):
    targets = _targets(context, paths)
    try:
        changed = context.catalog.set(targets, **fields)
    except ValueError as exc:
        raise CommandError(str(exc))
    return dict(changed=changed, photos=[_meta(context, p) for p in targets])


@command('library.rate', 'Set the star rating (0–5) of photos.',
         dict(rating=dict(type='integer', minimum=0, maximum=5), paths=PATHS), required=('rating',), tool='rate_photos')
def library_rate(context, rating, paths=None):
    return _rate(context, paths, rating=rating)


@command('library.flag', 'Flag photos as pick or reject, or clear the flag (none).',
         dict(flag=dict(type='string', enum=['pick', 'reject', 'none']), paths=PATHS), required=('flag',), tool='flag_photos')
def library_flag(context, flag, paths=None):
    return _rate(context, paths, flag=flag)


@command('library.label', 'Set the colour label of photos.',
         dict(label=dict(type='string', enum=list(catalog.LABELS) + ['none']), paths=PATHS), required=('label',),
         tool='label_photos')
def library_label(context, label, paths=None):
    return _rate(context, paths, label=label)


@command('library.read_xmp', 'Read ratings and labels from XMP (sidecars or embedded) into the library.',
         dict(paths=PATHS))
def library_read_xmp(context, paths=None):
    targets = _targets(context, paths)
    values = catalog.read_xmp(targets)
    return dict(changed=context.catalog.merge(values, overwrite=True))


@command('library.write_xmp', 'Write ratings and labels into XMP sidecar files (<name>.xmp); photos are never modified.',
         dict(paths=PATHS))
def library_write_xmp(context, paths=None):
    targets = _targets(context, paths)
    written, errors = catalog.write_xmp({p: context.catalog.get(p) for p in targets})
    return dict(written=[str(catalog.sidecar(p)) for p in written], errors=errors)


@command('library.import_lightroom', 'Import ratings, pick flags and colour labels from a Lightroom Classic catalog (.lrcat).',
         dict(catalog=dict(type='string'), add=dict(type='boolean', description='Also add the photos found to the library.')),
         required=('catalog',))
def library_import_lightroom(context, catalog=None, add=False):
    from . import catalog as library
    try:
        values = library.read_lightroom(catalog)
    except Exception as exc:
        raise CommandError(f'cannot read the Lightroom catalog: {exc}')
    for photo, fields in values.items():
        context.catalog.set([photo], **{k: fields.get(k, library.empty()[k]) for k in ('rating', 'flag', 'label')})
    found = [p for p in values if Path(p).is_file()]
    added = context.add(found[:max(0, 2000 - len(context.photos()))]) if add else []
    return dict(imported=len(values), found=len(found), added=len(added))


# --------------------------------------------------------------------------- the library catalog (2.0.0)

def _catalog_photo(row):
    result = dict(path=row['path'], name=row['name'], captured=row.get('captured') or None,
                  camera=' '.join(x for x in (row.get('make'), row.get('model')) if x) or None, lens=row.get('lens') or None,
                  iso=row.get('iso'), aperture=row.get('aperture'), shutter=row.get('shutter'), focal=row.get('focal'),
                  rating=row.get('rating', 0), flag=row.get('flag') or 'none', label=row.get('label') or 'none',
                  keywords=row.get('keywords') or [], missing=bool(row.get('missing')))
    if row.get('analysis'):
        result.update(sharpness=None if row.get('sharpness') is None else round(row['sharpness'], 1), faces=row.get('faces'),
                      eyes_closed=row.get('eyes_closed'), blurry=bool(row.get('blurry')))
    if row.get('burst'):
        result['burst'] = dict(group=row['burst'], index=row['burst_index'], size=row['burst_size'], best=bool(row.get('best')))
    return result


def _catalog_source(context, folder=None, keyword=None):
    from . import library_db
    db = context.library_db
    if folder:
        wanted = library_db.key(folder)
        match = next((f for f in db.folders() if f['key'] == wanted), None)
        if match is None:
            raise CommandError(f'folder not in the library (import_folder first): {folder}')
        return ('folder', match['id'], True)
    if keyword:
        match = next((k for k in db.keywords() if k['name'].lower() == str(keyword).lower()), None)
        if match is None:
            return ('ids', [])
        return ('keyword', match['id'])
    return ('all',)


@command('catalog.search', 'Photos of the library catalog (every imported folder) with capture data, keywords, ratings, '
         'culling hints and burst groups, optionally filtered.',
         dict(query=dict(type='string', description='Filter, as library.list plus keyword (any/none/name), camera, lens, iso, f, '
                                                    'focal, date (2026-10 or date>=2026-10-01), faces, eyes:closed, sharpness '
                                                    '(0-100), burst / burst:best and the words blurry, best, missing.'),
              folder=dict(type='string', description='Only this folder and its sub-folders.'),
              keyword=dict(type='string', description='Only photos with this keyword.'),
              sort=dict(type='string', enum=['captured', 'name', 'rating', 'added']),
              limit=dict(type='integer', minimum=1, maximum=5000, description='Default 200.')),
         read_only=True, tool='search_library')
def catalog_search(context, query='', folder=None, keyword=None, sort='captured', limit=200):
    try:
        rows = context.library_db.search(context.catalog, query, _catalog_source(context, folder, keyword), sort=sort)
    except ValueError as exc:
        raise CommandError(str(exc))
    return dict(count=len(rows), photos=[_catalog_photo(r) for r in rows[:limit]])


@command('catalog.import', 'Add folders (with their sub-folders) or photos to the library catalog and read their capture '
         'data, ratings and keywords from EXIF / XMP.',
         dict(paths=dict(type='array', items=dict(type='string'), description='Folders or photo files.'),
              recursive=dict(type='boolean', description='Also sub-folders (default true).')),
         required=('paths',), tool='import_folder')
def catalog_import(context, paths, recursive=True):
    from . import library_db
    db = context.library_db
    added = 0
    for name in paths:
        p = Path(name).expanduser()
        if p.is_dir():
            added += len(db.import_folder(str(p.resolve()), recursive)[0])
        elif p.is_file() and p.suffix.lower() in library_db.extensions():
            added += len(db.add([str(p.resolve())])[0])
        else:
            raise CommandError(f'not a supported photo or folder: {name}')
    db.index_all(context.catalog)
    context.library_changed()
    return dict(added=added, photos=db.count())


@command('library.keyword', 'Add or remove keywords of photos (kept in the library catalog; written to XMP sidecars when '
         'the editor writes ratings there).',
         dict(add=dict(type='array', items=dict(type='string')), remove=dict(type='array', items=dict(type='string')),
              paths=PATHS), tool='tag_photos')
def library_keyword(context, add=None, remove=None, paths=None):
    from . import library_db
    if not add and not remove:
        raise CommandError('give keywords to add or remove')
    targets = _targets(context, paths)
    db = context.library_db
    db.add([p for p in targets if Path(p).is_file()])
    ids = db.ids_of(targets)
    if add:
        db.add_keywords(ids, library_db.clean_keywords(add))
    if remove:
        db.remove_keywords(ids, library_db.clean_keywords(remove))
    rows = db.rows(('ids', ids))
    if context.catalog.write_xmp_on_change:
        library_db.write_keywords_xmp({r['path']: r['keywords'] for r in rows})
    context.library_changed()
    return dict(photos=[dict(path=r['path'], keywords=r['keywords']) for r in rows])


@command('catalog.analyze', 'Culling hints: sharpness (0-100), faces and closed eyes per photo, and burst groups with the '
         'best frame of each (eyes open, then sharpest).  Computed from the camera previews; photos analysed before '
         'are not computed again.',
         dict(paths=PATHS, query=dict(type='string', description='Or the library photos matching this filter (catalog.search).'),
              limit=dict(type='integer', minimum=1, maximum=2000, description='Default 300.')),
         tool='analyze_photos')
def catalog_analyze(context, paths=None, query=None, limit=300):
    from . import culling, library_db, preview_cache
    db = context.library_db
    if query is not None:
        try:
            targets = [r['path'] for r in db.search(context.catalog, query)][:limit]
        except ValueError as exc:
            raise CommandError(str(exc))
    else:
        targets = _targets(context, paths)[:limit]
        db.add([p for p in targets if Path(p).is_file()])
        db.index_all(context.catalog)
    cache = getattr(getattr(context, 'window', None), 'previews', None) or preview_cache.default()
    for row in db.unanalysed(db.ids_of(targets)):
        try:
            result = culling.analyse(cache.view(row['path']))
        except Exception as exc:
            result = {}
            context.notify(f'{Path(row["path"]).name}: {exc}')
        db.set_analysis(row['id'], result)
    rows = db.rows(('paths', targets))
    library_db.annotate_bursts(rows)
    library_db.annotate_hints(rows)
    library_db.with_ratings(rows, context.catalog)
    bursts = {}
    for row in rows:
        if row.get('burst'):
            group = bursts.setdefault(row['burst'], dict(group=row['burst'], photos=[], best=None))
            group['photos'].append(row['path'])
            if row.get('best'):
                group['best'] = row['path']
    context.library_changed()
    return dict(count=len(rows), photos=[_catalog_photo(r) for r in rows], bursts=list(bursts.values()))


# --------------------------------------------------------------------------- photos and edits

@command('photo.open', 'Open a photo (or a .lumen project) in the editor; it becomes the current photo.',
         dict(path=dict(type='string')), required=('path',), tool='open_photo')
def photo_open(context, path):
    path = _resolve(path)
    if not Path(path).is_file():
        raise CommandError(f'file not found: {path}')
    context.open(path)
    current = context.current()
    _, info = context.source(current)
    return {**_info(info), **_summary(current, context.edits(current))}


def _info(info):
    photo = info.get('photo') or {}
    keep = {k: v for k, v in photo.items() if k not in ('gps',) and isinstance(v, (str, int, float))}
    return dict(name=info.get('name'), width=info.get('width'), height=info.get('height'), format=info.get('format'),
                raw=bool(info.get('raw')), note=info.get('note', ''), exif=keep)


@command('photo.info', 'Size, format and shooting data (camera, lens, exposure) of a photo.',
         dict(path=PATH), read_only=True, tool='photo_info')
def photo_info(context, path=None):
    path = _target(context, path)
    _, info = context.source(path)
    return dict(path=path, **_info(info))


@command('edit.get', 'Current edit values of a photo, in the vocabulary edit.apply accepts.',
         dict(path=PATH, full=dict(type='boolean', description='Also return the complete recipe.')),
         read_only=True, tool='get_edits')
def edit_get(context, path=None, full=False):
    path = _target(context, path)
    edits = context.edits(path)
    result = _summary(path, edits)
    if full:
        from . import lut
        result['recipe'] = {k: v for k, v in model.validate(edits).items() if k != 'masks'} | dict(
            masks=[{k: v for k, v in m.items() if k not in ('raster', 'strokes')} for m in edits['masks']],
            lut=lut.summary(edits.get('lut')))
    return result


CHANGES = dict(type='object', description=(
    'Final values (not increments); only what should change. Keys: '
    'adjustments {exposure -5…5 EV, contrast, highlights, shadows, whites, blacks, temperature (warm +), tint (magenta +), '
    'saturation, vibrance, dehaze, clarity, texture: -100…100; sharpness, denoise, color_noise: 0…100}; '
    'hsl {red|orange|yellow|green|aqua|blue|purple|magenta: {hue, saturation, luminance: -100…100}}; '
    'grading {shadows|midtones|highlights: {hue 0…360, saturation 0…100}, balance -100…100}; '
    'effects {vignette -100…100, midpoint, feather, grain, grain_size 0…100}; monochrome true/false; '
    'curve [[input, output] 0…255, 3–8 points incl. 0 and 255]; '
    'lut {path: ".cube file (1D or 3D LUT)", amount 0…100} to apply a creative LUT, {amount} to change its strength, '
    'null to remove it; '
    'masks [{region: sky|subject|person|background|foreground|water|vegetation|mountain|building|ground|top|bottom|center|edges, adjustments {…}}] '
    '(sky…ground are detected locally by AI; at most 4 new masks; existing masks: {index, adjustments}).'))


def _original(context, path):
    """The photo's original-resolution pixels for fine masks (1.7.0), decoded for this call."""
    full = getattr(context, 'original', None)
    if full is not None:
        return full(path)
    return engine.load_image(path, None)[0]


def _fine(context, path, edits, items):
    """Fine versions of quick AI masks ``[(kind, alpha)]`` (None where a kind is not refined),
    computed on the photo's original."""
    from . import selection
    if not any(kind in selection.FINE_KINDS for kind, _ in items):
        return [None] * len(items)
    guide = selection.fine_guide(_original(context, path), edits)
    return [selection.refine_fine(kind, alpha, guide) for kind, alpha in items]


def _regions(context, path, plan, fine=True):
    """Compute the AI-selected regions a plan asks for, on the photo's developed preview, and
    with ``fine`` refine their edges on the original (1.7.0)."""
    if not plan.regions:
        return nl_edit.attach_regions(plan, [])
    from . import selection
    source, _ = context.source(path)
    edits = context.edits(path)
    rgb = engine.selection_view(source, edits)
    alphas = []
    for kind, _ in plan.regions:
        try:
            alpha, _ = selection.automatic(rgb, kind, True)
            alphas.append(alpha if float((alpha > .5).mean()) >= .0003 else None)
        except Exception as exc:
            plan.notes.append(f'{kind}: {exc}')
            alphas.append(None)
    if fine and any(a is not None for a in alphas):
        try:
            context.notify('fine mask edges')
            present = [(i, kind, a) for i, ((kind, _), a) in enumerate(zip(plan.regions, alphas)) if a is not None]
            for (i, _, _), better in zip(present, _fine(context, path, edits, [(k, a) for _, k, a in present])):
                if better is not None:
                    alphas[i] = better
        except Exception as exc:
            plan.notes.append(f'fine edges: {exc}')
    return nl_edit.attach_regions(plan, [selection.encode(a) if a is not None else None for a in alphas])


@command('edit.apply', 'Apply edit values to a photo as one undoable step (same vocabulary as edit.get).',
         dict(changes=CHANGES, path=PATH, reset=dict(type='boolean', description='Reset the global look first.'),
              amount=dict(type='number', minimum=0, maximum=150, description='Strength in percent of the change (default 100).'),
              label=dict(type='string', description='Short description shown in the editor.'),
              fine=dict(type='boolean', description='Refine the edges of new AI region masks (sky, subject, person, '
                                                    'background) on the original photo: hair, fur, branches. A few '
                                                    'seconds slower. Default true.')),
         required=('changes',), tool='apply_edits')
def edit_apply(context, changes, path=None, reset=False, amount=100, label='', fine=True):
    path = _target(context, path)
    before = context.edits(path)
    try:
        plan = nl_edit.plan(before, dict(changes, reset=bool(reset)), files=True)
    except nl_edit.ServiceError as exc:
        raise CommandError(str(exc))
    plan = _regions(context, path, plan, fine)
    target = nl_edit.blend(plan.base, plan.target, amount / 100) if amount != 100 else plan.target
    target = model.validate(target)
    lines = nl_edit.changes(before, target)
    context.apply(path, target, label or '; '.join(lines[:3]) or 'edit.apply')
    return dict(path=path, applied=lines, notes=plan.notes, state=nl_edit.current_state(target))


@command('mask.refine', 'Refine the edges of a photo\'s AI masks (sky, subject, person, background) on the original '
                       'photo, e.g. masks made before 1.7.0 or with fine edges off. One undoable step.',
         dict(path=PATH, index=dict(type='integer', minimum=0, maximum=31,
                                    description='Mask number (see edit.get); omit for every AI mask.')),
         tool='refine_masks')
def mask_refine(context, path=None, index=None):
    from . import selection
    path = _target(context, path)
    edits = context.edits(path)
    masks = edits['masks']
    if index is not None and index >= len(masks):
        raise CommandError(f'no mask {index}; the photo has {len(masks)}')
    chosen = [i for i in ([index] if index is not None else range(len(masks)))
              if masks[i]['kind'] in selection.FINE_KINDS and masks[i].get('raster')]
    if not chosen:
        raise CommandError('no AI mask (sky, subject, person, background) to refine')
    context.notify('fine mask edges')
    refined = _fine(context, path, edits, [(masks[i]['kind'], selection.decode(masks[i]['raster'])) for i in chosen])
    for i, alpha in zip(chosen, refined):
        masks[i]['raster'] = selection.encode(alpha)
    edits = model.validate(edits)
    context.apply(path, edits, 'mask.refine')
    return dict(path=path, refined=[dict(index=i, name=masks[i]['name'], size=list(selection.decode(masks[i]['raster']).shape[1::-1]))
                                    for i in chosen])


@command('edit.reset','Reset every edit of a photo (keeps its camera white balance and develop reference).',
         dict(path=PATH), tool='reset_edits')
def edit_reset(context, path=None):
    path = _target(context, path)
    before = context.edits(path)
    fresh = model.recipe()
    for key in ('white_balance', 'develop', 'lens'):
        fresh[key] = copy.deepcopy(before[key])
    fresh['lens']['enabled'] = before['lens'].get('enabled', False)
    context.apply(path, fresh, 'edit.reset')
    return _summary(path, fresh)


@command('edit.undo', 'Undo the last edit step of a photo.', dict(path=PATH), tool='undo')
def edit_undo(context, path=None):
    path = _target(context, path)
    return _summary(path, context.step(path, -1))


@command('edit.redo', 'Redo an undone edit step.', dict(path=PATH), tool='redo')
def edit_redo(context, path=None):
    path = _target(context, path)
    return _summary(path, context.step(path, 1))


@command('edit.auto', 'Automatic exposure and tone for a photo.', dict(path=PATH), tool='auto_tone')
def edit_auto(context, path=None):
    path = _target(context, path)
    source, _ = context.source(path)
    edits = context.edits(path)
    edits['adjustments'].update(engine.auto_tone_for(source, edits))
    context.apply(path, model.validate(edits), 'edit.auto')
    return _summary(path, edits)


@command('edit.presets', 'Presets for edit.preset: the built-in looks and the presets the user imported '
                        '(Lightroom .xmp / .lrtemplate, LUTs, LUMEN presets).', read_only=True, tool='list_presets')
def edit_presets(context):
    from . import presets
    return dict(presets=[dict(name=p['name'], description=p['subtitle'], group=p.get('group', ''),
                              kind='built-in' if p.get('source') == 'builtin' else p.get('source', 'lumen'),
                              notes=presets.lightroom.describe(p.get('notes', []))) for p in presets.all_presets()])


@command('edit.preset', 'Apply a preset (see edit.presets) at a strength to a photo, or to several (paths) as a batch.',
         dict(name=dict(type='string'), amount=dict(type='number', minimum=0, maximum=100), path=PATH,
              paths=dict(type='array', items=dict(type='string'), description='Apply to these photos instead of one.')),
         required=('name',), tool='apply_preset')
def edit_preset(context, name, amount=100, path=None, paths=None):
    from . import presets
    preset = presets.find(name)
    if preset is None:
        raise CommandError(f'unknown preset {name!r}; see edit.presets')
    if paths:
        done = []
        for other in [_resolve(p) for p in paths]:
            if not Path(other).is_file():
                raise CommandError(f'file not found: {other}')
            if other not in context.photos():
                context.add([other])
            edits = model.apply_look(context.edits(other), preset['look'], amount)
            context.apply(other, model.validate(edits), f'edit.preset {preset["name"]}')
            done.append(other)
        return dict(preset=preset['name'], applied=done)
    path = _target(context, path)
    edits = model.apply_look(context.edits(path), preset['look'], amount)
    context.apply(path, model.validate(edits), f'edit.preset {preset["name"]}')
    return _summary(path, edits)


@command('preset.import', 'Import presets into the preset list: Lightroom Classic / Camera Raw .xmp, Lightroom '
                          '.lrtemplate, .cube LUTs, LUMEN .lumenpreset, or a .zip pack of them. Lightroom settings are '
                          'converted approximately; the result lists what could not be carried over.',
         dict(path=dict(type='string', description='Preset file or .zip pack.')), required=('path',), tool='import_presets')
def preset_import(context, path):
    from . import presets
    source = Path(path).expanduser()
    if not source.is_file():
        raise CommandError(f'file not found: {source}')
    try:
        result = presets.import_path(source)
    except (OSError, ValueError) as exc:
        raise CommandError(str(exc))
    context.notify(f'imported {len(result["imported"])} presets')
    context.presets_changed()
    return dict(imported=[dict(name=i['name'], group=i['group'], kind=i['source'],
                               not_carried_over=presets.lightroom.describe(i['notes'])) for i in result['imported']],
                skipped=[dict(file=n, reason=r) for n, r in result['skipped']])


@command('edit.copy', 'Copy the look (tone, colour, effects, process version, lens switches) of a photo to others; '
                      'crops, masks and retouching stay per photo.',
         dict(to=dict(type='array', items=dict(type='string')), path=PATH), required=('to',), tool='copy_edits')
def edit_copy(context, to, path=None):
    path = _target(context, path)
    source = context.edits(path)
    look = model.copy_look(source)
    done = []
    for other in [_resolve(p) for p in to]:
        if other == path:
            continue
        if other not in context.photos():
            context.add([other])
        edits = model.apply_look(context.edits(other), look)
        edits['process'] = source.get('process', 1)
        settings = copy.deepcopy(edits.get('lens') or {})
        settings.update({k: source['lens'][k] for k in ('enabled', 'distortion', 'vignetting', 'chromatic') if k in source['lens']})
        edits['lens'] = settings
        context.apply(other, model.validate(edits), f'edit.copy from {Path(path).name}')
        done.append(other)
    return dict(source=path, copied=done)


@command('edit.process', 'Set the process version: 2 renders scene-referred (1.6.0), 1 as LUMEN RAW 1.x.',
         dict(version=dict(type='integer', enum=[1, 2]), path=PATH), required=('version',), tool='set_process_version')
def edit_process(context, version, path=None):
    path = _target(context, path)
    edits = context.edits(path)
    edits['process'] = version
    context.apply(path, edits, f'edit.process {version}')
    return dict(path=path, process=version)


# --------------------------------------------------------------------------- output

def preview_image(context, path, size=1024, original=False):
    source, _ = context.source(path)
    edits = context.edits(path)
    if original:
        rgb = engine.develop_view(source, edits)
    else:
        rgb = engine.crop_rotate(engine.process(source, edits, context.backend(), apply_crop=False), edits)
    return engine.resize_limit(np.ascontiguousarray(rgb), size), edits


def jpeg(rgb, quality=88):
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(buffer, 'JPEG', quality=quality)
    return buffer.getvalue()


@command('render.preview', 'Render the edited photo (or its unedited original) as a JPEG image to look at.',
         dict(path=PATH, size=dict(type='integer', minimum=128, maximum=2048, description='Long edge, default 1024.'),
              original=dict(type='boolean', description='The developed original without edits.'),
              output=dict(type='string', description='Also save the JPEG to this file.')),
         read_only=True, tool='render_preview')
def render_preview(context, path=None, size=1024, original=False, output=None):
    path = _target(context, path)
    rgb, edits = preview_image(context, path, size, original)
    data = jpeg(rgb)
    if output:
        Path(output).expanduser().write_bytes(data)
    stats = nl_edit.image_statistics(rgb)
    return dict(path=path, width=rgb.shape[1], height=rgb.shape[0], mime='image/jpeg',
                image=base64.b64encode(data).decode('ascii'), statistics=stats, output=output)


FORMATS = {'.jpg': '.jpg', '.jpeg': '.jpg', '.png': '.png', '.tif': '.tif', '.tiff': '.tif', '.dng': '.dng'}
#: 1.6.1 (GitHub issue #5): exports carry the original's metadata unless asked not to.
METADATA = dict(type='boolean', description="Copy the original's EXIF, IPTC and XMP (camera, lens, date, "
                "exposure, copyright, keywords, rating) into the file. Default true.")
LOCATION = dict(type='boolean', description='With metadata: include the GPS position. Default true.')
SPACE = dict(type='string', enum=['srgb', 'p3', 'adobe-rgb'],
             description='Output colour space: srgb (default), p3 (Display P3) or adobe-rgb (Adobe RGB 1998).')
MAX_KB = dict(type='integer', minimum=10, maximum=500000,
              description='JPEG only: keep the file under this many KB (metadata included) by lowering the quality.')


@command('photo.export', 'Export the edited photo at full resolution (JPEG, PNG, 16-bit TIFF or linear DNG).',
         dict(output=dict(type='string', description='Output file; the extension picks the format.'), path=PATH,
              quality=dict(type='integer', minimum=50, maximum=100), long_edge=dict(type='integer', minimum=16, maximum=20000),
              overwrite=dict(type='boolean', description='Replace an existing output file (never the original).'),
              metadata=METADATA, location=LOCATION, space=SPACE, max_kb=MAX_KB),
         required=('output',), tool='export_photo')
def photo_export(context, output, path=None, quality=92, long_edge=None, overwrite=False, metadata=True, location=True,
                 space='srgb', max_kb=None):
    from . import batch_export
    path = _target(context, path)
    output = Path(output).expanduser().resolve()
    if output.suffix.lower() not in FORMATS:
        raise CommandError('output must end in .jpg, .png, .tif or .dng')
    if output == Path(path).resolve():
        raise CommandError('choose a different file name; the original is never overwritten')
    if not output.parent.is_dir():
        raise CommandError(f'folder not found: {output.parent}')
    if output.exists() and not overwrite:
        raise CommandError(f'{output} exists; pass overwrite=true to replace it')
    context.notify(f'export {output.name}')
    edits = context.edits(path)
    results = batch_export.run([(path, edits, context.initialized(path))],
                               dict(extension=FORMATS[output.suffix.lower()], long_edge=long_edge, quality=quality,
                                    folder=str(output.parent), names={path: str(output)}, metadata=metadata,
                                    location=location, space=space, max_kb=max_kb),
                               context.backend(), lambda *_: None, _Never())
    _, target, error = results[0]
    if not target:
        raise CommandError(error)
    result = dict(path=path, output=target, bytes=Path(target).stat().st_size)
    result.update(_notes(error))
    return result


def _notes(text):
    """``note`` for any export note, plus ``metadata_error`` (the 1.6.1 field) when the original's
    metadata could not be written."""
    if not text:
        return {}
    prefix = tr('未能写入原片元数据：')
    metadata = [line[len(prefix):] for line in text.splitlines() if line.startswith(prefix)]
    return dict(note=text, **({'metadata_error': metadata[0]} if metadata else {}))


class _Never:
    def is_set(self):
        return False


@command('photo.export_batch', 'Export several photos with their own edits into a folder (names <photo>-Lumen.<ext>, never overwriting).',
         dict(folder=dict(type='string'), paths=PATHS, query=dict(type='string', description='Library filter instead of paths.'),
              format=dict(type='string', enum=['jpg', 'png', 'tif', 'dng']), quality=dict(type='integer', minimum=50, maximum=100),
              long_edge=dict(type='integer', minimum=16, maximum=20000), metadata=METADATA, location=LOCATION,
              space=SPACE, max_kb=MAX_KB),
         required=('folder',), tool='export_photos')
def photo_export_batch(context, folder, paths=None, query=None, format='jpg', quality=92, long_edge=None,
                       metadata=True, location=True, space='srgb', max_kb=None):
    from . import batch_export
    if not Path(folder).expanduser().is_dir():
        raise CommandError(f'folder not found: {folder}')
    if query is not None:
        targets = [p['path'] for p in library_list(context, query, 5000)['photos']]
    else:
        targets = _targets(context, paths)
    if not targets:
        raise CommandError('no photos to export')
    context.notify(f'export {len(targets)} photos')
    items = [(p, context.edits(p), context.initialized(p)) for p in targets]
    results = batch_export.run(items, dict(extension='.' + format, long_edge=long_edge, quality=quality,
                                           folder=str(Path(folder).expanduser().resolve()), metadata=metadata,
                                           location=location, space=space, max_kb=max_kb),
                               context.backend(), lambda *_: None, _Never())
    return dict(exported=[dict(path=p, output=t, **_notes(e)) for p, t, e in results if t],
                failed=[dict(path=p, error=e) for p, t, e in results if not t])


@command('video.timelapse', 'Make a time-lapse video of photos in capture order (interval shots).  Keyframes carry their '
         'edits; the photos between two keyframes blend smoothly between them (exposure, white balance, colour, crop).',
         dict(output=dict(type='string', description='Video file (.mp4, or .mov for ProRes).'),
              paths=PATHS, query=dict(type='string', description='Library catalog filter instead of paths (catalog.search).'),
              keyframes=dict(type='array', items=dict(type='string'),
                             description='Photos whose edits are keyframes; default: first, last and every edited photo.'),
              format=dict(type='string', enum=['h264', 'hevc', 'av1', 'prores']),
              resolution=dict(type='string', enum=['original', '4k', '2k', 'fhd']),
              aspect=dict(type='string', enum=['original', '16:9', '4:3', '1:1', '9:16']),
              fps=dict(type='integer', enum=[24, 25, 30, 50, 60]),
              frames_per_photo=dict(type='integer', minimum=1, maximum=10),
              max_mbps=dict(type='integer', minimum=2, maximum=800, description='Maximum bit rate; default by size.'),
              ten_bit=dict(type='boolean'), hardware=dict(type='boolean', description='Graphics card encoder first (default).'),
              deflicker=dict(type='string', enum=['off', 'light', 'medium', 'strong'])),
         required=('output',), tool='make_timelapse')
def video_timelapse(context, output, paths=None, query=None, keyframes=None, format='h264', resolution='4k',
                    aspect='original', fps=30, frames_per_photo=1, max_mbps=None, ten_bit=False, hardware=True,
                    deflicker='light'):
    from . import timelapse
    target = Path(output).expanduser().resolve()
    if not target.parent.is_dir():
        raise CommandError(f'folder not found: {target.parent}')
    if target.suffix.lower() != timelapse.FORMATS[format]['suffix']:
        raise CommandError(f'{format} is written as {timelapse.FORMATS[format]["suffix"]}')
    if query is not None:
        try:
            targets = [r['path'] for r in context.library_db.search(context.catalog, query)]
        except ValueError as exc:
            raise CommandError(str(exc))
    else:
        targets = _targets(context, paths)
    targets = [p for p in targets if Path(p).is_file()]
    if len(targets) < 2:
        raise CommandError('a time-lapse needs at least 2 photos')
    context.library_db.add(targets)
    records = {r['path']: r for r in context.library_db.rows(('paths', targets))}
    ordered = timelapse.ordered(targets, records)
    known = set(context.photos())
    frames = []
    for path in ordered:
        edits = context.edits(path) if path in known else None
        frames.append(dict(path=path, edits=edits, initialized=bool(edits) and context.initialized(path),
                           edited=bool(edits) and edits != _fresh(edits)))
    marks = None
    if keyframes:
        wanted = {_resolve(p) for p in keyframes}
        marks = [i for i, frame in enumerate(frames) if frame['path'] in wanted]
        if not marks:
            raise CommandError('none of the keyframes is in the sequence')
    context.notify(f'time-lapse of {len(frames)} photos')
    result = timelapse.run(frames, dict(output=str(target), format=format, resolution=resolution, aspect=aspect, fps=fps,
                                        repeat=frames_per_photo, mbps=max_mbps, ten_bit=ten_bit, hardware=hardware,
                                        deflicker=deflicker, keyframes=marks), context.backend())
    result['keyframes'] = [frames[i]['path'] for i in result['keyframes']]
    result['bytes'] = Path(result['output']).stat().st_size
    return result


@command('project.save', 'Save a photo\'s edits as a .lumen project (default: next to the photo).',
         dict(path=PATH, output=dict(type='string')), tool='save_project')
def project_save(context, path=None, output=None):
    path = _target(context, path)
    target = Path(output).expanduser().resolve() if output else Path(path).with_suffix('.lumen')
    if target.suffix.lower() != '.lumen':
        target = target.with_suffix('.lumen')
    return dict(path=path, project=context.save_project(path, str(target)))
