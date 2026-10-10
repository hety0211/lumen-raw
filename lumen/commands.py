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
    photos = []
    for path in context.photos():
        meta = context.catalog.get(path)
        edited = any(t[1] == 'edited' for t in terms) and context.edits(path) != _fresh(context.edits(path))
        if catalog.matches(terms, path, meta, edited):
            photos.append(dict(_meta(context, path), current=path == current, selected=path in selected))
    return dict(count=len(photos), photos=photos[:limit])


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
        result['recipe'] = {k: v for k, v in model.validate(edits).items() if k != 'masks'} | dict(
            masks=[{k: v for k, v in m.items() if k not in ('raster', 'strokes')} for m in edits['masks']])
    return result


CHANGES = dict(type='object', description=(
    'Final values (not increments); only what should change. Keys: '
    'adjustments {exposure -5…5 EV, contrast, highlights, shadows, whites, blacks, temperature (warm +), tint (magenta +), '
    'saturation, vibrance, dehaze, clarity, texture: -100…100; sharpness, denoise, color_noise: 0…100}; '
    'hsl {red|orange|yellow|green|aqua|blue|purple|magenta: {hue, saturation, luminance: -100…100}}; '
    'grading {shadows|midtones|highlights: {hue 0…360, saturation 0…100}, balance -100…100}; '
    'effects {vignette -100…100, midpoint, feather, grain, grain_size 0…100}; monochrome true/false; '
    'curve [[input, output] 0…255, 3–8 points incl. 0 and 255]; '
    'masks [{region: sky|subject|person|background|foreground|top|bottom|center|edges, adjustments {…}}] '
    '(sky…foreground are detected locally by AI; at most 4 new masks; existing masks: {index, adjustments}).'))


def _regions(context, path, plan):
    """Compute the AI-selected regions a plan asks for, on the photo's developed preview."""
    if not plan.regions:
        return nl_edit.attach_regions(plan, [])
    from . import lens, selection
    source, _ = context.source(path)
    edits = context.edits(path)
    rgb = engine.develop_view(lens.correct(source, edits), edits)
    alphas = []
    for kind, _ in plan.regions:
        try:
            alpha, _ = selection.automatic(rgb, kind, True)
            alphas.append(selection.encode(alpha) if float((alpha > .5).mean()) >= .0003 else None)
        except Exception as exc:
            plan.notes.append(f'{kind}: {exc}')
            alphas.append(None)
    return nl_edit.attach_regions(plan, alphas)


@command('edit.apply', 'Apply edit values to a photo as one undoable step (same vocabulary as edit.get).',
         dict(changes=CHANGES, path=PATH, reset=dict(type='boolean', description='Reset the global look first.'),
              amount=dict(type='number', minimum=0, maximum=150, description='Strength in percent of the change (default 100).'),
              label=dict(type='string', description='Short description shown in the editor.')),
         required=('changes',), tool='apply_edits')
def edit_apply(context, changes, path=None, reset=False, amount=100, label=''):
    path = _target(context, path)
    before = context.edits(path)
    try:
        plan = nl_edit.plan(before, dict(changes, reset=bool(reset)))
    except nl_edit.ServiceError as exc:
        raise CommandError(str(exc))
    plan = _regions(context, path, plan)
    target = nl_edit.blend(plan.base, plan.target, amount / 100) if amount != 100 else plan.target
    target = model.validate(target)
    lines = nl_edit.changes(before, target)
    context.apply(path, target, label or '; '.join(lines[:3]) or 'edit.apply')
    return dict(path=path, applied=lines, notes=plan.notes, state=nl_edit.current_state(target))


@command('edit.reset', 'Reset every edit of a photo (keeps its camera white balance and develop reference).',
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


@command('edit.presets', 'Names of the built-in looks for edit.preset.', read_only=True, tool='list_presets')
def edit_presets(context):
    return dict(presets=[dict(name=p['name'], description=p['subtitle']) for p in model.builtin_presets()])


@command('edit.preset', 'Apply a built-in look (see edit.presets) at a strength.',
         dict(name=dict(type='string'), amount=dict(type='number', minimum=0, maximum=100), path=PATH),
         required=('name',), tool='apply_preset')
def edit_preset(context, name, amount=100, path=None):
    presets = {p['name']: p for p in model.builtin_presets()}
    preset = presets.get(name) or next((p for p in presets.values() if name.lower() in p['subtitle'].lower()), None)
    if preset is None:
        raise CommandError(f'unknown preset {name!r}; see edit.presets')
    path = _target(context, path)
    edits = model.apply_look(context.edits(path), preset['look'], amount)
    context.apply(path, model.validate(edits), f'edit.preset {preset["name"]}')
    return _summary(path, edits)


@command('edit.copy', 'Copy the look (tone, colour, effects, process version, lens switches) of a photo to others; '
                      'crops, masks and retouching stay per photo.',
         dict(to=dict(type='array', items=dict(type='string')), path=PATH), required=('to',), tool='copy_edits')
def edit_copy(context, to, path=None):
    path = _target(context, path)
    source = context.edits(path)
    look = model.extract_look(source)
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


@command('photo.export', 'Export the edited photo at full resolution (JPEG, PNG, 16-bit TIFF or linear DNG).',
         dict(output=dict(type='string', description='Output file; the extension picks the format.'), path=PATH,
              quality=dict(type='integer', minimum=50, maximum=100), long_edge=dict(type='integer', minimum=16, maximum=20000),
              overwrite=dict(type='boolean', description='Replace an existing output file (never the original).')),
         required=('output',), tool='export_photo')
def photo_export(context, output, path=None, quality=92, long_edge=None, overwrite=False):
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
                                    folder=str(output.parent), names={path: str(output)}),
                               context.backend(), lambda *_: None, _Never())
    _, target, error = results[0]
    if error:
        raise CommandError(error)
    return dict(path=path, output=target)


class _Never:
    def is_set(self):
        return False


@command('photo.export_batch', 'Export several photos with their own edits into a folder (names <photo>-Lumen.<ext>, never overwriting).',
         dict(folder=dict(type='string'), paths=PATHS, query=dict(type='string', description='Library filter instead of paths.'),
              format=dict(type='string', enum=['jpg', 'png', 'tif', 'dng']), quality=dict(type='integer', minimum=50, maximum=100),
              long_edge=dict(type='integer', minimum=16, maximum=20000)),
         required=('folder',), tool='export_photos')
def photo_export_batch(context, folder, paths=None, query=None, format='jpg', quality=92, long_edge=None):
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
                                           folder=str(Path(folder).expanduser().resolve())),
                               context.backend(), lambda *_: None, _Never())
    return dict(exported=[dict(path=p, output=t) for p, t, e in results if t],
                failed=[dict(path=p, error=e) for p, t, e in results if not t])


@command('project.save', 'Save a photo\'s edits as a .lumen project (default: next to the photo).',
         dict(path=PATH, output=dict(type='string')), tool='save_project')
def project_save(context, path=None, output=None):
    path = _target(context, path)
    target = Path(output).expanduser().resolve() if output else Path(path).with_suffix('.lumen')
    if target.suffix.lower() != '.lumen':
        target = target.with_suffix('.lumen')
    return dict(path=path, project=context.save_project(path, str(target)))
