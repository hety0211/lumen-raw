"""The preset library (1.7.0): built-in looks and the user's own presets.

Imported presets are stored as .lumenpreset files in ``<data folder>/presets`` (``LUMEN_PRESETS``
overrides it), so they are there the next time LUMEN RAW starts and the command layer sees the
same list as the window.  ``import_path`` reads:

* LUMEN presets (.lumenpreset);
* Lightroom Classic / Camera Raw presets (.xmp) and Lightroom 1–7 templates (.lrtemplate),
  converted by ``lightroom``; what could not be carried over is kept with the preset (``notes``);
* .cube LUTs, as LUT presets (a partial look that only sets the LUT);
* .zip preset packs holding any of these (file names in GBK, as Windows zip tools write them
  for Chinese names, are read correctly).  A pack that ships both .xmp and .lrtemplate copies
  of a preset imports it once.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import zipfile
from pathlib import Path, PurePosixPath

from . import lightroom, lut, model
from .i18n import tr, N_

log = logging.getLogger(__name__)

EXTENSIONS = ('.lumenpreset', '.xmp', '.lrtemplate', '.cube')
#: Preset files of a pack: count and total size read.
MAX_MEMBERS = 4000
MAX_PACK_BYTES = 1024 * 1024 * 1024
#: One import adds at most this many presets.
MAX_IMPORT = 2000
FORMATS = {'.xmp': 'Lightroom', '.lrtemplate': 'Lightroom', '.cube': 'LUT', '.lumenpreset': 'LUMEN'}


def folder():
    override = os.environ.get('LUMEN_PRESETS')
    if override:
        return Path(override)
    from .host import data_folder
    return data_folder() / 'presets'


def subtitle(preset):
    """The second line of a preset in the list."""
    if preset.get('source') == 'builtin':
        return preset['subtitle']
    kind = {'lightroom': tr('Lightroom 预设'), 'cube': tr('LUT 预设')}.get(preset.get('source'), tr('自定义风格'))
    return f'{preset["group"]} · {kind}' if preset.get('group') else kind


def user_presets():
    """The user's presets, sorted by group and name; unreadable files are skipped."""
    result = []
    root = folder()
    if not root.is_dir():
        return result
    for path in sorted(root.glob('*.lumenpreset')):
        try:
            preset = model.load_preset(path)
        except Exception as exc:
            log.warning('preset %s skipped: %s', path.name, exc)
            continue
        preset['path'] = str(path)
        preset['subtitle'] = subtitle(preset)
        result.append(preset)
    return sorted(result, key=lambda p: (p['group'].lower(), p['name'].lower()))


def all_presets():
    return model.builtin_presets() + user_presets()


def find(name, presets=None):
    """A preset by name (exact, then ignoring case, then a subtitle containing it), or None."""
    presets = all_presets() if presets is None else presets
    text = str(name).strip()
    for test in (lambda p: p['name'] == text, lambda p: p['name'].lower() == text.lower(),
                 lambda p: text.lower() in p.get('subtitle', '').lower()):
        match = next((p for p in presets if test(p)), None)
        if match is not None:
            return match
    return None


def groups(presets):
    """Group names of the user's presets in list order."""
    seen = []
    for preset in presets:
        if preset.get('source') != 'builtin' and preset.get('group') and preset['group'] not in seen:
            seen.append(preset['group'])
    return seen


def delete(preset):
    path = preset.get('path')
    if not path or Path(path).parent.resolve() != folder().resolve():
        raise ValueError(tr('内置预设不能删除。'))
    Path(path).unlink(missing_ok=True)


def save(name, look, group='', notes=(), source='lumen'):
    """Store a preset in the library; the same preset imported again replaces its file."""
    look = model.clean_look(look)
    root = folder()
    root.mkdir(parents=True, exist_ok=True)
    content = json.dumps([name, group, look], sort_keys=True, ensure_ascii=False).encode('utf-8')
    digest = hashlib.blake2b(content, digest_size=5).hexdigest()
    stem = '-'.join(part for part in (lut.safe_name(group)[:40] if group else '', lut.safe_name(name)[:60]) if part)
    path = root / f'{stem}-{digest}.lumenpreset'
    model.write_preset(path, name, look, group, notes, source)
    return path


# --------------------------------------------------------------------------- import

def _read(name, data, group):
    """Presets in one file: a list of ``(name, group, look, notes, source)``."""
    suffix = PurePosixPath(name).suffix.lower()
    stem = PurePosixPath(name).stem
    if suffix == '.cube':
        setting, notes = lut.parse_cube(data, stem)
        return [(setting['name'] or stem, group, dict(fields=['lut'], lut=setting),
                 [(N_('LUT 说明'), n) for n in notes], 'cube')]
    if suffix == '.lumenpreset':
        if len(data) > model.PRESET_BYTES:
            raise ValueError(tr('预设文件过大。'))
        try:
            parsed = json.loads(data.decode('utf-8-sig'))
        except (ValueError, UnicodeDecodeError):
            raise ValueError(tr('请选择 Lumen 的 .lumenpreset 预设文件。'))
        preset = model.read_preset(parsed)
        return [(preset['name'], preset['group'] or group, preset['look'], preset['notes'], preset['source'])]
    if suffix == '.xmp':
        source = lightroom.read_xmp(data, stem, group)
    elif suffix == '.lrtemplate':
        source = lightroom.read_lrtemplate(data, stem, group)
    else:
        raise ValueError(tr('不支持的预设格式：{suffix}', suffix=suffix or name))
    look, notes, _ = lightroom.convert(source.settings)
    if look is None:
        raise ValueError(tr('这个预设没有 LUMEN 能套用的设置。'))
    return [(source.name, source.group or group, look, notes, 'lightroom')]


def _member_name(info):
    """A zip member's name; names without the UTF-8 flag are tried as UTF-8, GBK, Big5, Shift-JIS."""
    if info.flag_bits & 0x800:
        return info.filename
    raw = info.filename.encode('cp437', 'replace')
    for encoding in ('utf-8', 'gb18030', 'big5', 'shift_jis'):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return info.filename


def _members(path):
    """``(name, bytes)`` of the preset files in a zip pack, and the members skipped."""
    files, skipped, total = [], [], 0
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_MEMBERS * 4:
            raise ValueError(tr('预设包中的文件过多。'))
        for info in infos:
            if info.is_dir():
                continue
            name = _member_name(info)
            parts = PurePosixPath(name.replace('\\', '/')).parts
            if not parts or any(p.startswith('.') or p == '__MACOSX' for p in parts):
                continue
            suffix = PurePosixPath(name).suffix.lower()
            if suffix not in EXTENSIONS:
                if suffix in ('.dng', '.zip'):
                    skipped.append((name, tr('不支持的文件（{kind}）', kind=suffix[1:].upper())))
                continue
            limit = lut.MAX_FILE if suffix == '.cube' else model.PRESET_BYTES if suffix == '.lumenpreset' else lightroom.MAX_BYTES
            if info.file_size > limit:
                skipped.append((name, tr('文件过大')))
                continue
            total += info.file_size
            if total > MAX_PACK_BYTES or len(files) >= MAX_MEMBERS:
                raise ValueError(tr('预设包过大。'))
            with archive.open(info) as handle:
                data = handle.read(limit + 1)
            if len(data) > limit:
                skipped.append((name, tr('文件过大')))
                continue
            files.append(('/'.join(parts), data))
    return files, skipped


def import_path(path):
    """Import a preset file or a .zip pack into the library.

    Returns ``dict(imported=[...], skipped=[(name, reason), ...])``; each imported entry has
    ``name``, ``group``, ``path`` (its .lumenpreset), ``notes`` and ``source``."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.zip':
        files, skipped = _members(path)
        group = path.stem
    elif suffix in EXTENSIONS:
        limit = lut.MAX_FILE if suffix == '.cube' else model.PRESET_BYTES
        if path.stat().st_size > limit:
            raise ValueError(tr('预设文件过大。'))
        # Lightroom groups presets by folder; the folder of a single file is its group too.
        files, skipped, group = [(path.name, path.read_bytes())], [], path.resolve().parent.name
    else:
        raise ValueError(tr('不支持的预设格式：{suffix}', suffix=suffix or path.name))
    # A pack's .lrtemplate copy of an .xmp preset (for Lightroom 7.2 and older) is the same preset.
    order = sorted(files, key=lambda item: (PurePosixPath(item[0]).suffix.lower() == '.lrtemplate', item[0]))
    xmp_names, imported = set(), []
    for name, data in order:
        template = PurePosixPath(name).suffix.lower() == '.lrtemplate'
        try:
            presets = _read(name, data, group)
        except (ValueError, KeyError, TypeError, RecursionError) as exc:
            skipped.append((name, str(exc)))
            continue
        for title, preset_group, look, notes, source in presets:
            if template and title.lower() in xmp_names:
                continue
            if len(imported) >= MAX_IMPORT:
                skipped.append((name, tr('一次最多导入 {count} 个预设', count=MAX_IMPORT)))
                break
            if PurePosixPath(name).suffix.lower() == '.xmp':
                xmp_names.add(title.lower())
            saved = save(title, look, preset_group, notes, source)
            imported.append(dict(name=title, group=preset_group, path=str(saved), notes=[list(n) for n in notes],
                                 source=source))
    if not imported and len(files) == 1 and skipped:
        raise ValueError(skipped[0][1])
    return dict(imported=imported, skipped=skipped)


def report(result):
    """Interface text of an import: what was imported and what could not be carried over."""
    lines = [tr('已导入 {count} 个预设。', count=len(result['imported']))]
    with_notes = [item for item in result['imported'] if item['notes']]
    if with_notes:
        lines += ['', tr('其余设置已近似换算；以下设置在 LUMEN 中没有对应项，未能套用：')]
        for item in with_notes:
            lines.append(f'· {item["name"]}')
            lines += [f'    {line}' for line in lightroom.describe(item['notes'])]
    if result['skipped']:
        lines += ['', tr('未导入：')]
        lines += [f'· {name}：{reason}' for name, reason in result['skipped'][:200]]
    return '\n'.join(lines)
