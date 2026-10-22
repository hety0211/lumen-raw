"""Versioned, portable, non-destructive edit recipes."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from .i18n import tr, N_

COLORS = [(tr('红'), 0, '#ec6c75'), (tr('橙'), 30, '#e7a06c'), (tr('黄'), 60, '#ddd17a'),
          (tr('绿'), 120, '#83c697'), (tr('青'), 180, '#71c8c1'), (tr('蓝'), 240, '#789fea'),
          (tr('紫'), 280, '#b091db'), (tr('洋红'), 320, '#d887bb')]


def adjustments():
    return dict(exposure=0., temperature=0., tint=0., contrast=0., shadows=0.,
                highlights=0., blacks=0., whites=0., saturation=0., vibrance=0.,
                dehaze=0., clarity=0., texture=0., sharpness=0., denoise=0., color_noise=0.)


def grading():
    return dict(shadows=[220., 0.], midtones=[30., 0.], highlights=[45., 0.], balance=0.)


def effects():
    return dict(vignette=0., midpoint=50., feather=70., grain=0., grain_size=30.)


#: Recipe format written by this version; 6 adds ``process`` (1.6.0), 7 ``lut`` (1.7.0), 8 AI
#: removal in ``retouch`` and the landscape and object masks (2.1.0).
VERSION = 8
#: Process version of new photos: 2 is the scene-referred pipeline of 1.6.0 (see engine).
PROCESS = 2


def recipe():
    from .watermark import defaults
    return dict(version=VERSION, process=PROCESS, watermark=defaults(), develop=dict(mode='linear', curve=[[0.,0.],[1.,1.]], source=N_('线性起点')), adjustments=adjustments(), curve_mode='smooth',
                hsl=[[0., 0., 0.] for _ in COLORS],
                curves={c: [[0., 0.], [1., 1.]] for c in ['RGB', 'R', 'G', 'B']},
                tone_curve=[[0., 0.], [1., 1.]],
                masks=[], crop=None, rotation=0, straighten=0., monochrome=False,
                wb_gain=[1., 1., 1.], grading=grading(), effects=effects(), retouch=[],
                white_balance=dict(camera_kelvin=None, kelvin=None, estimated=False), lens=lens_defaults(),
                lut=None)


def lens_defaults():
    from .lens import defaults
    return defaults()


#: 2.1.0: landscape masks, one network for all five (lumen/selection.py), and picked objects.
LANDSCAPE_KINDS = ('water', 'vegetation', 'mountain', 'building', 'ground')
#: Masks stored as a raster (AI selections and colour picks); ``object`` also keeps its clicks.
RASTER_KINDS = ('sky', 'person', 'background', 'color', 'subject', 'foreground') + LANDSCAPE_KINDS + ('object',)
#: Clicks of an object mask: ``[x, y, label]`` (1 adds, 0 removes).
OBJECT_POINTS = 32


def new_mask(kind, index):
    names = dict(brush=tr('画笔'), linear=tr('线性渐变'), radial=tr('径向渐变'), luminance=tr('亮度范围'), sky=tr('天空'),
                 person=tr('人物'), background=tr('背景'), color=tr('相似颜色'), subject=tr('主体'), foreground=tr('近景'),
                 water=tr('水面'), vegetation=tr('植被'), mountain=tr('山体'), building=tr('建筑'), ground=tr('地面'),
                 object=tr('物体'))
    return dict(name=f'{names[kind]} {index}',
                kind=kind, enabled=True, invert=False, opacity=100., feather=50.,
                start=[.25, .25], end=[.75, .75], strokes=[], adjustments=adjustments(),
                luminance_range=[50., 100.], range_falloff=20.)


def validate(data):
    """Reject malformed projects before they enter either the UI or renderer."""
    import math
    def number(value, low, high):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(tr('工程中的数值超出支持范围。'))
        return float(value)
    def point(p):
        if not isinstance(p, list) or len(p) != 2:
            raise ValueError(tr('无效坐标。'))
        return [number(v, 0, 1) for v in p]
    def adj(a):
        base = adjustments()
        for k in base:
            base[k] = number(a.get(k, 0), -5 if k == 'exposure' else -100,
                             5 if k == 'exposure' else 100)
        return base
    if not isinstance(data, dict) or data.get('version') not in (1, 2, 3, 4, 5, 6, 7, 8):
        raise ValueError(tr('不支持的工程版本。'))
    r = recipe()
    # Projects before 1.6.0 have no process version: they keep rendering as they always did.
    r['process'] = data.get('process', 1)
    if r['process'] not in (1, 2) or isinstance(r['process'], bool):
        raise ValueError(tr('不支持的处理版本。'))
    from .watermark import validate as validate_watermark
    r['watermark'] = validate_watermark(data.get('watermark', r['watermark']))
    profile = data.get('develop', r['develop'])
    if profile.get('mode') not in ('camera', 'linear'):
        raise ValueError(tr('不支持的显影起点。'))
    curve = [point(p) for p in profile.get('curve', [[0,0],[1,1]])]
    if not 2 <= len(curve) <= 32 or curve[0] != [0,0] or curve[-1] != [1,1] or any(b[0] <= a[0] or b[1] < a[1] for a,b in zip(curve,curve[1:])):
        raise ValueError(tr('无效的默认显影曲线。'))
    r['develop'] = dict(mode=profile['mode'],curve=curve,source=str(profile.get('source',''))[:80])
    r['adjustments'] = adj(data.get('adjustments', {}))
    hsl = data.get('hsl', r['hsl'])
    if len(hsl) != 8 or any(len(row) != 3 for row in hsl):
        raise ValueError(tr('无效 HSL 数据。'))
    r['hsl'] = [[number(v, -100, 100) for v in row] for row in hsl]
    for c in r['curves']:
        points = sorted([point(p) for p in data.get('curves', r['curves'])[c]])
        if not 2 <= len(points) <= 256 or points[0][0] != 0 or points[-1][0] != 1 or any(b[0] - a[0] < .001 for a, b in zip(points, points[1:])):
            raise ValueError(tr('曲线必须含有 0、1 端点，且横坐标不能重复。'))
        r['curves'][c] = points
    # 1.4.1: the exposure curve's fine tone curve (absent in older projects: identity).
    tone = [point(p) for p in data.get('tone_curve', r['tone_curve'])]
    if not 2 <= len(tone) <= 256 or tone[0][0] != 0 or tone[-1][0] != 1 or             any(b[0] <= a[0] or b[1] < a[1] for a, b in zip(tone, tone[1:])):
        raise ValueError(tr('无效的曝光曲线。'))
    r['tone_curve'] = tone
    r['curve_mode'] = data.get('curve_mode', 'linear' if data['version'] < 3 else 'smooth')
    if r['curve_mode'] not in ('linear', 'smooth'):
        raise ValueError(tr('不支持的曲线模式。'))
    wb = data.get('white_balance', {})
    for k in ('camera_kelvin', 'kelvin'):
        r['white_balance'][k] = None if wb.get(k) is None else number(wb[k], 1500, 25000)
    r['white_balance']['estimated'] = bool(wb.get('estimated', False))
    if (r['white_balance']['camera_kelvin'] is None) != (r['white_balance']['kelvin'] is None):
        raise ValueError(tr('色温需要相机基准值。'))
    ops = data.get('retouch', [])
    if not isinstance(ops, list) or len(ops) > 500:
        raise ValueError(tr('最多支持 500 个修复笔划。'))
    for op in ops:
        if op.get('kind') not in ('heal', 'clone', 'remove') or not 1 <= len(op.get('points', [])) <= 50000:
            raise ValueError(tr('无效修复笔划。'))
        item = dict(kind=op['kind'], enabled=bool(op.get('enabled', True)),
                    points=[point(p) for p in op['points']], radius=number(op['radius'], .001, .1),
                    opacity=number(op.get('opacity', 100), 0, 100), feather=number(op.get('feather', 50), 0, 100))
        if op['kind'] == 'remove':
            # 2.1.0: the LaMa fill computed when the removal was made (lumen/remove.py).
            from . import remove
            item['patch'] = remove.validate(op.get('patch'))
            item['picked'] = bool(op.get('picked', False))
        if op['kind'] == 'clone':
            offset = op['offset']
            if not isinstance(offset, list) or len(offset) != 2:
                raise ValueError(tr('无效仿制取样偏移。'))
            item['offset'] = [number(v, -1, 1) for v in offset]
        r['retouch'].append(item)
    crop = data.get('crop')
    if crop is not None:
        if len(crop) != 4:
            raise ValueError(tr('无效裁切。'))
        crop = [number(v, 0, 1) for v in crop]
        if crop[2] - crop[0] < .001 or crop[3] - crop[1] < .001:
            raise ValueError(tr('裁切区域太小。'))
    r['crop'] = crop
    r['rotation'] = int(number(data.get('rotation', 0), 0, 3))
    r['straighten'] = number(data.get('straighten', 0), -15, 15)
    r['monochrome'] = bool(data.get('monochrome', False))
    gains = data.get('wb_gain', [1., 1., 1.])
    if not isinstance(gains, list) or len(gains) != 3:
        raise ValueError(tr('白平衡增益必须为三个通道。'))
    r['wb_gain'] = [number(v, .125, 8) for v in gains]
    g = data.get('grading', grading())
    for zone in ('shadows', 'midtones', 'highlights'):
        pair = g.get(zone, grading()[zone])
        if len(pair) != 2:
            raise ValueError(tr('无效的色彩分级。'))
        r['grading'][zone] = [number(pair[0], 0, 360), number(pair[1], 0, 100)]
    r['grading']['balance'] = number(g.get('balance', 0), -100, 100)
    for key, default in effects().items():
        r['effects'][key] = number(data.get('effects', {}).get(key, default), -100 if key == 'vignette' else 0, 100)
    # 1.5.1: lens profile corrections (absent in older projects: off).
    from .lens import validate as validate_lens
    try:
        r['lens'] = validate_lens(data.get('lens'))
    except (ValueError, TypeError, KeyError):
        raise ValueError(tr('无效的镜头校正数据。'))
    # 1.7.0: a creative LUT with its tables (absent in older projects: none).
    from . import lut
    try:
        r['lut'] = lut.validate(data.get('lut'))
    except (TypeError, KeyError):
        raise ValueError(tr('无效的 LUT 数据。'))
    masks = data.get('masks', [])
    if len(masks) > 32:
        raise ValueError(tr('最多支持 32 个蒙版。'))
    for m in masks:
        if m['kind'] not in ('brush', 'linear', 'radial', 'luminance') + RASTER_KINDS:
            raise ValueError(tr('不支持的蒙版。'))
        item = new_mask(m['kind'], 1)
        item.update(name=str(m['name'])[:80], enabled=bool(m.get('enabled', True)),
                    invert=bool(m.get('invert', False)), opacity=number(m['opacity'], 0, 100),
                    feather=number(m['feather'], 0, 100), start=point(m['start']), end=point(m['end']),
                    adjustments=adj(m['adjustments']))
        if m['kind'] in RASTER_KINDS:
            from .selection import decode
            decode(m.get('raster'))
            item['raster'] = m['raster']
        if m['kind'] == 'object':
            clicks = m.get('points', [])
            if not isinstance(clicks, list) or not 1 <= len(clicks) <= OBJECT_POINTS:
                raise ValueError(tr('不支持的蒙版。'))
            item['points'] = []
            for c in clicks:
                if not isinstance(c, list) or len(c) != 3 or c[2] not in (0, 1) or isinstance(c[2], bool):
                    raise ValueError(tr('无效坐标。'))
                item['points'].append(point(c[:2]) + [int(c[2])])
        levels = m.get('luminance_range', [50., 100.])
        if len(levels) != 2:
            raise ValueError(tr('无效亮度范围。'))
        item['luminance_range'] = [number(v, 0, 100) for v in levels]
        if levels[0] > levels[1]:
            raise ValueError(tr('亮度范围下限不能超过上限。'))
        item['range_falloff'] = number(m.get('range_falloff', 20), 0, 100)
        if len(m.get('strokes', [])) > 20000:
            raise ValueError(tr('画笔笔划过多。'))
        for s in m.get('strokes', []):
            if len(s['points']) > 50000:
                raise ValueError(tr('笔划过长。'))
            item['strokes'].append(dict(points=[point(p) for p in s['points']],
                                        radius=number(s['radius'], .001, .5), erase=bool(s.get('erase', False))))
        r['masks'].append(item)
    return r


#: Largest project file written and read (1.7.0: fine AI masks are up to 4096 pixels; 32 MB before).
#: 2.1.0: 512 MB like albums (AI removal patches), and checked before a save replaces the file:
#: until then a project could be saved that LUMEN RAW then refused to open.
PROJECT_BYTES = 512 * 1024 * 1024


def stored_source(source, folder):
    """The photo's path as project and album files store it: relative to their folder with '/'
    (2.1.0: Windows wrote ``RAW\\IMG.ARW``, which macOS read as one file name), or absolute on
    another drive."""
    import os
    try:
        return Path(os.path.relpath(Path(source).resolve(), Path(folder).resolve())).as_posix()
    except ValueError:
        return str(Path(source).resolve())


def source_path(stored, folder):
    """The photo a stored path names, read on any system: a relative path written by older versions
    on Windows (``RAW\\IMG.ARW``) is found on macOS too."""
    import os
    import re
    text = str(stored)
    if os.sep == '/' and '\\' in text and not re.match(r'^[A-Za-z]:\\|^\\\\', text):
        text = text.replace('\\', '/')
    return (Path(folder) / text).resolve()


def save_project(path, source, edits, snapshots=None):
    path = Path(path)
    relative = stored_source(source, path.parent)
    snapshots = validate_snapshots(snapshots or [])
    # 1.7.0: LUT tables and mask rasters are stored once, however many snapshots use them.
    recipes, tables = pack([validate(edits)] + [s['edits'] for s in snapshots])
    payload = dict(application='LUMEN RAW', source=relative, edits=recipes[0],
                   snapshots=[dict(s, edits=e) for s, e in zip(snapshots, recipes[1:])], **tables)
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8')
    if len(data) > PROJECT_BYTES:
        raise ValueError(tr('工程过大（{size} MB，上限 {limit} MB），未保存，原工程文件保持不变；可删除部分 AI 移除或快照后再保存。',
                            size=len(data) // 2 ** 20, limit=PROJECT_BYTES // 2 ** 20))
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes(data)
    tmp.replace(path)


def load_project(path, include_snapshots=False):
    path = Path(path)
    if path.stat().st_size > PROJECT_BYTES:
        raise ValueError(tr('工程文件过大。'))
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('application') not in ('LUMEN RAW', 'Lumen ARW'):
        raise ValueError(tr('这不是 LUMEN RAW 工程。'))
    tables = stored_tables(data)
    source = source_path(data['source'], path.parent)
    result = (str(source), validate(unpack(data['edits'], tables)))
    return (*result, validate_snapshots(data.get('snapshots', []), tables)) if include_snapshots else result


def pack(recipes):
    """Recipes as project and album files store them (1.7.0): LUT tables in ``luts`` and mask
    rasters in ``rasters``, each once, referenced by digest.  Returns ``(recipes, tables)``;
    the recipes given are not changed."""
    import hashlib
    from . import lut
    recipes, luts_ = lut.pack(recipes)
    rasters, result = {}, []
    def ref(text):
        digest = hashlib.blake2b(text.encode('ascii'), digest_size=12).hexdigest()
        rasters[digest] = text
        return digest
    for edits in recipes:
        masks = edits.get('masks', []) if isinstance(edits, dict) else []
        if any(isinstance(m.get('raster'), str) for m in masks):
            packed = []
            for m in masks:
                if isinstance(m.get('raster'), str):
                    m = {k: v for k, v in m.items() if k != 'raster'} | dict(raster_ref=ref(m['raster']))
                packed.append(m)
            edits = dict(edits, masks=packed)
        # 2.1.0: AI removal patches are stored the same way.
        ops = edits.get('retouch', []) if isinstance(edits, dict) else []
        if any(isinstance(o.get('patch'), dict) and isinstance(o['patch'].get('png'), str) for o in ops):
            packed = []
            for o in ops:
                if isinstance(o.get('patch'), dict) and isinstance(o['patch'].get('png'), str):
                    patch = {k: v for k, v in o['patch'].items() if k != 'png'} | dict(png_ref=ref(o['patch']['png']))
                    o = dict(o, patch=patch)
                packed.append(o)
            edits = dict(edits, retouch=packed)
        result.append(edits)
    tables = {}
    if luts_:
        tables['luts'] = luts_
    if rasters:
        tables['rasters'] = rasters
    return result, tables


def stored_tables(data):
    """The ``luts`` and ``rasters`` storage of a project or album file (see ``pack``)."""
    tables = {}
    for key in ('luts', 'rasters'):
        value = data.get(key, {})
        if not isinstance(value, dict) or not all(isinstance(v, str) for v in value.values()):
            raise ValueError(tr('工程中的 LUT 数据缺失或已损坏。') if key == 'luts' else tr('工程中的蒙版数据缺失或已损坏。'))
        tables[key] = value
    return tables


def unpack(edits, tables):
    """A stored recipe with its LUT tables and mask rasters taken back from ``tables``."""
    import hashlib
    from . import lut
    tables = tables or {}
    edits = lut.unpack(edits, tables.get('luts', {}))
    rasters = tables.get('rasters', {})
    def stored(ref):
        raster = rasters.get(ref)
        digest = hashlib.blake2b(raster.encode('ascii'), digest_size=12).hexdigest() if isinstance(raster, str) else None
        if digest != ref:
            raise ValueError(tr('工程中的蒙版数据缺失或已损坏。'))
        return raster
    masks = edits.get('masks', []) if isinstance(edits, dict) else []
    if any(isinstance(m, dict) and 'raster_ref' in m for m in masks):
        restored = []
        for m in masks:
            if isinstance(m, dict) and 'raster_ref' in m:
                m = {k: v for k, v in m.items() if k != 'raster_ref'} | dict(raster=stored(m['raster_ref']))
            restored.append(m)
        edits = dict(edits, masks=restored)
    ops = edits.get('retouch', []) if isinstance(edits, dict) else []
    if any(isinstance(o, dict) and isinstance(o.get('patch'), dict) and 'png_ref' in o['patch'] for o in ops):
        restored = []
        for o in ops:
            if isinstance(o, dict) and isinstance(o.get('patch'), dict) and 'png_ref' in o['patch']:
                patch = {k: v for k, v in o['patch'].items() if k != 'png_ref'} | dict(png=stored(o['patch']['png_ref']))
                o = dict(o, patch=patch)
            restored.append(o)
        edits = dict(edits, retouch=restored)
    return edits


def luts(data):
    """The ``{digest: table}`` LUT storage of a project or album file (1.7.0)."""
    return stored_tables(data)['luts']


def validate_snapshots(snapshots, tables=None):
    if not isinstance(snapshots, list) or len(snapshots) > 20:
        raise ValueError(tr('最多支持 20 个快照。'))
    return [dict(name=str(s['name'])[:80], edits=validate(unpack(s['edits'], tables))) for s in snapshots]


#: What a look (preset, copied edits) carries; ``lut`` since 1.7.0.
LOOK_KEYS = ('adjustments', 'hsl', 'curves', 'tone_curve', 'curve_mode', 'grading', 'effects', 'monochrome', 'lut')


def look_fields(look):
    """What applying a look changes: its ``fields`` (a partial look, 1.7.0), else the keys it holds.

    Fields are ``LOOK_KEYS``, single sliders (``adjustments.exposure``, ``effects.grain``) and
    ``white_balance`` (a target Kelvin, applied to photos with a camera Kelvin)."""
    if 'fields' in look:
        return set(look['fields'])
    return {k for k in LOOK_KEYS + ('white_balance',) if k in look}


def _all_fields():
    return set(LOOK_KEYS) | {'white_balance'} | {f'adjustments.{k}' for k in adjustments()} | \
        {f'effects.{k}' for k in effects()}


def clean_look(look):
    """A validated copy of a look that holds only what it changes."""
    if not isinstance(look, dict):
        raise ValueError(tr('无效的预设数据。'))
    fields = look_fields(look)
    if not isinstance(look.get('fields', []), list) or not fields or fields - _all_fields():
        raise ValueError(tr('无效的预设数据。'))
    full = recipe()
    full['curve_mode'] = look.get('curve_mode', 'linear')
    full.update({k: copy.deepcopy(look[k]) for k in LOOK_KEYS if k in look})
    full = validate(full)
    groups = {f.split('.')[0] for f in fields}
    result = {k: full[k] for k in LOOK_KEYS if k in groups or (k == 'curve_mode' and 'curves' in groups)}
    if 'white_balance' in fields:
        kelvin = (look.get('white_balance') or {}).get('kelvin')
        if isinstance(kelvin, bool) or not isinstance(kelvin, (int, float)) or not 1500 <= kelvin <= 25000:
            raise ValueError(tr('无效的预设数据。'))
        result['white_balance'] = dict(kelvin=float(kelvin))
    if 'fields' in look:
        result['fields'] = sorted(fields)
    return result


def extract_look(edits):
    """The look of a recipe, for a preset: one without a LUT leaves a photo's LUT alone."""
    look = copy_look(edits)
    if look['lut'] is None:
        del look['lut']
    return look


def copy_look(edits):
    """The whole look of a recipe, for copying it to other photos (removes their LUT if it has none)."""
    clean = validate(edits)
    return {key: copy.deepcopy(clean[key]) for key in LOOK_KEYS}


def apply_look(edits, look, amount=100):
    """Blend a look against neutral; preserve WB sampling, masks and framing.

    Only what the look holds changes (``look_fields``): a partial look (1.7.0) such as a grain
    or LUT preset leaves the other settings of the photo as they are."""
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not 0 <= amount <= 100:
        raise ValueError(tr('预设强度需介于 0–100。'))
    from . import lut
    look = clean_look(look)
    fields = look_fields(look)
    base = recipe()
    result = copy.deepcopy(edits)
    t = amount / 100
    for group in ('adjustments', 'effects'):
        for k, v in look.get(group, {}).items():
            if group in fields or f'{group}.{k}' in fields:
                result[group][k] = base[group][k] + (v - base[group][k]) * t
    if 'hsl' in fields:
        result['hsl'] = [[v * t for v in row] for row in look['hsl']]
    if 'curves' in fields:
        result['curves'] = {c: [[x, x + (y - x) * t] for x, y in points] for c, points in look['curves'].items()}
    if 'curves' in fields or 'curve_mode' in fields:
        result['curve_mode'] = look['curve_mode']
    if 'tone_curve' in fields:
        result['tone_curve'] = [[x, x + (y - x) * t] for x, y in look['tone_curve']]
    if 'grading' in fields:
        result['grading'] = {z: [look['grading'][z][0], look['grading'][z][1] * t]
                             for z in ('shadows', 'midtones', 'highlights')}
        result['grading']['balance'] = look['grading']['balance'] * t
    if 'monochrome' in fields:
        result['monochrome'] = look['monochrome'] if t > 0 else False
    if 'lut' in fields:
        result['lut'] = lut.with_amount(look['lut'], look['lut']['amount'] * t) if look['lut'] else None
    if 'white_balance' in fields:
        camera = (result.get('white_balance') or {}).get('camera_kelvin')
        if camera:
            # An absolute white balance (Lightroom's Temperature), interpolated in mired.
            mired = 1e6 / camera + (1e6 / look['white_balance']['kelvin'] - 1e6 / camera) * t
            result['white_balance'] = dict(result['white_balance'], kelvin=float(min(25000., max(1500., 1e6 / mired))))
            result['wb_gain'] = [1., 1., 1.]
    return result


#: Preset files with a partial look, a LUT or a white balance are format 2 (1.7.0).
PRESET_VERSION = 2
#: Largest preset file read (a LUT is embedded in it).
PRESET_BYTES = 32 * 1024 * 1024


def write_preset(path, name, look, group='', notes=(), source='lumen'):
    """Write a .lumenpreset file; ``notes`` are (label, detail) pairs of settings an imported
    preset could not carry over (labels are source-language interface texts)."""
    path = Path(path)
    look = clean_look(look)
    modern = 'fields' in look or 'lut' in look or 'white_balance' in look
    payload = dict(application='Lumen preset', version=PRESET_VERSION if modern else 1, name=str(name)[:80], look=look)
    if group:
        payload['group'] = str(group)[:80]
    if source != 'lumen':
        payload['source'] = str(source)[:20]
    if notes:
        payload['notes'] = [[str(a)[:200], str(b)[:400]] for a, b in notes][:200]
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def save_preset(path, name, edits):
    write_preset(path, name, extract_look(edits))


def read_preset(data):
    """A preset (``name``, ``look``, ``subtitle``, ``group``, ``notes``, ``source``) from the
    parsed JSON of a .lumenpreset file."""
    if not isinstance(data, dict) or data.get('application') != 'Lumen preset' or data.get('version') not in (1, 2):
        raise ValueError(tr('请选择 Lumen 的 .lumenpreset 预设文件。'))
    look = data.get('look')
    if data['version'] == 1 and isinstance(look, dict):
        # Format 1 holds a whole look of the 1.x settings; keys an older version did not write are neutral.
        neutral = recipe()
        legacy = [k for k in LOOK_KEYS if k != 'lut']
        look = dict({k: neutral[k] for k in legacy}, curve_mode='linear') | {k: look[k] for k in legacy if k in look}
    notes = data.get('notes', [])
    if not isinstance(notes, list) or any(not isinstance(n, list) or len(n) != 2 for n in notes):
        raise ValueError(tr('无效的预设数据。'))
    return dict(name=str(data['name'])[:80], look=clean_look(look), subtitle=tr('自定义风格'),
                group=str(data.get('group', ''))[:80], notes=[[str(a), str(b)] for a, b in notes],
                source=str(data.get('source', 'lumen'))[:20])


def load_preset(path):
    path = Path(path)
    if path.stat().st_size > PRESET_BYTES:
        raise ValueError(tr('预设文件过大。'))
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
    except (ValueError, UnicodeDecodeError):
        raise ValueError(tr('请选择 Lumen 的 .lumenpreset 预设文件。'))
    return read_preset(data)


def builtin_presets():
    presets = []
    def add(name, subtitle, a, g=None, e=None, curves=None, hsl=None, mono=False):
        r = recipe()
        r['adjustments'].update(a)
        if g: r['grading'].update(g)
        if e: r['effects'].update(e)
        if curves: r['curves']['RGB'] = curves
        if hsl: r['hsl'] = hsl
        r['monochrome'] = mono
        presets.append(dict(name=name, subtitle=subtitle, look=extract_look(r), group='', notes=[], source='builtin'))
    add(tr('自然风光'), tr('NATURAL · 通透而克制'), dict(shadows=20, highlights=-25, vibrance=12, texture=15, dehaze=8))
    add(tr('海岸微风'), tr('COASTAL · 清冷蓝绿'), dict(temperature=-9, shadows=22, highlights=-20, saturation=-5),
        dict(shadows=[195, 18], highlights=[40, 8]))
    add(tr('金色时刻'), tr('GOLDEN · 温暖光线'), dict(temperature=18, highlights=-25, contrast=6, vibrance=10),
        dict(shadows=[225, 10], highlights=[38, 26]), dict(vignette=-14))
    add(tr('山野胶片'), tr('WOODLAND · 柔和绿意'), dict(shadows=12, highlights=-32, saturation=-8, texture=10),
        dict(shadows=[170, 12], highlights=[42, 14]), dict(grain=12, vignette=-12),
        [[0., .035], [.2, .19], [.75, .79], [1., .97]])
    add(tr('城市夜景'), tr('NIGHT · 冷暖对照'), dict(highlights=-40, shadows=28, contrast=10, color_noise=25, denoise=12),
        dict(shadows=[220, 22], highlights=[28, 20]), dict(vignette=-20))
    add(tr('黑白纪行'), tr('MONO · 光影与纹理'), dict(contrast=20, highlights=-22, shadows=16, texture=22),
        e=dict(grain=16, vignette=-16), mono=True)
    # 1.7.0: more looks.  The mixer rows follow COLORS: red, orange, yellow, green, aqua, blue, purple, magenta.
    def mixer(**rows):
        names = ('red', 'orange', 'yellow', 'green', 'aqua', 'blue', 'purple', 'magenta')
        return [list(map(float, rows.get(n, (0, 0, 0)))) for n in names]
    add(tr('日系清新'), tr('AIRY · 明亮通透'),
        dict(exposure=.3, contrast=-12, highlights=-20, shadows=25, blacks=12, saturation=-10, vibrance=8, temperature=-4, tint=3),
        dict(shadows=[210, 8]), curves=[[0., .06], [.25, .29], [.75, .8], [1., .98]],
        hsl=mixer(orange=(0, 0, 8), green=(15, -15, 0), aqua=(0, -5, 0), blue=(0, 0, 10)))
    add(tr('青橙电影'), tr('CINEMA · 青橙对比'), dict(contrast=14, highlights=-30, shadows=10, saturation=-6, vibrance=10),
        dict(shadows=[195, 22], highlights=[35, 18], balance=-10), dict(vignette=-10),
        hsl=mixer(orange=(-5, 8, 0), yellow=(-20, -20, 0), green=(40, -35, 0), aqua=(10, 0, 0), blue=(-10, 5, 0)))
    add(tr('复古胶片'), tr('VINTAGE · 褪色暖调'),
        dict(contrast=-8, highlights=-25, shadows=12, whites=-10, blacks=15, saturation=-15, temperature=10),
        dict(shadows=[200, 10], highlights=[42, 16]), dict(grain=22, grain_size=35, vignette=-12),
        [[0., .08], [.3, .3], [.7, .72], [1., .93]])
    add(tr('柔和人像'), tr('PORTRAIT · 温润肤色'),
        dict(exposure=.15, contrast=-6, highlights=-20, shadows=15, texture=-15, clarity=-8, vibrance=6, temperature=5),
        dict(highlights=[40, 8]), hsl=mixer(red=(0, -5, 0), orange=(0, -8, 12), yellow=(0, 0, 5)))
    add(tr('中画幅影调'), tr('MEDIUM FORMAT · 细腻层次'),
        dict(contrast=-5, highlights=-30, shadows=18, whites=8, blacks=-6, saturation=-4, vibrance=6, texture=6),
        dict(shadows=[215, 6], highlights=[45, 6]), curves=[[0., 0.], [.22, .2], [.5, .505], [.78, .8], [1., 1.]],
        hsl=mixer(orange=(-3, 4, 0), yellow=(0, -8, 0), green=(0, -12, -5), blue=(0, -6, -6)))
    add(tr('暗调情绪'), tr('MOODY · 低调深沉'),
        dict(exposure=-.3, contrast=18, highlights=-35, shadows=-5, whites=-12, blacks=-12, saturation=-22, vibrance=5,
             clarity=10, dehaze=6),
        dict(shadows=[205, 15], highlights=[35, 10]), dict(vignette=-22),
        hsl=mixer(yellow=(0, -20, 0), green=(0, -30, -15), blue=(0, -20, -15)))
    return presets


class History:
    def __init__(self, value):
        self.items = [copy.deepcopy(value)]
        self.index = 0

    def push(self, value):
        if value == self.items[self.index]:
            return
        del self.items[self.index + 1:]
        self.items.append(copy.deepcopy(value))
        if len(self.items) > 60:
            self.items.pop(0)
        self.index = len(self.items) - 1

    def move(self, delta):
        self.index = max(0, min(len(self.items) - 1, self.index + delta))
        return copy.deepcopy(self.items[self.index])
