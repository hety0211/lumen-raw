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


#: Recipe format written by this version; 6 adds ``process`` (1.6.0).
VERSION = 6
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
                white_balance=dict(camera_kelvin=None, kelvin=None, estimated=False), lens=lens_defaults())


def lens_defaults():
    from .lens import defaults
    return defaults()


def new_mask(kind, index):
    names = dict(brush=tr('画笔'), linear=tr('线性渐变'), radial=tr('径向渐变'), luminance=tr('亮度范围'), sky=tr('天空'),
                 person=tr('人物'), background=tr('背景'), color=tr('相似颜色'), subject=tr('主体'), foreground=tr('近景'))
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
    if not isinstance(data, dict) or data.get('version') not in (1, 2, 3, 4, 5, 6):
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
        if op.get('kind') not in ('heal', 'clone') or not 1 <= len(op.get('points', [])) <= 50000:
            raise ValueError(tr('无效修复笔划。'))
        item = dict(kind=op['kind'], enabled=bool(op.get('enabled', True)),
                    points=[point(p) for p in op['points']], radius=number(op['radius'], .001, .1),
                    opacity=number(op.get('opacity', 100), 0, 100), feather=number(op.get('feather', 50), 0, 100))
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
    masks = data.get('masks', [])
    if len(masks) > 32:
        raise ValueError(tr('最多支持 32 个蒙版。'))
    for m in masks:
        if m['kind'] not in ('brush', 'linear', 'radial', 'luminance', 'sky', 'person', 'background', 'color', 'subject', 'foreground'):
            raise ValueError(tr('不支持的蒙版。'))
        item = new_mask(m['kind'], 1)
        item.update(name=str(m['name'])[:80], enabled=bool(m.get('enabled', True)),
                    invert=bool(m.get('invert', False)), opacity=number(m['opacity'], 0, 100),
                    feather=number(m['feather'], 0, 100), start=point(m['start']), end=point(m['end']),
                    adjustments=adj(m['adjustments']))
        if m['kind'] in ('sky','person','background','color','subject','foreground'):
            from .selection import decode
            decode(m.get('raster'))
            item['raster'] = m['raster']
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


def save_project(path, source, edits, snapshots=None):
    import os
    path = Path(path)
    source = Path(source).resolve()
    try:
        relative = os.path.relpath(source, path.parent.resolve())
    except ValueError:
        relative = str(source)
    payload = dict(application='LUMEN RAW', source=relative, edits=validate(edits),
                   snapshots=validate_snapshots(snapshots or []))
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def load_project(path, include_snapshots=False):
    path = Path(path)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError(tr('工程文件过大。'))
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('application') not in ('LUMEN RAW', 'Lumen ARW'):
        raise ValueError(tr('这不是 LUMEN RAW 工程。'))
    source = (path.parent / data['source']).resolve()
    result = (str(source), validate(data['edits']))
    return (*result, validate_snapshots(data.get('snapshots', []))) if include_snapshots else result


def validate_snapshots(snapshots):
    if not isinstance(snapshots, list) or len(snapshots) > 20:
        raise ValueError(tr('最多支持 20 个快照。'))
    return [dict(name=str(s['name'])[:80], edits=validate(s['edits'])) for s in snapshots]


LOOK_KEYS = ('adjustments', 'hsl', 'curves', 'tone_curve', 'curve_mode', 'grading', 'effects', 'monochrome')


def extract_look(edits):
    clean = validate(edits)
    return {key: copy.deepcopy(clean[key]) for key in LOOK_KEYS}


def apply_look(edits, look, amount=100):
    """Blend a look against neutral; preserve WB sampling, masks and framing."""
    if not isinstance(amount, (int, float)) or not 0 <= amount <= 100:
        raise ValueError(tr('预设强度需介于 0–100。'))
    base = recipe()
    selected = recipe()
    selected['curve_mode'] = look.get('curve_mode', 'linear')
    selected.update({k: copy.deepcopy(look[k]) for k in LOOK_KEYS if k in look})
    selected = validate(selected)
    result = copy.deepcopy(edits)
    t = amount / 100
    for key in ('adjustments', 'effects'):
        result[key] = {k: base[key][k] + (v - base[key][k]) * t for k, v in selected[key].items()}
    result['hsl'] = [[v * t for v in row] for row in selected['hsl']]
    result['curves'] = {c: [[x, x + (y - x) * t] for x, y in points] for c, points in selected['curves'].items()}
    result['tone_curve'] = [[x, x + (y - x) * t] for x, y in selected['tone_curve']]
    result['curve_mode'] = selected['curve_mode']
    result['grading'] = {z: [selected['grading'][z][0], selected['grading'][z][1] * t]
                         for z in ('shadows', 'midtones', 'highlights')}
    result['grading']['balance'] = selected['grading']['balance'] * t
    result['monochrome'] = selected['monochrome'] if t > 0 else False
    return result


def save_preset(path, name, edits):
    path = Path(path)
    payload = dict(application='Lumen preset', version=1, name=str(name)[:80], look=extract_look(edits))
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def load_preset(path):
    path = Path(path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError(tr('预设文件过大。'))
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('application') != 'Lumen preset' or data.get('version') != 1:
        raise ValueError(tr('请选择 Lumen 的 .lumenpreset 预设文件。'))
    clean = apply_look(recipe(), data['look'])
    return dict(name=str(data['name'])[:80], look=extract_look(clean), subtitle=tr('自定义风格'))


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
        presets.append(dict(name=name, subtitle=subtitle, look=extract_look(r)))
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
