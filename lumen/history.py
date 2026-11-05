"""Names of edit steps for the history list (2.3.0, the 历史 page beside 预设).

``describe(before, after)`` names what changed between two recipes the way the panels call it:
one slider with its new value ("曝光 EV +0.80"), several of one kind by their group ("光影 ·
3 项"), a mask, retouching or face added or removed by what it is.  Actions that change many
things at once (a preset, 自动, a reset) pass their own name to ``History.push`` instead.
"""
from __future__ import annotations

from .i18n import tr, N_

#: Slider names as the panels show them (source texts, translated when shown).
SLIDERS = dict(exposure=N_('曝光 EV'), contrast=N_('对比度'), shadows=N_('暗部'), highlights=N_('亮部'),
               blacks=N_('黑色'), whites=N_('白色'), temperature=N_('冷暖微调'), tint=N_('色调 · 绿 / 洋红'),
               saturation=N_('饱和度'), vibrance=N_('自然饱和度'), dehaze=N_('去薄雾'), clarity=N_('清晰度'),
               texture=N_('纹理'), sharpness=N_('锐化'), denoise=N_('明度降噪'), color_noise=N_('彩色杂点'))
EFFECTS = dict(vignette=N_('暗角'), midpoint=N_('暗角'), feather=N_('暗角'), grain=N_('胶片颗粒'),
               grain_size=N_('胶片颗粒'))
PORTRAIT = dict(smooth=N_('磨皮'), whiten=N_('美白'), acne=N_('祛痘'), circles=N_('去黑眼圈'), teeth=N_('白牙'),
                eyes=N_('大眼'), slim=N_('瘦脸'))
RETOUCH = dict(heal=N_('污点修复'), clone=N_('仿制图章'), remove=N_('AI 移除'))


def _value(key, value):
    if key == 'exposure':
        return f'{value:+.2f}'
    return f'{value:+.0f}' if key in SLIDERS else f'{value:.0f}'


def describe(before, after):
    """A short name for the step from recipe ``before`` to ``after``."""
    if before is None:
        return tr('打开照片')
    a, b = before.get('adjustments', {}), after.get('adjustments', {})
    sliders = [k for k in b if a.get(k) != b.get(k)]
    if sliders and before.get('tone_curve') != after.get('tone_curve'):
        return tr('曝光曲线')
    if len(sliders) == 1:
        key = sliders[0]
        return f'{tr(SLIDERS.get(key, key))} {_value(key, b[key])}'
    if sliders:
        return tr('光影与色彩 · {count} 项', count=len(sliders))
    if before.get('tone_curve') != after.get('tone_curve'):
        return tr('曝光曲线')
    if before.get('white_balance', {}).get('kelvin') != after.get('white_balance', {}).get('kelvin'):
        kelvin = after.get('white_balance', {}).get('kelvin')
        return tr('色温 {kelvin} K', kelvin=round(kelvin)) if kelvin else tr('白平衡')
    if before.get('wb_gain') != after.get('wb_gain'):
        return tr('白平衡取样')
    if before.get('hsl') != after.get('hsl'):
        from .model import COLORS
        rows = [i for i, (x, y) in enumerate(zip(before.get('hsl', []), after.get('hsl', []))) if x != y]
        return tr('色彩混合器') + (f' · {COLORS[rows[0]][0]}' if len(rows) == 1 else '')
    if before.get('curves') != after.get('curves') or before.get('curve_mode') != after.get('curve_mode'):
        channels = [c for c in after.get('curves', {}) if before.get('curves', {}).get(c) != after['curves'][c]]
        return tr('曲线') + (f' · {channels[0]}' if len(channels) == 1 else '')
    if before.get('grading') != after.get('grading'):
        return tr('色彩分级')
    if before.get('monochrome') != after.get('monochrome'):
        return tr('黑白处理') if after.get('monochrome') else tr('彩色')
    masks_a, masks_b = before.get('masks', []), after.get('masks', [])
    if len(masks_b) > len(masks_a):
        return tr('添加蒙版：{name}', name=masks_b[-1].get('name', ''))
    if len(masks_b) < len(masks_a):
        return tr('删除蒙版')
    for x, m in zip(masks_a, masks_b):
        if x != m:
            if len(m.get('strokes', [])) != len(x.get('strokes', [])):
                return tr('蒙版画笔：{name}', name=m.get('name', ''))
            return tr('蒙版：{name}', name=m.get('name', ''))
    ops_a, ops_b = before.get('retouch', []), after.get('retouch', [])
    if len(ops_b) > len(ops_a):
        return tr(RETOUCH.get(ops_b[-1].get('kind'), N_('修复')))
    if len(ops_b) < len(ops_a):
        return tr('删除修复')
    if ops_a != ops_b:
        return tr('修复')
    faces_a = (before.get('portrait') or {}).get('faces', [])
    faces_b = (after.get('portrait') or {}).get('faces', [])
    if len(faces_a) != len(faces_b) and not any(any(f['settings'].values()) for f in faces_b):
        return tr('检测人脸')
    if faces_a != faces_b:
        for x, y in zip(faces_a, faces_b):
            keys = [k for k in y['settings'] if x['settings'].get(k) != y['settings'][k]]
            if len(keys) == 1:
                return f'{tr("人像")} · {tr(PORTRAIT.get(keys[0], keys[0]))} {y["settings"][keys[0]]:.0f}'
        return tr('人像')
    if before.get('crop') != after.get('crop'):
        return tr('裁切')
    if before.get('straighten') != after.get('straighten'):
        return tr('拉直 {angle}°', angle=f'{after.get("straighten", 0):+.1f}')
    if before.get('rotation') != after.get('rotation'):
        return tr('旋转 90°')
    if before.get('flip') != after.get('flip'):
        return tr('翻转')
    if before.get('effects') != after.get('effects'):
        keys = {EFFECTS.get(k, k) for k in after.get('effects', {}) if before.get('effects', {}).get(k) != after['effects'][k]}
        return tr(keys.pop()) if len(keys) == 1 else tr('效果')
    if before.get('lens') != after.get('lens'):
        return tr('镜头校正')
    if before.get('lut') != after.get('lut'):
        return 'LUT'
    if before.get('watermark') != after.get('watermark'):
        return tr('水印')
    if before.get('develop') != after.get('develop') or before.get('process') != after.get('process'):
        return tr('显影起点')
    return tr('编辑')
