"""1.7.0: Lightroom presets, preset packs and creative LUTs."""
import copy
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lumen import catalog, commands, engine, lightroom, lut, model, nl_edit, presets

HERE = Path(__file__).resolve().parent / 'presets'


def cube(size, function, header=''):
    """A 3D .cube table of ``function`` (red fastest)."""
    axis = np.linspace(0, 1, size)
    b, g, r = np.meshgrid(axis, axis, axis, indexing='ij')
    rows = function(np.stack([r, g, b], -1).reshape(-1, 3))
    return header + f'LUT_3D_SIZE {size}\n' + '\n'.join('%.7f %.7f %.7f' % tuple(v) for v in rows) + '\n'


def warm(v):
    return np.clip(v ** np.array([.9, 1., 1.15]) * np.array([1.04, 1., .95]), 0, 1)


def frame(h=96, w=128, seed=1):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = np.stack([xx / w, yy / h, .3 + .2 * np.sin(xx / 9)], -1).astype(np.float32)
    return np.clip(x * (1 + .05 * rng.standard_normal((h, w, 1))).astype(np.float32), 0, 1.4)


@pytest.fixture
def library(tmp_path, monkeypatch):
    folder = tmp_path / 'presets'
    monkeypatch.setenv('LUMEN_PRESETS', str(folder))
    return folder


# ----------------------------------------------------------------------------- mapping

def test_xmp_preset_maps_onto_lumen_settings():
    preset = lightroom.read_xmp((HERE / 'warm-film.xmp').read_bytes(), 'warm-film')
    assert preset.name == '暖调胶片 01' and preset.group == '胶片质感' and preset.format == 'xmp'
    # The look's own settings (profile parameters, masks) stay inside their structures.
    assert preset.settings['ToneCurvePV2012'][0] == '0, 18'
    assert isinstance(preset.settings['Look'], dict) and len(preset.settings['MaskGroupBasedCorrections']) == 2
    look, notes, mapped = lightroom.convert(preset.settings)
    s = lightroom.SCALE
    a = look['adjustments']
    assert a['exposure'] == .35
    assert a['contrast'] == pytest.approx(round(-15 * s['contrast'], 1))
    assert a['highlights'] == pytest.approx(round(-45 * s['highlights'], 1))
    assert a['shadows'] == pytest.approx(round(30 * s['shadows'], 1))
    assert a['vibrance'] == pytest.approx(round(15 * s['vibrance'], 1))
    assert a['tint'] == pytest.approx(round(12 * s['tint'], 1))
    assert look['white_balance'] == dict(kelvin=6200.)
    # Absolute white balance replaces Lightroom's temperature: the relative slider is not part of it.
    assert 'adjustments.temperature' not in look['fields'] and 'white_balance' in look['fields']
    # Mixer: red, orange, yellow, green, aqua, blue, purple, magenta.
    assert look['hsl'][3] == [round(40 * s['hue'], 1), round(-40 * s['hsl_saturation'], 1), round(-15 * s['hsl_luminance'], 1)]
    assert look['hsl'][0] == [0., 0., 0.]
    # Point curves (0–255 → 0–1) and the parametric regions as the fine tone curve.
    assert look['curves']['RGB'][0] == [0., round(18 / 255, 5)] and look['curves']['RGB'][-1] == [1., round(240 / 255, 5)]
    assert look['curves']['B'] == [[0., round(12 / 255, 5)], [1., round(245 / 255, 5)]]
    assert look['curve_mode'] == 'smooth'
    tone = np.asarray(look['tone_curve'])
    assert tone[0].tolist() == [0., 0.] and tone[-1].tolist() == [1., 1.] and np.all(np.diff(tone[:, 1]) >= 0)
    assert np.interp(.125, *tone.T) > .125 and np.interp(.875, *tone.T) < .875   # shadows +10, highlights −20
    g = look['grading']
    assert g['shadows'] == [210., round(18 * s['grading'], 1)] and g['highlights'] == [42., round(22 * s['grading'], 1)]
    assert g['midtones'] == [30., round(6 * s['grading'], 1)] and g['balance'] == 15.
    assert look['effects']['vignette'] == pytest.approx(-18 * s['vignette']) and look['effects']['midpoint'] == 40
    assert look['effects']['grain'] == pytest.approx(min(100, 25 * s['grain'])) and look['effects']['grain_size'] == 36
    assert look['monochrome'] is False and 'monochrome' in look['fields']
    assert a['sharpness'] == round(40 * s['sharpness'], 1) and a['color_noise'] == round(25 * s['color_noise'], 1)
    assert mapped == len(look['fields']) and model.clean_look(look) == look


def test_settings_without_a_counterpart_are_reported_not_dropped():
    preset = lightroom.read_xmp((HERE / 'warm-film.xmp').read_bytes())
    _, notes, _ = lightroom.convert(preset.settings)
    labels = dict(notes)
    assert labels['Adobe 相机配置文件'] == 'Adobe Color'
    assert 'Adobe 创意配置文件（Look）' not in labels          # the same profile, named once
    assert labels['局部调整与蒙版'] == 'MaskGroupBasedCorrections × 2'
    assert labels['校准面板'] == 'RedHue +10, BlueHue -15, BlueSaturation +20'
    assert labels['镜头校正与去边'] == 'LensProfileEnable +1'
    assert labels['锐化半径、细节与蒙版'] == 'SharpenEdgeMasking +30'
    assert labels['暗角圆度、样式与高光'] == 'PostCropVignetteRoundness +20'
    assert labels['颗粒粗糙度'] == 'GrainFrequency +60'
    assert labels['色彩分级明亮度与混合'] == 'ColorGradeShadowLum -10'
    assert '6200 K' in labels['绝对色温只对有相机色温的 RAW 照片生效']
    # Neutral values are not reported: no transform, defringe, crop or noise-detail lines.
    assert not {'变换与几何', '裁切与方向', '降噪细节', '其他未映射的设置'} & set(labels)
    lines = lightroom.describe(notes)
    assert any(line.startswith('校准面板：') for line in lines)


def test_unknown_settings_and_creative_looks_are_listed():
    settings = dict(Exposure2012='+1.00', Look=dict(Name='Vintage 07', Amount='1'), CameraProfile='Adobe Standard',
                    FancyNewSlider='+12', RetouchAreas=[dict(SpotType='heal')], PointColors=['1, 2, 3'],
                    PerspectiveVertical='+15', HasCrop='True', CropTop='0.1')
    look, notes, mapped = lightroom.convert(settings)
    labels = dict(notes)
    assert look['adjustments']['exposure'] == 1. and look['fields'] == ['adjustments.exposure'] and mapped == 1
    assert labels['Adobe 创意配置文件（Look）'] == 'Vintage 07'
    assert labels['其他未映射的设置'] == 'FancyNewSlider'
    assert labels['局部调整与蒙版'] == 'RetouchAreas × 1' and labels['点颜色'] == '1'
    assert labels['变换与几何'] == 'PerspectiveVertical +15'
    assert labels['裁切与方向'] == 'HasCrop True, CropTop +0.10'
    assert lightroom.convert(dict(CameraProfile='Adobe Standard'))[0] is None


def test_lrtemplate_black_and_white_preset():
    preset = lightroom.read_lrtemplate((HERE / 'mono-classic.lrtemplate').read_bytes(), 'mono', '经典')
    assert preset.name == '经典黑白' and preset.group == '经典' and preset.format == 'lrtemplate'
    assert preset.settings['ToneCurvePV2012'] == [0., 0., 255., 255.] and preset.settings['IncrementalTemperature'] == 8
    look, notes, _ = lightroom.convert(preset.settings)
    s = lightroom.SCALE
    assert look['monochrome'] is True
    # The black & white mix becomes the mixer's lightness; Lightroom ignores HSL in B&W.
    assert look['hsl'][5] == [0., 0., round(-30 * s['gray'], 1)] and look['hsl'][1] == [0., 0., round(15 * s['gray'], 1)]
    assert look['adjustments']['temperature'] == pytest.approx(round(8 * s['temperature'], 1))
    assert look['curves']['RGB'] == [[0., 0.], [1., 1.]] and look['tone_curve'] == [[0., 0.], [1., 1.]]
    assert dict(notes)['局部调整与蒙版'] == 'PaintBasedCorrections × 1'
    # "As Shot" with an incremental temperature keeps the incremental value.
    assert look['adjustments']['tint'] == 0 or 'adjustments.tint' not in look['fields']


def test_lua_reader_handles_lightroom_syntax_without_running_code():
    text = '''-- comment
    s = { title = ZSTR "$$$/Name=A \\"quoted\\" name", type = "Develop", value = { settings = {
        Exposure2012 = -0.5, Contrast2012 = -1e1, ["Clarity2012"] = 0x0A, long = [[multi
line]], nested = { 1, 2; 3, }, flag = false, gone = nil, ToneCurvePV2012 = { 0, 10, 128, 128, 255, 250 } } } }'''
    preset = lightroom.read_lrtemplate(text.encode())
    assert preset.name == 'A "quoted" name'
    s = preset.settings
    assert s['Exposure2012'] == -.5 and s['Contrast2012'] == -10 and s['Clarity2012'] == 10
    assert s['long'] == 'multi\nline' and s['nested'] == [1., 2., 3.] and s['flag'] is False and s['gone'] is None
    look, _, _ = lightroom.convert(s)
    assert look['curves']['RGB'][0] == [0., round(10 / 255, 5)]
    with pytest.raises(ValueError):
        lightroom.read_lrtemplate(b's = { type = "Filter", value = { } }')
    with pytest.raises(ValueError):
        lightroom.read_lrtemplate(b's = { value = { settings = { a = os.execute("x") ')
    gbk = 's = { title = "人像 胶片", type = "Develop", value = { settings = { Exposure2012 = 0.3 } } }'.encode('gb18030')
    assert lightroom.read_lrtemplate(gbk).name == '人像 胶片'


def test_xmp_reader_forms_and_refusals():
    attributes = b'''<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
      <rdf:Description xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" crs:Name="Plain" crs:Saturation="-20">
      <crs:Exposure2012>+0.70</crs:Exposure2012></rdf:Description></rdf:RDF></x:xmpmeta>'''
    preset = lightroom.read_xmp(attributes)
    assert preset.name == 'Plain'
    look, _, _ = lightroom.convert(preset.settings)
    assert look['adjustments']['exposure'] == .7
    assert look['adjustments']['saturation'] == pytest.approx(round(-20 * lightroom.SCALE['saturation'], 1))
    profile = attributes.replace(b'crs:Name="Plain"', b'crs:PresetType="Look"')
    with pytest.raises(ValueError, match='配置文件'):
        lightroom.read_xmp(profile)
    with pytest.raises(ValueError):
        lightroom.read_xmp(b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><x>&a;</x>')
    with pytest.raises(ValueError):
        lightroom.read_xmp(b'<x:xmpmeta xmlns:x="adobe:ns:meta/"/>')
    with pytest.raises(ValueError):
        lightroom.read_xmp(b'not xml')


def test_legacy_process_presets_and_white_balance_cases():
    look, _, _ = lightroom.convert(dict(ProcessVersion='5.0', Exposure='+0.50', Contrast='50', FillLight='20',
                                        HighlightRecovery='40', Shadows='5'))
    a = look['adjustments']
    assert a['exposure'] == .5 and a['contrast'] == pytest.approx(25 * lightroom.SCALE['contrast'], abs=.06)
    assert a['shadows'] > 0 and a['highlights'] < 0 and a['blacks'] == 0
    auto, notes, _ = lightroom.convert(dict(WhiteBalance='Auto', Temperature='5000', Exposure2012='0'))
    assert 'white_balance' not in auto and dict(notes)['自动白平衡'] == 'Auto'
    shot, _, _ = lightroom.convert(dict(WhiteBalance='As Shot'))
    assert shot['adjustments']['temperature'] == 0 and {'adjustments.temperature', 'adjustments.tint'} <= set(shot['fields'])


def test_parametric_curve_shape():
    flat = lightroom.parametric_curve(0, 0, 0, 0)
    assert all(abs(x - y) < 1e-9 for x, y in flat)
    for amount in (-100, 100):
        curve = np.asarray(lightroom.parametric_curve(amount, amount, amount, amount, splits=(10, 50, 95)))
        assert curve[0].tolist() == [0., 0.] and curve[-1].tolist() == [1., 1.]
        assert np.all(np.diff(curve[:, 1]) >= 0) and len(curve) <= 256
    lights = np.asarray(lightroom.parametric_curve(0, 0, 100, 0))
    assert np.interp(.62, *lights.T) > .62 and abs(np.interp(.1, *lights.T) - .1) < 1e-9


def test_global_color_grading_adds_to_every_zone():
    look, _, _ = lightroom.convert(dict(ColorGradeGlobalHue='200', ColorGradeGlobalSat='20', SplitToningShadowHue='30',
                                        SplitToningShadowSaturation='0'))
    g = look['grading']
    for zone in ('shadows', 'midtones', 'highlights'):
        assert abs(g[zone][0] - 200) < 1 and g[zone][1] == pytest.approx(20 * lightroom.SCALE['grading'], abs=.1)
    both, _, _ = lightroom.convert(dict(ColorGradeGlobalHue='0', ColorGradeGlobalSat='20', SplitToningShadowHue='0',
                                        SplitToningShadowSaturation='20'))
    assert abs(both['grading']['shadows'][0]) < 1 or abs(both['grading']['shadows'][0] - 360) < 1
    assert both['grading']['shadows'][1] == pytest.approx(40 * lightroom.SCALE['grading'], abs=.2)


# ----------------------------------------------------------------------------- partial looks

def test_lightroom_preset_only_changes_what_it_holds():
    look, _, _ = lightroom.convert(dict(GrainAmount='40', GrainSize='25'))
    assert look['fields'] == ['effects.grain', 'effects.grain_size']
    edits = model.recipe()
    edits['adjustments'].update(exposure=.8, contrast=12)
    edits['effects'].update(vignette=-30)
    edits['lut'], _ = lut.parse_cube(cube(5, warm))
    applied = model.apply_look(edits, look)
    assert applied['adjustments'] == edits['adjustments'] and applied['effects']['vignette'] == -30
    assert applied['effects']['grain'] == pytest.approx(min(100, 40 * lightroom.SCALE['grain'])) and applied['lut'] == edits['lut']
    half = model.apply_look(applied, look, 50)
    assert half['effects']['grain'] == pytest.approx(min(100, 40 * lightroom.SCALE['grain']) / 2)
    assert model.apply_look(half, look, 0)['effects']['grain'] == 0


def test_absolute_white_balance_needs_a_camera_kelvin():
    look = model.clean_look(dict(fields=['white_balance'], white_balance=dict(kelvin=6500)))
    raw = model.recipe()
    raw['white_balance'] = dict(camera_kelvin=5000., kelvin=5000., estimated=False)
    raw['wb_gain'] = [1.1, 1., .9]
    full = model.apply_look(raw, look)
    assert full['white_balance']['kelvin'] == pytest.approx(6500) and full['wb_gain'] == [1., 1., 1.]
    half = model.apply_look(raw, look, 50)['white_balance']['kelvin']
    assert 1e6 / half == pytest.approx((1e6 / 5000 + 1e6 / 6500) / 2)
    jpeg = model.recipe()
    assert model.apply_look(jpeg, look) == jpeg
    with pytest.raises(ValueError):
        model.clean_look(dict(fields=['white_balance'], white_balance=dict(kelvin=100)))
    with pytest.raises(ValueError):
        model.clean_look(dict(fields=['adjustments.nonsense']))


def test_builtin_and_old_presets_keep_the_photo_lut(tmp_path):
    edits = model.recipe()
    edits['lut'], _ = lut.parse_cube(cube(5, warm))
    builtin = model.builtin_presets()
    assert len(builtin) == 12 and all('lut' not in p['look'] for p in builtin)
    assert model.apply_look(edits, builtin[4]['look'])['lut'] == edits['lut']
    # A 1.x preset file holds a whole look of the 1.x settings, so missing keys are neutral as before.
    old = tmp_path / 'old.lumenpreset'
    legacy = {k: v for k, v in model.extract_look(model.recipe()).items() if k != 'tone_curve'}
    old.write_text(json.dumps(dict(application='Lumen preset', version=1, name='old', look=legacy)), encoding='utf-8')
    loaded = model.load_preset(old)
    assert loaded['look']['tone_curve'] == [[0., 0.], [1., 1.]] and 'lut' not in loaded['look']
    curved = dict(edits, tone_curve=[[0., 0.], [.5, .6], [1., 1.]])
    assert model.apply_look(curved, loaded['look'])['tone_curve'] == [[0., 0.], [1., 1.]]
    # Copying a look to other photos copies (or removes) the LUT.
    assert model.apply_look(edits, model.copy_look(model.recipe()))['lut'] is None
    # A saved preset with a LUT carries the table and is format 2; without one it stays format 1.
    model.save_preset(tmp_path / 'with.lumenpreset', 'with', edits)
    model.save_preset(tmp_path / 'plain.lumenpreset', 'plain', model.recipe())
    assert json.loads((tmp_path / 'with.lumenpreset').read_text(encoding='utf-8'))['version'] == 2
    assert json.loads((tmp_path / 'plain.lumenpreset').read_text(encoding='utf-8'))['version'] == 1
    assert model.load_preset(tmp_path / 'with.lumenpreset')['look']['lut'] == edits['lut']


# ----------------------------------------------------------------------------- library and packs

def _zip_member(archive, name, data, utf8=True):
    info = zipfile.ZipInfo(name if utf8 else name.encode('gbk').decode('cp437'))
    if utf8:
        info.flag_bits |= 0x800
    archive.writestr(info, data)


def test_zip_pack_with_gbk_names_duplicates_and_luts(library, tmp_path):
    pack = tmp_path / '大神预设包.zip'
    xmp = (HERE / 'warm-film.xmp').read_bytes()
    template = (HERE / 'mono-classic.lrtemplate').read_bytes()
    plain = xmp.replace('暖调胶片 01'.encode(), '日落暖阳'.encode()).replace('胶片质感'.encode(), b'')
    with zipfile.ZipFile(pack, 'w') as archive:
        _zip_member(archive, '大神预设包/XMP/暖调胶片.xmp', xmp, utf8=False)
        _zip_member(archive, '大神预设包/LRTEMPLATE/暖调胶片.lrtemplate',
                    template.replace('经典黑白'.encode(), '暖调胶片 01'.encode()), utf8=False)
        _zip_member(archive, '大神预设包/黑白/经典黑白.lrtemplate', template)
        _zip_member(archive, '大神预设包/日落.xmp', plain)
        _zip_member(archive, '大神预设包/LUT/青橙.cube', cube(9, lambda v: v[:, [0, 2, 1]]).encode())
        _zip_member(archive, '__MACOSX/大神预设包/._日落.xmp', b'junk')
        _zip_member(archive, '大神预设包/移动版/日落.dng', b'II*\x00')
        _zip_member(archive, '大神预设包/坏的.xmp', b'<broken')
    result = presets.import_path(pack)
    names = sorted(i['name'] for i in result['imported'])
    assert names == ['日落暖阳', '暖调胶片 01', '经典黑白', '青橙']
    # The .xmp copy of a preset wins over its .lrtemplate copy, and carries its own group.
    film = next(i for i in result['imported'] if i['name'] == '暖调胶片 01')
    assert film['group'] == '胶片质感' and film['source'] == 'lightroom' and film['notes']
    assert next(i for i in result['imported'] if i['name'] == '日落暖阳')['group'] == '大神预设包'
    skipped = dict(result['skipped'])
    assert '大神预设包/移动版/日落.dng' in skipped and '大神预设包/坏的.xmp' in skipped
    assert not any(name.startswith('__MACOSX') for name in skipped)
    listed = presets.user_presets()
    assert len(listed) == 4 and len(list(library.glob('*.lumenpreset'))) == 4
    lut_preset = presets.find('青橙')
    assert lut_preset['look']['fields'] == ['lut'] and lut_preset['source'] == 'cube'
    assert presets.subtitle(lut_preset) == '大神预设包 · LUT 预设'
    # Importing the pack again replaces the same files.
    presets.import_path(pack)
    assert len(list(library.glob('*.lumenpreset'))) == 4
    report = presets.report(result)
    assert '已导入 4 个预设。' in report and '校准面板' in report and '未导入：' in report
    presets.delete(lut_preset)
    assert all(p['name'] != '青橙' for p in presets.user_presets())
    with pytest.raises(ValueError):
        presets.delete(model.builtin_presets()[0])


def test_single_files_and_bad_input(library, tmp_path):
    lone = tmp_path / '我的胶片' / 'Film.lrtemplate'
    lone.parent.mkdir()
    lone.write_bytes((HERE / 'mono-classic.lrtemplate').read_bytes())
    assert presets.import_path(lone)['imported'][0]['group'] == '我的胶片'
    with pytest.raises(ValueError):
        presets.import_path(tmp_path / 'x.txt')
    broken = tmp_path / 'broken.cube'
    broken.write_text('LUT_3D_SIZE 2\n0 0 0\n', encoding='utf-8')
    with pytest.raises(ValueError, match='行数'):
        presets.import_path(broken)


# ----------------------------------------------------------------------------- LUT parsing

def test_cube_3d_identity_title_comments_and_keywords():
    text = cube(17, lambda v: v, '# made by hand\nTITLE "Neutral 17"\nLUT_IN_VIDEO_RANGE\n\n')
    setting, notes = lut.parse_cube(text, 'file-name')
    assert setting['name'] == 'Neutral 17' and setting['size_3d'] == 17 and setting['size_1d'] == 0
    assert notes == ['已忽略 LUT 关键字 LUT_IN_VIDEO_RANGE']
    x = np.random.default_rng(2).random((40, 50, 3), dtype=np.float32)
    np.testing.assert_allclose(lut.evaluate(x, setting), x, atol=2e-7)
    assert lut.parse_cube(cube(5, lambda v: v), 'stem')[0]['name'] == 'stem'


def test_cube_1d_table_and_domain():
    rows = '\n'.join(f'{v:.6f} {v * v:.6f} {1 - v:.6f}' for v in np.linspace(0, 1, 5))
    setting, _ = lut.parse_cube(f'LUT_1D_SIZE 5\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 2 2 2\n{rows}\n')
    assert setting['size_1d'] == 5 and setting['size_3d'] == 0 and setting['domain_1d'] == [[0., 0., 0.], [2., 2., 2.]]
    out = lut.evaluate(np.array([[1., 1., 1.], [2., 0., 3.]], np.float32), setting)
    np.testing.assert_allclose(out[0], [.5, .25, .5], atol=1e-6)     # 1 is the middle of the 0–2 domain
    np.testing.assert_allclose(out[1], [1., 0., 0.], atol=1e-6)      # inputs are clamped to the domain


def test_cube_3d_domain_and_resolve_shaper():
    setting, _ = lut.parse_cube(cube(3, lambda v: v * 2, 'DOMAIN_MIN -1 -1 -1\nDOMAIN_MAX 1 1 1\n'))
    assert setting['domain_3d'] == [[-1.] * 3, [1.] * 3]
    np.testing.assert_allclose(lut.evaluate(np.array([[0., 1., -1.]], np.float32), setting), [[1., 2., 0.]], atol=1e-6)
    shaper = '\n'.join(f'{v:.6f} {v:.6f} {v:.6f}' for v in np.sqrt(np.linspace(0, 1, 9)))
    resolve = f'LUT_1D_SIZE 9\nLUT_1D_INPUT_RANGE 0 4\nLUT_3D_SIZE 2\nLUT_3D_INPUT_RANGE 0 1\n{shaper}\n' + \
        '\n'.join('%g %g %g' % tuple(v) for v in np.stack(np.meshgrid(*[[0, 1]] * 3, indexing='ij')[::-1], -1).reshape(-1, 3))
    setting, _ = lut.parse_cube(resolve)
    assert setting['size_1d'] == 9 and setting['size_3d'] == 2 and setting['domain_1d'] == [[0.] * 3, [4.] * 3]
    np.testing.assert_allclose(lut.evaluate(np.array([[1., 2., 4.]], np.float32), setting), [[.5, np.sqrt(.5), 1.]], atol=2e-3)
    assert 'LUT_1D_INPUT_RANGE' in lut.to_cube(setting)


@pytest.mark.parametrize('text, message', [
    ('LUT_3D_SIZE 2\n0 0 0\n', '行数'),
    ('0 0 0\n1 1 1\n', 'LUT_1D_SIZE'),
    ('LUT_3D_SIZE 300\n', '尺寸'),
    ('LUT_1D_SIZE 2\n0 0 x\n1 1 1\n', '格式'),
    ('LUT_1D_SIZE 2\n0 0\n1 1 1\n', '三个'),
    ('LUT_1D_SIZE 2\nDOMAIN_MIN 1 0 0\nDOMAIN_MAX 0 1 1\n0 0 0\n1 1 1\n', 'DOMAIN'),
    ('LUT_1D_SIZE 2\n0 0 0\n1 1 nan\n', '无效'),
    ('LUT_1D_SIZE 2\n0 0 0\nTITLE "late"\n1 1 1\n', '格式'),
])
def test_cube_errors(text, message):
    with pytest.raises(ValueError, match=message):
        lut.parse_cube(text)


def test_tetrahedral_interpolation_is_exact_for_affine_tables_and_keeps_greys_grey():
    matrix = np.array([[.8, .15, .05], [.1, .85, .05], [.05, .1, .85]])
    setting, _ = lut.parse_cube(cube(5, lambda v: v @ matrix.T * .9 + .05))
    x = np.random.default_rng(4).random((1000, 3), dtype=np.float32)
    np.testing.assert_allclose(lut.evaluate(x, setting), x @ matrix.T * .9 + .05, atol=2e-6)
    bent, _ = lut.parse_cube(cube(9, lambda v: v ** 1.8 + .1 * (v[:, [1, 2, 0]] - v)))
    grey = np.repeat(np.linspace(0, 1, 50, dtype=np.float32)[:, None], 3, axis=1)
    out = lut.evaluate(grey, bent)
    assert np.abs(out - out[:, :1]).max() < 1e-6        # only the diagonal's corners are used


def test_large_tables_are_resampled_and_amount_blends():
    setting, notes = lut.parse_cube(cube(70, warm))
    assert setting['size_3d'] == lut.MAX_3D and '70³ LUT 已重采样为 65³' in notes
    x = np.random.default_rng(6).random((64, 64, 3), dtype=np.float32)
    np.testing.assert_allclose(lut.evaluate(x, setting), warm(x.astype(np.float64)), atol=4e-3)
    half = lut.with_amount(setting, 50)
    full = lut.apply(x, setting)
    np.testing.assert_allclose(lut.apply(x, half), x + (full - x) / 2, atol=1e-6)
    assert lut.apply(x, lut.with_amount(setting, 0)) is x and lut.apply(x, None) is x
    assert lut.validate(setting) == setting
    with pytest.raises(ValueError):
        lut.validate(dict(setting, data=setting['data'][:-8]))
    with pytest.raises(ValueError):
        lut.validate(dict(setting, size_3d=64))
    with pytest.raises(ValueError):
        lut.validate(dict(setting, amount=101))


def test_to_cube_round_trip():
    setting, _ = lut.parse_cube(cube(9, warm, 'TITLE "Warm"\n'))
    again, _ = lut.parse_cube(lut.to_cube(setting))
    x = np.random.default_rng(8).random((100, 3), dtype=np.float32)
    np.testing.assert_allclose(lut.evaluate(x, again), lut.evaluate(x, setting), atol=1e-6)
    assert again['name'] == 'Warm'


# ----------------------------------------------------------------------------- pipeline

def _pointwise(process, setting, **effects):
    edits = model.recipe()
    edits['process'] = process
    edits['adjustments'].update(contrast=12, saturation=8, exposure=.2)
    edits['hsl'][5] = [10., 15., -5.]
    edits['grading']['shadows'] = [210., 15.]
    edits['effects'].update(effects)
    edits['lut'] = lut.with_amount(setting, 70)
    return model.validate(edits)


@pytest.mark.parametrize('process', [1, 2])
def test_block_renders_match_the_whole_frame_pixel_for_pixel(process):
    setting, _ = lut.parse_cube(cube(17, lambda v: np.clip(v[:, [1, 2, 0]] * .9 + .05, 0, 1)))
    source = frame(120, 160)
    edits = _pointwise(process, setting, vignette=-25)
    whole = engine.process(source, edits, engine.Backend('cpu'), apply_crop=False)
    rects = ((0, 0, 160, 120), (13, 7, 101, 77), (150, 100, 160, 120))
    for x0, y0, x1, y1 in rects:
        block = engine.process_region(source, edits, engine.Backend('cpu'), (x0, y0, x1, y1))
        np.testing.assert_array_equal(block, whole[y0:y1, x0:x1])
    # The stage itself is pointwise: any block of its input gives the same pixels.
    plain = engine.process(source, dict(edits, lut=None), engine.Backend('cpu'), apply_crop=False)
    staged = lut.apply(plain, edits['lut'])
    np.testing.assert_array_equal(lut.apply(np.ascontiguousarray(plain[30:70, 40:90]), edits['lut']), staged[30:70, 40:90])
    assert np.abs(whole - plain).mean() > .02
    # Grain comes after the LUT; its resampled noise field differs by at most a float rounding
    # between a block and the frame, with or without a LUT.
    grained = dict(edits, effects=dict(edits['effects'], grain=12.))
    for setting_ in (grained['lut'], None):
        e = dict(grained, lut=setting_)
        frame_ = engine.process(source, e, engine.Backend('cpu'), apply_crop=False)
        for x0, y0, x1, y1 in rects:
            block = engine.process_region(source, e, engine.Backend('cpu'), (x0, y0, x1, y1))
            assert np.abs(block - frame_[y0:y1, x0:x1]).max() <= 1.2e-7


def test_spatial_edits_and_wide_space_blocks_with_a_lut():
    setting, _ = lut.parse_cube(cube(17, warm))
    source = frame(120, 160)
    edits = _pointwise(2, setting)
    edits['adjustments'].update(clarity=25, highlights=-30, shadows=20)
    for space in (None, 'p3'):
        whole = engine.process(source, edits, engine.Backend('cpu'), apply_crop=False, space=space)
        block = engine.process_region(source, edits, engine.Backend('cpu'), (21, 15, 133, 97), space=space)
        assert np.abs(block - whole[15:97, 21:133]).max() < 2e-4
    # Wide gamut: colours inside sRGB get the same LUT change as an sRGB render.
    p3 = engine.process(source, edits, engine.Backend('cpu'), apply_crop=False, space='p3')
    srgb = engine.process(source, edits, engine.Backend('cpu'), apply_crop=False)
    assert np.abs(engine.convert_display(srgb, 'p3') - p3).mean() < 6e-3


def test_large_images_stream_the_lut_stage(monkeypatch):
    from lumen import large_image
    setting, _ = lut.parse_cube(cube(9, warm))
    source = frame(260, 300)
    edits = _pointwise(2, setting)
    whole = engine.process(source, edits, engine.Backend('cpu'), apply_crop=False, _stream=False)
    monkeypatch.setattr(large_image, 'MAP_BYTES', 1)
    monkeypatch.setattr(lut, 'CHUNK', 4096)
    streamed = engine.process(source, edits, engine.Backend('cpu'), apply_crop=False)
    np.testing.assert_array_equal(streamed, whole)


def test_old_projects_render_exactly_as_before(tmp_path):
    source = frame()
    edits = model.recipe()
    edits['adjustments'].update(exposure=.4, contrast=10, clarity=15)
    old = {k: v for k, v in copy.deepcopy(edits).items() if k != 'lut'}
    old['version'] = 6
    project = tmp_path / 'old.lumen'
    project.write_text(json.dumps(dict(application='LUMEN RAW', source='a.png', edits=old, snapshots=[])), encoding='utf-8')
    _, loaded = model.load_project(project)
    assert loaded['lut'] is None and loaded['process'] == old['process']
    reference = engine.process(source, dict(edits, lut=None))
    np.testing.assert_array_equal(engine.process(source, loaded), reference)
    setting, _ = lut.parse_cube(cube(5, warm))
    np.testing.assert_array_equal(engine.process(source, dict(edits, lut=lut.with_amount(setting, 0))), reference)


def test_projects_embed_the_lut_once_and_open_without_the_file(tmp_path):
    path = tmp_path / 'Film.cube'
    path.write_text(cube(17, warm, 'TITLE "Film"\n'), encoding='utf-8')
    setting, _ = lut.load_cube(path)
    edits = model.recipe()
    edits['lut'] = lut.with_amount(setting, 65)
    snapshots = [dict(name=f'v{i}', edits=copy.deepcopy(edits)) for i in range(20)]
    project = tmp_path / 'a.lumen'
    model.save_project(project, tmp_path / 'a.arw', edits, snapshots)
    path.unlink()        # another computer: no .cube file there
    raw = json.loads(project.read_text(encoding='utf-8'))
    assert len(raw['luts']) == 1 and 'data' not in raw['edits']['lut'] and raw['edits']['lut']['amount'] == 65
    assert project.stat().st_size < 600_000
    _, loaded, snaps = model.load_project(project, include_snapshots=True)
    assert loaded == model.validate(edits) and all(s['edits'] == loaded for s in snaps)
    raw['luts'] = {}
    project.write_text(json.dumps(raw), encoding='utf-8')
    with pytest.raises(ValueError, match='LUT'):
        model.load_project(project)
    packed, tables = lut.pack([edits, model.recipe()])
    assert lut.unpack(packed[0], tables) == edits and packed[1] == model.recipe() and edits['lut']['data']


# ----------------------------------------------------------------------------- commands and MCP

@pytest.fixture
def photo(tmp_path):
    yy, xx = np.mgrid[0:120, 0:180]
    rgb = np.stack([xx / 180 * .8, yy / 120 * .5, .3 + .2 * np.sin(xx / 15)], axis=2)
    path = tmp_path / 'photo one.png'
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)
    return str(path)


@pytest.fixture
def headless(tmp_path):
    from lumen.session import Session
    return Session(catalog.Catalog(tmp_path / 'catalog.jsonl'), engine.Backend('cpu'))


def test_edit_apply_loads_changes_and_removes_a_lut(headless, photo, tmp_path):
    path = tmp_path / 'look.cube'
    path.write_text(cube(9, warm, 'TITLE "Warm look"\n'), encoding='utf-8')
    run = lambda command, **args: commands.run(headless, command, args)
    run('photo.open', path=photo)
    result = run('edit.apply', changes=dict(lut=dict(path=str(path), amount=60)))
    assert result['state']['lut'] == dict(name='Warm look', amount=60., size='9³')
    assert any(line.startswith('LUT  Warm look') for line in result['applied'])
    recipe = run('edit.get', full=True)['recipe']
    assert recipe['lut'] == dict(name='Warm look', amount=60., size='9³')          # never the tables
    assert run('edit.apply', changes=dict(lut=dict(amount=30)))['state']['lut']['amount'] == 30
    assert run('edit.apply', changes=dict(lut=dict(path=str(path))), amount=50)['state']['lut']['amount'] == pytest.approx(65)
    assert 'lut' not in run('edit.apply', changes=dict(lut=None))['state']
    with pytest.raises(commands.CommandError, match='LUT'):
        run('edit.apply', changes=dict(lut=dict(path=str(tmp_path / 'missing.cube'))))
    assert 'lut' in commands.REGISTRY['edit.apply'].params['changes']['description']


def test_language_models_cannot_load_lut_files(tmp_path):
    path = tmp_path / 'x.cube'
    path.write_text(cube(3, warm), encoding='utf-8')
    edits = model.recipe()
    plan = nl_edit.plan(edits, dict(lut=dict(path=str(path))))
    assert plan.target['lut'] is None and plan.notes
    edits['lut'], _ = lut.load_cube(path)
    plan = nl_edit.plan(edits, dict(lut=dict(amount=40), explanation='softer'))
    assert plan.target['lut']['amount'] == 40 and nl_edit.current_state(plan.target)['lut']['amount'] == 40
    assert 'lut' in nl_edit.system_prompt()
    assert nl_edit.plan(edits, dict(reset=True)).target['lut'] is None


def test_preset_commands_import_list_and_batch_apply(library, headless, photo, tmp_path):
    other = tmp_path / 'photo two.png'
    Image.open(photo).transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(other)
    run = lambda command, **args: commands.run(headless, command, args)
    imported = run('preset.import', path=str(HERE / 'warm-film.xmp'))
    assert imported['imported'][0]['name'] == '暖调胶片 01' and imported['imported'][0]['not_carried_over']
    listed = run('edit.presets')['presets']
    film = next(p for p in listed if p['name'] == '暖调胶片 01')
    assert film['kind'] == 'lightroom' and film['group'] == '胶片质感' and film['notes']
    assert sum(p['kind'] == 'built-in' for p in listed) == 12
    run('photo.open', path=photo)
    batch = run('edit.preset', name='暖调胶片 01', amount=50, paths=[photo, str(other)])
    assert len(batch['applied']) == 2
    edits = headless.edits(str(Path(other).resolve()))
    assert edits['adjustments']['exposure'] == pytest.approx(.175)
    with pytest.raises(commands.CommandError):
        run('preset.import', path=str(tmp_path / 'nothing.zip'))
    from lumen import mcp
    tools = {t['name']: t for t in mcp.tools()}
    assert 'import_presets' in tools and 'paths' in tools['apply_preset']['inputSchema']['properties']


def test_cli_render_with_a_lut(photo, tmp_path):
    from lumen import cli
    path = tmp_path / 'cli.cube'
    path.write_text(cube(9, lambda v: 1 - v), encoding='utf-8')
    plain, inverted = tmp_path / 'plain.png', tmp_path / 'inverted.png'
    assert cli.main(['render', photo, str(plain), '--size', '90']) == 0
    assert cli.main(['render', photo, str(inverted), '--size', '90', '--lut', str(path)]) == 0
    a = np.asarray(Image.open(plain), np.float32)
    b = np.asarray(Image.open(inverted), np.float32)
    assert np.abs(a + b - 255).mean() < 3


# ----------------------------------------------------------------------------- window

from test_ui import app, window, wait_until  # noqa: E402,F401


def test_preset_panel_imports_groups_luts_and_batch(window, library, tmp_path, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    w = window
    reports = []
    monkeypatch.setattr(type(w), 'show_report', lambda self, title, text: reports.append(text))
    w.import_preset_files([str(HERE / 'warm-film.xmp'), str(HERE / 'mono-classic.lrtemplate')])
    wait_until(lambda: any(p.get('source') == 'lightroom' for p in w.presets) and not w.jobs)
    assert reports and '校准面板' in reports[0]
    groups = [w.preset_group.itemData(i) for i in range(w.preset_group.count())]
    assert '胶片质感' in groups and 'presets' in groups
    w.preset_group.setCurrentIndex(groups.index('胶片质感'))
    assert w.preset_list.count() == 1
    item = w.preset_list.item(0)
    assert item.text().startswith('暖调胶片 01') and '导入时未能套用' in item.toolTip()
    w.choose_preset(item.data(Qt.ItemDataRole.UserRole))
    assert w.edits['adjustments']['exposure'] == .35
    w.preset_group.setCurrentIndex(1)                       # built-in looks only
    assert w.preset_list.count() == 12
    # A LUT on the current photo, its strength and removal.
    path = tmp_path / 'panel.cube'
    path.write_text(cube(9, warm, 'TITLE "Panel"\n'), encoding='utf-8')
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (str(path), ''))
    w.load_lut()
    assert w.edits['lut']['name'] == 'Panel' and not w.lut_amount.isHidden()
    w.lut_amount.spin.setValue(40)
    assert w.edits['lut']['amount'] == 40
    wait_until(lambda: not w.render_running and not w.timer.isActive())
    w.remove_lut()
    assert w.edits['lut'] is None and w.lut_amount.isHidden()
    # Batch: one preset on every selected photo of the filmstrip.
    index = next(i for i, p in enumerate(w.presets) if p['name'] == '经典黑白')
    for row in range(w.filmstrip.count()):
        w.filmstrip.item(row).setSelected(True)
    w.apply_preset_to_selection(index)
    assert w.edits['monochrome'] is True
    # Deleting an imported preset.
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.StandardButton.Yes)
    w.delete_preset(index)
    assert all(p['name'] != '经典黑白' for p in w.presets)
