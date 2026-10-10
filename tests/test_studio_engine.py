import copy
import numpy as np
import pytest
from lumen import model, engine


def test_old_recipe_migrates_without_changing_pixels():
    old = model.recipe()
    old['version'] = 1
    for key in ('wb_gain', 'grading', 'effects', 'straighten', 'monochrome', 'process'):
        old.pop(key)
    old['adjustments'].pop('texture')
    new = model.validate(old)
    assert new['version'] == model.VERSION and new['process'] == 1
    source = np.random.default_rng(5).random((30, 40, 3)).astype(np.float32) * .6
    np.testing.assert_allclose(engine.process(source, new), engine.process(source, old), atol=1e-6)


def test_white_balance_neutralizes_cast_and_rejects_clipping():
    source = np.full((30, 30, 3), [.32, .2, .12], dtype=np.float32)
    gains = engine.sample_white_balance(source, [.5, .5])
    balanced = source * gains
    assert np.ptp(balanced[15, 15]) < 1e-6
    for level in (0., 1.):
        with pytest.raises(ValueError):
            engine.sample_white_balance(np.full_like(source, level), [.5, .5])


def test_auto_tone_improves_underexposure_and_handles_black():
    source = np.full((20, 20, 3), .04, dtype=np.float32)
    settings = engine.auto_tone(source)
    assert 1 < settings['exposure'] < 3
    assert abs(.04 * 2 ** settings['exposure'] - .18) < .01
    black = engine.auto_tone(np.zeros_like(source))
    assert black['exposure'] == 0 and np.isfinite(list(black.values())).all()


def test_three_way_grading_targets_shadows():
    rgb = np.full((4, 10, 3), .2, dtype=np.float32)
    rgb[:, 5:] = .8
    grading = model.grading()
    np.testing.assert_array_equal(engine.color_grade(rgb, grading), rgb)
    grading['shadows'] = [220., 60.]
    out = engine.color_grade(rgb, grading)
    delta = out - rgb
    assert delta[:, :5, 2].mean() > delta[:, 5:, 2].mean() * 5
    assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1


def test_vignette_crop_center_and_deterministic_grain():
    rgb = np.full((100, 160, 3), .5, dtype=np.float32)
    settings = model.effects()
    settings['vignette'] = -70
    out = engine.finishing(rgb, settings)
    assert out[50, 80, 0] == .5 and out[0, 0, 0] < .3
    crop = [.0, .0, .5, 1.]
    out = engine.finishing(rgb, settings, crop)
    assert out[50, 40, 0] == .5
    settings.update(vignette=0., grain=70)
    first = engine.finishing(rgb, settings)
    second = engine.finishing(rgb, settings)
    np.testing.assert_array_equal(first, second)
    assert first.std() > .01
    np.testing.assert_array_equal(first[..., 0], first[..., 1])


@pytest.mark.parametrize('shape', [(80, 160, 3), (160, 80, 3), (100, 100, 3)])
@pytest.mark.parametrize('angle', [-15., 3., 15.])
def test_straightening_has_no_black_edges(shape, angle):
    rgb = np.ones(shape, dtype=np.float32) * .7
    edits = model.recipe()
    edits['straighten'] = angle
    out = engine.crop_rotate(rgb, edits)
    assert out.shape == shape
    np.testing.assert_allclose(out, .7, atol=1e-6)


def test_luminance_range_mask_is_selective():
    m = model.new_mask('luminance', 1)
    m['luminance_range'] = [60., 90.]
    m['range_falloff'] = 10.
    rgb = np.repeat(np.linspace(0, 1, 101, dtype=np.float32)[None, :, None], 3, axis=2)
    alpha = engine.mask_alpha(m, rgb.shape, rgb)
    assert alpha[0, 10] == 0 and alpha[0, 75] == 1 and 0 < alpha[0, 55] < 1
    m['adjustments']['exposure'] = -.7
    edits = model.recipe()
    edits['masks'] = [m]
    out = engine.process(engine.to_linear(rgb), edits)
    np.testing.assert_allclose(out[0, 10], rgb[0, 10], atol=1e-6)
    assert out[0, 75, 0] < rgb[0, 75, 0] - .1


def test_presets_strength_and_framing_preserved():
    r = model.recipe()
    r.update(crop=[.1, .2, .8, .9], rotation=1, straighten=2., wb_gain=[1.1, 1., .8])
    r['masks'].append(model.new_mask('brush', 1))
    preset = model.builtin_presets()[3]
    full = model.apply_look(r, preset['look'], 100)
    half = model.apply_look(full, preset['look'], 50)
    zero = model.apply_look(half, preset['look'], 0)
    for key in ('crop', 'rotation', 'straighten', 'masks', 'wb_gain'):
        assert full[key] == half[key] == zero[key] == r[key]
    assert half['adjustments']['highlights'] == full['adjustments']['highlights'] / 2
    assert zero['effects'] == model.effects() and zero['adjustments'] == model.adjustments()


def test_preset_and_snapshots_persist(tmp_path):
    r = model.apply_look(model.recipe(), model.builtin_presets()[2]['look'])
    r['masks'].append(model.new_mask('luminance', 1))
    r['wb_gain'] = [.8, 1.1, 1.2]
    path = tmp_path / '日落.lumenpreset'
    model.save_preset(path, '日落', r)
    loaded = model.load_preset(path)
    assert loaded['look'] == model.extract_look(r)
    assert 'masks' not in loaded['look'] and 'wb_gain' not in loaded['look']
    snapshots = [dict(name='柔和', edits=copy.deepcopy(r))]
    p = tmp_path / '旅途.lumen'
    model.save_project(p, tmp_path / 'a.arw', r, snapshots)
    _, result, saved_snapshots = model.load_project(p, include_snapshots=True)
    assert result == r and saved_snapshots == snapshots


def test_new_project_validation():
    for key, value in [('wb_gain', [0, 1, 1]), ('straighten', 60), ('grading', dict(shadows=[360, float('nan')]))]:
        r = model.recipe()
        r[key] = value
        with pytest.raises(ValueError):
            model.validate(r)


def test_texture_and_monochrome_have_real_pixel_effects():
    y, x = np.mgrid[0:100, 0:150]
    gray = (.25 + .1 * np.sin(x / 3) + .03 * np.sin(y / 2)).astype(np.float32)
    source = np.stack([gray, gray * .8, gray * .6], axis=2)
    r = model.recipe()
    baseline = engine.process(source, r)
    r['adjustments']['texture'] = 60
    textured = engine.process(source, r)
    assert np.mean(abs(baseline - textured)) > .002
    r['monochrome'] = True
    mono = engine.process(source, r)
    np.testing.assert_allclose(mono[..., 0], mono[..., 2], atol=1e-6)
