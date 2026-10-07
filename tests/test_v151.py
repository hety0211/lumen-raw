"""1.5.1: lens profile corrections from the lensfun database and the interface languages."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lumen import engine, i18n, large_image, lens, model
from lumen.engine import Area

ROOT = Path(__file__).resolve().parents[1]


def exif(**values):
    base = dict(make='', model='', lens_make='', lens_model='', lens='', lens_id='', focal=None, aperture=None,
                distance=None, crop35=None)
    base.update(values)
    return base


# ----------------------------------------------------------------------------- matching

@pytest.mark.parametrize('tags,maker,model_name,crop', [
    # The four lenses of the RAW test samples (Sony A7R V, Nikon Z8, Canon R5 II, the user's A7 III).
    (exif(make='SONY', model='ILCE-7RM5', lens_model='FE 50mm F1.2 GM', focal=50, aperture=1.2, crop35=1.0),
     'Sony', 'FE 50mm f/1.2 GM', 1.0),
    (exif(make='NIKON CORPORATION', model='NIKON Z 8', lens_make='NIKON', lens_model='NIKKOR Z 14-24mm f/2.8 S',
          focal=24, aperture=8, distance=11.9, crop35=1.0), 'Nikon', 'Nikkor Z 14-24mm f/2.8 S', 1.0),
    (exif(make='Canon', model='Canon EOS R5m2', lens_model='RF135mm F1.8 L IS USM', focal=135, aperture=1.8),
     'Canon', 'Canon RF 135mm F1.8L IS USM', 1.0),
    # APS-C crop mode on a full-frame body: the 35 mm equivalent decides the crop factor.
    (exif(make='SONY', model='ILCE-7M3', lens_model='E 28-200mm F2.8-5.6 A071',
          lens_id='Tamron 28-200mm F2.8-5.6 Di III RXD', focal=28, aperture=5, crop35=1.5),
     'Tamron', 'E 28-200mm F2.8-5.6 A071', 1.5),
])
def test_exif_finds_the_lens_and_crop_factor(tags, maker, model_name, crop):
    match = lens.identify(tags)
    assert match['lens'] == dict(maker=maker, model=model_name)
    assert match['crop'] == pytest.approx(crop)


def test_zoom_entries_need_a_matching_focal_range_and_unknown_lenses_stay_unmatched():
    # Same words, other focal range: never picked.
    assert lens.identify(exif(make='SONY', model='ILCE-7M3', lens_model='FE 24-70mm F2.8 GM II', focal=35))['lens'] \
        == dict(maker='Sony', model='FE 24-70mm f/2.8 GM II')
    assert lens.identify(exif(make='SONY', model='ILCE-7M3', lens_model='FE 24-70mm F2.8 GM', focal=35))['lens'] \
        == dict(maker='Sony', model='FE 24-70mm f/2.8 GM')
    assert lens.identify(exif(make='SONY', model='ILCE-7M3', lens_model='Mystery 13-37mm F9', focal=20))['lens'] is None
    assert lens.identify(exif()) is None


def test_exif_tags_give_crop_and_distance():
    tags = {'Make': 'SONY', 'Model': 'ILCE-7M3', 'LensModel': 'E 28-200mm F2.8-5.6 A071', 'FocalLength': 28,
            'FocalLengthIn35mmFormat': 42, 'FNumber': 5, 'FocusDistance2': 'inf', 'LensID': 'A or B'}
    facts = lens.exif_from_tags(tags)
    assert facts['crop35'] == pytest.approx(1.5) and facts['distance'] is None and facts['lens_id'] == ''


# ----------------------------------------------------------------------------- models

REFERENCE_POINTS = [[0, 0], [300, 0], [599, 399], [200, 100], [10, 200], [450, 330]]


def test_lateral_chromatic_aberration_matches_lensfun():
    """Sigma 50mm f/1.4 DG HSM | A at 50 mm on a 600 × 400 full-frame image, red and blue sample
    positions as computed by lensfun 0.3.4 (lensfunpy) for the same calibration."""
    red = [[0.02095, 0.01397], [300.00024, -0.01710], [598.97943, 398.98630], [199.98814, 99.98808],
           [9.99731, 200.00014], [450.01324, 330.01141]]
    blue = [[-0.02335, -0.01556], [300.00015, 0.01657], [599.02374, 399.01581], [200.01198, 100.01191],
            [10.00119, 200.00014], [449.98776, 329.98935]]
    entry = lens.database().find('Sigma', 'Sigma 50mm f/1.4 DG HSM [A]')
    profile, _ = lens.solve(entry, 50, 2, 1000, 1.0)
    terms = dict(lens.effective(dict(enabled=True, profile=profile)), vignetting=None)
    xs, ys = (np.array([p[i] for p in REFERENCE_POINTS], float) for i in (0, 1))
    positions, _ = lens._sample(terms, xs, ys, 600, 400, 1.0)
    assert np.abs(np.stack(positions[0], -1) - red).max() < .002
    assert np.abs(np.stack(positions[2], -1) - blue).max() < .002


def test_distortion_matches_lensfun_up_to_its_centre_scale():
    """Canon EF 24-105mm f/4L at 24 mm: lensfun 0.3.4 keeps the PTLens centre scale d = 1 - a - b - c,
    current lensfun (and LUMEN) normalise it away; with zoom d the positions agree."""
    reference = [[8.063, 5.365], [300.001, -0.005], [590.946, 393.630], [198.652, 98.636], [15.581, 199.991],
                 [450.068, 330.036]]
    entry = lens.database().find('Canon', 'Canon EF 24-105mm f/4L IS USM')
    profile, _ = lens.solve(entry, 24, 4, 10, 1.0)
    terms = dict(lens.effective(dict(enabled=True, profile=profile)), tca=None, vignetting=None)
    xs, ys = (np.array([p[i] for p in REFERENCE_POINTS], float) for i in (0, 1))
    positions, _ = lens._sample(terms, xs, ys, 600, 400, 1 - 0.017263 + 0.049244)
    assert np.abs(np.stack(positions[0], -1) - reference).max() < .2


def test_profiles_are_independent_of_resolution_and_interpolated_between_calibrations():
    entry = lens.database().find('Canon', 'Canon EF 24-105mm f/4L IS USM')
    at24, _ = lens.solve(entry, 24, 4, 1000, 1.0)
    at28, _ = lens.solve(entry, 28, 4, 1000, 1.0)
    between, _ = lens.solve(entry, 26, 4, 1000, 1.0)
    for i in range(4):
        low, high = sorted((at24['distortion'][i], at28['distortion'][i]))
        assert low - 1e-3 <= between['distortion'][i] <= high + 1e-3
    # A crop body sees the centre of the full-frame calibration: weaker distortion and vignetting.
    crop, _ = lens.solve(entry, 24, 4, 1000, 1.6)
    assert abs(crop['distortion'][1]) < abs(at24['distortion'][1])
    assert abs(crop['vignetting'][0]) < abs(at24['vignetting'][0])
    # A calibration for a smaller sensor does not cover a full-frame image (lensfun's rule).
    aps = next(l for l in lens.database().lenses if l.crop > 1.4 and l.distortion)
    assert lens.solve(aps, aps.distortion[0][0], None, None, 1.0) == (None, 'crop')


def photo(width=320, height=220):
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    grid = ((xx // 16 + yy // 16) % 2) * .6 + .2
    return np.ascontiguousarray(np.stack([grid * (xx / width), grid, grid * (yy / height)], -1), np.float32)


def settings(**changes):
    entry = lens.database().find('Nikon', 'Nikkor Z 14-24mm f/2.8 S')
    profile, _ = lens.solve(entry, 14, 4, 1000, 1.0)
    return {**lens.defaults(), 'enabled': True, 'profile': profile, **changes}


def test_blocks_render_exactly_like_the_whole_frame():
    source = photo()
    s = settings()
    whole = lens.apply(source, s)
    h, w = source.shape[:2]
    for rect in [(0, 0, 64, 48), (100, 30, 260, 200), (250, 150, 320, 220)]:
        block = lens.apply(source, s, Area(w, h, *rect))
        assert np.array_equal(block, whole[rect[1]:rect[3], rect[0]:rect[2]])
    edits = model.recipe()
    edits['lens'] = s
    edits['adjustments'].update(clarity=30, shadows=20)
    edits['retouch'] = [dict(kind='heal', enabled=True, points=[[.4, .4], [.45, .42]], radius=.02, opacity=100, feather=50)]
    full = engine.process(source, edits, apply_crop=False)
    region = engine.process_region(source, edits, None, (90, 60, 230, 180))
    assert np.abs(full[60:180, 90:230] - region).max() < 1e-6


def test_the_corrected_frame_samples_only_inside_the_photo():
    s = settings()
    terms = lens.effective(s)
    zoom = lens.fill_zoom(terms, 320, 220)
    ys, xs = np.mgrid[0:220, 0:320]
    positions, _ = lens._sample(dict(terms, vignetting=None), xs.ravel(), ys.ravel(), 320, 220, zoom)
    for px, py in positions:
        assert px.min() >= -.51 and px.max() <= 319.51 and py.min() >= -.51 and py.max() <= 219.51
    # Barrel distortion is corrected by bowing straight lines back: the edge midpoints move most.
    assert zoom != 1


def test_amounts_switches_and_vignetting():
    source = np.full((120, 180, 3), .25, np.float32)
    off = lens.apply(source, settings(enabled=False))
    assert off is source
    vignette = lens.apply(source, settings(distortion=0, chromatic=False))
    assert vignette[60, 90, 1] == pytest.approx(.25, abs=1e-3) and vignette[0, 0, 1] > .3
    half = lens.apply(source, settings(distortion=0, chromatic=False, vignetting=50))
    assert .25 < half[0, 0, 1] < vignette[0, 0, 1]
    assert lens.key(settings()) != lens.key(settings(distortion=50))


def test_streamed_large_images_correct_the_lens_once():
    source = photo(200, 140)
    edits = model.recipe()
    edits['lens'] = settings()
    edits['adjustments']['exposure'] = .3
    streamed = large_image.process(source, edits, engine.Backend('cpu'), False, 1.)
    direct = engine.process(source, edits, _stream=False, apply_crop=False)
    assert np.abs(streamed - direct).max() < 1e-6


# ----------------------------------------------------------------------------- recipes

def test_profiles_follow_each_photo_and_survive_projects(tmp_path):
    at14 = lens.identify(exif(make='NIKON CORPORATION', model='NIKON Z 8', lens_model='NIKKOR Z 14-24mm f/2.8 S',
                              focal=14, aperture=4))
    at24 = dict(at14, focal=24)
    first, reason = lens.prepare(dict(enabled=True), at14)
    assert reason == '' and first['profile']['focal'] == 14
    again, _ = lens.prepare(first, at14)
    assert again == first                      # reopening keeps the stored profile
    other, _ = lens.prepare(first, at24)       # synchronised to another photo: solved for it
    assert other['profile']['focal'] == 24
    manual, _ = lens.prepare(dict(first, manual=dict(maker='Nikon', model='Nikkor Z 24-70mm f/4 S')), at24)
    assert manual['profile']['model'] == 'Nikkor Z 24-70mm f/4 S'
    nothing, reason = lens.prepare(dict(enabled=True), None)
    assert nothing['profile'] is None and reason == 'noexif'
    edits = model.recipe()
    edits['lens'] = other
    project = tmp_path / 'p.lumen'
    model.save_project(project, tmp_path / 'p.arw', edits)
    assert model.load_project(project)[1]['lens'] == other
    # Projects of earlier versions have no lens entry: corrections off.
    assert model.validate({k: v for k, v in edits.items() if k != 'lens'})['lens'] == lens.defaults()
    broken = copy.deepcopy(edits)
    broken['lens']['profile']['distortion'] = [1e9, 0, 0, 0]
    with pytest.raises(ValueError):
        model.validate(broken)


def test_synchronising_a_look_copies_the_switches_not_the_profile():
    from lumen.lens_panel import LensMixin

    class Window(LensMixin):
        edits = dict(lens=settings(distortion=60, chromatic=False))
    target = dict(lens=dict(lens.defaults(), profile={'model': 'other'}))
    Window().sync_lens(target)
    assert target['lens']['enabled'] and target['lens']['distortion'] == 60 and not target['lens']['chromatic']
    assert target['lens']['profile'] == {'model': 'other'}


# ----------------------------------------------------------------------------- languages

def test_language_codes_normalise_from_settings_installers_and_locales():
    assert i18n.normalize('zhcn') == 'zh_CN' and i18n.normalize('zhtw') == 'zh_TW'
    assert i18n.normalize('zh-Hant-TW') == 'zh_TW' and i18n.normalize('zh-Hans-CN') == 'zh_CN'
    assert i18n.normalize('de-AT') == 'de' and i18n.normalize('pt-BR') is None and i18n.normalize('') is None


def test_macos_takes_the_language_list_of_system_settings(monkeypatch):
    # An app opened from the Finder has no LANG, and Qt then reports the "C" locale.
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.setattr(i18n, 'macos_languages', lambda: ['pt-BR', 'zh-Hans-CN', 'en-CN'])
    assert i18n.system_language() == 'zh_CN'
    monkeypatch.setattr(i18n, 'macos_languages', lambda: [])
    assert i18n.system_language() in i18n.CODES


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS only')
def test_macos_language_list_is_read_from_foundation():
    assert i18n.macos_languages() and all(isinstance(name, str) for name in i18n.macos_languages())


def test_every_source_string_is_translated_with_the_same_fields():
    sys.path.insert(0, str(ROOT / 'tools'))
    import i18n_catalog
    assert i18n_catalog.check() == 0


def test_translation_lookup_formats_and_falls_back(monkeypatch):
    try:
        i18n.select('de')
        assert i18n.tr('打开原片') == 'Original öffnen'
        assert i18n.tr('导出 {count} 张', count=3) == '3 exportieren'
        assert i18n.tr('不在目录里的文本') == '不在目录里的文本'
        i18n.select('zh_CN')
        assert i18n.tr('导出 {count} 张', count=3) == '导出 3 张'
    finally:
        i18n.select('zh_CN')
    assert i18n.tr_in('ja', '立即重新启动') == '今すぐ再起動'


def test_settings_file_is_shared_with_the_installer(tmp_path):
    path = tmp_path / 'settings.ini'
    path.write_text('[General]\nlanguage=fr\n', encoding='ascii')   # as Inno Setup's [INI] writes it
    assert i18n.setting('language', path=path) == 'fr'
    i18n.save_setting('lens_auto_enable', '1', path=path)
    assert i18n.setting('language', path=path) == 'fr' and i18n.setting('lens_auto_enable', path=path) == '1'
    assert 'language=fr' in path.read_text(encoding='utf-8')


def test_installer_offers_the_same_languages_and_writes_the_choice():
    iss = (ROOT / 'installer.iss').read_text(encoding='utf-8')
    names = re.findall(r'^Name: "(\w+)"; MessagesFile', iss, re.M)
    assert sorted(i18n.normalize(n) for n in names) == sorted(i18n.CODES)
    assert 'Key: "language"; String: "{language}"' in iss
    assert '{cm:CreateDesktopIcon}' in iss and '{cm:LaunchProgram,LUMEN RAW}' in iss


def test_a_fresh_process_starts_in_the_saved_language(tmp_path):
    env = dict(os.environ, LOCALAPPDATA=str(tmp_path), PYTHONIOENCODING='utf-8')
    env.pop('LUMEN_LANGUAGE', None)
    folder = tmp_path
    if sys.platform == 'darwin':  # ~/Library/Application Support there, not LOCALAPPDATA
        env['HOME'] = str(tmp_path)
        folder = tmp_path / 'Library' / 'Application Support'
    (folder / 'LUMEN RAW').mkdir(parents=True)
    (folder / 'LUMEN RAW' / 'settings.ini').write_text('[General]\nlanguage=ko\n', encoding='ascii')
    code = 'from lumen import i18n, model; print(i18n.language(), model.COLORS[0][0])'
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env, capture_output=True, text=True,
                         encoding='utf-8', timeout=60)
    assert out.stdout.split() == ['ko', '빨강'], out.stderr


# ----------------------------------------------------------------------------- window

from test_ui import app, window, wait_until  # noqa: E402,F401


def test_language_menu_lists_every_language_and_marks_the_current_one(window):
    menu = window.language_menu
    assert menu.title() == '语言 / Language'
    actions = menu.actions()
    assert [a.text() for a in actions] == [name for _, name in i18n.LANGUAGES]
    assert [a.text() for a in actions if a.isChecked()] == ['简体中文']


def test_tab_strip_wraps_the_panels_and_follows_the_selection(window):
    strip = window.tab_strip
    assert [b.text() for b in strip.buttons] == [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert window.tabs.tabText(5) == '裁切·镜头'
    strip.buttons[5].click()
    assert window.tabs.currentIndex() == 5
    window.tabs.setCurrentIndex(2)
    assert strip.buttons[2].isChecked()


def test_lens_panel_detects_enables_and_renders(window):
    w = window
    w.info['lens_match'] = lens.identify(exif(make='NIKON CORPORATION', model='NIKON Z 8',
                                              lens_model='NIKKOR Z 14-24mm f/2.8 S', focal=14, aperture=4))
    w.init_lens_states([w.edits], False)   # as opening a photo with this EXIF does
    w.refresh_lens()
    assert '14-24mm' in w.lens_status.text() and not w.lens_enable.isChecked()
    before = w.rendered.copy()
    w.lens_enable.setChecked(True)
    assert w.edits['lens']['enabled'] and w.edits['lens']['profile']['focal'] == 14
    wait_until(lambda: w.rendered is not None and not w.render_running and not w.timer.isActive()
               and not np.array_equal(w.rendered, before))
    assert w.lens_controls['distortion'].isEnabled()
    w.undo(-1)
    assert not w.edits['lens']['enabled']
