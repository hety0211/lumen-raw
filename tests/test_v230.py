"""2.3.0: the crop tool on the photo (fixed frame, flips, straightening to 45°, automatic level),
eight panels with colour grading in 色彩 and lens corrections in 细节, the edit history list,
automatic exposure and colour, and portrait retouching (faces, skin, eyes, teeth, face shape)."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest

from lumen import auto_adjust, culling, engine, geometry, history, model, portrait, straighten
from lumen.crop_tool import View, clean, fill_scale, geometry_of
from test_ui import app, wait_until, window  # noqa: E402,F401

SAMPLES = Path(os.environ.get('LUMEN_SAMPLES', r'E:\LumenSamples'))


def synthetic_face(size=512):
    """A drawn face YuNet and Face Mesh both take for one (no photograph needed)."""
    img = np.full((size, size, 3), 110, np.uint8)
    c = size // 2
    cv2.ellipse(img, (c, c + 10), (int(size * .27), int(size * .36)), 0, 0, 360, (224, 172, 140), -1, cv2.LINE_AA)
    for dx in (-1, 1):
        eye = (c + dx * int(size * .11), c - int(size * .05))
        cv2.ellipse(img, eye, (int(size * .06), int(size * .028)), 0, 0, 360, (250, 250, 250), -1, cv2.LINE_AA)
        cv2.circle(img, eye, int(size * .024), (60, 40, 30), -1, cv2.LINE_AA)
        cv2.line(img, (c + dx * int(size * .06), c - int(size * .12)), (c + dx * int(size * .17), c - int(size * .13)),
                 (70, 50, 40), 6, cv2.LINE_AA)
    cv2.line(img, (c, c - int(size * .02)), (c - 8, c + int(size * .08)), (190, 140, 115), 4, cv2.LINE_AA)
    cv2.ellipse(img, (c, c + int(size * .17)), (int(size * .09), int(size * .03)), 0, 0, 360, (180, 70, 80), -1, cv2.LINE_AA)
    return cv2.GaussianBlur(img, (0, 0), 1.5)


@pytest.fixture(scope='module')
def face_photo():
    if culling.faces() is None:
        pytest.skip('face models not available')
    rgb = synthetic_face()
    # Skin texture and a few blemishes, so smoothing and acne have something to do.
    noise = np.random.default_rng(5).normal(0, 6, rgb.shape[:2])[..., None]
    rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    faces = portrait.detect(rgb)
    assert len(faces) == 1
    return rgb, faces


# ------------------------------------------------------------------------------- recipe and engine

def test_new_recipes_are_version_9_and_older_projects_still_open(tmp_path):
    r = model.recipe()
    assert r['version'] == 9 and r['flip'] == [False, False] and r['portrait'] == dict(faces=[])
    old = {k: v for k, v in r.items() if k not in ('flip', 'portrait')}
    old.update(version=8, straighten=12.)
    checked = model.validate(old)
    assert checked['flip'] == [False, False] and checked['portrait'] == dict(faces=[]) and checked['straighten'] == 12
    assert model.validate(dict(r, straighten=44.5))['straighten'] == 44.5
    for bad in (dict(r, straighten=46.), dict(r, flip=[1, 0]), dict(r, flip=[True])):
        with pytest.raises(ValueError):
            model.validate(bad)


def test_flips_mirror_the_output_everywhere(tmp_path):
    rng = np.random.default_rng(1)
    image = rng.random((60, 90, 3)).astype(np.float32)
    edits = dict(model.recipe(), crop=[.1, .2, .9, .8], rotation=1, straighten=4.)
    plain = engine.crop_rotate(image, edits)
    np.testing.assert_array_equal(engine.crop_rotate(image, dict(edits, flip=[True, False])), plain[:, ::-1])
    np.testing.assert_array_equal(engine.crop_rotate(image, dict(edits, flip=[False, True])), plain[::-1])
    flipped = dict(edits, flip=[True, True])
    whole = engine.crop_rotate(image, flipped)
    # Blocks of the displayed output (detail tiles) come from the right place.
    def render(rect):
        return image[rect[1]:rect[3], rect[0]:rect[2]]
    block = engine.display_block(render, flipped, (90, 60), (3, 5, 20, 30))
    np.testing.assert_allclose(block, whole[5:30, 3:20], atol=1e-5)
    # The canvas mapping (geometry.frame) puts a source point where crop_rotate puts its pixel.
    marked = np.zeros((60, 90, 3), np.float32)
    marked[30, 40] = 1
    out = engine.crop_rotate(marked, dict(flipped, straighten=0.))
    y, x = np.unravel_index(np.argmax(out[..., 0]), out.shape[:2])
    matrix, size, _ = geometry.frame(dict(flipped, straighten=0.), (90, 60))
    px, py = geometry.point(matrix, [40 / 89, 30 / 59])
    assert abs(px * (size[0] - 1) - x) < 1.01 and abs(py * (size[1] - 1) - y) < 1.01


@pytest.mark.parametrize('angle,turns,flip', [(0., 0, [False, False]), (7.5, 0, [False, False]), (-12., 1, [True, False]),
                                              (30., 3, [False, True]), (-44., 2, [True, True])])
def test_the_crop_view_shows_exactly_what_the_crop_keeps(angle, turns, flip):
    yy, xx = np.mgrid[0:300, 0:450].astype(np.float32)
    image = np.stack([xx / 450, yy / 300, .5 + .5 * np.sin(xx / 23) * np.cos(yy / 17)], axis=2).astype(np.float32)
    edits = dict(model.recipe(), straighten=angle, rotation=turns, flip=flip, crop=[.25, .2, .7, .75])
    view = View(edits, (450, 300))
    frame = view.frame(edits['crop'])
    np.testing.assert_allclose(view.crop_for(frame), edits['crop'], atol=1e-9)
    shown = view.warp(image)
    x0, y0, x1, y1 = [round(v) for v in frame]
    inside = shown[y0 + 2:y1 - 2, x0 + 2:x1 - 2]
    kept = engine.crop_rotate(image, edits)
    kept = cv2.resize(kept, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)[2:-2, 2:-2]
    assert inside.shape == kept.shape
    assert float(np.abs(inside - kept).mean()) < .02


def test_a_moved_frame_stops_at_the_photo_and_a_large_one_shrinks_to_fit():
    edits = dict(model.recipe(), straighten=10., crop=[.3, .3, .6, .6])
    view = View(edits, (400, 300))
    moved = view.moved(edits['crop'], -10_000, 0)
    assert moved[0] == 0 and abs((moved[2] - moved[0]) - .3) < 1e-9
    huge = [-500, -500, 2000, 2000]
    fitted = view.shrink(huge, (200, 150))
    assert fitted is not None and View.fits(view.crop_for(fitted))
    assert clean([0, 0, 1, 1]) is None and clean([-.001, .2, .5, 1.0001]) == [0.0, 0.2, 0.5, 1.0]
    assert fill_scale(0, 3, 2) == 1


def test_automatic_level_finds_a_tilted_horizon_and_keeps_level_photos_level():
    scene = np.full((800, 1200, 3), .65, np.float32)
    scene[430:] = .22
    for x in (300, 520, 800):
        scene[160:430, x:x + 40] = .92
    tilted = cv2.warpAffine(scene, cv2.getRotationMatrix2D((600, 400), -3.5, 1.2), (1200, 800), borderMode=cv2.BORDER_REFLECT)
    angle, confidence = straighten.estimate(tilted)
    assert abs(angle - 3.5) < .25 and confidence > .5
    level = engine.crop_rotate(tilted, dict(model.recipe(), straighten=angle))
    assert abs(straighten.estimate(level)[0]) < .2
    noise = np.random.default_rng(0).random((400, 600, 3)).astype(np.float32)
    assert straighten.estimate(cv2.GaussianBlur(noise, (0, 0), 4))[0] is None


def test_auto_white_balance_sliders_cancel_the_light():
    from lumen import tone
    for light in ([1.3, 1., .7], [.8, 1., 1.25], [1.1, 1., 1.1]):
        temperature, tint = auto_adjust.white_balance_sliders(np.array(light))
        gains = tone.wb_gains(dict(temperature=temperature, tint=tint)) if hasattr(tone, 'wb_gains') else np.array(
            [2 ** (.4 * temperature / 100 + .15 * tint / 100), 2 ** (-.15 * tint / 100),
             2 ** (-.4 * temperature / 100 + .15 * tint / 100)])
        neutral = np.array(light) * gains
        np.testing.assert_allclose(neutral / neutral[1], 1, atol=1e-6)


def test_auto_adjust_brightens_a_dark_photo_and_warms_a_cold_cast_gently():
    yy, xx = np.mgrid[0:240, 0:360].astype(np.float32)
    scene = np.stack([.3 + .5 * xx / 360, .25 + .4 * yy / 240, .2 + .3 * np.sin(xx / 30) ** 2], axis=2)
    grey = np.repeat(((xx // 40 + yy // 40) % 2 * .5 + .2)[..., None], 3, axis=2)
    scene[:, :120] = grey[:, :120]
    dark = (scene * .08 * np.array([.85, 1., 1.25], np.float32)).astype(np.float32)
    edits = model.recipe()
    # The rules (the fallback without a fitted model) warm the blue cast, gently.
    rules = auto_adjust.suggest(dark, edits, use_model=False)
    assert set(rules) == set(auto_adjust.KEYS)
    assert rules['exposure'] > 1 and rules['temperature'] > 0
    assert -25 <= rules['temperature'] <= 30 and -20 <= rules['tint'] <= 20
    # The shipped model brightens it as much, and is deterministic.
    assert auto_adjust.model() is not None
    values = auto_adjust.suggest(dark, edits)
    assert set(values) == set(auto_adjust.KEYS) and values['exposure'] > 1
    after = engine.process(dark, dict(edits, adjustments=dict(edits['adjustments'], **values)))
    before = engine.process(dark, edits)
    assert np.median(after) > np.median(before) * 1.8
    assert auto_adjust.suggest(dark, edits) == values
    assert len(auto_adjust.features(dark, edits)) == len(auto_adjust.FEATURES)


def test_the_shipped_auto_model_matches_the_features_and_stays_in_range():
    fitted = auto_adjust.model()
    assert fitted['format'] == 1 and fitted['photos'] >= 200
    assert fitted['features'] == list(auto_adjust.FEATURES)
    assert set(fitted['targets']) == {'median'} | set(auto_adjust.KEYS) - {'exposure'}
    for target in fitted['targets'].values():
        assert len(target['coef']) == len(auto_adjust.FEATURES)
    wild = auto_adjust.predict(fitted, np.full(len(auto_adjust.FEATURES), 1e6))
    for target, value in wild.items():
        low, high = fitted['targets'][target]['range']
        assert low <= value <= high


def test_the_training_tool_learns_from_edited_projects(tmp_path):
    """tools/train_auto_adjust.py on 36 synthetic photos whose 'editor' always lifted dark photos."""
    import subprocess
    import sys
    rng = np.random.default_rng(11)
    yy, xx = np.mgrid[0:80, 0:120].astype(np.float32)
    for i in range(36):
        level = rng.uniform(.05, .6)
        rgb = np.stack([xx / 120, yy / 80, .5 + .4 * np.sin(xx / 9)], axis=2) * level + .02
        path = tmp_path / f'{i:02d}.png'
        Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)
        edits = model.recipe()
        edits['adjustments'].update(exposure=round(float(np.log2(.3 / level)), 2), contrast=10., vibrance=8.)
        model.save_project(path.with_suffix('.lumen'), path, edits)
    out = tmp_path / 'model.json'
    report = tmp_path / 'report.json'
    tool = Path(__file__).resolve().parents[1] / 'tools' / 'train_auto_adjust.py'
    result = subprocess.run([sys.executable, str(tool), str(tmp_path), '--out', str(out), '--report', str(report),
                             '--workers', '1', '--folds', '4'], capture_output=True, timeout=600)
    assert result.returncode == 0, result.stderr.decode('utf-8', 'replace')[-2000:]
    fitted = json.loads(out.read_text(encoding='utf-8'))
    summary = json.loads(report.read_text(encoding='utf-8'))
    assert fitted['photos'] == 36 and fitted['features'] == list(auto_adjust.FEATURES)
    # The median the edits reach is predictable from the photo, far better than its average.
    median = summary['targets']['median']
    assert median['model_mae'] < median['mean_mae'] * .6
    assert abs(fitted['targets']['contrast']['intercept'] - 10) < 1


def test_history_names_each_step_and_jumps_back():
    r = model.recipe()
    h = model.History(r)
    a = copy.deepcopy(r)
    a['adjustments']['exposure'] = .8
    h.push(a)
    b = copy.deepcopy(a)
    b['masks'].append(model.new_mask('radial', 1))
    h.push(b)
    c = copy.deepcopy(b)
    c['crop'] = [.1, .1, .9, .9]
    h.push(c, '裁切')
    assert h.labels == ['打开照片', '曝光 EV +0.80', '添加蒙版：径向渐变 1', '裁切']
    assert h.goto(1) == a and h.index == 1
    d = copy.deepcopy(a)
    d['flip'] = [True, False]
    h.push(d)
    assert h.labels == ['打开照片', '曝光 EV +0.80', '翻转'] and len(h.items) == 3
    e = copy.deepcopy(d)
    e['portrait'] = dict(faces=[dict(box=[0, 0, .1, .1], points=[[0, 0]] * 478, settings=dict(portrait.settings(), smooth=40.))])
    assert history.describe(d, e) == '检测人脸' or history.describe(d, e).startswith('人像')
    f = copy.deepcopy(e)
    f['portrait']['faces'][0]['settings']['slim'] = 30.
    assert history.describe(e, f) == '人像 · 瘦脸 30'
    assert history.describe(f, dict(f, straighten=2.5)) == '拉直 +2.5°'


# ------------------------------------------------------------------------------------- portrait

def _linear(rgb):
    return engine.to_linear(rgb.astype(np.float32) / 255)


def test_faces_are_found_with_landmarks_and_stored_in_the_recipe(face_photo, tmp_path):
    rgb, faces = face_photo
    face = faces[0]
    assert len(face['points']) == 478 and face['settings'] == portrait.settings()
    x, y, w, h = face['box']
    assert .15 < x < .35 and .1 < y < .3 and w > .4 and h > .5
    edits = dict(model.recipe(), portrait=dict(faces=[dict(face, settings=dict(face['settings'], smooth=50.))]))
    project = tmp_path / 'face.lumen'
    model.save_project(project, tmp_path / 'face.png', edits)
    _, loaded = model.load_project(project)
    assert loaded['portrait'] == model.validate(edits)['portrait']


def test_faces_found_again_keep_their_settings_by_box():
    a = dict(box=[.1, .1, .2, .2], points=[[0, 0]] * 478, settings=dict(portrait.settings(), whiten=30.))
    b = dict(box=[.6, .1, .2, .2], points=[[0, 0]] * 478, settings=dict(portrait.settings(), slim=50.))
    found = [dict(b, box=[.61, .11, .2, .2], settings=portrait.settings()), dict(a, box=[.09, .1, .21, .2], settings=portrait.settings()),
             dict(a, box=[.4, .7, .1, .1], settings=portrait.settings())]
    merged = portrait.merge([a, b], found)
    assert [f['settings']['slim'] for f in merged] == [50., 0., 0.]
    assert [f['settings']['whiten'] for f in merged] == [0., 30., 0.]
    assert portrait.face_at(merged, (.7, .2)) == 0 and portrait.face_at(merged, (.9, .9)) is None


def test_portrait_retouching_stays_inside_the_face_window(face_photo):
    rgb, faces = face_photo
    linear = _linear(rgb)
    face = copy.deepcopy(faces[0])
    assert portrait.apply(linear, [face], linear.shape) is linear            # nothing on: untouched
    face['settings'].update(smooth=70., whiten=60., eyes=60., slim=60., circles=50., teeth=60., acne=50.)
    out = portrait.apply(linear, [face], linear.shape)
    x0, y0, x1, y1 = portrait.window(face, *linear.shape[:2])
    outside = np.ones(linear.shape[:2], bool)
    outside[y0:y1, x0:x1] = False
    np.testing.assert_array_equal(out[outside], linear[outside])
    # The warp leaves the window's border where it was (no seam).
    np.testing.assert_allclose(out[y0:y1, x0:x1][[0, -1]], linear[y0:y1, x0:x1][[0, -1]], atol=1e-6)
    assert np.isfinite(out).all()


def test_whitening_and_smoothing_work_on_the_skin(face_photo):
    rgb, faces = face_photo
    linear = _linear(rgb)
    h, w = linear.shape[:2]
    points = portrait.pixels(faces[0], h, w)
    cheek = points[50].round().astype(int)
    patch = (slice(cheek[1] - 12, cheek[1] + 12), slice(cheek[0] - 12, cheek[0] + 12))

    def run(**values):
        face = copy.deepcopy(faces[0])
        face['settings'].update(values)
        return portrait.apply(linear, [face], linear.shape)
    whiter = run(whiten=80.)
    assert whiter[patch].mean() > linear[patch].mean() * 1.15
    smoother = run(smooth=80.)
    texture = lambda img: (img[patch] @ np.array([.2126, .7152, .0722], np.float32)).std()
    assert texture(smoother) < texture(linear) * .7
    background = (slice(5, 30), slice(5, 30))
    np.testing.assert_allclose(whiter[background], linear[background], atol=1e-6)


def test_bigger_eyes_and_a_slimmer_face_move_the_features(face_photo):
    rgb, faces = face_photo
    linear = _linear(rgb)
    face = copy.deepcopy(faces[0])
    face['settings'].update(eyes=100.)
    out = portrait.apply(linear, [face], linear.shape)
    eye = portrait.pixels(face, *linear.shape[:2])[468]
    row = int(round(eye[1]))
    dark = lambda img: (img[row, :, 0] < .05).sum()
    assert dark(out) > dark(linear)                                           # the iris is wider
    face['settings'].update(eyes=0., slim=100.)
    out = portrait.apply(linear, [face], linear.shape)
    jaw = int(round(portrait.pixels(face, *linear.shape[:2])[172][1]))
    skin = lambda img: (img[jaw, :, 0] > .3).sum()
    assert skin(out) < skin(linear)                                           # the jaw is narrower


def test_portrait_blocks_match_the_whole_frame(face_photo):
    rgb, faces = face_photo
    source = _linear(rgb)
    face = copy.deepcopy(faces[0])
    face['settings'].update(smooth=60., whiten=40., acne=40., eyes=50., slim=50., teeth=50., circles=40.)
    edits = dict(model.recipe(), portrait=dict(faces=[face]))
    edits['adjustments']['clarity'] = 20
    whole = engine.process(source, edits, apply_crop=False)
    for rect in ((0, 0, 512, 512), (180, 200, 300, 330), (10, 400, 140, 512)):
        block = engine.process_region(source, edits, rect=rect)
        np.testing.assert_allclose(block, whole[rect[1]:rect[3], rect[0]:rect[2]], atol=2e-4)


def test_large_images_take_the_whole_frame_path_with_faces():
    from lumen import large_image
    face = dict(box=[.1, .1, .2, .2], points=[[.2, .2]] * 478, settings=dict(portrait.settings(), smooth=10.))
    edits = dict(model.recipe(), portrait=dict(faces=[face]))
    assert portrait.active(edits['portrait'])
    assert 'portrait.active' in Path(large_image.__file__).read_text(encoding='utf-8')


@pytest.mark.skipif(not (SAMPLES / 'portraits').is_dir(), reason='portrait test photos not available')
def test_faces_in_the_test_portraits():
    found = {}
    for name in ('korean-woman.jpg', 'studio-portrait.jpg', 'group-women.jpg', 'elderly-woman.jpg'):
        bgr = cv2.imdecode(np.fromfile(str(SAMPLES / 'portraits' / name), np.uint8), cv2.IMREAD_COLOR)
        scale = 1600 / max(bgr.shape[:2])
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        found[name] = len(portrait.detect(np.ascontiguousarray(bgr[..., ::-1])))
    assert found['korean-woman.jpg'] == 1 and found['studio-portrait.jpg'] == 1
    assert found['elderly-woman.jpg'] == 1                     # fills the frame: found by the coarse look
    assert found['group-women.jpg'] >= 4


# ------------------------------------------------------------------------------------------- the window

def test_eight_panels_in_one_row_with_grading_in_colour_and_lens_in_detail(window):
    w = window
    assert w.tab_keys == ['basic', 'color', 'details', 'masks', 'portrait', 'effects', 'retouch', 'watermark']
    color = w.tabs.widget(w.tab_index('color'))
    assert color.isAncestorOf(w.wheels['shadows']) and color.isAncestorOf(w.curve) and color.isAncestorOf(w.mono_check)
    details = w.tabs.widget(w.tab_index('details'))
    assert details.isAncestorOf(w.lens_enable) and not w.lens_section.content.isVisible()
    w.show_tab('details')
    w.lens_section.header.click()
    assert w.lens_section.content.isVisible()
    rows = {button.geometry().top() for button in w.tab_strip.buttons}
    assert len(rows) == 1
    assert [w.library_tabs.tabText(i) for i in range(w.library_tabs.count())] == ['预设', '自然语言输入', '历史']


def _drag(canvas, a, b, steps=6):
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=a)
    for i in range(1, steps + 1):
        QTest.mouseMove(canvas, QPoint(round(a.x() + (b.x() - a.x()) * i / steps), round(a.y() + (b.y() - a.y()) * i / steps)))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=b)


def test_crop_tool_moves_the_photo_under_a_fixed_frame(window):
    w = window
    w.canvas.setFocus()
    QTest.keyClick(w.canvas, Qt.Key.Key_R)
    assert w.crop_active and w.canvas.tool == 'crop' and w.crop_bar.isVisible() and not w.final_view.isEnabled()
    # A new frame on a photo not cropped yet: drawn by dragging.
    _drag(w.canvas, w.canvas.screen([.3, .3]).toPoint(), w.canvas.screen([.6, .7]).toPoint())
    crop = w.edits['crop']
    assert crop is not None and abs(crop[0] - .3) < .02 and abs(crop[3] - .7) < .02
    tool = w.crop_tool
    frame = tool.frame()
    before = (tool.to_screen(w.canvas, frame[:2]), tool.to_screen(w.canvas, frame[2:]))
    centre = (before[0] + before[1]) / 2
    _drag(w.canvas, centre.toPoint(), (centre + QPointF(40, 25)).toPoint())
    moved = w.edits['crop']
    assert moved[0] < crop[0] - .05 and moved[1] < crop[1] - .05             # the photo moved, not the frame
    frame = tool.frame()
    after = (tool.to_screen(w.canvas, frame[:2]), tool.to_screen(w.canvas, frame[2:]))
    for p, q in zip(before, after):
        assert abs(p.x() - q.x()) < 1.5 and abs(p.y() - q.y()) < 1.5
    # A corner resizes the frame; the ratio menu keeps 1 : 1.
    w.crop_bar.ratio.setCurrentIndex(2)
    frame = tool.frame()
    assert abs((frame[2] - frame[0]) - (frame[3] - frame[1])) < 1
    corner = tool.to_screen(w.canvas, frame[2:]).toPoint()
    _drag(w.canvas, corner, corner + QPoint(-30, -10))
    frame = tool.frame()
    assert abs((frame[2] - frame[0]) - (frame[3] - frame[1])) < 1.5
    # Esc gives everything back; R again and Enter keep it.
    QTest.keyClick(w.canvas, Qt.Key.Key_Escape)
    assert not w.crop_active and w.edits['crop'] is None
    w.start_crop()
    w.flip_image(0)
    w.rotate(-1)
    w.crop_bar.angle.spin.setValue(20)
    assert w.edits['straighten'] == 20 and w.edits['rotation'] == 3 and w.edits['flip'] == [True, False]
    QTest.keyClick(w.canvas, Qt.Key.Key_Return)
    assert not w.crop_active and w.final_view.isChecked() and w.history.labels[-1] == '裁切'
    wait_until(lambda: not w.render_running and not w.timer.isActive())
    assert w.canvas.image.width() == engine.crop_rotate(w.rendered, w.edits).shape[1]


def test_r_crops_only_in_the_editor_and_the_library_keeps_a_crop_in_progress(window):
    w = window
    w.start_crop()
    w.edits['crop'] = [.2, .2, .7, .7]
    w.set_mode('library')
    assert not w.crop_active and w.edits['crop'] == [.2, .2, .7, .7] and not w.crop_bar.isVisible()
    w.toggle_crop()
    assert not w.crop_active
    w.set_mode('develop')
    w.toggle_crop()
    assert w.crop_active
    w.finish_crop(False)
    assert w.edits['crop'] == [.2, .2, .7, .7]


def test_straightening_along_a_line_and_the_automatic_level(window, tmp_path):
    w = window
    scene = np.full((400, 600, 3), 170, np.uint8)
    scene[215:] = 60
    tilted = cv2.warpAffine(scene, cv2.getRotationMatrix2D((300, 200), -4, 1.25), (600, 400), borderMode=cv2.BORDER_REFLECT)
    path = tmp_path / '海平线.png'
    Image.fromarray(tilted).save(path)
    w.open_path(str(path))
    wait_until(lambda: not w.loading and w.rendered is not None and not w.render_running)
    w.start_crop()
    w.auto_level()
    wait_until(lambda: w.crop_bar.auto.isEnabled() and not w.jobs)
    assert abs(w.edits['straighten'] - 4) < .4 and abs(w.crop_bar.angle.spin.value() - w.edits['straighten']) < .06
    assert w.history.labels[-1].startswith('自动水平')
    w.crop_bar.angle.spin.setValue(0)
    w.crop_bar.line.setChecked(True)
    view = w.crop_view
    a = view.to_view((100, 200 + math.tan(math.radians(6)) * -200))
    b = view.to_view((500, 200 + math.tan(math.radians(6)) * 200))
    vw, vh = view.size
    _drag(w.canvas, w.canvas.screen(view.normalised()[0]([a[0] / vw, a[1] / vh])).toPoint(),
          w.canvas.screen(view.normalised()[0]([b[0] / vw, b[1] / vh])).toPoint())
    assert abs(w.edits['straighten'] - 6) < .6 and not w.crop_bar.line.isChecked()
    w.finish_crop(True)


def test_history_list_shows_steps_and_goes_back(window):
    w = window
    w.controls['exposure'].spin.setValue(.5)
    w.commit()
    w.controls['contrast'].spin.setValue(20)
    w.commit()
    w.add_mask('radial')
    labels = [w.history_list.item(i).text().split('  ', 1)[1] for i in range(w.history_list.count())]
    assert labels[:3] == ['添加蒙版：径向渐变 1', '对比度 +20', '曝光 EV +0.50'] and labels[-1] == '打开照片'
    item = w.history_list.item(2)
    w.history_list.itemClicked.emit(item)
    assert w.edits['adjustments']['exposure'] == .5 and w.edits['adjustments']['contrast'] == 0 and not w.edits['masks']
    assert w.controls['contrast'].spin.value() == 0
    assert w.history_list.currentItem().text().endswith('曝光 EV +0.50')
    w.snapshot_from_history(2)
    assert w.snapshots[-1]['name'] == '对比度 +20' and w.snapshots[-1]['edits']['adjustments']['contrast'] == 20
    w.controls['clarity'].spin.setValue(10)
    w.commit()
    assert [w.history.labels[i] for i in range(len(w.history.labels))][-2:] == ['曝光 EV +0.50', '清晰度 +10']


def test_auto_button_sets_editable_sliders_in_the_background(window, monkeypatch):
    w = window
    calls = []
    real = auto_adjust.suggest
    monkeypatch.setattr(auto_adjust, 'suggest', lambda source, edits: calls.append(1) or real(source, edits))
    w.auto_button.click()
    wait_until(lambda: w.auto_button.isEnabled() and not w.jobs)
    assert calls and w.history.labels[-1] == '自动调整'
    for key in auto_adjust.KEYS:
        assert w.controls[key].spin.value() == pytest.approx(w.edits['adjustments'][key], abs=.01)


def test_portrait_panel_finds_numbers_selects_and_retouches_faces(window, tmp_path, face_photo):
    w = window
    rgb, _ = face_photo
    pair = np.concatenate([rgb, rgb[:, ::-1]], axis=1)
    path = tmp_path / '两张脸.png'
    Image.fromarray(pair).save(path)
    w.open_path(str(path))
    wait_until(lambda: not w.loading and w.rendered is not None and not w.render_running)
    w.show_tab('portrait')
    wait_until(lambda: not w.portrait_running and not w.jobs)
    assert len(w.faces()) == 2 and w.face_selection == {0, 1}
    assert [c.text() for c in w.face_chips] == ['全部', '1', '2'] and w.canvas.tool == 'face'
    assert [f['number'] for f in w.canvas.faces] == [1, 2] and w.history.labels[-1] == '检测人脸'
    # A click on face 2 chooses it alone; the sliders act on it only.
    box = w.faces()[1]['box']
    QTest.mouseClick(w.canvas, Qt.MouseButton.LeftButton, pos=w.canvas.screen([box[0] + box[2] / 2, box[1] + box[3] / 2]).toPoint())
    assert w.face_selection == {1}
    w.portrait_controls['whiten'].spin.setValue(45)
    w.commit()
    assert w.faces()[1]['settings']['whiten'] == 45 and w.faces()[0]['settings']['whiten'] == 0
    assert w.history.labels[-1] == '人像 · 美白 45'
    w.face_chips[0].click()
    assert w.face_selection == {0, 1}
    w.portrait_controls['slim'].spin.setValue(30)
    assert all(f['settings']['slim'] == 30 for f in w.faces())
    wait_until(lambda: not w.render_running and not w.timer.isActive())
    # Faces stay with the photo's recipe and keep their settings when found again.
    w.detect_faces()
    wait_until(lambda: not w.portrait_running and not w.jobs)
    assert w.faces()[1]['settings']['whiten'] == 45
    w.show_tab('basic')
    assert w.canvas.faces == []


# ------------------------------------------------------------------------------- memory (2.3.0 review)

def test_available_memory_is_the_smaller_of_free_memory_and_remaining_commit():
    from lumen import large_image
    status = large_image.memory_status()
    if status is None:
        pytest.skip('Windows only')
    available = large_image.available_memory()
    assert available <= status['physical_available'] + 2 ** 26 and available <= status['commit_available'] + 2 ** 26


def test_running_out_of_memory_is_explained_with_the_real_numbers():
    from lumen import large_image
    assert not large_image.out_of_memory(ValueError('无效裁切。'))
    assert large_image.explain(ValueError('无效裁切。')) == '无效裁切。'
    for error in (MemoryError(), MemoryError('Unable to allocate 1.57 MiB for an array with shape (114, 1200, 3)'),
                  cv2.error('OpenCV(5.0.0) alloc.cpp:73: error: (-4:Insufficient memory) Failed to allocate 1179648 bytes')):
        assert large_image.out_of_memory(error)
        text = large_image.explain(error)
        assert text.startswith('内存不足')
        if large_image.memory_status():
            assert '已提交' in text and '页面文件' in text and 'GB' in text
    gpu = RuntimeError('DmlExecutionProvider: 887A0005 E_OUTOFMEMORY while allocating a buffer')
    assert large_image.explain(gpu).startswith('显卡内存不足') and 'CPU 模式' in large_image.explain(gpu)


def test_a_failed_job_reports_running_out_of_memory_in_plain_words(app):
    from lumen.scheduler import Job
    messages = []
    def fail():
        raise MemoryError('Unable to allocate 698. MiB for an array with shape (6374, 9566, 3)')
    job = Job(fail)
    job.signals.failed.connect(messages.append)
    job.run()
    assert messages and messages[0].startswith('内存不足') and '698. MiB' in messages[0]


def test_strips_hold_a_bounded_block_and_give_the_same_values():
    seen = []
    image = np.random.default_rng(2).random((2400, 3000, 3)).astype(np.float32)
    def stage(block, y0, y1):
        seen.append(block.shape[0] * block.shape[1])
        return block * 2 + y0 * 0
    out = engine.in_strips(image, stage)
    np.testing.assert_array_equal(out, image * 2)
    assert max(seen) <= engine.STRIP_WORK + 3000 and len(seen) >= image.shape[0] * image.shape[1] // engine.STRIP_WORK


def test_the_detail_tools_never_change_their_input():
    image = np.random.default_rng(4).random((300, 400, 3)).astype(np.float32)
    keep = image.copy()
    a = dict.fromkeys(engine.SPATIAL_KEYS, 0)
    a.update(dehaze=-20, clarity=25, texture=-30, sharpness=50)
    out = engine.spatial_details(image, a)
    np.testing.assert_array_equal(image, keep)
    assert not np.shares_memory(out, image) and not np.array_equal(out, image)


def test_a_frame_too_large_for_memory_is_exported_in_blocks_not_refused(monkeypatch):
    from lumen import large_image
    yy, xx = np.mgrid[0:1100, 0:1600].astype(np.float32)
    source = np.stack([xx / 1600, yy / 1100, .4 + .3 * np.sin(xx / 40) * np.cos(yy / 55)], axis=2).astype(np.float32) * 1.3
    edits = model.recipe()
    edits['adjustments'].update(exposure=.3, shadows=25, highlights=-35, clarity=20, texture=15, sharpness=40, dehaze=20)
    mask = model.new_mask('radial', 1)
    mask['adjustments'].update(exposure=.5, clarity=15)
    edits['masks'].append(mask)
    edits['effects'].update(vignette=-25, grain=20)
    edits.update(crop=[.1, .1, .9, .85], flip=[True, False], rotation=1)
    monkeypatch.setattr(large_image, 'MAP_BYTES', 1 << 20)
    for angle, tolerance in ((0., 1e-5), (2., 3e-3)):
        # Straightened, the blocks are clipped to 0–1 before the turn (as streamed exports always
        # were) and the whole frame after it: under half an 8-bit step apart.
        edits['straighten'] = angle
        whole = engine.process(source, edits, _stream=False)
        monkeypatch.setattr(large_image, 'fits_in_memory', lambda *a, **k: False)
        monkeypatch.setattr(large_image, 'available_memory', lambda: 60 * 2 ** 20)
        blocks = np.asarray(engine.process(source, edits))
        monkeypatch.undo()
        monkeypatch.setattr(large_image, 'MAP_BYTES', 1 << 20)
        assert blocks.shape == whole.shape
        assert float(np.abs(blocks - whole).max()) < tolerance


def test_large_streamed_exports_flip_once(monkeypatch):
    from lumen import large_image
    image = np.random.default_rng(6).random((200, 300, 3)).astype(np.float32)
    for edits in (dict(model.recipe(), straighten=3., flip=[True, False]), dict(model.recipe(), crop=[.1, .2, .9, .8], flip=[False, True], rotation=3)):
        np.testing.assert_allclose(large_image.transform(image, edits), engine.crop_rotate(image, edits), atol=1e-6)


def test_tiles_of_a_retouched_face_reuse_the_face(face_photo):
    import time as clock
    rgb, faces = face_photo
    big = cv2.resize(rgb, (1400, 1400), interpolation=cv2.INTER_CUBIC)
    source = _linear(big)
    face = copy.deepcopy(faces[0])
    face['settings'].update(smooth=50, whiten=30, slim=40)
    edits = dict(model.recipe(), portrait=dict(faces=[face]))
    whole = engine.process(source, edits, _stream=False)
    started = clock.perf_counter()
    first = engine.process_region(source, edits, rect=(500, 500, 756, 756))
    first_time = clock.perf_counter() - started
    started = clock.perf_counter()
    second = engine.process_region(source, edits, rect=(756, 500, 1012, 756))
    second_time = clock.perf_counter() - started
    np.testing.assert_allclose(first, whole[500:756, 500:756], atol=1e-6)
    np.testing.assert_allclose(second, whole[500:756, 756:1012], atol=1e-6)
    assert second_time < first_time / 3
    edits['portrait']['faces'][0]['settings']['smooth'] = 80        # a new setting is computed afresh
    changed = engine.process_region(source, edits, rect=(500, 500, 756, 756))
    assert not np.array_equal(changed, first)


def test_a_mismatched_auto_model_is_ignored(monkeypatch, tmp_path):
    fitted = copy.deepcopy(auto_adjust.model())
    fitted['features'] = fitted['features'][:-1] + ['something_new']
    monkeypatch.setattr(auto_adjust, '_model', None)
    monkeypatch.setattr(auto_adjust.Path if hasattr(auto_adjust, 'Path') else Path, 'read_text',
                        lambda self, *a, **k: json.dumps(fitted), raising=False)
    assert auto_adjust.model() is None
    monkeypatch.setattr(auto_adjust, '_model', None)


def test_another_photo_starts_with_all_its_faces_chosen_and_turned_photos_reopen_framed(window, tmp_path, face_photo):
    w = window
    rgb, _ = face_photo
    paths = []
    for name in ('一.png', '二.png'):
        path = tmp_path / name
        Image.fromarray(np.concatenate([rgb, rgb[:, ::-1]], axis=1)).save(path)
        paths.append(str(path))
    w.import_paths(paths)
    wait_until(lambda: not w.loading and w.rendered is not None and not w.render_running)
    w.show_tab('portrait')
    wait_until(lambda: not w.portrait_running and not w.jobs)
    w.select_faces({1})
    w.rotate(1)
    w.open_path(paths[1])
    wait_until(lambda: not w.loading and not w.render_running)
    wait_until(lambda: not w.portrait_running and not w.jobs)
    assert w.face_selection == set(range(len(w.faces())))
    w.open_path(paths[0])
    wait_until(lambda: not w.loading and not w.render_running)
    assert w.edits['rotation'] == 1 and w.final_view.isChecked()
    for path in paths:
        w.documents[path]['saved_edits'] = copy.deepcopy(w.documents[path]['edits'])
    w.saved_edits = copy.deepcopy(w.edits)


# ------------------------------------------------------------------------------- 祛痘, frames, snow (2.3.0 review)

def _clear_face(size=512):
    """The drawn face without noise: clear skin, where any change is damage."""
    rgb = synthetic_face(size)
    faces = portrait.detect(rgb)
    assert len(faces) == 1
    return rgb, faces


def _retouch(rgb, faces, **values):
    linear = _linear(rgb)
    h, w = linear.shape[:2]
    settings = dict(portrait.settings(), **{k: float(v) for k, v in values.items()})
    return portrait.retouch_face(linear, portrait.pixels(faces[0], h, w), settings)


def _display(linear):
    return np.maximum(linear, 0) ** (1 / 2.2) @ np.array([.2126, .7152, .0722], np.float32)


def _open_points(faces, shape, rgb, count):
    """Points of the face's open skin, apart from each other."""
    h, w = shape
    linear = _linear(rgb)
    face = portrait.Face(np.maximum(linear, 0) ** (1 / 2.2), portrait.pixels(faces[0], h, w))
    open_skin = face.open_skin() & (face.skin() > .5)
    far = cv2.erode(open_skin.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    ys, xs = np.nonzero(far)
    order = np.random.default_rng(2).permutation(len(xs))
    points = []
    for i in order:
        if all(math.hypot(xs[i] - x, ys[i] - y) > 28 for x, y in points):
            points.append((int(xs[i]), int(ys[i])))
        if len(points) == count:
            break
    assert len(points) == count
    return points


@pytest.mark.parametrize('amount', (30, 70))
def test_blemishes_are_filled_and_clear_skin_is_left_alone(amount):
    if culling.faces() is None:
        pytest.skip('face models not available')
    rgb, faces = _clear_face()
    # Clear skin: next to nothing to do (the drawn brows do not quite lie where a real brow would).
    clean = _linear(rgb)
    assert (np.abs(_retouch(rgb, faces, acne=amount) - clean).max(axis=2) > 1e-3).mean() < .002
    # Round spots, darker and redder, come out.
    spotted = rgb.copy()
    points = _open_points(faces, rgb.shape[:2], rgb, 5)
    for x, y in points:
        cv2.circle(spotted, (x, y), 4, (176, 118, 98), -1, cv2.LINE_AA)
    spotted = cv2.GaussianBlur(spotted, (0, 0), .8)
    out = _display(_retouch(spotted, faces, acne=amount))
    before = _display(_linear(spotted))
    for x, y in points:
        ring = before[y - 12:y + 13, x - 12:x + 13].max()
        assert ring - out[y, x] < .35 * (ring - before[y, x])
    # A mark much darker than the skin is a mole or a tattoo: it stays.
    marked = rgb.copy()
    for x, y in points:
        cv2.circle(marked, (x, y), 4, (70, 50, 45), -1, cv2.LINE_AA)
    marked = cv2.GaussianBlur(marked, (0, 0), .8)
    out = _display(_retouch(marked, faces, acne=amount))
    before = _display(_linear(marked))
    for x, y in points:
        assert out[y, x] < before[y, x] + .02


def test_creases_shadows_and_frames_are_not_blemishes():
    if culling.faces() is None:
        pytest.skip('face models not available')
    rgb, faces = _clear_face()
    h, w = rgb.shape[:2]
    pts = portrait.pixels(faces[0], h, w).round().astype(int)
    marked = rgb.copy()
    # Smile lines, a crease across the forehead and the frame of a pair of glasses.
    for a, b in ((64, 212), (294, 432)):
        cv2.line(marked, tuple(pts[a]), tuple(pts[b]), (190, 138, 110), 2, cv2.LINE_AA)
    cv2.line(marked, tuple(pts[67]), tuple(pts[297]), (190, 138, 110), 2, cv2.LINE_AA)
    for iris in (468, 473):
        cv2.circle(marked, tuple(pts[iris]), 30, (120, 120, 120), 1, cv2.LINE_AA)
    before = _display(_linear(marked))
    for values in (dict(acne=80), dict(smooth=70)):
        out = _display(_retouch(marked, faces, **values))
        # What was drawn across the skin is still there: the frame most of all.
        for iris in (468, 473):
            x, y = pts[iris]
            rim = [(x + round(30 * math.cos(t)), y + round(30 * math.sin(t))) for t in (math.pi * .75, math.pi * .25)]
            for rx, ry in rim:
                block = (slice(ry - 7, ry + 8), slice(rx - 7, rx + 8))
                assert out[block].max() - out[block].min() > .6 * (before[block].max() - before[block].min())
    # 祛痘 changes no more than a trace of the crease and smile lines.
    out = _display(_retouch(marked, faces, acne=80))
    assert np.abs(out - before).max() < .03


def test_frames_are_found_only_where_there_are_thin_long_lines():
    if culling.faces() is None:
        pytest.skip('face models not available')
    rgb, faces = _clear_face()
    h, w = rgb.shape[:2]
    pts = portrait.pixels(faces[0], h, w)

    def frames(image):
        p = np.maximum(_linear(image), 0) ** (1 / 2.2)
        face = portrait.Face(p, pts)
        return face._frames(face.small(p), face.size * face.scale)
    worn = rgb.copy()
    for iris in (468, 473):
        cv2.circle(worn, tuple(pts[iris].round().astype(int)), 30, (120, 120, 120), 1, cv2.LINE_AA)
    assert frames(worn).sum() > frames(rgb).sum() + 60
    # Thick shapes and round spots are not lines.
    spotted = rgb.copy()
    cv2.circle(spotted, tuple(pts[50].round().astype(int)), 5, (176, 118, 98), -1, cv2.LINE_AA)
    assert frames(spotted).sum() < frames(rgb).sum() + 5


def _bright_scene(gain=1.):
    """A snowfield with a dark figure: bright and unclipped, or clipped when pushed by ``gain``."""
    rng = np.random.default_rng(4)
    yy, xx = np.mgrid[0:240, 0:360].astype(np.float32)
    snow = .72 + .08 * np.sin(xx / 45) * np.cos(yy / 35) + rng.normal(0, .012, xx.shape).astype(np.float32)
    scene = np.repeat(snow[..., None], 3, axis=2)
    scene[90:180, 150:210] = (.06, .05, .05)
    return np.clip(scene * gain, 0, None).astype(np.float32)


def test_automatic_keeps_snow_bright_but_still_recovers_clipped_photos():
    edits = dict(model.recipe(), process=2)
    snow = _bright_scene()
    median = lambda image, values: auto_adjust.measure(engine.process(image, dict(edits, adjustments=dict(edits['adjustments'], **values))))['median']
    before = auto_adjust.measure(engine.process(snow, edits))
    assert before['median'] > .8 and before['high_clip'] < .01
    for use_model in (True, False):
        assert median(snow, auto_adjust.suggest(snow, edits, use_model=use_model)) > before['median'] - .06
    # The same scene burnt out is not "high key": it comes down.
    burnt = _bright_scene(3.)
    start = auto_adjust.measure(engine.process(burnt, edits))
    assert start['high_clip'] > .3
    assert auto_adjust._keep_high_key(.6, start, .7) == .6
    assert auto_adjust._keep_high_key(.6, dict(before), .7) > before['median'] - .01
    assert auto_adjust._keep_high_key(.6, dict(before), .1) == .6        # bright, but not a high key


def test_opening_a_photo_with_the_portrait_panel_open_finds_its_faces(window, tmp_path, face_photo):
    w = window
    rgb, _ = face_photo
    paths = []
    for name in ('甲.png', '乙.png', '丙.png'):
        path = tmp_path / name
        Image.fromarray(rgb).save(path)
        paths.append(str(path))
    w.import_paths(paths)
    wait_until(lambda: not w.loading and w.rendered is not None and not w.render_running)
    w.show_tab('portrait')
    wait_until(lambda: not w.portrait_running and not w.jobs)
    assert len(w.faces()) == 1
    w.open_path(paths[1])
    wait_until(lambda: not w.loading and not w.render_running)
    wait_until(lambda: not w.portrait_running and not w.jobs)
    assert w.source_path == paths[1] and len(w.faces()) == 1 and w.canvas.faces
    # Another photo while the first is still being searched: it is searched when that ends.
    w.detect_faces()
    assert w.portrait_running
    w.open_path(paths[2])
    wait_until(lambda: not w.loading and not w.portrait_running and not w.jobs)
    wait_until(lambda: not w.render_running and not w.portrait_running and not w.jobs)
    assert w.source_path == paths[2] and len(w.faces()) == 1
    w.show_tab('basic')
    for path in paths:
        w.documents[path]['saved_edits'] = copy.deepcopy(w.documents[path]['edits'])
    w.saved_edits = copy.deepcopy(w.edits)


def test_the_macos_app_is_a_foreground_app():
    """Issue #11: PyInstaller makes a bundle with a console EXE (lumen-cli) background-only, so the
    menu bar and the key equivalents never reach it."""
    root = Path(__file__).resolve().parents[1]
    spec = (root / 'LumenRAW-macOS.spec').read_text(encoding='utf-8')
    assert "'LSBackgroundOnly': False" in spec and 'console=True' in spec
    assert 'LSBackgroundOnly must be false' in (root / 'tools' / 'build_macos.sh').read_text(encoding='utf-8')
