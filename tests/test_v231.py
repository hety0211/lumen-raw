"""2.3.1: the camera colour reference of the develop starting point (the editor showed photos
less colourful than the culling view), lens facts in copies LUMEN renders (AI copies read "no
lens data"), working copies of agents' edits (GitHub issue #13), and crop, positioned masks,
spots, white-balance sampling and 100 % previews for agents (issues #14 and #15)."""
import base64
import copy
import io
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from lumen import catalog, color, commands, develop, engine, gpu_graphs, lens, model, photo_metadata, tone, working

SAMPLES = Path(os.environ.get('LUMEN_SAMPLES', r'E:\LumenSamples'))


def scene(seed=3, shape=(96, 144)):
    """A colourful linear picture: smooth ramps with patches of every hue."""
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([.15 + .5 * xx / w, .2 + .45 * yy / h, .55 - .35 * xx / w], axis=2)
    for _ in range(14):
        y, x = rng.integers(0, h - 16), rng.integers(0, w - 16)
        base[y:y + 16, x:x + 16] = rng.random(3) * .8 + .05
    return base.astype(np.float32)


def chroma_of(srgb):
    lab = color.to_oklab(color.srgb_to_linear(np.clip(srgb, 0, 1)))
    return float(np.hypot(lab[..., 1], lab[..., 2]).mean())


# ------------------------------------------------------------------------------- camera colour reference

def test_camera_chroma_measures_how_much_more_colourful_the_camera_picture_is():
    ours = engine.to_srgb(scene())
    for gain in (1.4, .85):
        camera = color.adjust(ours, (gain - 1) * 100)
        assert develop.camera_chroma(ours, camera) == pytest.approx(gain, abs=.06)
    # Other proportions (a 16:9 preview of a 3:2 frame) and a blurred preview still compare.
    camera = color.adjust(ours, 40.)[8:-8]
    assert develop.camera_chroma(ours, cv2.GaussianBlur(camera, (0, 0), 1.2)) == pytest.approx(1.4, abs=.12)
    # A monochrome picture style does not make the RAW monochrome; a grey scene tells nothing.
    grey = np.repeat(ours.mean(axis=2, keepdims=True), 3, axis=2)
    assert develop.camera_chroma(ours, grey) == 1.
    assert develop.camera_chroma(grey, ours) == 1.
    # Never beyond what a camera style plausibly does.
    assert develop.camera_chroma(ours, color.adjust(ours, 250.)) == develop.CHROMA_RANGE[1]
    assert develop.camera_chroma(color.adjust(ours, 60.), color.adjust(ours, -30.)) == develop.CHROMA_RANGE[0]
    # Sensor noise in the fresh decode is not colour: the gain of a noisy copy stays the same.
    noisy = np.clip(ours + np.random.default_rng(1).normal(0, .06, ours.shape).astype(np.float32), 0, 1)
    assert develop.camera_chroma(noisy, color.adjust(ours, 40.)) == pytest.approx(1.4, abs=.1)


def test_chroma_gain_rides_on_saturation_in_every_stage():
    x = scene()
    edits = model.recipe()
    edits['develop'] = dict(mode='camera', curve=[[0., 0.], [.2, .45], [.6, .9], [1., 1.]], source='', chroma=1.5)
    assert tone.look_saturation(edits) == pytest.approx(50)
    edits['adjustments']['saturation'] = -100.
    assert tone.look_saturation(edits) == pytest.approx(-100)       # fully desaturated stays grey
    edits['adjustments']['saturation'] = 20.
    assert tone.look_saturation(edits) == pytest.approx(80)
    # The same picture as the plain reference with the saturation slider at the combined value.
    plain = copy.deepcopy(edits)
    plain['develop']['chroma'] = 1.
    plain['adjustments']['saturation'] = 80.
    np.testing.assert_array_equal(engine.process(x, edits), engine.process(x, plain))
    # The linear starting point and recipes without the key have no gain.
    linear = copy.deepcopy(edits)
    linear['develop']['mode'] = 'linear'
    assert tone.look_saturation(linear) == 20 and develop.chroma(linear['develop']) == 1
    assert develop.chroma(dict(mode='camera', curve=[[0, 0], [1, 1]])) == 1 and develop.chroma(None) == 1
    # Block renders (zoomed tiles, large exports) match the whole frame.
    whole = engine.process(x, edits, apply_crop=False)
    np.testing.assert_allclose(engine.process_region(x, edits, rect=(20, 10, 90, 70)), whole[10:70, 20:90], atol=1e-6)
    # A mask's own saturation is not multiplied again.
    masked = copy.deepcopy(edits)
    mask = model.new_mask('linear', 1)
    mask['adjustments']['exposure'] = .4
    masked['masks'].append(mask)
    lifted = engine.process(x, masked)
    assert chroma_of(lifted) < chroma_of(engine.process(x, edits)) * 1.25


def test_develop_view_shows_the_starting_point_with_its_colour():
    x = scene()
    edits = model.recipe()
    edits['develop'] = dict(mode='camera', curve=[[0., 0.], [.25, .5], [1., 1.]], source='', chroma=1.35)
    view = engine.develop_view(x, edits)
    np.testing.assert_allclose(view, engine.process(x, edits), atol=1e-6)      # before = after without edits
    plain = engine.develop_view(x, dict(edits, develop=dict(edits['develop'], chroma=1.)))
    assert chroma_of(view) == pytest.approx(chroma_of(plain) * 1.35, rel=.08)
    # Lightness is kept: only colourfulness changes.
    lab = lambda v: color.to_oklab(color.srgb_to_linear(v))[..., 0]
    assert np.abs(lab(view) - lab(plain)).mean() < .004


def test_gpu_inputs_carry_the_chroma_gain():
    import onnxruntime as ort
    x = scene()
    edits = model.recipe()
    edits['develop'] = dict(mode='camera', curve=[[0., 0.], [.2, .45], [.6, .9], [1., 1.]], source='', chroma=1.6)
    assert gpu_graphs.lab_stage(edits)                       # the OkLCh pass runs without any slider
    inputs = gpu_graphs.color2_inputs(edits)
    assert float(inputs['saturation'][0]) == pytest.approx(1.6)
    edits['adjustments'].update(saturation=-25., vibrance=15.)
    assert float(gpu_graphs.color2_inputs(edits)['saturation'][0]) == pytest.approx(1.6 * .75)
    a = edits['adjustments']
    spec = tone.spec(edits)
    tonal = tone.tonal(x, a, spec)
    expected = engine.color_stage(tonal, edits)

    def run(kind, image, feed):
        session = ort.InferenceSession(gpu_graphs.model(kind), providers=['CPUExecutionProvider'])
        return session.run(None, dict(feed, image=image[None]))[0][0]
    np.testing.assert_allclose(run('color2lab', tonal, gpu_graphs.color2_inputs(edits)), expected, atol=5e-4)
    fused = run('fused2lab', x, dict(gpu_graphs.tonal2_inputs(a, spec, x.shape), **gpu_graphs.color2_inputs(edits)))
    assert np.abs(fused - expected).max() < 2e-3
    plain = model.recipe()
    assert not gpu_graphs.lab_stage(plain) and 'saturation' not in gpu_graphs.color2_inputs(plain)


def test_recipes_keep_the_chroma_gain_and_older_ones_render_as_before(tmp_path):
    r = model.recipe()
    assert r['develop']['chroma'] == 1.
    old = copy.deepcopy(r)
    old['develop'] = dict(mode='camera', curve=[[0., 0.], [.3, .5], [1., 1.]], source='相机预览亮度参考')
    checked = model.validate(old)
    assert checked['develop']['chroma'] == 1.           # a 2.3.0 project: colours exactly as decoded
    x = scene()
    np.testing.assert_array_equal(engine.process(x, checked), engine.process(x, old))
    new = copy.deepcopy(checked)
    new['develop']['chroma'] = 1.42
    project = tmp_path / 'look.lumen'
    photo = tmp_path / 'photo.png'
    Image.fromarray((engine.to_srgb(x) * 255).astype(np.uint8)).save(photo)
    model.save_project(project, photo, new)
    assert model.load_project(project)[1]['develop']['chroma'] == 1.42
    for bad in (0, -1, 9, True, 'x'):
        with pytest.raises(ValueError):
            model.validate(dict(new, develop=dict(new['develop'], chroma=bad)))
    # The labels of both generations stay translatable.
    assert '相机预览亮度参考' in develop.SOURCES and '相机预览亮度与色彩参考' in develop.SOURCES


@pytest.mark.parametrize('name', ['7RM5-LosslessCompressedLarge.ARW', 'Nikon_Z8_raw_14_bit_lossless_compression.NEF'])
def test_a_raw_photo_starts_as_colourful_as_the_camera_showed_it(name):
    path = SAMPLES / name
    if not path.is_file():
        pytest.skip('RAW samples not available')
    from lumen import previews
    rgb, info = engine.load_image(path, 1600)
    reference = info['develop']
    assert reference['mode'] == 'camera' and develop.CHROMA_RANGE[0] <= reference['chroma'] <= develop.CHROMA_RANGE[1]
    assert reference['chroma'] != 1 and reference['source'] == '相机预览亮度与色彩参考'
    camera = previews.embedded_view(path, 1600)[0].astype(np.float32) / 255
    matched = develop._colourfulness(engine.develop_view(rgb, dict(develop=reference, process=2)))
    plain = develop._colourfulness(engine.develop_view(rgb, dict(develop=dict(reference, chroma=1.), process=2)))
    target = develop._colourfulness(camera)
    assert abs(matched - target) < .35 * abs(plain - target) + .002


def test_hdr_previews_give_brightness_only():
    path = SAMPLES / 'R5m2_RAW.CR3'
    if not path.is_file():
        pytest.skip('RAW samples not available')
    _, info = engine.load_image(path, 1600)
    assert info['develop']['chroma'] == 1. and 'HDR' in info['develop']['source']


# ------------------------------------------------------------------------------- lens facts in rendered copies

EXIF = dict(make='SONY', model='ILCE-7M3', lens_make='', lens_model='FE 200-600mm F5.6-6.3 G OSS', lens='',
            lens_id='Sony FE 200-600mm F5.6-6.3 G OSS', focal=200., aperture=5.6, distance=5.1, crop35=1.5)
PHOTO = dict(body='ILCE-7M3', brand='SONY', lens='FE 200-600mm F5.6-6.3 G OSS', focal='200 mm', aperture='f/5.6')


def corrected():
    edits = model.recipe()
    edits['lens'] = lens.prepare(dict(lens.defaults(), enabled=True), lens.identify(EXIF))[0]
    assert lens.active(edits['lens'])
    return edits


def test_copies_record_their_lens_and_what_was_done_to_the_frame():
    plain = lens.copy_record(EXIF, model.recipe())
    assert plain == dict(exif=EXIF, corrected=False, frame='full')
    assert lens.copy_record(EXIF, corrected())['corrected'] is True
    assert lens.copy_record(EXIF, dict(model.recipe(), crop=[.1, .1, .9, .9]))['frame'] == 'cropped'
    assert lens.copy_record(EXIF, dict(model.recipe(), straighten=2.))['frame'] == 'cropped'
    # Quarter turns and mirroring keep the lens axis in the middle.
    assert lens.copy_record(EXIF, dict(model.recipe(), rotation=1, flip=[True, False]))['frame'] == 'full'
    assert lens.copy_record(None, model.recipe()) is None and lens.copy_record({}, model.recipe()) is None


def test_reopened_copies_are_recognised_and_never_corrected_twice():
    def opened(record, **kw):
        exif, match = lens.from_copy(dict(operation='super', source='A.ARW', scale=2, lens=record), PHOTO, **kw)
        return match, lens.prepare(dict(lens.defaults(), enabled=True), match)
    match, (settings, reason) = opened(lens.copy_record(EXIF, model.recipe()))
    assert 'copy' not in match and match['crop'] == 1.5 and reason == '' and settings['profile']['focal'] == 200
    assert lens.auto_enables(match) and '200-600' in match['label']
    match, (settings, reason) = opened(lens.copy_record(EXIF, corrected()))
    assert match['copy'] == reason == lens.COPY_APPLIED and settings['profile'] is None and not lens.auto_enables(match)
    match, (settings, reason) = opened(lens.copy_record(EXIF, dict(model.recipe(), crop=[0., 0., .5, .5])))
    assert match['copy'] == reason == lens.COPY_CROPPED and settings['profile'] is None
    # A lens chosen by hand is the user's own call.
    manual = dict(lens.defaults(), enabled=True, manual=dict(maker='Sony', model='FE 200-600mm f/5.6-6.3 G OSS'))
    assert lens.prepare(manual, match)[0]['profile'] is not None
    # A copy of a corrected copy stays corrected; of a cropped one, cropped.
    again = lens.copy_record(EXIF, model.recipe(), dict(copy=lens.COPY_APPLIED))
    assert again['corrected'] and opened(again)[0]['copy'] == lens.COPY_APPLIED
    assert lens.copy_record(EXIF, model.recipe(), dict(copy=lens.COPY_CROPPED))['frame'] == 'cropped'


def test_copies_of_earlier_versions_take_the_lens_from_the_original_or_their_shooting_data(monkeypatch, tmp_path):
    legacy = dict(operation='super', source='A.ARW', backend='x', scale=2, amount=0)
    # No original next to the copy: the shooting data alone (the in-camera crop mode is unknown).
    exif, match = lens.from_copy(legacy, PHOTO, tmp_path / 'A-增强.dng', (7872, 5296))
    assert match['copy'] == lens.COPY_LEGACY and match['crop'] == 1. and '200-600' in match['label']
    assert exif['focal'] == 200 and exif['aperture'] == 5.6
    settings, reason = lens.prepare(dict(lens.defaults(), enabled=True), match)
    assert settings['profile'] is not None and reason == '' and not lens.auto_enables(match)
    # The original lies next to it: its exact facts, and the frame is compared.
    monkeypatch.setattr(lens, '_original_of', lambda processing, path: (EXIF, (3936, 2648)))
    _, match = lens.from_copy(legacy, PHOTO, tmp_path / 'A-增强.dng', (7872, 5296))
    assert match['copy'] == lens.COPY_LEGACY and match['crop'] == 1.5
    _, match = lens.from_copy(legacy, PHOTO, tmp_path / 'A-增强.dng', (5296, 7872))      # turned a quarter
    assert match['copy'] == lens.COPY_LEGACY
    _, match = lens.from_copy(legacy, PHOTO, tmp_path / 'A-增强.dng', (6000, 4000))      # cropped before the copy
    assert match['copy'] == lens.COPY_CROPPED
    # A copy of such a copy stays uncertain.
    record = lens.copy_record(EXIF, model.recipe(), dict(copy=lens.COPY_LEGACY))
    assert lens.from_copy(dict(lens=record), PHOTO)[1]['copy'] == lens.COPY_LEGACY
    # Nothing known at all.
    assert lens.from_copy({}, {}) == (None, None) and lens.from_copy(None, None) == (None, None)
    assert lens._original_of.__name__ == '<lambda>'
    monkeypatch.undo()
    assert lens._original_of(dict(source='../A.ARW'), tmp_path / 'c.dng') is None       # only a name next to it
    assert lens._original_of(dict(source='missing.ARW'), tmp_path / 'c.dng') is None


@pytest.mark.parametrize('extension', ['.dng', '.tif'])
def test_rendered_files_open_with_their_lens(tmp_path, extension):
    picture = np.clip(engine.to_srgb(scene()), 0, 1)
    info = dict(lens_exif=EXIF, lens_match=lens.identify(EXIF))
    target = tmp_path / f'copy{extension}'
    provenance = engine.export_provenance(tmp_path / 'A.ARW', info, model.recipe())
    assert provenance['operation'] == 'export' and provenance['source'] == 'A.ARW'
    engine.export_image(target, picture, photo=PHOTO, provenance=provenance)
    assert photo_metadata.processing(target)['lens']['exif']['focal'] == 200
    _, opened = engine.load_image(target, 800)
    assert '200-600' in opened['lens_match']['label'] and 'copy' not in opened['lens_match']
    assert opened['lens_exif']['crop35'] == 1.5
    # Exported with the corrections on: recognised, and left alone.
    engine.export_image(target, picture, photo=PHOTO, provenance=engine.export_provenance('A.ARW', info, corrected()))
    _, opened = engine.load_image(target, 800)
    assert opened['lens_match']['copy'] == lens.COPY_APPLIED
    # Photos without lens data record nothing, and files of 2.3.0 open as before.
    assert engine.export_provenance('A.ARW', {}, model.recipe()) is None
    engine.export_image(target, picture, photo={})
    _, opened = engine.load_image(target, 800)
    assert opened['lens_match'] is None and photo_metadata.processing(target) == {}


def test_a_legacy_copy_next_to_its_raw_original(tmp_path):
    original = SAMPLES / '7RM5-LosslessCompressedLarge.ARW'
    if not original.is_file():
        pytest.skip('RAW samples not available')
    link = tmp_path / original.name
    try:
        os.link(original, link)
    except OSError:
        import shutil
        shutil.copyfile(original, link)           # another drive
    _, info = engine.load_image(link, 800)
    w, h = info['width'], info['height']
    small = np.clip(engine.to_srgb(scene(shape=(h // 16, w // 16))), 0, 1)
    copy_path = tmp_path / 'copy.dng'
    engine.export_image(copy_path, small, photo=info['photo'],
                        provenance=dict(operation='super', source=link.name, backend='x', scale=1 / 16, amount=0))
    _, opened = engine.load_image(copy_path, 800)
    assert opened['lens_match']['copy'] == lens.COPY_LEGACY
    assert opened['lens_exif']['lens_model'] == info['lens_exif']['lens_model']
    assert opened['lens_exif']['focal'] == info['lens_exif']['focal']


# ------------------------------------------------------------------------------- working copies (issue #13)

@pytest.fixture
def photos(tmp_path, monkeypatch):
    monkeypatch.setenv('LUMEN_WORKING', str(tmp_path / 'working'))
    folder = tmp_path / 'photos'
    folder.mkdir()
    paths = []
    for i, tint in enumerate(((.8, .5, .3), (.3, .5, .8))):
        yy, xx = np.mgrid[0:240, 0:360]
        rgb = np.stack([xx / 360 * tint[0], yy / 240 * tint[1], .5 * tint[2] + .2 * np.sin(xx / 15)], axis=2)
        path = folder / f'photo {i}.png'
        Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)
        paths.append(str(path.resolve()))
    return paths


def session(tmp_path, keep=True):
    from lumen.session import Session
    return Session(catalog.Catalog(tmp_path / 'catalog.jsonl'), engine.Backend('cpu'), keep=keep)


def test_edits_survive_a_new_session(tmp_path, photos):
    first = session(tmp_path)
    run = lambda command, **args: commands.run(first, command, args)
    opened = run('photo.open', path=photos[0])
    assert 'restored' not in opened
    applied = run('edit.apply', path=photos[0], changes=dict(adjustments=dict(exposure=-.42, contrast=18)))
    assert 'working copy' in applied['persistence'] and 'working copy' in run('app.status')['edits']
    run('edit.apply', path=photos[1], changes=dict(adjustments=dict(shadows=30)))
    # Back to the first photo with an explicit path, in the same session.
    assert run('edit.get', path=photos[0])['state']['adjustments']['exposure'] == -.42
    # The process ends; a new one reads them back (the report: 0.42 in, 0.00 out).
    second = session(tmp_path)
    run = lambda command, **args: commands.run(second, command, args)
    got = run('edit.get', path=photos[0])
    assert got['state']['adjustments']['exposure'] == -.42 and got['state']['adjustments']['contrast'] == 18
    assert 'restored' in got and 'restored' in run('photo.open', path=photos[1])
    assert run('edit.get', path=photos[1])['state']['adjustments']['shadows'] == 30
    # Undo, redo and reset are kept as well.
    run('edit.apply', path=photos[0], changes=dict(adjustments=dict(exposure=.3)))
    assert run('edit.undo', path=photos[0])['state']['adjustments']['exposure'] == -.42
    third = session(tmp_path)
    assert commands.run(third, 'edit.get', dict(path=photos[0]))['state']['adjustments']['exposure'] == -.42
    assert commands.run(third, 'edit.reset', dict(path=photos[0]))['state']['adjustments']['exposure'] == 0
    assert commands.run(session(tmp_path), 'edit.get', dict(path=photos[0]))['state']['adjustments']['exposure'] == 0
    # Nothing was written next to the photos.
    assert sorted(p.name for p in Path(photos[0]).parent.iterdir()) == ['photo 0.png', 'photo 1.png']


def test_sessions_that_do_not_keep_write_nothing(tmp_path, photos):
    plain = session(tmp_path, keep=False)
    result = commands.run(plain, 'edit.apply', dict(path=photos[0], changes=dict(adjustments=dict(exposure=1.))))
    assert 'this process only' in result['persistence']
    assert not (tmp_path / 'working').exists()
    assert commands.run(session(tmp_path), 'edit.get', dict(path=photos[0]))['state']['adjustments']['exposure'] == 0


def test_the_newer_of_project_and_working_copy_wins(tmp_path, photos):
    first = session(tmp_path)
    run = lambda command, **args: commands.run(first, command, args)
    run('edit.apply', path=photos[0], changes=dict(adjustments=dict(exposure=.5)))
    project = run('project.save', path=photos[0])['project']
    # Saved next to the photo: the project takes the working copy's place.
    assert Path(project).is_file() and not list((tmp_path / 'working').glob('*.lumen'))
    run('edit.apply', path=photos[0], changes=dict(adjustments=dict(exposure=.9)))
    # Opening the photo again in the same session keeps what was applied since the save
    # (it used to fall back to the saved project).
    run('photo.open', path=photos[1])
    assert run('photo.open', path=photos[0])['state']['adjustments']['exposure'] == .9
    # A new session: the working copy is newer than the project.
    second = session(tmp_path)
    assert commands.run(second, 'photo.open', dict(path=photos[0]))['state']['adjustments']['exposure'] == .9
    # The user saves the project in the editor afterwards: that is the newer state.
    edits = model.load_project(project)[1]
    edits['adjustments']['exposure'] = -1.
    time.sleep(.05)
    model.save_project(project, photos[0], edits)
    os.utime(project, (time.time() + 5, time.time() + 5))
    third = session(tmp_path)
    opened = commands.run(third, 'photo.open', dict(path=photos[0]))
    assert opened['state']['adjustments']['exposure'] == -1 and 'restored' not in opened
    # Opening the project file itself always shows the project.
    commands.run(third, 'edit.apply', dict(path=photos[0], changes=dict(adjustments=dict(exposure=2.))))
    assert commands.run(third, 'photo.open', dict(path=project))['state']['adjustments']['exposure'] == -1
    # A project saved somewhere else does not replace the working copy.
    commands.run(third, 'edit.apply', dict(path=photos[1], changes=dict(adjustments=dict(contrast=40))))
    commands.run(third, 'project.save', dict(path=photos[1], output=str(tmp_path / 'elsewhere.lumen')))
    assert commands.run(session(tmp_path), 'edit.get', dict(path=photos[1]))['state']['adjustments']['contrast'] == 40


def test_working_copies_belong_to_their_photo_and_expire(tmp_path, photos, monkeypatch):
    edits = model.recipe()
    edits['adjustments']['exposure'] = 1.
    assert working.save(photos[0], edits) and working.load(photos[0])['adjustments']['exposure'] == 1
    assert working.load(photos[1]) is None
    stored = next((tmp_path / 'working').glob('*.lumen'))
    # A file that names another photo is not used.
    other = working._file(photos[1])
    other.write_bytes(stored.read_bytes())
    assert working.load(photos[1]) is None
    other.write_text('not json', encoding='utf-8')
    assert working.load(photos[1]) is None
    # Unused for a month: removed the next time something is kept.
    old = time.time() - (working.KEEP_DAYS + 2) * 86400
    os.utime(other, (old, old))
    monkeypatch.setattr(working, '_pruned', False)
    working.save(photos[0], edits)
    assert not other.exists() and stored.exists()
    working.discard(photos[0])
    assert working.load(photos[0]) is None
    # A folder that cannot be written never fails an edit.
    monkeypatch.setenv('LUMEN_WORKING', str(Path(photos[0])))      # a file, not a folder
    assert working.save(photos[0], edits) is False


def test_mcp_tells_when_the_editor_window_goes_away(tmp_path, photos, monkeypatch):
    from lumen import control, mcp

    class Window:
        """An editor window that answers once and then closes in the middle of a command."""
        calls = 0

        def __init__(self, connect_timeout=1.):
            if Window.calls >= 2:
                raise control.Unavailable('LUMEN RAW is not running')

        def call(self, command, args=None):
            Window.calls += 1
            if Window.calls == 2:
                raise ConnectionError('LUMEN RAW closed the connection')
            # What the window does with an applied edit: it also keeps a working copy.
            edits = model.recipe()
            edits['adjustments']['exposure'] = args['changes']['adjustments']['exposure']
            working.save(args['path'], edits)
            return dict(applied=['exposure'])

        def close(self):
            pass
    monkeypatch.setattr(control, 'Client', Window)
    dispatcher = mcp.Dispatcher()
    dispatcher._session = session(tmp_path)
    change = dict(path=photos[0], changes=dict(adjustments=dict(exposure=-.6)))
    assert dispatcher.run('edit.apply', change) == dict(applied=['exposure'])
    with pytest.raises(commands.CommandError, match='editor window closed while running render.preview'):
        dispatcher.run('render.preview', dict(path=photos[0]))
    # The next call continues without the window, says so once, and has the edits.
    got = dispatcher.run('edit.get', dict(path=photos[0]))
    assert got['state']['adjustments']['exposure'] == -.6 and 'restored' in got
    assert 'notice' not in got                                   # the failed call already told
    assert 'notice' not in dispatcher.run('edit.get', dict(path=photos[0]))
    # The window simply gone between two calls: the first headless result carries the notice.
    Window.calls = 0
    dispatcher = mcp.Dispatcher()
    dispatcher._session = session(tmp_path)
    dispatcher.run('edit.apply', dict(path=photos[1], changes=dict(adjustments=dict(exposure=.25))))
    Window.calls = 2
    got = dispatcher.run('edit.get', dict(path=photos[1]))
    assert 'no longer open' in got['notice'] and got['state']['adjustments']['exposure'] == .25
    assert 'notice' not in dispatcher.run('app.status', {})
    # The MCP tool result is an error an agent can read, not a bare exception.
    Window.calls = 1
    server = mcp.Server(mcp.Dispatcher(), io.BytesIO())
    reply = server.call('render_preview', dict(path=photos[0]))
    assert reply['isError'] and 'working copies' in reply['content'][0]['text']
    assert 'persistence' in mcp.INSTRUCTIONS and 'notice' in mcp.INSTRUCTIONS


def test_the_editor_window_keeps_working_copies_too():
    import inspect
    from lumen import control
    source = inspect.getsource(control.WindowContext.apply) + inspect.getsource(control.WindowContext.step)
    assert source.count('working.save(path, edits)') == 2
    assert 'working.discard(path)' in inspect.getsource(control.WindowContext.save_project)
    assert 'working copy' in control.WindowContext.persistence(None)


# ------------------------------------------------------------------------------- crop for agents (issue #14)

def test_crop_rotate_and_straighten_through_edit_apply(tmp_path, photos):
    context = session(tmp_path)
    run = lambda **changes: commands.run(context, 'edit.apply', dict(path=photos[0], changes=changes))
    result = run(crop=dict(rect=[.1, .2, .9, .8], angle=1.8))
    assert result['state']['crop'] == dict(rect=[.1, .2, .9, .8], angle=1.8, rotate=0, flip_horizontal=False,
                                           flip_vertical=False)
    assert result['size'] == dict(photo=[360, 240], result=[288, 144]) and any(l.startswith('crop') for l in result['applied'])
    edits = context.edits(photos[0])
    assert edits['crop'] == [.1, .2, .9, .8] and edits['straighten'] == 1.8
    # What is read back can be written back.
    state = commands.run(context, 'edit.get', dict(path=photos[0]))['state']['crop']
    assert run(crop=state)['applied'] == []
    # Only what is named changes; the shape is the final picture's.
    result = run(crop=dict(aspect='1:1'))
    x0, y0, x1, y1 = result['state']['crop']['rect']
    assert (x1 - x0) * 360 == pytest.approx((y1 - y0) * 240, abs=.01) and result['state']['crop']['angle'] == 1.8
    assert (y0, y1) == (.2, .8) and (x0 + x1) / 2 == pytest.approx(.5)
    result = run(crop=dict(rect=None, rotate=90, aspect='4:5', flip_horizontal=True))
    x0, y0, x1, y1 = result['state']['crop']['rect']
    assert result['state']['crop']['rotate'] == 90 and result['state']['crop']['flip_horizontal'] is True
    width, height = result['size']['result']
    assert width / height == pytest.approx(4 / 5, abs=.01)             # after the quarter turn
    preview = commands.run(context, 'render.preview', dict(path=photos[0], size=256))
    assert preview['width'] / preview['height'] == pytest.approx(4 / 5, abs=.02)
    assert run(crop=dict(aspect='original', rect=None, rotate=0))['size']['result'] == [360, 240]
    # null shows the whole photo again.
    result = run(crop=None)
    assert 'crop' not in result['state'] and result['size']['result'] == [360, 240]
    assert context.edits(photos[0])['crop'] is None and context.edits(photos[0])['straighten'] == 0
    for bad in (dict(rect=[.5, .1, .4, .9]), dict(rect=[0, 0, 2, 1]), dict(angle=60), dict(rotate=45),
                dict(aspect='wide'), dict(flip_horizontal=1), dict(zoom=2), 'square'):
        with pytest.raises(commands.CommandError):
            run(crop=bad)
    # The look's strength does not weaken the crop.
    half = commands.run(context, 'edit.apply', dict(path=photos[0], amount=50,
                                                    changes=dict(adjustments=dict(exposure=1.), crop=dict(rect=[0, 0, .5, .5]))))
    assert half['state']['adjustments']['exposure'] == .5 and half['state']['crop']['rect'] == [0., 0., .5, .5]


def test_crop_angle_auto_levels_a_tilted_horizon(tmp_path, monkeypatch):
    monkeypatch.setenv('LUMEN_WORKING', str(tmp_path / 'working'))
    picture = np.full((400, 600, 3), 170, np.uint8)
    picture[215:] = 60
    tilted = cv2.warpAffine(picture, cv2.getRotationMatrix2D((300, 200), -4, 1.25), (600, 400), borderMode=cv2.BORDER_REFLECT)
    path = tmp_path / 'horizon.png'
    Image.fromarray(tilted).save(path)
    context = session(tmp_path)
    result = commands.run(context, 'edit.apply', dict(path=str(path), changes=dict(crop=dict(angle='auto'))))
    assert abs(result['state']['crop']['angle'] - 4) < .4
    flat = tmp_path / 'flat.png'
    Image.fromarray(np.full((200, 300, 3), 120, np.uint8)).save(flat)
    result = commands.run(context, 'edit.apply', dict(path=str(flat), changes=dict(crop=dict(angle='auto'))))
    assert 'crop' not in result['state'] and any('no clear horizon' in n for n in result['notes'])


# ------------------------------------------------------------------------------- masks, spots, white balance (issue #15)

def test_masks_by_position(tmp_path, photos):
    context = session(tmp_path)
    run = lambda **changes: commands.run(context, 'edit.apply', dict(path=photos[0], changes=changes))
    result = run(masks=[dict(shape='radial', center=[.3, .4], radius=.2, adjustments=dict(exposure=-.5)),
                        dict(shape='linear', **{'from': [.5, .6], 'to': [.5, 0.]}, adjustments=dict(highlights=-30), invert=True),
                        dict(shape='brush', points=[[.2, .2], [.6, .3], [.8, .7]], size=.04, feather=30,
                             adjustments=dict(saturation=20), name='cat face')])
    masks = context.edits(photos[0])['masks']
    assert [m['kind'] for m in masks] == ['radial', 'linear', 'brush'] and len(result['applied']) == 3
    assert masks[0]['start'] == pytest.approx([.1, .2]) and masks[0]['end'] == pytest.approx([.5, .6])
    assert masks[1]['start'] == [.5, .6] and masks[1]['end'] == [.5, 0.] and masks[1]['invert'] is True
    assert masks[2]['strokes'][0]['radius'] == .04 and masks[2]['feather'] == 30 and masks[2]['name'] == 'cat face'
    # The radial mask covers its centre and leaves a far corner alone; inverted, the other way round.
    alpha = engine.mask_alpha(masks[0], (240, 360))
    assert alpha[96, 108] > .95 and alpha[230, 350] == 0
    assert engine.mask_alpha(dict(masks[0], invert=True), (240, 360))[230, 350] == 1
    painted = engine.mask_alpha(masks[2], (240, 360))
    assert painted[48, 72] > .9 and painted[200, 40] == 0
    # An ellipse over the edge is kept inside the photo, and says so.
    result = run(masks=[dict(shape='radial', center=[.05, .05], radius=[.2, .3], adjustments=dict(exposure=.3))])
    assert context.edits(photos[0])['masks'][3]['start'] == [0., 0.] and len(result['notes']) == 1
    # Existing masks: switch off, invert, remove.
    run(masks=[dict(index=1, enabled=False), dict(index=0, invert=True)])
    masks = context.edits(photos[0])['masks']
    assert masks[1]['enabled'] is False and masks[0]['invert'] is True
    run(masks=[dict(index=0, remove=True), dict(index=2, remove=True)])
    assert [m['kind'] for m in context.edits(photos[0])['masks']] == ['linear', 'radial']
    # Bad coordinates are skipped with a note, like an unknown region.
    result = run(masks=[dict(shape='radial', center=[1.5, .5], adjustments=dict(exposure=.3)),
                        dict(shape='linear', **{'from': [.5, .5], 'to': [.5, .5]}, adjustments=dict(exposure=.3)),
                        dict(shape='brush', points=[], adjustments=dict(exposure=.3)),
                        dict(shape='star', adjustments=dict(exposure=.3))])
    assert len(result['notes']) == 4 and len(context.edits(photos[0])['masks']) == 2
    # At most four new masks at once, as before.
    many = [dict(shape='radial', center=[.5, .5], radius=.1, adjustments=dict(exposure=.1))] * 6
    result = run(masks=many)
    assert len(context.edits(photos[0])['masks']) == 6 and len(result['notes']) == 1


def test_spots_heal_and_clone(tmp_path, photos):
    context = session(tmp_path)
    run = lambda **changes: commands.run(context, 'edit.apply', dict(path=photos[0], changes=changes))
    result = run(spots=[dict(x=.3, y=.4, radius=.02), dict(x=.6, y=.5, source=[.7, .55])])
    ops = context.edits(photos[0])['retouch']
    assert [op['kind'] for op in ops] == ['heal', 'clone'] and ops[0]['points'] == [[.3, .4]] and ops[0]['radius'] == .02
    assert ops[1]['offset'] == pytest.approx([.1, .05]) and ops[1]['radius'] == .01
    assert result['state']['spots'] == [dict(x=.3, y=.4, radius=.02), dict(x=.6, y=.5, radius=.01, source=[.7, .55])]
    assert 'spots  0 → 2' in result['applied']
    # The healed spot changes the picture there and nowhere else.
    source, _ = context.source(photos[0])
    before = engine.process(source, dict(context.edits(photos[0]), retouch=[]))
    after = engine.process(source, context.edits(photos[0]))
    changed = np.abs(after - before).max(axis=2) > 1e-4
    assert changed[90:102, 102:114].any() and not changed[:40].any() and not changed[:, :60].any()
    run(spots=[dict(x=.1, y=.1)])
    assert len(context.edits(photos[0])['retouch']) == 3
    assert 'spots' not in run(spots=None)['state'] and context.edits(photos[0])['retouch'] == []
    for bad in ([dict(x=2, y=.5)], [dict(x=.5, y=.5, radius=.5)], [dict(x=.5)], [dict(x=.5, y=.5, colour='red')], 'dust'):
        with pytest.raises(commands.CommandError):
            run(spots=bad)


def test_white_balance_from_a_sampled_patch(tmp_path, monkeypatch):
    monkeypatch.setenv('LUMEN_WORKING', str(tmp_path / 'working'))
    picture = np.full((200, 300, 3), (150, 120, 90), np.uint8)       # a warm cast over a grey wall
    picture[:, 200:] = (60, 120, 200)
    path = tmp_path / 'wall.png'
    Image.fromarray(picture).save(path)
    context = session(tmp_path)
    run = lambda **changes: commands.run(context, 'edit.apply', dict(path=str(path), changes=changes))
    run(adjustments=dict(temperature=30))
    result = run(white_balance=dict(sample=[.3, .5]))
    assert result['state']['white_balance']['sampled'] and 'white balance  sampled' in result['applied']
    assert result['state']['adjustments']['temperature'] == 0
    source, _ = context.source(str(path))
    wall = engine.process(source, context.edits(str(path)))[100, 90]
    assert np.ptp(wall) < .01                                          # the sampled wall is neutral now
    result = run(white_balance=None)
    assert 'white_balance' not in result['state'] and context.edits(str(path))['wb_gain'] == [1., 1., 1.]
    for bad in (dict(sample=[.5]), dict(sample=[.5, 1.5]), dict(kelvin=5000), [.5, .5]):
        with pytest.raises(commands.CommandError):
            run(white_balance=bad)
    black = tmp_path / 'black.png'
    Image.fromarray(np.zeros((100, 100, 3), np.uint8)).save(black)
    with pytest.raises(commands.CommandError):
        commands.run(context, 'edit.apply', dict(path=str(black), changes=dict(white_balance=dict(sample=[.5, .5]))))


def test_preview_of_a_region_at_100_percent(tmp_path, monkeypatch):
    monkeypatch.setenv('LUMEN_WORKING', str(tmp_path / 'working'))
    rng = np.random.default_rng(2)
    blocks = (rng.random((200, 300, 3)) * 200 + 28).astype(np.uint8)
    picture = np.repeat(np.repeat(blocks, 8, axis=0), 8, axis=1)     # 2400 x 1600: larger than the 1600-pixel preview
    path = tmp_path / 'detail.png'
    Image.fromarray(picture).save(path)
    context = session(tmp_path)
    run = lambda **args: commands.run(context, 'render.preview', dict(path=str(path), **args))
    whole = run(size=4096)                                           # the limit is 4096 now
    assert whole['width'] == 1600 and 'region' not in whole
    part = run(region=[.25, .2, .5, .6], size=2048)
    assert (part['width'], part['height']) == (600, 640) and part['scale'] == 1 and part['region'] == [.25, .2, .5, .6]
    image = np.asarray(Image.open(io.BytesIO(base64.b64decode(part['image']))).convert('RGB')).astype(np.float32)
    # True pixels of the original: its 8-pixel blocks are still sharp (the 1600-pixel preview blurs them).
    error = np.abs(image - picture[320:960, 600:1200]).mean()
    blurred = np.asarray(Image.open(io.BytesIO(base64.b64decode(whole['image']))).convert('RGB').resize((2400, 1600)))
    assert error < 10 and np.abs(blurred[320:960, 600:1200].astype(np.float32) - picture[320:960, 600:1200]).mean() > 2 * error
    original = run(region=[.25, .2, .5, .6], original=True)
    assert original['width'] == 600 and original['scale'] == 1
    reduced = run(region=[0, 0, .5, .5], size=512)
    assert reduced['width'] == 512 and reduced['scale'] == pytest.approx(512 / 1200, abs=.001)
    # Edits show in the region as in the whole picture.
    commands.run(context, 'edit.apply', dict(path=str(path), changes=dict(adjustments=dict(exposure=1.))))
    brighter = run(region=[.25, .2, .5, .6])
    assert brighter['statistics'] != part['statistics']
    for bad in ([.5, .2, .4, .6], [0, 0, 1], [0, 0, 1, 2]):
        with pytest.raises(commands.CommandError):
            run(region=bad)
    monkeypatch.setattr(commands, 'REGION_PIXELS', 1000)
    with pytest.raises(commands.CommandError, match='megapixels'):
        run(region=[0, 0, .5, .5])


def test_the_tool_descriptions_name_the_new_vocabulary():
    schema = commands.REGISTRY['edit.apply'].schema()['properties']['changes']['description']
    for word in ('crop {rect', 'shape: "radial"', 'spots [', 'white_balance {sample', 'uncropped photo'):
        assert word in schema
    preview = commands.REGISTRY['render.preview'].schema()['properties']
    assert preview['size']['maximum'] == 4096 and 'region' in preview
    assert commands.EDITING <= set(commands.REGISTRY)
    json.dumps(commands.describe())
