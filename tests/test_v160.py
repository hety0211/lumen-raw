"""1.6.0: process version 2 (scene-referred tone, OkLCh colour, gamut mapping), the rating
catalog with XMP / Lightroom interchange, the command layer, the control channel and MCP."""
import base64
import copy
import io
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lumen import catalog, color, commands, engine, gpu_graphs, model, tone

SAMPLES = Path(os.environ.get('LUMEN_SAMPLES', 'E:/LumenSamples'))
A7RV = SAMPLES / '7RM5-LosslessCompressedLarge.ARW'


def need(path):
    if not Path(path).is_file():
        pytest.skip(f'sample missing: {path}')


def scene(h=120, w=180, seed=4):
    """Scene-linear test frame with out-of-gamut colours and values above white."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([xx / w, yy / h, .5 + .4 * np.sin(xx / 13)], axis=2) * .8
    texture = rng.normal(0, .03, (h, w, 3)).astype(np.float32)
    x = np.clip(base + texture, 0, None) ** 2.2
    x[:20, :40] = [1.6, .9, .2]          # a bright sunset beyond white
    x[20:40, :40] = [.6, -.04, .02]      # a red outside sRGB
    x[40:60, :40] = [-.02, .1, .55]      # a blue outside sRGB
    return x.astype(np.float32)


def recipe(**adjustments):
    edits = model.recipe()
    edits['develop'] = dict(mode='camera', curve=[[0., 0.], [.08, .22], [.3, .7], [.6, .93], [1., 1.]], source='test')
    edits['adjustments'].update(adjustments)
    return edits


# --------------------------------------------------------------------------- colour

def test_oklab_round_trip_and_white():
    x = np.random.default_rng(1).random((500, 3)).astype(np.float32) * 1.4 - .2
    np.testing.assert_allclose(color.from_oklab(color.to_oklab(x)), x, atol=5e-6)
    np.testing.assert_allclose(color.to_oklab(np.ones((1, 3), np.float32)), [[1, 0, 0]], atol=1e-6)


def test_gamut_mapping_keeps_inside_colours_and_hue_and_lightness_outside():
    rng = np.random.default_rng(2)
    inside = rng.random((400, 3)).astype(np.float32)
    np.testing.assert_array_equal(color.gamut_map(inside), inside)
    outside = np.array([[1.3, .2, .1], [.6, -.05, .03], [-.03, .12, .6], [.2, .9, -.1]], np.float32)
    mapped = color.gamut_map(outside)
    assert mapped.min() >= 0 and mapped.max() <= 1
    before, after = color.to_oklab(outside), color.to_oklab(mapped)
    # Lightness moves only a little (towards mid grey) where a colour is too bright or dark for its chroma.
    np.testing.assert_allclose(after[:, 0], np.clip(before[:, 0], 0, 1), atol=.04)
    hue = lambda lab: np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))
    np.testing.assert_allclose((hue(after) - hue(before) + 180) % 360 - 180, 0, atol=.5)
    # Continuous: a tiny change of the input never jumps a bisection step.
    nudged = color.gamut_map(outside * np.float32(1.00001))
    assert np.abs(nudged - mapped).max() < 1e-4


def test_mixer_hues_follow_the_hsv_circle_in_oklch():
    centers = [color.ok_hue(h) for h in (0, 30, 60, 120, 180, 240, 280, 320)]
    assert all(b > a for a, b in zip(centers, centers[1:]))
    hsl = [[0, 0, 0] for _ in range(8)]
    hsl[0] = [100, 0, 0]                                  # red +100 → 45° towards yellow
    shift, _, _ = color.mixer_tables(hsl)
    red = int(round(centers[0]))
    assert centers[1] - centers[0] < shift[red] < centers[2] - centers[0] + 5


def test_saturating_a_blue_sky_keeps_hue_and_lightness():
    sky = np.full((8, 8, 3), [.42, .62, .88], np.float32)      # display sRGB
    edits = model.recipe()
    edits['adjustments']['saturation'] = 60
    out = engine.color_stage(sky, edits)
    lab = lambda x: color.to_oklab(color.srgb_to_linear(x[0, 0]))
    before, after = lab(sky), lab(out)
    assert abs(after[0] - before[0]) < 6e-3        # beyond sRGB: a little lightness for chroma
    hue = lambda v: np.degrees(np.arctan2(v[2], v[1]))
    assert abs(hue(after) - hue(before)) < 1
    assert np.hypot(*after[1:]) > 1.4 * np.hypot(*before[1:])
    old = copy.deepcopy(edits)
    old['process'] = 1          # 1.x: HSV in gamma space drifts the sky's hue and lightness
    lab_old = lab(engine.color_stage(sky, old))
    assert abs(hue(lab_old) - hue(before)) > 1 or abs(lab_old[0] - before[0]) > 5e-3


def test_monochrome_keeps_lightness_and_toned_grading_survives():
    x = np.random.default_rng(3).random((10, 10, 3)).astype(np.float32)
    edits = model.recipe()
    edits['monochrome'] = True
    grey = engine.color_stage(x, edits)
    assert np.ptp(grey, axis=2).max() < 2e-3
    edits['grading']['highlights'] = [40., 50.]
    toned = engine.color_stage(x, edits)
    assert np.ptp(toned, axis=2).max() > .01


# --------------------------------------------------------------------------- tone

def test_process_two_recovers_highlights_the_develop_curve_compressed():
    x = np.repeat(np.linspace(.2, 1., 200, dtype=np.float32)[None, :, None], 4, axis=0).repeat(3, axis=2)
    contrast = {}
    for version in (1, 2):
        edits = recipe(exposure=-1.)
        edits['process'] = version
        out = engine.process(x, edits)[0, :, 1]
        contrast[version] = float(out[150:].max() - out[150:].min())   # the brightest quarter
    assert contrast[2] > 2 * contrast[1]


def test_values_above_white_and_outside_srgb_are_mapped_not_clipped_per_channel():
    x = scene()
    out = engine.process(x, recipe(exposure=.3))
    assert out.min() >= 0 and out.max() <= 1
    sunset = out[5, 5]
    assert sunset[0] >= sunset[1] >= sunset[2]                     # still orange, not yellow
    lab = color.to_oklab(color.srgb_to_linear(out[30, 10]))
    assert np.degrees(np.arctan2(lab[2], lab[1])) % 360 < 60         # the red keeps a red hue


def test_local_highlights_keep_texture_in_bright_areas():
    rng = np.random.default_rng(9)
    x = np.full((160, 240, 3), .05, np.float32)
    x[:, 120:] = .7 + rng.normal(0, .06, (160, 120, 1)).astype(np.float32)
    flat = recipe(highlights=-100)
    out = engine.process(x, flat)
    texture_local = out[20:140, 140:220, 1].std()
    spec = tone.spec(flat, None)        # per pixel: the 1.x way of weighting
    global_ = engine.color_stage(tone.tonal(x, flat['adjustments'], spec), flat)
    texture_global = global_[20:140, 140:220, 1].std()
    assert texture_local > 1.15 * texture_global
    assert out[80, 60, 1] == pytest.approx(global_[80, 60, 1], abs=.02)   # dark side untouched


def test_tone_context_tracks_exposure_without_recomputing():
    x = scene()
    a = recipe(highlights=-60, shadows=40)['adjustments']
    context = tone.context(x, a)
    for ev in (-1.3, .7):
        moved = dict(a, exposure=ev)
        direct = tone.context(x * np.float32(2 ** ev), moved)
        spec = tone.Spec(tone.identity(), context)
        shifted = tone.maps(spec, x.shape, ev)
        expected = tone.maps(tone.Spec(tone.identity(), direct), x.shape, 0.)
        np.testing.assert_allclose(shifted[..., 1], expected[..., 1], atol=2e-3)


def test_gpu_graphs_match_the_numpy_reference():
    import onnxruntime as ort
    x = scene()
    edits = recipe(exposure=.4, highlights=-50, shadows=35, blacks=-10, whites=12, contrast=18, temperature=9, tint=-4,
                   saturation=15, vibrance=20)
    edits['hsl'][5] = [12., 30., -15.]
    edits['curves']['RGB'] = [[0., 0.], [.5, .56], [1., 1.]]
    edits['curve_mode'] = 'smooth'
    edits['grading'].update(shadows=[215., 30.], highlights=[40., 25.], balance=10.)
    a = edits['adjustments']
    spec = tone.spec(edits, tone.context(x, a))
    tonal = tone.tonal(x, a, spec)
    expected = engine.color_stage(tonal, edits)

    def run(kind, image, inputs):
        session = ort.InferenceSession(gpu_graphs.model(kind), providers=['CPUExecutionProvider'])
        return session.run(None, dict(inputs, image=image[None]))[0][0]
    tonal_inputs = gpu_graphs.tonal2_inputs(a, spec, x.shape)
    np.testing.assert_allclose(run('tonal2', x, tonal_inputs), tonal, atol=2e-4)
    np.testing.assert_allclose(run('color2lab', tonal, gpu_graphs.color2_inputs(edits)), expected, atol=5e-4)
    fused = run('fused2lab', x, dict(tonal_inputs, **gpu_graphs.color2_inputs(edits)))
    assert np.abs(fused - expected).max() < 2e-3 and np.abs(fused - expected).mean() < 2e-5
    plain = model.recipe()
    np.testing.assert_allclose(run('color2', tonal, gpu_graphs.color2_inputs(plain)), engine.color_stage(tonal, plain), atol=2e-6)


def test_block_renders_match_the_whole_frame_with_local_tone_mapping():
    x = scene()
    edits = recipe(highlights=-70, shadows=45, exposure=.2, clarity=20, saturation=10)
    whole = engine.process(x, edits, apply_crop=False)
    for rect in ((0, 0, 180, 120), (31, 17, 97, 88), (150, 90, 180, 120)):
        block = engine.process_region(x, edits, rect=rect)
        x0, y0, x1, y1 = rect
        assert np.abs(block - whole[y0:y1, x0:x1]).max() < 2e-4


def test_large_frames_stream_with_one_tone_context(monkeypatch):
    from lumen import large_image
    x = scene(260, 300)
    edits = recipe(highlights=-60, shadows=30)
    whole = engine.process(x, edits, _stream=False)
    monkeypatch.setattr(large_image, 'STRIP_ROWS', 64)
    streamed = large_image.process(x, edits, engine.Backend('cpu'), True, 1.)
    assert np.abs(streamed - whole).max() < 2e-4


# --------------------------------------------------------------------------- recipes and decoding

def test_recipes_carry_a_process_version():
    fresh = model.recipe()
    assert fresh['version'] == model.VERSION == 6 and fresh['process'] == 2
    old = model.recipe()
    old['version'] = 5
    old.pop('process')
    assert model.validate(old)['process'] == 1
    for bad in (3, 0, True, '2'):
        with pytest.raises(ValueError):
            model.validate(dict(fresh, process=bad))
    look = model.extract_look(fresh)
    assert 'process' not in look and model.apply_look(dict(old, process=1), look)['process'] == 1


def test_process_one_keeps_the_clipped_pipeline_bit_for_bit():
    x = scene()
    edits = recipe(exposure=.3, highlights=-40, saturation=20)
    edits['process'] = 1
    clipped = np.clip(x, 0, 1)
    np.testing.assert_array_equal(engine.process(x, edits), engine.process(clipped, edits))
    np.testing.assert_array_equal(engine.develop_view(x, edits), engine.develop_view(clipped, edits))


def test_raw_decode_keeps_colours_outside_srgb():
    need(A7RV)
    import rawpy
    source, _ = engine.load_image(A7RV)
    assert source.min() < 0
    with rawpy.imread(str(A7RV)) as raw:
        libraw = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16, gamma=(1, 1),
                                 output_color=rawpy.ColorSpace.sRGB, half_size=False, user_flip=None,
                                 highlight_mode=rawpy.HighlightMode.Blend).astype(np.float32) / 65535
    full, _ = engine.load_image(A7RV, None, clip=True)
    # LibRaw truncates to 16 bits; otherwise the conversion is the same.
    assert np.abs(full - libraw).max() < 3e-5


# --------------------------------------------------------------------------- catalog

def test_catalog_log_survives_restarts_torn_lines_and_compaction(tmp_path):
    path = tmp_path / 'catalog.jsonl'
    one = catalog.Catalog(path)
    a, b = str(tmp_path / 'a.ARW'), str(tmp_path / 'b.NEF')
    assert one.set([a, b], rating=3) == [a, b]
    assert one.set([a], rating=3) == []
    one.set([b], flag='reject')
    one.set([a], label='red')
    with open(path, 'ab') as handle:
        handle.write(b'{"path": "torn')                 # a crash in the middle of a write
    two = catalog.Catalog(path)
    assert two.get(a) == dict(rating=3, flag='', label='red') and two.get(b)['flag'] == 'reject'
    one.set([a], rating=5)
    assert two.refresh() and two.get(a)['rating'] == 5
    two.compact()
    assert catalog.Catalog(path).get(a) == dict(rating=5, flag='', label='red')
    with pytest.raises(ValueError):
        one.set([a], rating=7)


def test_queries():
    q = catalog.parse_query
    meta = dict(rating=4, flag='pick', label='red')
    assert catalog.matches(q('rating>=3 flag:pick label:red name:dsc0*'), 'x/DSC01.ARW', meta)
    assert not catalog.matches(q('-flag:pick'), 'x/DSC01.ARW', meta)
    assert catalog.matches(q('★★★ 红色 picked ext:arw'), 'x/DSC01.ARW', meta)
    assert not catalog.matches(q('rating<4'), 'x/a.jpg', meta)
    assert catalog.matches(q('label:none unflagged'), 'x/a.jpg', catalog.empty())
    assert catalog.matches(q('edited'), 'x/a.jpg', catalog.empty(), True)
    with pytest.raises(ValueError):
        q('rating>=many')


def test_label_names_in_other_languages():
    assert [catalog.label_from_text(n) for n in ('Rot', '紅色', 'Violet', 'BLUE', '赤', '')] == \
        ['red', 'red', 'purple', 'blue', 'red', '']


def test_xmp_sidecars_round_trip_without_touching_the_photo(tmp_path):
    from lumen import white_balance
    if white_balance.exiftool() is None:
        pytest.skip('ExifTool missing')
    photo = tmp_path / '照片 one.jpg'
    Image.new('RGB', (8, 8), (120, 90, 60)).save(photo)
    data = photo.read_bytes()
    written, errors = catalog.write_xmp({str(photo): dict(rating=4, label='green', flag='')})
    assert written and not errors and photo.read_bytes() == data
    assert catalog.read_xmp([str(photo)])[str(photo)] == dict(rating=4, label='green')
    catalog.write_xmp({str(photo): dict(rating=2, flag='reject', label='')})
    assert catalog.read_xmp([str(photo)])[str(photo)] == dict(flag='reject')


def test_lightroom_catalog_import(tmp_path):
    lrcat = tmp_path / 'Lightroom Catalog.lrcat'
    db = sqlite3.connect(lrcat)
    db.executescript('''
        CREATE TABLE AgLibraryRootFolder (id_local INTEGER PRIMARY KEY, absolutePath TEXT);
        CREATE TABLE AgLibraryFolder (id_local INTEGER PRIMARY KEY, pathFromRoot TEXT, rootFolder INTEGER);
        CREATE TABLE AgLibraryFile (id_local INTEGER PRIMARY KEY, baseName TEXT, extension TEXT, folder INTEGER);
        CREATE TABLE Adobe_images (id_local INTEGER PRIMARY KEY, rootFile INTEGER, rating REAL, pick REAL,
                                   colorLabels TEXT, masterImage INTEGER);
        INSERT INTO AgLibraryRootFolder VALUES (1, 'D:/Photos/');
        INSERT INTO AgLibraryFolder VALUES (2, '2024/Iceland/', 1);
        INSERT INTO AgLibraryFile VALUES (3, 'DSC0001', 'ARW', 2), (4, 'DSC0002', 'ARW', 2), (5, 'DSC0003', 'NEF', 2);
        INSERT INTO Adobe_images VALUES (10, 3, 5, 1, '红色', NULL), (11, 4, NULL, -1, '', NULL),
                                        (12, 5, 2, 0, 'Blue', NULL), (13, 3, 1, 0, '', 10);
    ''')
    db.commit()
    db.close()
    values = catalog.read_lightroom(lrcat)
    first = str(Path('D:/Photos/2024/Iceland/DSC0001.ARW'))
    assert values[first] == dict(rating=5, flag='pick', label='red')
    assert values[str(Path('D:/Photos/2024/Iceland/DSC0002.ARW'))] == dict(flag='reject')
    assert values[str(Path('D:/Photos/2024/Iceland/DSC0003.NEF'))] == dict(rating=2, label='blue')
    assert len(values) == 3                              # the virtual copy is skipped


# --------------------------------------------------------------------------- commands, CLI and MCP

@pytest.fixture
def photos(tmp_path):
    paths = []
    for i, tint in enumerate(((.8, .5, .3), (.3, .5, .8))):
        yy, xx = np.mgrid[0:120, 0:180]
        rgb = np.stack([xx / 180 * tint[0], yy / 120 * tint[1], .5 * tint[2] + .2 * np.sin(xx / 15)], axis=2)
        path = tmp_path / f'photo {i}.png'
        Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)
        paths.append(str(path))
    return paths


@pytest.fixture
def headless(tmp_path):
    from lumen.session import Session
    return Session(catalog.Catalog(tmp_path / 'catalog.jsonl'), engine.Backend('cpu'))


def test_arguments_are_checked():
    cmd = commands.REGISTRY['library.rate']
    assert commands.arguments(cmd, dict(rating=4.0)) == dict(rating=4)
    for bad in (dict(rating=6), dict(rating='4'), dict(rating=True), dict(), dict(rating=1, colour='red')):
        with pytest.raises(commands.CommandError):
            commands.arguments(cmd, bad)
    with pytest.raises(commands.CommandError):
        commands.run(None, 'no.such.command')
    assert all(c['parameters']['type'] == 'object' for c in commands.describe())


def test_headless_session_edits_previews_exports_and_saves(headless, photos, tmp_path):
    run = lambda command, **args: commands.run(headless, command, args)
    assert run('library.add', paths=[str(tmp_path)])['count'] == 2
    opened = run('photo.open', path=photos[0])
    assert opened['process'] == 2 and opened['width'] == 180
    applied = run('edit.apply', changes=dict(adjustments=dict(exposure=.5, highlights=-30),
                                             hsl=dict(blue=dict(saturation=25)), effects=dict(vignette=-20),
                                             masks=[dict(region='top', adjustments=dict(exposure=-.3))]))
    assert len(applied['applied']) == 5 and applied['state']['adjustments']['exposure'] == .5
    assert run('edit.get')['state']['effects']['vignette'] == -20
    half = run('edit.apply', changes=dict(adjustments=dict(exposure=1.)), amount=50)
    assert half['state']['adjustments']['exposure'] == pytest.approx(.75)
    assert run('edit.undo')['state']['adjustments']['exposure'] == .5
    assert run('edit.redo')['state']['adjustments']['exposure'] == pytest.approx(.75)
    preview = run('render.preview', size=128)
    image = Image.open(io.BytesIO(base64.b64decode(preview['image'])))
    assert image.size == (128, 85) and preview['statistics']
    output = tmp_path / 'out' / 'photo.jpg'
    output.parent.mkdir()
    assert Path(run('photo.export', output=str(output), long_edge=100)['output']).is_file()
    with pytest.raises(commands.CommandError):
        run('photo.export', output=str(output))            # never overwrite without asking
    with pytest.raises(commands.CommandError):
        run('photo.export', output=photos[0])              # never the original
    project = run('project.save')['project']
    assert Path(project).with_suffix('.png') == Path(photos[0]).with_suffix('.png')
    copied = run('edit.copy', to=[photos[1]])['copied']
    assert copied == [str(Path(photos[1]).resolve())]
    assert headless.edits(copied[0])['adjustments']['exposure'] == pytest.approx(.75)
    run('library.rate', rating=4, paths=photos)
    run('library.flag', flag='reject', paths=[photos[1]])
    assert [p['name'] for p in run('library.list', query='rating>=4 -flag:reject')['photos']] == ['photo 0.png']
    batch = run('photo.export_batch', folder=str(output.parent), query='rating>=4', format='png', long_edge=64)
    assert len(batch['exported']) == 2 and not batch['failed']
    # A new session continues from the saved project next to the photo.
    from lumen.session import Session
    again = Session(headless.catalog, engine.Backend('cpu'))
    assert commands.run(again, 'photo.open', dict(path=photos[0]))['state']['adjustments']['exposure'] == pytest.approx(.75)


def test_edit_commands_need_a_photo(headless):
    with pytest.raises(commands.CommandError, match='no photo is open'):
        commands.run(headless, 'edit.get')
    assert commands.run(headless, 'app.status')['mode'] == 'headless'


def mcp_exchange(messages, headless_session=None):
    from lumen import mcp
    out = io.BytesIO()
    dispatcher = mcp.Dispatcher(headless=True)
    dispatcher._session = headless_session
    server = mcp.Server(dispatcher, out)
    server.serve(io.BytesIO(b''.join(json.dumps(m).encode() + b'\n' for m in messages) + b'not json\n'))
    return [json.loads(line) for line in out.getvalue().splitlines()]


def test_mcp_legacy_and_modern_clients(headless, photos):
    modern = {'_meta': {'io.modelcontextprotocol/protocolVersion': '2026-07-28'}}
    replies = mcp_exchange([
        dict(jsonrpc='2.0', id=1, method='initialize', params=dict(protocolVersion='2025-06-18', capabilities={},
                                                                  clientInfo=dict(name='t', version='1'))),
        dict(jsonrpc='2.0', method='notifications/initialized'),
        dict(jsonrpc='2.0', id=2, method='tools/list'),
        dict(jsonrpc='2.0', id=3, method='server/discover', params=modern),
        dict(jsonrpc='2.0', id=4, method='tools/call', params=dict(name='open_photo', arguments=dict(path=photos[0]), **modern)),
        dict(jsonrpc='2.0', id=5, method='tools/call', params=dict(name='render_preview', arguments=dict(size=128))),
        dict(jsonrpc='2.0', id=6, method='tools/call', params=dict(name='rate_photos', arguments=dict(rating=9))),
        dict(jsonrpc='2.0', id=7, method='tools/call', params={'name': 'lumen_status', 'arguments': {},
                                                              '_meta': {'io.modelcontextprotocol/protocolVersion': '1999-01-01'}}),
        dict(jsonrpc='2.0', id=8, method='no/such'),
        dict(jsonrpc='2.0', id=9, method='tools/call', params=dict(name='run_command', arguments=dict(command='edit.presets'))),
    ], headless)
    by_id = {r.get('id'): r for r in replies}
    assert by_id[1]['result']['protocolVersion'] == '2025-06-18' and 'tools' in by_id[1]['result']['capabilities']
    names = [t['name'] for t in by_id[2]['result']['tools']]
    assert {'open_photo', 'apply_edits', 'render_preview', 'export_photo', 'rate_photos', 'run_command'} <= set(names)
    assert all(t['inputSchema']['type'] == 'object' for t in by_id[2]['result']['tools'])
    assert '2026-07-28' in by_id[3]['result']['supportedVersions'] and by_id[3]['result']['resultType'] == 'complete'
    assert by_id[4]['result']['structuredContent']['width'] == 180 and by_id[4]['result']['resultType'] == 'complete'
    content = by_id[5]['result']['content']
    assert content[0]['type'] == 'image' and content[0]['mimeType'] == 'image/jpeg'
    assert by_id[6]['result']['isError'] and 'rating' in by_id[6]['result']['content'][0]['text']
    assert by_id[7]['error']['code'] == -32022
    assert by_id[8]['error']['code'] == -32601
    assert by_id[9]['result']['structuredContent']['presets']
    assert by_id[None]['error']['code'] == -32700       # the stray non-JSON line


def test_cli_render_and_config(photos, tmp_path, capsys):
    from lumen import cli
    output = tmp_path / 'render.png'
    assert cli.main(['render', photos[0], str(output), '--size', '90', '--set', 'exposure=0.4',
                     '--set', 'hsl.blue.saturation=20', '--preset', '金色时刻', '--amount', '50']) == 0
    assert Image.open(output).size == (90, 60)
    assert cli.main(['render', photos[0], str(output)]) == 1                      # exists: needs --overwrite
    capsys.readouterr()
    assert cli.main(['mcp-config', 'json']) == 0
    config = json.loads(capsys.readouterr().out)
    assert config['mcpServers']['lumen-raw']['args'][-1] == 'mcp'
    assert cli.main(['mcp-config', 'claude-code']) == 0
    assert capsys.readouterr().out.startswith('claude mcp add lumen-raw -- ')


# --------------------------------------------------------------------------- the window

@pytest.fixture(scope='module')
def app():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from lumen.app import STYLE
    instance = QApplication.instance() or QApplication([])
    instance.setStyle('Fusion')
    instance.setStyleSheet(STYLE)
    return instance


def wait_until(predicate, timeout=20):
    import time
    from PySide6.QtTest import QTest
    start = time.monotonic()
    while not predicate():
        QTest.qWait(20)
        time.sleep(.002)
        if time.monotonic() - start > timeout:
            raise AssertionError('timeout')


@pytest.fixture
def window(app, photos, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from lumen.app import MainWindow
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a: None)
    w = MainWindow()
    w.show()
    w.import_paths(photos)
    wait_until(lambda: w.rendered is not None and not w.render_running)
    yield w
    w.saved_edits = copy.deepcopy(w.edits)
    for document in w.documents.values():
        document['saved_edits'] = copy.deepcopy(document['edits'])
    w.library_structure_dirty = False
    w.close()


def test_ratings_in_the_filmstrip(window, photos):
    from lumen import rating
    from PySide6.QtCore import Qt
    w = window
    w.filmstrip.clearSelection()
    w.film_item(str(Path(photos[1]).resolve())).setSelected(True)
    w.rate_selected(rating=4)
    w.rate_selected(label='red', toggle=True)
    item = w.film_item(str(Path(photos[1]).resolve()))
    wait_until(lambda: item.data(rating.RATING_ROLE) == 4)
    assert item.data(rating.LABEL_ROLE) == 'red'
    w.rate_selected(label='red', toggle=True)                 # the same label again clears it
    wait_until(lambda: not item.data(rating.LABEL_ROLE))
    w.film_search.setText('rating>=4')
    assert item.isHidden() is False and w.film_item(str(Path(photos[0]).resolve())).isHidden()
    w.film_search.setText('')
    assert w.catalog.get(photos[1])['rating'] == 4
    assert w.rating_bar.rating == 4


def test_agents_drive_the_window_through_the_control_channel(window, photos):
    import threading
    from lumen import control
    w = window
    assert w.control.listening
    results = {}

    def agent():
        client = control.Client(connect_timeout=2)
        try:
            results['status'] = client.call('app.status')
            results['apply'] = client.call('edit.apply', dict(changes=dict(adjustments=dict(exposure=.6)), label='agent'))
            results['rate'] = client.call('library.rate', dict(rating=3))
            try:
                client.call('library.rate', dict(rating=11))
            except commands.CommandError as exc:
                results['error'] = str(exc)
        finally:
            client.close()
    thread = threading.Thread(target=agent)
    thread.start()
    wait_until(lambda: not thread.is_alive(), 60)
    assert results['status']['mode'] == 'gui' and results['status']['current'] == w.source_path
    assert w.edits['adjustments']['exposure'] == .6
    assert 'rating' in results['error'] and w.catalog.get(w.source_path)['rating'] == 3
    w.undo(-1)                                                # the agent's edit is one undo step
    assert w.edits['adjustments']['exposure'] == 0
    assert any('agent' in text for _, text in w.agent_log)


def test_process_upgrade_and_palette(window):
    from lumen.palette import CommandPalette, TITLES
    w = window
    assert w.edits['process'] == 2 and not w.process_button.isVisible()
    w.edits['process'] = 1
    w.refresh()
    assert w.process_button.isVisibleTo(w)
    w.upgrade_process()
    assert w.edits['process'] == 2
    w.undo(-1)
    assert w.edits['process'] == 1
    palette = CommandPalette(w)
    assert palette.list.count() == len(commands.REGISTRY) and set(TITLES) == set(commands.REGISTRY)
    palette.search.setText('rate')
    visible = [palette.list.item(i).data(256) for i in range(palette.list.count()) if not palette.list.item(i).isHidden()]
    assert 'library.rate' in visible and 'edit.auto' not in visible
