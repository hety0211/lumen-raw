"""1.8.0: float16-weight models, slimmer packages, speech model on first use, parallel CPU
strips, faster gamut mapping and the amber dark / light themes."""
import hashlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from lumen import color, engine, model, onnx_graph, speech, tone
from tools import bundle, compact_models

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- float16 weights

def _matmul_model(rows=64, columns=48, as_inputs=False):
    rng = np.random.default_rng(4)
    graph = onnx_graph.Graph('compact')
    x = graph.input('x', [1, rows])
    weight = graph.const(rng.standard_normal((rows, columns)).astype(np.float32) * .1)
    bias = graph.const(rng.standard_normal(columns).astype(np.float32))
    graph.output(graph.op('Add', graph.op('MatMul', x, weight), bias), 'y', [1, columns])
    if as_inputs:
        # IR < 4 style: initializers are graph inputs too.
        for name, array in graph.initializers.values():
            graph.inputs.append(onnx_graph._value_info(name, onnx_graph.FLOAT, list(array.shape)))
    return graph.serialize()


def _run(data, x):
    import onnxruntime as ort
    session = ort.InferenceSession(data, providers=['CPUExecutionProvider'])
    assert [i.name for i in session.get_inputs()] == ['x']
    return session.run(None, {'x': x})[0]


@pytest.mark.parametrize('as_inputs', [False, True])
def test_large_weights_are_stored_as_float16_and_computed_in_float32(as_inputs):
    data = _matmul_model(as_inputs=as_inputs)
    compact, count = compact_models.compact_model(data)
    assert count == 1                      # the 64 x 48 weight; the 48-value bias stays float32
    assert len(compact) < len(data) * .6
    x = np.random.default_rng(1).standard_normal((1, 64)).astype(np.float32)
    reference, result = _run(data, x), _run(compact, x)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, reference, rtol=2e-3, atol=2e-3)
    assert compact_models.compact_model(compact)[1] == 0    # nothing left to convert


def test_small_or_out_of_range_tensors_stay_float32():
    graph = onnx_graph.Graph('small')
    x = graph.input('x', [1, 4])
    graph.output(graph.op('Mul', x, graph.const(np.full(4, 1e6, np.float32))), 'y', [1, 4])
    data = graph.serialize()
    assert compact_models.compact_model(data) == (data, 0)


def test_compact_folder_skips_unchanged_sources_and_drops_stale_files(tmp_path):
    source, target = tmp_path / 'models', tmp_path / 'compact'
    source.mkdir()
    (source / 'net.onnx').write_bytes(_matmul_model())
    (source / 'info.json').write_text('{}', encoding='utf-8')
    (source / 'u2netp.onnx').write_bytes(_matmul_model(32, 32))
    (source / 'sensevoice').mkdir()
    logged = []
    compact_models.compact_folder(source, target, logged.append)
    assert sorted(p.name for p in target.iterdir()) == ['info.json', 'net.onnx', 'u2netp.onnx']
    assert (target / 'u2netp.onnx').read_bytes() == (source / 'u2netp.onnx').read_bytes()   # kept float32
    assert len(logged) == 1
    (target / 'old.onnx').write_bytes(b'stale')
    compact_models.compact_folder(source, target, logged.append)
    assert len(logged) == 1 and not (target / 'old.onnx').exists()


@pytest.mark.skipif(not (ROOT / 'assets/models/ffdnet-color.onnx').is_file(), reason='runtime assets not installed')
def test_a_bundled_model_keeps_its_output():
    import onnxruntime as ort
    data = (ROOT / 'assets/models/ffdnet-color.onnx').read_bytes()
    compact, count = compact_models.compact_model(data)
    assert count and len(compact) < len(data) * .55
    a = ort.InferenceSession(data, providers=['CPUExecutionProvider'])
    b = ort.InferenceSession(compact, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(2)
    feeds = {i.name: rng.random([d if isinstance(d, int) else 64 for d in i.shape], np.float32) for i in a.get_inputs()}
    assert np.abs(a.run(None, feeds)[0] - b.run(None, feeds)[0]).max() < 5e-3


# ----------------------------------------------------------------------------- package contents

@pytest.mark.parametrize('path, windows, macos', [
    ('PySide6/opengl32sw.dll', True, False),
    ('PySide6/Qt6Quick.dll', True, True),
    ('PySide6/Qt6Pdf.dll', True, True),
    ('PySide6/plugins/platforminputcontexts/qtvirtualkeyboardplugin.dll', True, True),
    ('PySide6/plugins/imageformats/qpdf.dll', True, True),
    ('PySide6/avcodec-61.dll', True, False),
    ('PySide6/plugins/multimedia/ffmpegmediaplugin.dll', True, False),
    ('PySide6/translations/qt_de.qm', True, True),
    ('PySide6/translations/qtmultimedia_ja.qm', True, True),
    ('PySide6/translations/qtbase_zh_CN.qm', False, False),
    ('PySide6/translations/qtbase_ru.qm', False, False),
    ('PySide6/Qt6Network.dll', False, False),
    ('PySide6/Qt6Multimedia.dll', False, False),
    ('PySide6/plugins/multimedia/windowsmediaplugin.dll', False, False),
    ('PySide6/plugins/platforms/qoffscreen.dll', False, False),
    ('PySide6/plugins/platforms/qwindows.dll', False, False),
    ('PySide6/plugins/imageformats/qsvg.dll', False, False),
    ('cv2/opencv_videoio_ffmpeg500_64.dll', False, False),   # CR3 HEVC previews (lumen/previews.py)
])
def test_unused_qt_files_are_left_out_of_the_packages(path, windows, macos):
    assert bundle.unused(path, 'windows') is windows
    assert bundle.unused(path, 'macos') is macos


def test_onnx_runtime_tooling_is_not_collected():
    assert bundle.runtime_module('onnxruntime.capi._pybind_state')
    assert not bundle.runtime_module('onnxruntime.transformers.models.bert')
    assert not bundle.runtime_module('onnxruntime.quantization')


def test_assets_bundle_compact_models_without_the_speech_model(tmp_path, monkeypatch):
    calls = []

    def fake(source, destination, log=print):
        # 2.0.0: never into the real build folder: a dummy model there once reached a package.
        calls.append(Path(source))
        folder = tmp_path / 'compact'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'skyseg.onnx').write_bytes(b'x')
        return folder
    monkeypatch.setattr(bundle, 'compact_folder', fake)
    pairs = bundle.asset_datas(ROOT)
    destinations = {Path(source).name: target for source, target in pairs}
    assert calls == [ROOT / 'assets' / 'models']
    assert destinations['skyseg.onnx'] == 'assets/models'
    assert 'runtime-assets.json' not in destinations and 'speech-assets.json' in destinations
    assert not any('sensevoice' in source for source, _ in pairs)
    assert not any(Path(source) == ROOT / 'assets' / 'models' for source, _ in pairs)


# ----------------------------------------------------------------------------- speech model download

class _Files(BaseHTTPRequestHandler):
    files = {}
    requests = []

    def log_message(self, *_):
        pass

    def do_GET(self):
        self.requests.append((self.path, self.headers.get('Range')))
        data = self.files.get(self.path)
        if data is None:
            self.send_error(404)
            return
        start = 0
        if self.headers.get('Range'):
            start = int(self.headers['Range'].split('=')[1].rstrip('-'))
            self.send_response(206)
        else:
            self.send_response(200)
        self.send_header('Content-Length', str(len(data) - start))
        self.end_headers()
        self.wfile.write(data[start:])


@pytest.fixture
def server(tmp_path, monkeypatch):
    _Files.files, _Files.requests = {}, []
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), _Files)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{httpd.server_address[1]}/'
    model_bytes = os.urandom(300_000)
    tokens = 'a 1\nb 2\n'.encode('utf-8')
    files = []
    for name, data in (('model.int8.onnx', model_bytes), ('tokens.txt', tokens)):
        files.append(dict(path=f'assets/models/sensevoice/{name}', url=f'{base}up/{name}', bytes=len(data),
                          sha256=hashlib.sha256(data).hexdigest()))
    manifest = dict(version='test', mirrors=[base + 'mirror/'], files=files)
    monkeypatch.setattr(speech, 'manifest', lambda: manifest)
    monkeypatch.setattr(speech, 'UPSTREAM', base + 'up/')
    monkeypatch.setattr(speech, 'BUNDLED', tmp_path / 'not-bundled')
    monkeypatch.setattr(speech, 'user_folder', lambda: tmp_path / 'user' / 'sensevoice')
    monkeypatch.setattr(speech, 'CHUNK', 65536)
    yield dict(base=base, model=model_bytes, tokens=tokens, folder=tmp_path / 'user' / 'sensevoice')
    httpd.shutdown()
    httpd.server_close()


def test_the_speech_model_downloads_on_first_use(server, monkeypatch):
    monkeypatch.setattr(speech, 'language', lambda: 'en')
    _Files.files = {'/up/model.int8.onnx': server['model'], '/up/tokens.txt': server['tokens']}
    assert not speech.available()
    reports = []
    folder = speech.download(lambda done, total: reports.append((done, total)))
    assert folder == server['folder'] and speech.model_folder() == folder
    assert (folder / 'model.int8.onnx').read_bytes() == server['model']
    assert reports[-1] == (len(server['model']) + len(server['tokens']),) * 2
    assert all(b >= a for (a, _), (b, _) in zip(reports, reports[1:]))
    assert not list(folder.glob('*.part'))
    # Already there: nothing is fetched again.
    count = len(_Files.requests)
    speech.download()
    assert len(_Files.requests) == count


def test_a_failing_source_falls_back_to_the_next_and_bad_files_are_rejected(server, monkeypatch):
    monkeypatch.setattr(speech, 'language', lambda: 'zh_CN')      # the mirror is tried first
    corrupt = bytearray(server['model'])
    corrupt[10] ^= 1
    _Files.files = {'/mirror/model.int8.onnx': bytes(corrupt), '/up/model.int8.onnx': server['model'],
                    '/up/tokens.txt': server['tokens']}
    speech.download()
    assert (server['folder'] / 'model.int8.onnx').read_bytes() == server['model']
    assert [p for p, _ in _Files.requests][:2] == ['/mirror/model.int8.onnx', '/up/model.int8.onnx']
    _Files.files = {}
    (server['folder'] / 'tokens.txt').unlink()
    with pytest.raises(OSError, match='tokens.txt|404'):
        speech.download()


def test_a_cancelled_download_resumes_where_it_stopped(server, monkeypatch):
    monkeypatch.setattr(speech, 'language', lambda: 'en')
    _Files.files = {'/up/model.int8.onnx': server['model'], '/up/tokens.txt': server['tokens']}
    seen = []
    with pytest.raises(speech.Cancelled):
        speech.download(lambda done, total: seen.append(done), lambda: bool(seen) and seen[-1] >= 100_000)
    part = server['folder'] / 'model.int8.onnx.part'
    have = part.stat().st_size if part.is_file() else 0
    assert 0 < have < len(server['model'])
    speech.download()
    ranges = [r for p, r in _Files.requests if p == '/up/model.int8.onnx']
    assert ranges == [None, f'bytes={have}-']
    assert (server['folder'] / 'model.int8.onnx').read_bytes() == server['model']


def test_the_manifest_lists_the_pinned_files_and_a_mirror():
    data = json.loads((ROOT / 'assets/speech-assets.json').read_text(encoding='utf-8'))
    assert data['mirrors'] and all(m.startswith('https://') and m.endswith('/') for m in data['mirrors'])
    # 2.0.0: LUMEN RAW's own release of the same files.
    assert data['releases'] and all(r.startswith('https://github.com/') and r.endswith('/') for r in data['releases'])
    assert all(f['url'].startswith(speech.UPSTREAM) and len(f['sha256']) == 64 for f in data['files'])
    assert 200 < speech.download_megabytes() < 260


# ----------------------------------------------------------------------------- CPU strips and gamut

def _frame(h=600, w=800):
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.stack([.2 + .6 * xx / w, .1 + .5 * yy / h, .4 + .3 * np.sin(xx / 40)], -1)
    return np.ascontiguousarray(np.clip(rgb * (1 + .05 * rng.standard_normal((h, w, 1))), 0, 1.6).astype(np.float32))


@pytest.mark.parametrize('version', [1, 2])
def test_cpu_strips_give_the_same_render_as_one_strip(version, monkeypatch):
    source = _frame()
    e = model.recipe()
    e['process'] = version
    e['adjustments'].update(exposure=.4, shadows=35, highlights=-30, saturation=12, vibrance=20, contrast=10)
    e['hsl'][5] = [10, 25, -5]
    e['grading'] = dict(e.get('grading', {}), shadows=[220, 30], highlights=[40, 25])
    backend = engine.Backend('cpu')
    stages = []
    original = engine.in_strips

    def counting(image, stage):
        stages.append(image.shape)
        return original(image, stage)
    monkeypatch.setattr(engine, 'in_strips', counting)
    many = engine.process(source, e, backend)
    assert stages
    monkeypatch.setattr(engine, 'STRIP_PIXELS', 1 << 40)
    single = engine.process(source, e, backend)
    np.testing.assert_array_equal(many, single)


def test_strip_spec_samples_the_local_tone_context_at_the_strip_rows():
    source = _frame(300, 400)
    a = dict(model.recipe()['adjustments'], shadows=40, highlights=-40)
    context = tone.context(source, a)
    spec = tone.spec(model.recipe(), context)
    whole = tone.tonal(source, a, spec)
    strip = tone.tonal(source[120:200], a, engine.strip_spec(spec, source.shape, 120, 200))
    np.testing.assert_array_equal(strip, whole[120:200])
    block = engine.Area(400, 300, 50, 100, 350, 260)
    inner = tone.spec(model.recipe(), context, block)
    rows = tone.tonal(source[140:180, 50:350], a, engine.strip_spec(inner, (160, 300), 40, 80))
    np.testing.assert_array_equal(rows, whole[140:180, 50:350])


def test_channel_extremes_match_numpy_reductions():
    x = np.random.default_rng(3).normal(.5, .6, (1000, 3)).astype(np.float32)
    np.testing.assert_array_equal(color._excess(x), np.maximum(x.max(axis=-1) - 1, -x.min(axis=-1)))
    np.testing.assert_array_equal(color._inside(x), (x.min(axis=-1) >= -color.GAMUT_EPS) & (x.max(axis=-1) <= 1 + color.GAMUT_EPS))


# ----------------------------------------------------------------------------- themes and Qt texts

@pytest.fixture(scope='module')
def app():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from lumen.app import STYLE
    instance = QApplication.instance() or QApplication([])
    instance.setStyle('Fusion')
    instance.setStyleSheet(STYLE)
    return instance


@pytest.fixture
def restore_style(app):
    sheet, palette = app.styleSheet(), app.palette()
    yield
    from lumen import theme
    theme._mode = 'dark'
    app.setStyleSheet(sheet)
    app.setPalette(palette)


def test_both_looks_fill_every_stylesheet_token():
    from lumen import theme
    assert set(theme.DARK) == set(theme.LIGHT)
    for look in ('dark', 'light'):
        sheet = theme.stylesheet(look)
        assert '@' not in sheet and theme.PALETTES[look]['accent'] in sheet


def test_appearance_is_saved_applied_and_repaints(app, restore_style, tmp_path, monkeypatch):
    from lumen import theme
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    monkeypatch.setattr(theme, 'system_mode', lambda: 'light')
    assert theme.preference() == 'system'
    assert theme.apply(app) == 'light' and theme.color('canvas').name() == theme.LIGHT['canvas']
    theme.save_preference('dark')
    assert theme.preference() == 'dark' and theme.apply(app) == 'dark'
    assert app.styleSheet() == theme.stylesheet('dark')
    assert theme.color('bubble').alpha() == 0xc8 and theme.color('ink', 20).alpha() == 20
    theme.save_preference('nonsense')
    assert theme.preference() == 'system'


def test_window_switches_appearance_from_the_view_menu(app, restore_style, tmp_path, monkeypatch):
    from PySide6.QtTest import QTest
    from lumen import theme
    from lumen.app import MainWindow
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    w = MainWindow()
    try:
        assert set(w.appearance_actions) == {'system', 'dark', 'light'}
        w.appearance_actions['light'].trigger()
        QTest.qWait(20)
        assert theme.mode() == 'light' and theme.preference() == 'light'
        assert w.appearance_actions['light'].isChecked()
        w.canvas.resize(200, 120)
        assert w.canvas.grab().toImage().pixelColor(5, 5).name() == theme.LIGHT['canvas']
        w.appearance_actions['dark'].trigger()
        assert w.canvas.grab().toImage().pixelColor(5, 5).name() == theme.DARK['canvas']
    finally:
        w.saved_edits = w.edits
        w.close()


def test_selected_thumbnails_are_not_tinted(app):
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QIcon, QImage
    from lumen.widgets import photo_icon
    image = QImage(16, 12, QImage.Format.Format_RGB32)
    image.fill(0xff3060a0)
    icon = photo_icon(image)
    normal = icon.pixmap(QSize(16, 12), QIcon.Mode.Normal).toImage()
    selected = icon.pixmap(QSize(16, 12), QIcon.Mode.Selected).toImage()
    assert normal.pixelColor(4, 4) == selected.pixelColor(4, 4)


def test_qt_dialog_buttons_follow_the_interface_language(app, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from lumen import i18n
    monkeypatch.setattr(i18n, '_language', 'zh_CN')
    translator = i18n.install_qt_translator(app)
    try:
        box = QMessageBox()
        box.setStandardButtons(QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel)
        assert {b.text() for b in box.buttons()} == {'保存', '取消'}
    finally:
        app.removeTranslator(translator)
    monkeypatch.setattr(i18n, '_language', 'en')
    assert i18n.install_qt_translator(app) is None


def test_voice_offers_the_download_and_reports_progress(app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from PySide6.QtTest import QTest
    from lumen.app import MainWindow
    monkeypatch.setattr(speech, 'BUNDLED', tmp_path / 'none')
    monkeypatch.setattr(speech, 'user_folder', lambda: tmp_path / 'user')
    asked = []
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: asked.append(a[2]) or QMessageBox.StandardButton.Yes)
    gate = threading.Event()

    def fake_download(progress, cancelled):
        progress(50, 100)
        gate.wait(5)
        folder = tmp_path / 'user'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'model.int8.onnx').write_bytes(b'm')
        (folder / 'tokens.txt').write_text('a 1', encoding='utf-8')
        return folder
    monkeypatch.setattr(speech, 'download', fake_download)
    w = MainWindow()
    try:
        w.source = np.zeros((4, 4, 3), np.float32)
        w.refresh_natural_language()
        assert w.nl_voice.isEnabled() and 'MB' in w.nl_voice.toolTip()
        w.nl_toggle_voice()
        assert asked and str(speech.download_megabytes()) in asked[0]
        assert w.nl_download is not None and not w.nl_recorder.recording
        QTest.qWait(50)
        assert w.nl_meter.value() == 500 and '正在下载' in w.nl_status.text()
        assert w.nl_voice.isEnabled() and w.nl_voice.text() == '取消下载'
        gate.set()
        for _ in range(100):
            QTest.qWait(20)
            if w.nl_download is None:
                break
        assert w.nl_download is None and speech.available() and not w.nl_meter.isVisible()
    finally:
        gate.set()
        w.source = None
        w.saved_edits = w.edits
        w.close()
