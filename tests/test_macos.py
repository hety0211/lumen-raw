"""1.3.1 macOS: Metal pixel kernels, Core ML provider plan, folders and Mac input handling.

Platform-independent parts (parameter packing, provider plans, Mach-O scanning,
event batching) run everywhere; the Metal / Core ML parity checks need an Apple
silicon Mac and are skipped elsewhere.
"""
import copy
import importlib.util
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from lumen import compute, engine, gpu_graphs, host, metal, model, white_balance

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'assets' / 'models' / 'realesr-general-x4v3.onnx'
MAC = sys.platform == 'darwin'
needs_metal = pytest.mark.skipif(not MAC or metal.pipeline() is None, reason='needs Metal on macOS')


def source(seed=11, shape=(83, 121, 3)):
    rng = np.random.default_rng(seed)
    image = rng.random(shape, dtype=np.float32) ** 2 * 1.2
    image[0, :6] = [[0, 0, 0], [1, 1, 1], [.5, .5, .5], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    return np.ascontiguousarray(image)


def color_edits(mode='smooth', monochrome=False, process=1):
    e = model.recipe()
    e['process'] = process      # the 1.x kernel these checks describe; 1.6.0 recipes default to 2
    e['adjustments'].update(exposure=.6, contrast=-22, shadows=41, highlights=-35, blacks=-18, whites=25,
                            temperature=-30, tint=17, saturation=-12, vibrance=33)
    e['hsl'] = [[20, -30, 15], [0, 40, -20], [-50, 10, 0], [10, 0, 30],
                [0, -60, 0], [35, 20, -10], [0, 0, 0], [-15, 5, 25]]
    e['curves']['RGB'] = [[0., 0.], [.25, .2], [.75, .82], [1., 1.]]
    e['curves']['G'] = [[0., .05], [.5, .55], [1., .95]]
    e['curve_mode'], e['monochrome'] = mode, monochrome
    e['grading'] = dict(shadows=[220., 40.], midtones=[30., 25.], highlights=[45., 60.], balance=20.)
    return e


# ----------------------------------------------------------------------------- everywhere

def test_kernel_parameters_follow_the_graph_inputs():
    e = color_edits()
    inputs = dict(gpu_graphs.tonal_inputs(e['adjustments']), **gpu_graphs.color_inputs(e))
    values, luts = metal.parameters('fused', inputs, 5)
    assert metal._PARAMS.size == 108 and len(values) == 27
    assert luts.dtype == np.float32 and luts.size == 3 * metal.HUE_SAMPLES + 4 * metal.CURVE_SAMPLES
    count, stages, hsl, curves, grading = values[-5:]
    assert (count, stages, hsl, grading) == (5, 3, 1, 1)
    assert curves == 0b0101                      # RGB and G curves; R and B are identities
    assert values[3] == pytest.approx(2 ** .6) and values[8] == pytest.approx(1 - 22 / 125)
    neutral = dict(gpu_graphs.tonal_inputs(model.adjustments()), **gpu_graphs.color_inputs(model.recipe()))
    assert metal.parameters('fused', neutral)[0][-4:] == [3, 0, 0, 0]
    tonal_only, table = metal.parameters('tonal', gpu_graphs.tonal_inputs(e['adjustments']))
    assert tonal_only[-4] == 1 and table.size == 4


def test_kernel_source_declares_the_packed_parameter_block():
    for name in ('gain_r', 'grade[9]', 'uint count', 'uint grading', 'kernel void develop'):
        assert name in metal.SOURCE
    assert 'precise::pow' in metal.SOURCE and 'packed_float3' in metal.SOURCE


def test_coreml_plan_and_options(monkeypatch):
    monkeypatch.setenv('LUMEN_COMPUTE', 'auto')
    monkeypatch.delenv('LUMEN_COREML_UNITS', raising=False)
    available = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    apple = (0, 'Apple M4 Pro', 24 << 30, metal.APPLE, '15.6')
    plan = compute.provider_plan(available, apple, True, models=True)
    assert [p[1] for p in plan] == ['CoreMLExecutionProvider'] and plan[0][2] == 'Apple M4 Pro'
    provider, options = plan[0][0][0]
    assert options['MLComputeUnits'] == 'CPUAndGPU' and options['ModelFormat'] == 'MLProgram'
    assert 'ModelCacheDirectory' not in options          # no persistent copy of every model on disk
    monkeypatch.setenv('LUMEN_COREML_UNITS', 'ALL')
    assert compute.coreml_options()['MLComputeUnits'] == 'ALL'
    monkeypatch.setenv('LUMEN_COREML_UNITS', 'bogus')
    assert compute.coreml_options()['MLComputeUnits'] == 'CPUAndGPU'
    # Pixel graphs, recorded crashes and CPU mode never use Core ML.
    assert compute.provider_plan(available, apple, True, models=False) == []
    assert compute.provider_plan(available, apple, True, models=True, excluded={'CoreMLExecutionProvider'}) == []
    monkeypatch.setenv('LUMEN_COMPUTE', 'metal')
    assert [p[1] for p in compute.provider_plan(available, apple, True, models=True)] == ['CoreMLExecutionProvider']
    monkeypatch.setenv('LUMEN_COMPUTE', 'cpu')
    assert compute.provider_plan(available, apple, True, models=True) == []


def test_shaped_sessions_fix_dynamic_dimensions_per_tile_size(monkeypatch):
    """Core ML gets one static-shape session per input size; results equal the dynamic model."""
    import onnxruntime as ort
    monkeypatch.setattr(compute, 'COREML', 'CPUExecutionProvider')      # the CPU provider stands in for Core ML
    described = compute.signature(ort, str(MODEL))
    assert described[2] == {'height': ('image', 2), 'width': ('image', 3)}
    shaped = compute.ShapedSessions(ort, str(MODEL), ['CPUExecutionProvider'], described, None)
    reference = ort.InferenceSession(str(MODEL), providers=['CPUExecutionProvider'])
    assert [i.shape for i in shaped.get_inputs()] == [[1, 3, 'height', 'width']]
    rng = np.random.default_rng(6)
    sizes = [(24, 31), (16, 16), (24, 31), (8, 12), (12, 8), (20, 20)]
    for h, w in sizes:
        image = rng.random((1, 3, h, w), dtype=np.float32)
        np.testing.assert_allclose(shaped.run(None, {'image': image})[0],
                                   reference.run(None, {'image': image})[0], atol=1e-5)
        assert shaped.current.get_inputs()[0].shape == [1, 3, h, w]
    # Five distinct sizes were used; the four most recently used sessions are kept.
    assert len(shaped.sessions) == shaped.LIMIT == 4
    assert (('image', (1, 3, 16, 16)),) not in shaped.sessions
    assert (('image', (1, 3, 24, 31)),) in shaped.sessions


def test_coreml_housekeeping_clears_the_never_reused_runtime_cache(monkeypatch, tmp_path):
    """Core ML's E5 cache keys on ONNX Runtime's random model paths, so it only ever grows."""
    import tempfile
    cache = tmp_path / 'Library' / 'Caches' / host.BUNDLE_ID / compute.E5_CACHE / '26A428' / 'ABC.bundle'
    cache.mkdir(parents=True)
    (tmp_path / 'T').mkdir()
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(tempfile, 'gettempdir', lambda: str(tmp_path / 'T'))
    monkeypatch.setattr(sys, 'platform', 'darwin')
    # A source run shares Python's cache folder: only entries added while the worker ran go.
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'venv' / 'bin' / 'python'))
    monkeypatch.setattr(compute, '_E5_BEFORE', None)
    python_cache = tmp_path / 'Library' / 'Caches' / 'python' / compute.E5_CACHE / '26A428'
    (python_cache / 'other-program.bundle').mkdir(parents=True)
    compute.coreml_housekeeping()                    # worker start
    (python_cache / 'this-worker.bundle').mkdir()
    compute.coreml_housekeeping()                    # worker stop
    assert [p.name for p in python_cache.iterdir()] == ['other-program.bundle']
    assert cache.exists()                            # the app's folder is not touched by source runs
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    compute.coreml_housekeeping()
    assert not (tmp_path / 'Library' / 'Caches' / host.BUNDLE_ID / compute.E5_CACHE).exists()
    assert (tmp_path / 'Library' / 'Caches' / host.BUNDLE_ID).is_dir()


@pytest.mark.skipif(os.name == 'nt', reason='process liveness via os.kill(pid, 0) is POSIX only')
def test_coreml_housekeeping_removes_models_left_by_dead_workers(monkeypatch, tmp_path):
    import subprocess
    import tempfile
    finished = subprocess.Popen([sys.executable, '-c', 'pass'])
    finished.wait()
    names = {pid: f'onnxruntime-0FF3AB56-5993-45E1-B15D-58F5CDA8B48C-{pid}-000000107D40F406'
             for pid in (finished.pid, os.getpid())}
    for name in names.values():
        (tmp_path / name).mkdir()
        (tmp_path / (name + '.mlmodelc')).mkdir()
    (tmp_path / 'onnxruntime-unrelated').mkdir()
    monkeypatch.setattr(tempfile, 'gettempdir', lambda: str(tmp_path))
    monkeypatch.setattr(sys, 'platform', 'darwin')
    compute.coreml_housekeeping()
    left = sorted(p.name for p in tmp_path.iterdir())
    assert left == sorted([names[os.getpid()], names[os.getpid()] + '.mlmodelc', 'onnxruntime-unrelated'])


def test_platform_folders(monkeypatch, tmp_path):
    monkeypatch.setattr(host, 'MACOS', True)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.delenv('LOCALAPPDATA', raising=False)
    assert host.data_folder() == tmp_path / 'Library' / 'Application Support' / 'LUMEN RAW'
    assert host.log_folder() == tmp_path / 'Library' / 'Logs' / 'LUMEN RAW'
    assert host.cache_folder() == tmp_path / 'Library' / 'Caches' / 'LUMEN RAW'
    # 2.1.0: LOCALAPPDATA, never set on macOS, moves the data folder (the tests' own folders).
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'Tests'))
    assert host.data_folder() == tmp_path / 'Tests' / 'LUMEN RAW'
    monkeypatch.setattr(host, 'MACOS', False)
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'Local'))
    # Windows keeps the 1.3 locations: logs and gpu-compat.json under %LOCALAPPDATA%\LUMEN RAW.
    assert host.log_folder() == tmp_path / 'Local' / 'LUMEN RAW' / 'logs'
    assert host.data_folder() == tmp_path / 'Local' / 'LUMEN RAW'
    from lumen import ai_worker, logs
    assert logs.folder() == host.log_folder()
    assert ai_worker._record_path() == host.data_folder() / 'gpu-compat.json'


def test_key_names(monkeypatch):
    monkeypatch.setattr(host, 'MACOS', True)
    assert host.keys('Ctrl+O') == '⌘O' and host.keys('Ctrl+Shift+Z') == '⌘⇧Z'
    monkeypatch.setattr(host, 'MACOS', False)
    assert host.keys('Ctrl+O') == 'Ctrl+O'


def test_exiftool_command_per_platform(monkeypatch, tmp_path):
    command = white_balance.exiftool()
    folder = ROOT / 'assets' / 'exiftool'
    if os.name == 'nt':
        assert command == ([str(folder / 'exiftool.exe')] if (folder / 'exiftool.exe').exists() else None)
    elif command is not None:
        assert command[1] == str(folder / 'unix' / 'exiftool') and Path(command[0]).name.startswith('perl')
    monkeypatch.setattr(white_balance, 'exiftool', lambda: None)
    assert white_balance.metadata(tmp_path / 'missing.ARW') == ({}, '未安装元数据读取器')


def _macho(minos, platform=1):
    command = struct.pack('<6I', 0x32, 24, platform, minos, minos, 0)
    return struct.pack('<8I', 0xFEEDFACF, 0x0100000C, 0, 6, 1, len(command), 0, 0) + command


def test_minimum_macos_scan(tmp_path):
    spec = importlib.util.spec_from_file_location('package_macos', ROOT / 'tools' / 'package_macos.py')
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    app = tmp_path / 'Test.app' / 'Contents'
    (app / 'MacOS').mkdir(parents=True)
    (app / 'MacOS' / 'main').write_bytes(_macho(13 << 16))
    (app / 'Frameworks').mkdir()
    (app / 'Frameworks' / 'lib.so').write_bytes(_macho((14 << 16) | (2 << 8)))
    (app / 'Frameworks' / 'ios.so').write_bytes(_macho(17 << 16, platform=2))   # not macOS: ignored
    thin = _macho(12 << 16)
    fat = struct.pack('>2I', 0xCAFEBABE, 1) + struct.pack('>5I', 0x0100000C, 0, 28, len(thin), 0) + thin
    (app / 'Frameworks' / 'universal.dylib').write_bytes(fat)
    (app / 'Resources').mkdir()
    (app / 'Resources' / 'model.onnx').write_bytes(b'\x08\x07' * 10)
    version, found = package.minimum(tmp_path / 'Test.app')
    assert version == '14.2' and len(found) == 3


def test_file_open_events_are_batched(qapp_offscreen):
    from PySide6.QtCore import QObject
    from PySide6.QtGui import QFileOpenEvent
    from PySide6.QtTest import QTest
    from lumen.app import FileOpenEvents

    class Window(QObject):
        def __init__(self):
            super().__init__()
            self.calls = []

        def import_paths(self, paths):
            self.calls.append(paths)

    window = Window()
    events = FileOpenEvents(window)
    assert events.eventFilter(None, QFileOpenEvent('/Photos/a.ARW'))
    assert events.eventFilter(None, QFileOpenEvent('/Photos/b.ARW'))
    QTest.qWait(400)
    assert window.calls == [['/Photos/a.ARW', '/Photos/b.ARW']]


def test_canvas_trackpad_pans_and_wheel_zooms_on_macos(qapp_offscreen, monkeypatch):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from lumen.widgets import Canvas
    canvas = Canvas()
    canvas.resize(800, 600)
    canvas.set_image(np.full((300, 400, 3), .5, np.float32))

    def wheel(pixel, angle, modifiers=Qt.KeyboardModifier.NoModifier, phase=Qt.ScrollPhase.NoScrollPhase):
        return QWheelEvent(QPointF(400, 300), QPointF(400, 300), QPoint(*pixel), QPoint(*angle),
                           Qt.MouseButton.NoButton, modifiers, phase, False)

    monkeypatch.setattr(sys, 'platform', 'darwin')
    canvas.wheelEvent(wheel((12, -30), (24, -60), phase=Qt.ScrollPhase.ScrollUpdate))   # trackpad: pan
    assert canvas.zoom == 1 and (canvas.offset.x(), canvas.offset.y()) == (12, -30)
    canvas.wheelEvent(wheel((0, 20), (0, 60)))      # mouse wheel: Qt adds a pixel delta, but it zooms
    assert canvas.zoom == pytest.approx(1.15 ** .5)
    canvas.wheelEvent(wheel((0, 8), (0, 120), Qt.KeyboardModifier.ControlModifier, Qt.ScrollPhase.ScrollUpdate))
    assert canvas.zoom == pytest.approx(1.15 ** 1.5)  # ⌘ + two fingers zooms
    monkeypatch.setattr(sys, 'platform', 'win32')
    canvas.fit()
    canvas.wheelEvent(wheel((0, 0), (0, 30)))
    assert canvas.zoom == pytest.approx(1.15)       # Windows keeps one step per wheel event


@pytest.fixture(scope='module')
def qapp_offscreen():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# ----------------------------------------------------------------------------- Apple silicon

@needs_metal
@pytest.mark.parametrize('mode,monochrome', [('smooth', False), ('linear', False), ('smooth', True)])
def test_metal_stages_match_the_cpu_reference(mode, monochrome):
    image = source()
    e = color_edits(mode, monochrome)
    pipeline = metal.pipeline()
    tonal_inputs = gpu_graphs.tonal_inputs(e['adjustments'])
    tonal = engine.Backend('cpu').tonal(image, e['adjustments'])
    np.testing.assert_allclose(pipeline.run('tonal', image, tonal_inputs), tonal, rtol=2e-5, atol=3e-6)
    expected = engine.color_stage(tonal, e)
    tolerance = 3e-5 if mode == 'linear' else 2e-5
    np.testing.assert_allclose(pipeline.run('color', tonal, gpu_graphs.color_inputs(e)), expected, atol=tolerance)
    fused = dict(tonal_inputs, **gpu_graphs.color_inputs(e))
    np.testing.assert_allclose(pipeline.run('fused', image, fused), expected, atol=tolerance)


@needs_metal
def test_metal_identity_and_strips(monkeypatch):
    image = source(3, (157, 64, 3))
    pipeline = metal.pipeline()
    neutral = gpu_graphs.color_inputs(model.recipe())
    np.testing.assert_allclose(pipeline.run('color', image.clip(0, 1), neutral), image.clip(0, 1), atol=2e-6)
    e = color_edits()
    inputs = dict(gpu_graphs.tonal_inputs(e['adjustments']), **gpu_graphs.color_inputs(e))
    whole = pipeline.run('fused', image, inputs)
    monkeypatch.setattr(metal, 'STRIP_BYTES', 64 * 12 * 10)       # ten rows per dispatch
    np.testing.assert_array_equal(pipeline.run('fused', image, inputs), whole)
    view = np.asfortranarray(image)[::-1]                          # non-contiguous source
    np.testing.assert_array_equal(pipeline.run('fused', view, inputs), whole[::-1])


def masked_edits(process=1):
    e = color_edits(process=process)
    e['adjustments'].update(clarity=30, texture=20, sharpness=40, denoise=15)
    radial = model.new_mask('radial', 1)
    radial.update(start=[.1, .15], end=[.35, .5])
    radial['adjustments'].update(exposure=.6, clarity=40, saturation=-20)
    luminance = model.new_mask('luminance', 2)
    luminance['luminance_range'] = [60., 100.]
    luminance['adjustments'].update(highlights=-30, temperature=15)
    e['masks'] += [radial, luminance]
    return e


@needs_metal
@pytest.mark.parametrize('process', [1, 2])
def test_metal_backend_pipeline_matches_cpu_with_masks(process):
    src = source(9, (90, 130, 3)).clip(0, 1)
    backend = engine.Backend('auto')
    assert backend.metal is not None and backend.gpu_pointwise and backend.name.startswith('Metal')
    for spatial in (False, True):
        e = masked_edits(process)
        e['adjustments']['denoise'] = 0.
        if not spatial:
            for k in engine.SPATIAL_KEYS:
                e['adjustments'][k] = 0.
        gpu = engine.process(src, e, backend, apply_crop=False)
        assert compute.state.snapshot()[0] == metal.PROVIDER
        cpu = engine.process(src, e, engine.Backend('cpu'), apply_crop=False)
        if process == 1:
            np.testing.assert_allclose(gpu, cpu, atol=2e-5)
        else:
            # Process 2's OkLCh pass (cube roots, gamut mapping) in float32: the fused-graph bound
            # of test_v160 and Pipeline.self_test2.
            difference = np.abs(gpu - cpu)
            assert difference.max() < 2e-3 and difference.mean() < 2e-5
    # OpenCV's float bilateral filter (luminance noise reduction) looks colour distances up in a
    # quantized table: a 1e-7 difference from the GPU stage can move a few pixels across a bin.
    e = masked_edits(process)
    difference = np.abs(engine.process(src, e, backend, apply_crop=False)
                        - engine.process(src, e, engine.Backend('cpu'), apply_crop=False))
    assert difference.mean() < 1e-6 and difference.max() < 5e-3 and (difference > 2e-5).mean() < .002


@needs_metal
def test_metal_failure_falls_back_to_cpu(monkeypatch):
    backend = engine.Backend('auto')
    monkeypatch.setattr(backend.metal, 'run', lambda *a: (_ for _ in ()).throw(RuntimeError('device lost')))
    e = color_edits()
    image = source(5, (40, 50, 3))
    result = backend.fused(image, e)
    np.testing.assert_allclose(result, engine.color_stage(engine.Backend('cpu').tonal(image, e['adjustments']), e), atol=1e-6)
    assert backend.metal is None and not backend.gpu_pointwise
    assert 'Metal' in backend.name and 'device lost' in backend.warning


@pytest.mark.skipif(not MAC, reason='macOS memory statistics')
def test_available_memory_on_macos():
    from lumen import large_image
    free = large_image.available_memory()
    assert isinstance(free, int) and 64 << 20 < free < os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')


@pytest.mark.skipif(not MAC or 'CoreMLExecutionProvider' not in __import__('onnxruntime').get_available_providers(),
                    reason='needs Core ML')
def test_coreml_session_runs_and_matches_cpu():
    image = np.random.default_rng(4).random((1, 3, 48, 40), dtype=np.float32)
    accelerated = compute.Session(MODEL, True)
    result = accelerated.run(None, {'image': image})[0]
    assert accelerated.get_providers() == ['CoreMLExecutionProvider']
    cpu = compute.Session(MODEL, False).run(None, {'image': image})[0]
    np.testing.assert_allclose(result, cpu, atol=2e-3)
    assert compute.preferred_adapter()[3] == metal.APPLE
