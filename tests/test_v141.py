"""1.4.1: stepless exposure curve, unsupported-RAW fallbacks and batch export."""
import copy
import numpy as np
import pytest
from lumen import engine, exposure_curve as ec, model


@pytest.mark.parametrize('x,dy', [(.15, .08), (.35, .1), (.5, .12), (.65, -.1), (.85, -.12), (.95, -.08), (.3, -.15)])
def test_curve_bends_locally_and_moves_the_sliders_of_that_range(x, dy):
    a = model.adjustments()
    before = ec.display(a, ec.IDENTITY, ec.SAMPLES)
    target = float(ec.display(a, ec.IDENTITY, x)) + dy
    b, points = ec.drag(a, x, target)
    after = ec.display(b, points, ec.SAMPLES)
    assert abs(float(ec.display(b, points, x)) - target) < .003
    assert np.all(np.diff(after) >= -1e-9)
    # Tones far from the pointer stay put (1.4.0 moved them more than the dragged tone).
    far = np.abs(ec.SAMPLES - x) > .4
    assert np.abs(after - before)[far].max() < .01
    moved = {k: b[k] for k in ec.KEYS if b[k] != a[k]}
    assert moved and all(np.sign(v) == np.sign(dy) for v in moved.values())
    # Only sliders whose tone range covers the pointer move (e.g. 亮部 / 白色 near white).
    assert all(ec.membership(x)[ec.KEYS.index(k)] >= .2 for k in moved)


def test_slider_changes_after_a_drag_still_move_the_curve():
    a, points = ec.drag(model.adjustments(), .4, float(ec.display(model.adjustments(), ec.IDENTITY, .4)) + .08)
    curve = ec.display(a, points, ec.SAMPLES)
    lifted = dict(a, shadows=min(100, a['shadows'] + 30))
    assert (ec.display(lifted, points, ec.SAMPLES) - curve)[ec.SAMPLES < .5].max() > .01
    assert np.all(np.diff(ec.display(lifted, points, ec.SAMPLES)) >= -1e-9)


def test_tone_curve_renders_like_the_widget_on_cpu_and_gpu():
    edits = model.recipe()
    edits['adjustments'], edits['tone_curve'] = ec.drag(edits['adjustments'], .55, .7)
    edits['curves']['RGB'] = [[0., 0.], [.5, .45], [1., 1.]]
    axis = np.linspace(0, 1, 513, dtype=np.float32)
    source = np.repeat(engine.to_linear(axis)[None, :, None], 3, axis=2)
    plain = dict(copy.deepcopy(edits), curves=model.recipe()['curves'])
    rendered = engine.process(source, plain, engine.Backend('cpu'))[0, :, 0]
    np.testing.assert_allclose(rendered, ec.display(edits['adjustments'], edits['tone_curve'], axis), atol=2e-3)
    cpu = engine.process(source, edits, engine.Backend('cpu'))
    gpu = engine.Backend('auto')
    if gpu.gpu_pointwise:
        np.testing.assert_allclose(engine.process(source, edits, gpu), cpu, atol=2e-4)
    # The RGB curve still applies after the tone curve.
    assert np.abs(cpu[0, :, 0] - rendered).max() > .01


def test_projects_without_tone_curve_render_unchanged_and_presets_blend_it():
    edits = model.recipe()
    edits['adjustments'].update(exposure=.3, shadows=20)
    legacy = {k: v for k, v in edits.items() if k != 'tone_curve'}
    validated = model.validate(legacy)
    assert validated['tone_curve'] == [[0., 0.], [1., 1.]]
    rgb = np.random.default_rng(2).random((40, 60, 3)).astype(np.float32)
    np.testing.assert_array_equal(engine.process(rgb, validated), engine.process(rgb, edits))
    look = dict(model.extract_look(dict(edits, tone_curve=[[0., 0.], [.5, .7], [1., 1.]])))
    half = model.apply_look(model.recipe(), look, 50)
    assert half['tone_curve'] == [[0., 0.], [.5, .6], [1., 1.]]
    for bad in ([[0., 0.], [.5, .7], [.4, .8], [1., 1.]], [[0., .5], [.5, .2], [1., 1.]], [[.1, 0.], [1., 1.]]):
        with pytest.raises(ValueError):
            model.validate(dict(model.recipe(), tone_curve=bad))


# --- UI: batch export from the filmstrip ---------------------------------------------------------

def _app():
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from lumen.app import STYLE
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(STYLE)
    return app


def _wait(predicate, timeout=60):
    import time
    from PySide6.QtTest import QTest
    start = time.monotonic()
    while not predicate():
        QTest.qWait(20)
        time.sleep(.002)
        assert time.monotonic() - start < timeout, '异步操作超时'


def test_batch_export_uses_each_photos_edits_and_never_overwrites(tmp_path, monkeypatch):
    from PIL import Image
    from PySide6.QtWidgets import QMessageBox, QDialog, QProgressDialog
    from lumen import batch_export
    from lumen.app import MainWindow
    _app()
    messages = []
    monkeypatch.setattr(QMessageBox, 'information', lambda *a: messages.append(a[2]))
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a: messages.append(a[2]))
    paths = []
    for i, color in enumerate(('#506070', '#807060', '#405a40')):
        path = tmp_path / f'照片{i}.png'
        Image.new('RGB', (900, 600), color).save(path)
        paths.append(str(path.resolve()))
    out = tmp_path / '导出'
    out.mkdir()
    (out / '照片1-Lumen.jpg').write_bytes(b'existing')
    w = MainWindow()
    w.show()
    w.import_paths(paths)
    _wait(lambda: not w.loading and not w.render_running and not w.timer.isActive())
    w.controls['exposure'].spin.setValue(1.)
    _wait(lambda: not w.render_running and not w.timer.isActive())
    for i in range(w.filmstrip.count()):
        w.filmstrip.item(i).setSelected(True)
    action = next(a for a in w.film_context_menu().actions() if a.text().startswith('批量导出'))
    assert action.isEnabled() and '3 张' in action.text()
    monkeypatch.setattr(batch_export.BatchExportDialog, 'exec', lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr(batch_export.BatchExportDialog, 'options',
                        lambda self: dict(extension='.jpg', long_edge=400, quality=90, folder=str(out)))
    action.trigger()
    _wait(lambda: not w.exporting)
    files = sorted(p.name for p in out.iterdir())
    assert files == ['照片0-Lumen.jpg', '照片1-Lumen-2.jpg', '照片1-Lumen.jpg', '照片2-Lumen.jpg']
    assert (out / '照片1-Lumen.jpg').read_bytes() == b'existing'
    sizes = {name: Image.open(out / name).size for name in files if name != '照片1-Lumen.jpg'}
    assert set(sizes.values()) == {(400, 267)}
    brightness = {name: np.asarray(Image.open(out / name)).mean() for name in sizes}
    # Only the first photo (opened and edited) carries +1 EV.
    assert brightness['照片0-Lumen.jpg'] > brightness['照片1-Lumen-2.jpg'] + 20
    assert messages and '已导出 3 张' in messages[-1]
    assert not [d for d in w.findChildren(QProgressDialog) if d.isVisible()]
    w.saved_edits = copy.deepcopy(w.edits)
    for document in w.documents.values():
        document['saved_edits'] = copy.deepcopy(document['edits'])
    w.close()


# --- RAW files LibRaw cannot fully read (real samples; skipped where they are absent) ----------

import os
from pathlib import Path
SAMPLES = Path(os.environ.get('LUMEN_SAMPLES', 'E:/LumenSamples'))
HE_NEF = SAMPLES / 'Nikon_Z8_raw_high_efficiency_hight.NEF'
PQ_CR3 = SAMPLES / 'R5m2_RAW.CR3'


def test_pq_signal_maps_reference_white():
    from lumen import previews
    assert abs(float(previews.pq_to_linear(.58)) - 203) < 2
    assert float(previews.pq_to_linear(0.)) == 0 and abs(float(previews.pq_to_linear(1.)) - 10000) < 1


@pytest.mark.skipif(not HE_NEF.exists(), reason='Nikon Z8 High Efficiency sample not available')
def test_high_efficiency_nef_opens_from_its_full_size_embedded_jpeg():
    from lumen import library
    preview, info = engine.load_image(HE_NEF)
    full, _ = engine.load_image(HE_NEF, None)
    assert (info['width'], info['height']) == (8256, 5504) and full.shape == (5504, 8256, 3)
    assert max(preview.shape[:2]) == 1600 and not info['raw'] and info['embedded']
    assert '高效率' in info['note'] and info['photo']['body'] == 'NIKON Z 8'
    assert library.thumbnail(HE_NEF).shape[1] == 120


@pytest.mark.skipif(not PQ_CR3.exists(), reason='Canon EOS R5 Mark II HDR PQ sample not available')
def test_canon_hdr_pq_cr3_uses_its_hevc_preview_for_develop_and_thumbnail():
    from lumen import library, previews
    preview = previews.hevc_preview(PQ_CR3)
    assert preview.shape == (1080, 1620, 3) and 40 < preview.mean() < 200
    _, info = engine.load_image(PQ_CR3)
    assert info['raw'] and 'HDR' in info['develop']['source']
    assert library.thumbnail(PQ_CR3).shape == (80, 120, 3)
