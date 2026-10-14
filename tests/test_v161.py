"""1.6.1: AI mask edges follow the photograph; local tone mapping leaves no visible halo."""
import os
import cv2
import numpy as np
from lumen import engine, model, selection, tone


def test_local_tone_mapping_leaves_no_halo_beside_a_one_stop_edge():
    # A subject one stop darker than a flat sky: up to 1.6.0 (epsilon 0.2 stops squared) the sky
    # beside it was ~7/255 brighter than the sky far away, a halo around people and objects.
    h, w = 800, 1200
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    subject = ((xx - 600) ** 2 / 160 ** 2 + (yy - 520) ** 2 / 280 ** 2) < 1
    x = np.full((h, w, 3), .6, np.float32)
    x[subject] = .3
    edits = model.recipe()
    edits['develop'] = dict(mode='linear', curve=[[0., 0.], [1., 1.]])
    edits['adjustments'].update(highlights=-60, shadows=40)
    assert edits['process'] == 2 and tone.CONTEXT_EPS == .05
    sky = engine.process(x, edits).mean(axis=2)[520, 760:]
    assert (sky.max() - sky[-60:].mean()) * 255 < 4


def test_refined_ai_mask_edge_follows_the_photo_not_the_network_ramp():
    # A dark subject on a light background, and what a 320 × 320 network returns for it:
    # the shape at its own resolution, softened over a couple of its pixels.
    h,w=1067,1600
    yy,xx=np.mgrid[0:h,0:w]
    distance=np.hypot(xx-800,yy-560)-380
    inside=distance<0
    rgb=np.where(inside[...,None],np.float32([.18,.14,.12]),np.float32([.75,.8,.85]))
    rgb=(rgb+np.random.default_rng(4).normal(0,.02,rgb.shape)).astype(np.float32)
    net=cv2.GaussianBlur(cv2.resize(inside.astype(np.float32),(320,320),interpolation=cv2.INTER_AREA),(0,0),1.5)
    alpha=selection.refine(net,rgb)
    # Up to 1.6.0 the ramp stayed ~10 pixels wide, mostly on the background side: darkening the
    # background left a light outline around the subject.
    assert np.percentile(alpha[(distance>4)&(distance<40)],99)<.05
    assert np.percentile(alpha[(distance<-4)&(distance>-40)], 1)>.95



# ----------------------------------------------------------------------------- issue #6

from test_ui import app, window, wait_until  # noqa: E402,F401


def _wheel(widget, global_point, notches=-1):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication
    local = widget.mapFromGlobal(global_point)
    event = QWheelEvent(QPointF(local), QPointF(global_point), QPoint(0, 0), QPoint(0, 120 * notches),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(widget, event)


def test_scrolling_a_panel_does_not_change_the_slider_arriving_under_the_pointer(window):
    from PySide6.QtGui import QCursor
    from PySide6.QtWidgets import QScrollArea
    w = window
    w.resize(1600, 700)
    w.tabs.setCurrentIndex(0)
    wait_until(lambda: not w.render_running)
    area = w.tabs.currentWidget()
    assert isinstance(area, QScrollArea) and area.verticalScrollBar().maximum() > 0
    control = w.controls['exposure']
    point = control.slider.mapToGlobal(control.slider.rect().center())
    QCursor.setPos(point)
    point = QCursor.pos()
    bar = area.verticalScrollBar()
    before, top = control.spin.value(), bar.value()
    # The wheel starts on the page (pointer resting), then a slider sits under the pointer.
    _wheel(area.viewport(), point)
    assert bar.value() > top
    scrolled = bar.value()
    _wheel(control.slider, point)
    assert control.spin.value() == before and bar.value() > scrolled
    # Moving the pointer onto the slider on purpose lets the wheel adjust it; Ctrl+Z undoes it.
    point = point + type(point)(20, 0)
    QCursor.setPos(point)
    _wheel(control.slider, QCursor.pos(), notches=1)
    assert control.spin.value() != before and w.edits['adjustments']['exposure'] != before
    w.undo(-1)
    assert w.edits['adjustments']['exposure'] == before


# ----------------------------------------------------------------------------- issue #5

def _exiftool(*arguments):
    """ExifTool with UTF-8 file names (Windows passes command-line arguments in the ANSI code page)."""
    import subprocess
    import tempfile
    from lumen.white_balance import exiftool
    with tempfile.NamedTemporaryFile('w', suffix='.args', delete=False, encoding='utf-8') as handle:
        handle.write('\n'.join(map(str, arguments)))
    try:
        result = subprocess.run([*exiftool(), '-charset', 'filename=UTF8', '-@', handle.name], capture_output=True)
    finally:
        os.unlink(handle.name)
    assert result.returncode == 0, result.stderr
    return result.stdout


def _tags(path, *names):
    import json
    record = json.loads(_exiftool('-j', '-G1', '-n', *names, path).decode('utf-8'))[0]
    record.pop('SourceFile')
    return record


def _tagged_original(tmp_path):
    """A camera JPEG: shooting data, orientation 6, GPS, keywords, Lightroom develop settings."""
    from PIL import Image
    original = tmp_path / '原片.jpg'
    Image.new('RGB', (64, 48), (120, 130, 140)).save(original)
    _exiftool('-overwrite_original', '-Make=SONY', '-Model=ILCE-7M3', '-LensModel=FE 24-70mm F2.8 GM',
              '-DateTimeOriginal=2025:10:01 08:30:00', '-ExposureTime=0.004', '-FNumber=5.6', '-ISO=200',
              '-Copyright=Example', '-Orientation#=6', '-ExifImageWidth=6000', '-ExifImageHeight=4000',
              '-GPSLatitude=31.2', '-GPSLatitudeRef=N', '-GPSLongitude=121.5', '-GPSLongitudeRef=E',
              '-XMP-dc:Subject=landscape', '-IPTC:Keywords=landscape', '-XMP-crs:Exposure2012=+1.0',
              '-XMP-exif:GPSLatitude=31.2', original)
    _exiftool('-o', original.with_suffix('.xmp'), '-XMP-xmp:Label=Red', '-XMP-xmp:Rating=4')
    return original


def test_exports_keep_the_originals_metadata_but_not_what_no_longer_applies(tmp_path):
    import pytest
    from lumen import dng, photo_metadata
    from lumen.white_balance import exiftool
    if exiftool() is None:
        pytest.skip('ExifTool is not installed')
    original = _tagged_original(tmp_path)
    picture = np.full((40, 30, 3), .4, np.float32)
    photo = dict(body='ILCE-7M3')
    for ext in ('.jpg', '.png', '.tif', '.dng'):
        target = tmp_path / f'成片{ext}'
        assert engine.export_image(target, picture, 90, photo=photo, metadata=dict(source=original)).notes == ()
        tags = _tags(target, '-all')
        assert tags['IFD0:Make'] == 'SONY' and tags['IFD0:Model'] == 'ILCE-7M3' and tags['IFD0:Copyright'] == 'Example'
        assert tags['ExifIFD:DateTimeOriginal'] == '2025:10:01 08:30:00' and tags['ExifIFD:ISO'] == 200
        assert tags['ExifIFD:LensModel'] == 'FE 24-70mm F2.8 GM' and abs(tags['ExifIFD:ExposureTime'] - .004) < 1e-9
        assert tags['GPS:GPSLatitude'] == pytest.approx(31.2) and tags['XMP-dc:Subject'] == 'landscape'
        assert tags['XMP-xmp:Label'] == 'Red' and tags['XMP-xmp:Rating'] == 4          # from the .xmp sidecar
        assert tags['IFD0:Software'].startswith('LUMEN RAW ')
        # The export is already upright, sized and sRGB; Lightroom must not develop it again.
        assert tags.get('IFD0:Orientation', 1) == 1 and 'ExifIFD:ExifImageWidth' not in tags
        assert not any(k.startswith(('XMP-crs:', 'IFD1:')) for k in tags)
        if ext != '.dng':
            assert tags['ExifIFD:ColorSpace'] == 1
        if ext in ('.tif', '.dng'):
            assert photo_metadata.embedded(target)['body'] == 'ILCE-7M3'      # LUMEN's own description kept
    assert dng.is_rendered(tmp_path / '成片.dng')
    assert engine.load_image(tmp_path / '成片.dng')[0].shape == (40, 30, 3)
    # Without the location, neither the EXIF nor the XMP position is written; without metadata, nothing.
    engine.export_image(tmp_path / 'no-gps.jpg', picture, metadata=dict(source=original, location=False))
    tags = _tags(tmp_path / 'no-gps.jpg', '-all')
    assert tags['IFD0:Make'] == 'SONY' and not any('GPS' in k for k in tags)
    engine.export_image(tmp_path / 'plain.jpg', picture)
    assert 'IFD0:Make' not in _tags(tmp_path / 'plain.jpg', '-all')


def test_batch_export_copies_metadata_and_reports_what_it_could_not_write(tmp_path):
    import pytest
    from lumen import batch_export
    from lumen.white_balance import exiftool
    if exiftool() is None:
        pytest.skip('ExifTool is not installed')
    original = _tagged_original(tmp_path)
    out = tmp_path / 'out'
    out.mkdir()

    class Never:
        def is_set(self):
            return False
    options = dict(extension='.jpg', long_edge=None, quality=90, folder=str(out), metadata=True, location=False)
    (path, target, note), = batch_export.run([(str(original), model.recipe(), False)], options, engine.Backend('cpu'),
                                             lambda *_: None, Never())
    tags = _tags(target, '-all')
    assert note == '' and tags['ExifIFD:ISO'] == 200 and not any('GPS' in k for k in tags)
    # A photo ExifTool cannot read still exports; the reason comes back with the result.
    broken = tmp_path / 'not-a-photo.jpg'
    from lumen import photo_metadata
    assert photo_metadata.copy_from(broken, target) != ''


def test_window_export_writes_the_originals_metadata(window, tmp_path):
    import pytest
    from lumen.enhance_dialog import ExportDialog
    from lumen.white_balance import exiftool
    if exiftool() is None:
        pytest.skip('ExifTool is not installed')
    w = window
    original = _tagged_original(tmp_path)
    w.open_path(str(original))
    wait_until(lambda: not w.loading and not w.render_running)
    dialog = ExportDialog(w, w.source_path)
    choices = dialog.metadata
    choices.keep.setChecked(True)
    assert choices.location.isEnabled()
    choices.keep.setChecked(False)
    assert not choices.location.isEnabled() and choices.options()['metadata'] is False
    dialog.reject()
    output = tmp_path / '导出.jpg'
    w.start_export(str(output), location=False)
    wait_until(lambda: not w.exporting)
    tags = _tags(output, '-all')
    assert tags['IFD0:Make'] == 'SONY' and tags['ExifIFD:ISO'] == 200 and not any('GPS' in k for k in tags)
    assert tags.get('IFD0:Orientation', 1) == 1
    assert '已保存' in w.test_messages[-1] and '元数据' not in w.test_messages[-1]   # no metadata note
