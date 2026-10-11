import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import copy
import time
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import Qt, QPoint, QPointF
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog
from lumen.app import MainWindow, STYLE
from lumen import model, engine


@pytest.fixture(scope='session')
def app():
    instance = QApplication.instance() or QApplication([])
    instance.setStyle('Fusion')
    instance.setStyleSheet(STYLE)
    return instance


def wait_until(predicate, timeout=15):
    start = time.monotonic()
    while not predicate():
        QTest.qWait(20)
        # PySide's synthetic wait can retain the GIL on Windows; yield so the
        # Python part of worker jobs can progress as it does under QApplication.exec.
        time.sleep(.002)
        if time.monotonic() - start > timeout:
            raise AssertionError('异步操作超时')


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    messages = []
    monkeypatch.setattr(QMessageBox, 'information', lambda *a: messages.append(a[2]))
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a: messages.append(a[2]))
    p = tmp_path / '测试原片.png'
    yy, xx = np.mgrid[0:240, 0:360]
    rgb = np.stack([xx / 360, yy / 240, .35 + .2 * np.sin(xx / 30)], axis=2)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(p)
    w = MainWindow()
    w.show()
    w.open_path(str(p))
    wait_until(lambda: w.rendered is not None and not w.render_running)
    w.test_messages = messages
    yield w
    w.saved_edits = copy.deepcopy(w.edits)
    w.saved_snapshots = copy.deepcopy(w.snapshots)
    wait_until(lambda: not w.exporting and not w.loading and not w.render_running)
    w.close()
    QTest.qWait(30)


def test_pending_selection_cannot_be_omitted_by_export_or_close(window, tmp_path):
    from PySide6.QtGui import QCloseEvent
    w = window
    w.selection_busy = True
    try:
        w.export()
        w.start_export(str(tmp_path / 'premature.dng'))
        assert not w.exporting and not (tmp_path / 'premature.dng').exists()
        event = QCloseEvent()
        w.closeEvent(event)
        assert not event.isAccepted()
    finally:
        w.selection_busy = False


def test_open_adjust_undo_redo_and_stale_preview(window):
    w = window
    w.controls['exposure'].spin.setValue(1)
    w.commit()
    w.undo(-1)
    assert w.edits['adjustments']['exposure'] == 0
    w.undo(1)
    assert w.edits['adjustments']['exposure'] == 1
    for value in [.5, 1.2, -.25]:
        w.controls['exposure'].spin.setValue(value)
    wait_until(lambda: not w.timer.isActive() and not w.render_running)
    expected = engine.process(w.source, w.edits, apply_crop=False, detail_scale=.45)
    np.testing.assert_allclose(w.rendered, expected, atol=1e-5)
    assert not w.test_messages


def test_actual_canvas_crop_brush_curve_gestures(window):
    w = window
    w.tabs.setCurrentIndex(5)
    a = w.canvas.screen([.1, .1]).toPoint()
    b = w.canvas.screen([.8, .8]).toPoint()
    QTest.mousePress(w.canvas, Qt.MouseButton.LeftButton, pos=a)
    QTest.mouseMove(w.canvas, b)
    QTest.mouseRelease(w.canvas, Qt.MouseButton.LeftButton, pos=b)
    assert w.edits['crop'] is not None
    w.tabs.setCurrentIndex(4)
    w.add_mask('brush')
    a = w.canvas.screen([.4, .4]).toPoint()
    b = w.canvas.screen([.6, .6]).toPoint()
    QTest.mousePress(w.canvas, Qt.MouseButton.LeftButton, pos=a)
    QTest.mouseMove(w.canvas, b)
    QTest.mouseRelease(w.canvas, Qt.MouseButton.LeftButton, pos=b)
    assert len(w.selected_mask()['strokes']) == 1
    w.local_controls['exposure'].spin.setValue(.8)
    assert w.selected_mask()['adjustments']['exposure'] == .8
    w.tabs.setCurrentIndex(1);w.tabs.widget(1).ensureWidgetVisible(w.curve)
    QTest.mouseDClick(w.curve, Qt.MouseButton.LeftButton, pos=w.curve.screen([.5, .6]).toPoint())
    assert len(w.edits['curves']['RGB']) == 3
    w.rotate()
    assert w.final_view.isChecked() and w.canvas.tool == 'view'


def test_project_reopen_and_export(window, tmp_path):
    w = window
    w.edits['crop'] = [.1, .1, .9, .9]
    w.edits['rotation'] = 1
    w.edits['adjustments']['exposure'] = .5
    w.project_path = str(tmp_path / '编辑.lumen')
    assert w.save()
    w.open_path(w.project_path)
    wait_until(lambda: not w.loading and not w.render_running and not w.timer.isActive())
    assert w.edits['adjustments']['exposure'] == .5
    output = tmp_path / '导出.png'
    before = Path(w.source_path).read_bytes()
    w.start_export(str(output), 2)
    wait_until(lambda: not w.exporting)
    assert output.exists() and Path(w.source_path).read_bytes() == before
    # The window-modal progress dialog must close with the export, not linger over the editor.
    assert not [d for d in w.findChildren(QProgressDialog) if d.isVisible()]
    assert QApplication.activeModalWidget() is None
    with Image.open(output) as img:
        assert img.size == (384, 576)


def test_failed_open_leaves_previous_document(window, tmp_path):
    w = window
    path = w.source_path
    bad = tmp_path / 'bad.arw'
    bad.write_bytes(b'bad')
    w.open_path(str(bad))
    wait_until(lambda: not w.loading)
    assert w.source_path == path and w.save_button.isEnabled()
    assert '无法打开文件' in w.test_messages[-1]


def test_preset_strength_snapshot_and_persistent_project(window, tmp_path):
    w = window
    w.choose_preset(2)
    expected = w.edits['adjustments']['temperature']
    assert expected > 0 and w.active_preset == 2
    w.preset_amount.spin.setValue(50)
    assert w.edits['adjustments']['temperature'] == expected / 2
    w.add_snapshot('暖阳')
    w.choose_preset(1)
    assert w.edits['adjustments']['temperature'] < 0
    w.snapshot_list.setCurrentRow(0)
    w.restore_snapshot()
    assert w.edits['adjustments']['temperature'] == expected / 2
    w.project_path = str(tmp_path / '版本.lumen')
    assert w.save()
    w.open_path(w.project_path)
    wait_until(lambda: not w.loading and not w.render_running and not w.timer.isActive())
    assert len(w.snapshots) == 1 and w.snapshots[0]['name'] == '暖阳'


def test_white_balance_sample_and_split_gestures(window):
    w = window
    w.wb_button.setChecked(True)
    assert w.canvas.tool == 'sample'
    point = [.5, .5]
    expected = engine.sample_white_balance(w.source, point)
    QTest.mouseClick(w.canvas, Qt.MouseButton.LeftButton, pos=w.canvas.screen(point).toPoint())
    np.testing.assert_allclose(w.edits['wb_gain'], expected, atol=.03)
    assert not w.wb_button.isChecked()
    w.split_check.setChecked(True)
    before = w.canvas.split_position
    QTest.mousePress(w.canvas, Qt.MouseButton.LeftButton, pos=w.canvas.screen([before, .5]).toPoint())
    QTest.mouseMove(w.canvas, w.canvas.screen([.7, .5]).toPoint())
    QTest.mouseRelease(w.canvas, Qt.MouseButton.LeftButton, pos=w.canvas.screen([.7, .5]).toPoint())
    assert w.canvas.split_position > .65 and w.canvas.before is not None
    w.toggle_clipping()
    assert w.canvas.clipping is not None


def test_luminance_mask_and_color_wheel_gestures(window):
    w = window
    w.tabs.setCurrentIndex(4)
    w.add_mask('luminance')
    w.range_controls['low'].spin.setValue(70)
    assert w.selected_mask()['luminance_range'] == [70., 100.]
    assert w.range_box.isVisible() and w.canvas.overlay is not None
    w.tabs.setCurrentIndex(6)
    wheel = w.wheels['shadows']
    center, radius = wheel.center_radius()
    pos = QPoint(round(center.x() + radius * .5), round(center.y()))
    QTest.mouseClick(wheel, Qt.MouseButton.LeftButton, pos=pos)
    assert w.edits['grading']['shadows'][1] > 40
    assert w.grade_zone.currentIndex() == 0
    w.effect_controls['grain'].spin.setValue(25)
    assert w.edits['effects']['grain'] == 25
    w.straighten.spin.setValue(3)
    assert w.edits['straighten'] == 3 and w.final_view.isChecked()


def test_free_curve_drag_precise_tone_and_undo(window):
    w = window
    w.tabs.setCurrentIndex(1);w.tabs.widget(1).ensureWidgetVisible(w.curve)
    QTest.qWait(50)
    for tone in (.25, .5, .75):
        a = w.curve.screen([tone,tone]).toPoint()
        b = w.curve.screen([tone,tone+.1]).toPoint()
        QTest.mousePress(w.curve, Qt.MouseButton.LeftButton, pos=a)
        QTest.mouseMove(w.curve,b)
        QTest.mouseRelease(w.curve,Qt.MouseButton.LeftButton,pos=b)
    assert len(w.edits['curves']['RGB']) == 5
    w.curve_input.setValue(127)
    w.curve_output.setValue(173)
    assert any(abs(x-127/255)<1e-9 and abs(y-173/255)<1e-9 for x,y in w.edits['curves']['RGB'])
    w.undo(-1)
    assert not any(abs(x-127/255)<1e-9 and abs(y-173/255)<1e-9 for x,y in w.edits['curves']['RGB'])


def test_heal_clone_actual_gestures_disable_and_undo(window):
    w = window
    w.tabs.setCurrentIndex(8)
    QTest.qWait(50)
    assert w.canvas.tool == 'heal'
    target = w.canvas.screen([.5,.5]).toPoint()
    QTest.mouseClick(w.canvas,Qt.MouseButton.LeftButton,pos=target)
    assert len(w.edits['retouch']) == 1 and w.edits['retouch'][0]['kind'] == 'heal'
    w.retouch_tool.setCurrentIndex(1)
    QTest.mouseClick(w.canvas,Qt.MouseButton.LeftButton,pos=target)
    assert len(w.edits['retouch']) == 1
    QTest.mouseClick(w.canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.AltModifier,pos=w.canvas.screen([.25,.25]).toPoint())
    QTest.mouseClick(w.canvas,Qt.MouseButton.LeftButton,pos=target)
    assert len(w.edits['retouch']) == 2
    np.testing.assert_allclose(w.edits['retouch'][1]['offset'],[-.25,-.25],atol=.01)
    w.retouch_enabled.setChecked(False)
    assert not w.edits['retouch'][1]['enabled']
    w.undo(-1)
    assert w.edits['retouch'][1]['enabled']
    w.remove_retouch()
    assert len(w.edits['retouch']) == 1
    w.tabs.setCurrentIndex(4)
    w.add_mask('brush')
    assert w.canvas.brush_radius == w.brush_size.spin.value()/200
    wait_until(lambda: not w.timer.isActive() and not w.render_running)
    assert not w.test_messages


def test_kelvin_ui_baseline_restore_and_export_cancel(window,tmp_path):
    w = window
    from lumen import white_balance
    wb = white_balance.from_metadata({'ColorTemperature':4800})
    w.info['white_balance'] = copy.deepcopy(wb)
    w.edits['white_balance'] = wb
    w.refresh()
    assert w.kelvin.spin.value() == 4800 and '4800' in w.camera_wb_label.text()
    w.kelvin.spin.setValue(6500)
    assert w.edits['white_balance']['kelvin'] == 6500
    w.reset_wb()
    assert w.kelvin.spin.value() == 4800
    output = tmp_path/'cancelled.tif'
    w.start_export(str(output),2,':builtin:')
    w.export_cancel.set()
    wait_until(lambda: not w.exporting)
    assert not output.exists() and not list(tmp_path.glob('*.lumen-tmp*'))
    assert not [d for d in w.findChildren(QProgressDialog) if d.isVisible()]


def test_neural_preview_dialog_works_offline(window):
    from lumen.enhance_dialog import ExportDialog
    w=window
    dialog=ExportDialog(w,w.source_path,True)
    dialog.show()
    assert dialog.model_choice() == ':builtin:'
    dialog.preview()
    wait_until(lambda: dialog.preview_button.isEnabled(),timeout=30)
    assert dialog.after.pixmap() is not None and 'Real-ESRGAN' in dialog.status.text()
    dialog.reject()


def test_multi_import_sync_retains_per_photo_edits_and_album(window,tmp_path):
    w=window
    second=tmp_path/'第二张.png'
    third=tmp_path/'第三张.png'
    Image.new('RGB',(360,240),'#52604a').save(second)
    Image.new('RGB',(360,240),'#304e78').save(third)
    first=w.source_path
    w.add_documents([str(second),str(third)])
    assert w.filmstrip.count()==3
    w.controls['exposure'].spin.setValue(.65)
    w.edits['crop']=[.1,.1,.9,.9]
    w.edits['retouch']=[dict(kind='heal',points=[[.5,.5]],radius=.02,enabled=True,feather=50,opacity=100)]
    for i in range(3):w.filmstrip.item(i).setSelected(True)
    w.sync_look()
    other=w.documents[str(second.resolve())]
    assert other['edits']['adjustments']['exposure']==.65
    assert other['edits']['crop'] is None and other['edits']['retouch']==[]
    w.open_path(str(second))
    wait_until(lambda:not w.loading and not w.render_running and not w.timer.isActive())
    assert w.edits['adjustments']['exposure']==.65 and w.edits['crop'] is None
    w.controls['exposure'].spin.setValue(-.25)
    w.open_path(first)
    wait_until(lambda:not w.loading and not w.render_running and not w.timer.isActive())
    assert w.edits['adjustments']['exposure']==.65 and w.edits['crop']==[.1,.1,.9,.9]
    album=tmp_path/'旅行.lumenalbum'
    assert w.save_album(path=str(album))
    w.open_album(str(album))
    wait_until(lambda:not w.loading and not w.render_running and not w.timer.isActive())
    assert len(w.documents)==3 and w.documents[str(second.resolve())]['edits']['adjustments']['exposure']==-.25
    assert not w.test_messages


def test_enter_crop_hides_outside_and_mask_coordinates_remain_original(window):
    w=window
    w.tabs.setCurrentIndex(5)
    w.edits.update(crop=[.2,.2,.8,.8],rotation=1,straighten=3)
    w.update_display()
    w.canvas.setFocus()
    QTest.keyClick(w.canvas,Qt.Key.Key_Return)
    assert w.final_view.isChecked() and w.canvas.crop is None
    assert w.canvas.image.width()==144 and w.canvas.image.height()==216
    w.tabs.setCurrentIndex(4)
    w.add_mask('brush')
    assert w.final_view.isChecked() and w.canvas.tool=='brush'
    point=[.5,.5]
    QTest.mouseClick(w.canvas,Qt.MouseButton.LeftButton,pos=w.canvas.screen(point).toPoint())
    np.testing.assert_allclose(w.selected_mask()['strokes'][0]['points'][0],point,atol=.004)
    assert w.canvas.overlay.width()==144 and w.canvas.overlay.height()==216
    w.tabs.setCurrentIndex(8)
    QTest.mouseClick(w.canvas,Qt.MouseButton.LeftButton,pos=w.canvas.screen(point).toPoint())
    np.testing.assert_allclose(w.edits['retouch'][-1]['points'][0],point,atol=.004)


def tile_pixels(tile):
    from PySide6.QtGui import QImage
    image=tile.image.convertToFormat(QImage.Format.Format_RGB888)
    rows=np.frombuffer(image.constBits(),np.uint8).reshape(image.height(),image.bytesPerLine())
    return rows[:,:image.width()*3].reshape(image.height(),image.width(),3).copy()


def detail_settled(w):
    return w.detail_ready() and not w.full_busy and not w.render_running and not w.timer.isActive() \
        and not w.detail_timer.isActive() and not w.jobs


def open_detail_photo(w,tmp_path):
    path=tmp_path/'细节.png'
    yy,xx=np.mgrid[0:1800,0:2400]
    checker=((xx//2+yy//2)%2*200+25+xx//48).astype(np.uint8)
    Image.fromarray(np.stack([checker,np.flipud(checker),checker//2],axis=2)).save(path)
    w.open_path(str(path))
    wait_until(lambda:not w.loading and not w.render_running and not w.timer.isActive())
    assert w.source.shape[1]==1600
    w.canvas.actual_size()
    wait_until(lambda:detail_settled(w),timeout=30)


def assert_tiles_match(w,layer,expected,view):
    for index in view.tiles:
        tile=layer[index]
        block=expected[tile.y0:tile.y0+tile.height,tile.x0:tile.x0+tile.width]
        np.testing.assert_array_equal(tile_pixels(tile),np.clip(block*255,0,255).astype(np.uint8))


def save_documents(w,tmp_path):
    w.saved_edits=copy.deepcopy(w.edits)
    # Save every document so fixture teardown never invokes a modal album question.
    assert w.save_album(path=str(tmp_path/'高清.lumenalbum'))


def test_true_one_to_one_zoom_renders_visible_original_pixels_and_drops_stale_tiles(window,tmp_path):
    w=window
    open_detail_photo(w,tmp_path)
    view=w.detail_view()
    assert view.level==0 and view.size==(2400,1800) and w.full_source.shape[:2]==(1800,2400)
    assert abs(w.canvas.image_rect().width()*w.canvas.devicePixelRatioF()-2400)<.01
    # 1.4.0: only the visible part of the 5 × 4 tile frame is rendered; the canvas keeps the preview.
    layer=w.canvas.detail.main
    assert set(view.tiles)<=set(layer) and len(layer)<20 and w.canvas.image.width()==1600
    assert_tiles_match(w,layer,engine.process(w.full_source,w.edits,w.backend,apply_crop=False,detail_scale=1.),view)
    w.controls['exposure'].spin.setValue(.3)
    wait_until(lambda:detail_settled(w),timeout=30)
    layer=w.canvas.detail.main
    assert all(name[1][1]==w.generation for name in w.tiles.layers if name[0]=='edited')
    assert_tiles_match(w,layer,engine.process(w.full_source,w.edits,w.backend,apply_crop=False,detail_scale=1.),w.detail_view())
    save_documents(w,tmp_path)


def test_panning_renders_only_newly_exposed_tiles(window,tmp_path):
    w=window
    open_detail_photo(w,tmp_path)
    before=dict(w.canvas.detail.main)
    w.canvas.offset+=QPointF(-700,0)
    w.canvas.viewport_changed.emit()
    wait_until(lambda:detail_settled(w),timeout=30)
    after=w.canvas.detail.main
    kept=set(before)&set(w.detail_view().tiles)
    assert kept and all(after[index] is before[index] for index in kept)
    assert set(w.detail_view().tiles)-set(before)
    # Zooming back out to the preview drops the detail layer; tiles stay cached.
    w.canvas.fit();QTest.qWait(50)
    assert w.canvas.detail is None and w.detail_view() is None
    save_documents(w,tmp_path)


def test_detail_tiles_cover_the_view_without_seams(window,tmp_path):
    from PySide6.QtGui import QColor, QImage
    w=window
    open_detail_photo(w,tmp_path)
    # 100 % with tile edges on half pixels (where unsnapped edges left a blended column), then zoomed.
    for fraction,zoom in ((.5,1.),(.25,1.),(.5,1.0003),(.37,1.6)):
        w.canvas.zoom*=zoom
        r=w.canvas.image_rect()
        w.canvas.offset+=QPointF(fraction-r.x()%1,fraction-r.y()%1)
        w.canvas.viewport_changed.emit()
        wait_until(lambda:detail_settled(w),timeout=30)
        # A pixel the tiles leave uncovered, even partly, changes with the preview underneath.
        shots=[]
        for color in ('#ff00ff','#00ff00'):
            base=QImage(w.canvas.image.size(),QImage.Format.Format_RGB32)
            base.fill(QColor(color))
            w.canvas.image=base
            shown=w.canvas.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)
            pixels=np.frombuffer(shown.constBits(),np.uint8).reshape(shown.height(),shown.bytesPerLine())
            shots.append(pixels[:,:shown.width()*4].copy())
        seams=np.argwhere(shots[0]!=shots[1])
        assert not len(seams),seams[:5]
        w.update_display()
    save_documents(w,tmp_path)


def test_no_detail_render_while_a_slider_is_dragged(window,tmp_path):
    w=window
    open_detail_photo(w,tmp_path)
    slider=w.controls['contrast'].slider
    QTest.mousePress(slider,Qt.MouseButton.LeftButton,pos=slider.rect().center())
    w.controls['contrast'].spin.setValue(20)
    started=time.monotonic()
    while time.monotonic()-started<1.:
        QTest.qWait(20);time.sleep(.002)
        assert not w.full_busy
    assert not w.detail_ready()
    QTest.mouseRelease(slider,Qt.MouseButton.LeftButton,pos=slider.rect().center())
    wait_until(lambda:detail_settled(w),timeout=30)
    save_documents(w,tmp_path)


def test_compare_and_split_use_original_detail_tiles(window,tmp_path):
    w=window
    open_detail_photo(w,tmp_path)
    w.controls['exposure'].spin.setValue(-.6)
    wait_until(lambda:detail_settled(w),timeout=30)
    original=np.clip(engine.to_srgb(engine.develop.apply(w.full_source,w.edits['develop'])),0,1)
    QTest.mousePress(w.compare,Qt.MouseButton.LeftButton)
    assert w.comparing
    wait_until(lambda:w.detail_ready() and not w.full_busy,timeout=30)
    assert_tiles_match(w,w.canvas.detail.main,original,w.detail_view())
    QTest.mouseRelease(w.compare,Qt.MouseButton.LeftButton)
    w.split_check.setChecked(True)
    wait_until(lambda:detail_settled(w),timeout=30)
    assert w.canvas.detail.before is not None
    assert_tiles_match(w,w.canvas.detail.before,original,w.detail_view())
    save_documents(w,tmp_path)


def test_color_selection_ui_and_dng_export(window,tmp_path):
    w=window
    w.tabs.setCurrentIndex(4)
    w.color_region_button.setChecked(True)
    assert w.canvas.tool=='color'
    QTest.mouseClick(w.canvas,Qt.MouseButton.LeftButton,pos=w.canvas.screen([.4,.4]).toPoint())
    wait_until(lambda:not w.selection_busy)
    assert w.selected_mask()['kind']=='color' and 'raster' in w.selected_mask()
    w.refine_mask_check.setChecked(True)
    assert w.canvas.tool=='brush'
    out=tmp_path/'成片.dng'
    w.start_export(str(out))
    wait_until(lambda:not w.exporting)
    assert out.exists()
    from lumen import dng
    assert dng.is_rendered(out)
