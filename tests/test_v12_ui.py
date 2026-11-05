import copy
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from test_ui import app,window,wait_until
from lumen import exposure_curve,model
from lumen.app import MainWindow
from lumen.ai_dialog import EnhancementDialog


def test_close_drains_active_job_and_drops_queued_previews(app):
    import threading
    w=MainWindow();w.show();release=threading.Event();started=threading.Event();calls=[]
    def first():started.set();release.wait(3);return 1
    w.job(first,lambda r:calls.append('active result'))
    wait_until(started.is_set)
    w.job(lambda:calls.append('queued work'),lambda r:None)
    w.close();assert w.closing and w.isVisible()
    release.set();wait_until(lambda:not w.isVisible())
    assert not calls and not w.queued_jobs


def test_tabs_have_inline_watermark_and_color_curves(app):
    w=MainWindow();w.show()
    assert [w.tabs.tabText(i) for i in range(2)]==['光影','色彩'] and w.tabs.tabText(w.tabs.count()-1)=='水印'
    assert '曲线' not in [w.tabs.tabText(i) for i in range(w.tabs.count())]
    assert w.tabs.widget(w.tab_index('color')).isAncestorOf(w.curve)
    assert w.tabs.widget(w.tab_index('watermark')).isAncestorOf(w.watermark_editor)
    w.show_tab('watermark')
    assert not w.watermark_editor.enabled.isEnabled()
    assert w.pool.maxThreadCount()==32
    assert all(a.text()!='水印' for a in w.menuBar().actions())
    w.close()


def test_curve_sliders_undo_and_inline_watermark(window):
    w=window;w.show_tab('basic');w.tabs.widget(w.tab_index('basic')).ensureWidgetVisible(w.exposure_curve)
    w.controls['shadows'].spin.setValue(25);w.commit()
    assert w.exposure_curve.values['shadows']==25
    baseline=copy.deepcopy(w.edits['adjustments'])
    a=w.exposure_curve.screen(.65,float(exposure_curve.evaluate(baseline,.65))).toPoint()
    b=w.exposure_curve.screen(.65,float(exposure_curve.evaluate(baseline,.65))+.07).toPoint()
    QTest.mousePress(w.exposure_curve,Qt.MouseButton.LeftButton,pos=a)
    QTest.mouseMove(w.exposure_curve,b)
    QTest.mouseRelease(w.exposure_curve,Qt.MouseButton.LeftButton,pos=b)
    assert w.edits['adjustments']['highlights']>baseline['highlights']
    assert abs(w.controls['highlights'].spin.value()-w.edits['adjustments']['highlights'])<=.5
    assert w.edits['curves']==model.recipe()['curves']
    w.undo(-1);assert w.edits['adjustments']==baseline
    assert w.exposure_curve.values==baseline
    w.show_tab('watermark')
    w.watermark_editor.enabled.setChecked(True)
    w.watermark_editor.sides['left'].setChecked(True)
    assert w.edits['watermark']['enabled'] and 'left' in w.edits['watermark']['sides']
    w.undo(-1);assert 'left' not in w.edits['watermark']['sides']
    assert not w.watermark_editor.sides['left'].isChecked()
    for kind in ('super','denoise'):
        dialog=EnhancementDialog(w,kind)
        assert dialog.method.currentIndex()==0 and dialog.method.count()==(2 if kind=='super' else 3)
        dialog.reject()
