"""1.7.0: a draggable histogram (Lightroom's five zones) and the colour-grading eyedropper."""
import colorsys

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from test_ui import app, window, wait_until  # noqa: F401


def _drag(widget, x0, x1, y=50):
    QTest.mousePress(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(x0, y))
    for x in np.linspace(x0, x1, 6)[1:]:
        QTest.mouseMove(widget, QPoint(int(round(x)), y))
    QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(x1, y))


def test_histogram_zones_follow_lightroom(app):
    from lumen.widgets import Histogram
    h = Histogram()
    h.resize(400, 105)
    assert [h.zone_at(x)[0] for x in (0, 39, 41, 139, 141, 259, 261, 359, 361, 399, 400)] == \
        ['blacks', 'blacks', 'shadows', 'shadows', 'exposure', 'exposure', 'highlights', 'highlights', 'whites', 'whites', 'whites']


def test_dragging_the_histogram_changes_the_zone_slider(window):
    w = window
    wait_until(lambda: w.histogram.hist is not None)
    hist = w.histogram
    width = hist.width()
    before = dict(w.edits['adjustments'])
    # A quarter of the width in the exposure zone: one stop.
    _drag(hist, int(width * .5), int(width * .75))
    assert w.edits['adjustments']['exposure'] == pytest.approx(before['exposure'] + 1, abs=.03)
    assert w.controls['exposure'].spin.value() == pytest.approx(w.edits['adjustments']['exposure'])
    # Shadows: a quarter of the width is 50 points; whites to the left lowers them.
    _drag(hist, int(width * .2), int(width * .45))
    assert w.edits['adjustments']['shadows'] == pytest.approx(50, abs=1.5)
    _drag(hist, int(width * .95), int(width * .8))
    assert w.edits['adjustments']['whites'] == pytest.approx(-30, abs=1.5)
    # Limits, and the label shows the zone under the pointer.
    _drag(hist, int(width * .05), int(width * .05) - width * 2)
    assert w.edits['adjustments']['blacks'] == -100
    assert hist.values['blacks'] == -100 and hist.hover[0] == 'blacks'
    # Each drag is one undo step; double-click resets the zone.
    w.undo(-1)
    assert w.edits['adjustments']['blacks'] == before['blacks'] and w.edits['adjustments']['whites'] == pytest.approx(-30, abs=1.5)
    QTest.mouseDClick(hist, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(int(width * .3), 50))
    assert w.edits['adjustments']['shadows'] == 0
    # Sliders moved elsewhere show in the histogram label.
    w.controls['highlights'].spin.setValue(-40)
    assert hist.values['highlights'] == -40


def test_grading_eyedropper_sets_the_zone_hue(window):
    w = window
    wait_until(lambda: w.rendered is not None and not w.render_running)
    w.tabs.setCurrentIndex(6)
    w.grade_zone.setCurrentIndex(0)
    w.grade_pick_button.setChecked(True)
    assert w.canvas.tool == 'pick'
    h, wd = w.rendered.shape[:2]
    point = (.9, .2)
    pixel = w.rendered[round(.2 * (h - 1)), round(.9 * (wd - 1))]
    hue, saturation, _ = colorsys.rgb_to_hsv(*map(float, pixel))
    w.canvas.picked.emit(point)
    shadows = w.edits['grading']['shadows']
    assert abs(shadows[0] - hue * 360) < 2
    assert shadows[1] == pytest.approx(min(40, max(8, saturation * 50)), abs=1)
    assert not w.grade_pick_button.isChecked() and w.canvas.tool != 'pick'
    assert '色相' in w.statusBar().currentMessage()
    # With the zone already toned, only its hue changes; one undo step back.
    w.edits['grading']['highlights'] = [40., 33.]
    w.grade_zone.setCurrentIndex(2)
    w.grade_pick_button.setChecked(True)
    w.canvas.picked.emit((.1, .9))
    assert w.edits['grading']['highlights'][1] == 33. and w.edits['grading']['highlights'][0] != 40.
    w.undo(-1)
    assert w.edits['grading']['highlights'] == [40., 33.]
    # Grey has no hue: nothing changes and the tool stays on.
    w.rendered = np.full_like(w.rendered, .5)
    w.grade_pick_button.setChecked(True)
    w.canvas.picked.emit((.5, .5))
    assert w.edits['grading']['highlights'] == [40., 33.] and w.grade_pick_button.isChecked()
    assert '灰色' in w.statusBar().currentMessage()
    # Leaving the panel turns the eyedropper off.
    w.tabs.setCurrentIndex(0)
    assert not w.grade_pick_button.isChecked()
