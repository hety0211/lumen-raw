from __future__ import annotations
import copy
import sys
import numpy as np
from .curves import evaluate
from PySide6.QtCore import Qt, Signal, QEvent, QPointF, QRectF, QPoint, QRect, QSize
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPainterPath, QConicalGradient, QRadialGradient
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QDoubleSpinBox, QLayout,
                               QToolButton, QButtonGroup)
from .i18n import tr


def qimage(rgb):
    from . import large_image
    data=large_image.allocate(rgb.shape,np.uint8)
    for y,block in large_image.strips(rgb):data[y:y+len(block)]=np.clip(block*255,0,255).astype(np.uint8)
    return QImage(data.data, data.shape[1], data.shape[0], data.strides[0], QImage.Format.Format_RGB888).copy()


class FlowLayout(QLayout):
    """Left-to-right items that wrap onto further rows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items = []
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self.items.append(item)

    def count(self):
        return len(self.items)

    def itemAt(self, index):
        return self.items[index] if 0 <= index < len(self.items) else None

    def takeAt(self, index):
        return self.items.pop(index) if 0 <= index < len(self.items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QRect(0, 0, width, 0), False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.items:
            size = size.expandedTo(item.minimumSize())
        return size

    def _arrange(self, rect, apply):
        x, y, line = rect.x(), rect.y(), 0
        for item in self.items:
            hint = item.sizeHint()
            if x + hint.width() > rect.right() + 1 and line:
                x, y, line = rect.x(), y + line, 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width()
            line = max(line, hint.height())
        return y + line - rect.y()


class TabStrip(QWidget):
    """Tab buttons for a QTabWidget that wrap onto a second row instead of scrolling (1.5.1:
    tab names in longer languages no longer hide behind scroll arrows)."""

    def __init__(self, tabs):
        super().__init__()
        self.tabs = tabs
        self.setObjectName('tabStrip')
        tabs.tabBar().hide()
        self.flow = FlowLayout(self)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = []
        tabs.currentChanged.connect(self.sync)

    def rebuild(self):
        for button in self.buttons:
            self.group.removeButton(button)
            button.deleteLater()
        while self.flow.takeAt(0) is not None:
            pass
        self.buttons = []
        for index in range(self.tabs.count()):
            button = QToolButton()
            button.setObjectName('tabButton')
            button.setText(self.tabs.tabText(index))
            button.setCheckable(True)
            button.setAutoRaise(True)
            button.clicked.connect(lambda checked=False, i=index: self.tabs.setCurrentIndex(i))
            self.group.addButton(button)
            self.flow.addWidget(button)
            self.buttons.append(button)
        self.sync(self.tabs.currentIndex())
        self.updateGeometry()

    def sync(self, index):
        if 0 <= index < len(self.buttons):
            self.buttons[index].setChecked(True)


class AdjustSlider(QWidget):
    changed = Signal(float)
    committed = Signal()

    def __init__(self, label, minimum=-100, maximum=100, step=1):
        super().__init__()
        self.factor = 1 / step
        self.default_value = 0
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 1, 0, 5)
        outer.setSpacing(2)
        row = QHBoxLayout()
        self.title = QLabel(label)
        row.addWidget(self.title)
        row.addStretch()
        self.spin = QDoubleSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(2 if step < 1 else 0)
        self.spin.setSingleStep(step)
        self.spin.setFixedWidth(70)
        self.spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.spin.setKeyboardTracking(False)
        row.addWidget(self.spin)
        outer.addLayout(row)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(round(minimum * self.factor), round(maximum * self.factor))
        outer.addWidget(self.slider)
        self.slider.valueChanged.connect(self._slide)
        self.spin.valueChanged.connect(self._spin)
        self.slider.sliderReleased.connect(self.committed)
        self.spin.editingFinished.connect(self.committed)
        self.slider.setToolTip(tr('拖动调整；双击数值可直接输入'))
        self.slider.mouseDoubleClickEvent = self.reset_value

    def reset_value(self, _):
        self.spin.setValue(max(self.spin.minimum(), self.default_value))
        self.committed.emit()

    def _slide(self, value):
        self.spin.blockSignals(True)
        self.spin.setValue(value / self.factor)
        self.spin.blockSignals(False)
        self.changed.emit(value / self.factor)

    def _spin(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(round(value * self.factor))
        self.slider.blockSignals(False)
        self.changed.emit(value)

    def setValue(self, value):
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(round(value * self.factor))
        self.spin.setValue(value)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)


class Histogram(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(105)
        self.hist = None

    def set_image(self, rgb):
        values = (np.clip(rgb[::3, ::3], 0, 1) * 255).astype(np.uint8)
        self.hist = [np.bincount(values[..., c].ravel(), minlength=256) for c in range(3)]
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor('#171b21'))
        if self.hist is None:
            p.setPen(QColor('#6d7684'))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, tr('RGB 直方图'))
            return
        peak = max(max(np.percentile(h, 99), 1) for h in self.hist)
        for h, color in zip(self.hist, ['#ea7d88', '#8ed7ac', '#86adf3']):
            path = QPainterPath(QPointF(0, self.height()))
            for i, val in enumerate(h):
                path.lineTo(i * self.width() / 255, self.height() - min(val / peak, 1) * (self.height() - 8))
            path.lineTo(self.width(), self.height())
            path.closeSubpath()
            c = QColor(color)
            c.setAlpha(80)
            p.fillPath(path, c)
            p.setPen(QPen(QColor(color), 1))
            p.drawPath(path)


class CurveEditor(QWidget):
    changed = Signal(list)
    selected = Signal(int, int)
    committed = Signal()

    def __init__(self):
        super().__init__()
        self.points = [[0., 0.], [1., 1.]]
        self.color = '#b5c5ff'
        self.mode = 'smooth'
        self.current_x = 128
        self.drag = None
        self.setMinimumHeight(215)
        self.setToolTip(tr('在任意亮度位置按下并拖动 · 右键删除节点 · 输入/输出支持 0–255 精确调整'))

    def area(self):
        return QRectF(18, 12, self.width() - 36, self.height() - 30)

    def screen(self, point):
        a = self.area()
        return QPointF(a.left() + point[0] * a.width(), a.bottom() - point[1] * a.height())

    def value(self, pos):
        a = self.area()
        return [float(np.clip((pos.x() - a.left()) / a.width(), 0, 1)),
                float(np.clip((a.bottom() - pos.y()) / a.height(), 0, 1))]

    def set_points(self, points, channel='RGB', mode='smooth'):
        self.points = copy.deepcopy(points)
        self.mode = mode
        self.color = {'RGB': '#b5c5ff', 'R': '#ef8693', 'G': '#8ed7ac', 'B': '#86adf3'}[channel]
        self.update()

    def hit(self, pos):
        for i, point in enumerate(self.points):
            if (self.screen(point) - pos).manhattanLength() < 16:
                return i
        return None

    def report(self, x):
        self.current_x = round(x * 255)
        self.selected.emit(self.current_x, round(float(evaluate(self.points, x, self.mode)) * 255))

    def insert(self, x, y):
        x = round(x * 255) / 255
        nearest = min(range(len(self.points)), key=lambda i: abs(self.points[i][0] - x))
        if abs(self.points[nearest][0] - x) < .001:
            self.points[nearest][1] = y
            return nearest
        if len(self.points) >= 256:
            return nearest
        self.points.append([x, y])
        self.points.sort()
        return next(i for i, p in enumerate(self.points) if p[0] == x)

    def set_output(self, x, y):
        self.insert(x / 255, y / 255)
        self.changed.emit(copy.deepcopy(self.points))
        self.report(x / 255)
        self.committed.emit()
        self.update()

    def mousePressEvent(self, event):
        if not self.area().contains(event.position()):
            return
        i = self.hit(event.position())
        if event.button() == Qt.MouseButton.RightButton:
            if i is not None and 0 < i < len(self.points) - 1:
                self.points.pop(i)
                self.changed.emit(copy.deepcopy(self.points))
                self.committed.emit()
                self.update()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x, y = self.value(event.position())
        if i is None:
            i = self.insert(x, y)
        self.drag = i
        self.report(self.points[i][0])
        self.changed.emit(copy.deepcopy(self.points))
        self.update()

    def mouseDoubleClickEvent(self, event):
        self.mousePressEvent(event)
        self.mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        x, y = self.value(event.position())
        if self.drag is not None:
            # Lock the input tone while dragging: any brightness can be pulled
            # vertically without accidentally changing neighboring tone anchors.
            self.points[self.drag][1] = y
            self.changed.emit(copy.deepcopy(self.points))
            self.report(self.points[self.drag][0])
            self.update()
        else:
            self.report(x)

    def mouseReleaseEvent(self, _):
        if self.drag is not None:
            self.drag = None
            self.committed.emit()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        a = self.area()
        p.fillRect(a, QColor('#161a20'))
        p.setPen(QPen(QColor('#303640'), 1))
        for i in range(5):
            t = i / 4
            p.drawLine(QPointF(a.left() + t * a.width(), a.top()), QPointF(a.left() + t * a.width(), a.bottom()))
            p.drawLine(QPointF(a.left(), a.top() + t * a.height()), QPointF(a.right(), a.top() + t * a.height()))
        p.setPen(QPen(QColor('#566070'), 1, Qt.PenStyle.DashLine))
        p.drawLine(a.bottomLeft(), a.topRight())
        path = QPainterPath(self.screen(self.points[0]))
        axis = np.linspace(0, 1, 512)
        for point in zip(axis, evaluate(self.points, axis, self.mode)):
            path.lineTo(self.screen(point))
        p.setPen(QPen(QColor(self.color), 2))
        p.drawPath(path)
        for point in self.points:
            p.setBrush(QColor('#1c2028'))
            p.drawEllipse(self.screen(point), 4, 4)


class Canvas(QWidget):
    opened = Signal()
    dropped = Signal(str)
    geometry_changed = Signal(object)
    stroke_finished = Signal(dict)
    zoom_changed = Signal(str)
    sampled = Signal(object)
    clone_sampled = Signal(object)
    color_sampled = Signal(object)
    viewport_changed = Signal()
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.image = None
        self.reference_size = None
        self.to_source = None
        self.to_display = None
        self.brush_scale = 1.
        self.overlay = None
        self.zoom = 1.
        self.offset = QPointF()
        self.tool = 'view'
        self.clone_source = None
        self.clone_offset = None
        self.crop = None
        self.ratio = None
        self.brush_radius = .045
        self.erase = False
        self.start = None
        self.end = None
        self.stroke = None
        self.pan = None
        self.pointer = None
        self.mask_geometry = None
        self.before = None
        self.split = False
        self.split_position = .5
        self.split_drag = False
        self.clipping = None
        self.show_shadows = self.show_highlights = False
        self.detail = None
        self.setMinimumSize(400, 380)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_image(self, rgb):
        self.image = qimage(rgb)
        self.update()

    def set_before(self, rgb, enabled=False):
        self.before = qimage(rgb) if enabled and rgb is not None else None
        self.split = enabled
        self.update()

    def set_detail(self, layer):
        """Original-resolution tiles drawn over the preview (``viewport.DetailLayer`` or None)."""
        self.detail = layer
        self.update()

    def set_clipping(self, rgb, shadows=False, highlights=False):
        self.clipping = None
        self.show_shadows, self.show_highlights = shadows, highlights
        if shadows or highlights:
            data = np.zeros((*rgb.shape[:2], 4), np.uint8)
            if shadows:
                data[np.max(rgb, axis=2) <= .005] = [75, 134, 255, 210]
            if highlights:
                data[np.max(rgb, axis=2) >= .995] = [255, 75, 90, 210]
            self.clipping = QImage(data.data, data.shape[1], data.shape[0], data.strides[0], QImage.Format.Format_RGBA8888).copy()
        self.update()

    def set_overlay(self, alpha):
        if alpha is None:
            self.overlay = None
        else:
            data = np.zeros((*alpha.shape, 4), np.uint8)
            data[..., :3] = [155, 122, 255]
            data[..., 3] = np.clip(alpha * 105, 0, 105).astype(np.uint8)
            self.overlay = QImage(data.data, data.shape[1], data.shape[0], data.strides[0], QImage.Format.Format_RGBA8888).copy()
        self.update()

    def image_rect(self):
        if self.image is None:
            return QRectF()
        iw, ih = self.reference_size or (self.image.width(), self.image.height())
        scale = min((self.width() - 64) / iw, (self.height() - 64) / ih) * self.zoom
        w, h = iw * scale, ih * scale
        return QRectF((self.width() - w) / 2 + self.offset.x(), (self.height() - h) / 2 + self.offset.y(), w, h)

    def view_pos(self, p):
        r = self.image_rect()
        return [float(np.clip((p.x() - r.x()) / max(1, r.width()), 0, 1)),
                float(np.clip((p.y() - r.y()) / max(1, r.height()), 0, 1))]

    def pos(self, p):
        point = self.view_pos(p)
        return np.clip(self.to_source(point),0,1).tolist() if self.to_source else point

    def screen(self, p):
        if self.to_display:
            p = self.to_display(p)
        r = self.image_rect()
        return QPointF(r.x() + p[0] * r.width(), r.y() + p[1] * r.height())

    def fit(self):
        self.zoom, self.offset = 1., QPointF()
        self.zoom_changed.emit(tr('适应窗口'))
        self.viewport_changed.emit()
        self.update()

    def actual_size(self):
        if self.image:
            iw,ih = self.reference_size or (self.image.width(),self.image.height())
            fit = min((self.width() - 64) / iw, (self.height() - 64) / ih)
            self.zoom = 1 / (fit * self.devicePixelRatioF())
            self.offset = QPointF()
            self.zoom_changed.emit(tr('原图 100%'))
            self.viewport_changed.emit()
            self.update()

    def wheelEvent(self, e):
        if self.image is None:
            return
        if sys.platform == 'darwin':
            # Trackpad (and Magic Mouse) scrolls carry a phase; plain mouse wheels do not, even though
            # Qt also gives them a pixel delta.  Two fingers pan; ⌘ + scroll and wheels zoom smoothly.
            if e.phase() != Qt.ScrollPhase.NoScrollPhase and not e.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.offset += QPointF(e.pixelDelta())
                self.viewport_changed.emit()
                self.update()
                return
            if e.angleDelta().y():
                self.zoom_by(1.15 ** (e.angleDelta().y() / 120), e.position())
            return
        self.zoom_by(1.15 if e.angleDelta().y() > 0 else 1 / 1.15, e.position())

    def zoom_by(self, factor, position):
        old_zoom = self.zoom
        iw,ih = self.reference_size or (self.image.width(),self.image.height())
        fit = min((self.width()-64)/iw,(self.height()-64)/ih)
        self.zoom = float(np.clip(self.zoom * factor, .3, max(12,8/fit)))
        center = QPointF(self.width() / 2, self.height() / 2)
        relative = position - center
        self.offset = relative - (relative - self.offset) * (self.zoom / old_zoom)
        hint = tr('双指平移') if sys.platform == 'darwin' else tr('中键平移')
        self.zoom_changed.emit(tr('原图 {v:.0f}% · {hint}', v=fit*self.zoom*self.devicePixelRatioF()*100, hint=hint))
        self.viewport_changed.emit()
        self.update()

    def event(self, e):
        # macOS trackpad pinch arrives as a native zoom gesture.
        if e.type() == QEvent.Type.NativeGesture and self.image is not None:
            if e.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                self.zoom_by(max(.2, 1 + e.value()), e.position())
                return True
            if e.gestureType() == Qt.NativeGestureType.SmartZoomNativeGesture:
                self.fit()
                return True
        return super().event(e)

    def mousePressEvent(self, e):
        if self.image is None:
            if e.button() == Qt.MouseButton.LeftButton:
                self.opened.emit()
            return
        if e.button() == Qt.MouseButton.LeftButton and self.tool in ('sample','color') and self.image_rect().contains(e.position()):
            (self.sampled if self.tool == 'sample' else self.color_sampled).emit(self.pos(e.position()))
            return
        if e.button() == Qt.MouseButton.LeftButton and self.split:
            line = self.image_rect().left() + self.split_position * self.image_rect().width()
            if abs(e.position().x() - line) < 20:
                self.split_drag = True
                return
        if e.button() == Qt.MouseButton.MiddleButton or (self.tool == 'view' and e.button() == Qt.MouseButton.LeftButton):
            self.pan = e.position()
            return
        if e.button() != Qt.MouseButton.LeftButton or not self.image_rect().contains(e.position()):
            return
        if self.tool == 'clone' and e.modifiers() & Qt.KeyboardModifier.AltModifier:
            self.clone_source = self.pos(e.position())
            self.clone_offset = None
            self.clone_sampled.emit(self.clone_source)
            self.update()
            return
        if self.tool == 'clone' and self.clone_source is None:
            self.clone_sampled.emit(None)
            return
        self.start = self.pos(e.position())
        self.end = self.start[:]
        if self.tool in ('brush', 'heal', 'clone'):
            self.stroke = dict(points=[self.start], radius=self.brush_radius, erase=self.erase)
            if self.tool in ('heal', 'clone'):
                self.stroke['kind'] = self.tool
            if self.tool == 'clone':
                if self.clone_offset is None:
                    self.clone_offset = [a - b for a, b in zip(self.clone_source, self.start)]
                self.stroke['offset'] = self.clone_offset[:]
        self.update()

    def mouseMoveEvent(self, e):
        self.pointer = e.position()
        if self.split_drag:
            self.split_position = self.view_pos(e.position())[0]
        elif self.pan is not None:
            self.offset += e.position() - self.pan
            self.pan = e.position()
            self.viewport_changed.emit()
        elif self.start is not None:
            self.end = self.pos(e.position())
            if self.tool == 'crop' and self.ratio:
                r = self.image_rect()
                dx = self.end[0] - self.start[0]
                dy = abs(dx) * r.width() / (self.ratio * r.height())
                sign = 1 if self.end[1] >= self.start[1] else -1
                dy = min(dy, 1 - self.start[1] if sign > 0 else self.start[1])
                self.end[1] = self.start[1] + sign * dy
                self.end[0] = self.start[0] + (1 if dx >= 0 else -1) * dy * self.ratio * r.height() / r.width()
            if self.stroke is not None:
                if np.linalg.norm(np.array(self.end) - self.stroke['points'][-1]) > .001:
                    self.stroke['points'].append(self.end)
        self.update()

    def mouseReleaseEvent(self, e):
        if self.split_drag:
            self.split_drag = False
            return
        if self.pan is not None:
            self.pan = None
            self.viewport_changed.emit()
            return
        if self.start is not None:
            if self.stroke is not None:
                self.stroke_finished.emit(self.stroke)
            elif self.tool in ('crop', 'linear', 'radial') and np.linalg.norm(np.array(self.end) - self.start) > .01:
                self.geometry_changed.emit([self.start, self.end])
        self.start = self.end = self.stroke = None
        self.update()

    def resizeEvent(self, event):
        self.viewport_changed.emit()
        super().resizeEvent(event)

    def leaveEvent(self, _):
        self.pointer = None
        self.update()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and e.mimeData().urls()[0].isLocalFile():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if len(paths)>1:self.files_dropped.emit(paths)
        elif paths:self.dropped.emit(paths[0])

    def draw_tiles(self, p, r, tiles, clipping=False):
        """Draw finished detail tiles that fall inside the widget over the preview."""
        width, height = self.detail.size
        dpr = self.devicePixelRatioF()
        sx, sy = r.width() * dpr / width, r.height() * dpr / height
        ox, oy = r.x() * dpr, r.y() * dpr
        view = QRectF(self.rect())
        p.save()
        # Neighbouring tiles share edges snapped to whole device pixels: no uncovered seam
        # column between them, and at 100 % each tile is a 1:1 copy without resampling.
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        for tile in tiles.values():
            if tile.image is None:
                continue
            left, top = round(ox + tile.x0 * sx), round(oy + tile.y0 * sy)
            right, bottom = round(ox + (tile.x0 + tile.width) * sx), round(oy + (tile.y0 + tile.height) * sy)
            target = QRectF(left / dpr, top / dpr, (right - left) / dpr, (bottom - top) / dpr)
            if not target.intersects(view):
                continue
            p.drawImage(target, tile.image)
            if clipping:
                for shown, warning in ((self.show_shadows, tile.shadows), (self.show_highlights, tile.highlights)):
                    if shown and warning is not None:
                        p.drawImage(target, warning)
        p.restore()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor('#101214'))
        if self.image is None:
            box = QRectF(self.width() / 2 - 195, self.height() / 2 - 100, 390, 200)
            p.setPen(QPen(QColor('#394250'), 1, Qt.PenStyle.DashLine))
            p.setBrush(QColor('#171b23'))
            p.drawRoundedRect(box, 14, 14)
            p.setPen(QColor('#c3cdf5'))
            font = p.font()
            font.setPixelSize(25)
            p.setFont(font)
            p.drawText(box.adjusted(0, 26, 0, -95), Qt.AlignmentFlag.AlignCenter, tr('每一束光，都有余地'))
            font.setPixelSize(14)
            p.setFont(font)
            p.setPen(QColor('#c1c8d4'))
            p.drawText(box.adjusted(0, 82, 0, -45), Qt.AlignmentFlag.AlignCenter, tr('点击打开，或将原片拖到这里'))
            p.setPen(QColor('#778292'))
            p.drawText(box.adjusted(0, 132, 0, -16), Qt.AlignmentFlag.AlignCenter, 'SONY / CANON / NIKON / FUJIFILM / PANASONIC')
            return
        r = self.image_rect()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.drawImage(r, self.image)
        if self.clipping is not None:
            p.drawImage(r, self.clipping)
        if self.detail is not None:
            self.draw_tiles(p, r, self.detail.main, True)
        if self.split and self.before is not None:
            split_x = r.left() + self.split_position * r.width()
            p.save()
            p.setClipRect(QRectF(r.left(), r.top(), r.width() * self.split_position, r.height()))
            p.drawImage(r, self.before)
            if self.detail is not None and self.detail.before is not None:
                self.draw_tiles(p, r, self.detail.before)
            p.restore()
            p.setPen(QPen(QColor('#f1f2e8'), 1.5))
            p.drawLine(QPointF(split_x, r.top()), QPointF(split_x, r.bottom()))
            p.setBrush(QColor('#242827'))
            p.drawEllipse(QPointF(split_x, r.center().y()), 13, 13)
            p.drawText(QRectF(split_x - 12, r.center().y() - 10, 24, 20), Qt.AlignmentFlag.AlignCenter, '↔')
            for text, at in [(tr('原片'), r.left() + 12), (tr('调整后'), r.right() - 70)]:
                rect = QRectF(at, r.top() + 12, 58, 24)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(18, 21, 24, 200))
                p.drawRoundedRect(rect, 4, 4)
                p.setPen(QColor('#e1e4dd'))
                p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        if self.overlay is not None:
            p.drawImage(r, self.overlay)
        crop = self.crop
        if self.tool == 'crop' and self.start is not None:
            crop = [min(self.start[0], self.end[0]), min(self.start[1], self.end[1]),
                    max(self.start[0], self.end[0]), max(self.start[1], self.end[1])]
        if crop:
            area = QRectF(self.screen(crop[:2]), self.screen(crop[2:]))
            shadow = QPainterPath()
            shadow.addRect(r)
            hole = QPainterPath()
            hole.addRect(area)
            p.fillPath(shadow.subtracted(hole), QColor(0, 0, 0, 150))
            p.setPen(QPen(QColor('#e0e5ef'), 1))
            p.drawRect(area)
            for t in (1 / 3, 2 / 3):
                p.drawLine(QPointF(area.x() + t * area.width(), area.top()), QPointF(area.x() + t * area.width(), area.bottom()))
                p.drawLine(QPointF(area.left(), area.y() + t * area.height()), QPointF(area.right(), area.y() + t * area.height()))
        geom = [self.start, self.end] if self.start is not None else self.mask_geometry
        if geom and self.tool in ('linear', 'radial'):
            a, b = self.screen(geom[0]), self.screen(geom[1])
            p.setPen(QPen(QColor('#d9c7ff'), 2, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            if self.tool == 'radial':
                p.drawEllipse(QRectF(a, b).normalized())
            else:
                p.drawLine(a, b)
            p.drawEllipse(a, 5, 5)
            p.drawEllipse(b, 5, 5)
        if self.stroke:
            p.setPen(QPen(QColor(183, 151, 255, 120), max(2, self.brush_radius * self.brush_scale * min(r.width(), r.height()) * 2),
                         Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            path = QPainterPath(self.screen(self.stroke['points'][0]))
            for pt in self.stroke['points'][1:]:
                path.lineTo(self.screen(pt))
            p.drawPath(path)
        if self.pointer and self.tool in ('brush', 'heal', 'clone') and r.contains(self.pointer):
            radius = self.brush_radius * self.brush_scale * min(r.width(), r.height())
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor('#ffffff'), 1))
            p.drawEllipse(self.pointer, radius, radius)
        if self.tool == 'clone' and self.clone_source is not None:
            source = self.clone_source
            if self.pointer and self.clone_offset is not None:
                source = [a + b for a, b in zip(self.pos(self.pointer), self.clone_offset)]
            mark = self.screen(source)
            p.setPen(QPen(QColor('#dcc48e'), 1.5))
            p.drawLine(mark - QPointF(9, 0), mark + QPointF(9, 0))
            p.drawLine(mark - QPointF(0, 9), mark + QPointF(0, 9))
            p.drawEllipse(mark, 5, 5)


class ColorWheel(QWidget):
    changed = Signal(float, float)
    committed = Signal()
    activated = Signal()

    def __init__(self, title):
        super().__init__()
        self.title = title
        self.hue, self.saturation = 0., 0.
        self.active = False
        self.dragging = False
        self.setMinimumSize(95, 140)
        self.setMaximumHeight(155)
        self.setToolTip(tr('拖动色轮：方向控制色相，离中心越远饱和度越高；双击归零。'))

    def center_radius(self):
        return QPointF(self.width() / 2, (self.height() - 32) / 2), min(self.width() - 20, self.height() - 48) / 2

    def set_value(self, hue, saturation):
        self.hue, self.saturation = hue, saturation
        self.update()

    def choose(self, p):
        center, radius = self.center_radius()
        delta = p - center
        self.hue = float(np.degrees(np.arctan2(-delta.y(), delta.x())) % 360)
        self.saturation = float(min(100, np.hypot(delta.x(), delta.y()) / radius * 100))
        self.changed.emit(self.hue, self.saturation)
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.activated.emit()
            center, radius = self.center_radius()
            if np.hypot(e.position().x() - center.x(), e.position().y() - center.y()) <= radius + 6:
                self.dragging = True
                self.choose(e.position())

    def mouseMoveEvent(self, e):
        if self.dragging:
            self.choose(e.position())

    def mouseReleaseEvent(self, _):
        if self.dragging:
            self.dragging = False
            self.committed.emit()

    def mouseDoubleClickEvent(self, _):
        self.saturation = 0.
        self.changed.emit(self.hue, 0.)
        self.committed.emit()
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        center, radius = self.center_radius()
        gradient = QConicalGradient(center, 0)
        for i in range(7):
            gradient.setColorAt(i / 6, QColor.fromHsvF((i / 6) % 1, .72, .76))
        p.setPen(QPen(QColor('#464d47' if not self.active else '#c6d2b7'), 2 if self.active else 1))
        p.setBrush(gradient)
        p.drawEllipse(center, radius, radius)
        radial = QRadialGradient(center, radius)
        radial.setColorAt(0, QColor('#9eaaa3'))
        radial.setColorAt(1, QColor(158, 170, 163, 0))
        p.setBrush(radial)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(center, radius, radius)
        angle = np.deg2rad(self.hue)
        marker = center + QPointF(np.cos(angle), -np.sin(angle)) * radius * self.saturation / 100
        p.setPen(QPen(QColor('#171b1a'), 4))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(marker, 4, 4)
        p.setPen(QPen(QColor('#f2f3ec'), 1.5))
        p.drawEllipse(marker, 4, 4)
        p.setPen(QColor('#d3d9ce' if self.active else '#8b948d'))
        p.drawText(QRectF(0, self.height() - 27, self.width(), 20), Qt.AlignmentFlag.AlignCenter, self.title)
