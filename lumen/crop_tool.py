"""Crop as a tool on the photo (2.3.0): R, or 裁切 above the photo, replaces the 裁切·镜头 panel.

While cropping, the canvas shows the whole photo as it will be oriented — turned by the straighten
angle and the quarter turns, mirrored by the flips — with the crop frame upright over it
(``View``).  The recipe keeps its meaning from 1.x (``engine.crop_rotate``): ``crop`` is a
rectangle of the lens-corrected frame and straightening turns it about its centre, enlarged to
fill.  So the visible part is a rectangle of the turned photo, ``1 / fill_scale`` of the crop's
size, and a frame in the view converts back to a crop exactly (``View.crop_for``).

Dragging inside the frame moves the photo under it while the frame stays where it is on screen
(the canvas pans by the same amount); the corners and edges resize it, keeping the chosen ratio;
dragging outside draws a new frame (also on a photo not cropped yet).  The frame never leaves the
photo: a move stops at its edge, a resize shrinks towards the fixed corner.

The floating ``CropBar`` holds the ratio, orientation swap, quarter turns, flips, the straighten
angle with 自动水平 (``straighten.estimate``) and a 拉直 line, the overlay grid, and 重置 / 取消 /
完成 (Esc / Enter).
"""
from __future__ import annotations

import copy
import math

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from . import model, theme
from .i18n import tr, N_
from .widgets import AdjustSlider

#: Ratio menu: (name, width / height of the frame as seen, or 'original').
RATIOS = ((N_('自由比例'), None), (N_('原片比例'), 'original'), ('1 : 1', 1.), ('4 : 5', .8), ('3 : 4', .75),
          ('2 : 3', 2 / 3), ('9 : 16', 9 / 16), ('5 : 4', 1.25), ('4 : 3', 4 / 3), ('3 : 2', 1.5),
          ('16 : 9', 16 / 9), ('2 : 1', 2.), ('65 : 24', 65 / 24))
GRIDS = (('thirds', N_('三分法')), ('golden', N_('黄金比例')), ('grid', N_('网格')), ('diagonal', N_('对角线')),
         ('none', N_('无参考线')))
#: Screen pixels around a corner or edge that grab it.
HANDLE = 11
#: Smallest frame, as a share of the photo's shorter side.
SMALLEST = .02


def fill_scale(angle, width, height):
    """How much ``engine.crop_rotate`` enlarges a ``width × height`` crop turned by ``angle``."""
    theta = math.radians(abs(angle))
    ratio = height / max(width, 1e-9)
    return max(math.cos(theta) + ratio * math.sin(theta), math.cos(theta) + math.sin(theta) / max(ratio, 1e-9))


def geometry_of(edits):
    """The parts of a recipe the crop tool changes."""
    return dict(crop=copy.deepcopy(edits.get('crop')), straighten=edits.get('straighten', 0.),
                rotation=edits.get('rotation', 0), flip=list(edits.get('flip') or [False, False]))


def shaped(edits):
    """Whether a recipe crops, turns, straightens or mirrors the photo."""
    return geometry_of(edits) != geometry_of(model.recipe())


class View:
    """The oriented photo shown while cropping: source pixels (x, y) of a ``size = (w, h)`` frame
    to view pixels, and back.  Positive angles turn counter-clockwise, as ``cv2`` does."""

    def __init__(self, edits, size):
        self.width, self.height = map(float, size)
        self.angle = float(edits.get('straighten', 0))
        self.turns = int(edits.get('rotation', 0)) % 4
        self.flip = list(edits.get('flip') or [False, False])[:2]
        theta = math.radians(self.angle)
        c, s = math.cos(theta), math.sin(theta)
        rotate = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.]])
        corners = np.array([[0, 0, 1], [self.width, 0, 1], [0, self.height, 1], [self.width, self.height, 1.]]) @ rotate.T
        low, high = corners[:, :2].min(axis=0), corners[:, :2].max(axis=0)
        matrix = np.array([[1, 0, -low[0]], [0, 1, -low[1]], [0, 0, 1.]]) @ rotate
        w, h = high - low
        for _ in range(self.turns):          # clockwise quarter turns, as np.rot90(x, -1)
            matrix = np.array([[0, -1, h], [1, 0, 0], [0, 0, 1.]]) @ matrix
            w, h = h, w
        if self.flip[0]:
            matrix = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1.]]) @ matrix
        if self.flip[1]:
            matrix = np.array([[1, 0, 0], [0, -1, h], [0, 0, 1.]]) @ matrix
        self.matrix = matrix
        self.inverse = np.linalg.inv(matrix)
        self.size = (float(w), float(h))

    def to_view(self, point):
        x, y, _ = self.matrix @ [point[0], point[1], 1.]
        return float(x), float(y)

    def to_source(self, point):
        x, y, _ = self.inverse @ [point[0], point[1], 1.]
        return float(x), float(y)

    def normalised(self):
        """``(to_source, to_display)`` between view and frame coordinates 0–1 (``Canvas``)."""
        vw, vh = self.size

        def to_source(p):
            x, y = self.to_source((p[0] * vw, p[1] * vh))
            return [x / self.width, y / self.height]

        def to_display(p):
            x, y = self.to_view((p[0] * self.width, p[1] * self.height))
            return [x / vw, y / vh]
        return to_source, to_display

    def warp(self, image, background=0.):
        """``image`` (the whole frame at any resolution) as the view, the same resolution."""
        h, w = image.shape[:2]
        k = w / self.width
        scale = np.diag([k, k, 1.])
        matrix = scale @ self.matrix @ np.diag([1 / k, 1 / (h / self.height), 1.])
        size = (max(1, round(self.size[0] * k)), max(1, round(self.size[1] * k)))
        return cv2.warpAffine(np.ascontiguousarray(image), matrix[:2], size, flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(background,) * 3)

    def _unturned(self, width, height):
        return (height, width) if self.turns % 2 else (width, height)

    def frame(self, crop):
        """The visible part of ``crop`` (None: the whole frame) as an upright view rectangle."""
        a, b, c, d = crop or (0., 0., 1., 1.)
        cw, ch = (c - a) * self.width, (d - b) * self.height
        scale = fill_scale(self.angle, cw, ch)
        fw, fh = self._unturned(cw / scale, ch / scale)
        x, y = self.to_view(((a + c) / 2 * self.width, (b + d) / 2 * self.height))
        return [x - fw / 2, y - fh / 2, x + fw / 2, y + fh / 2]

    def crop_for(self, frame):
        """The crop (frame coordinates 0–1, possibly outside 0–1) whose visible part is ``frame``."""
        x0, y0, x1, y1 = frame
        sw, sh = self._unturned(abs(x1 - x0), abs(y1 - y0))
        scale = fill_scale(self.angle, sw, sh)
        cw, ch = sw * scale, sh * scale
        cx, cy = self.to_source(((x0 + x1) / 2, (y0 + y1) / 2))
        return [(cx - cw / 2) / self.width, (cy - ch / 2) / self.height,
                (cx + cw / 2) / self.width, (cy + ch / 2) / self.height]

    @staticmethod
    def fits(crop, slack=1e-6):
        return crop[0] >= -slack and crop[1] >= -slack and crop[2] <= 1 + slack and crop[3] <= 1 + slack

    def shrink(self, frame, anchor):
        """``frame`` shrunk towards ``anchor`` until its crop fits the photo, or None."""
        if self.fits(self.crop_for(frame)):
            return frame
        ax, ay = anchor
        def scaled(t):
            return [ax + (frame[0] - ax) * t, ay + (frame[1] - ay) * t, ax + (frame[2] - ax) * t, ay + (frame[3] - ay) * t]
        low, high = 0., 1.
        for _ in range(24):
            middle = (low + high) / 2
            low, high = (middle, high) if self.fits(self.crop_for(scaled(middle))) else (low, middle)
        if low <= 0:
            return None
        return scaled(low)

    def moved(self, crop, dx, dy):
        """``crop`` with its visible part moved by (dx, dy) view pixels, stopped at the photo's edges."""
        frame = self.frame(crop)
        moved = self.crop_for([frame[0] + dx, frame[1] + dy, frame[2] + dx, frame[3] + dy])
        width, height = moved[2] - moved[0], moved[3] - moved[1]
        x0 = min(max(moved[0], 0.), 1 - width)
        y0 = min(max(moved[1], 0.), 1 - height)
        return [x0, y0, x0 + width, y0 + height]

    def largest(self, ratio, centre):
        """The largest upright frame of ``ratio`` (width / height) about ``centre`` that fits."""
        vw, vh = self.size
        diagonal = math.hypot(vw, vh)
        width = diagonal * ratio / math.hypot(ratio, 1)
        height = width / ratio
        frame = [centre[0] - width / 2, centre[1] - height / 2, centre[0] + width / 2, centre[1] + height / 2]
        return self.shrink(frame, centre)


def clean(crop):
    """A crop as the recipe stores it: inside 0–1, None for the whole frame."""
    if crop is None:
        return None
    crop = [float(np.clip(v, 0, 1)) for v in crop]
    if crop[0] <= 1e-4 and crop[1] <= 1e-4 and crop[2] >= 1 - 1e-4 and crop[3] >= 1 - 1e-4:
        return None
    return [round(v, 6) for v in crop]


class CropTool:
    """Mouse handling and drawing of the crop frame on the canvas (``Canvas.cropper``)."""

    def __init__(self, window):
        self.window = window
        self.drag = None
        self.line = None

    # -- geometry

    def view(self):
        return self.window.crop_view

    def frame(self):
        view = self.view()
        return view.frame(self.window.edits.get('crop')) if view else None

    @staticmethod
    def scale(canvas):
        rect, view = canvas.image_rect(), canvas.reference_size
        return rect.width() / max(view[0], 1e-9)

    def to_screen(self, canvas, point):
        rect, k = canvas.image_rect(), self.scale(canvas)
        return QPointF(rect.x() + point[0] * k, rect.y() + point[1] * k)

    def to_view(self, canvas, position):
        rect, k = canvas.image_rect(), self.scale(canvas)
        return ((position.x() - rect.x()) / k, (position.y() - rect.y()) / k)

    def hit(self, canvas, position):
        """'move', a handle ('l', 't', 'r', 'b' or two of them) or 'new' under the pointer."""
        frame = self.frame()
        if frame is None:
            return 'new'
        a, b = self.to_screen(canvas, frame[:2]), self.to_screen(canvas, frame[2:])
        x, y = position.x(), position.y()
        near_x = 'l' if abs(x - a.x()) <= HANDLE else 'r' if abs(x - b.x()) <= HANDLE else ''
        near_y = 't' if abs(y - a.y()) <= HANDLE else 'b' if abs(y - b.y()) <= HANDLE else ''
        inside_x = a.x() - HANDLE <= x <= b.x() + HANDLE
        inside_y = a.y() - HANDLE <= y <= b.y() + HANDLE
        if near_x and inside_y or near_y and inside_x:
            return (near_x if inside_y else '') + (near_y if inside_x else '')
        if a.x() < x < b.x() and a.y() < y < b.y():
            return 'move' if self.window.edits.get('crop') else 'new-or-move'
        return 'new'

    def cursor(self, canvas, position):
        if self.window.crop_line_mode():
            return Qt.CursorShape.CrossCursor
        part = self.hit(canvas, position)
        if part in ('move', 'new-or-move'):
            return Qt.CursorShape.OpenHandCursor if part == 'move' else Qt.CursorShape.CrossCursor
        if part in ('lt', 'rb'):
            return Qt.CursorShape.SizeFDiagCursor
        if part in ('rt', 'lb'):
            return Qt.CursorShape.SizeBDiagCursor
        if part in ('l', 'r'):
            return Qt.CursorShape.SizeHorCursor
        if part in ('t', 'b'):
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.CrossCursor

    # -- mouse

    def press(self, canvas, event):
        position = event.position()
        if self.window.crop_line_mode():
            point = self.to_view(canvas, position)
            self.line = [point, point]
            return
        part = self.hit(canvas, position)
        frame = self.frame()
        if part == 'new-or-move':
            part = 'new'
        self.drag = dict(part=part, start=position, last=position, frame=frame,
                         anchor=self.to_view(canvas, position), crop=copy.deepcopy(self.window.edits.get('crop')))
        if part == 'move':
            canvas.setCursor(Qt.CursorShape.ClosedHandCursor)

    def move(self, canvas, event):
        position = event.position()
        if self.line is not None:
            self.line[1] = self.to_view(canvas, position)
            canvas.update()
            return
        if self.drag is None:
            canvas.setCursor(self.cursor(canvas, position))
            return
        view = self.view()
        part = self.drag['part']
        if part == 'move':
            k = self.scale(canvas)
            delta = position - self.drag['last']
            self.drag['last'] = position
            before = view.frame(self.window.edits.get('crop'))
            crop = view.moved(self.window.edits.get('crop') or [0, 0, 1, 1], -delta.x() / k, -delta.y() / k)
            after = view.frame(crop)
            # The frame stays where it is on screen: the photo moves under it.
            canvas.offset += QPointF((before[0] - after[0]) * k, (before[1] - after[1]) * k)
            canvas.viewport_changed.emit()
            self.window.set_crop(crop, commit=False)
            return
        point = self.to_view(canvas, position)
        frame = self._resized(part, point)
        if frame is not None:
            self.window.set_crop(view.crop_for(frame), commit=False)

    def release(self, canvas, event):
        if self.line is not None:
            line, self.line = self.line, None
            canvas.update()
            if math.dist(line[0], line[1]) * self.scale(canvas) > 12:
                self.window.straighten_along(line)
            return
        if self.drag is not None:
            moved = self.drag['crop'] != self.window.edits.get('crop')
            self.drag = None
            canvas.setCursor(self.cursor(canvas, event.position()))
            if moved:
                self.window.commit()

    def _resized(self, part, point):
        """The frame a resize or new drag asks for, kept to the ratio and to the photo."""
        view = self.view()
        ratio = self.window.crop_ratio_value()
        start = self.drag['frame']
        if part == 'new':
            anchor = self.drag['anchor']
            vw, vh = view.size
            anchor = (min(max(anchor[0], 0.), vw), min(max(anchor[1], 0.), vh))
            x, y = point
            w, h = x - anchor[0], y - anchor[1]
            if ratio:
                if abs(w) / ratio > abs(h):
                    h = math.copysign(abs(w) / ratio, h or 1)
                else:
                    w = math.copysign(abs(h) * ratio, w or 1)
            frame = [min(anchor[0], anchor[0] + w), min(anchor[1], anchor[1] + h),
                     max(anchor[0], anchor[0] + w), max(anchor[1], anchor[1] + h)]
        else:
            x0, y0, x1, y1 = start
            x, y = point
            if 'l' in part:
                x0 = x
            if 'r' in part:
                x1 = x
            if 't' in part:
                y0 = y
            if 'b' in part:
                y1 = y
            if ratio:
                w, h = abs(x1 - x0), abs(y1 - y0)
                if part in ('l', 'r'):
                    h = w / ratio
                    middle = (start[1] + start[3]) / 2
                    y0, y1 = middle - h / 2, middle + h / 2
                elif part in ('t', 'b'):
                    w = h * ratio
                    middle = (start[0] + start[2]) / 2
                    x0, x1 = middle - w / 2, middle + w / 2
                else:
                    w, h = (w, w / ratio) if w / ratio > h else (h * ratio, h)
                    x0, x1 = (x1 - w, x1) if 'l' in part else (x0, x0 + w)
                    y0, y1 = (y1 - h, y1) if 't' in part else (y0, y0 + h)
            frame = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
            anchor = ((frame[0] if 'l' not in part else frame[2]) if part not in ('t', 'b') else (frame[0] + frame[2]) / 2,
                      (frame[1] if 't' not in part else frame[3]) if part not in ('l', 'r') else (frame[1] + frame[3]) / 2)
        smallest = SMALLEST * min(view.width, view.height)
        if frame[2] - frame[0] < smallest or frame[3] - frame[1] < smallest:
            return None
        return view.shrink(frame, anchor)

    # -- drawing

    def paint(self, canvas, painter, rect):
        frame = self.frame()
        if frame is None:
            return
        a, b = self.to_screen(canvas, frame[:2]), self.to_screen(canvas, frame[2:])
        area = QRectF(a, b)
        shadow = QPainterPath()
        shadow.addRect(QRectF(canvas.rect()))
        hole = QPainterPath()
        hole.addRect(area)
        painter.fillPath(shadow.subtracted(hole), QColor(0, 0, 0, 155))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 120), 1))
        for line in self.grid(area, self.window.crop_grid_kind()):
            painter.drawLine(*line)
        painter.setPen(QPen(QColor('#f4f1ea'), 1.2))
        painter.drawRect(area)
        painter.setPen(QPen(theme.color('accent'), 3.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap))
        arm = min(18., area.width() / 4, area.height() / 4)
        for corner, sx, sy in ((area.topLeft(), 1, 1), (area.topRight(), -1, 1), (area.bottomLeft(), 1, -1),
                               (area.bottomRight(), -1, -1)):
            painter.drawLine(corner, corner + QPointF(sx * arm, 0))
            painter.drawLine(corner, corner + QPointF(0, sy * arm))
        for middle, horizontal in ((QPointF(area.center().x(), area.top()), True), (QPointF(area.center().x(), area.bottom()), True),
                                   (QPointF(area.left(), area.center().y()), False), (QPointF(area.right(), area.center().y()), False)):
            offset = QPointF(arm / 2, 0) if horizontal else QPointF(0, arm / 2)
            painter.drawLine(middle - offset, middle + offset)
        if self.line is not None:
            painter.setPen(QPen(theme.color('accent'), 1.6, Qt.PenStyle.DashLine))
            painter.drawLine(self.to_screen(canvas, self.line[0]), self.to_screen(canvas, self.line[1]))

    @staticmethod
    def grid(area, kind):
        x, y, w, h = area.x(), area.y(), area.width(), area.height()
        lines = []
        def vertical(t):
            lines.append((QPointF(x + t * w, y), QPointF(x + t * w, y + h)))
        def horizontal(t):
            lines.append((QPointF(x, y + t * h), QPointF(x + w, y + t * h)))
        if kind == 'thirds':
            for t in (1 / 3, 2 / 3):
                vertical(t)
                horizontal(t)
        elif kind == 'golden':
            for t in (.382, .618):
                vertical(t)
                horizontal(t)
        elif kind == 'grid':
            for t in np.linspace(0, 1, 9)[1:-1]:
                vertical(float(t))
                horizontal(float(t))
        elif kind == 'diagonal':
            lines.append((QPointF(x, y), QPointF(x + w, y + h)))
            lines.append((QPointF(x + w, y), QPointF(x, y + h)))
        return lines


class CropBar(QFrame):
    """The crop tool's floating toolbar over the bottom of the photo."""

    def __init__(self, window, canvas):
        super().__init__(canvas)
        self.window = window
        self.setObjectName('cropBar')
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(6)

        def row():
            widget = QWidget()
            widget.setObjectName('cropRow')
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)
            outer.addWidget(widget)
            return layout

        def button(text, callback, tip='', checkable=False, primary=False):
            b = QPushButton(text)
            b.setCheckable(checkable)
            b.setToolTip(tip)
            if primary:
                b.setObjectName('primary')
            (b.toggled if checkable else b.clicked).connect(callback)
            return b
        first = row()
        self.ratio = QComboBox()
        for title, _ in RATIOS:
            self.ratio.addItem(tr(title))
        self.ratio.setToolTip(tr('裁切比例'))
        self.ratio.currentIndexChanged.connect(window.crop_ratio_changed)
        first.addWidget(self.ratio)
        self.swap = button('⇄', window.swap_crop_orientation, tr('横竖切换'))
        first.addWidget(self.swap)
        first.addSpacing(6)
        first.addWidget(button(tr('左转 90°'), lambda: window.rotate(-1), tr('逆时针旋转 90°')))
        first.addWidget(button(tr('右转 90°'), lambda: window.rotate(1), tr('顺时针旋转 90°')))
        first.addWidget(button(tr('水平翻转'), lambda: window.flip_image(0), tr('左右镜像')))
        first.addWidget(button(tr('垂直翻转'), lambda: window.flip_image(1), tr('上下镜像')))
        first.addStretch()
        self.grid = QComboBox()
        for _, title in GRIDS:
            self.grid.addItem(tr(title))
        self.grid.setToolTip(tr('参考线'))
        self.grid.currentIndexChanged.connect(lambda _: window.canvas.update())
        first.addWidget(self.grid)
        second = row()
        self.angle = AdjustSlider(tr('角度 °'), -model.STRAIGHTEN, model.STRAIGHTEN, .1)
        self.angle.setMinimumWidth(230)
        self.angle.changed.connect(window.straighten_changed)
        self.angle.committed.connect(window.commit)
        second.addWidget(self.angle, 1)
        self.auto = button(tr('自动水平'), window.auto_level,
                           tr('按画面中的水平线与竖直线自动拉直（海平线、建筑等）'))
        second.addWidget(self.auto)
        self.line = button(tr('拉直'), window.crop_line_toggled,
                           tr('沿画面中应当水平或竖直的线拖出一条直线，松开后按它拉直'), checkable=True)
        second.addWidget(self.line)
        second.addSpacing(6)
        second.addWidget(button(tr('重置'), window.reset_geometry, tr('清除裁切、角度、旋转与翻转')))
        second.addWidget(button(tr('取消'), lambda: window.finish_crop(False), tr('放弃这次裁切的改动（Esc）')))
        second.addWidget(button(tr('完成'), lambda: window.finish_crop(True), tr('确认裁切（Enter）'), primary=True))
        self.hide()

    def place(self):
        canvas = self.parentWidget()
        self.adjustSize()
        width = min(max(self.sizeHint().width(), 560), canvas.width() - 24)
        self.resize(width, self.sizeHint().height())
        self.move((canvas.width() - width) // 2, canvas.height() - self.height() - 14)


class CropMixin:
    """The crop tool's state on the main window."""

    def init_crop(self):
        self.crop_active = False
        self.crop_view = None
        self.crop_saved = None
        self.crop_final_before = False
        self.crop_tool = CropTool(self)

    def build_crop_tool(self):
        self.crop_bar = CropBar(self, self.canvas)
        self.canvas.cropper = self.crop_tool
        self.canvas.resized.connect(lambda: self.crop_bar.place() if self.crop_active else None)

    def crop_ratio_value(self):
        ratio = RATIOS[self.crop_bar.ratio.currentIndex()][1]
        if ratio == 'original':
            view = self.crop_view
            if view is None:
                return None
            ratio = view.width / view.height if view.turns % 2 == 0 else view.height / view.width
        return ratio

    def crop_grid_kind(self):
        return GRIDS[self.crop_bar.grid.currentIndex()][0]

    def crop_line_mode(self):
        return self.crop_bar.line.isChecked()

    def toggle_crop(self):
        if getattr(self, 'mode', 'develop') != 'develop':
            return                       # R in the library grid is not a crop
        if self.crop_active:
            self.finish_crop(True)
        else:
            self.start_crop()

    def start_crop(self):
        if self.source is None or self.crop_active or self.loading:
            self.crop_button.setChecked(self.crop_active)
            return
        self.crop_saved = geometry_of(self.edits)
        self.crop_final_before = self.final_view.isChecked()
        self.crop_active = True
        self.crop_button.setChecked(True)
        for name in ('wb_button', 'grade_pick_button', 'object_button', 'color_region_button'):
            button = getattr(self, name, None)
            if button is not None:
                button.setChecked(False)
        self.split_check.setChecked(False)
        self.final_view.blockSignals(True)
        self.final_view.setChecked(False)
        self.final_view.blockSignals(False)
        self.final_view.setEnabled(False)
        self.refresh_crop_bar()
        self.crop_bar.show()
        self.crop_bar.raise_()
        self.crop_bar.place()
        self.canvas.reserve = self.crop_bar.height() + 22
        self.update_display()
        self.canvas.fit()
        self.statusBar().showMessage(tr('裁切：拖动画面内部移动照片，拖动四角或四边调整范围，在框外拖动重画 · Enter 完成，Esc 取消'))

    def finish_crop(self, keep=True):
        if not self.crop_active:
            return
        if not keep and self.crop_saved is not None:
            self.edits.update(copy.deepcopy(self.crop_saved))
            self.changed()
        self.crop_active = False
        self.crop_view = None
        self.crop_bar.line.setChecked(False)
        self.crop_bar.hide()
        self.canvas.reserve = 0
        self.crop_button.setChecked(False)
        self.final_view.setEnabled(True)
        shaped = geometry_of(self.edits) != geometry_of(model.recipe())
        self.final_view.blockSignals(True)
        self.final_view.setChecked(shaped or (self.crop_final_before and not keep))
        self.final_view.blockSignals(False)
        self.commit(tr('裁切') if keep and geometry_of(self.edits) != self.crop_saved else None)
        self.canvas.fit()
        self.changed()
        self.update_display()
        self.statusBar().showMessage(tr('裁切已确认 · 仅显示保留部分，可继续调色或绘制蒙版') if keep else tr('已取消裁切的改动'))

    def refresh_crop_bar(self):
        if not hasattr(self, 'crop_bar'):
            return
        self.crop_bar.angle.setValue(self.edits.get('straighten', 0.))

    def set_crop(self, crop, commit=True):
        self.edits['crop'] = clean(crop)
        self.update_display()
        self.changed()
        if commit:
            self.commit()

    def crop_ratio_changed(self, *_):
        if not self.crop_active or self.crop_view is None:
            return
        ratio = self.crop_ratio_value()
        if ratio is None:
            return
        frame = self.crop_view.frame(self.edits.get('crop'))
        centre = ((frame[0] + frame[2]) / 2, (frame[1] + frame[3]) / 2)
        fitted = self.crop_view.largest(ratio, centre)
        if fitted is not None:
            self.set_crop(self.crop_view.crop_for(fitted))

    def swap_crop_orientation(self):
        if not self.crop_active or self.crop_view is None:
            return
        frame = self.crop_view.frame(self.edits.get('crop'))
        width, height = frame[2] - frame[0], frame[3] - frame[1]
        centre = ((frame[0] + frame[2]) / 2, (frame[1] + frame[3]) / 2)
        fitted = self.crop_view.largest(height / max(width, 1e-9), centre)
        if fitted is not None:
            index = self.crop_bar.ratio.currentIndex()
            ratio = RATIOS[index][1]
            if isinstance(ratio, float):
                swapped = next((i for i, (_, r) in enumerate(RATIOS) if isinstance(r, float) and abs(r - 1 / ratio) < 1e-6), index)
                self.crop_bar.ratio.blockSignals(True)
                self.crop_bar.ratio.setCurrentIndex(swapped)
                self.crop_bar.ratio.blockSignals(False)
            self.set_crop(self.crop_view.crop_for(fitted))

    def rotate(self, turns=1):
        """Quarter turns (clockwise positive); outside the crop tool the result is shown at once."""
        if self.source is None:
            return
        self.edits['rotation'] = (self.edits['rotation'] + turns) % 4
        if not self.crop_active:
            self.final_view.setChecked(True)
        self.update_display()
        self.changed()
        self.commit()

    def flip_image(self, axis):
        if self.source is None:
            return
        flip = list(self.edits.get('flip') or [False, False])
        flip[axis] = not flip[axis]
        self.edits['flip'] = flip
        if not self.crop_active:
            self.final_view.setChecked(True)
        self.update_display()
        self.changed()
        self.commit()

    def straighten_changed(self, value):
        """A new angle keeps the frame's centre and size where it fits, shrinking it where not."""
        if self.refreshing or self.source is None:
            return
        old = self.edits.get('straighten', 0.)
        crop = self.edits.get('crop')
        if crop is not None and self.info:
            size = (self.info.get('width', self.source.shape[1]), self.info.get('height', self.source.shape[0]))
            before = View(self.edits, size)
            frame = before.frame(crop)
            self.edits['straighten'] = float(value)
            after = View(self.edits, size)
            centre = after.to_view(before.to_source(((frame[0] + frame[2]) / 2, (frame[1] + frame[3]) / 2)))
            width, height = frame[2] - frame[0], frame[3] - frame[1]
            fitted = after.shrink([centre[0] - width / 2, centre[1] - height / 2, centre[0] + width / 2,
                                   centre[1] + height / 2], centre)
            if fitted is not None:
                self.edits['crop'] = clean(after.crop_for(fitted))
        self.edits['straighten'] = float(value)
        if old != value:
            if not self.crop_active:
                self.final_view.setChecked(True)
            self.update_display()
            self.changed()

    def set_straighten(self, angle, label=None):
        angle = float(np.clip(angle, -model.STRAIGHTEN, model.STRAIGHTEN))
        self.straighten_changed(angle)
        old, self.refreshing = self.refreshing, True
        self.crop_bar.angle.setValue(angle)
        self.refreshing = old
        self.commit(label)

    def auto_level(self):
        """自动水平: the angle from the photo's horizon and verticals (``straighten``)."""
        if self.source is None:
            return
        from . import engine, straighten
        source, edits = self.source, copy.deepcopy(self.edits)
        self.crop_bar.auto.setEnabled(False)
        self.statusBar().showMessage(tr('正在寻找水平线与竖直线…'))

        def work():
            return straighten.estimate(engine.selection_view(source, edits))

        def done(result):
            self.crop_bar.auto.setEnabled(True)
            angle, _ = result
            if source is not self.source:
                return
            if angle is None:
                self.statusBar().showMessage(tr('画面中没有足够清楚的水平线或竖直线，请用“拉直”沿一条线拖动。'))
                return
            self.set_straighten(angle, tr('自动水平 {angle}°', angle=f'{angle:+.1f}'))
            self.statusBar().showMessage(tr('已自动拉直 {angle}°，可继续微调角度。', angle=f'{angle:+.1f}'))

        def failed(text):
            self.crop_bar.auto.setEnabled(True)
            self.error(text)
        self.job(work, done, failed)

    def crop_line_toggled(self, checked):
        if checked:
            self.statusBar().showMessage(tr('拉直：沿一条应当水平或竖直的线拖动。'))
        self.canvas.setCursor(Qt.CursorShape.CrossCursor if checked else Qt.CursorShape.ArrowCursor)

    def straighten_along(self, line):
        """拉直 along a line drawn in the view: it becomes horizontal, or vertical when steep."""
        view = self.crop_view
        if view is None:
            return
        (x0, y0), (x1, y1) = view.to_source(line[0]), view.to_source(line[1])
        dx, dy = x1 - x0, y1 - y0
        if dx < 0:
            dx, dy = -dx, -dy
        slope = math.degrees(math.atan2(dy, dx))
        if abs(slope) <= 45:
            angle = slope
        else:
            if dy < 0:
                dx, dy = -dx, -dy
            angle = math.degrees(math.atan2(-dx, dy))
        self.crop_bar.line.setChecked(False)
        self.set_straighten(angle, tr('拉直 {angle}°', angle=f'{angle:+.1f}'))

    def reset_geometry(self):
        if self.source is None:
            return
        self.edits.update(geometry_of(model.recipe()))
        self.refresh_crop_bar()
        self.update_display()
        self.changed()
        self.commit()
