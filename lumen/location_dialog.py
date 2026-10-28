"""Choosing where a photo was taken on an offline map (2.1.2).

The 水印 panel shows the photo's GPS position (the camera's, or one chosen here) and opens this
window: the world map of ``world_map`` (Natural Earth, no network), a search for towns in every
interface language and for typed coordinates, and a pin placed with a click.  ``edit_location``
writes the choice into the photo's XMP sidecar (``geo.write``; the photo itself is never
changed), the library and the open photo, so the border's pin row and the exports use it.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMessageBox, QPushButton, QToolButton, QVBoxLayout, QWidget)

from . import geo, i18n, photo_metadata, theme, world_map
from .i18n import tr

log = logging.getLogger(__name__)

#: The pin's colour (a map pin, as in the border).
PIN = QColor(224, 70, 60)


def pin_path(tip, height):
    """A map pin whose point is at ``tip``: a round head and a point, ``height`` pixels tall."""
    radius = height * .36
    center = QPointF(tip.x(), tip.y() - height + radius)
    # The sides touch the head where the angle at its centre between the tip and that point is
    # acos(radius / distance); the arc runs from the right one over the top to the left one.
    alpha = math.degrees(math.acos(radius / (height - radius)))
    path = QPainterPath()
    path.moveTo(tip)
    path.arcTo(QRectF(center.x() - radius, center.y() - radius, 2 * radius, 2 * radius), -90 + alpha, 360 - 2 * alpha)
    path.closeSubpath()
    return path, center, radius


def draw_pin(painter, tip, height=30):
    path, center, radius = pin_path(tip, height)
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor(0, 0, 0, 60), 4))
    painter.drawPath(path.translated(1, 2))
    painter.setPen(QPen(PIN.darker(140), 1.2))
    painter.setBrush(PIN)
    painter.drawPath(path)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(255, 255, 255))
    painter.drawEllipse(center, radius * .42, radius * .42)
    painter.restore()


class MapView(QWidget):
    """The map: drag to move, wheel or double-click to zoom, click to place the pin."""
    picked = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(560, 340)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.language = i18n.language()
        self.renderer = world_map.Renderer(theme.mode())
        self.view = world_map.View(geo.to_world(30, 105 if self.language.startswith(('zh', 'ja', 'ko')) else 10), 2.,
                                   self.width(), self.height())
        self.position = None
        self._image = None
        self._press = None
        self._moved = False
        for text, step in (('+', 1), ('−', -1)):
            button = QToolButton(self)
            button.setText(text)
            button.setFixedSize(30, 30)
            button.setToolTip(tr('放大') if step > 0 else tr('缩小'))
            button.clicked.connect(lambda _=False, s=step: self.zoom_by(s))
            setattr(self, 'zoom_in' if step > 0 else 'zoom_out', button)

    # -- the view

    def _set(self, center, zoom):
        self.view = world_map.clamp(world_map.View(center, zoom, max(1, self.width()), max(1, self.height())))
        self._image = None
        self.update()

    def show_position(self, position, zoom=None):
        self._set(geo.to_world(*position), self.view.zoom if zoom is None else zoom)

    def set_pin(self, position):
        self.position = geo.normalized(position) if position is not None else None
        self.update()

    def zoom_by(self, steps, anchor=None):
        anchor = anchor or QPointF(self.width() / 2, self.height() / 2)
        before = self.view.to_world(anchor.x(), anchor.y())
        zoom = min(world_map.MAX_ZOOM, max(world_map.MIN_ZOOM, self.view.zoom + steps))
        scale = world_map.TILE * 2 ** zoom
        center = (before[0] - (anchor.x() - self.width() / 2) / scale, before[1] - (anchor.y() - self.height() / 2) / scale)
        self._set(center, zoom)

    def resizeEvent(self, event):
        self._set(self.view.center, self.view.zoom)
        self.zoom_in.move(10, 10)
        self.zoom_out.move(10, 44)
        super().resizeEvent(event)

    # -- drawing

    def _map(self):
        if self._image is None or self._image.size() != self.size():
            rgb = np.ascontiguousarray(self.renderer.render(self.view))
            self._image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format.Format_RGB888).copy()
        return self._image

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self._map())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self._labels(painter)
        if self.position is not None:
            x, y = geo.to_world(*self.position)
            for offset in self.view.offsets():
                px, py = self.view.to_pixel(x + offset, y)
                if -40 <= px <= self.width() + 40 and -10 <= py <= self.height() + 50:
                    draw_pin(painter, QPointF(px, py))
        font = QFont(self.font())
        font.setPointSizeF(max(7., font.pointSizeF() * .8))
        painter.setFont(font)
        dark = theme.mode() == 'dark'
        painter.setPen(QColor(200, 205, 210) if dark else QColor(70, 80, 88))
        painter.drawText(QRectF(0, 0, self.width() - 8, self.height() - 4),
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom, tr('地图数据：Natural Earth（公有领域）'))
        painter.end()

    def _labels(self, painter):
        dark = theme.mode() == 'dark'
        text, halo = (QColor(226, 224, 216), QColor(10, 14, 18, 200)) if dark else (QColor(48, 46, 42), QColor(255, 255, 255, 210))
        font = QFont(self.font())
        font.setPointSizeF(max(7.5, font.pointSizeF() * .92))
        painter.setFont(font)
        metrics = QFontMetricsF(font)
        taken = []
        shown = 0
        for px, py, name, place in self.renderer.labels(self.view, self.language):
            if not name:
                continue
            width = metrics.horizontalAdvance(name)
            box = QRectF(px + 5, py - metrics.height() / 2, width + 2, metrics.height())
            if box.right() > self.width() - 4:
                box.moveRight(px - 5)
            dot = QRectF(px - 3, py - 3, 6, 6)
            if any(box.intersects(other) or dot.intersects(other) for other in taken):
                continue
            taken += [box.adjusted(-3, -1, 3, 1), dot]
            painter.setPen(QPen(QColor(text).darker(110), 1))
            painter.setBrush(QColor(255, 255, 255) if dark else QColor(60, 60, 60))
            painter.drawEllipse(QPointF(px, py), 2.4 if place.zoom > 4 else 3.2, 2.4 if place.zoom > 4 else 3.2)
            path = QPainterPath()
            path.addText(QPointF(box.left() + 1, box.top() + metrics.ascent()), font, name)
            painter.setPen(QPen(halo, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(text)
            painter.drawPath(path)
            shown += 1
            if shown >= 120:
                break

    # -- the mouse and the keyboard

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press = (event.position(), self.view.center)
            self._moved = False
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._press is None or not event.buttons() & Qt.MouseButton.LeftButton:
            return
        start, center = self._press
        delta = event.position() - start
        if not self._moved and abs(delta.x()) + abs(delta.y()) < 4:
            return
        self._moved = True
        scale = self.view.scale
        self._set((center[0] - delta.x() / scale, center[1] - delta.y() / scale), self.view.zoom)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self._press is None:
            return
        self.setCursor(Qt.CursorShape.CrossCursor)
        clicked = not self._moved
        self._press = None
        if clicked:
            lat, lon = self.view.position(event.position().x(), event.position().y())
            self.set_pin((lat, lon))
            self.picked.emit(lat, lon)

    def mouseDoubleClickEvent(self, event):
        self.zoom_by(1, event.position())

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120
        if steps:
            self.zoom_by(steps * .5, event.position())

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_by(1)
        elif key == Qt.Key.Key_Minus:
            self.zoom_by(-1)
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            dx = {Qt.Key.Key_Left: -1, Qt.Key.Key_Right: 1}.get(key, 0) * self.width() / 4
            dy = {Qt.Key.Key_Up: -1, Qt.Key.Key_Down: 1}.get(key, 0) * self.height() / 4
            scale = self.view.scale
            self._set((self.view.center[0] + dx / scale, self.view.center[1] + dy / scale), self.view.zoom)
        else:
            super().keyPressEvent(event)


def describe(position):
    """``31°13′49″N 121°28′25″E · 上海附近 7 km`` for the info line."""
    text = geo.dms(position)
    found = world_map.nearest(position)
    if found is not None:
        place, km = found
        name = world_map.names(i18n.language())[place.index]
        text += ' · ' + (name if km < 3 else tr('{place}附近 {km} km', place=name, km=round(km)))
    return text


class LocationDialog(QDialog):
    """Where a photo was taken: ``position`` after ``accept`` (None: remove the chosen position)."""

    def __init__(self, parent, name, current=None, source=''):
        super().__init__(parent)
        self.setWindowTitle(tr('拍摄位置 — {name}', name=name))
        self.resize(980, 700)
        self.position = current
        self.source = source if current is not None else ''
        root = QVBoxLayout(self)
        search = QHBoxLayout()
        self.query = QLineEdit()
        self.query.setPlaceholderText(tr('搜索城市，或输入坐标（如 31.2304, 121.4737）'))
        self.query.setClearButtonEnabled(True)
        self.query.textChanged.connect(self.find)
        self.query.returnPressed.connect(self.take_first)
        search.addWidget(self.query, 1)
        root.addLayout(search)
        self.results = QListWidget()
        self.results.hide()
        self.results.itemActivated.connect(self.choose)
        self.results.itemClicked.connect(self.choose)
        root.addWidget(self.results)
        self.map = MapView(self)
        self.map.picked.connect(lambda lat, lon: self.set_position((lat, lon)))
        root.addWidget(self.map, 1)
        self.info = QLabel()
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.info)
        hint = QLabel(tr('点击地图放置图钉，滚轮或 +/− 缩放，拖动平移。坐标为 WGS-84（相机和手机 GPS 记录的坐标系）。\n'
                         '位置写入照片旁的 XMP 附属文件（Lightroom 等软件也能读取），原片不会被修改；导出时写入成片的 EXIF。'))
        hint.setObjectName('subtle')
        hint.setWordWrap(True)
        root.addWidget(hint)
        buttons = QHBoxLayout()
        self.clear_button = QPushButton(tr('清除位置'))
        self.clear_button.clicked.connect(self.clear)
        self.clear_button.setEnabled(source == 'xmp')
        self.clear_button.setToolTip(tr('删除写入 XMP 附属文件的位置（相机记录的 GPS 不受影响）'))
        buttons.addWidget(self.clear_button)
        buttons.addStretch(1)
        cancel = QPushButton(tr('取消'))
        cancel.clicked.connect(self.reject)
        self.ok_button = QPushButton(tr('写入照片信息'))
        self.ok_button.setObjectName('primary')
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(self.ok_button)
        root.addLayout(buttons)
        self.cleared = False
        if current is not None:
            self.map.set_pin(current)
            self.map.show_position(current, 10.)
        self.refresh()

    def set_position(self, position):
        self.position = geo.normalized(position)
        self.source = 'map'
        self.map.set_pin(self.position)
        self.refresh()

    def refresh(self):
        if self.position is None:
            self.info.setText(tr('这张照片没有 GPS 位置：搜索城市或在地图上点击放置图钉。'))
        else:
            origin = {'camera': tr('（相机记录）'), 'xmp': tr('（已写入 XMP）')}.get(self.source, '')
            self.info.setText(tr('位置：{place}', place=describe(self.position)) + origin + '    ' + geo.decimal(self.position))
        self.ok_button.setEnabled(self.position is not None and self.source == 'map')

    def find(self, text):
        self.results.clear()
        text = text.strip()
        if not text:
            self.results.hide()
            return
        try:
            position = geo.parse(text)
        except ValueError:
            position = None
        if position is not None:
            item = QListWidgetItem(tr('坐标 {place}', place=geo.dms(position)))
            item.setData(Qt.ItemDataRole.UserRole, (position, 12.))
            self.results.addItem(item)
        for name, place in world_map.search(text, i18n.language(), 10):
            item = QListWidgetItem(f'{name}    {geo.dms((place.lat, place.lon))}')
            item.setData(Qt.ItemDataRole.UserRole, ((place.lat, place.lon), 10. if place.zoom <= 4 else 11.))
            self.results.addItem(item)
        if not self.results.count():
            self.results.addItem(QListWidgetItem(tr('没有找到。试试其他写法，或直接输入坐标。')))
        rows = min(6, self.results.count())
        self.results.setFixedHeight(rows * max(18, self.results.sizeHintForRow(0)) + 2 * self.results.frameWidth() + 8)
        self.results.setVisible(True)

    def take_first(self):
        if self.results.count():
            self.choose(self.results.item(0))

    def choose(self, item):
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data:
            return
        position, zoom = data
        self.set_position(position)
        self.map.show_position(position, zoom)
        self.results.hide()

    def clear(self):
        self.position = None
        self.cleared = True
        super().accept()


def edit_location(window):
    """The 水印 panel's “在地图上选择…”: choose, write into the sidecar, apply to the open photo."""
    if window.source is None or not window.source_path:
        return False
    path = window.source_path
    photo = window.info.get('photo') or {}
    current = photo_metadata.position(photo)
    dialog = LocationDialog(window, Path(path).name, current, photo.get('gps_source', '') if current else '')
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return False
    position = None if dialog.cleared else dialog.position
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        errors = geo.write(path, position)
        if position is None:
            # The camera's own GPS, if any, shows again.
            from . import white_balance
            camera = photo_metadata.position(photo_metadata.from_tags(white_balance.metadata(path)[0]))
            updated = photo_metadata.with_position(photo, camera, 'camera')
        else:
            updated = photo_metadata.with_position(photo, position, 'xmp')
    finally:
        QApplication.restoreOverrideCursor()
    if errors:
        QMessageBox.warning(window, 'LUMEN RAW', tr('位置没有写入 XMP 附属文件：{error}', error='; '.join(errors[:3])))
        return False
    window.info['photo'] = updated
    try:
        window.library_db.set_position(path, photo_metadata.position(updated))
    except Exception:
        log.warning('library position not updated', exc_info=True)
    window.location_changed()
    return True
