"""Full-screen culling (2.0.0): go through a shoot quickly with the keyboard.

The photos show as the camera rendered them (``previews.camera_view``, cached), so moving to the
next one takes no RAW decode; the neighbours are read ahead.  Z (or a click) zooms to 100 % —
from the camera's full-size embedded JPEG when there is one, otherwise from the RAW.  C shows two
photos side by side (the select on the left, a candidate on the right that ← / → change; Tab
switches which side the keys rate; ↑ makes the candidate the select), with zoom and position
shared.  Frames of a burst are numbered and the best one (eyes open, sharpest) is marked; ↑ / ↓
in single view jump between bursts.  Sharpness and closed-eye hints are shown as they are
computed.  Keys: 0–5 stars, P pick, X reject, U unflag, 6–9 colour labels, A advance after
rating, F faces, Enter open in the develop view, Esc / G back to the library.
"""
from __future__ import annotations

import collections
import json
import logging
import threading

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, QSize, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QLabel, QListView, QPushButton, QVBoxLayout,
                               QWidget, QCheckBox)

from . import catalog, library_db, theme
from .i18n import tr
from .library_view import GridDelegate, LibraryModel, ROW_ROLE, camera_text, exposure_text, hint_text

log = logging.getLogger(__name__)

#: Camera views kept decoded (each about 13 MB at 2560 × 1707).
VIEWS = 14
#: Photos read ahead on each side of the current one.
AHEAD = 3
#: Full-size photos kept for 100 % views (one per compared side; each can be 150–200 MB).
FULL = 2


def qimage_of(array):
    array = np.ascontiguousarray(array)
    return QImage(array.data, array.shape[1], array.shape[0], array.strides[0], QImage.Format.Format_RGB888).copy()


class _Signals(QObject):
    loaded = Signal(str, QImage, bool)


class _Load(QRunnable):
    def __init__(self, loader, path, full):
        super().__init__()
        self.loader, self.path, self.full = loader, path, full

    def run(self):
        try:
            if self.full:
                array = full_view(self.path)
            else:
                array = self.loader.cache.view(self.path)
            image = qimage_of(array)
        except Exception:
            log.warning('culling view of %s failed', self.path, exc_info=True)
            image = QImage()
        self.loader.signals.loaded.emit(self.path, image, self.full)


def full_view(path):
    """The photo at full size for checking focus: the camera's full-size embedded JPEG, or the
    RAW (or other file) decoded and developed."""
    from . import engine, previews, dng
    from pathlib import Path
    suffix = Path(path).suffix.lower()
    if suffix in engine.RAW_EXTENSIONS and not (suffix == '.dng' and dng.is_rendered(path)):
        found = previews.embedded_view(path, None)
        if found is not None and found[1]:
            return found[0]
        rgb, info = engine.load_image(path, None)
        developed = engine.develop_view(rgb, dict(develop=info.get('develop', {}), process=2))
        return np.round(np.clip(developed, 0, 1) * 255).astype(np.uint8)
    return previews.camera_view(path, None)


class ViewLoader(QObject):
    """Camera views by path (LRU), read in two background threads."""
    loaded = Signal(str)

    def __init__(self, cache, parent=None):
        super().__init__(parent)
        self.cache = cache
        self.views = collections.OrderedDict()
        self.full = collections.OrderedDict()
        self.pending = set()
        self.signals = _Signals()
        self.signals.loaded.connect(self._loaded)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)

    def image(self, path, full=False):
        if full and path in self.full:
            self.full.move_to_end(path)
            return self.full[path]
        image = self.views.get(path)
        if image is not None:
            self.views.move_to_end(path)
        self.request(path, full)
        return image

    def request(self, path, full=False, front=True):
        if (path, full) in self.pending or (full and path in self.full) or (not full and path in self.views):
            return
        self.pending.add((path, full))
        self.pool.start(_Load(self, path, full), 1 if front else 0)

    def _loaded(self, path, image, full):
        self.pending.discard((path, full))
        if image.isNull():
            return
        if full:
            self.full[path] = image
            self.full.move_to_end(path)
            while len(self.full) > FULL:
                self.full.popitem(last=False)
        else:
            self.views[path] = image
            self.views.move_to_end(path)
            while len(self.views) > VIEWS:
                self.views.popitem(last=False)
        self.loaded.emit(path)

    def shutdown(self):
        self.pool.clear()
        self.pool.waitForDone(3000)


class Pane(QWidget):
    """One photo, fitted or at 100 %, with its faces and a frame when it takes the keys."""
    clicked = Signal(object)
    zoom_changed = Signal(object)

    def __init__(self, cull, parent=None):
        super().__init__(parent)
        self.cull = cull
        self.row = None
        self.zoomed = False
        self.center = QPointF(.5, .5)      # point of the photo at the middle of the pane (0–1)
        self.active = False
        self.drag = None
        self.setMouseTracking(True)
        self.setMinimumSize(200, 150)

    def set_row(self, row):
        self.row = row
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#050506'))
        if self.row is None:
            return
        path = self.row['path']
        full = self.cull.loader.image(path, True) if self.zoomed else None
        image = full if full is not None else self.cull.loader.image(path)
        if image is None:
            painter.setPen(QColor('#8a8780'))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, tr('正在读取…'))
            return
        area = QRectF(self.rect()).adjusted(8, 8, -8, -8)
        if self.zoomed:
            # 100 % of the full-size photo (the view, enlarged, until it arrives).
            enlarge = 1. if full is not None else self.full_edge() / max(image.width(), image.height())
            scale = enlarge / self.devicePixelRatioF()
            width, height = image.width() * scale, image.height() * scale
            x = area.center().x() - self.center.x() * width
            y = area.center().y() - self.center.y() * height
            target = QRectF(x, y, width, height)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, full is None)
            painter.drawImage(target, image)
        else:
            scale = min(area.width() / image.width(), area.height() / image.height())
            width, height = image.width() * scale, image.height() * scale
            target = QRectF(area.center().x() - width / 2, area.center().y() - height / 2, width, height)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawImage(target, image)
        self.target = target
        if self.cull.show_faces and self.row.get('face_data'):
            try:
                boxes = json.loads(self.row['face_data'])
            except ValueError:
                boxes = []
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            for x, y, w, h, closed in boxes:
                rect = QRectF(target.left() + x * target.width(), target.top() + y * target.height(),
                              w * target.width(), h * target.height())
                painter.setPen(QPen(QColor('#e6c84f') if closed else QColor('#8ed7ac'), 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(rect, 6, 6)
                if closed:
                    painter.drawText(rect.adjusted(0, -20, 0, 0).topLeft() + QPointF(2, 14), tr('闭眼？'))
        if self.cull.compare:
            painter.setPen(QPen(theme.color('accent') if self.active else QColor('#2a2a30'), 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5))
        if full is None and self.zoomed and self.row is not None:
            painter.setPen(QColor('#d8d4cc'))
            painter.drawText(self.rect().adjusted(0, 0, -16, -12), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                             tr('正在读取原图…'))

    def full_edge(self):
        """Long edge of the full-size photo (from its capture data) while only the view is at hand."""
        edge = max(self.row.get('width') or 0, self.row.get('height') or 0) if self.row else 0
        return edge or 6000

    def mousePressEvent(self, event):
        self.clicked.emit(self)
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag = (event.position(), QPointF(self.center))
            self.moved = False

    def mouseMoveEvent(self, event):
        if self.drag is None or not self.zoomed or not hasattr(self, 'target'):
            return
        start, center = self.drag
        delta = event.position() - start
        if abs(delta.x()) + abs(delta.y()) > 3:
            self.moved = True
        self.center = QPointF(min(1, max(0, center.x() - delta.x() / self.target.width())),
                              min(1, max(0, center.y() - delta.y() / self.target.height())))
        self.zoom_changed.emit(self)
        self.update()

    def mouseReleaseEvent(self, event):
        if self.drag is not None and not getattr(self, 'moved', False) and event.button() == Qt.MouseButton.LeftButton:
            self.toggle_zoom(event.position())
        self.drag = None

    def toggle_zoom(self, position=None):
        if self.row is None:
            return
        if not self.zoomed and position is not None and hasattr(self, 'target') and self.target.contains(position):
            self.center = QPointF((position.x() - self.target.left()) / self.target.width(),
                                  (position.y() - self.target.top()) / self.target.height())
        elif not self.zoomed:
            self.center = QPointF(.5, .5)
        self.zoomed = not self.zoomed
        if self.zoomed:
            self.cull.loader.request(self.row['path'], True)
        self.zoom_changed.emit(self)
        self.update()


class CullWindow(QWidget):
    closed = Signal(str)

    def __init__(self, window, rows, start=0):
        super().__init__(None, Qt.WindowType.Window)
        self.window = window
        self.setWindowTitle(tr('选片 · LUMEN RAW'))
        self.setObjectName('cullWindow')
        self.setStyleSheet('QWidget#cullWindow { background: #050506; } QLabel { background: transparent; color: #d8d4cc; }'
                           ' QLabel#cullHints { color: #e6c84f; } QLabel#cullHelp { color: #7c7972; font-size: 11px; }')
        self.paths = [r['path'] for r in rows]
        self.rows = {}
        self.index = max(0, min(start, len(self.paths) - 1))
        self.candidate = min(self.index + 1, len(self.paths) - 1)
        self.compare = False
        self.active_right = False
        self.advance = False
        self.show_faces = True
        self.loader = ViewLoader(window.previews, self)
        self.loader.loaded.connect(self.view_loaded)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QHBoxLayout()
        header.setContentsMargins(18, 10, 18, 6)
        header.setSpacing(14)
        self.title = QLabel()
        font = self.title.font()
        font.setPointSizeF(font.pointSizeF() * 1.25)
        self.title.setFont(font)
        header.addWidget(self.title)
        self.details = QLabel()
        header.addWidget(self.details, 1)
        self.hints = QLabel()
        self.hints.setObjectName('cullHints')
        header.addWidget(self.hints)
        self.advance_check = QCheckBox(tr('评级后前进（A）'))
        self.advance_check.toggled.connect(self.set_advance)
        header.addWidget(self.advance_check)
        self.compare_button = QPushButton(tr('对比（C）'))
        self.compare_button.setCheckable(True)
        self.compare_button.toggled.connect(self.set_compare)
        header.addWidget(self.compare_button)
        close = QPushButton(tr('返回（Esc）'))
        close.clicked.connect(self.close)
        header.addWidget(close)
        layout.addLayout(header)
        panes = QHBoxLayout()
        panes.setContentsMargins(6, 0, 6, 0)
        panes.setSpacing(6)
        self.left, self.right = Pane(self), Pane(self)
        for pane in (self.left, self.right):
            pane.clicked.connect(self.pane_clicked)
            pane.zoom_changed.connect(self.sync_zoom)
            panes.addWidget(pane, 1)
        self.right.hide()
        layout.addLayout(panes, 1)
        self.strip_model = LibraryModel(self)
        self.strip = QListView()
        self.strip.setModel(self.strip_model)
        self.strip.setViewMode(QListView.ViewMode.IconMode)
        self.strip.setFlow(QListView.Flow.LeftToRight)
        self.strip.setWrapping(False)
        self.strip.setMovement(QListView.Movement.Static)
        self.strip.setUniformItemSizes(True)
        self.strip.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.strip.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.strip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.strip_delegate = GridDelegate(window.thumb_images, self.strip, 96, names=False)
        self.strip.setItemDelegate(self.strip_delegate)
        self.strip.setGridSize(self.strip_delegate.cell())
        self.strip.setFixedHeight(self.strip_delegate.cell().height() + 12)
        self.strip.setStyleSheet('QListView { background: #08080a; border: 0; }')
        self.strip.clicked.connect(lambda index: self.go(index.row()))
        layout.addWidget(self.strip)
        self.help = QLabel(tr('← → 切换  ·  ↑ ↓ 上／下一组连拍  ·  0–5 星级  ·  P 留用  X 排除  U 取消  ·  6–9 色标  ·  '
                              'Z 或单击 100%  ·  C 对比  Tab 切换左右  ·  F 人脸框  ·  Enter 显影  ·  Esc 返回'))
        self.help.setObjectName('cullHelp')
        self.help.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.help.setContentsMargins(0, 2, 0, 8)
        layout.addWidget(self.help)
        self.refresh_rows()
        self.strip_model.set_rows([self.rows[p] for p in self.paths])
        self.go(self.index)
        # Every key belongs to the culling view: buttons never take the focus (Space, arrows, Tab).
        for widget in (self.advance_check, self.compare_button, close):
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def focusNextPrevChild(self, forward):
        return False                     # Tab switches the compared side instead

    # -- showing

    def show_fullscreen(self):
        screen = self.window.screen() if hasattr(self.window, 'screen') else None
        if screen is not None:
            self.setGeometry(screen.geometry())
        self.showFullScreen()
        self.activateWindow()
        self.setFocus()

    def refresh_rows(self):
        """Rows of the photos being culled, from the library, with bursts within this set."""
        rows = self.window.library_db.rows(('paths', self.paths))
        library_db.annotate_bursts(rows)
        library_db.annotate_hints(rows)
        found = {r['path']: r for r in rows}
        fresh = [found.get(path) or dict(path=path, name=path.rsplit('\\', 1)[-1].rsplit('/', 1)[-1], id=None)
                 for path in self.paths]
        library_db.with_ratings(fresh, self.window.catalog)
        for path, row in zip(self.paths, fresh):
            if path in self.rows:
                self.rows[path].clear()
                self.rows[path].update(row)
            else:
                self.rows[path] = row

    def rows_changed(self, ids):
        if not any(self.rows[p].get('id') in ids for p in self.paths):
            return
        self.refresh_rows()
        self.strip_model.refresh_all()
        self.update_header()
        self.left.update()
        self.right.update()

    def ratings_changed(self):
        library_db.with_ratings([self.rows[p] for p in self.paths], self.window.catalog)
        self.strip_model.refresh_all()
        self.update_header()

    def thumbnail_arrived(self, path):
        self.strip_model.changed(path)

    def view_loaded(self, path):
        for pane in (self.left, self.right):
            if pane.row is not None and pane.row['path'] == path:
                pane.update()

    def current_row(self):
        return self.rows[self.paths[self.index]]

    def target_row(self):
        if self.compare and self.active_right:
            return self.rows[self.paths[self.candidate]]
        return self.current_row()

    def go(self, index):
        if not self.paths:
            return
        if self.compare:
            index = max(0, min(index, len(self.paths) - 1))
            if index == self.index:
                return
            self.candidate = index
        else:
            self.index = max(0, min(index, len(self.paths) - 1))
        self.show_current()

    def show_current(self):
        # The zoom and position stay when moving on, to check focus frame after frame.
        self.left.set_row(self.current_row())
        if self.compare:
            self.right.set_row(self.rows[self.paths[self.candidate]])
        for pane in (self.left, self.right):
            if pane.zoomed and pane.row is not None and pane.isVisible():
                self.loader.request(pane.row['path'], True)
        focus = self.candidate if self.compare else self.index
        self.strip.setCurrentIndex(self.strip_model.index(focus))
        self.strip.scrollTo(self.strip_model.index(focus), QAbstractItemView.ScrollHint.PositionAtCenter)
        for offset in range(1, AHEAD + 1):
            for position in (focus + offset, focus - offset):
                if 0 <= position < len(self.paths):
                    self.loader.request(self.paths[position], front=False)
        from .library_tasks import possible
        from .library_view import analysis_requests
        unanalysed = analysis_requests([dict(self.rows[p], path=p) for p in self.paths[max(0, focus - 2):focus + 6]],
                                       possible(self.window.library_worker.parts))
        if unanalysed:
            self.window.library_worker.want_analysis(unanalysed, first=True)
        self.update_header()

    def update_header(self):
        row = self.target_row()
        position = self.paths.index(row['path']) + 1
        self.title.setText(f'{row["name"]}   {position} / {len(self.paths)}')
        details = [row.get('captured') or '', camera_text(row), exposure_text(row)]
        meta = []
        if row.get('rating'):
            meta.append('★' * int(row['rating']))
        if row.get('flag'):
            meta.append({'pick': tr('留用'), 'reject': tr('排除')}[row['flag']])
        self.details.setText('    ' + '   ·   '.join(d for d in details + meta if d))
        self.hints.setText(hint_text(row) if row.get('id') else '')
        self.left.active = not self.active_right
        self.right.active = self.active_right
        self.left.update()
        self.right.update()

    # -- compare and zoom

    def set_compare(self, on):
        if on == self.compare:
            return
        if on and len(self.paths) < 2:
            self.compare_button.setChecked(False)
            return
        self.compare = on
        self.compare_button.setChecked(on)
        if on:
            self.candidate = self.index + 1 if self.index + 1 < len(self.paths) else self.index - 1
            self.active_right = True
            self.right.zoomed = self.left.zoomed
            self.right.center = QPointF(self.left.center)
            self.right.show()
        else:
            self.index = self.candidate if self.active_right else self.index
            self.active_right = False
            self.right.hide()
        self.show_current()

    def pane_clicked(self, pane):
        if self.compare:
            self.active_right = pane is self.right
            self.update_header()

    def sync_zoom(self, pane):
        if not self.compare:
            return
        other = self.right if pane is self.left else self.left
        if other.zoomed != pane.zoomed and pane.zoomed and other.row is not None:
            self.loader.request(other.row['path'], True)
        other.zoomed, other.center = pane.zoomed, QPointF(pane.center)
        other.update()

    def promote(self):
        """↑ in comparison: the candidate becomes the select."""
        if self.compare:
            self.index, self.candidate = self.candidate, self.index
            self.show_current()

    def jump_burst(self, step):
        row = self.current_row()
        group = row.get('burst') or 0
        position = self.index
        while 0 <= position + step < len(self.paths):
            position += step
            other = self.rows[self.paths[position]]
            if other.get('burst') != group or not group:
                if step < 0 and other.get('burst'):
                    # to the start of the previous burst
                    while position > 0 and self.rows[self.paths[position - 1]].get('burst') == other.get('burst'):
                        position -= 1
                self.go(position)
                return

    # -- ratings

    def rate(self, **fields):
        row = self.target_row()
        (field, value), = fields.items()
        if field == 'label' and value and row.get('label') == value:
            value = ''
        self.window.rate_paths([row['path']], **{field: value})
        row.update(self.window.catalog.get(row['path']))
        self.strip_model.changed(row['path'])
        self.update_header()
        if self.advance:
            if self.compare:
                self.go(self.candidate + 1)
            else:
                self.go(self.index + 1)

    def set_advance(self, on):
        self.advance = on
        self.advance_check.setChecked(on)

    # -- keys

    def keyPressEvent(self, event):
        key, text = event.key(), event.text().upper()
        modifiers = event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
        if modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier):
            return super().keyPressEvent(event)
        position = self.candidate if self.compare else self.index
        if key in (Qt.Key.Key_Right, Qt.Key.Key_PageDown):
            self.go(position + 1)
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_PageUp):
            self.go(position - 1)
        elif key == Qt.Key.Key_Home:
            self.go(0)
        elif key == Qt.Key.Key_End:
            self.go(len(self.paths) - 1)
        elif key == Qt.Key.Key_Up:
            self.promote() if self.compare else self.jump_burst(-1)
        elif key == Qt.Key.Key_Down:
            if not self.compare:
                self.jump_burst(1)
        elif key == Qt.Key.Key_Tab or key == Qt.Key.Key_Backtab:
            if self.compare:
                self.active_right = not self.active_right
                self.update_header()
        elif key in (Qt.Key.Key_Escape,) or text == 'G':
            self.close()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) or text == 'D':
            self.open_in_develop()
        elif text in '012345' and text:
            self.rate(rating=int(text))
        elif text in '6789' and text:
            self.rate(label=catalog.LABELS['6789'.index(text)])
        elif text == 'P':
            self.rate(flag='pick')
        elif text == 'X':
            self.rate(flag='reject')
        elif text == 'U':
            self.rate(flag='')
        elif text == 'Z' or key == Qt.Key.Key_Space:
            pane = self.right if self.compare and self.active_right else self.left
            pane.toggle_zoom()
        elif text == 'C':
            self.set_compare(not self.compare)
        elif text == 'A':
            self.set_advance(not self.advance)
        elif text == 'F':
            self.show_faces = not self.show_faces
            self.left.update()
            self.right.update()
        else:
            super().keyPressEvent(event)

    def open_in_develop(self):
        path = self.target_row()['path']
        self.close()
        from pathlib import Path
        if Path(path).is_file():
            self.window.add_documents([path])
            self.window.set_mode('develop')
            if path != self.window.source_path:
                self.window.open_path(path)

    def closeEvent(self, event):
        self.loader.shutdown()
        path = self.target_row()['path'] if self.paths else ''
        super().closeEvent(event)
        self.closed.emit(path)
        self.deleteLater()
