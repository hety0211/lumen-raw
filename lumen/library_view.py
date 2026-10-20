"""Library mode (2.0.0): the catalog as a grid, with folders, smart albums, keywords and an
information panel.

The window has two pages (G / D): the library and the develop view.  The library shows a source
(all photos, the last import, the open 选片集, a smart album, a folder with its sub-folders, or a
keyword), filtered by the quick filter and the search box (``catalog.parse_query``), sorted by
capture time, name, rating or import.  Double-click or Enter opens photos in the develop view
(they join the 选片集); C or Space starts culling (``cull_view``).  Ratings, flags and labels
work on the grid's selection with the usual keys.  Thumbnails, capture data and culling hints are
produced in the background (``library_tasks``) and appear as they arrive.
"""
from __future__ import annotations

import collections
import json
import logging
import os
import threading
from pathlib import Path

from PySide6.QtCore import (QAbstractListModel, QItemSelection, QItemSelectionModel, QModelIndex, QObject, QPointF,
                            QRectF, QSize, QStringListModel, Qt, QTimer, QUrl, Signal)
from PySide6.QtGui import QAction, QColor, QDesktopServices, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QPixmapCache
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QBoxLayout, QComboBox, QCompleter, QDialog,
                               QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
                               QListView, QMenu,
                               QMessageBox, QPushButton, QSlider, QSplitter, QStyle, QStyledItemDelegate, QToolButton,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QButtonGroup, QCheckBox)

from . import catalog, host, i18n, library_db, preview_cache, theme
from .i18n import tr, N_
from .rating import QUICK_FILTERS, RatingBar, flag_path, star_path
from .scheduler import Activity as A
from .widgets import FlowLayout

log = logging.getLogger(__name__)

ROW_ROLE = Qt.ItemDataRole.UserRole + 20
PATH_ROLE = Qt.ItemDataRole.UserRole
SOURCE_ROLE = Qt.ItemDataRole.UserRole + 21
#: Sort orders: (key, title).
SORTS = (('captured', N_('拍摄时间')), ('name', N_('文件名')), ('rating', N_('星级')), ('added', N_('导入时间')))
#: Extra quick filters of the library (after the filmstrip's).
LIBRARY_FILTERS = ((N_('可能模糊'), 'blurry'), (N_('可能闭眼'), 'eyes:closed'), (N_('连拍中最佳'), 'burst:best'),
                   (N_('有人脸'), 'faces'))
#: Grid thumbnail sizes (pixels, long edge).
THUMB_MIN, THUMB_MAX, THUMB_DEFAULT = 96, 320, 176
#: Thumbnails kept decoded in memory.
THUMB_MEMORY = 900
SETTING_SIZE = 'library_thumbnail'
SETTING_ANALYSIS = 'library_analysis'
#: Query fields that depend on capture data or culling hints (a change to them re-filters).
DERIVED = {'blurry', 'eyes', 'faces', 'sharpness', 'burst', 'iso', 'f', 'focal', 'date', 'camera', 'lens', 'keyword', 'name'}


def exposure_text(row):
    parts = []
    if row.get('aperture'):
        parts.append(f'f/{row["aperture"]:g}')
    shutter = row.get('shutter')
    if shutter:
        parts.append(f'1/{round(1 / shutter)} s' if shutter < 1 and abs(1 / shutter - round(1 / shutter)) < .02 / shutter
                     else f'{shutter:.3g} s')
    if row.get('iso'):
        parts.append(f'ISO {row["iso"]:g}')
    if row.get('focal'):
        parts.append(f'{row["focal"]:g} mm')
    return '  ·  '.join(parts)


def size_text(size):
    if not size:
        return ''
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size < 1024 or unit == 'GB':
            return f'{size:.0f} {unit}' if unit == 'B' else f'{size:.1f} {unit}'
        size /= 1024


def camera_text(row):
    make, model = (row.get('make') or '').strip(), (row.get('model') or '').strip()
    if make and model.lower().startswith(make.split()[0].lower()):
        return model
    return f'{make} {model}'.strip()


# --------------------------------------------------------------------------- model and delegate

class LibraryModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []
        self.positions = {}

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self.rows):
            return None
        row = self.rows[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return row['name']
        if role == ROW_ROLE:
            return row
        if role == PATH_ROLE:
            return row['path']
        if role == Qt.ItemDataRole.ToolTipRole:
            lines = [row['path']]
            if row.get('captured'):
                lines.append(row['captured'])
            if camera_text(row):
                lines.append(camera_text(row))
            return '\n'.join(lines)
        return None

    def flags(self, index):
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def set_rows(self, rows):
        """Show ``rows``; the same photos in the same order only repaint (selection kept)."""
        if [r['path'] for r in rows] == [r['path'] for r in self.rows]:
            self.rows = rows
            if rows:
                self.dataChanged.emit(self.index(0), self.index(len(rows) - 1))
            return False
        self.beginResetModel()
        self.rows = rows
        self.positions = {r['path']: i for i, r in enumerate(rows)}
        self.endResetModel()
        return True

    def changed(self, path):
        position = self.positions.get(path)
        if position is not None:
            index = self.index(position)
            self.dataChanged.emit(index, index)

    def refresh_all(self):
        if self.rows:
            self.dataChanged.emit(self.index(0), self.index(len(self.rows) - 1))

    def row_of(self, path):
        position = self.positions.get(path)
        return self.rows[position] if position is not None else None


def paint_badges(painter, area, row, compact=False):
    """Flag, colour label, stars, burst and hint marks over a thumbnail rectangle."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    flag, label, rating = row.get('flag') or '', row.get('label') or '', int(row.get('rating') or 0)
    if flag == 'reject':
        painter.fillRect(area, QColor(10, 12, 10, 150))
        painter.setPen(QPen(QColor('#f0a0a0'), 2))
        c = QPointF(area.right() - 10, area.top() + 10)
        painter.drawLine(c + QPointF(-4, -4), c + QPointF(4, 4))
        painter.drawLine(c + QPointF(-4, 4), c + QPointF(4, -4))
    elif flag == 'pick':
        painter.setPen(QPen(QColor(0, 0, 0, 160), 3))
        painter.drawPath(flag_path(area.left() + 7, area.top() + 5, 11))
        painter.setPen(QPen(QColor('#f4f7ee'), 1.6))
        painter.drawPath(flag_path(area.left() + 7, area.top() + 5, 11))
    if label in catalog.LABEL_COLORS:
        painter.setPen(QPen(QColor(0, 0, 0, 150), 1))
        painter.setBrush(QColor(catalog.LABEL_COLORS[label]))
        center = QPointF(area.right() - 9, area.bottom() - 9) if flag == 'reject' else QPointF(area.right() - 9, area.top() + 9)
        painter.drawEllipse(center, 4.5, 4.5)
    if rating:
        strip = QRectF(area.left() + 2, area.bottom() - 15, 8 + rating * 11, 13)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 140))
        painter.drawRoundedRect(strip, 3, 3)
        painter.setBrush(QColor('#f2d36b'))
        for i in range(rating):
            painter.drawPath(star_path(strip.left() + 9 + i * 11, strip.center().y() + .5, 4.6))
    marks = []
    if row.get('missing'):
        marks.append(('!', QColor('#e0585b')))
    if row.get('blurry'):
        marks.append(('blur', QColor('#e6c84f')))
    if row.get('eyes_closed'):
        marks.append(('eyes', QColor('#e6c84f')))
    x = area.right() - 4
    for kind, color in marks:
        box = QRectF(x - 17, area.bottom() - 17, 15, 15)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.drawRoundedRect(box, 4, 4)
        painter.setPen(QPen(color, 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if kind == '!':
            painter.drawLine(QPointF(box.center().x(), box.top() + 3), QPointF(box.center().x(), box.bottom() - 6))
            painter.drawPoint(QPointF(box.center().x(), box.bottom() - 3))
        elif kind == 'blur':
            for dy in (-3, 0, 3):
                path = QPainterPath(QPointF(box.left() + 3, box.center().y() + dy))
                path.cubicTo(QPointF(box.left() + 6, box.center().y() + dy - 2), QPointF(box.left() + 9, box.center().y() + dy + 2),
                             QPointF(box.right() - 3, box.center().y() + dy))
                painter.drawPath(path)
        else:
            path = QPainterPath(QPointF(box.left() + 3, box.center().y()))
            path.quadTo(QPointF(box.center().x(), box.center().y() + 5), QPointF(box.right() - 3, box.center().y()))
            painter.drawPath(path)
            for dx in (-3, 0, 3):
                painter.drawLine(QPointF(box.center().x() + dx, box.center().y() + 2.5),
                                 QPointF(box.center().x() + dx * 1.3, box.center().y() + 5))
        x -= 18
    if row.get('burst') and not compact:
        text = f'{row["burst_index"]}/{row["burst_size"]}'
        if row.get('collapsed'):
            text = f'×{row["burst_size"]}'
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + 12
        box = QRectF(area.left() + (24 if flag == 'pick' else 4), area.top() + 4, width, 16)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(theme.color('accent') if row.get('best') else QColor(0, 0, 0, 160))
        painter.drawRoundedRect(box, 8, 8)
        painter.setPen(theme.color('accent_text') if row.get('best') else QColor('#f4f1ea'))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
    painter.restore()


class GridDelegate(QStyledItemDelegate):
    """A thumbnail cell: image, badges and the file name."""

    def __init__(self, images, view, edge=THUMB_DEFAULT, names=True, marks=None):
        super().__init__(view)
        self.images, self.edge, self.names, self.marks = images, edge, names, marks
        self.stretched = None              # the cell the grid widened to fill its width

    def cell(self):
        return QSize(self.edge + 16, round(self.edge * .74) + (36 if self.names else 14))

    def sizeHint(self, option, index):
        return self.stretched or self.cell()

    def paint(self, painter, option, index):
        row = index.data(ROW_ROLE)
        if row is None:
            return
        rect = QRectF(option.rect).adjusted(3, 3, -3, -3)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(theme.color('accent'), 2) if selected else QPen(theme.color('border'), 1))
        painter.setBrush(theme.color('selection') if selected else theme.color('card'))
        painter.drawRoundedRect(rect, 7, 7)
        area = rect.adjusted(6, 6, -6, -(28 if self.names else 6))
        pixmap = self.images.pixmap(row['path'], area.size().toSize())
        if pixmap is not None:
            size = pixmap.deviceIndependentSize()
            target = QRectF(area.center().x() - size.width() / 2, area.center().y() - size.height() / 2,
                            size.width(), size.height())
            painter.drawPixmap(target.toRect(), pixmap)
            image = target
        else:
            image = area
            painter.fillRect(area, theme.color('plot'))
            painter.setPen(theme.color('dim'))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, (row.get('ext') or '').lstrip('.').upper())
        painter.restore()
        paint_badges(painter, image, row, compact=not self.names)
        if self.marks is not None:
            mark = self.marks(row['path'])
            if mark:
                painter.save()
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(theme.color('accent'))
                painter.drawEllipse(QPointF(rect.left() + 9, rect.bottom() - 13), 3.2, 3.2)
                painter.restore()
        if self.names:
            painter.save()
            painter.setPen(theme.color('head' if selected else 'dim'))
            text_rect = QRectF(rect.left() + (16 if self.marks else 8), rect.bottom() - 23, rect.width() - 24, 18)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             painter.fontMetrics().elidedText(row['name'] or '', Qt.TextElideMode.ElideMiddle, int(text_rect.width())))
            painter.restore()


class ThumbnailImages(QObject):
    """Decoded thumbnails by path (LRU) and their scaled pixmaps; asks the worker for missing ones."""
    updated = Signal(str)

    def __init__(self, worker, parent=None):
        super().__init__(parent)
        self.worker = worker
        self.images = collections.OrderedDict()
        self.requested = set()
        worker.thumbnail_ready.connect(self.ready)
        QPixmapCache.setCacheLimit(max(QPixmapCache.cacheLimit(), 160 * 1024))

    def image(self, path):
        image = self.images.get(path)
        if image is not None:
            self.images.move_to_end(path)
            return image
        if path not in self.requested:
            self.requested.add(path)
            self.worker.want_thumbnail(path)
        return None

    def pixmap(self, path, size):
        key = f'{path}|{size.width()}x{size.height()}'
        found = QPixmapCache.find(key)
        if found is not None and not found.isNull():
            return found
        image = self.image(path)
        if image is None:
            return None
        pixmap = QPixmap.fromImage(image.scaled(size, Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation))
        QPixmapCache.insert(key, pixmap)
        return pixmap

    def ready(self, path, image):
        self.requested.discard(path)
        self.images[path] = image
        self.images.move_to_end(path)
        while len(self.images) > THUMB_MEMORY:
            old, _ = self.images.popitem(last=False)
        self.updated.emit(path)

    def forget(self, paths=None):
        """Drop thumbnails (all, or of ``paths``) so they are fetched again."""
        for path in list(self.images) if paths is None else paths:
            self.images.pop(path, None)
            self.requested.discard(path)
        QPixmapCache.clear()


class ButtonPair(QWidget):
    """Buttons side by side, or one above the other when their labels do not fit (longer
    translations, a narrow panel)."""

    def __init__(self, buttons, parent=None):
        super().__init__(parent)
        self.buttons = buttons
        self.box = QBoxLayout(QBoxLayout.Direction.LeftToRight, self)
        self.box.setContentsMargins(0, 0, 0, 0)
        self.box.setSizeConstraint(QBoxLayout.SizeConstraint.SetNoConstraint)
        for button in buttons:
            self.box.addWidget(button)

    def minimumSizeHint(self):
        return QSize(max(b.minimumSizeHint().width() for b in self.buttons), super().minimumSizeHint().height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        needed = sum(b.sizeHint().width() for b in self.buttons) + self.box.spacing() * (len(self.buttons) - 1)
        direction = (QBoxLayout.Direction.LeftToRight if needed <= event.size().width()
                     else QBoxLayout.Direction.TopToBottom)
        if self.box.direction() != direction:
            self.box.setDirection(direction)


class GridView(QListView):
    opened = Signal()
    cull = Signal()
    dropped = Signal(list)
    zoomed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('libraryGrid')
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMovement(QListView.Movement.Static)
        self.setUniformItemSizes(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setLayoutMode(QListView.LayoutMode.Batched)
        self.setBatchSize(400)
        self.setSpacing(0)
        self.setWrapping(True)
        self.setAcceptDrops(True)
        self.setDragEnabled(False)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.verticalScrollBar().setSingleStep(24)

    def layout_width(self):
        """The width QListView wraps items at: it keeps room for the vertical scroll bar even
        while that is hidden, so cells sized to the bare viewport wrap one column early."""
        style, bar = self.style(), self.verticalScrollBar()
        width = self.maximumViewportSize().width()
        if (self.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
                and not style.pixelMetric(QStyle.PixelMetric.PM_ScrollView_ScrollBarOverlap, None, bar)):
            width -= style.pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent, None, bar)
            if style.styleHint(QStyle.StyleHint.SH_ScrollView_FrameOnlyAroundContents, None, self):
                width -= style.pixelMetric(QStyle.PixelMetric.PM_DefaultFrameWidth) * 2
        return width

    def fit_cells(self, cell):
        """Cells as wide as the view divides evenly, so the grid fills it edge to edge."""
        self.cell = cell
        width = max(1, self.layout_width() - 1)
        columns = max(1, width // max(1, cell.width()))
        size = QSize(width // columns, cell.height())
        delegate = self.itemDelegate()
        if isinstance(delegate, GridDelegate):
            delegate.stretched = size
        if size != self.gridSize():
            self.setGridSize(size)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, 'cell', None) is not None:
            self.fit_cells(self.cell)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not event.modifiers():
            self.opened.emit()
            return
        if event.key() == Qt.Key.Key_Space and not event.modifiers():
            self.cull.emit()
            return
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.indexAt(event.position().toPoint()).isValid():
            self.opened.emit()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoomed.emit(1 if event.angleDelta().y() > 0 else -1)
            return
        super().wheelEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            event.acceptProposedAction()
            self.dropped.emit(paths)
        else:
            super().dropEvent(event)


# --------------------------------------------------------------------------- side panels

class SourceTree(QTreeWidget):
    chosen = Signal(tuple)
    menu_requested = Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('sourceTree')
        self.setHeaderHidden(True)
        self.setIndentation(14)
        self.setUniformRowHeights(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda point: self.menu_requested.emit(self.itemAt(point), self.viewport().mapToGlobal(point)))
        self.itemClicked.connect(self._clicked)
        self.currentItemChanged.connect(lambda current, _: self._clicked(current))
        self.sections = {}

    def _clicked(self, item, _column=0):
        if item is None:
            return
        source = item.data(0, SOURCE_ROLE)
        if source:
            self.chosen.emit(tuple(source))

    def section(self, key, title):
        item = self.sections.get(key)
        if item is None:
            item = QTreeWidgetItem([title])
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
            self.addTopLevelItem(item)
            item.setExpanded(True)
            self.sections[key] = item
        return item

    @staticmethod
    def entry(parent, title, source, tip=''):
        item = QTreeWidgetItem([title])
        item.setData(0, SOURCE_ROLE, list(source))
        if tip:
            item.setToolTip(0, tip)
        parent.addChild(item)
        return item

    def items(self):
        found, stack = [], [self.topLevelItem(i) for i in range(self.topLevelItemCount())]
        while stack:
            item = stack.pop()
            found.append(item)
            stack.extend(item.child(i) for i in range(item.childCount()))
        return found

    def find(self, source):
        wanted = list(source)
        return next((item for item in self.items() if item.data(0, SOURCE_ROLE) == wanted), None)


class KeywordChips(QWidget):
    removed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.flow = FlowLayout(self)

    def set_keywords(self, counts, total):
        while self.flow.count():
            item = self.flow.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        for name, count in counts:
            button = QToolButton()
            button.setObjectName('keywordChip')
            partial = count < total
            button.setText(f'{name}  ×' if not partial else f'{name} ({count}/{total})  ×')
            button.setToolTip(tr('从所选照片移除关键词“{name}”', name=name))
            button.clicked.connect(lambda checked=False, n=name: self.removed.emit(n))
            self.flow.addWidget(button)
        self.updateGeometry()


class InfoPanel(QWidget):
    open_requested = Signal()
    cull_requested = Signal()
    keywords_entered = Signal(list)
    keyword_removed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('library')
        self.setMinimumWidth(250)
        self.setMaximumWidth(340)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 14, 12, 12)
        layout.setSpacing(7)
        heading = QLabel(tr('信息'))
        heading.setObjectName('section')
        layout.addWidget(heading)
        self.preview = QLabel()
        self.preview.setObjectName('preview')
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setFixedHeight(190)
        layout.addWidget(self.preview)
        self.title = QLabel()
        self.title.setWordWrap(True)
        self.title.setObjectName('infoTitle')
        layout.addWidget(self.title)
        self.details = QLabel()
        self.details.setObjectName('subtle')
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.details)
        self.hints = QLabel()
        self.hints.setWordWrap(True)
        self.hints.setObjectName('hints')
        layout.addWidget(self.hints)
        self.rating_bar = RatingBar()
        layout.addWidget(self.rating_bar)
        keywords = QLabel(tr('关键词'))
        keywords.setObjectName('section')
        layout.addWidget(keywords)
        self.chips = KeywordChips()
        self.chips.removed.connect(self.keyword_removed)
        layout.addWidget(self.chips)
        self.keyword_input = QLineEdit()
        self.keyword_input.setPlaceholderText(tr('添加关键词，逗号分隔，回车确认'))
        self.keyword_input.returnPressed.connect(self._entered)
        self.completions = QStringListModel(self)
        self.completer = QCompleter(self.completions, self)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.keyword_input.setCompleter(self.completer)
        layout.addWidget(self.keyword_input)
        layout.addStretch(1)
        row = QHBoxLayout()
        self.open_button = QPushButton(tr('在显影中打开'))
        self.open_button.setObjectName('primary')
        self.open_button.clicked.connect(self.open_requested)
        self.cull_button = QPushButton(tr('选片模式'))
        self.cull_button.clicked.connect(self.cull_requested)
        row.addWidget(self.open_button)
        row.addWidget(self.cull_button)
        layout.addLayout(row)

    def _entered(self):
        words = library_db.clean_keywords(self.keyword_input.text())
        if words:
            self.keywords_entered.emit(words)
        self.keyword_input.clear()

    def set_completions(self, names):
        self.completions.setStringList(names)

    def show_rows(self, rows, image=None):
        enabled = bool(rows)
        for widget in (self.rating_bar, self.keyword_input, self.open_button, self.cull_button):
            widget.setEnabled(enabled)
        if not rows:
            self.preview.clear()
            self.preview.setText(tr('未选择照片'))
            self.title.clear()
            self.details.clear()
            self.hints.clear()
            self.chips.set_keywords([], 0)
            return
        row = rows[0]
        if image is not None:
            self.preview.setPixmap(QPixmap.fromImage(image).scaled(self.preview.width() - 4, self.preview.height() - 4,
                                                                   Qt.AspectRatioMode.KeepAspectRatio,
                                                                   Qt.TransformationMode.SmoothTransformation))
        else:
            self.preview.clear()
            self.preview.setText((row.get('ext') or '').lstrip('.').upper())
        if len(rows) > 1:
            self.title.setText(tr('已选 {n} 张照片', n=len(rows)))
            self.details.setText(tr('当前：{name}', name=row['name']))
        else:
            self.title.setText(row['name'])
            lines = []
            if row.get('captured'):
                lines.append(row['captured'])
            if camera_text(row):
                lines.append(camera_text(row))
            if row.get('lens'):
                lines.append(row['lens'])
            if exposure_text(row):
                lines.append(exposure_text(row))
            size = []
            if row.get('width') and row.get('height'):
                size.append(f'{row["width"]} × {row["height"]}')
            if row.get('size'):
                size.append(size_text(row['size']))
            if size:
                lines.append('  ·  '.join(size))
            lines.append(str(Path(row['path']).parent))
            if row.get('missing'):
                lines.append(tr('文件已移动或删除'))
            self.details.setText('\n'.join(lines))
        self.hints.setText(hint_text(row))
        counts = collections.Counter(k for r in rows for k in (r.get('keywords') or ()))
        self.chips.set_keywords(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower())), len(rows))


def hint_text(row):
    """The culling hints of a photo as one line (or an explanation when not analysed yet)."""
    if (row.get('analysis') or 0) < library_db.ANALYSIS:
        return tr('清晰度与闭眼提示：分析中…') if not row.get('missing') else ''
    if row.get('sharpness') is None:
        return ''
    parts = [tr('清晰度 {value:.0f}', value=row['sharpness']) + (tr('（可能模糊）') if row.get('blurry') else '')]
    if row.get('faces'):
        parts.append(tr('{n} 张人脸', n=row['faces']) + (tr(' · {n} 张可能闭眼', n=row['eyes_closed']) if row.get('eyes_closed') else ''))
    if row.get('burst'):
        parts.append(tr('连拍 {index}/{size}', index=row['burst_index'], size=row['burst_size']) +
                     (tr(' · 本组最佳') if row.get('best') else ''))
    return '  ·  '.join(parts)


class SmartAlbumDialog(QDialog):
    def __init__(self, window, name='', query=''):
        super().__init__(window)
        self.window = window
        self.setWindowTitle(tr('智能相册'))
        self.setMinimumWidth(520)
        form = QFormLayout(self)
        self.name = QLineEdit(name)
        self.query = QLineEdit(query)
        self.query.setPlaceholderText('rating>=3 -flag:reject keyword:海边 camera:a7 date:2026-10')
        form.addRow(tr('名称'), self.name)
        form.addRow(tr('条件'), self.query)
        help_text = QLabel(tr('条件与筛选框相同，多个条件同时满足，前加 - 表示排除。字段：rating、flag、label、keyword（any / none / 名称）、'
                              'camera、lens、iso、f（光圈）、focal、date（如 2026-10 或 date>=2026-10-01）、faces、eyes:closed、'
                              'sharpness（0–100）、burst、name、ext、folder；词语：blurry、burst、best、missing。'))
        help_text.setWordWrap(True)
        help_text.setObjectName('subtle')
        form.addRow(help_text)
        self.count = QLabel()
        self.count.setObjectName('subtle')
        form.addRow(self.count)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        form.addRow(buttons)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.update_count)
        self.query.textChanged.connect(self.timer.start)
        self.name.textChanged.connect(lambda: self.ok.setEnabled(self.valid))
        self.valid = False
        self.update_count()

    def update_count(self):
        try:
            rows = self.window.library_db.search(self.window.catalog, self.query.text())
            self.count.setText(tr('目前有 {n} 张照片符合', n=len(rows)))
            self.query.setProperty('invalid', False)
            self.valid = bool(self.query.text().strip())
        except ValueError as exc:
            self.count.setText(tr('条件无效：') + str(exc))
            self.query.setProperty('invalid', True)
            self.valid = False
        self.query.style().unpolish(self.query)
        self.query.style().polish(self.query)
        self.ok.setEnabled(self.valid and bool(self.name.text().strip()))

    def values(self):
        return self.name.text().strip(), self.query.text().strip()


class _Relay(QObject):
    library = Signal(str, list)
    imported = Signal(object)


# --------------------------------------------------------------------------- the window's library mode

class LibraryModeMixin:
    def init_library_mode(self):
        self.library_db = library_db.LibraryDB()
        self.previews = preview_cache.default()
        self.previews.paused = lambda: self.work.busy(A.AI, A.EXPORTING)
        from .library_tasks import LibraryWorker
        self.library_worker = LibraryWorker(self.library_db, self.previews, self.catalog,
                                            paused=lambda: self.work.busy(A.AI, A.EXPORTING), parent=self)
        self.thumb_images = ThumbnailImages(self.library_worker, self)
        self.library_relay = _Relay(self)
        self.library_relay.library.connect(self.library_changed)
        self.library_relay.imported.connect(self.import_finished)
        self.library_db.listeners.append(self.library_relay.library.emit)
        self.catalog.listeners.append(lambda paths: self.library_relay.library.emit('ratings', []))
        self.work.changed.connect(self.library_worker.kick)
        self.mode = 'develop'
        self.library_source = ('all',)
        self.library_rows = []
        self.library_dirty = True
        self.library_pending = set()
        self.library_refresh_timer = QTimer(self)
        self.library_refresh_timer.setSingleShot(True)
        self.library_refresh_timer.setInterval(350)
        self.library_refresh_timer.timeout.connect(lambda: self.refresh_library_grid())
        self.library_update_timer = QTimer(self)
        self.library_update_timer.setSingleShot(True)
        self.library_update_timer.setInterval(1200)
        self.library_update_timer.timeout.connect(self.apply_library_updates)
        self.source_timer = QTimer(self)
        self.source_timer.setSingleShot(True)
        self.source_timer.setInterval(500)
        self.source_timer.timeout.connect(self.refresh_sources)
        self.info_timer = QTimer(self)
        self.info_timer.setSingleShot(True)
        self.info_timer.setInterval(60)
        self.info_timer.timeout.connect(self.update_library_info)
        self.prefetch_timer = QTimer(self)
        self.prefetch_timer.setSingleShot(True)
        self.prefetch_timer.setInterval(700)
        self.prefetch_timer.timeout.connect(self.prefetch_current)
        self.library_indexed = False
        self.library_worker.analysis_enabled = i18n.setting(SETTING_ANALYSIS, '1') != '0'
        self.cull_window = None
        try:
            size = int(i18n.setting(SETTING_SIZE, str(THUMB_DEFAULT)))
        except (TypeError, ValueError):
            size = THUMB_DEFAULT
        self.library_thumb_edge = max(THUMB_MIN, min(THUMB_MAX, size))

    # -- page

    def build_library_page(self):
        page = QSplitter(Qt.Orientation.Horizontal)
        page.setObjectName('libraryPage')
        left = QWidget()
        left.setObjectName('library')
        left.setMinimumWidth(205)
        left.setMaximumWidth(300)
        column = QVBoxLayout(left)
        column.setContentsMargins(12, 14, 8, 10)
        column.setSpacing(8)
        heading = QLabel(tr('图库'))
        heading.setObjectName('section')
        column.addWidget(heading)
        self.source_tree = SourceTree()
        self.source_tree.chosen.connect(self.choose_library_source)
        self.source_tree.menu_requested.connect(self.source_menu)
        column.addWidget(self.source_tree, 1)
        column.addWidget(ButtonPair([self.button(tr('导入文件夹…'), self.import_folder_dialog, True),
                                     self.button(tr('＋ 智能相册'), self.new_smart_album)]))
        page.addWidget(left)
        center = QWidget()
        middle = QVBoxLayout(center)
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(14, 8, 14, 8)
        self.library_title = QLabel(tr('全部照片'))
        self.library_title.setObjectName('section')
        bar.addWidget(self.library_title)
        self.library_summary = QLabel()
        self.library_summary.setObjectName('subtle')
        bar.addWidget(self.library_summary, 1)
        self.library_filter = QComboBox()
        for title, _ in QUICK_FILTERS + LIBRARY_FILTERS:
            self.library_filter.addItem(tr(title))
        self.library_filter.setToolTip(tr('筛选图库'))
        self.library_filter.currentIndexChanged.connect(lambda _: self.refresh_library_grid())
        bar.addWidget(self.library_filter)
        self.library_search = QLineEdit()
        self.library_search.setPlaceholderText(tr('搜索：文件名、关键词、相机 或 rating>=3 keyword:海边'))
        self.library_search.setToolTip(tr('与智能相册的条件相同；直接输入文字时同时查找文件名、关键词、相机与镜头。'))
        self.library_search.setClearButtonEnabled(True)
        self.library_search.setFixedWidth(250)
        self.library_search.textChanged.connect(lambda _: self.library_refresh_timer.start())
        bar.addWidget(self.library_search)
        self.library_sort = QComboBox()
        for _, title in SORTS:
            self.library_sort.addItem(tr(title))
        self.library_sort.setToolTip(tr('排序'))
        self.library_sort.currentIndexChanged.connect(lambda _: self.refresh_library_grid())
        bar.addWidget(self.library_sort)
        self.collapse_bursts = QCheckBox(tr('折叠连拍'))
        self.collapse_bursts.setToolTip(tr('每组连拍只显示最佳的一张（睁眼且最清晰）'))
        self.collapse_bursts.toggled.connect(lambda _: self.refresh_library_grid())
        bar.addWidget(self.collapse_bursts)
        self.thumb_slider = QSlider(Qt.Orientation.Horizontal)
        self.thumb_slider.setRange(THUMB_MIN, THUMB_MAX)
        self.thumb_slider.setValue(self.library_thumb_edge)
        self.thumb_slider.setFixedWidth(110)
        self.thumb_slider.setToolTip(tr('缩略图大小（{key} + 滚轮）', key=host.COMMAND))
        self.thumb_slider.valueChanged.connect(self.set_thumbnail_size)
        bar.addWidget(self.thumb_slider)
        middle.addLayout(bar)
        self.library_model = LibraryModel(self)
        self.grid = GridView()
        self.grid.setModel(self.library_model)
        self.grid_delegate = GridDelegate(self.thumb_images, self.grid, self.library_thumb_edge,
                                          marks=lambda path: path in self.documents)
        self.grid.setItemDelegate(self.grid_delegate)
        self.grid.fit_cells(self.grid_delegate.cell())
        self.grid.selectionModel().selectionChanged.connect(lambda *_: self.library_selection_changed())
        self.grid.selectionModel().currentChanged.connect(lambda *_: self.library_selection_changed())
        self.grid.opened.connect(self.open_library_selection)
        self.grid.cull.connect(self.start_culling)
        self.grid.dropped.connect(self.import_dropped)
        self.grid.zoomed.connect(lambda step: self.thumb_slider.setValue(self.thumb_slider.value() + step * 24))
        self.grid.customContextMenuRequested.connect(self.grid_menu)
        self.thumb_images.updated.connect(self.thumbnail_arrived)
        middle.addWidget(self.grid, 1)
        self.library_status = QLabel()
        self.library_status.setObjectName('subtle')
        self.library_status.setContentsMargins(14, 4, 14, 6)
        middle.addWidget(self.library_status)
        page.addWidget(center)
        self.info_panel = InfoPanel()
        self.info_panel.open_requested.connect(self.open_library_selection)
        self.info_panel.cull_requested.connect(self.start_culling)
        self.info_panel.keywords_entered.connect(self.add_keywords_to_selection)
        self.info_panel.keyword_removed.connect(self.remove_keyword_from_selection)
        self.info_panel.rating_bar.rated.connect(lambda v: self.rate_selected(rating=v))
        self.info_panel.rating_bar.flagged.connect(lambda v: self.rate_selected(flag=v))
        self.info_panel.rating_bar.labelled.connect(lambda v: self.rate_selected(label=v))
        page.addWidget(self.info_panel)
        page.setStretchFactor(1, 1)
        page.setSizes([230, 1000, 290])
        self.library_worker.progress.connect(self.analysis_progress)
        self.refresh_sources()
        return page

    def build_mode_switch(self, bar):
        group = QButtonGroup(self)
        group.setExclusive(True)
        self.mode_buttons = {}
        for mode, title, key in (('library', tr('图库'), 'G'), ('develop', tr('显影'), 'D')):
            button = QPushButton(title)
            button.setCheckable(True)
            button.setObjectName('modeButton')
            button.setToolTip(tr('{title}（{key}）', title=title, key=key))
            button.clicked.connect(lambda checked=False, m=mode: self.set_mode(m))
            group.addButton(button)
            bar.addWidget(button)
            self.mode_buttons[mode] = button
        cull = QPushButton(tr('选片'))
        cull.setObjectName('modeButton')
        cull.setToolTip(tr('全屏选片：并排对比、连拍分组、清晰度与闭眼提示（C）'))
        cull.clicked.connect(self.start_culling)
        bar.addWidget(cull)
        self.cull_button = cull
        self.mode_buttons['develop'].setChecked(True)

    def library_shortcuts(self):
        for key, callback in (('G', lambda: self.set_mode('library')), ('D', lambda: self.set_mode('develop')),
                              ('C', self.start_culling), ('Ctrl+Shift+I', self.import_folder_dialog)):
            action = QAction(self)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(callback)
            self.addAction(action)

    def set_mode(self, mode):
        if mode == self.mode:
            return
        self.mode = mode
        self.pages.setCurrentIndex(0 if mode == 'library' else 1)
        self.mode_buttons[mode].setChecked(True)
        if mode == 'library':
            if self.library_dirty or not self.library_model.rows:
                self.refresh_library_grid()
            self.grid.setFocus()
            self.select_library_path(self.source_path)
        self.update_rating_bar()

    # -- sources

    def refresh_sources(self):
        tree = self.source_tree
        current = list(self.library_source)
        expanded = {tuple(item.data(0, SOURCE_ROLE) or ()) for item in tree.items() if item.isExpanded()}
        tree.blockSignals(True)
        tree.clear()
        tree.sections = {}
        db = self.library_db
        catalog_section = tree.section('catalog', tr('目录'))
        tree.entry(catalog_section, tr('全部照片  {n}', n=db.count()), ('all',))
        tree.entry(catalog_section, tr('最近导入'), ('recent',))
        tree.entry(catalog_section, tr('当前选片集  {n}', n=len(self.documents)), ('album',))
        smart = tree.section('smart', tr('智能相册'))
        for key, title, query in library_db.BUILTIN:
            tree.entry(smart, tr(title), ('smart', query), query)
        for collection in db.collections():
            tree.entry(smart, collection['name'], ('collection', collection['id']), collection['query'])
        folders = tree.section('folders', tr('文件夹'))
        rows = sorted(db.folders(), key=lambda r: len(r['key']))
        parents, totals = {}, collections.Counter()
        known = {r['key'].rstrip('\\/') for r in rows}
        for row in rows:
            name = row['key'].rstrip('\\/')
            ancestor, parent = os.path.dirname(name), None
            while ancestor and ancestor != os.path.dirname(ancestor):
                if ancestor in known:
                    parent = ancestor
                    break
                ancestor = os.path.dirname(ancestor)
            if parent is None and ancestor in known and ancestor != name:
                parent = ancestor
            parents[name] = parent
            step = name
            while step is not None:
                totals[step] += row['count']
                step = parents.get(step)
        items = {}
        for row in rows:
            name = row['key'].rstrip('\\/')
            parent = items.get(parents[name], folders)
            label = row['path'] if parent is folders else (Path(row['path']).name or row['path'])
            count = totals[name]
            items[name] = tree.entry(parent, f'{label}  {count}' if count else label, ('folder', row['id']), row['path'])
        for item in items.values():
            item.setExpanded(tuple(item.data(0, SOURCE_ROLE)) in expanded or item.parent() is folders)
        keywords = tree.section('keywords', tr('关键词'))
        names = []
        for keyword in db.keywords():
            tree.entry(keywords, f'{keyword["name"]}  {keyword["count"]}', ('keyword', keyword['id']))
            names.append(keyword['name'])
        self.info_panel.set_completions(names)
        found = tree.find(current)
        if found is None:
            self.library_source = ('all',)
            found = tree.find(['all'])
        tree.setCurrentItem(found)
        self.library_title.setText(found.text(0).split('  ')[0])
        tree.blockSignals(False)

    def choose_library_source(self, source):
        if tuple(source) == tuple(self.library_source):
            return
        self.library_source = tuple(source)
        item = self.source_tree.currentItem()
        self.library_title.setText(item.text(0).split('  ')[0] if item else tr('全部照片'))
        self.refresh_library_grid(scroll_top=True)

    def source_rows_spec(self):
        source = self.library_source
        kind = source[0]
        if kind == 'album':
            return ('paths', list(self.documents)), ''
        if kind == 'smart':
            return ('all',), source[1]
        if kind == 'collection':
            collection = self.library_db.collection(source[1])
            return ('all',), collection['query'] if collection else ''
        if kind == 'folder':
            return ('folder', source[1], True), ''
        return source, ''

    def library_query(self):
        spec, base = self.source_rows_spec()
        filters = QUICK_FILTERS + LIBRARY_FILTERS
        quick = filters[self.library_filter.currentIndex()][1] if self.library_filter.currentIndex() >= 0 else ''
        return spec, ' '.join(q for q in (base, quick, self.library_search.text()) if q)

    # -- the grid

    def refresh_library_grid(self, scroll_top=False):
        if not hasattr(self, 'grid'):
            return
        self.library_refresh_timer.stop()
        if self.mode != 'library' and self.cull_window is None:
            self.library_dirty = True
            return
        self.library_dirty = False
        spec, query = self.library_query()
        sort = SORTS[max(0, self.library_sort.currentIndex())][0]
        try:
            rows = self.library_db.search(self.catalog, query, spec, edited=self.document_edited, sort=sort)
            self.library_search.setProperty('invalid', False)
        except ValueError:
            self.library_search.setProperty('invalid', True)
            rows = None
        self.library_search.style().unpolish(self.library_search)
        self.library_search.style().polish(self.library_search)
        if rows is None:
            return
        if self.collapse_bursts.isChecked():
            rows = [r for r in rows if not r.get('burst') or r.get('best')]
            for r in rows:
                r['collapsed'] = bool(r.get('burst'))
        selected, current = self.library_selected_paths(), self.library_current_path()
        self.library_rows = rows
        reset = self.library_model.set_rows(rows)
        if reset:
            self.restore_library_selection(selected, current)
            if scroll_top:
                self.grid.scrollToTop()
            elif current in self.library_model.positions:
                self.grid.scrollTo(self.library_model.index(self.library_model.positions[current]))
        total = self.library_db.count()
        self.library_summary.setText(tr('  {n} 张', n=len(rows)) + (tr(' · 图库共 {total} 张', total=total) if total != len(rows) else ''))
        self.want_library_work(rows)
        self.update_library_info()

    def want_library_work(self, rows):
        unanalysed = [dict(id=r['id'], path=r['path']) for r in rows if r.get('analysis', 0) < library_db.ANALYSIS
                      and not r.get('missing')]
        if unanalysed:
            self.library_worker.want_analysis(unanalysed[:5000])
        if any(not r.get('indexed') for r in rows):
            self.library_worker.want_index()

    def restore_library_selection(self, paths, current):
        model = self.library_model
        selection = QItemSelection()
        for path in paths:
            position = model.positions.get(path)
            if position is not None:
                index = model.index(position)
                selection.select(index, index)
        flags = QItemSelectionModel.SelectionFlag.ClearAndSelect
        self.grid.selectionModel().select(selection, flags)
        position = model.positions.get(current)
        if position is not None:
            self.grid.selectionModel().setCurrentIndex(model.index(position), QItemSelectionModel.SelectionFlag.NoUpdate)

    def select_library_path(self, path):
        position = self.library_model.positions.get(path)
        if position is None:
            return
        index = self.library_model.index(position)
        if not self.grid.selectionModel().isSelected(index):
            self.grid.selectionModel().setCurrentIndex(index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        self.grid.scrollTo(index)

    def library_selected_paths(self):
        if not hasattr(self, 'grid'):
            return []
        rows = sorted(index.row() for index in self.grid.selectionModel().selectedIndexes())
        return [self.library_model.rows[r]['path'] for r in rows if r < len(self.library_model.rows)]

    def library_current_path(self):
        index = self.grid.currentIndex() if hasattr(self, 'grid') else QModelIndex()
        return index.data(PATH_ROLE) if index.isValid() else None

    def library_selection_changed(self):
        self.info_timer.start()
        self.update_rating_bar()
        if self.mode == 'library':
            self.prefetch_timer.start()

    def prefetch_current(self):
        """Decode the photo the grid rests on, so opening it in the develop view is immediate."""
        current = self.library_current_path()
        if current and self.mode == 'library' and current != self.source_path and Path(current).is_file():
            self.previews.prefetch([current])

    def update_library_info(self):
        if not hasattr(self, 'info_panel'):
            return
        paths = self.library_selected_paths()
        current = self.library_current_path()
        if current in paths:
            paths.remove(current)
            paths.insert(0, current)
        rows = [r for r in (self.library_model.row_of(p) for p in paths) if r is not None]
        image = self.thumb_images.image(rows[0]['path']) if rows else None
        self.info_panel.show_rows(rows, image)
        meta = rows[0] if rows else catalog.empty()
        self.info_panel.rating_bar.set_state(int(meta.get('rating') or 0), meta.get('flag') or '', meta.get('label') or '')

    def thumbnail_arrived(self, path):
        self.library_model.changed(path)
        rows = self.info_panel_paths() if hasattr(self, 'info_panel') else []
        if rows and rows[0] == path:
            self.info_timer.start()
        if hasattr(self, 'cull_window') and self.cull_window is not None:
            self.cull_window.thumbnail_arrived(path)

    def info_panel_paths(self):
        current = self.library_current_path()
        return [current] if current else []

    def set_thumbnail_size(self, value):
        value = max(THUMB_MIN, min(THUMB_MAX, int(value)))
        self.library_thumb_edge = value
        self.grid_delegate.edge = value
        self.grid.fit_cells(self.grid_delegate.cell())
        self.grid.doItemsLayout()
        try:
            i18n.save_setting(SETTING_SIZE, value)
        except OSError:
            pass

    # -- changes from the database, the worker and ratings

    def library_changed(self, kind, ids):
        if kind in ('indexed', 'analysed'):
            self.library_pending.update(ids)
            if kind == 'indexed':
                self.library_indexed = True
                self.source_timer.start()
            if not self.library_update_timer.isActive():
                self.library_update_timer.start()
            return
        if kind == 'ratings':
            for row in self.library_model.rows:
                row.update(self.catalog.get(row['path']))
            self.library_model.refresh_all()
            self.info_timer.start()
            spec, query = self.library_query()
            if any(field in ('rating', 'flag', 'label') for _, field, _, _ in self.safe_terms(query)):
                self.library_refresh_timer.start()
            if self.cull_window is not None:
                self.cull_window.ratings_changed()
            return
        self.source_timer.start()
        self.library_refresh_timer.start()

    @staticmethod
    def safe_terms(query):
        try:
            return catalog.parse_query(query)
        except ValueError:
            return []

    def apply_library_updates(self):
        """New capture data or hints for some photos, at most every 1.2 s while the worker is busy.
        Capture data may change the order, so the grid is searched again; new hints only update
        the rows shown, unless the filter or burst folding depends on them."""
        if not self.library_pending:
            return
        ids, indexed = self.library_pending, self.library_indexed
        self.library_pending, self.library_indexed = set(), False
        if self.cull_window is not None:
            self.cull_window.rows_changed(ids)
        if self.mode != 'library':
            self.library_dirty = True
            return
        fields = {field for _, field, _, _ in self.safe_terms(self.library_query()[1])}
        if indexed or self.collapse_bursts.isChecked() or fields & DERIVED:
            return self.refresh_library_grid()
        fresh = {r['id']: r for r in self.library_db.rows(('ids', list(ids)))}
        rows = self.library_model.rows
        for row in rows:
            update = fresh.get(row['id'])
            if update is not None:
                row.update({k: v for k, v in update.items() if k not in ('rating', 'flag', 'label')})
        library_db.annotate_bursts(rows)
        library_db.annotate_hints(rows)
        self.library_model.refresh_all()
        self.info_timer.start()

    def analysis_progress(self, done, total):
        if hasattr(self, 'library_status'):
            if done >= total:
                self.library_status.setText(tr('清晰度与闭眼提示已更新'))
            else:
                self.library_status.setText(tr('正在分析清晰度与闭眼：{done} / {total}', done=done, total=total))

    # -- importing

    def import_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, tr('导入文件夹到图库'))
        if folder:
            self.import_folder(folder)

    def import_dropped(self, paths):
        folders = [p for p in paths if Path(p).is_dir()]
        files = [p for p in paths if Path(p).is_file() and Path(p).suffix.lower() in library_db.extensions()]
        for folder in folders:
            self.import_folder(folder)
        if files:
            self.library_db.add(files)
            self.library_worker.want_index()
            self.library_source = ('recent',)
            self.refresh_sources()
            self.refresh_library_grid(scroll_top=True)

    def import_folder(self, folder, recursive=True):
        """Add a folder's photos to the library in the background (scanning a card or a network
        folder can take a while); the folder is shown when done."""
        self.set_mode('library')
        self.library_status.setText(tr('正在扫描 {folder}…', folder=folder))
        db, relay = self.library_db, self.library_relay

        def work():
            try:
                files = db.scan(folder, recursive)
                added, ids = db.add(files, roots=[folder])
                relay.imported.emit(dict(folder=str(Path(folder).resolve()), added=len(added), total=len(ids)))
            except Exception as exc:
                log.exception('import failed')
                relay.imported.emit(dict(folder=folder, error=str(exc)))
        threading.Thread(target=work, name='library-import', daemon=True).start()

    def import_finished(self, result):
        if 'error' in result:
            self.library_status.setText('')
            return self.error(tr('无法导入文件夹：') + result['error'])
        self.library_worker.want_index()
        self.refresh_sources()
        row = next((f for f in self.library_db.folders() if f['key'] == library_db.key(result['folder'])), None)
        if row is not None:
            item = self.source_tree.find(['folder', row['id']])
            if item is not None:
                self.source_tree.setCurrentItem(item)
        self.library_status.setText(tr('已导入 {added} 张新照片（文件夹中共 {total} 张）', added=result['added'], total=result['total']))
        self.statusBar().showMessage(tr('已导入 {added} 张新照片 · 正在读取拍摄信息与生成缩略图', added=result['added']), 10000)

    def register_in_library(self, paths):
        """Photos opened or imported into the 选片集 join the library too."""
        try:
            added, _ = self.library_db.add([p for p in paths if Path(p).suffix.lower() in library_db.extensions()])
        except Exception:
            log.warning('library registration failed', exc_info=True)
            return
        if added:
            self.library_worker.want_index()

    # -- actions on the selection

    def library_targets(self):
        paths = self.library_selected_paths()
        current = self.library_current_path()
        if not paths and current:
            paths = [current]
        return paths

    def open_library_selection(self):
        paths = [p for p in self.library_targets() if Path(p).is_file()]
        if not paths:
            return
        if len(self.documents) + len([p for p in paths if p not in self.documents]) > 2000:
            return self.error(tr('选片集最多 2000 张照片。'))
        current = self.library_current_path()
        first = current if current in paths else paths[0]
        self.add_documents(paths)
        self.set_mode('develop')
        if first != self.source_path:
            self.open_path(first)
        self.filmstrip.clearSelection()
        for path in paths:
            item = self.film_item(path)
            if item:
                item.setSelected(True)

    def add_to_album(self):
        paths = [p for p in self.library_targets() if Path(p).is_file()]
        if paths:
            self.add_documents(paths[:max(0, 2000 - len(self.documents))])
            self.statusBar().showMessage(tr('已加入选片集 {n} 张', n=len(paths)))
            self.library_model.refresh_all()
            self.source_timer.start()

    def add_keywords_to_selection(self, words):
        ids = self.library_db.ids_of(self.library_targets())
        if not ids:
            return
        self.library_db.add_keywords(ids, words)
        self.after_keyword_change(ids)
        self.statusBar().showMessage(tr('已为 {n} 张照片添加关键词：{words}', n=len(ids), words='、'.join(words)))

    def remove_keyword_from_selection(self, word):
        ids = self.library_db.ids_of(self.library_targets())
        if ids:
            self.library_db.remove_keywords(ids, [word])
            self.after_keyword_change(ids)

    def after_keyword_change(self, ids):
        self.refresh_library_grid()
        self.refresh_sources()
        if self.catalog.write_xmp_on_change:
            rows = self.library_db.rows(('ids', ids))
            library_db.write_keywords_xmp_later({r['path']: r['keywords'] for r in rows})

    def edit_keywords_dialog(self):
        paths = self.library_targets()
        if not paths:
            return
        text, ok = QInputDialog.getText(self, tr('添加关键词'), tr('为所选 {n} 张照片添加关键词（逗号分隔）：', n=len(paths)))
        if ok:
            words = library_db.clean_keywords(text)
            if words:
                self.add_keywords_to_selection(words)

    def reveal_in_folder(self, path):
        if host.MACOS:
            import subprocess
            subprocess.Popen(['open', '-R', path])
        elif host.WINDOWS:
            import subprocess
            subprocess.Popen(['explorer', '/select,', os.path.normpath(path)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))

    def remove_from_library(self):
        paths = self.library_targets()
        if not paths:
            return
        answer = QMessageBox.question(self, tr('从图库移除'), tr('从图库中移除所选 {n} 张照片？\n磁盘上的原片不会删除，评级与选片集不受影响。', n=len(paths)))
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.library_db.remove(self.library_db.ids_of(paths))
        self.refresh_sources()
        self.refresh_library_grid()

    def export_library_selection(self):
        paths = [p for p in self.library_targets() if Path(p).is_file()]
        if not paths:
            return
        self.add_documents(paths[:max(0, 2000 - len(self.documents))])
        self.filmstrip.clearSelection()
        for path in paths:
            item = self.film_item(path)
            if item:
                item.setSelected(True)
        self.batch_export()

    def timelapse_from_library(self):
        """The selected photos (or all the grid shows) join the 选片集 and become a time-lapse."""
        paths = self.library_targets()
        if len(paths) < 2:
            paths = [r['path'] for r in self.library_model.rows]
        paths = [p for p in paths if Path(p).is_file()]
        room = max(0, 2000 - len(self.documents))
        if len([p for p in paths if p not in self.documents]) > room:
            return self.error(tr('选片集最多 2000 张照片。'))
        self.add_documents(paths)
        self.make_timelapse(paths=paths)

    def grid_menu(self, point):
        index = self.grid.indexAt(point)
        if index.isValid() and not self.grid.selectionModel().isSelected(index):
            self.grid.selectionModel().setCurrentIndex(index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        paths = self.library_targets()
        menu = QMenu(self)
        count = len(paths)
        menu.addAction(tr('在显影中打开') + '\tEnter', self.open_library_selection).setEnabled(count > 0)
        menu.addAction(tr('选片模式') + '\tC', self.start_culling).setEnabled(count > 0 or bool(self.library_model.rows))
        menu.addAction(tr('加入选片集'), self.add_to_album).setEnabled(count > 0)
        menu.addAction(tr('批量导出 {count} 张…', count=count) if count > 1 else tr('批量导出…'),
                       self.export_library_selection).setEnabled(count > 0 and not self.work.busy())
        menu.addAction(tr('制作延时视频…'), self.timelapse_from_library).setEnabled(
            (count > 1 or len(self.library_model.rows) > 1) and not self.work.busy())
        menu.addSeparator()
        self.add_rating_actions(menu)
        menu.addAction(tr('添加关键词…'), self.edit_keywords_dialog).setEnabled(count > 0)
        menu.addAction(tr('重新分析清晰度与闭眼'), self.reanalyse_selection).setEnabled(count > 0)
        menu.addSeparator()
        menu.addAction(tr('从 XMP 读取评级'), self.read_selected_xmp).setEnabled(count > 0)
        menu.addAction(tr('评级写入 XMP 附属文件'), self.write_selected_xmp).setEnabled(count > 0)
        menu.addSeparator()
        menu.addAction(tr('在文件夹中显示'), lambda: self.reveal_in_folder(paths[0])).setEnabled(count == 1)
        menu.addAction(tr('从图库移除…'), self.remove_from_library).setEnabled(count > 0)
        menu.exec(self.grid.viewport().mapToGlobal(point))

    def reanalyse_selection(self):
        ids = self.library_db.ids_of(self.library_targets())
        with self.library_db._lock:
            for start in range(0, len(ids), 500):
                chunk = ids[start:start + 500]
                self.library_db._db().execute(f'UPDATE photos SET analysis=0 WHERE id IN ({",".join("?" * len(chunk))})', chunk)
        rows = self.library_db.rows(('ids', ids))
        self.previews_forget(rows)
        self.library_worker.want_analysis([dict(id=r['id'], path=r['path']) for r in rows], first=True)

    def previews_forget(self, rows):
        self.thumb_images.forget([r['path'] for r in rows])

    # -- sources: menus and smart albums

    def source_menu(self, item, position):
        source = tuple(item.data(0, SOURCE_ROLE) or ()) if item is not None else ()
        menu = QMenu(self)
        if source and source[0] == 'folder':
            folder = self.library_db.folder(source[1])
            menu.addAction(tr('重新扫描文件夹'), lambda: self.rescan_folder(source[1]))
            if folder:
                menu.addAction(tr('在文件夹中显示'), lambda: self.reveal_in_folder(folder['path']))
            menu.addAction(tr('从图库移除文件夹…'), lambda: self.remove_folder(source[1]))
        elif source and source[0] == 'collection':
            menu.addAction(tr('编辑智能相册…'), lambda: self.edit_smart_album(source[1]))
            menu.addAction(tr('删除智能相册'), lambda: self.delete_smart_album(source[1]))
        elif source and source[0] == 'keyword':
            menu.addAction(tr('重命名关键词…'), lambda: self.rename_keyword(source[1]))
        menu.addAction(tr('导入文件夹…'), self.import_folder_dialog)
        menu.addAction(tr('＋ 智能相册'), self.new_smart_album)
        menu.exec(position)

    def rescan_folder(self, folder_id):
        added, missing = self.library_db.rescan(folder_id)
        self.library_worker.want_index()
        self.refresh_sources()
        self.refresh_library_grid()
        self.statusBar().showMessage(tr('重新扫描：新增 {added} 张，{missing} 张已找不到', added=len(added), missing=len(missing)))

    def remove_folder(self, folder_id):
        folder = self.library_db.folder(folder_id)
        if folder is None:
            return
        answer = QMessageBox.question(self, tr('从图库移除'), tr('从图库中移除文件夹“{folder}”及其子文件夹中的照片？\n磁盘上的文件不会删除。', folder=folder['path']))
        if answer == QMessageBox.StandardButton.Yes:
            self.library_db.remove_folder(folder_id)
            self.library_source = ('all',)
            self.refresh_sources()
            self.refresh_library_grid(scroll_top=True)

    def new_smart_album(self):
        dialog = SmartAlbumDialog(self, query=self.library_query()[1] if self.mode == 'library' else '')
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name, query = dialog.values()
            collection = self.library_db.add_collection(name, query)
            self.library_source = ('collection', collection)
            self.refresh_sources()
            self.set_mode('library')
            self.refresh_library_grid(scroll_top=True)

    def edit_smart_album(self, collection_id):
        collection = self.library_db.collection(collection_id)
        if collection is None:
            return
        dialog = SmartAlbumDialog(self, collection['name'], collection['query'])
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name, query = dialog.values()
            self.library_db.update_collection(collection_id, name, query)
            self.refresh_sources()
            self.refresh_library_grid()

    def delete_smart_album(self, collection_id):
        self.library_db.remove_collection(collection_id)
        if self.library_source == ('collection', collection_id):
            self.library_source = ('all',)
        self.refresh_sources()
        self.refresh_library_grid()

    def rename_keyword(self, keyword_id):
        row = next((k for k in self.library_db.keywords() if k['id'] == keyword_id), None)
        if row is None:
            return
        name, ok = QInputDialog.getText(self, tr('重命名关键词'), tr('新名称：'), text=row['name'])
        name = name.strip()
        if ok and name and name != row['name']:
            ids = [r['id'] for r in self.library_db.rows(('keyword', keyword_id))]
            self.library_db.add_keywords(ids, [name])
            self.library_db.remove_keywords(ids, [row['name']])
            self.after_keyword_change(ids)
            self.refresh_sources()

    # -- the preview cache

    def build_cache_menu(self, menu):
        cache = menu.addMenu(tr('预览缓存'))
        from PySide6.QtGui import QActionGroup
        group = QActionGroup(self)
        group.setExclusive(True)
        current = preview_cache.limit_mb()
        for megabytes in preview_cache.CHOICES:
            title = tr('关闭') if megabytes == 0 else f'{megabytes // 1024} GB'
            action = cache.addAction(title, lambda checked=False, m=megabytes: self.set_cache_limit(m))
            action.setCheckable(True)
            action.setChecked(megabytes == current)
            group.addAction(action)
        cache.addSeparator()
        cache.addAction(tr('清空预览缓存'), self.clear_preview_cache)
        cache.aboutToShow.connect(lambda: cache.setToolTip(''))
        return cache

    def set_cache_limit(self, megabytes):
        try:
            i18n.save_setting(preview_cache.SETTING, megabytes)
        except OSError as exc:
            return self.error(str(exc))
        self.previews.set_limit(megabytes)
        self.statusBar().showMessage(tr('预览缓存上限：{size}', size=tr('关闭') if not megabytes else f'{megabytes // 1024} GB'))

    def clear_preview_cache(self):
        size = self.previews.total_bytes()
        self.previews.clear()
        self.thumb_images.forget()
        self.statusBar().showMessage(tr('已清空预览缓存（{size}）', size=size_text(size)))
        self.library_model.refresh_all()

    # -- culling

    def start_culling(self):
        from .cull_view import CullWindow
        if self.cull_window is not None:
            self.cull_window.raise_()
            self.cull_window.activateWindow()
            return
        if self.mode == 'library':
            if self.library_dirty:
                self.refresh_library_grid()
            rows = list(self.library_model.rows)
            selected = self.library_selected_paths()
            if len(selected) > 1:
                rows = [r for r in rows if r['path'] in set(selected)]
            current = self.library_current_path()
        else:
            paths = [self.filmstrip.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.filmstrip.count())
                     if not self.filmstrip.item(i).isHidden()]
            selected = self.selected_paths()
            if len(selected) > 1:
                paths = [p for p in paths if p in set(selected)]
            self.register_in_library(paths)
            rows = self.library_db.rows(('paths', paths))
            library_db.annotate_bursts(rows)
            library_db.annotate_hints(rows)
            for row in rows:
                row.update(self.catalog.get(row['path']))
            current = self.source_path
        rows = [r for r in rows if not r.get('missing')]
        if not rows:
            self.statusBar().showMessage(tr('没有可选片的照片：先导入文件夹或照片。'))
            return
        start = next((i for i, r in enumerate(rows) if r['path'] == current), 0)
        self.want_library_work(rows)
        self.cull_window = CullWindow(self, rows, start)
        self.cull_window.closed.connect(self.culling_closed)
        self.cull_window.show_fullscreen()

    def culling_closed(self, path):
        self.cull_window = None
        if self.mode == 'library':
            self.refresh_library_grid()
            if path:
                self.select_library_path(path)

