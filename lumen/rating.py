"""Filmstrip ratings, flags, colour labels and filters (1.6.0).

The values live in ``catalog.Catalog`` (an append-only log in the data folder), so they are kept
at once and need no saving; albums carry a copy so they travel with the album.  Shortcuts follow
Lightroom: 0–5 stars, P pick, X reject, U unflag, 6–9 red / yellow / green / blue labels.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from PySide6.QtCore import Qt, QPointF, QRectF, QTimer, Signal, QObject
from PySide6.QtGui import QAction, QColor, QKeySequence, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLineEdit, QMenu, QMessageBox,
                               QStyledItemDelegate, QToolButton, QWidget)

from . import catalog, i18n
from .i18n import tr, N_

log = logging.getLogger(__name__)

RATING_ROLE = Qt.ItemDataRole.UserRole + 1
FLAG_ROLE = Qt.ItemDataRole.UserRole + 2
LABEL_ROLE = Qt.ItemDataRole.UserRole + 3
LABEL_NAMES = dict(red=N_('红色'), yellow=N_('黄色'), green=N_('绿色'), blue=N_('蓝色'), purple=N_('紫色'))
#: Quick filters: (title, query).
QUICK_FILTERS = ((N_('全部照片'), ''), (N_('已留用'), 'flag:pick'), (N_('未排除'), '-flag:reject'),
                 (N_('★ 1 以上'), 'rating>=1'), (N_('★ 3 以上'), 'rating>=3'), (N_('★ 5'), 'rating=5'),
                 (N_('有色标'), 'label:any'), (N_('已排除'), 'flag:reject'))


def star_path(cx, cy, outer, inner=None):
    inner = inner or outer * .45
    import math
    points = []
    for i in range(10):
        radius = outer if i % 2 == 0 else inner
        angle = -math.pi / 2 + i * math.pi / 5
        points.append(QPointF(cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    path = QPainterPath()
    path.addPolygon(QPolygonF(points))
    path.closeSubpath()
    return path


def flag_path(x, y, size):
    path = QPainterPath()
    path.moveTo(x, y)
    path.lineTo(x, y + size)
    path.moveTo(x, y)
    path.lineTo(x + size * .8, y + size * .25)
    path.lineTo(x, y + size * .5)
    return path


class FilmDelegate(QStyledItemDelegate):
    """Draws stars, the pick / reject flag and the colour label over filmstrip thumbnails."""

    def __init__(self, view, icon_size):
        super().__init__(view)
        self.icon_size = icon_size

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        rating = int(index.data(RATING_ROLE) or 0)
        flag = index.data(FLAG_ROLE) or ''
        label = index.data(LABEL_ROLE) or ''
        if not (rating or flag or label):
            return
        rect = option.rect
        w, h = self.icon_size.width(), self.icon_size.height()
        decoration = index.data(Qt.ItemDataRole.DecorationRole)
        if decoration is not None and not decoration.isNull():
            # The thumbnail keeps its aspect ratio inside the icon box (portrait photos are narrower).
            actual = decoration.actualSize(self.icon_size)
            w, h = actual.width(), actual.height()
        top = rect.y() + 3 + (self.icon_size.height() - h) / 2
        icon = QRectF(rect.x() + (rect.width() - w) / 2, top, w, h)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if flag == 'reject':
            painter.fillRect(icon, QColor(10, 12, 10, 150))
            painter.setPen(QPen(QColor('#f0a0a0'), 2))
            c = QPointF(icon.right() - 9, icon.top() + 9)
            painter.drawLine(c + QPointF(-4, -4), c + QPointF(4, 4))
            painter.drawLine(c + QPointF(-4, 4), c + QPointF(4, -4))
        elif flag == 'pick':
            painter.setPen(QPen(QColor(0, 0, 0, 160), 3))
            painter.drawPath(flag_path(icon.left() + 6, icon.top() + 4, 11))
            painter.setPen(QPen(QColor('#f4f7ee'), 1.6))
            painter.drawPath(flag_path(icon.left() + 6, icon.top() + 4, 11))
        if label in catalog.LABEL_COLORS:
            painter.setPen(QPen(QColor(0, 0, 0, 150), 1))
            painter.setBrush(QColor(catalog.LABEL_COLORS[label]))
            painter.drawEllipse(QPointF(icon.right() - 8, icon.bottom() - 8) if flag == 'reject' else
                                QPointF(icon.right() - 8, icon.top() + 8), 4.5, 4.5)
        if rating:
            strip = QRectF(icon.left(), icon.bottom() - 13, 8 + rating * 11, 13)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 140))
            painter.drawRoundedRect(strip, 3, 3)
            painter.setBrush(QColor('#f2d36b'))
            for i in range(rating):
                painter.drawPath(star_path(strip.left() + 9 + i * 11, strip.center().y() + .5, 4.6))
        painter.restore()


class RatingBar(QWidget):
    """Stars, pick / reject and colour labels for the selected photos."""
    rated = Signal(int)
    flagged = Signal(str)
    labelled = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(1)
        self.stars = []
        for value in range(1, 6):
            button = self._button(tr('{n} 星（按 {n}）', n=value), lambda checked=False, v=value: self.rated.emit(
                0 if self.rating == v else v))
            self.stars.append(button)
            row.addWidget(button)
        row.addSpacing(6)
        self.pick = self._button(tr('留用（P）'), lambda: self.flagged.emit('' if self.flag == 'pick' else 'pick'))
        self.reject = self._button(tr('排除（X）'), lambda: self.flagged.emit('' if self.flag == 'reject' else 'reject'))
        row.addWidget(self.pick)
        row.addWidget(self.reject)
        row.addSpacing(6)
        self.labels = {}
        for number, key in zip((6, 7, 8, 9, None), catalog.LABELS):
            hint = tr('{label}（按 {n}）', label=tr(LABEL_NAMES[key]), n=number) if number else tr(LABEL_NAMES[key])
            button = self._button(hint, lambda checked=False, k=key: self.labelled.emit('' if self.label == k else k))
            self.labels[key] = button
            row.addWidget(button)
        self.rating, self.flag, self.label = 0, '', ''
        self.setEnabled(False)

    def _button(self, tip, callback):
        button = QToolButton()
        button.setToolTip(tip)
        button.setFixedSize(19, 22)
        button.setAutoRaise(True)
        button.clicked.connect(callback)
        button.paintEvent = lambda event, b=button: self._paint(b)
        return button

    def set_state(self, rating, flag, label):
        self.rating, self.flag, self.label = rating, flag, label
        self.update()
        for button in self.stars + [self.pick, self.reject] + list(self.labels.values()):
            button.update()

    def _paint(self, button):
        painter = QPainter(button)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        enabled = self.isEnabled()
        hover = button.underMouse() and enabled
        center = QPointF(button.width() / 2, button.height() / 2)
        if button in self.stars:
            filled = self.stars.index(button) < self.rating
            painter.setPen(QPen(QColor('#d9c060' if enabled else '#5d6358'), 1))
            painter.setBrush(QColor('#f2d36b') if filled and enabled else QColor('#ffffff22') if hover else Qt.BrushStyle.NoBrush)
            painter.drawPath(star_path(center.x(), center.y(), 6.5))
        elif button is self.pick:
            active = self.flag == 'pick'
            painter.setPen(QPen(QColor('#f4f7ee' if active else '#9fb091' if enabled else '#5d6358'), 2 if active else 1.3))
            painter.drawPath(flag_path(center.x() - 4, center.y() - 6, 12))
        elif button is self.reject:
            active = self.flag == 'reject'
            painter.setPen(QPen(QColor('#f0a0a0' if active else '#9fb091' if enabled else '#5d6358'), 2 if active else 1.3))
            painter.drawLine(center + QPointF(-4, -4), center + QPointF(4, 4))
            painter.drawLine(center + QPointF(-4, 4), center + QPointF(4, -4))
        else:
            key = next(k for k, b in self.labels.items() if b is button)
            color = QColor(catalog.LABEL_COLORS[key])
            if not enabled:
                color.setAlpha(70)
            painter.setBrush(color)
            painter.setPen(QPen(QColor('#f4f7ee'), 2) if self.label == key else QPen(QColor(0, 0, 0, 120), 1))
            painter.drawEllipse(center, 5.5 if self.label == key or hover else 4.5, 5.5 if self.label == key or hover else 4.5)
        painter.end()


class _Relay(QObject):
    changed = Signal(list)
    read = Signal(dict)


class RatingMixin:
    def init_rating(self):
        self.catalog = catalog.Catalog()
        self.rating_relay = _Relay(self)
        self.rating_relay.changed.connect(self.refresh_ratings)
        self.rating_relay.read.connect(self.xmp_read_done)
        self.catalog.listeners.append(self.rating_relay.changed.emit)
        self.film_filter_terms = []
        self.catalog_timer = QTimer(self)
        self.catalog_timer.setInterval(3000)
        self.catalog_timer.timeout.connect(self.poll_catalog)
        self.catalog_timer.start()

    # -- widgets

    def build_rating_controls(self, row):
        self.rating_bar = RatingBar()
        self.rating_bar.rated.connect(lambda v: self.rate_selected(rating=v))
        self.rating_bar.flagged.connect(lambda v: self.rate_selected(flag=v))
        self.rating_bar.labelled.connect(lambda v: self.rate_selected(label=v))
        row.addWidget(self.rating_bar)
        row.addSpacing(10)
        self.film_quick = QComboBox()
        for title, _ in QUICK_FILTERS:
            self.film_quick.addItem(tr(title))
        self.film_quick.currentIndexChanged.connect(self.quick_filter)
        self.film_quick.setToolTip(tr('筛选图集'))
        row.addWidget(self.film_quick)
        self.film_search = QLineEdit()
        self.film_search.setPlaceholderText(tr('筛选：rating>=3 label:red flag:pick name:DSC'))
        self.film_search.setToolTip(tr('字段：rating（星级，可用 = > >= < <=）、flag（pick / reject / none）、label（red / yellow / green / blue / purple / none / any）、name（文件名，可用 * ?）、ext、folder、edited。\n前加 - 表示排除，多个条件同时满足。例：rating>=3 -flag:reject label:red'))
        self.film_search.setClearButtonEnabled(True)
        self.film_search.setFixedWidth(230)
        self.film_search.textChanged.connect(self.apply_film_filter)
        row.addWidget(self.film_search)

    def install_rating_delegate(self):
        self.filmstrip.setItemDelegate(FilmDelegate(self.filmstrip, self.filmstrip.iconSize()))

    def rating_shortcuts(self):
        bindings = [(str(n), lambda checked=False, v=n: self.rate_selected(rating=v)) for n in range(6)]
        bindings += [('P', lambda: self.rate_selected(flag='pick')), ('X', lambda: self.rate_selected(flag='reject')),
                     ('U', lambda: self.rate_selected(flag=''))]
        bindings += [(str(n), lambda checked=False, k=k: self.rate_selected(label=k, toggle=True))
                     for n, k in zip((6, 7, 8, 9), catalog.LABELS)]
        for shortcut, callback in bindings:
            action = QAction(self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(callback)
            self.addAction(action)

    def build_library_menu(self, menu):
        library = menu.addMenu(tr('图库'))
        self.add_rating_actions(library)
        library.addSeparator()
        library.addAction(tr('从 XMP 读取所选照片的评级'), self.read_selected_xmp)
        library.addAction(tr('将所选照片的评级写入 XMP 附属文件'), self.write_selected_xmp)
        self.xmp_auto_action = library.addAction(tr('评级变化时自动写入 XMP 附属文件'))
        self.xmp_auto_action.setCheckable(True)
        self.xmp_auto_action.setChecked(self.catalog.write_xmp_on_change)
        self.xmp_auto_action.toggled.connect(self.toggle_xmp_sync)
        library.addSeparator()
        library.addAction(tr('导入 Lightroom 目录中的评级…'), self.import_lightroom)
        library.addAction(tr('清除筛选'), lambda: (self.film_quick.setCurrentIndex(0), self.film_search.clear()))
        return library

    def toggle_xmp_sync(self, on):
        i18n.save_setting(catalog.XMP_SETTING, '1' if on else '0')
        self.catalog.write_xmp_on_change = on

    def add_rating_actions(self, menu):
        stars = menu.addMenu(tr('星级'))
        for value in range(6):
            stars.addAction(tr('无星级') if value == 0 else '★' * value, lambda checked=False, v=value: self.rate_selected(rating=v))
        flags = menu.addMenu(tr('旗标'))
        flags.addAction(tr('留用'), lambda: self.rate_selected(flag='pick'))
        flags.addAction(tr('排除'), lambda: self.rate_selected(flag='reject'))
        flags.addAction(tr('取消旗标'), lambda: self.rate_selected(flag=''))
        labels = menu.addMenu(tr('色标'))
        for key in catalog.LABELS:
            labels.addAction(tr(LABEL_NAMES[key]), lambda checked=False, k=key: self.rate_selected(label=k))
        labels.addAction(tr('无色标'), lambda: self.rate_selected(label=''))

    # -- actions

    def rating_targets(self):
        paths = [p for p in self.selected_paths() if p in self.documents]
        if not paths and self.source_path:
            paths = [self.source_path]
        return paths

    def rate_selected(self, toggle=False, **fields):
        """Stars, flags and labels from the bar, menus and shortcuts: the same commands
        (library.rate / flag / label) the command line and MCP agents use."""
        from . import commands
        paths = self.rating_targets()
        if not paths:
            return
        if toggle and 'label' in fields:
            if all(self.catalog.get(p)['label'] == fields['label'] for p in paths):
                fields['label'] = ''
        (field, value), = fields.items()
        command = dict(rating='library.rate', flag='library.flag', label='library.label')[field]
        if field != 'rating' and not value:
            value = 'none'
        try:
            commands.run(self.control.context, command, {field: value, 'paths': paths})
        except commands.CommandError as exc:
            return self.error(str(exc))
        self.statusBar().showMessage(self.rating_message(len(paths), fields))

    def rating_message(self, count, fields):
        if 'rating' in fields:
            value = fields['rating']
            return tr('{count} 张照片：{stars}', count=count, stars='★' * value if value else tr('无星级'))
        if 'flag' in fields:
            return tr('{count} 张照片：{flag}', count=count,
                      flag={'pick': tr('留用'), 'reject': tr('排除')}.get(fields['flag'], tr('取消旗标')))
        label = fields.get('label')
        return tr('{count} 张照片：{label}', count=count, label=tr(LABEL_NAMES[label]) if label else tr('无色标'))

    def refresh_ratings(self, paths=None):
        if not hasattr(self, 'filmstrip'):
            return
        wanted = None if paths is None else {catalog.key(p) for p in paths}
        for i in range(self.filmstrip.count()):
            item = self.filmstrip.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            if wanted is not None and catalog.key(path) not in wanted:
                continue
            meta = self.catalog.get(path)
            item.setData(RATING_ROLE, meta['rating'])
            item.setData(FLAG_ROLE, meta['flag'])
            item.setData(LABEL_ROLE, meta['label'])
        self.apply_film_filter()
        self.update_rating_bar()

    def update_rating_bar(self):
        if not hasattr(self, 'rating_bar'):
            return
        paths = self.rating_targets()
        self.rating_bar.setEnabled(bool(paths))
        meta = self.catalog.get(paths[0]) if paths else catalog.empty()
        self.rating_bar.set_state(meta['rating'], meta['flag'], meta['label'])

    def poll_catalog(self):
        # The command line and the MCP server append to the same log.
        try:
            if self.catalog.refresh():
                self.refresh_ratings()
        except OSError:
            pass

    # -- filters

    def quick_filter(self, index):
        self.apply_film_filter()

    def apply_film_filter(self, *_):
        if not hasattr(self, 'film_search'):
            return
        query = ' '.join(q for q in (QUICK_FILTERS[self.film_quick.currentIndex()][1], self.film_search.text()) if q)
        try:
            terms = catalog.parse_query(query)
            self.film_search.setStyleSheet('')
        except ValueError:
            self.film_search.setStyleSheet('QLineEdit { border-color: #b65a4c; }')
            return
        shown = 0
        needs_edits = any(field == 'edited' for _, field, _, _ in terms)
        for i in range(self.filmstrip.count()):
            item = self.filmstrip.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            edited = needs_edits and self.document_edited(path)
            visible = catalog.matches(terms, path, self.catalog.get(path), edited)
            item.setHidden(not visible)
            shown += visible
        self.film_filter_terms = terms
        if hasattr(self, 'library_count'):
            self.update_library_status()
            if terms:
                self.library_count.setText(self.library_count.text() + tr(' · 筛选后 {n} 张', n=shown))

    def document_edited(self, path):
        """True when a photo has edits beyond what opening it sets (for the ``edited`` filter)."""
        from . import model
        document = self.documents.get(path)
        if not document:
            return False
        edits = self.edits if path == self.source_path else document['edits']
        fresh = model.recipe()
        for key in ('white_balance', 'develop', 'lens', 'process'):
            fresh[key] = edits.get(key, fresh[key])
        return bool(document.get('project')) or edits != fresh

    # -- XMP and Lightroom

    def read_xmp_async(self, paths, overwrite=False):
        paths = [p for p in paths if overwrite or not self.catalog.known(p)]
        if not paths:
            return

        def work():
            try:
                values = catalog.read_xmp(paths)
            except Exception:
                log.exception('XMP read failed')
                values = {}
            self.rating_relay.read.emit(dict(values=values, overwrite=overwrite, count=len(paths)))
        threading.Thread(target=work, name='xmp-read', daemon=True).start()

    def xmp_read_done(self, result):
        changed = self.catalog.merge(result['values'], overwrite=result['overwrite'])
        if result['overwrite']:
            self.statusBar().showMessage(tr('已从 XMP 读取 {n} 张照片的评级', n=len(changed)))

    def read_selected_xmp(self):
        paths = self.rating_targets()
        if paths:
            self.read_xmp_async(paths, overwrite=True)

    def write_xmp_async(self, paths):
        entries = {p: self.catalog.get(p) for p in paths}

        def work():
            try:
                written, errors = catalog.write_xmp(entries)
                if errors:
                    log.warning('XMP write: %s', '; '.join(errors[:5]))
            except Exception:
                log.exception('XMP write failed')
        threading.Thread(target=work, name='xmp-write', daemon=True).start()

    def write_selected_xmp(self):
        paths = self.rating_targets()
        if not paths:
            return
        self.write_xmp_async(paths)
        self.statusBar().showMessage(tr('正在写入 {n} 个 XMP 附属文件（原片不会被修改）', n=len(paths)))

    def import_lightroom(self):
        path, _ = QFileDialog.getOpenFileName(self, tr('选择 Lightroom Classic 目录'), '', tr('Lightroom 目录 (*.lrcat)'))
        if not path:
            return
        try:
            values = catalog.read_lightroom(path)
        except Exception as exc:
            return self.error(tr('无法读取 Lightroom 目录：') + str(exc))
        existing = {p: v for p, v in values.items() if Path(p).is_file()}
        if not values:
            QMessageBox.information(self, 'LUMEN RAW', tr('这个目录中没有评级、旗标或色标。'))
            return
        answer = QMessageBox.question(
            self, tr('导入 Lightroom 评级'),
            tr('目录中有 {total} 张照片带有评级、旗标或色标，其中 {found} 张在本机找到。\n\n'
               '导入后会覆盖这些照片在 LUMEN RAW 中的评级。是否同时把找到的照片加入图集？', total=len(values), found=len(existing)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Cancel:
            return
        for photo, fields in values.items():
            self.catalog.set([photo], **{k: fields.get(k, catalog.empty()[k]) for k in ('rating', 'flag', 'label')})
        if answer == QMessageBox.StandardButton.Yes and existing:
            self.add_documents(list(existing)[:2000 - len(self.documents)])
        self.statusBar().showMessage(tr('已导入 {n} 张照片的 Lightroom 评级', n=len(values)))

    # -- albums

    def album_ratings(self, path):
        meta = self.catalog.get(path)
        return {k: v for k, v in meta.items() if v}

    def restore_album_ratings(self, records):
        """Albums carry ratings; they fill photos the catalog does not know yet."""
        values = {path: fields for path, fields in records.items() if fields}
        self.catalog.merge(values)
