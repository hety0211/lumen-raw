"""The time-lapse dialog (2.0.0): the sequence with its keyframes, and the video settings."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListView, QListWidget, QListWidgetItem,
                               QPushButton, QSpinBox, QStyledItemDelegate, QVBoxLayout)

from . import i18n, theme, timelapse
from .i18n import tr

KEYFRAME_ROLE = Qt.ItemDataRole.UserRole + 30
PIXMAP_ROLE = Qt.ItemDataRole.UserRole + 31
EDITED_ROLE = Qt.ItemDataRole.UserRole + 32
CELL = QSize(138, 112)


class FrameDelegate(QStyledItemDelegate):
    """A frame of the sequence: thumbnail, number and name; keyframes are outlined and marked."""

    def sizeHint(self, option, index):
        return CELL

    def paint(self, painter, option, index):
        rect = QRectF(option.rect).adjusted(3, 3, -3, -3)
        key = index.data(Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked.value or \
            index.data(Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(theme.color('accent'), 2) if key else QPen(theme.color('border'), 1))
        painter.setBrush(theme.color('accent_soft') if key else theme.color('card'))
        painter.drawRoundedRect(rect, 7, 7)
        area = rect.adjusted(5, 5, -5, -24)
        pixmap = index.data(PIXMAP_ROLE)
        if isinstance(pixmap, QPixmap) and not pixmap.isNull():
            scaled = pixmap.scaled(area.size().toSize(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            target = QRectF(area.center().x() - scaled.width() / 2, area.center().y() - scaled.height() / 2,
                            scaled.width(), scaled.height())
            painter.drawPixmap(target.toRect(), scaled)
        else:
            painter.fillRect(area, theme.color('plot'))
        if key:
            badge = QRectF(area.left() + 3, area.top() + 3, 50, 17)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(theme.color('accent'))
            painter.drawRoundedRect(badge, 8, 8)
            diamond = QPainterPath()
            c = QPointF(badge.left() + 9, badge.center().y())
            diamond.moveTo(c + QPointF(0, -4.5))
            diamond.lineTo(c + QPointF(4.5, 0))
            diamond.lineTo(c + QPointF(0, 4.5))
            diamond.lineTo(c + QPointF(-4.5, 0))
            diamond.closeSubpath()
            painter.setBrush(theme.color('accent_text'))
            painter.drawPath(diamond)
            painter.setPen(theme.color('accent_text'))
            painter.drawText(badge.adjusted(15, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, tr('关键帧'))
        if index.data(EDITED_ROLE):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(theme.color('accent'))
            painter.drawEllipse(QPointF(area.right() - 7, area.top() + 7), 3.5, 3.5)
        painter.setPen(theme.color('head' if key else 'dim'))
        text = QRectF(rect.left() + 6, rect.bottom() - 21, rect.width() - 12, 18)
        painter.drawText(text, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         painter.fontMetrics().elidedText(index.data(Qt.ItemDataRole.DisplayRole) or '', Qt.TextElideMode.ElideMiddle,
                                                          int(text.width())))
        painter.restore()
SETTINGS = dict(format='timelapse_format', resolution='timelapse_resolution', aspect='timelapse_aspect',
                fps='timelapse_fps', ten_bit='timelapse_ten_bit', hardware='timelapse_hardware',
                deflicker='timelapse_deflicker')


def _setting(name, default):
    return i18n.setting(SETTINGS[name], default)


def _save(name, value):
    try:
        i18n.save_setting(SETTINGS[name], value)
    except OSError:
        pass


class TimelapseDialog(QDialog):
    def __init__(self, window, frames, keyframes=None):
        super().__init__(window)
        self.window, self.frames = window, frames
        self.setWindowTitle(tr('延时摄影视频'))
        self.setMinimumSize(760, 640)
        root = QVBoxLayout(self)
        heading = QLabel(tr('延时摄影视频  /  {n} 张照片', n=len(frames)))
        heading.setObjectName('section')
        root.addWidget(heading)
        text = QLabel(tr('照片按拍摄时间排列，每张一帧。勾选的照片是关键帧：它们在选片集中的调整（曝光、白平衡、色彩、曲线、裁切等）决定画面，'
                         '中间的照片在相邻两个关键帧之间平滑过渡；裁切不同时画面会缓慢平移、缩放。先在显影中调好关键帧，再来这里生成。'
                         '拍摄时固定白平衡与曝光效果最好；“去闪烁”可抹平相邻照片的亮度与色彩跳动。'))
        text.setWordWrap(True)
        text.setObjectName('subtle')
        root.addWidget(text)
        self.list = QListWidget()
        self.list.setObjectName('framesList')
        self.list.setViewMode(QListView.ViewMode.IconMode)
        self.list.setFlow(QListView.Flow.LeftToRight)
        self.list.setWrapping(True)
        self.list.setResizeMode(QListView.ResizeMode.Adjust)
        self.list.setMovement(QListView.Movement.Static)
        self.list.setGridSize(CELL)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.list.setUniformItemSizes(True)
        self.list.setItemDelegate(FrameDelegate(self.list))
        marks = set(keyframes if keyframes is not None else timelapse.default_keyframes(frames))
        for index, frame in enumerate(frames):
            item = QListWidgetItem(f'{index + 1}  {frame["name"]}')
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if index in marks else Qt.CheckState.Unchecked)
            tip = [frame['path'], frame.get('captured') or '', tr('有自己的调整') if frame.get('edited') else '',
                   tr('单击设为或取消关键帧')]
            item.setToolTip('\n'.join(x for x in tip if x))
            item.setData(KEYFRAME_ROLE, index)
            item.setData(EDITED_ROLE, bool(frame.get('edited')))
            self.list.addItem(item)
        self.list.itemClicked.connect(self.toggle)
        self.list.itemChanged.connect(lambda _: self.refresh())
        root.addWidget(self.list, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr('关键帧：')))
        for title, choose in ((tr('首尾两张'), lambda: self.set_marks({0, len(frames) - 1})),
                              (tr('首尾与已调整的照片'), lambda: self.set_marks(set(timelapse.default_keyframes(frames)))),
                              (tr('全部取消'), lambda: self.set_marks(set()))):
            button = QPushButton(title)
            button.clicked.connect(choose)
            row.addWidget(button)
        row.addStretch()
        self.marks_label = QLabel()
        self.marks_label.setObjectName('subtle')
        row.addWidget(self.marks_label)
        root.addLayout(row)
        form = QFormLayout()
        self.format = QComboBox()
        for key, spec in timelapse.FORMATS.items():
            self.format.addItem(spec['title'], key)
        self.resolution = QComboBox()
        for key, title, _ in timelapse.RESOLUTIONS:
            self.resolution.addItem(tr(title), key)
        self.aspect = QComboBox()
        for key, title, _ in timelapse.ASPECTS:
            self.aspect.addItem(tr(title), key)
        self.fps = QComboBox()
        for value in timelapse.FRAME_RATES:
            self.fps.addItem(f'{value} fps', value)
        self.repeat = QSpinBox()
        self.repeat.setRange(1, 10)
        self.repeat.setSuffix(tr(' 帧'))
        self.repeat.setToolTip(tr('每张照片在视频中重复的帧数：照片少时让视频慢一些'))
        self.mbps = QSpinBox()
        self.mbps.setRange(2, 800)
        self.mbps.setSuffix(' Mbps')
        self.mbps.setToolTip(tr('最大码率：画面复杂时也不会超过；平静的画面会更小'))
        self.mbps_auto = QCheckBox(tr('自动'))
        self.mbps_auto.setChecked(True)
        rate = QHBoxLayout()
        rate.addWidget(self.mbps, 1)
        rate.addWidget(self.mbps_auto)
        self.ten_bit = QCheckBox(tr('10 位色深（天空渐变更平滑）'))
        self.hardware = QCheckBox(tr('优先用显卡编码（AMD / NVIDIA / Intel，不可用时自动改用 CPU）'))
        self.deflicker = QComboBox()
        for key, title, _ in timelapse.DEFLICKER:
            self.deflicker.addItem(tr(title), key)
        self.deflicker.setToolTip(tr('按相邻几张照片的平均亮度与白平衡校正每一帧：去掉亮度闪烁和自动白平衡的色彩跳动，保留日出日落等缓慢变化'))
        self.output = QLineEdit()
        browse = QPushButton(tr('选择…'))
        browse.clicked.connect(self.choose_output)
        target = QHBoxLayout()
        target.addWidget(self.output, 1)
        target.addWidget(browse)
        for title, widget in ((tr('格式'), self.format), (tr('分辨率'), self.resolution), (tr('画面比例'), self.aspect),
                              (tr('帧率'), self.fps), (tr('每张照片'), self.repeat), (tr('最大码率'), rate),
                              ('', self.ten_bit), ('', self.hardware), (tr('去闪烁'), self.deflicker), (tr('保存到'), target)):
            form.addRow(title, widget)
        root.addLayout(form)
        self.summary = QLabel()
        self.summary.setObjectName('subtle')
        self.summary.setWordWrap(True)
        root.addWidget(self.summary)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.ok.setText(tr('生成视频'))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr('取消'))
        buttons.accepted.connect(self.confirm)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.frame = self._frame_size()
        self._restore()
        for signal in (self.format.currentIndexChanged, self.resolution.currentIndexChanged, self.aspect.currentIndexChanged,
                       self.fps.currentIndexChanged, self.repeat.valueChanged, self.mbps.valueChanged,
                       self.mbps_auto.toggled, self.ten_bit.toggled):
            signal.connect(lambda *_: self.refresh())
        self.format.currentIndexChanged.connect(lambda *_: self.suffix())
        self.refresh()
        self.load_icons()

    # -- state

    def _frame_size(self):
        frame = self.frames[0]
        try:
            from . import model
            return timelapse.frame_size(frame.get('edits') or model.recipe(), timelapse.photo_size(frame['path']))
        except Exception:
            return 6000, 4000

    def _restore(self):
        def pick(combo, value):
            index = combo.findData(value)
            if index >= 0:
                combo.setCurrentIndex(index)
        pick(self.format, _setting('format', 'h264'))
        pick(self.resolution, _setting('resolution', '4k'))
        pick(self.aspect, _setting('aspect', 'original'))
        try:
            pick(self.fps, int(_setting('fps', '30')))
        except ValueError:
            pick(self.fps, 30)
        self.ten_bit.setChecked(_setting('ten_bit', '0') == '1')
        self.hardware.setChecked(_setting('hardware', '1') != '0')
        pick(self.deflicker, _setting('deflicker', 'light'))
        first, last = Path(self.frames[0]['path']), Path(self.frames[-1]['path'])
        name = f'{first.stem}-{last.stem}-timelapse' + timelapse.FORMATS[self.format.currentData()]['suffix']
        self.output.setText(str(first.with_name(name)))

    def marks(self):
        return [self.list.item(i).data(KEYFRAME_ROLE) for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.CheckState.Checked]

    def set_marks(self, marks):
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.CheckState.Checked if i in marks else Qt.CheckState.Unchecked)
        self.list.blockSignals(False)
        self.refresh()

    def video_size(self):
        long_edge = dict((k, v) for k, _, v in timelapse.RESOLUTIONS)[self.resolution.currentData()]
        aspect = dict((k, v) for k, _, v in timelapse.ASPECTS)[self.aspect.currentData()]
        return timelapse.output_size(*self.frame, long_edge, aspect, self.format.currentData())

    def suffix(self):
        path = Path(self.output.text())
        wanted = timelapse.FORMATS[self.format.currentData()]['suffix']
        if path.suffix.lower() != wanted:
            self.output.setText(str(path.with_suffix(wanted)))

    def label_resolutions(self):
        """Each size with its pixels for the chosen aspect ratio (the sensor's by default)."""
        aspect = dict((k, v) for k, _, v in timelapse.ASPECTS)[self.aspect.currentData()]
        for index, (key, title, long_edge) in enumerate(timelapse.RESOLUTIONS):
            width, height = timelapse.output_size(*self.frame, long_edge, aspect, self.format.currentData())
            self.resolution.setItemText(index, f'{tr(title)}  ·  {width} × {height}')

    def refresh(self):
        fmt = self.format.currentData()
        self.label_resolutions()
        width, height = self.video_size()
        prores = fmt == 'prores'
        self.mbps.setEnabled(not prores and not self.mbps_auto.isChecked())
        self.mbps_auto.setEnabled(not prores)
        if self.mbps_auto.isChecked() and not prores:
            self.mbps.blockSignals(True)
            self.mbps.setValue(timelapse.default_mbps(fmt, width, height))
            self.mbps.blockSignals(False)
        ten = timelapse.FORMATS[fmt]['pixel']
        self.ten_bit.setEnabled(bool(ten[0] and ten[1]))
        if not ten[0]:
            self.ten_bit.setChecked(True)
        elif not ten[1]:
            self.ten_bit.setChecked(False)
        frames = len(self.frames) * self.repeat.value()
        seconds = frames / self.fps.currentData()
        text = tr('{width} × {height} · {frames} 帧 · 时长 {seconds:.1f} 秒', width=width, height=height, frames=frames, seconds=seconds)
        if not prores:
            text += tr(' · 最大约 {size} MB', size=max(1, round(seconds * self.mbps.value() / 8)))
        else:
            text += tr(' · ProRes 约 {size} MB', size=max(1, round(seconds * width * height * 30 * 8.2e-6 / 8)))
        marks = self.marks()
        self.marks_label.setText(tr('已选 {n} 个关键帧', n=len(marks)) if marks else tr('未选关键帧：全部使用第一张的调整'))
        self.summary.setText(text)

    def toggle(self, item):
        item.setCheckState(Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked)

    def load_icons(self):
        images = getattr(self.window, 'thumb_images', None)
        if images is None:
            return
        images.updated.connect(self.icon_arrived)
        for i in range(self.list.count()):
            self.icon_arrived(self.frames[i]['path'])

    def icon_arrived(self, path):
        image = self.window.thumb_images.image(path)
        if image is None:
            return
        for i, frame in enumerate(self.frames):
            if frame['path'] == path:
                pixmap = QPixmap.fromImage(image).scaled(QSize(256, 170), Qt.AspectRatioMode.KeepAspectRatio,
                                                          Qt.TransformationMode.SmoothTransformation)
                self.list.item(i).setData(PIXMAP_ROLE, pixmap)

    def choose_output(self):
        fmt = timelapse.FORMATS[self.format.currentData()]
        path, _ = QFileDialog.getSaveFileName(self, tr('保存延时视频'), self.output.text(), f'{fmt["title"]} (*{fmt["suffix"]})')
        if path:
            self.output.setText(path)
            self.suffix()

    def confirm(self):
        path = Path(self.output.text())
        if not path.parent.is_dir():
            self.summary.setText(tr('请选择已存在的文件夹。'))
            return
        if any(path.resolve() == Path(f['path']).resolve() for f in self.frames):
            self.summary.setText(tr('请选择不同的文件名，以保留原片。'))
            return
        _save('format', self.format.currentData())
        _save('resolution', self.resolution.currentData())
        _save('aspect', self.aspect.currentData())
        _save('fps', self.fps.currentData())
        _save('ten_bit', '1' if self.ten_bit.isChecked() else '0')
        _save('hardware', '1' if self.hardware.isChecked() else '0')
        _save('deflicker', self.deflicker.currentData())
        self.accept()

    def options(self):
        return dict(output=self.output.text(), format=self.format.currentData(), resolution=self.resolution.currentData(),
                    aspect=self.aspect.currentData(), fps=self.fps.currentData(), repeat=self.repeat.value(),
                    mbps=None if self.mbps_auto.isChecked() else self.mbps.value(), ten_bit=self.ten_bit.isChecked(),
                    hardware=self.hardware.isChecked(), deflicker=self.deflicker.currentData(), keyframes=self.marks())
