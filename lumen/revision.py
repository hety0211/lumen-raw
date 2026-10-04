"""Camera Kelvin controls and the source-space retouch workspace."""
import copy
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QCheckBox, QListWidget, QLabel
from .widgets import AdjustSlider
from . import host
from .i18n import tr


class RevisionMixin:
    def build_camera_wb(self, layout):
        self.camera_wb_label = QLabel(tr('打开 RAW 后读取相机白平衡'))
        self.camera_wb_label.setWordWrap(True)
        self.camera_wb_label.setObjectName('subtle')
        layout.addWidget(self.camera_wb_label)
        self.kelvin = AdjustSlider(tr('色温  K'), 1500, 25000, 10)
        self.kelvin.changed.connect(self.kelvin_changed)
        self.kelvin.committed.connect(self.commit)
        layout.addWidget(self.kelvin)

    def kelvin_changed(self, value):
        if not self.refreshing and self.edits['white_balance']['camera_kelvin'] is not None:
            self.edits['white_balance']['kelvin'] = value
            self.changed()

    def refresh_camera_wb(self):
        settings = self.edits['white_balance']
        base = settings['camera_kelvin']
        self.kelvin.setEnabled(base is not None)
        self.kelvin.setValue(settings['kelvin'] or 5500)
        self.kelvin.default_value = base or 5500
        self.kelvin.title.setText(tr('色温  K') if base else tr('色温  K · 无可用基准'))
        if base is None:
            text = tr('未读取到可用的相机色温；可用相对冷暖和吸管调整。')
        elif settings['estimated']:
            text = tr('相机未记录有效 K 值 · 估算基准 ≈ {base:.0f} K\n原始通道增益已保留；估算值可能与 LR 不同。', base=base)
        else:
            text = tr('相机记录：{base:.0f} K · 在原始白平衡基础上调整', base=base)
        if self.edits['wb_gain'] != [1., 1., 1.]:
            text += tr('\n当前还应用了吸管取样校正。')
        self.camera_wb_label.setText(text)
        self.camera_wb_label.setToolTip(self.info.get('wb_warning', ''))

    def build_retouch(self):
        layout = self.panel(tr('修复'))
        title = QLabel(tr('修复与仿制  /  RETOUCH'))
        title.setObjectName('section')
        layout.addWidget(title)
        self.retouch_tool = QComboBox()
        self.retouch_tool.addItems([tr('污点修复 · 邻域修补'), tr('仿制图章 · 对齐取样')])
        self.retouch_tool.currentIndexChanged.connect(self.update_tool)
        layout.addWidget(self.retouch_tool)
        tip = QLabel(tr('污点修复：在灰尘、小污点上点击或涂抹。\n仿制图章：{OPTION} + 单击取样，再涂抹目标区域。\n取样位置随笔触对齐；重新 {OPTION} 取样可更换来源。\n{NAVIGATION}；修复在裁切与调色之前应用。', OPTION=host.OPTION, NAVIGATION=host.NAVIGATION))
        tip.setWordWrap(True)
        tip.setObjectName('subtle')
        layout.addWidget(tip)
        self.retouch_controls = {}
        for key, title, lo, hi, default in [('size', tr('画笔直径 %'), .2, 20, 3),
              ('feather', tr('羽化'), 0, 100, 50), ('opacity', tr('不透明度'), 0, 100, 100)]:
            c = AdjustSlider(title, lo, hi, .2 if key == 'size' else 1)
            c.setValue(default)
            c.default_value = default
            c.changed.connect(self.update_tool)
            self.retouch_controls[key] = c
            layout.addWidget(c)
        self.retouch_hint = QLabel(tr('每个笔划都可单独隐藏、删除或撤销。'))
        self.retouch_hint.setWordWrap(True)
        self.retouch_hint.setObjectName('subtle')
        layout.addWidget(self.retouch_hint)
        self.retouch_list = QListWidget()
        self.retouch_list.setMaximumHeight(235)
        self.retouch_list.currentRowChanged.connect(self.select_retouch)
        layout.addWidget(self.retouch_list)
        self.retouch_enabled = QCheckBox(tr('启用选中笔划'))
        self.retouch_enabled.toggled.connect(self.toggle_retouch)
        layout.addWidget(self.retouch_enabled)
        row = QHBoxLayout()
        row.addWidget(self.button(tr('删除选中'), self.remove_retouch))
        row.addWidget(self.button(tr('清除取样点'), self.clear_clone_source))
        layout.addLayout(row)
        layout.addStretch()
        self.canvas.clone_sampled.connect(self.clone_sampled)

    def clear_clone_source(self):
        self.canvas.clone_source = self.canvas.clone_offset = None
        self.canvas.update()
        self.retouch_hint.setText(tr('{OPTION} + 单击设置新的图章取样点。', OPTION=host.OPTION))

    def clone_sampled(self, point):
        self.retouch_hint.setText(tr('已取样：在目标区域涂抹。') if point else tr('请先按住 {OPTION} 并单击画面，设置取样来源。', OPTION=host.OPTION))

    def add_retouch(self, stroke):
        if len(self.edits['retouch']) >= 500:
            self.statusBar().showMessage(tr('修复笔划达到 500 个上限，请删除不需要的笔划。'))
            return
        op = {k: copy.deepcopy(v) for k, v in stroke.items() if k != 'erase'}
        op.update(enabled=True, feather=self.retouch_controls['feather'].spin.value(),
                  opacity=self.retouch_controls['opacity'].spin.value())
        self.edits['retouch'].append(op)
        self.refresh_retouch(len(self.edits['retouch']) - 1)
        self.changed()
        self.commit()

    def refresh_retouch(self, row=None):
        if row is None:
            row = self.retouch_list.currentRow()
        self.retouch_list.blockSignals(True)
        self.retouch_list.clear()
        self.retouch_list.addItems([f'{i + 1:02d}  {tr("污点修复") if op["kind"] == "heal" else tr("仿制图章")}{"" if op["enabled"] else tr(" · 已隐藏")}'
                                  for i, op in enumerate(self.edits['retouch'])])
        self.retouch_list.setCurrentRow(min(row, len(self.edits['retouch']) - 1))
        self.retouch_list.blockSignals(False)
        self.select_retouch()

    def select_retouch(self, *_):
        i = self.retouch_list.currentRow()
        valid = 0 <= i < len(self.edits['retouch'])
        self.retouch_enabled.blockSignals(True)
        self.retouch_enabled.setEnabled(valid)
        self.retouch_enabled.setChecked(valid and self.edits['retouch'][i]['enabled'])
        self.retouch_enabled.blockSignals(False)

    def toggle_retouch(self, enabled):
        i = self.retouch_list.currentRow()
        if 0 <= i < len(self.edits['retouch']) and not self.refreshing:
            self.edits['retouch'][i]['enabled'] = enabled
            self.refresh_retouch(i)
            self.changed()
            self.commit()

    def remove_retouch(self):
        i = self.retouch_list.currentRow()
        if 0 <= i < len(self.edits['retouch']):
            self.edits['retouch'].pop(i)
            self.refresh_retouch(i)
            self.changed()
            self.commit()
