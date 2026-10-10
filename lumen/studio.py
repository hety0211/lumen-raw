"""Photographic workflow panels: presets, versions, color grading and finishing."""
import copy
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QTabWidget, QComboBox, QCheckBox, QFileDialog, QInputDialog)
from . import model, engine, lens
from .widgets import AdjustSlider, ColorWheel, TabStrip, qimage
from .i18n import tr


def label(text, style='subtle'):
    widget = QLabel(text)
    widget.setObjectName(style)
    widget.setWordWrap(True)
    return widget


class StudioMixin:
    def init_studio(self):
        self.presets = model.builtin_presets()
        self.active_preset = None
        self.snapshots = []
        self.saved_snapshots = []
        self.thumbnail_token = 0
        self.wb_sampling = False
        self.grading_controls = {}
        self.effect_controls = {}

    def build_library(self):
        side = QWidget()
        side.setObjectName('library')
        side.setMinimumWidth(205)
        side.setMaximumWidth(280)
        layout = QVBoxLayout(side)
        layout.setContentsMargins(14, 17, 14, 14)
        layout.setSpacing(11)
        layout.addWidget(label(tr('导航器'), 'section'))
        self.navigator = QLabel('LUMEN  /  RAW')
        self.navigator.setObjectName('navigator')
        self.navigator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.navigator.setFixedHeight(116)
        self.navigator.setToolTip(tr('点击适应窗口'))
        self.navigator.mousePressEvent = lambda _: self.canvas.fit()
        layout.addWidget(self.navigator)
        self.library_info = label(tr('风光与旅行\n从一张原片开始'))
        layout.addWidget(self.library_info)
        library_tabs = QTabWidget()
        self.library_tabs = library_tabs
        self.library_strip = TabStrip(library_tabs)
        layout.addWidget(self.library_strip)
        layout.addWidget(library_tabs, 1)
        preset_page = QWidget()
        p = QVBoxLayout(preset_page)
        p.setContentsMargins(0, 12, 0, 0)
        p.addWidget(label(tr('旅行精选  /  TRAVEL COLLECTION')))
        self.preset_list = QListWidget()
        self.preset_list.setObjectName('presetList')
        self.preset_list.setIconSize(QSize(76, 52))
        self.preset_list.setSpacing(4)
        self.preset_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for preset in self.presets:
            item = QListWidgetItem(preset['name'] + '\n' + preset['subtitle'].split(' · ')[0])
            item.setSizeHint(QSize(180, 67))
            item.setToolTip(preset['subtitle'])
            self.preset_list.addItem(item)
        self.preset_list.itemClicked.connect(lambda item: self.choose_preset(self.preset_list.row(item)))
        p.addWidget(self.preset_list, 1)
        self.preset_amount = AdjustSlider(tr('预设强度'), 0, 100)
        self.preset_amount.setValue(100)
        self.preset_amount.default_value = 100
        self.preset_amount.changed.connect(self.change_preset_amount)
        self.preset_amount.committed.connect(self.commit)
        self.preset_amount.setEnabled(False)
        p.addWidget(self.preset_amount)
        row = QHBoxLayout()
        row.addWidget(self.button(tr('导入'), self.import_preset))
        row.addWidget(self.button(tr('保存预设'), self.export_preset))
        p.addLayout(row)
        library_tabs.addTab(preset_page, tr('预设'))
        library_tabs.addTab(self.build_natural_language(), tr('自然语言输入'))
        versions = QWidget()
        v = QVBoxLayout(versions)
        v.setContentsMargins(0, 12, 0, 0)
        v.addWidget(label(tr('保留不同处理版本，随时切换比较。快照随工程保存。')))
        self.snapshot_list = QListWidget()
        self.snapshot_list.itemDoubleClicked.connect(lambda _: self.restore_snapshot())
        v.addWidget(self.snapshot_list, 1)
        v.addWidget(self.button(tr('＋ 创建快照'), lambda: self.add_snapshot()))
        row = QHBoxLayout()
        row.addWidget(self.button(tr('恢复'), self.restore_snapshot))
        row.addWidget(self.button(tr('删除'), self.delete_snapshot))
        v.addLayout(row)
        v.addWidget(label(tr('双击快照恢复 · 最多 20 个')))
        library_tabs.addTab(versions, tr('快照'))
        self.library_strip.rebuild()
        return side

    def update_thumbnails(self):
        if self.source is None or self.ai_busy:
            return
        self.thumbnail_token += 1
        token = self.thumbnail_token
        source = engine.resize_limit(self.source, 180).copy()
        baseline = copy.deepcopy(self.edits['develop'])
        version = engine.process_version(self.edits)
        looks = copy.deepcopy(self.presets)
        def work():
            return [engine.process(source, model.apply_look(dict(model.recipe(), develop=baseline, process=version), p['look']))
                    for p in looks]
        def ready(images):
            if token != self.thumbnail_token:
                return
            for index, image in enumerate(images):
                self.preset_list.item(index).setIcon(QIcon(QPixmap.fromImage(qimage(image))))
        self.job(work, ready, lambda text: self.statusBar().showMessage(tr('预设缩略图暂时不可用：') + text))

    def update_navigator(self):
        if self.rendered is None:
            return
        pix = QPixmap.fromImage(qimage(engine.crop_rotate(self.rendered, self.edits)))
        self.navigator.setPixmap(pix.scaled(max(150, self.navigator.width()), 116,
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        if self.info:
            self.library_info.setText(tr('{format}  ·  {width} × {height}\n相机白平衡  /  无损编辑', format=self.info["format"], width=self.info["width"], height=self.info["height"]))

    def clear_preset_selection(self):
        self.active_preset = None
        self.preset_amount.setEnabled(False)
        self.preset_list.clearSelection()

    def choose_preset(self, index):
        if self.source is None or self.loading or not 0 <= index < len(self.presets):
            return
        self.commit()
        self.active_preset = index
        self.preset_amount.setEnabled(True)
        self.preset_amount.setValue(100)
        self.edits = model.apply_look(self.edits, self.presets[index]['look'])
        self.refresh()
        self.changed()
        self.commit()
        self.statusBar().showMessage(tr('已应用：') + self.presets[index]['name'] + tr(' · 保留蒙版、取样白平衡与构图'))

    def change_preset_amount(self, value):
        if self.active_preset is not None and not self.refreshing:
            self.edits = model.apply_look(self.edits, self.presets[self.active_preset]['look'], value)
            self.refresh()
            self.changed()

    def import_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, tr('导入 Lumen 预设'), '', tr('Lumen 预设 (*.lumenpreset)'))
        if not path:
            return
        try:
            preset = model.load_preset(path)
            self.presets.append(preset)
            item = QListWidgetItem(preset['name'] + '\nCUSTOM')
            item.setSizeHint(QSize(180, 67))
            self.preset_list.addItem(item)
            self.update_thumbnails()
            self.choose_preset(len(self.presets) - 1)
        except Exception as exc:
            self.error(str(exc))

    def export_preset(self):
        if self.source is None:
            return
        name, ok = QInputDialog.getText(self, tr('保存预设'), tr('为这个风格命名：'), text=tr('我的旅行风格'))
        if not ok or not name.strip():
            return
        path, _ = QFileDialog.getSaveFileName(self, tr('保存风格预设'), name + '.lumenpreset', tr('Lumen 预设 (*.lumenpreset)'))
        if path:
            try:
                model.save_preset(path, name.strip(), self.edits)
                self.statusBar().showMessage(tr('预设已保存，之后可导入复用。'))
            except Exception as exc:
                self.error(str(exc))

    def add_snapshot(self, name=None):
        if self.source is None:
            return
        if len(self.snapshots) >= 20:
            return self.error(tr('最多保留 20 个快照，请先删除不需要的版本。'))
        if name is None:
            name, ok = QInputDialog.getText(self, tr('创建快照'), tr('版本名称：'), text=tr('旅行版本 {v:02d}', v=len(self.snapshots) + 1))
            if not ok or not name.strip():
                return
        self.snapshots.append(dict(name=name[:80], edits=copy.deepcopy(self.edits)))
        self.refresh_snapshots()
        self.snapshot_list.setCurrentRow(len(self.snapshots) - 1)
        self.state_label.setText(tr('快照尚未保存'))

    def restore_snapshot(self):
        index = self.snapshot_list.currentRow()
        if 0 <= index < len(self.snapshots):
            self.commit()
            self.edits = copy.deepcopy(self.snapshots[index]['edits'])
            self.clear_preset_selection()
            self.current_mask = -1
            self.refresh()
            self.changed()
            self.commit()

    def delete_snapshot(self):
        index = self.snapshot_list.currentRow()
        if 0 <= index < len(self.snapshots):
            self.snapshots.pop(index)
            self.refresh_snapshots()
            self.state_label.setText(tr('快照尚未保存'))

    def refresh_snapshots(self):
        selected = self.snapshot_list.currentRow()
        self.snapshot_list.clear()
        for i, snap in enumerate(self.snapshots):
            self.snapshot_list.addItem(f'{i + 1:02d}  {snap["name"]}')
        self.snapshot_list.setCurrentRow(min(selected, len(self.snapshots) - 1))

    def build_grading(self):
        l = self.panel(tr('调色'))
        l.addWidget(label(tr('色彩分级'), 'section'))
        l.addWidget(label(tr('为暗部、中间调与高光分别赋予色彩。')))
        row = QHBoxLayout()
        row.setSpacing(0)
        self.wheels = {}
        for key, title in [('shadows', tr('暗部')), ('midtones', tr('中间调')), ('highlights', tr('高光'))]:
            wheel = ColorWheel(title)
            wheel.activated.connect(lambda k=key: self.select_grading_zone(k))
            wheel.changed.connect(lambda h, s, k=key: self.set_wheel(k, h, s))
            wheel.committed.connect(self.commit)
            self.wheels[key] = wheel
            row.addWidget(wheel)
        l.addLayout(row)
        self.grade_zone = QComboBox()
        self.grade_zone.addItems([tr('暗部'), tr('中间调'), tr('高光')])
        self.grade_zone.currentIndexChanged.connect(self.refresh_grading)
        l.addWidget(self.grade_zone)
        for index, title, maximum in [(0, tr('色相'), 360), (1, tr('饱和度'), 100)]:
            c = AdjustSlider(title, 0, maximum)
            c.changed.connect(lambda value, i=index: self.grading_value(i, value))
            c.committed.connect(self.commit)
            l.addWidget(c)
            self.grading_controls[index] = c
        self.grade_balance = AdjustSlider(tr('平衡 · 暗部 / 高光'))
        self.grade_balance.changed.connect(self.change_balance)
        self.grade_balance.committed.connect(self.commit)
        l.addWidget(self.grade_balance)
        l.addWidget(self.button(tr('重置色彩分级'), self.reset_grading))
        l.addWidget(label(tr('提示：给阴影加入轻微青蓝色、给高光加入暖金色，适合海岸和日落场景。双击色轮清除该区域颜色。')))
        l.addStretch()

    def select_grading_zone(self, zone):
        self.grade_zone.setCurrentIndex(('shadows', 'midtones', 'highlights').index(zone))
        self.refresh_grading()

    def set_wheel(self, zone, hue, saturation):
        if self.refreshing:
            return
        self.edits['grading'][zone] = [round(hue, 2), round(saturation, 2)]
        self.clear_preset_selection()
        self.refresh_grading()
        self.changed()

    def grading_value(self, index, value):
        if self.refreshing:
            return
        zone = ('shadows', 'midtones', 'highlights')[self.grade_zone.currentIndex()]
        self.edits['grading'][zone][index] = value
        self.clear_preset_selection()
        self.refresh_grading()
        self.changed()

    def change_balance(self, value):
        if not self.refreshing:
            self.edits['grading']['balance'] = value
            self.clear_preset_selection()
            self.changed()

    def refresh_grading(self, *_):
        if not hasattr(self, 'grade_zone'):
            return
        zone = ('shadows', 'midtones', 'highlights')[self.grade_zone.currentIndex()]
        for key, wheel in self.wheels.items():
            wheel.set_value(*self.edits['grading'][key])
            wheel.active = key == zone
        for index, control in self.grading_controls.items():
            control.setValue(self.edits['grading'][zone][index])
        self.grade_balance.setValue(self.edits['grading']['balance'])

    def reset_grading(self):
        self.edits['grading'] = model.grading()
        self.refresh_grading()
        self.clear_preset_selection()
        self.changed()
        self.commit()

    def build_effects(self):
        l = self.panel(tr('效果'))
        l.addWidget(label(tr('镜头氛围'), 'section'))
        l.addWidget(label(tr('暗角以当前裁切框为中心，支持压暗或提亮边缘。')))
        for key, title, low in [('vignette', tr('暗角 · 暗 / 亮'), -100), ('midpoint', tr('中点'), 0), ('feather', tr('羽化'), 0)]:
            c = AdjustSlider(title, low, 100)
            c.changed.connect(lambda value, k=key: self.effect_value(k, value))
            c.committed.connect(self.commit)
            l.addWidget(c)
            self.effect_controls[key] = c
            c.default_value = model.effects()[key]
        l.addWidget(label(tr('胶片颗粒'), 'section'))
        for key, title in [('grain', tr('数量')), ('grain_size', tr('大小'))]:
            c = AdjustSlider(title, 0, 100)
            c.changed.connect(lambda value, k=key: self.effect_value(k, value))
            c.committed.connect(self.commit)
            l.addWidget(c)
            self.effect_controls[key] = c
            c.default_value = model.effects()[key]
        l.addWidget(label(tr('颗粒使用固定随机场，相同参数重复预览与导出不会产生随机跳变。')))
        l.addStretch()

    def effect_value(self, key, value):
        if not self.refreshing:
            self.edits['effects'][key] = value
            self.clear_preset_selection()
            self.changed()

    def automatic(self):
        if self.source is None or self.loading:
            return
        self.commit()
        self.edits['adjustments'].update(engine.auto_tone_for(self.source, self.edits))
        self.clear_preset_selection()
        self.refresh()
        self.changed()
        self.commit()

    def toggle_wb(self, checked):
        self.wb_sampling = checked
        if checked:
            self.split_check.setChecked(False)
            self.statusBar().showMessage(tr('白平衡吸管：点击画面中有细节的中性灰／白色区域，Esc 取消。'))
        self.update_tool()

    def pick_wb(self, point):
        if self.source is None:
            return
        try:
            gains = engine.sample_white_balance(lens.correct(self.source, self.edits), point)
            self.commit()
            self.edits['white_balance']['kelvin'] = self.edits['white_balance']['camera_kelvin']
            self.edits['wb_gain'] = gains
            self.edits['adjustments'].update(temperature=0., tint=0.)
            self.clear_preset_selection()
            self.refresh()
            self.changed()
            self.commit()
            self.wb_button.setChecked(False)
            self.statusBar().showMessage(tr('已取样白平衡，可继续微调色温。'))
        except ValueError as exc:
            self.statusBar().showMessage(str(exc))

    def reset_wb(self):
        self.edits['white_balance'] = copy.deepcopy(self.info.get('white_balance', model.recipe()['white_balance']))
        self.edits['wb_gain'] = [1., 1., 1.]
        self.edits['adjustments'].update(temperature=0., tint=0.)
        self.clear_preset_selection()
        self.refresh()
        self.changed()
        self.commit()

    def monochrome_changed(self, checked):
        if not self.refreshing:
            self.edits['monochrome'] = checked
            self.clear_preset_selection()
            self.changed()
            self.commit()

    def straighten_changed(self, value):
        if not self.refreshing:
            self.edits['straighten'] = value
            self.final_view.setChecked(True)
            self.update_display()
            self.changed()

    def range_value(self, key, value):
        mask = self.selected_mask()
        if not mask or self.refreshing:
            return
        if key == 'falloff':
            mask['range_falloff'] = value
        else:
            lo, hi = mask['luminance_range']
            mask['luminance_range'] = [min(value, hi), hi] if key == 'low' else [lo, max(value, lo)]
            self.range_controls['low'].setValue(mask['luminance_range'][0])
            self.range_controls['high'].setValue(mask['luminance_range'][1])
        self.update_overlay()
        self.changed()

    def refresh_studio(self):
        self.mono_check.setChecked(self.edits['monochrome'])
        self.straighten.setValue(self.edits['straighten'])
        self.refresh_grading()
        for key, control in self.effect_controls.items():
            control.setValue(self.edits['effects'][key])
        self.refresh_snapshots()
        self.refresh_natural_language()
        self.update_navigator()
        self.auto_button.setEnabled(self.source is not None and not self.loading)
        self.preset_list.setEnabled(self.source is not None and not self.loading)
        self.split_check.setEnabled(self.source is not None)
        self.wb_button.setEnabled(self.source is not None and not self.loading)

    def toggle_clipping(self):
        checked = not (self.shadow_warning.isChecked() and self.highlight_warning.isChecked())
        self.shadow_warning.setChecked(checked)
        self.highlight_warning.setChecked(checked)
