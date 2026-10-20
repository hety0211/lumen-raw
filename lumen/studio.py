"""Photographic workflow panels: presets, versions, color grading and finishing."""
import colorsys
import copy
import zipfile
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QTabWidget, QComboBox, QCheckBox, QFileDialog, QInputDialog, QMenu, QMessageBox,
    QDialog, QDialogButtonBox, QPlainTextEdit, QPushButton)
from . import model, engine, lens, lightroom, lut
from . import presets as library_presets
from .widgets import AdjustSlider, ColorWheel, TabStrip, photo_icon, qimage
from .i18n import tr


#: Group filter entry of the built-in looks.
BUILTIN = ':builtin:'


def label(text, style='subtle'):
    widget = QLabel(text)
    widget.setObjectName(style)
    widget.setWordWrap(True)
    return widget


class StudioMixin:
    def init_studio(self):
        self.presets = library_presets.all_presets()
        self.preset_icons, self.preset_icons_source = {}, None
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
        # 1.7.0: built-in looks plus imported Lightroom presets, LUT presets and the user's own, by group.
        self.preset_group = QComboBox()
        self.preset_group.setToolTip(tr('按分组筛选预设'))
        self.preset_group.currentIndexChanged.connect(lambda _: self.fill_presets())
        row = QHBoxLayout()
        row.addWidget(self.preset_group, 1)
        # The photo's LUT: a menu button here, and its strength (titled with its name) once one is loaded.
        self.lut_button = QPushButton('LUT')
        lut_menu = QMenu(self.lut_button)
        self.lut_load_action = lut_menu.addAction(tr('载入 LUT…'), self.load_lut)
        self.lut_remove_action = lut_menu.addAction(tr('移除 LUT'), self.remove_lut)
        self.lut_button.setMenu(lut_menu)
        row.addWidget(self.lut_button)
        p.addLayout(row)
        self.preset_list = QListWidget()
        self.preset_list.setObjectName('presetList')
        self.preset_list.setIconSize(QSize(76, 52))
        self.preset_list.setSpacing(4)
        self.preset_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.preset_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.preset_list.customContextMenuRequested.connect(self.preset_menu)
        self.preset_list.itemClicked.connect(lambda item: self.choose_preset(item.data(Qt.ItemDataRole.UserRole)))
        p.addWidget(self.preset_list, 1)
        self.preset_amount = AdjustSlider(tr('预设强度'), 0, 100)
        self.preset_amount.setValue(100)
        self.preset_amount.default_value = 100
        self.preset_amount.changed.connect(self.change_preset_amount)
        self.preset_amount.committed.connect(self.commit)
        self.preset_amount.setEnabled(False)
        p.addWidget(self.preset_amount)
        row = QHBoxLayout()
        importer = self.button(tr('导入…'), self.import_preset)
        importer.setToolTip(tr('Lightroom 预设（.xmp / .lrtemplate）、预设包（.zip）、LUT（.cube）或 Lumen 预设'))
        row.addWidget(importer)
        row.addWidget(self.button(tr('保存预设'), self.export_preset))
        p.addLayout(row)
        self.lut_amount = AdjustSlider(tr('LUT 强度'), 0, 100)
        self.lut_amount.setValue(100)
        self.lut_amount.default_value = 100
        self.lut_amount.changed.connect(self.change_lut_amount)
        self.lut_amount.committed.connect(self.commit)
        self.lut_amount.setVisible(False)
        p.addWidget(self.lut_amount)
        self.fill_presets()
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
        """Render the presets shown in the list on a small copy of the photo (visible ones only:
        an imported pack may hold hundreds)."""
        if self.source is None or self.ai_busy:
            return
        self.thumbnail_token += 1
        token = self.thumbnail_token
        if self.preset_icons_source is not self.source:
            self.preset_icons, self.preset_icons_source = {}, self.source
        source = engine.resize_limit(self.source, 180).copy()
        baseline = copy.deepcopy(self.edits['develop'])
        version = engine.process_version(self.edits)
        rows = [self.preset_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.preset_list.count())]
        looks = [(i, copy.deepcopy(self.presets[i]['look'])) for i in rows if i not in self.preset_icons]
        if not looks:
            return
        presets = self.presets
        def work():
            neutral = dict(model.recipe(), develop=baseline, process=version)
            return [(i, engine.process(source, model.apply_look(neutral, look))) for i, look in looks]
        def ready(images):
            if token != self.thumbnail_token or presets is not self.presets:
                return
            for index, image in images:
                self.preset_icons[index] = photo_icon(qimage(image))
            self.show_preset_icons()
        self.job(work, ready, lambda text: self.statusBar().showMessage(tr('预设缩略图暂时不可用：') + text))

    def show_preset_icons(self):
        for row in range(self.preset_list.count()):
            item = self.preset_list.item(row)
            icon = self.preset_icons.get(item.data(Qt.ItemDataRole.UserRole))
            if icon is not None:
                item.setIcon(icon)

    def fill_presets(self):
        """The preset list for the chosen group (all, built-in, or one group of imported presets)."""
        chosen = self.preset_group.currentData() if self.preset_group.count() else ''
        self.preset_group.blockSignals(True)
        self.preset_group.clear()
        self.preset_group.addItem(tr('全部预设'), '')
        self.preset_group.addItem(tr('内置风格'), BUILTIN)
        for name in library_presets.groups(self.presets):
            self.preset_group.addItem(name, name)
        index = self.preset_group.findData(chosen)
        self.preset_group.setCurrentIndex(max(0, index))
        self.preset_group.blockSignals(False)
        chosen = self.preset_group.currentData()
        self.preset_list.clear()
        for index, preset in enumerate(self.presets):
            builtin = preset.get('source') == 'builtin'
            if chosen == BUILTIN and not builtin or chosen not in ('', BUILTIN) and preset.get('group') != chosen:
                continue
            item = QListWidgetItem(preset['name'] + '\n' + preset['subtitle'].split(' · ')[0])
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setSizeHint(QSize(180, 67))
            tip = [preset['subtitle']]
            if preset.get('notes'):
                tip += ['', tr('导入时未能套用：')] + lightroom.describe(preset['notes'])
            item.setToolTip('\n'.join(tip))
            self.preset_list.addItem(item)
            if index == self.active_preset:
                item.setSelected(True)
        self.show_preset_icons()
        if self.source is not None:
            self.update_thumbnails()

    def reload_presets(self):
        """Read the preset library again (after an import, here or through the command layer)."""
        self.presets = library_presets.all_presets()
        self.preset_icons = {}
        self.clear_preset_selection()
        self.fill_presets()

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
        if self.source is None or self.loading or index is None or not 0 <= index < len(self.presets):
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

    def preset_menu(self, position):
        item = self.preset_list.itemAt(position)
        if item is None:
            return
        index = item.data(Qt.ItemDataRole.UserRole)
        preset = self.presets[index]
        menu = QMenu(self)
        count = len(self.selected_paths()) if hasattr(self, 'filmstrip') else 0
        batch = menu.addAction(tr('套用到选中的 {count} 张照片', count=count), lambda: self.apply_preset_to_selection(index))
        batch.setEnabled(count > 0 and not self.loading)
        menu.addAction(tr('导出预设文件…'), lambda: self.export_preset_file(index))
        if preset.get('notes'):
            menu.addAction(tr('查看导入说明'), lambda: self.show_report(preset['name'], '\n'.join(lightroom.describe(preset['notes']))))
        if preset.get('path'):
            menu.addSeparator()
            menu.addAction(tr('删除预设'), lambda: self.delete_preset(index))
        menu.exec(self.preset_list.viewport().mapToGlobal(position))

    def apply_preset_to_selection(self, index):
        """Batch: one preset on every selected photo of the filmstrip (each keeps its framing,
        masks, retouching and sampled white balance)."""
        if self.loading or not 0 <= index < len(self.presets):
            return
        look = self.presets[index]['look']
        amount = self.preset_amount.spin.value() if self.active_preset == index else 100
        self.commit()
        self.stash_document()
        count = 0
        for item in self.filmstrip.selectedItems():
            path = item.data(Qt.ItemDataRole.UserRole)
            if path == self.source_path:
                self.edits = model.apply_look(self.edits, look, amount)
                self.refresh()
                self.changed()
                self.commit()
            else:
                document = self.documents[path]
                before = copy.deepcopy(document['edits'])
                document['edits'] = model.apply_look(before, look, amount)
                history = document.setdefault('history', model.History(before))
                history.push(document['edits'])
                item.setText(Path(path).name + ' •')
            count += 1
        self.statusBar().showMessage(tr('已将“{name}”套用到 {count} 张照片', name=self.presets[index]['name'], count=count))

    def import_preset(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr('导入预设或 LUT'), '',
            tr('预设与 LUT (*.lumenpreset *.xmp *.lrtemplate *.zip *.cube);;Lightroom 预设 (*.xmp *.lrtemplate);;'
               '预设包 (*.zip);;LUT (*.cube);;Lumen 预设 (*.lumenpreset)'))
        if paths:
            self.import_preset_files(paths)

    def import_preset_files(self, paths):
        self.statusBar().showMessage(tr('正在导入预设…'))
        def work():
            merged = dict(imported=[], skipped=[])
            for path in paths:
                try:
                    result = library_presets.import_path(path)
                except (OSError, ValueError, zipfile.BadZipFile) as exc:
                    result = dict(imported=[], skipped=[(Path(path).name, str(exc))])
                merged['imported'] += result['imported']
                merged['skipped'] += result['skipped']
            return merged
        def ready(result):
            self.reload_presets()
            imported = result['imported']
            if imported:
                group = self.preset_group.findData(imported[0]['group'])
                if group >= 0:
                    self.preset_group.setCurrentIndex(group)
            self.statusBar().showMessage(tr('已导入 {count} 个预设。', count=len(imported)))
            if result['skipped'] or any(item['notes'] for item in imported) or not imported:
                self.show_report(tr('导入预设'), library_presets.report(result))
            if len(imported) == 1 and self.source is not None:
                index = next((i for i, p in enumerate(self.presets) if p.get('path') == imported[0]['path']), None)
                self.choose_preset(index)
        self.job(work, ready)

    def show_report(self, title, text):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setMinimumSize(520, 380)
        layout = QVBoxLayout(dialog)
        note = label(tr('Lightroom 预设按近似换算：LUMEN 的滑块与 Adobe 不同，效果接近但不会完全一致。'))
        layout.addWidget(note)
        view = QPlainTextEdit(text)
        view.setReadOnly(True)
        layout.addWidget(view, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()

    def export_preset(self):
        """Save the current look into the preset list (group 我的预设)."""
        if self.source is None:
            return
        name, ok = QInputDialog.getText(self, tr('保存预设'), tr('为这个风格命名：'), text=tr('我的旅行风格'))
        if not ok or not name.strip():
            return
        try:
            library_presets.save(name.strip()[:80], model.extract_look(self.edits), tr('我的预设'))
        except (OSError, ValueError) as exc:
            return self.error(str(exc))
        self.reload_presets()
        self.statusBar().showMessage(tr('预设已保存到预设列表，可在其他照片上套用，或右键导出为文件分享。'))

    def export_preset_file(self, index):
        preset = self.presets[index]
        path, _ = QFileDialog.getSaveFileName(self, tr('导出预设文件'), lut.safe_name(preset['name']) + '.lumenpreset',
                                              tr('Lumen 预设 (*.lumenpreset)'))
        if not path:
            return
        try:
            source = preset.get('source', 'lumen')
            model.write_preset(path, preset['name'], preset['look'], preset.get('group', ''), preset.get('notes', ()),
                               'lumen' if source == 'builtin' else source)
            self.statusBar().showMessage(tr('预设已导出：') + path)
        except (OSError, ValueError) as exc:
            self.error(str(exc))

    def delete_preset(self, index):
        preset = self.presets[index]
        answer = QMessageBox.question(self, tr('删除预设'), tr('从预设列表删除“{name}”？', name=preset['name']))
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            library_presets.delete(preset)
        except (OSError, ValueError) as exc:
            return self.error(str(exc))
        self.reload_presets()

    # 1.7.0: a creative LUT on the current photo.

    def load_lut(self):
        if self.source is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, tr('载入 LUT'), '', tr('LUT (*.cube)'))
        if not path:
            return
        try:
            setting, notes = lut.load_cube(path)
        except (OSError, ValueError) as exc:
            return self.error(str(exc))
        self.commit()
        self.edits['lut'] = setting
        self.clear_preset_selection()
        self.refresh()
        self.changed()
        self.commit()
        self.statusBar().showMessage(tr('已载入 LUT：') + setting['name'] + (' · ' + '；'.join(notes) if notes else ''))

    def remove_lut(self):
        if self.source is None or not self.edits.get('lut'):
            return
        self.commit()
        self.edits['lut'] = None
        self.clear_preset_selection()
        self.refresh()
        self.changed()
        self.commit()

    def change_lut_amount(self, value):
        if not self.refreshing and self.edits.get('lut'):
            self.edits['lut'] = lut.with_amount(self.edits['lut'], value)
            self.clear_preset_selection()
            self.changed()

    def refresh_lut(self):
        setting = self.edits.get('lut')
        summary = lut.summary(setting)
        tip = tr('LUT 作用在色调映射与调色之后的显示 sRGB 上，位于暗角与颗粒之前；数据已随工程保存。')
        if summary:
            text = f'LUT · {summary["name"]}'
            title = self.lut_amount.title
            title.setText(title.fontMetrics().elidedText(text, Qt.TextElideMode.ElideMiddle, 130))
            self.lut_amount.setToolTip(f'{tr("LUT 强度")} · {summary["name"]} · {summary["size"]}\n{tip}')
        self.lut_button.setToolTip(tip)
        self.lut_amount.setVisible(bool(setting))
        self.lut_amount.setEnabled(bool(setting) and self.source is not None)
        self.lut_amount.setValue(setting['amount'] if setting else 100)
        self.lut_button.setEnabled(self.source is not None and not self.loading)
        self.lut_remove_action.setEnabled(bool(setting))

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
        # 1.7.0: an eyedropper: the colour clicked in the photo becomes the zone's hue.
        row = QHBoxLayout()
        row.addWidget(self.grade_zone, 1)
        self.grade_pick_button = QPushButton(tr('吸管取色'))
        self.grade_pick_button.setCheckable(True)
        self.grade_pick_button.setToolTip(tr('点选画面中的颜色，作为所选区域（暗部／中间调／高光）的色相；该区域尚未着色时按取样颜色的浓淡设置强度。Esc 取消。'))
        self.grade_pick_button.toggled.connect(self.grade_pick_mode)
        row.addWidget(self.grade_pick_button)
        l.addLayout(row)
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

    def grade_pick_mode(self, checked):
        if checked:
            if self.source is None:
                self.grade_pick_button.setChecked(False)
                return
            self.wb_button.setChecked(False)
            self.split_check.setChecked(False)
            self.statusBar().showMessage(tr('吸管取色：点击画面中的颜色，作为{zone}的色相。', zone=self.grade_zone.currentText()))
        self.update_tool()

    def pick_grade_color(self, point):
        """The colour-grading eyedropper (1.7.0): the hue of the edited photo around ``point``
        (frame coordinates 0–1) becomes the selected zone's hue."""
        if self.rendered is None or self.source is None:
            return
        rgb = self.rendered
        h, w = rgb.shape[:2]
        x, y = round(point[0] * (w - 1)), round(point[1] * (h - 1))
        r = max(1, round(min(h, w) / 300))
        sample = np.asarray(rgb[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1], np.float32).reshape(-1, 3).mean(axis=0)
        hue, saturation, value = colorsys.rgb_to_hsv(*map(float, np.clip(sample, 0, 1)))
        if saturation < .04 or value < .03:
            self.statusBar().showMessage(tr('取样处接近灰色，没有可用的色相；请点有颜色的地方。'))
            return
        zone = ('shadows', 'midtones', 'highlights')[self.grade_zone.currentIndex()]
        self.commit()
        strength = self.edits['grading'][zone][1] or round(min(40., max(8., saturation * 50)), 1)
        self.edits['grading'][zone] = [round(hue * 360, 1), strength]
        self.clear_preset_selection()
        self.refresh_grading()
        self.changed()
        self.commit()
        self.grade_pick_button.setChecked(False)
        self.statusBar().showMessage(tr('已从画面取色：{zone} · 色相 {hue}° · 强度 {strength}', zone=self.grade_zone.currentText(),
                                        hue=round(hue * 360), strength=f'{strength:g}'))

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
        self.refresh_lut()
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
