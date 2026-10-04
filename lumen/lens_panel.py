"""Lens corrections in the crop tab and the manual lens choice (1.5.1).

The crop tab holds everything that changes the frame's geometry; profile corrections
for distortion, lateral chromatic aberration and vignetting sit below the crop tools.
Profiles come from the open lensfun database (see ``lens``).
"""
import copy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QVBoxLayout)

from . import i18n, lens
from .i18n import tr
from .widgets import AdjustSlider

#: settings.ini key: enable profile corrections for photos opened the first time.
AUTO_KEY = 'lens_auto_enable'


def auto_enable():
    return i18n.setting(AUTO_KEY, '0') == '1'


class LensDialog(QDialog):
    """Search the database for a lens when EXIF names none, or names it differently."""

    def __init__(self, parent, match, current=None):
        super().__init__(parent)
        self.setWindowTitle(tr('选择镜头配置文件'))
        self.setMinimumSize(560, 520)
        self.fitting, self.everything = lens.candidates(match)
        root = QVBoxLayout(self)
        title = QLabel(tr('镜头配置文件  /  LENSFUN'))
        title.setObjectName('section')
        root.addWidget(title)
        camera = (match or {}).get('camera') or tr('未知机身')
        hint = QLabel(tr('机身：{camera}。照片记录的镜头：{lens}', camera=camera,
                         lens=(match or {}).get('query') or tr('无')))
        hint.setObjectName('subtle')
        hint.setWordWrap(True)
        root.addWidget(hint)
        self.search = QLineEdit()
        self.search.setPlaceholderText(tr('搜索品牌或型号，例如 24-70 或 Sigma'))
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.fill)
        root.addWidget(self.search)
        self.compatible = QCheckBox(tr('只显示适合此机身卡口的镜头'))
        self.compatible.setChecked(self.fitting is not self.everything)
        self.compatible.setEnabled(self.fitting is not self.everything)
        self.compatible.toggled.connect(self.fill)
        root.addWidget(self.compatible)
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda _: self.accept())
        root.addWidget(self.list, 1)
        self.count = QLabel()
        self.count.setObjectName('subtle')
        root.addWidget(self.count)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr('使用此镜头'))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr('取消'))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.current = current
        self.fill()

    def fill(self, *_):
        terms = self.search.text().lower().split()
        source = self.fitting if self.compatible.isChecked() else self.everything
        self.list.clear()
        shown = 0
        for entry in source:
            text = f'{entry.label}   ·   {", ".join(entry.mounts)}'
            if terms and not all(t in text.lower() for t in terms):
                continue
            item = QListWidgetItem(text)
            have = entry.has()
            item.setToolTip(tr('畸变：{d} · 色差：{c} · 暗角：{v}', d=tr('有') if have['distortion'] else tr('无'),
                               c=tr('有') if have['tca'] else tr('无'), v=tr('有') if have['vignetting'] else tr('无')))
            item.setData(Qt.ItemDataRole.UserRole, entry.ident)
            self.list.addItem(item)
            if self.current and entry.ident == self.current:
                self.list.setCurrentItem(item)
            shown += 1
            if shown >= 800:
                break
        self.count.setText(tr('显示 {n} 个镜头', n=shown))

    def chosen(self):
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None


class LensMixin:
    def build_lens(self, layout):
        title = QLabel(tr('镜头校正  /  LENS'))
        title.setObjectName('section')
        layout.addWidget(title)
        self.lens_enable = QCheckBox(tr('启用配置文件校正'))
        self.lens_enable.toggled.connect(self.lens_enabled)
        layout.addWidget(self.lens_enable)
        self.lens_status = QLabel()
        self.lens_status.setObjectName('subtle')
        self.lens_status.setWordWrap(True)
        layout.addWidget(self.lens_status)
        row = QHBoxLayout()
        row.addWidget(self.button(tr('选择镜头…'), self.choose_lens))
        self.lens_auto_button = self.button(tr('按照片自动识别'), self.automatic_lens)
        row.addWidget(self.lens_auto_button)
        layout.addLayout(row)
        self.lens_controls = {}
        for key, title in (('distortion', tr('畸变校正 %')), ('vignetting', tr('暗角校正 %'))):
            control = AdjustSlider(title, 0, 200)
            control.default_value = 100
            control.changed.connect(lambda value, k=key: self.lens_amount(k, value))
            control.committed.connect(self.commit)
            self.lens_controls[key] = control
            layout.addWidget(control)
        self.lens_chromatic = QCheckBox(tr('去除横向色差'))
        self.lens_chromatic.setToolTip(tr('去除横向色差（边缘紫边／绿边）'))
        self.lens_chromatic.toggled.connect(self.lens_chromatic_changed)
        layout.addWidget(self.lens_chromatic)
        self.lens_auto = QCheckBox(tr('新打开的照片自动启用'))
        self.lens_auto.setChecked(auto_enable())
        self.lens_auto.toggled.connect(lambda checked: i18n.save_setting(AUTO_KEY, '1' if checked else '0'))
        layout.addWidget(self.lens_auto)
        note = QLabel(tr('配置文件来自开源 lensfun 镜头数据库（CC BY-SA 3.0），按照片的焦距、光圈与对焦距离插值。'
                         '畸变校正后自动放大以填满画面；裁切、蒙版与修复都基于校正后的画面。'))
        note.setObjectName('subtle')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.lens_reason = ''

    # -- photo state

    def lens_match(self):
        return (self.info or {}).get('lens_match')

    def prepare_lens(self, edits):
        """Solve the profile for the open photo; keeps an existing profile that already fits."""
        edits['lens'], reason = lens.prepare(edits.get('lens'), self.lens_match())
        return reason

    def init_lens_states(self, states, new_photo):
        """``states``: every recipe of the photo being opened (edits, saved edits, history)."""
        enable = new_photo and auto_enable()
        reason = ''
        for index, state in enumerate(states):
            if enable:
                state.setdefault('lens', lens.defaults())['enabled'] = True
            found = self.prepare_lens(state)
            if index == 0:
                reason = found
        self.lens_reason = reason

    # -- controls

    def lens_enabled(self, checked):
        if self.refreshing or self.source is None:
            return
        self.edits['lens']['enabled'] = checked
        self.lens_reason = self.prepare_lens(self.edits)
        self.refresh_lens()
        self.changed()
        self.commit()

    def lens_amount(self, key, value):
        if self.refreshing or self.source is None:
            return
        self.edits['lens'][key] = value
        self.changed()

    def lens_chromatic_changed(self, checked):
        if self.refreshing or self.source is None:
            return
        self.edits['lens']['chromatic'] = checked
        self.changed()
        self.commit()

    def choose_lens(self):
        if self.source is None:
            return
        settings = self.edits['lens']
        current = settings.get('manual') or (
            {k: settings['profile'][k] for k in ('maker', 'model')} if settings.get('profile') else None)
        try:
            dialog = LensDialog(self, self.lens_match(), current)
        except Exception as exc:
            return self.error(tr('镜头数据库不可用：') + str(exc))
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.chosen() is None:
            return
        self.commit()
        self.edits['lens']['manual'] = dialog.chosen()
        self.edits['lens']['enabled'] = True
        self.lens_reason = self.prepare_lens(self.edits)
        self.refresh_lens()
        self.changed()
        self.commit()

    def automatic_lens(self):
        if self.source is None:
            return
        self.commit()
        self.edits['lens']['manual'] = None
        self.lens_reason = self.prepare_lens(self.edits)
        self.refresh_lens()
        self.changed()
        self.commit()

    def refresh_lens(self):
        if not hasattr(self, 'lens_enable'):
            return
        settings = self.edits.get('lens') or lens.defaults()
        old = self.refreshing
        self.refreshing = True
        self.lens_enable.setChecked(bool(settings.get('enabled')))
        for key, control in self.lens_controls.items():
            control.setValue(settings.get(key, 100.))
        self.lens_chromatic.setChecked(bool(settings.get('chromatic', True)))
        self.refreshing = old
        profile = settings.get('profile')
        has = {key: bool(profile and profile.get(key)) for key in ('distortion', 'tca', 'vignetting')}
        on = bool(settings.get('enabled')) and profile is not None
        self.lens_controls['distortion'].setEnabled(on and has['distortion'])
        self.lens_controls['vignetting'].setEnabled(on and has['vignetting'])
        self.lens_chromatic.setEnabled(on and has['tca'])
        self.lens_auto_button.setEnabled(bool(settings.get('manual')))
        self.lens_status.setText(self.lens_description(settings))

    def lens_description(self, settings):
        if self.source is None:
            return tr('打开照片后按 EXIF 识别机身与镜头。')
        profile = settings.get('profile')
        match = self.lens_match() or {}
        if profile is None:
            reason = self.lens_reason or ('nolens' if match.get('query') else 'noexif')
            text = {'noexif': tr('照片没有记录镜头信息，可手动选择镜头。'),
                    'nolens': tr('数据库中没有找到“{lens}”，可手动选择镜头。', lens=match.get('query', '')),
                    'unknown': tr('所选镜头不在当前数据库中，请重新选择。'),
                    'focal': tr('照片没有记录焦距，变焦镜头无法插值。'),
                    'crop': tr('此镜头按更小的画幅标定，不适用于这台机身的画面。')}.get(reason, '')
            return text
        parts = [profile.get('label') or profile.get('model', '')]
        parts.append(tr('手动选择') if settings.get('manual') else tr('自动识别'))
        facts = [f'{profile["focal"]:g} mm']
        if profile.get('aperture'):
            facts.append(f'f/{profile["aperture"]:g}')
        facts.append(tr('裁切系数 {crop}', crop=f'{profile["crop"]:.2f}'))
        missing = [name for key, name in (('distortion', tr('畸变')), ('tca', tr('色差')), ('vignetting', tr('暗角')))
                   if not profile.get(key)]
        text = ' · '.join(parts) + '\n' + ' · '.join(facts)
        if missing:
            text += '\n' + tr('配置文件不含：{items}', items='、'.join(missing) if i18n.language().startswith('zh')
                                 else ', '.join(missing))
        return text

    def lens_states_for_document(self, document):
        states = [document['edits'], document['saved_edits']]
        if 'history' in document:
            states += document['history'].items
        return states

    def sync_lens(self, target):
        """Copy the correction switches (not the per-photo profile) to another photo's recipe."""
        current = self.edits.get('lens') or lens.defaults()
        settings = copy.deepcopy(target.get('lens') or lens.defaults())
        settings.update({k: current[k] for k in ('enabled', 'distortion', 'vignetting', 'chromatic')})
        target['lens'] = settings
