"""The 人像 panel (2.3.0): faces numbered on the photo, chosen alone, several or all, and the
retouching sliders of ``portrait`` for the chosen faces.

Faces are looked for the first time the panel is opened for a photo (and again with 重新检测);
what was found is stored in the recipe with the landmarks, so the photo never needs the models
again.  A face found again keeps its settings (``portrait.merge`` matches faces by box).
"""
from __future__ import annotations

import copy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QToolButton, QWidget

from . import engine, portrait
from .i18n import tr, N_
from .widgets import AdjustSlider, FlowLayout, Section

#: The sliders, by section: (key, name).
SKIN = (('smooth', N_('磨皮')), ('whiten', N_('美白')), ('acne', N_('祛痘')), ('circles', N_('去黑眼圈')))
FEATURES = (('eyes', N_('大眼')), ('slim', N_('瘦脸')), ('teeth', N_('白牙')))


class PortraitMixin:
    def init_portrait(self):
        self.face_selection = set()
        self.face_chips = []
        self.portrait_controls = {}
        #: Photos whose faces were looked for in this session (also when none were found).
        self.portrait_scanned = set()
        self.portrait_running = False
        #: The photo ``face_selection`` belongs to: another photo starts with all its faces chosen.
        self.face_photo = None

    def build_portrait(self):
        l = self.panel(tr('人像'), 'portrait')
        title = QLabel(tr('人像修饰  /  PORTRAIT'))
        title.setObjectName('section')
        l.addWidget(title)
        row = QHBoxLayout()
        self.detect_button = self.button(tr('重新检测人脸'), self.detect_faces)
        row.addWidget(self.detect_button)
        self.face_status = QLabel()
        self.face_status.setObjectName('subtle')
        self.face_status.setWordWrap(True)
        row.addWidget(self.face_status, 1)
        l.addLayout(row)
        chips = QWidget()
        chips.setObjectName('tabStrip')
        self.face_flow = FlowLayout(chips)
        l.addWidget(chips)
        self.face_numbers = QCheckBox(tr('在画面上显示人脸编号'))
        self.face_numbers.setChecked(True)
        self.face_numbers.toggled.connect(lambda _: self.update_faces())
        l.addWidget(self.face_numbers)
        for title, items in ((tr('皮肤'), SKIN), (tr('五官与牙齿'), FEATURES)):
            section = Section(title)
            l.addWidget(section)
            for key, name in items:
                control = AdjustSlider(tr(name), 0, 100)
                control.changed.connect(lambda value, k=key: self.portrait_value(k, value))
                control.committed.connect(self.commit)
                section.addWidget(control)
                self.portrait_controls[key] = control
        l.addWidget(self.button(tr('重置所选人脸'), self.reset_faces))
        note = QLabel(tr('点画面上的编号或上方的编号按钮选择人脸，Ctrl 或 Shift 点击可多选；滑块作用于所选的每张人脸。'
                         '人像修饰在调色、蒙版与裁切之前作用于原片，人脸位置随工程保存。') + '\n' +
                      tr('人脸检测：YuNet（MIT）；面部关键点：MediaPipe Face Mesh（Apache-2.0）。侧脸、遮挡较多或很小的人脸可能检测不到。'))
        note.setObjectName('subtle')
        note.setWordWrap(True)
        l.addWidget(note)
        l.addStretch()

    # -- faces

    def faces(self):
        return (self.edits.get('portrait') or {}).get('faces', [])

    def portrait_shown(self):
        """Opening the panel, or opening a photo with the panel open, looks for faces once per photo."""
        if self.source is not None and not self.faces() and self.source_path not in self.portrait_scanned:
            self.detect_faces()

    def detect_faces(self):
        if self.source is None or self.portrait_running or self.loading:
            return
        source, edits, path = self.source, copy.deepcopy(self.edits), self.source_path
        self.portrait_running = True
        self.detect_button.setEnabled(False)
        self.face_status.setText(tr('正在检测人脸…'))

        def work():
            return portrait.detect(engine.selection_view(source, edits))

        def done(found):
            self.portrait_running = False
            self.detect_button.setEnabled(True)
            self.portrait_scanned.add(path)
            if source is not self.source:
                # Another photo was opened meanwhile: it was not looked at, because this one ran.
                self.refresh_portrait()
                if self.on_tab('portrait'):
                    self.portrait_shown()
                return
            if found is None:
                self.face_status.setText(tr('缺少人脸模型，请在“帮助 → AI 模型”中下载。'))
                return
            self.commit()
            merged = portrait.merge(self.faces(), found)
            self.edits['portrait'] = dict(faces=merged)
            self.face_selection = set(range(len(merged)))
            self.refresh_portrait()
            self.update_faces()
            self.changed()
            self.commit(tr('检测人脸'))

        def failed(text):
            self.portrait_running = False
            self.detect_button.setEnabled(True)
            self.portrait_scanned.add(path)
            if source is not self.source:
                self.refresh_portrait()
                if self.on_tab('portrait'):
                    self.portrait_shown()
                return
            self.face_status.setText(tr('人脸检测失败：') + text)
        self.job(work, done, failed)

    def selected_faces(self):
        faces = self.faces()
        return [faces[i] for i in sorted(self.face_selection) if 0 <= i < len(faces)]

    def select_faces(self, indices):
        self.face_selection = set(indices)
        self.refresh_portrait()
        self.update_faces()

    def face_clicked(self, point, mode):
        index = portrait.face_at(self.faces(), point)
        if index is None:
            return
        if mode == 'add':
            self.select_faces(self.face_selection ^ {index})
        else:
            self.select_faces({index})

    def chip_clicked(self, index, checked):
        from PySide6.QtWidgets import QApplication
        modifiers = QApplication.keyboardModifiers()
        if index < 0:
            self.select_faces(range(len(self.faces())))
        elif modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
            self.select_faces(self.face_selection ^ {index})
        else:
            self.select_faces({index})

    def portrait_value(self, key, value):
        if self.refreshing or self.source is None:
            return
        chosen = self.selected_faces()
        if not chosen:
            self.statusBar().showMessage(tr('请先选择要修饰的人脸。'))
            return
        for face in chosen:
            face['settings'][key] = float(value)
        self.clear_preset_selection()
        self.changed()

    def reset_faces(self):
        chosen = self.selected_faces()
        if not chosen:
            return
        self.commit()
        for face in chosen:
            face['settings'] = portrait.settings()
        self.refresh_portrait()
        self.changed()
        self.commit(tr('重置人像'))

    # -- display

    def refresh_portrait(self):
        if not hasattr(self, 'face_flow'):
            return
        faces = self.faces()
        if self.face_photo != self.source_path:
            self.face_photo, self.face_selection = self.source_path, set()
        self.face_selection = {i for i in self.face_selection if i < len(faces)}
        if faces and not self.face_selection:
            self.face_selection = set(range(len(faces)))
        if len(self.face_chips) != len(faces) + (1 if faces else 0):
            for chip in self.face_chips:
                self.face_flow.removeWidget(chip)
                chip.deleteLater()
            self.face_chips = []
            for index in ([-1] + list(range(len(faces)))) if faces else []:
                chip = QToolButton()
                chip.setObjectName('faceChip')
                chip.setCheckable(True)
                chip.setText(tr('全部') if index < 0 else str(index + 1))
                chip.clicked.connect(lambda checked=False, i=index: self.chip_clicked(i, checked))
                self.face_flow.addWidget(chip)
                self.face_chips.append(chip)
        for chip, index in zip(self.face_chips, [-1] + list(range(len(faces)))):
            chip.setChecked(len(self.face_selection) == len(faces) if index < 0 else index in self.face_selection)
        chosen = self.selected_faces()
        old, self.refreshing = self.refreshing, True
        for key, control in self.portrait_controls.items():
            control.setEnabled(bool(chosen))
            control.setValue(chosen[0]['settings'].get(key, 0.) if chosen else 0.)
        self.refreshing = old
        self.detect_button.setEnabled(self.source is not None and not self.portrait_running)
        if self.portrait_running:
            return
        if self.source is None:
            self.face_status.setText('')
        elif faces:
            self.face_status.setText(tr('{count} 张人脸 · 已选 {chosen} 张', count=len(faces), chosen=len(chosen)))
        elif self.source_path in self.portrait_scanned:
            self.face_status.setText(tr('没有找到人脸'))
        else:
            self.face_status.setText(tr('打开此面板时自动检测人脸'))

    def update_faces(self):
        """Numbered face boxes on the canvas while the 人像 panel is open."""
        if not hasattr(self, 'face_numbers'):
            return
        shown = self.on_tab('portrait') and self.face_numbers.isChecked() and not self.crop_active and self.source is not None
        self.canvas.faces = [dict(box=face['box'], number=i + 1, selected=i in self.face_selection)
                             for i, face in enumerate(self.faces())] if shown else []
        self.canvas.update()
