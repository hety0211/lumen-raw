"""Export and a real full-resolution neural-enhancement crop preview."""
import copy
import threading
import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QFormLayout,
    QComboBox, QSpinBox, QLineEdit, QPushButton, QFileDialog, QDialogButtonBox)
from . import engine, host
from .widgets import qimage
from .i18n import tr


class ExportDialog(QDialog):
    def __init__(self, parent, source, enhance=False):
        super().__init__(parent)
        self.owner = parent
        self.source_path = source
        self.preview_cancel = threading.Event()
        self.finished.connect(lambda _: self.preview_cancel.set())
        self.setWindowTitle(tr('增强与导出'))
        self.setMinimumWidth(620)
        root = QVBoxLayout(self)
        title = QLabel(tr('增强与导出  /  ENHANCE'))
        title.setObjectName('section')
        root.addWidget(title)
        text = QLabel(tr('从全尺寸原片生成成片。2× 为宽高各翻倍，像素数量增加至 4 倍。'))
        text.setWordWrap(True)
        root.addWidget(text)
        dng_note=QLabel(tr('DNG 保存已应用调整的 16 位线性 RGB；原始 ARW 传感器数据继续保留在原片中。'))
        dng_note.setWordWrap(True)
        dng_note.setObjectName('subtle')
        root.addWidget(dng_note)
        form = QFormLayout()
        self.format = QComboBox()
        self.format.addItems(['JPEG · 8-bit sRGB', 'PNG · 8-bit sRGB', 'TIFF · 16-bit sRGB', tr('DNG · 16-bit 线性成片')])
        self.scale = QComboBox()
        self.scale.addItems([tr('原始尺寸'), tr('2× 超分辨率'), tr('4× 超分辨率')])
        self.method = QComboBox()
        self.method.addItems([tr('AI 增强 · Real-ESRGAN（已内置）'), tr('快速增强 · Lanczos + 反投影'), tr('自定义 ONNX 模型')])
        self.quality = QSpinBox()
        self.quality.setRange(50, 100)
        self.quality.setValue(95)
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText(tr('动态尺寸 NCHW / RGB / float32，倍率与所选一致'))
        self.browse = QPushButton(tr('选择 ONNX'))
        self.browse.clicked.connect(self.choose_model)
        row = QHBoxLayout()
        row.addWidget(self.model_path)
        row.addWidget(self.browse)
        for name, widget in [(tr('输出格式'), self.format), (tr('输出尺寸'), self.scale), (tr('增强算法'), self.method), (tr('JPEG 质量'), self.quality)]:
            form.addRow(name, widget)
        form.addRow(tr('自定义模型'), row)
        root.addLayout(form)
        self.preview_button = QPushButton(tr('预览中央细节 · 读取全尺寸原片'))
        self.preview_button.clicked.connect(self.preview)
        root.addWidget(self.preview_button)
        previews = QHBoxLayout()
        self.before = QLabel(tr('增强前'))
        self.after = QLabel(tr('增强后'))
        for label in (self.before, self.after):
            label.setFixedSize(256, 256)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet('background: #121713; border: 1px solid #3c4738;')
            previews.addWidget(label)
        root.addLayout(previews)
        self.status = QLabel(tr('预览为成片中央 128 × 128 原始像素；左侧为普通放大，右侧为增强。'))
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        note = QLabel(tr('内置模型可离线运行。') + host.GPU_NOTE + tr('\n这是 RGB 图像增强，不是 Adobe 的 RAW 增强算法；细小纹理可能被重建。\n最多输出 4 亿像素。大图增强可能需要数分钟，支持分块进度与取消。'))
        note.setWordWrap(True)
        note.setObjectName('subtle')
        root.addWidget(note)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText(tr('选择保存位置'))
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr('取消'))
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)
        self.method.currentIndexChanged.connect(self.refresh_options)
        self.scale.currentIndexChanged.connect(self.refresh_options)
        if enhance:
            self.scale.setCurrentIndex(1)
            self.format.setCurrentIndex(2)
        self.refresh_options()
        if not enhance:
            self.setWindowTitle(tr('导出成片'))
            self.setMinimumWidth(520)
            title.setText(tr('导出成片  /  EXPORT'))
            text.setText(tr('输出当前成片。AI 超分和 AI 去杂色请在「照片」菜单中单独处理。'))
            dng_note.setText(tr('可输出 JPEG、PNG、16 位 TIFF 或线性 DNG。水印设置会在这里应用。'))
            for row_index in (1,2,4):form.setRowVisible(row_index,False)
            for widget in (self.preview_button,self.before,self.after,self.status,note):widget.hide()
            self.adjustSize()

    def refresh_options(self, *_):
        active = self.scale.currentIndex() > 0
        self.method.setEnabled(active)
        custom = active and self.method.currentIndex() == 2
        self.model_path.setEnabled(custom)
        self.browse.setEnabled(custom)
        self.preview_button.setEnabled(active)
        self.before.setText(tr('增强前'))
        self.after.setText(tr('增强后'))
        self.status.setText(tr('点击预览，检查当前倍率和算法的中央细节。'))

    def model_choice(self):
        return [':builtin:', '', self.model_path.text().strip()][self.method.currentIndex()]

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, tr('选择超分模型'), '', tr('ONNX 模型 (*.onnx)'))
        if path:
            self.model_path.setText(path)

    def accept(self):
        if self.scale.currentIndex() and self.method.currentIndex() == 2 and not self.model_path.text().strip():
            self.status.setText(tr('请先选择自定义 ONNX 模型。'))
            return
        super().accept()

    def preview(self):
        scale = [1, 2, 4][self.scale.currentIndex()]
        path = self.model_choice()
        if self.method.currentIndex() == 2 and not path:
            self.status.setText(tr('请先选择自定义 ONNX 模型。'))
            return
        edits = copy.deepcopy(self.owner.edits)
        cuda = self.owner.backend_combo.currentIndex() == 0
        self.status.setText(tr('正在读取原片并计算中央细节…'))
        for widget in (self.preview_button, self.method, self.scale, self.buttons.button(QDialogButtonBox.StandardButton.Save)):
            widget.setEnabled(False)
        def work():
            source, _ = engine.load_image(self.source_path, preview_limit=None)
            if self.preview_cancel.is_set():
                raise InterruptedError(tr('预览已取消'))
            # 1.4.0: render only the central patch of the finished frame, not the whole photograph.
            w, h = engine.display_size(edits, (source.shape[1], source.shape[0]))
            rect = (max(0, w//2-64), max(0, h//2-64), min(w, w//2+64), min(h, h//2+64))
            patch = np.clip(engine.render_display(source, edits, engine.Backend('auto' if cuda else 'cpu'), rect, final=True), 0, 1)
            del source
            result, name = engine.super_resolve(patch, scale, path, cuda, cancel=self.preview_cancel)
            return patch, result, name
        def enable():
            self.scale.setEnabled(True)
            self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(True)
            self.refresh_options()
        def success(result):
            if self.preview_cancel.is_set():
                return
            enable()
            before, after, name = result
            self.before.setPixmap(QPixmap.fromImage(qimage(cv2.resize(before, (256,256), interpolation=cv2.INTER_CUBIC))))
            self.after.setPixmap(QPixmap.fromImage(qimage(cv2.resize(after, (256,256), interpolation=cv2.INTER_AREA))))
            self.status.setText(tr('中央细节 · 左：普通放大 / 右：{scale}× 增强\n{name}', scale=scale, name=name))
        def failed(text):
            if not self.preview_cancel.is_set():
                enable()
                self.status.setText(tr('预览失败：') + text)
        self.owner.job(work, success, failed)
