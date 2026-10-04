"""Watermark settings with a live export preview; no image coordinates are changed."""
import base64
import copy
from pathlib import Path
from PySide6.QtCore import Qt,Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QFormLayout,QWidget,QLabel,
    QCheckBox,QComboBox,QSpinBox,QPushButton,QFileDialog,QDialogButtonBox)
from . import watermark,engine
from .widgets import qimage
from .i18n import tr


class WatermarkEditor(QWidget):
    changed=Signal(dict)

    def __init__(self,owner,compact=False):
        super().__init__(owner)
        self.owner=owner;self.refreshing=True;self.compact=compact
        self.settings=copy.deepcopy(owner.edits['watermark'])
        root=QVBoxLayout(self);root.setContentsMargins(0,0,0,0)
        columns=QVBoxLayout() if compact else QHBoxLayout();root.addLayout(columns)
        self.preview=QLabel(tr('导入照片后可预览水印边框'))
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if compact:self.preview.setMinimumSize(240,200);self.preview.setMaximumHeight(280)
        else:self.preview.setMinimumSize(540,500);columns.addWidget(self.preview,1)
        self.form_box=QWidget();form=QFormLayout(self.form_box);columns.addWidget(self.form_box)
        self.enabled=QCheckBox(tr('导出时添加水印边框'));self.enabled.setChecked(self.settings['enabled']);form.addRow(self.enabled)
        self.preset=QComboBox()
        for key,(title,*_) in watermark.PRESETS.items():self.preset.addItem(title,key)
        self.preset.setCurrentIndex(list(watermark.PRESETS).index(self.settings['preset']));form.addRow(tr('边框预设'),self.preset)
        side_row=QHBoxLayout();self.sides={}
        for key,title in [('top',tr('上')),('bottom',tr('下')),('left',tr('左')),('right',tr('右'))]:
            control=QCheckBox(title);control.setChecked(key in self.settings['sides']);self.sides[key]=control;side_row.addWidget(control)
        form.addRow(tr('显示位置'),side_row)
        self.size=QSpinBox();self.size.setRange(8,35);self.size.setSuffix(' %');self.size.setValue(round(self.settings['size']));form.addRow(tr('边框宽度'),self.size)
        self.fields={}
        field_row=None
        for i,(key,title) in enumerate(watermark.LABELS.items()):
            control=QCheckBox(title);control.setChecked(key in self.settings['fields']);self.fields[key]=control
            if compact:
                if i%2==0:field_row=QHBoxLayout();form.addRow(field_row)
                field_row.addWidget(control,1)
            else:form.addRow(control)
        for key,title in [('camera_logo',tr('机身标志')),('lens_logo',tr('镜头标志'))]:
            row=QHBoxLayout();add=QPushButton(tr('导入 PNG'));clear=QPushButton(tr('清除'))
            add.clicked.connect(lambda checked=False,k=key:self.choose_logo(k));clear.clicked.connect(lambda checked=False,k=key:self.clear_logo(k))
            row.addWidget(add);row.addWidget(clear);form.addRow(title,row)
        helptext=QLabel(tr('内置通用字体品牌名称；图形标志请导入自己有权使用的 PNG。\n缺失 EXIF 会显示“未记录”，不采用文件修改时间。\n水印仅在最终导出时添加，AI 副本保留无边框图像。'))
        helptext.setWordWrap(True);form.addRow(helptext)
        self.status=QLabel();self.status.setWordWrap(True);root.addWidget(self.status)
        if compact:columns.addWidget(self.preview)
        for c in [self.enabled,*self.sides.values(),*self.fields.values()]:c.toggled.connect(self.update_preview)
        self.preset.currentIndexChanged.connect(self.update_preview);self.size.valueChanged.connect(self.update_preview)
        active=owner.source is not None
        self.form_box.setEnabled(active)
        self.refreshing=False;self.update_preview()

    def values(self):
        return dict(self.settings,enabled=self.enabled.isChecked(),preset=self.preset.currentData(),
                    sides=[k for k,c in self.sides.items() if c.isChecked()],size=self.size.value(),
                    fields=[k for k,c in self.fields.items() if c.isChecked()])

    def update_preview(self,*_):
        if self.refreshing:return
        if self.owner.rendered is None:self.preview.setText(tr('导入照片后可预览水印边框'));return
        settings=self.values()
        if settings!=self.settings:
            self.settings=copy.deepcopy(settings);self.changed.emit(copy.deepcopy(settings))
        image=engine.crop_rotate(self.owner.rendered,self.owner.edits)
        image=engine.resize_limit(image,750)
        try:
            result=watermark.apply(image,self.values(),self.owner.info.get('photo',{}))
            pix=QPixmap.fromImage(qimage(result))
            self.preview.setPixmap(pix.scaled(350 if self.compact else 580,260 if self.compact else 550,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
            self.status.setText(tr('只在保留画面之外添加边框，照片内容不会被遮挡。'))
        except Exception as e:self.status.setText(str(e))

    def choose_logo(self,key):
        path,_=QFileDialog.getOpenFileName(self,tr('选择有权使用的标志'),'',tr('PNG 标志 (*.png)'))
        if not path:return
        try:
            p=Path(path)
            if p.stat().st_size>2_000_000:raise ValueError(tr('标志请使用小于 2MB 的 PNG。'))
            data=base64.b64encode(p.read_bytes()).decode('ascii');watermark.logo(data)
            self.settings[key]=data;self.changed.emit(self.values());self.update_preview()
        except Exception as e:self.status.setText(str(e))

    def clear_logo(self,key):
        self.settings[key]='';self.changed.emit(self.values());self.update_preview()

    def set_settings(self,settings):
        self.refreshing=True;self.settings=copy.deepcopy(settings)
        self.enabled.setChecked(settings['enabled'])
        self.preset.setCurrentIndex(list(watermark.PRESETS).index(settings['preset']))
        self.size.setValue(round(settings['size']))
        for key,control in self.sides.items():control.setChecked(key in settings['sides'])
        for key,control in self.fields.items():control.setChecked(key in settings['fields'])
        self.form_box.setEnabled(self.owner.source is not None)
        self.refreshing=False;self.update_preview()


class WatermarkDialog(QDialog):
    """Optional large preview, sharing the same editor as the Watermark tab."""
    def __init__(self,owner):
        super().__init__(owner);self.owner=owner
        self.setWindowTitle(tr('水印与边框'));self.resize(1000,650)
        root=QVBoxLayout(self);self.editor=WatermarkEditor(owner)
        self.form_box=self.editor.form_box;root.addWidget(self.editor)
        self.buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr('应用'))
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr('取消'))
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(owner.source is not None)
        self.buttons.accepted.connect(self.accept);self.buttons.rejected.connect(self.reject);root.addWidget(self.buttons)
        self.settings=copy.deepcopy(owner.edits['watermark'])

    def accept(self):
        if self.owner.source is None:return
        self.settings=watermark.validate(self.editor.values());super().accept()
