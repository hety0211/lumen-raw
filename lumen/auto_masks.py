"""Automatic masks integrated with the existing non-destructive mask stack."""
import copy
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QLabel, QCheckBox, QComboBox
from .widgets import AdjustSlider
from .scheduler import Activity as A
from . import engine, model, selection, develop, lens
from .i18n import tr


class AutoMaskMixin:
    def build_auto_masks(self,layout):
        title=QLabel(tr('自动选择  /  AI MASKS'))
        title.setObjectName('section')
        layout.addWidget(title)
        row=QHBoxLayout()
        self.ai_mask_buttons=[]
        for text,kind in [(tr('天空'),'sky'),(tr('人物'),'person'),(tr('背景'),'background')]:
            b=self.button(text,lambda checked=False,k=kind:self.create_auto_mask(k))
            row.addWidget(b);self.ai_mask_buttons.append(b)
        layout.addLayout(row)
        second=QHBoxLayout()
        for text,kind in [(tr('主体'),'subject'),(tr('近景'),'foreground')]:
            b=self.button(text,lambda checked=False,k=kind:self.create_auto_mask(k))
            second.addWidget(b);self.ai_mask_buttons.append(b)
        layout.addLayout(second)
        self.color_region_button=QPushButton(tr('点选相似颜色区域'))
        self.color_region_button.setCheckable(True)
        self.color_region_button.toggled.connect(self.color_selection_mode)
        layout.addWidget(self.color_region_button)
        self.color_tolerance=AdjustSlider(tr('相似颜色容差'),1,60)
        self.color_tolerance.default_value=18
        self.color_tolerance.setValue(18)
        layout.addWidget(self.color_tolerance)
        tip=QLabel(tr('主体识别显著对象；近景按相对深度估算较近区域。\n背景为主体的反选。自动蒙版可用画笔修整，结果随工程保存。'))
        tip.setWordWrap(True);tip.setObjectName('subtle');layout.addWidget(tip)
        self.refine_mask_check=QCheckBox(tr('画笔修整自动蒙版'))
        self.refine_mask_check.toggled.connect(self.update_tool)
        layout.addWidget(self.refine_mask_check)
        self.canvas.color_sampled.connect(self.pick_color_region)

    def color_selection_mode(self,checked):
        if checked:
            self.wb_button.setChecked(False)
            self.statusBar().showMessage(tr('点击画面中的颜色区域；容差越大，连通选区越宽。'))
        self.update_tool()

    def pick_color_region(self,point):
        self.create_auto_mask('color',point)
        self.color_region_button.setChecked(False)

    def create_auto_mask(self,kind,point=None):
        if self.source is None or not self.work.can_start(A.SELECTION):return
        if len(self.edits['masks'])>=32:
            return self.error(tr('最多支持 32 个蒙版。'))
        self.work.begin(A.SELECTION)
        for b in self.ai_mask_buttons:b.setEnabled(False)
        self.color_region_button.setEnabled(False)
        token=self.document_token
        source=self.source
        profile=self.edits['develop'].copy()
        corrections=copy.deepcopy(self.edits)
        cuda=self.backend_combo.currentIndex()==0
        tolerance=self.color_tolerance.spin.value()
        self.statusBar().showMessage(tr('正在本地识别选区…'))
        def work():
            # Masks are drawn on the lens-corrected frame (1.5.1).
            rgb=engine.to_srgb(develop.apply(lens.correct(source,corrections),profile)).clip(0,1)
            if kind=='color':
                alpha=selection.color_region(rgb,point,tolerance)
                provider=tr('连通颜色选择')
            else:alpha,provider=selection.automatic(rgb,kind,cuda)
            return alpha,provider
        def release():
            self.work.end(A.SELECTION)
            for b in self.ai_mask_buttons:b.setEnabled(True)
            self.color_region_button.setEnabled(True)
        def ready(result):
            release()
            if token!=self.document_token:return
            alpha,provider=result
            if float((alpha>.5).mean())<.0003:
                self.statusBar().showMessage(tr('没有识别到明确区域；可改用颜色点选或画笔蒙版。'))
                return
            self.commit()
            mask=model.new_mask(kind,len(self.edits['masks'])+1)
            mask.update(raster=selection.encode(alpha),feather=10.)
            self.edits['masks'].append(mask)
            self.current_mask=len(self.edits['masks'])-1
            self.refresh()
            self.changed();self.commit()
            self.statusBar().showMessage(tr('已创建{name} · {provider} · 可继续调整局部参数', name=mask["name"], provider=provider))
        def fail(text):
            release()
            if token==self.document_token:self.error(tr('自动选择失败：')+text)
        self.job(work,ready,fail)

    def build_develop_profile(self,layout):
        self.develop_combo=QComboBox()
        self.develop_combo.addItems([tr('相机参考显影'),tr('线性显影（旧版）')])
        self.develop_combo.currentIndexChanged.connect(self.change_develop_profile)
        layout.addWidget(self.develop_combo)
        self.develop_hint=QLabel(tr('RAW 默认使用相机预览作为亮度参考，曝光滑块仍从 0 EV 开始。'))
        self.develop_hint.setObjectName('subtle');self.develop_hint.setWordWrap(True)
        layout.addWidget(self.develop_hint)

    def change_develop_profile(self,index):
        if self.refreshing or self.source is None:return
        self.edits['develop']['mode']='camera' if index==0 else 'linear'
        if index==0 and self.edits['develop']['curve']==[[0.,0.],[1.,1.]]:
            self.edits['develop']=self.info.get('develop',self.edits['develop']).copy()
        self.changed();self.commit()
