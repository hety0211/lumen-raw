"""Automatic masks integrated with the existing non-destructive mask stack."""
import copy
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QLabel, QCheckBox, QComboBox
from .widgets import AdjustSlider
from .scheduler import Activity as A
from . import engine, model, selection, lens
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
        corrections=copy.deepcopy(self.edits)
        cuda=self.backend_combo.currentIndex()==0
        tolerance=self.color_tolerance.spin.value()
        self.statusBar().showMessage(tr('正在本地识别选区…'))
        def work():
            # Masks are drawn on the lens-corrected frame (1.5.1).
            rgb=engine.develop_view(lens.correct(source,corrections),corrections)
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
        # 1.6.0: process version of the recipe; projects from 1.x keep version 1 until upgraded.
        row=QHBoxLayout()
        self.process_label=QLabel();self.process_label.setObjectName('subtle');self.process_label.setWordWrap(True)
        row.addWidget(self.process_label,1)
        self.process_button=self.button(tr('升级'),self.upgrade_process)
        self.process_button.setToolTip(tr('改用处理版本 2：曝光、白平衡与亮部／暗部在线性光下计算，高光和饱和色彩保留更多层次；数值不变，画面会有变化，可撤销。'))
        row.addWidget(self.process_button)
        layout.addLayout(row)

    def refresh_process(self):
        version=self.edits.get('process',1)
        self.process_label.setText(tr('处理版本 2 · 场景参考') if version==2 else tr('处理版本 1 · 1.x 旧版渲染'))
        self.process_label.setToolTip(tr('处理版本 2（1.6.0）：RAW 解码不截断色域外颜色；曝光、白平衡与亮部／暗部在线性光下计算（亮部／暗部为边缘感知的局部色调映射），显影曲线作为色调映射，超出 sRGB 的颜色按 OkLCh 保持色相与明度压缩；饱和度、色彩混合器与色彩分级在 OkLCh 中计算。') if version==2
                                      else tr('1.6.0 之前保存的工程使用处理版本 1，画面与旧版完全一致。'))
        self.process_button.setVisible(version==1)
        self.process_button.setEnabled(self.source is not None)

    def upgrade_process(self):
        if self.source is None or self.edits.get('process',1)==2:return
        self.commit()
        self.edits['process']=2
        self.refresh();self.changed();self.commit()
        from . import host
        self.statusBar().showMessage(tr('已改用处理版本 2 · {v} 可撤销', v=host.keys('Ctrl+Z')))

    def change_develop_profile(self,index):
        if self.refreshing or self.source is None:return
        self.edits['develop']['mode']='camera' if index==0 else 'linear'
        if index==0 and self.edits['develop']['curve']==[[0.,0.],[1.,1.]]:
            self.edits['develop']=self.info.get('develop',self.edits['develop']).copy()
        self.changed();self.commit()
