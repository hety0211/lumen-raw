"""Exclusive AI enhancement workspace; completed DNG copies join the filmstrip."""
import copy
import os
import threading
from pathlib import Path
import cv2
from PySide6.QtCore import Qt,QTimer,Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,QWidget,
    QComboBox,QSpinBox,QPushButton,QLineEdit,QFileDialog,QProgressBar)
from . import engine,denoise,model,restoration,large_image,performance
from .scheduler import Activity as A
from . import ai_worker, compute
from .widgets import qimage
from .i18n import tr


def reserve_copy(folder,stem,kind):
    folder=Path(folder).expanduser().resolve()
    if not folder.is_dir():raise ValueError(tr('副本目录不存在，请选择有效文件夹。'))
    suffix=tr('增强') if kind=='super' else tr('去杂色')
    for index in range(1,10000):
        target=folder/f'{stem}-{suffix}{"" if index==1 else "-"+str(index)}.dng'
        try:
            fd=os.open(target,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            os.close(fd)
            return target
        except FileExistsError:continue
    raise ValueError(tr('同名副本过多，请更换目录。'))


class EnhancementDialog(QDialog):
    progressed=Signal(int,str)

    def __init__(self,owner,kind='super'):
        super().__init__(owner)
        if kind not in ('super','denoise'):raise ValueError(tr('未知增强类型'))
        self.owner=owner;self.kind=kind;self.busy=False;self.close_after=False
        self.cancel_event=threading.Event();self.output_path=None
        self.setWindowTitle(tr('AI 超分辨率') if kind=='super' else tr('AI 去杂色'))
        self.resize(850,720)
        root=QVBoxLayout(self)
        title=QLabel(tr('AI 超分辨率  /  SUPER RESOLUTION') if kind=='super' else tr('AI 去杂色  /  DENOISE'))
        title.setObjectName('section');root.addWidget(title)
        root.addWidget(QLabel(tr('左：处理前细节    /    右：AI 处理后的细节')))
        previews=QHBoxLayout();self.before=QLabel(tr('原始细节'));self.after=QLabel(tr('处理后细节'))
        for label in (self.before,self.after):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter);label.setFixedSize(384,300)
            label.setStyleSheet('background:#111712;border:1px solid #3f4c3b');previews.addWidget(label)
        root.addLayout(previews)
        self.options=QWidget();form=QFormLayout(self.options);root.addWidget(self.options)
        self.method=QComboBox()
        self.method.addItems([tr('高画质 · Real-ESRGAN x4plus / RRDB'),tr('快速 · Real-ESRGAN compact')] if kind=='super' else [tr('高画质 · DRUNet 细节保留'),tr('真实噪声 · NAFNet SIDD'),tr('快速 · FFDNet 自适应降噪')])
        form.addRow(tr('算法'),self.method)
        self.scale=QComboBox();self.scale.addItems([tr('2× · 像素数量 4 倍'),tr('4× · 像素数量 16 倍')])
        self.amount=QSpinBox();self.amount.setRange(0,100);self.amount.setValue(35)
        if kind=='denoise':self.method.currentIndexChanged.connect(lambda index:self.amount.setValue(70 if index==1 else 35))
        form.addRow(tr('增强倍率') if kind=='super' else tr('去杂色强度'),self.scale if kind=='super' else self.amount)
        if kind=='denoise':form.addRow(QLabel(tr('DRUNet / FFDNet：35 为噪声基准；NAFNet：70 为混合强度。')))
        resource=QLabel(tr('高画质耗时更长，建议先预览。最多 4 亿像素；线程上限 32，本机使用 {threads}。', threads=performance.THREADS))
        resource.setWordWrap(True);form.addRow(resource)
        self.folder=QLineEdit(str(Path(owner.source_path).parent) if owner.source_path else '')
        browse=QPushButton(tr('选择目录'));browse.clicked.connect(self.choose_folder)
        row=QHBoxLayout();row.addWidget(self.folder);row.addWidget(browse);form.addRow(tr('副本保存目录'),row)
        info=QLabel(tr('处理当前成片并生成 16 位线性 DNG 副本，自动加入下方选片栏。\n当前调色与裁切会写入副本；原片保留。水印在最终导出时添加。\n运算期间优先执行此任务，暂停新的预览和缩略图计算，可取消。'))
        info.setWordWrap(True);root.addWidget(info)
        self.progress=QProgressBar();self.progress.setRange(0,100);root.addWidget(self.progress)
        self.status=QLabel(tr('先预览中央 192 × 192 原图像素，再生成完整副本。'));self.status.setWordWrap(True);root.addWidget(self.status)
        buttons=QHBoxLayout();self.preview_button=QPushButton(tr('预览中央细节'));self.run_button=QPushButton(tr('生成增强副本') if kind=='super' else tr('生成去杂色副本'))
        self.run_button.setObjectName('primary');self.cancel_button=QPushButton(tr('关闭'))
        self.preview_button.clicked.connect(lambda:self.start(True));self.run_button.clicked.connect(lambda:self.start(False));self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.preview_button);buttons.addStretch();buttons.addWidget(self.cancel_button);buttons.addWidget(self.run_button);root.addLayout(buttons)
        self.progressed.connect(self.on_progress)
        active=owner.source is not None
        self.options.setEnabled(active);self.preview_button.setEnabled(active);self.run_button.setEnabled(active)

    def choose_folder(self):
        path=QFileDialog.getExistingDirectory(self,tr('选择副本保存目录'),self.folder.text())
        if path:self.folder.setText(path)

    def on_progress(self,value,text):
        self.progress.setValue(value);self.status.setText(text)

    def start(self,preview=False):
        w=self.owner
        if self.busy or w.source is None or not w.work.can_start(A.AI):return
        if not preview and not Path(self.folder.text()).is_dir():
            self.status.setText(tr('请选择有效的副本保存目录。'));return
        self.busy=True;w.work.begin(A.AI);self.cancel_event.clear()
        self.options.setEnabled(False);self.preview_button.setEnabled(False);self.run_button.setEnabled(False)
        self.cancel_button.setText(tr('取消运算'));self.progress.setValue(0)
        w.timer.stop();w.detail_timer.stop();w.cancel_detail()
        self.status.setText(tr('等待已开始的任务结束，然后优先进行 AI 运算…'))
        self._request=dict(preview=preview,path=w.source_path,edits=copy.deepcopy(w.edits),
                           photo=copy.deepcopy(w.info.get('photo',{})),scale=[2,4][self.scale.currentIndex()],
                           amount=self.amount.value(),method=self.method.currentIndex(),cuda=w.backend_combo.currentIndex()==0,folder=self.folder.text())
        self.wait_for_idle()

    def wait_for_idle(self):
        if self.cancel_event.is_set():self.finished_work();return
        if self.owner.jobs:
            QTimer.singleShot(30,self.wait_for_idle);return
        w=self.owner;request=self._request
        # Free full-frame caches before allocating the potentially 4x larger copy.
        w.reset_resolution();w.update_display()
        def work():
            r=request;target=None
            try:
                self.progressed.emit(2,tr('正在读取全尺寸原片…'))
                source,_=engine.load_image(r['path'],None)
                if self.cancel_event.is_set():raise InterruptedError(tr('已取消'))
                self.progressed.emit(8,tr('正在应用当前编辑…'))
                rgb=engine.process(source,r['edits'],engine.Backend('auto' if r['cuda'] else 'cpu'));del source
                if not r['preview']:large_image.validate_size((rgb.shape[0]*(r['scale'] if self.kind=='super' else 1),rgb.shape[1]*(r['scale'] if self.kind=='super' else 1)))
                noise_reference=denoise.noise_level(rgb) if self.kind=='denoise' else None
                region=None
                if r['preview']:
                    h,w=rgb.shape[:2];region=(max(0,w//2-96),max(0,h//2-96),min(w,w//2+96),min(h,h//2+96))
                    before=rgb[region[1]:region[3],region[0]:region[2]].copy()
                self.progressed.emit(15,tr('正在执行 AI 运算…'))
                ai_worker.set_status_listener(lambda text:self.progressed.emit(15,text))
                def progress(done,total):self.progressed.emit(15+round(78*done/total),tr('正在计算 · 分块 {done} / {total}', done=done, total=total))
                if self.kind=='super':
                    if r['method']==0:output,backend=restoration.super_resolution(rgb,r['scale'],r['cuda'],progress,self.cancel_event,region=region)
                    else:
                        source=before if region else rgb
                        output,backend=engine.super_resolve(source,r['scale'],':builtin:',r['cuda'],progress,self.cancel_event)
                else:
                    algorithm=(restoration.drunet_denoise,restoration.denoise,denoise.process)[r['method']]
                    options={'noise_reference':noise_reference}
                    if r['method']<2:options['region']=region
                    source=before if region and r['method']==2 else rgb
                    output,backend=algorithm(source,r['amount'],r['cuda'],progress,self.cancel_event,**options)
                if self.cancel_event.is_set():raise InterruptedError(tr('已取消'))
                ai_worker.set_status_listener(None)
                notice=compute.state.snapshot()[3]
                if notice in ai_worker.CRASH_NOTICES:backend+='\n'+notice
                if r['preview']:return dict(before=before,after=output,backend=backend)
                del rgb
                self.progressed.emit(96,tr('正在保存 DNG 副本…'))
                target=reserve_copy(r['folder'],Path(r['path']).stem,self.kind)
                engine.export_image(target,output,photo=r['photo'],provenance=dict(operation=self.kind,source=Path(r['path']).name,backend=backend,scale=r['scale'] if self.kind=='super' else 1,amount=r['amount'] if self.kind=='denoise' else 0))
                if self.cancel_event.is_set():raise InterruptedError(tr('已取消'))
                return dict(path=str(target),backend=backend,shape=output.shape)
            except BaseException:
                ai_worker.set_status_listener(None)
                if target is not None and target.exists():target.unlink()
                raise
        def ready(result):
            if self.cancel_event.is_set():
                if 'path' in result:Path(result['path']).unlink(missing_ok=True)
                self.finished_work()
                return
            self.finished_work()
            if request['preview']:
                for label,rgb in ((self.before,result['before']),(self.after,result['after'])):
                    image=cv2.resize(rgb,(300,300),interpolation=cv2.INTER_CUBIC if label is self.before or self.kind=='denoise' else cv2.INTER_AREA)
                    label.setPixmap(QPixmap.fromImage(qimage(image)))
                self.progress.setValue(100);self.status.setText(tr('中央细节 · ')+result['backend'])
            else:
                self.output_path=result['path'];w.add_documents([self.output_path])
                document=w.documents[self.output_path]
                document['edits']['watermark']=copy.deepcopy(request['edits']['watermark'])
                document['initialized']=True
                w.open_path(self.output_path)
                self.status.setText(tr('已生成副本 · ')+result['backend']);super(EnhancementDialog,self).accept()
        def fail(text):
            cancelled=self.cancel_event.is_set();self.finished_work()
            if not self.close_after:self.status.setText(tr('已取消，未生成副本。') if cancelled else tr('处理失败：')+text)
        w.job(work,ready,fail,priority=10)

    def finished_work(self):
        self.busy=False;self.owner.work.end(A.AI)
        self.options.setEnabled(True);self.preview_button.setEnabled(True);self.run_button.setEnabled(True);self.cancel_button.setText(tr('关闭'))
        self.owner.resume_after_ai()
        if self.close_after:super().reject()

    def reject(self):
        if self.busy:
            self.close_after=True;self.cancel_event.set();self.cancel_button.setEnabled(False)
            self.status.setText(tr('正在取消… 当前解码或保存阶段结束后关闭。'))
        else:super().reject()

    def closeEvent(self,event):
        if self.busy:self.reject();event.ignore()
        else:event.accept()
