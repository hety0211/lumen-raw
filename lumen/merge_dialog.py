"""Exclusive merge workspace; originals stay intact and DNG copies join the album."""
import copy
import os
import threading
from pathlib import Path
import cv2
from PySide6.QtCore import Qt,QTimer,Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,QWidget,
    QComboBox,QCheckBox,QPushButton,QLineEdit,QFileDialog,QProgressBar,QListWidget)
from . import engine,model,merge,large_image,host
from .scheduler import Activity as A
from .widgets import qimage
from .i18n import tr


def reference_index(infos,requested=None):
    if requested is not None:
        if not 0<=requested<len(infos):raise ValueError(tr('无效参考照片。'))
        return requested
    return max(range(len(infos)),key=lambda i:(infos[i]['width']*infos[i]['height'],-abs(i-(len(infos)-1)/2)))


def reserve_result(folder,stem,kind):
    folder=Path(folder).expanduser().resolve()
    if not folder.is_dir():raise ValueError(tr('请选择有效的副本目录。'))
    for index in range(1,10000):
        target=folder/f'{stem}-{merge.METHODS[kind]}{"" if index==1 else "-"+str(index)}.dng'
        try:
            descriptor=os.open(target,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(descriptor);return target
        except FileExistsError:continue
    raise ValueError(tr('同名副本过多，请更换目录。'))


def run_records(records,kind,requested_reference=None,preview=True,use_edits=False,ghost='medium',crop=True,align=True,progress=None,cancel=None):
    if not 2<=len(records)<=32:raise ValueError(tr('合成需选择 2–32 张照片。'))
    previews=[];infos=[]
    for i,record in enumerate(records):
        merge.notify(progress,int(10*i/len(records)),tr('读取照片 {v} / {n_records}：{name}', v=i+1, n_records=len(records), name=Path(record["path"]).name),cancel)
        source,info=engine.load_image(record['path'],1400);previews.append(source);infos.append(info)
    reference=reference_index(infos,requested_reference)
    if not preview and kind!='panorama':merge.check_memory(infos[reference]['width']*infos[reference]['height'],len(records))
    shared_develop=infos[reference].get('develop',model.recipe()['develop'])
    frames=[]
    for i,record in enumerate(records):
        merge.notify(progress,10+int(10*i/len(records)),tr('准备合成像素 {v} / {n_records}', v=i+1, n_records=len(records)),cancel)
        edits=copy.deepcopy(record['edits']) if use_edits else model.recipe()
        if not use_edits:edits['develop']=copy.deepcopy(shared_develop)
        elif not record.get('initialized',False):edits['develop']=copy.deepcopy(infos[i].get('develop',edits['develop']))
        source=previews[i] if preview else engine.load_image(record['path'],None)[0]
        frames.append(engine.process(source,edits,engine.Backend('cpu')))
    result,details=merge.merge_images(frames,kind,reference,ghost,crop,align,progress,cancel)
    return dict(image=result,reference=reference,photo=infos[reference].get('photo',{}),details=details,
                source=records[reference]['path'],watermark=copy.deepcopy(records[reference]['edits']['watermark']))


class MergeDialog(QDialog):
    progressed=Signal(int,str)

    def __init__(self,owner,kind='hdr',paths=None):
        super().__init__(owner)
        self.owner=owner;self.busy=False;self.close_after=False;self.output_path=None;self.cancel_event=threading.Event()
        owner.stash_document()
        self.records=[copy.deepcopy(owner.documents[p]) for p in (paths or []) if p in owner.documents]
        self.setWindowTitle(tr('照片合成'));self.resize(900,840)
        root=QVBoxLayout(self);title=QLabel(tr('照片合成  /  PHOTO MERGE'));title.setObjectName('section');root.addWidget(title)
        self.preview=QLabel(tr('选择合成方法，先预览构图与边缘，再生成完整副本。'));self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setFixedHeight(285);self.preview.setStyleSheet('background:#111712;border:1px solid #3f4c3b');root.addWidget(self.preview)
        self.options=QWidget();form=QFormLayout(self.options);root.addWidget(self.options)
        self.method=QComboBox()
        for key,name in merge.METHODS.items():self.method.addItem(name,key)
        self.method.setCurrentIndex(list(merge.METHODS).index(kind));form.addRow(tr('合成方法'),self.method)
        self.files=QListWidget();self.files.addItems([Path(r['path']).name for r in self.records]);self.files.setFixedHeight(70);form.addRow(tr('所选照片 · {n_items} 张', n_items=len(self.records)),self.files)
        self.reference=QComboBox();self.reference.addItem(tr('自动 · 像素最多（同尺寸取中间帧）'),None)
        for i,record in enumerate(self.records):self.reference.addItem(Path(record['path']).name,i)
        form.addRow(tr('参考照片 / 命名基准'),self.reference)
        self.ghost=QComboBox();self.ghost.addItems([tr('低 · 少量参考帧覆盖'),tr('中 · 平衡运动与细节'),tr('高 · 更大范围抑制重影')]);self.ghost.setCurrentIndex(1)
        self.ghost_label=QLabel(tr('HDR 去伪影'));form.addRow(self.ghost_label,self.ghost)
        self.align=QCheckBox(tr('自动对齐（校正轻微位移与焦点呼吸）'));self.align.setChecked(True);form.addRow(self.align)
        self.crop=QCheckBox(tr('自动裁去不完整边缘'));self.crop.setChecked(True);form.addRow(self.crop)
        self.use_edits=QCheckBox(tr('使用每张照片的当前调色、蒙版、修复和裁切'));form.addRow(self.use_edits)
        self.folder=QLineEdit(str(Path(self.records[0]['path']).parent) if self.records else '')
        browse=QPushButton(tr('选择目录'));browse.clicked.connect(self.choose_folder);row=QHBoxLayout();row.addWidget(self.folder);row.addWidget(browse);form.addRow(tr('副本保存目录'),row)
        self.info=QLabel();self.info.setWordWrap(True);root.addWidget(self.info)
        self.method.currentIndexChanged.connect(self.method_changed);self.method_changed()
        self.progress=QProgressBar();root.addWidget(self.progress)
        self.status=QLabel(tr('副本按「参考照片名-合成方法.dng」命名，同名自动编号；原片保留。'));self.status.setWordWrap(True);root.addWidget(self.status)
        buttons=QHBoxLayout();self.preview_button=QPushButton(tr('预览合成'));self.run_button=QPushButton(tr('生成合成副本'));self.run_button.setObjectName('primary');self.cancel_button=QPushButton(tr('关闭'))
        self.preview_button.clicked.connect(lambda:self.start(True));self.run_button.clicked.connect(lambda:self.start(False));self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.preview_button);buttons.addStretch();buttons.addWidget(self.cancel_button);buttons.addWidget(self.run_button);root.addLayout(buttons)
        self.progressed.connect(self.on_progress)
        enabled=2<=len(self.records)<=32;self.preview_button.setEnabled(enabled);self.run_button.setEnabled(enabled)
        if not enabled:self.status.setText(tr('请先在下方图集中按 {COMMAND} / Shift 选择 2–32 张照片，再打开合成窗口。', COMMAND=host.COMMAND))

    def method_changed(self):
        kind=self.method.currentData();hdr=kind=='hdr';self.ghost.setVisible(hdr);self.ghost_label.setVisible(hdr);self.align.setVisible(kind!='panorama')
        text={'hdr':tr('Mertens 曝光融合：将包围曝光合为自然成片。去伪影越高，运动区域越多采用参考帧；不能恢复参考帧已丢失的运动细节。'),
              'focus':tr('景深合成：选择不同焦平面中较清晰的细节。适合静物与稳定构图，移动物体、透明物体和严重焦点呼吸可能留有接缝。'),
              'panorama':tr('球面全景：需要约 30% 以上重叠、接近同一视点的照片。按原生细节拼接，画布超过 2 亿像素时等比缩小；较大视差可能产生接缝。')}[kind]
        self.info.setText(text+tr('\n默认统一使用参考帧显影基准，保留曝光差异；水印在最终导出时添加。'))

    def choose_folder(self):
        folder=QFileDialog.getExistingDirectory(self,tr('选择合成副本目录'),self.folder.text())
        if folder:self.folder.setText(folder)

    def on_progress(self,value,message):self.progress.setValue(value);self.status.setText(message)

    def start(self,preview=False):
        w=self.owner
        if self.busy or not w.work.can_start(A.AI) or not 2<=len(self.records)<=32:return
        if not preview and not Path(self.folder.text()).is_dir():self.status.setText(tr('请选择有效的副本保存目录。'));return
        self.request=dict(kind=self.method.currentData(),requested_reference=self.reference.currentData(),preview=preview,
                          use_edits=self.use_edits.isChecked(),ghost=('low','medium','high')[self.ghost.currentIndex()],crop=self.crop.isChecked(),align=self.align.isChecked())
        self.destination=self.folder.text();self.busy=True;w.work.begin(A.AI);self.cancel_event.clear()
        self.options.setEnabled(False);self.preview_button.setEnabled(False);self.run_button.setEnabled(False);self.cancel_button.setText(tr('取消合成'));self.progress.setValue(0)
        w.timer.stop();w.detail_timer.stop();w.cancel_detail();w.refresh_access();self.status.setText(tr('等待已有任务结束，随后优先执行合成…'));self.wait_for_idle()

    def wait_for_idle(self):
        if self.cancel_event.is_set():self.finished_work();return
        if self.owner.jobs:QTimer.singleShot(30,self.wait_for_idle);return
        self.owner.reset_resolution();self.owner.update_display()
        def work():
            target=None
            try:
                result=run_records(self.records,**self.request,progress=self.progressed.emit,cancel=self.cancel_event)
                if self.request['preview']:return result
                merge.check(self.cancel_event);target=reserve_result(self.destination,Path(result['source']).stem,self.request['kind'])
                self.progressed.emit(96,tr('正在保存 16 位线性 DNG…'))
                engine.export_image(target,result['image'],photo=result['photo'],provenance=dict(operation=self.request['kind'],sources=[Path(r['path']).name for r in self.records],reference=Path(result['source']).name,**result['details']))
                merge.check(self.cancel_event);result.pop('image');result['path']=str(target);return result
            except BaseException:
                if target is not None:target.unlink(missing_ok=True)
                raise
        def ready(result):
            if self.cancel_event.is_set():
                if 'path' in result:Path(result['path']).unlink(missing_ok=True)
                self.finished_work();return
            self.finished_work()
            if self.request['preview']:
                pix=QPixmap.fromImage(qimage(engine.resize_limit(result['image'],1400)))
                self.preview.setPixmap(pix.scaled(self.preview.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
                self.progress.setValue(100);self.status.setText(tr('预览完成 · 参考：{name} · 完整处理将使用原图像素', name=Path(result["source"]).name))
            else:
                self.output_path=result['path'];w=self.owner;w.add_documents([self.output_path]);record=w.documents[self.output_path]
                record['initialized']=True;record['edits']['watermark']=result['watermark'];w.open_path(self.output_path)
                self.status.setText(tr('合成副本已加入图集'));super(MergeDialog,self).accept()
        def fail(message):
            cancelled=self.cancel_event.is_set();self.finished_work()
            if not self.close_after:self.status.setText(tr('已取消，未生成副本。') if cancelled else tr('合成失败：')+message)
        self.owner.job(work,ready,fail,priority=10)

    def finished_work(self):
        self.busy=False;self.owner.work.end(A.AI);self.options.setEnabled(True);self.preview_button.setEnabled(True);self.run_button.setEnabled(True);self.cancel_button.setText(tr('关闭'))
        self.owner.refresh_access();self.owner.resume_after_ai()
        if self.close_after:super().reject()

    def reject(self):
        if self.busy:
            self.close_after=True;self.cancel_event.set();self.cancel_button.setEnabled(False)
            self.status.setText(tr('正在取消… 整图融合或保存阶段需要等待当前步骤结束。'))
        else:super().reject()

    def closeEvent(self,event):
        if self.busy:self.reject();event.ignore()
        else:event.accept()
