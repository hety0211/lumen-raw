"""Filmstrip, per-photo edit state, batch look synchronization and portable albums."""
import copy
import io
import json
import os
import threading
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QAbstractItemView, QFileDialog, QMessageBox,QMenu,QDialog,QProgressDialog)
from . import model, engine, host, catalog
from .scheduler import Activity as A
from .widgets import qimage
from .i18n import tr


def new_document(path):
    edits=model.recipe()
    return dict(path=str(Path(path).resolve()), edits=edits, saved_edits=copy.deepcopy(edits),
                snapshots=[],saved_snapshots=[],project='',initialized=False)


def dirty(document):
    return document['edits']!=document['saved_edits'] or document['snapshots']!=document['saved_snapshots']


def thumbnail(path):
    path=Path(path)
    if path.suffix.lower() in engine.RAW_EXTENSIONS and path.suffix.lower()!='.dng':
        import rawpy
        from . import previews
        with rawpy.imread(str(path)) as raw:
            preview=previews.libraw_preview(raw)
            if preview is None:
                # 1.4.1: e.g. Canon HDR PQ CR3 (HEVC preview); last resort, a small RAW decode.
                preview=previews.hevc_preview(path)
                preview=previews.orient(preview,raw.sizes.flip) if preview is not None else None
        if preview is None:
            rgb,info=engine.load_image(path,144)
            return engine.develop_view(rgb,dict(develop=info.get('develop',{}),process=2))
        im=Image.fromarray(np.ascontiguousarray(preview))
    else:
        try:
            im=Image.open(path)
        except Exception:
            return engine.develop_view(engine.load_image(path,144)[0],dict(process=2))
    with im:
        pic=ImageOps.exif_transpose(im).convert('RGB')
        pic.thumbnail((128,80))
        return np.asarray(pic,np.float32)/255


class LibraryMixin:
    def init_library(self):
        self.documents={}
        self.album_path=''
        self.thumbnail_queue=[]
        self.library_structure_dirty=False

    def build_filmstrip(self):
        box=QWidget()
        box.setObjectName('filmstrip')
        layout=QVBoxLayout(box)
        layout.setContentsMargins(14,6,14,6)
        layout.setSpacing(4)
        row=QHBoxLayout()
        self.library_count=QLabel(tr('选片集 · 0 张'))
        row.addWidget(self.library_count)
        row.addWidget(self.button(tr('＋ 导入多张'),self.open_file))
        row.addWidget(self.button(tr('保存选片集'),self.save_album))
        row.addSpacing(12)
        # 1.6.0: ratings, flags, colour labels and filters (rating.RatingMixin).
        self.build_rating_controls(row)
        row.addStretch()
        self.sync_button=self.button(tr('将当前调色套用到选中照片'),self.sync_look,True)
        row.addWidget(self.sync_button)
        layout.addLayout(row)
        self.filmstrip=QListWidget()
        self.filmstrip.setObjectName('filmstripList')
        self.filmstrip.setViewMode(QListWidget.ViewMode.IconMode)
        self.filmstrip.setFlow(QListWidget.Flow.LeftToRight)
        self.filmstrip.setWrapping(False)
        self.filmstrip.setMovement(QListWidget.Movement.Static)
        self.filmstrip.setIconSize(QSize(118,76))
        self.filmstrip.setGridSize(QSize(144,102))
        self.filmstrip.setFixedHeight(119)
        self.filmstrip.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.filmstrip.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.filmstrip.itemClicked.connect(self.film_clicked)
        self.filmstrip.itemActivated.connect(lambda item:self.open_path(item.data(Qt.ItemDataRole.UserRole)))
        self.filmstrip.itemSelectionChanged.connect(self.update_library_status)
        self.filmstrip.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.filmstrip.customContextMenuRequested.connect(self.show_film_menu)
        self.filmstrip.setToolTip(tr('{COMMAND} / Shift 多选 · 单击切换照片', COMMAND=host.COMMAND))
        self.install_rating_delegate()
        layout.addWidget(self.filmstrip)
        return box

    def add_documents(self, paths):
        added=[]
        for path in paths:
            path=str(Path(path).resolve())
            if path in self.documents:
                continue
            self.documents[path]=new_document(path)
            if self.album_path:self.library_structure_dirty=True
            item=QListWidgetItem(Path(path).name)
            item.setData(Qt.ItemDataRole.UserRole,path)
            item.setToolTip(path)
            self.filmstrip.addItem(item)
            self.thumbnail_queue.append(path)
            added.append(path)
        if added:
            self.refresh_ratings(added)
            # 1.6.0: ratings Lightroom, Bridge or a camera wrote into XMP, for photos the catalog does not know.
            self.read_xmp_async(added)
        self.update_library_status()
        self.next_film_thumbnails()

    def next_film_thumbnails(self):
        if not self.thumbnail_queue or not self.work.can_start(A.THUMBNAILS):
            return
        self.work.begin(A.THUMBNAILS)
        batch=self.thumbnail_queue[:4]
        del self.thumbnail_queue[:4]
        def work():
            images=[]
            for path in batch:
                try:images.append((path,thumbnail(path)))
                except Exception:images.append((path,None))
            return images
        def ready(images):
            self.work.end(A.THUMBNAILS)
            for path,rgb in images:
                if rgb is not None:
                    item=self.film_item(path)
                    if item:item.setIcon(QIcon(QPixmap.fromImage(qimage(rgb))))
            self.next_film_thumbnails()
        def fail(_):
            self.work.end(A.THUMBNAILS)
            self.next_film_thumbnails()
        self.job(work,ready,fail)

    def film_item(self,path):
        for i in range(self.filmstrip.count()):
            item=self.filmstrip.item(i)
            if item.data(Qt.ItemDataRole.UserRole)==path:return item
        return None

    def film_clicked(self,item):
        modifiers=QApplication.keyboardModifiers()
        if not modifiers & (Qt.KeyboardModifier.ControlModifier|Qt.KeyboardModifier.ShiftModifier):
            self.open_path(item.data(Qt.ItemDataRole.UserRole))

    def selected_paths(self):
        return [self.filmstrip.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.filmstrip.count()) if self.filmstrip.item(i).isSelected()]

    def film_context_menu(self):
        menu=QMenu(self);count=len(self.selected_paths())
        busy=self.work.busy()
        remove=menu.addAction(tr('删除（从图集移除）'),self.remove_selected)
        remove.setEnabled(count>0 and not busy and not self.jobs)
        group=menu.addMenu(tr('合成'))
        from .merge import METHODS
        for kind,name in METHODS.items():
            action=group.addAction(name+'…',lambda checked=False,key=kind:self.open_merge(key))
            action.setEnabled(2<=count<=32 and not busy)
        menu.addSeparator()
        export=menu.addAction(tr('批量导出 {count} 张…',count=count) if count>1 else tr('批量导出…'),self.batch_export)
        export.setEnabled(count>0 and not busy)
        # 1.6.0: ratings, flags, labels and XMP.
        menu.addSeparator()
        self.add_rating_actions(menu)
        menu.addAction(tr('从 XMP 读取评级'),self.read_selected_xmp).setEnabled(count>0)
        menu.addAction(tr('评级写入 XMP 附属文件'),self.write_selected_xmp).setEnabled(count>0)
        return menu

    def show_film_menu(self,point):
        item=self.filmstrip.itemAt(point)
        if item is not None and not item.isSelected():
            self.filmstrip.clearSelection();item.setSelected(True)
        menu=self.film_context_menu();menu.exec(self.filmstrip.viewport().mapToGlobal(point))

    def batch_export(self):
        """Export every selected photo with its own edits (1.4.1)."""
        paths=[p for p in self.selected_paths() if p in self.documents]
        if not paths or not self.work.can_start(A.EXPORTING):
            return
        self.stash_document()
        from .batch_export import BatchExportDialog,run
        dialog=BatchExportDialog(self,len(paths),Path(paths[0]).parent)
        if dialog.exec()!=QDialog.DialogCode.Accepted:
            return
        options=dialog.options()
        items=[(p,copy.deepcopy(self.documents[p]['edits']),bool(self.documents[p]['initialized'])) for p in paths]
        backend=engine.Backend('auto' if self.backend_combo.currentIndex()==0 else 'cpu')
        self.work.begin(A.EXPORTING)
        self.cancel_detail()
        self.export_cancel=threading.Event()
        self.export_dialog=QProgressDialog(tr('正在准备批量导出…'),tr('取消批量导出'),0,100,self)
        self.export_dialog.setWindowTitle(tr('批量导出'))
        self.export_dialog.setAutoClose(False)
        self.export_dialog.setAutoReset(False)
        self.export_dialog.setMinimumDuration(0)
        self.export_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.export_dialog.canceled.connect(self.export_cancel.set)
        self.export_dialog.show()
        self.export_button.setEnabled(False)
        self.open_button.setEnabled(False)
        self.statusBar().showMessage(tr('正在批量导出 {n_items} 张照片…', n_items=len(items)))
        cancel=self.export_cancel
        def finish():
            self.export_dialog.reset()
            self.work.end(A.EXPORTING)
            self.open_button.setEnabled(True)
            self.export_button.setEnabled(self.source is not None)
        def success(results):
            finish()
            done=[r for r in results if r[1]]
            failed=[r for r in results if not r[1]]
            skipped=len(items)-len(results)
            text=tr('已导出 {n_done} 张到：\n{folder}', n_done=len(done), folder=options["folder"])
            if failed:
                text+=tr('\n\n{n_failed} 张未能导出：\n', n_failed=len(failed))+'\n'.join(tr('{name}：{e}', name=Path(p).name, e=e) for p,_,e in failed[:8])
            if skipped:
                text+=tr('\n\n已取消，其余 {skipped} 张未导出。', skipped=skipped)
            self.statusBar().showMessage(tr('批量导出完成 · {n_done} 张成功', n_done=len(done))+(tr(' · {n_failed} 张失败', n_failed=len(failed)) if failed else '')+(tr(' · {skipped} 张已取消', skipped=skipped) if skipped else ''))
            QMessageBox.information(self,tr('批量导出'),text)
        def failed(text):
            finish()
            self.error(tr('批量导出失败：\n')+text)
        self.job(lambda:run(items,options,backend,self.export_progress.emit,cancel),success,failed)

    def clear_current_photo(self):
        self.timer.stop();self.detail_timer.stop();self.history_timer.stop();self.generation+=1;self.reset_resolution()
        self.source_path='';self.source=self.rendered=None;self.info={};self.project_path=''
        self.edits=model.recipe();self.saved_edits=copy.deepcopy(self.edits);self.snapshots=[];self.saved_snapshots=[];self.history=model.History(self.edits)
        self.current_mask=-1;self.clear_clone_source();self.pending=False
        self.canvas.image=self.canvas.before=self.canvas.clipping=None;self.canvas.crop=None;self.canvas.set_overlay(None);self.canvas.update()
        self.histogram.hist=None;self.histogram.update();self.navigator.clear();self.navigator.setText('LUMEN / RAW');self.library_info.clear()
        self.file_label.setText(tr('尚未打开原片'));self.file_label.setToolTip('');self.clear_preset_selection();self.refresh()

    def remove_selected(self):
        if self.work.busy() or self.jobs:return
        paths=self.selected_paths()
        if not paths:return
        self.stash_document()
        count=sum(dirty(self.documents[p]) for p in paths)
        if count:
            answer=QMessageBox.question(self,tr('从图集移除'),tr('所选照片中 {count} 张有未保存编辑。是否先保存选片集？\n磁盘原片不会删除。', count=count),QMessageBox.StandardButton.Save|QMessageBox.StandardButton.Discard|QMessageBox.StandardButton.Cancel)
            if answer==QMessageBox.StandardButton.Cancel:return
            if answer==QMessageBox.StandardButton.Save and not self.save_album():return
        removing_current=self.source_path in paths
        if removing_current:self.clear_current_photo()
        for path in paths:
            item=self.film_item(path)
            if item:self.filmstrip.takeItem(self.filmstrip.row(item))
            self.documents.pop(path,None)
        self.thumbnail_queue=[p for p in self.thumbnail_queue if p not in paths]
        self.library_structure_dirty=True;self.state_label.setText(tr('图集列表尚未保存'));self.update_library_status();self.refresh_access()
        if removing_current:
            available=next((p for p in self.documents if Path(p).is_file()),None)
            if available:self.open_path(available)
        self.statusBar().showMessage(tr('已从图集移除 {n_paths} 张 · 磁盘原片保留', n_paths=len(paths)))

    def stash_document(self):
        if not self.source_path or self.source is None:
            return
        document=self.documents.setdefault(self.source_path,new_document(self.source_path))
        document.update(edits=copy.deepcopy(self.edits),saved_edits=copy.deepcopy(self.saved_edits),
            snapshots=copy.deepcopy(self.snapshots),saved_snapshots=copy.deepcopy(self.saved_snapshots),
            project=self.project_path,history=copy.deepcopy(self.history),initialized=True,
            final=self.final_view.isChecked())

    def loaded_document(self,path,explicit_project=False):
        self.add_documents([path])
        document=self.documents[path]
        if not explicit_project:
            self.saved_edits=copy.deepcopy(document['saved_edits'])
            self.saved_snapshots=copy.deepcopy(document['saved_snapshots'])
            self.project_path=document['project']
            self.history=copy.deepcopy(document.get('history',model.History(self.edits)))
        self.final_view.setChecked(bool(document.get('final',bool(self.edits['crop']))) if not explicit_project else bool(self.edits['crop']))
        item=self.film_item(path)
        if item:
            if not item.isSelected():
                self.filmstrip.clearSelection()
                item.setSelected(True)
            self.filmstrip.scrollToItem(item)
        self.stash_document()
        self.update_library_status()

    def update_library_status(self):
        if not hasattr(self,'filmstrip'):return
        count=len(self.filmstrip.selectedItems())
        self.library_count.setText(tr('选片集 · {n_items} 张 / 已选 {count} 张', n_items=len(self.documents), count=count))
        self.sync_button.setEnabled(self.source is not None and count>0 and not self.loading)
        self.update_rating_bar()

    def sync_look(self):
        if self.source is None or self.loading:return
        self.commit()
        self.stash_document()
        look=model.extract_look(self.edits)
        count=0
        for item in self.filmstrip.selectedItems():
            path=item.data(Qt.ItemDataRole.UserRole)
            if path==self.source_path:continue
            document=self.documents[path]
            before=copy.deepcopy(document['edits'])
            document['edits']=model.apply_look(before,look)
            document['edits']['process']=self.edits.get('process',1)  # 1.6.0: the look's process version too
            self.sync_lens(document['edits'])
            history=document.setdefault('history',model.History(before))
            history.push(document['edits'])
            item.setText(Path(path).name+' •')
            count+=1
        self.statusBar().showMessage(tr('已套用到 {count} 张照片 · 保留各自裁切、蒙版、修复与相机显影基准', count=count))

    def save_album(self,checked=False,path=None):
        self.stash_document()
        if not self.documents and not self.library_structure_dirty:return False
        if path is None:
            path,_=QFileDialog.getSaveFileName(self,tr('保存全部照片的编辑'),self.album_path or tr('旅行选片.lumenalbum'),tr('Lumen 选片集 (*.lumenalbum)'))
        if not path:return False
        path=Path(path)
        if path.suffix.lower()!='.lumenalbum':path=path.with_suffix('.lumenalbum')
        try:
            documents=[]
            for document in self.documents.values():
                try:source=os.path.relpath(document['path'],path.parent.resolve())
                except ValueError:source=document['path']
                documents.append(dict(source=source,edits=model.validate(document['edits']),
                    snapshots=model.validate_snapshots(document['snapshots']),initialized=document['initialized'],
                    **self.album_ratings(document['path'])))
            data=json.dumps(dict(application='Lumen album',version=1,documents=documents),ensure_ascii=False)
            if len(data.encode('utf-8'))>128*1024*1024:raise ValueError(tr('选片集过大，请拆分保存。'))
            temp=path.with_suffix('.lumenalbum.tmp');temp.write_text(data,encoding='utf-8');temp.replace(path)
            self.album_path=str(path)
            self.library_structure_dirty=False
            for document in self.documents.values():
                document['saved_edits']=copy.deepcopy(document['edits'])
                document['saved_snapshots']=copy.deepcopy(document['snapshots'])
            self.saved_edits=copy.deepcopy(self.edits)
            self.saved_snapshots=copy.deepcopy(self.snapshots)
            self.state_label.setText(tr('选片集已保存'))
            self.statusBar().showMessage(tr('全部照片的编辑已保存 · ')+str(path))
            return True
        except Exception as exc:
            self.error(str(exc));return False

    def open_album(self,path):
        if not self.confirm_close_library():return
        try:
            path=Path(path)
            if path.stat().st_size>128*1024*1024:raise ValueError(tr('选片集过大。'))
            data=json.loads(path.read_text(encoding='utf-8'))
            if data.get('application')!='Lumen album' or not 0<=len(data['documents'])<=2000:
                raise ValueError(tr('无效选片集。'))
            records={};ratings={}
            for item in data['documents']:
                source=str((path.parent/item['source']).resolve())
                try:ratings[source]=catalog.validate({k:item[k] for k in ('rating','flag','label') if k in item})
                except ValueError:pass
                record=new_document(source)
                record.update(edits=model.validate(item['edits']),snapshots=model.validate_snapshots(item.get('snapshots',[])),initialized=bool(item.get('initialized',True)))
                record['saved_edits']=copy.deepcopy(record['edits']);record['saved_snapshots']=copy.deepcopy(record['snapshots'])
                records[source]=record
            available=[p for p in records if Path(p).exists()]
            if records and not available:raise ValueError(tr('选片集的原片均已移动或存储设备未连接。'))
            self.restore_album_ratings(ratings)
            self.source_path='';self.source=None
            self.documents={};self.filmstrip.clear();self.thumbnail_queue=[]
            self.add_documents(records)
            self.documents=records
            self.refresh_ratings()
            self.album_path=str(path)
            self.library_structure_dirty=False
            if available:self.open_path(available[0])
            else:self.clear_current_photo()
        except Exception as exc:self.error(str(exc))

    def confirm_close_library(self):
        self.stash_document()
        count=sum(dirty(d) for d in self.documents.values())
        if not count and not self.library_structure_dirty:return True
        message=tr('{count} 张照片的编辑尚未保存。是否保存全部编辑为选片集？', count=count) if count else tr('图集照片列表有变化。是否保存选片集？')
        answer=QMessageBox.question(self,tr('保存选片集'),message,
            QMessageBox.StandardButton.Save|QMessageBox.StandardButton.Discard|QMessageBox.StandardButton.Cancel)
        if answer==QMessageBox.StandardButton.Save:return self.save_album()
        return answer==QMessageBox.StandardButton.Discard
