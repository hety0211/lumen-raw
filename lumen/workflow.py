"""Browseable menus, watermark preferences and exclusive AI task scheduling."""
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import QDialog,QMessageBox
from . import i18n, host
from .ai_dialog import EnhancementDialog
from .scheduler import Activity as A
from .watermark_dialog import WatermarkDialog
from .i18n import tr


class WorkflowMixin:
    def build_menus(self):
        menu=self.menuBar()
        file=menu.addMenu(tr('文件'));file.addAction(tr('导入照片…'),self.open_file)
        self.menu_save=file.addAction(tr('保存当前工程'),self.save)
        self.menu_album=file.addAction(tr('保存选片集'),self.save_album)
        self.menu_export=file.addAction(tr('导出成片…'),self.export)
        photo=menu.addMenu(tr('照片'))
        photo.addAction(tr('AI 超分辨率…'),lambda:self.open_ai('super'))
        photo.addAction(tr('AI 去杂色…'),lambda:self.open_ai('denoise'))
        group=photo.addMenu(tr('合成'))
        from .merge import METHODS
        for kind,name in METHODS.items():group.addAction(name+'…',lambda checked=False,key=kind:self.open_merge(key))
        view=menu.addMenu(tr('查看'))
        for i in range(self.tabs.count()):view.addAction(self.tabs.tabText(i),lambda checked=False,index=i:self.tabs.setCurrentIndex(index))
        self.build_library_menu(menu)
        self.build_language_menu(menu)
        helpmenu=menu.addMenu(tr('帮助'))
        helpmenu.addAction(tr('AI 助手接入（MCP）…'),self.open_agent_dialog)
        helpmenu.addAction(tr('命令面板…')+'\t'+host.keys('Ctrl+K'),self.open_palette)
        helpmenu.addAction(tr('支持的 RAW 格式'),lambda:QMessageBox.information(self,tr('RAW 支持'),tr('Sony ARW / SR2 / SRF\nCanon CRW / CR2 / CR3\nNikon NEF / NRW\nFujifilm RAF（含 X-Trans）\nPanasonic RW2 / RAW\nDNG\n\n具体机型和压缩方式以内置 LibRaw 支持为准。')))

    def build_language_menu(self,menu):
        """1.5.1: interface language; always also labelled in English so anyone can find it."""
        title=tr('语言')
        language=menu.addMenu(title if title=='Language' else f'{title} / Language')
        group=QActionGroup(self);group.setExclusive(True)
        for code,name in i18n.LANGUAGES:
            action=language.addAction(name,lambda checked=False,c=code:self.change_language(c))
            action.setCheckable(True);action.setChecked(code==i18n.language());group.addAction(action)
        self.language_menu=language

    def change_language(self,code):
        if code==i18n.language():return
        try:i18n.save_language(code)
        except OSError as exc:return self.error(tr('无法保存语言设置：')+str(exc))
        # Ask in the chosen language, and in the current one when the user may not read it.
        lines=[i18n.tr_in(code,'界面语言将在重新启动 LUMEN RAW 后切换为 {name}。现在重新启动吗？',name=i18n.language_name(code))]
        current=tr('界面语言将在重新启动 LUMEN RAW 后切换为 {name}。现在重新启动吗？',name=i18n.language_name(code))
        if current not in lines:lines.append(current)
        box=QMessageBox(self);box.setWindowTitle('LUMEN RAW');box.setIcon(QMessageBox.Icon.Question)
        box.setText('\n\n'.join(lines))
        restart=box.addButton(i18n.tr_in(code,'立即重新启动'),QMessageBox.ButtonRole.AcceptRole)
        box.addButton(i18n.tr_in(code,'稍后'),QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is restart:
            self.restart_requested=True
            self.close()
            if not self.closing and self.isVisible():
                self.restart_requested=False  # closing was cancelled (unsaved edits or work in progress)
        else:
            self.statusBar().showMessage(i18n.tr_in(code,'语言设置已保存，下次启动时生效。'))

    def configure_watermark(self):
        if self.work.busy(A.AI, A.EXPORTING, A.LOADING):return
        self.watermark_dialog=WatermarkDialog(self)
        if self.watermark_dialog.exec()==QDialog.DialogCode.Accepted:
            self.edits['watermark']=self.watermark_dialog.settings
            self.watermark_editor.set_settings(self.edits['watermark'])
            self.changed();self.commit()

    def open_ai(self,kind):
        if not self.work.can_start(A.AI):return
        self.ai_dialog=EnhancementDialog(self,kind)
        self.ai_dialog.exec()

    def open_merge(self,kind):
        if not self.work.can_start(A.AI):return
        from .merge_dialog import MergeDialog
        self.merge_dialog=MergeDialog(self,kind,self.selected_paths())
        self.merge_dialog.exec()

    def resume_after_ai(self):
        self.next_film_thumbnails()
        if self.source is not None:
            self.timer.start();self.detail_timer.start(220)
            self.update_thumbnails()

    def refresh_access(self):
        # The tab bar and scroll areas remain operable even without an image.
        active=self.source is not None and not self.work.busy(A.LOADING, A.AI)
        self.tabs.setEnabled(True)
        for index in range(self.tabs.count()):self.tabs.widget(index).widget().setEnabled(active)
        self.backend_combo.setEnabled(active)
        self.menu_save.setEnabled(active);self.menu_export.setEnabled(active and not self.work.busy(A.EXPORTING))
        self.menu_album.setEnabled((bool(self.documents) or self.library_structure_dirty) and not self.work.busy(A.AI))
