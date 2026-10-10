"""Command palette (Ctrl+K, 1.6.0): every command of ``commands`` by id, run in this window.

The same commands serve ``lumen-cli`` and MCP agents; here they act on the open photo and each
edit is one undo step.  Commands with parameters ask for them as a JSON object.
"""
from __future__ import annotations

import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                               QVBoxLayout)

from . import commands, host
from .i18n import tr, N_

#: Interface titles of the commands (the summaries are written for agents, in English).
TITLES = {
    'app.status': N_('当前状态'), 'commands.list': N_('列出全部命令'), 'library.list': N_('列出图集照片'),
    'library.add': N_('导入照片或文件夹'), 'library.select': N_('选择照片'), 'library.rate': N_('设置星级'),
    'library.flag': N_('设置旗标'), 'library.label': N_('设置色标'), 'library.read_xmp': N_('从 XMP 读取评级'),
    'library.write_xmp': N_('评级写入 XMP 附属文件'), 'library.import_lightroom': N_('导入 Lightroom 目录中的评级'),
    'photo.open': N_('打开照片'), 'photo.info': N_('照片信息'), 'edit.get': N_('查看当前调整'),
    'edit.apply': N_('应用调整（JSON）'), 'edit.reset': N_('重置全部调整'), 'edit.undo': N_('撤销'), 'edit.redo': N_('重做'),
    'edit.auto': N_('自动色调'), 'edit.presets': N_('列出内置预设'), 'edit.preset': N_('套用内置预设'),
    'edit.copy': N_('将调色复制到其他照片'), 'edit.process': N_('设置处理版本'), 'render.preview': N_('渲染预览'),
    'photo.export': N_('导出成片'), 'photo.export_batch': N_('批量导出'), 'project.save': N_('保存工程'),
}


def template(cmd):
    """A JSON object with the required parameters, as a starting point."""
    sample = {}
    for name in cmd.required:
        spec = cmd.params[name]
        kind = spec.get('type')
        sample[name] = spec['enum'][0] if 'enum' in spec else {'integer': 0, 'number': 0, 'boolean': False,
                                                                'array': [], 'object': {}}.get(kind, '')
    return json.dumps(sample, ensure_ascii=False)


class CommandPalette(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle(tr('命令面板'))
        self.setMinimumSize(560, 460)
        root = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText(tr('输入命令名称或 id，例如 rate、edit.auto、export'))
        self.search.textChanged.connect(self.filter)
        self.search.returnPressed.connect(self.run_current)
        root.addWidget(self.search)
        self.list = QListWidget()
        self.list.itemActivated.connect(lambda item: self.run(item.data(Qt.ItemDataRole.UserRole)))
        root.addWidget(self.list, 1)
        note = QLabel(tr('这些命令也可由 lumen-cli 与 MCP 助手调用（帮助 → AI 助手接入）。'))
        note.setObjectName('subtle')
        note.setWordWrap(True)
        root.addWidget(note)
        for cmd in sorted(commands.REGISTRY.values(), key=lambda c: c.id):
            title = tr(TITLES[cmd.id]) if cmd.id in TITLES else cmd.id
            item = QListWidgetItem(f'{title}    ·    {cmd.id}')
            item.setToolTip(cmd.summary)
            item.setData(Qt.ItemDataRole.UserRole, cmd.id)
            self.list.addItem(item)
        self.list.setCurrentRow(0)

    def filter(self, text):
        words = text.lower().split()
        first = None
        for i in range(self.list.count()):
            item = self.list.item(i)
            visible = all(w in (item.text() + ' ' + item.toolTip()).lower() for w in words)
            item.setHidden(not visible)
            if visible and first is None:
                first = item
        if first is not None:
            self.list.setCurrentItem(first)

    def run_current(self):
        item = self.list.currentItem()
        if item is not None and not item.isHidden():
            self.run(item.data(Qt.ItemDataRole.UserRole))

    def run(self, command_id):
        cmd = commands.REGISTRY[command_id]
        args = {}
        if cmd.params:
            text, ok = QInputDialog.getMultiLineText(
                self, tr('命令参数'), tr('{command} 的参数（JSON 对象）：\n{summary}', command=cmd.id, summary=cmd.summary),
                template(cmd))
            if not ok:
                return
            try:
                args = json.loads(text or '{}')
            except ValueError as exc:
                return QMessageBox.warning(self, 'LUMEN RAW', tr('参数不是有效的 JSON：') + str(exc))
        self.accept()
        self.window.statusBar().showMessage(tr('正在执行 {command}…', command=cmd.id))
        # Commands may wait for the window (opening a photo, exporting): never on the GUI thread.
        self.window.control.submit(cmd.id, args, lambda outcome: self.done_command(cmd, outcome))

    def done_command(self, cmd, result):
        if isinstance(result, Exception):
            return QMessageBox.warning(self.window, 'LUMEN RAW', str(result))
        result = {k: v for k, v in result.items() if k != 'image'}
        if cmd.read_only:
            box = QMessageBox(self.window)
            box.setWindowTitle(cmd.id)
            box.setText(json.dumps(result, ensure_ascii=False, indent=1)[:3000])
            box.exec()
        else:
            self.window.statusBar().showMessage(tr('已执行 {command}', command=cmd.id) + tr(' · {v} 可撤销', v=host.keys('Ctrl+Z'))
                                                if cmd.id.startswith('edit.') else tr('已执行 {command}', command=cmd.id))
