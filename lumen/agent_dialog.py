"""AI assistant (MCP) setup and activity (1.6.0)."""
from __future__ import annotations

import time

from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QListWidget, QPlainTextEdit,
                               QPushButton, QTabWidget, QVBoxLayout, QWidget)

from . import cli, control, host, i18n
from .i18n import tr


class AgentMixin:
    def init_agents(self):
        self.agent_log = []
        self.agent_dialog = None
        self.control = control.Server(self)
        if control.enabled():
            self.control.start()

    def agent_activity(self, text, path=None):
        """An agent's command reached the window: status bar and activity list."""
        self.agent_log.append((time.strftime('%H:%M:%S'), str(text)))
        del self.agent_log[:-50]
        undo = path is not None and path == self.source_path
        self.statusBar().showMessage(tr('AI 助手：{action}', action=text) + (tr(' · {v} 可撤销', v=host.keys('Ctrl+Z')) if undo else ''))
        if self.agent_dialog is not None and self.agent_dialog.isVisible():
            self.agent_dialog.refresh_log()

    def open_agent_dialog(self):
        if self.agent_dialog is None:
            self.agent_dialog = AgentDialog(self)
        self.agent_dialog.refresh()
        self.agent_dialog.show()
        self.agent_dialog.raise_()


class AgentDialog(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle(tr('AI 助手接入（MCP）'))
        self.setMinimumSize(660, 600)
        root = QVBoxLayout(self)
        title = QLabel(tr('AI 助手接入  /  MCP'))
        title.setObjectName('section')
        root.addWidget(title)
        text = QLabel(tr('让本机支持 MCP 的 AI 助手（Claude Code、Codex、WorkBuddy、Claude Desktop 等）直接操作 LUMEN RAW：'
                         '打开照片、调色、评级、预览与导出。LUMEN RAW 打开时，助手的每一步修改都显示在窗口中，可用 {undo} 撤销；'
                         '未打开时在后台处理。原片始终不会被修改。', undo=host.keys('Ctrl+Z')))
        text.setWordWrap(True)
        text.setObjectName('subtle')
        root.addWidget(text)
        row = QHBoxLayout()
        self.allow = QCheckBox(tr('允许本机 AI 助手控制此窗口'))
        self.allow.setChecked(control.enabled())
        self.allow.toggled.connect(self.toggle)
        row.addWidget(self.allow)
        row.addStretch()
        self.state = QLabel()
        self.state.setObjectName('subtle')
        row.addWidget(self.state)
        root.addLayout(row)
        mono = QFont('Consolas' if host.WINDOWS else 'Menlo', 9)
        self.snippets = []
        clients = QTabWidget()
        for title, heading, kind, note in (
                ('Claude Code', tr('Claude Code（在终端运行）'), 'claude-code', ''),
                ('Codex', tr('Codex（在终端运行，或写入 ~/.codex/config.toml）'), 'codex', ''),
                ('WorkBuddy · JSON', tr('WorkBuddy、Claude Desktop 及其他 MCP 客户端（JSON 配置）'), 'json',
                 tr('WorkBuddy：在设置的 MCP（自定义连接）中添加服务器并粘贴这段 JSON；Claude Desktop：合并到 claude_desktop_config.json。'))):
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(4, 10, 4, 4)
            label = QLabel(heading)
            label.setObjectName('subtle')
            label.setWordWrap(True)
            layout.addWidget(label)
            if note:
                hint = QLabel(note)
                hint.setWordWrap(True)
                hint.setObjectName('subtle')
                layout.addWidget(hint)
            box = QPlainTextEdit(cli.mcp_config(kind))
            box.setReadOnly(True)
            box.setFont(mono)
            box.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            layout.addWidget(box, 1)
            copy = QPushButton(tr('复制'))
            copy.clicked.connect(lambda checked=False, b=box: self.copy(b))
            row = QHBoxLayout()
            row.addStretch()
            row.addWidget(copy)
            layout.addLayout(row)
            clients.addTab(page, title)
            self.snippets.append(box)
        root.addWidget(clients, 1)
        label = QLabel(tr('最近的助手操作'))
        label.setObjectName('section')
        root.addWidget(label)
        self.activity = QListWidget()
        self.activity.setFixedHeight(110)
        root.addWidget(self.activity)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText(tr('关闭'))
        buttons.rejected.connect(self.close)
        root.addWidget(buttons)

    def copy(self, box):
        QGuiApplication.clipboard().setText(box.toPlainText())
        self.window.statusBar().showMessage(tr('已复制到剪贴板'))

    def toggle(self, on):
        i18n.save_setting(control.SETTING, '1' if on else '0')
        if on:
            self.window.control.start()
        else:
            self.window.control.stop()
        self.refresh()

    def refresh(self):
        listening = self.window.control.listening
        self.state.setText(tr('正在等待连接') if listening else tr('已关闭：助手会在后台单独处理照片'))
        self.refresh_log()

    def refresh_log(self):
        self.activity.clear()
        for stamp, text in reversed(self.window.agent_log):
            self.activity.addItem(f'{stamp}  {text}')
