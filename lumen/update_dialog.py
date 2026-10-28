"""Software update and AI model download windows (2.1.1); the logic is in ``updates`` and ``ai_models``.

2.1.1: when a check or a download fails, the windows say where to download by hand (the newest
release, whose full installer carries every AI model) with links that open the page or folder.
"""
from __future__ import annotations

import html
import math
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (QDialog, QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QScrollArea,
                               QVBoxLayout, QWidget)

from . import __version__, ai_models, downloads, theme, updates
from .i18n import tr


class _Relay(QObject):
    """Delivers results from worker threads on the GUI thread."""
    done = Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.done.connect(lambda callback, value: callback(value))


def _run(relay, work, finished):
    def target():
        try:
            result = work()
        except BaseException as exc:  # reported in the window
            result = exc
        try:
            relay.done.emit(finished, result)
        except RuntimeError:
            pass  # the window closed meanwhile
    threading.Thread(target=target, daemon=True, name='lumen-update').start()


def megabytes(count):
    return max(1, round(count / 1e6)) if count else 0


def page_link(url):
    """A web page as a link that shows its address, so it can also be copied into a browser."""
    return url.split('://', 1)[-1], url


def folder_link(folder, label):
    return label, QUrl.fromLocalFile(str(folder)).toString()


def help_html(*lines):
    """Rich text for a QLabel: each line is ``(text, [(label, url), …])``; the links go on a line
    of their own, so an address is not broken in the middle."""
    rows = []
    for text, links in lines:
        anchors = ' · '.join(f'<a href="{html.escape(url, quote=True)}">{html.escape(label)}</a>' for label, url in links)
        rows.append(html.escape(text) + ('<br>' + anchors if anchors else ''))
    return ''.join(f'<p style="margin: 0 0 6px 0">{row}</p>' for row in rows) if len(rows) > 1 else rows[0]


def open_link(url):
    """Opens a web page, or a local folder (made first when it is not there yet)."""
    link = QUrl(url)
    if link.isLocalFile():
        try:
            Path(link.toLocalFile()).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    QDesktopServices.openUrl(link)


def help_label():
    """A small label for ``help_html`` whose links open with ``open_link``; hidden until used."""
    label = QLabel()
    label.setObjectName('subtle')
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    label.linkActivated.connect(open_link)
    label.hide()
    return label


class RefreshIcon(QWidget):
    """Two circling arrows, drawn so that no icon font is needed."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(26, 26)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(theme.color('text'), 2.2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        box = QRectF(4, 4, 18, 18)
        centre, radius = box.center(), 9
        for start in (20, 200):
            path = QPainterPath()
            path.arcMoveTo(box, start)
            path.arcTo(box, start, 130)
            painter.drawPath(path)
            end = math.radians(start + 130)
            tip = QPointF(centre.x() + radius * math.cos(end), centre.y() - radius * math.sin(end))
            ahead = QPointF(-math.sin(end), -math.cos(end))     # direction of travel at the arc's end
            side = QPointF(math.cos(end), -math.sin(end))
            head = QPolygonF([tip + ahead * 3.2, tip - side * 3.4, tip + side * 3.4])
            painter.setBrush(theme.color('text'))
            painter.drawPolygon(head)


class _Card(QDialog):
    """A window laid out like a card: title row with a badge, rule, then content; the buttons at
    the bottom right are the app's own (orange main button, as Export, and plain ones)."""

    def __init__(self, window, title, badge='', caption='', badge_style='okBadge'):
        super().__init__(window)
        self.window = window
        self.relay = _Relay(self)
        self.setWindowTitle(caption or title)
        self.setMinimumWidth(560)
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(28, 22, 28, 24)
        self.root.setSpacing(14)
        head = QHBoxLayout()
        head.setSpacing(12)
        head.addWidget(RefreshIcon())
        label = QLabel(title)
        label.setObjectName('cardTitle')
        head.addWidget(label)
        head.addStretch()
        self.badge = QLabel(badge)
        self.badge.setObjectName(badge_style)
        self.badge.setVisible(bool(badge))
        head.addWidget(self.badge)
        self.root.addLayout(head)
        rule = QFrame()
        rule.setObjectName('rule')
        rule.setFixedHeight(1)
        self.root.addWidget(rule)

    def buttons(self):
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addStretch()
        self.primary = QPushButton()
        self.primary.setObjectName('primary')
        self.secondary = QPushButton()
        self.actions = {}
        for button in (self.primary, self.secondary):
            button.setMinimumWidth(116)
            button.clicked.connect(lambda checked=False, b=button: self.actions[b]())
            row.addWidget(button)
        self.root.addSpacing(4)
        self.root.addLayout(row)
        self.primary.setDefault(True)

    def rebind(self, button, text, callback, visible=True):
        """Give one of the two buttons a new label and action."""
        self.actions[button] = callback
        button.setText(text)
        button.setVisible(visible)
        button.setEnabled(True)

    def add_help(self):
        """2.1.1: the manual-download links, below the status line."""
        self.help = help_label()
        self.root.addWidget(self.help)

    def show_help(self, *lines):
        self.help.setText(help_html(*lines))
        self.help.show()


class UpdateDialog(_Card):
    """A newer version: current → latest, what changed, then download and install (the layout
    the 2.1.1 design asked for)."""

    def __init__(self, window, update):
        super().__init__(window, tr('LUMEN RAW — 软件更新'), tr('发现新版本'), tr('软件更新'))
        self.update_info = update
        self.cancel = None
        versions = QHBoxLayout()
        versions.setSpacing(24)
        for caption, version, name in ((tr('当前版本'), __version__, 'versionOld'), (tr('最新版本'), update.version, 'versionNew')):
            column = QVBoxLayout()
            column.setSpacing(4)
            small = QLabel(caption)
            small.setObjectName('subtle')
            big = QLabel('v' + version)
            big.setObjectName(name)
            column.addWidget(small)
            column.addWidget(big)
            versions.addLayout(column, 1)
            if name == 'versionOld':
                arrow = QLabel('→')
                arrow.setObjectName('versionArrow')
                versions.addWidget(arrow, 0, Qt.AlignmentFlag.AlignBottom)
        self.root.addLayout(versions)
        self.size_label = QLabel()
        self.size_label.setObjectName('subtle')
        self.size_label.setWordWrap(True)
        self.root.addWidget(self.size_label)
        if update.source in {downloads.host(base) for base in downloads.mirrors()}:
            # 2.1.2: GitHub could not be reached; the signed feed came from a mirror, the files will too.
            mirror = QLabel(tr('GitHub 暂时无法连接，更新信息来自镜像 {host}（已验证签名），文件也会从镜像下载。', host=update.source))
            mirror.setObjectName('subtle')
            mirror.setWordWrap(True)
            self.root.addWidget(mirror)
        heading = QLabel(tr('更新内容'))
        heading.setObjectName('cardHeading')
        self.root.addWidget(heading)
        notes = QLabel(update.notes or tr('请查看发布页面了解更新内容。'))
        notes.setObjectName('notes')
        notes.setWordWrap(True)
        notes.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        notes.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        scroll = QScrollArea()
        scroll.setObjectName('notesScroll')
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(notes)
        scroll.setMaximumHeight(220)
        scroll.setMinimumHeight(min(220, notes.sizeHint().height() + 8))
        self.root.addWidget(scroll)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.hide()
        self.root.addWidget(self.progress)
        self.status = QLabel()
        self.status.setObjectName('subtle')
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.hide()
        self.root.addWidget(self.status)
        self.add_help()
        self.buttons()
        self.show_offer()

    def show_offer(self):
        update = self.update_info
        if update.installable():
            size = update.download_bytes()
            models = ai_models.size(update.missing_models())
            if not size:
                text = tr('更新已下载，可以安装。')
            elif models:
                text = tr('下载大小约 {total} MB（程序 {program} MB，变化的 AI 模型 {models} MB）。已有的 AI 模型不会重新下载。',
                          total=megabytes(size), program=megabytes(size - models), models=megabytes(models))
            else:
                text = tr('下载大小约 {total} MB。AI 模型没有变化，不会重新下载。', total=megabytes(size))
            self.size_label.setText(text)
            self.rebind(self.primary, tr('立即更新'), self.start)
        else:
            self.size_label.setText(tr('这份 LUMEN RAW 不是用安装程序安装的（便携版、源码或 macOS），请在发布页面下载新版本。'))
            self.rebind(self.primary, tr('打开下载页面'), self.open_page)
        self.rebind(self.secondary, tr('下次提醒'), self.reject)

    def open_page(self):
        QDesktopServices.openUrl(QUrl(self.update_info.page))
        self.accept()

    def start(self):
        cancel = threading.Event()
        self.cancel = cancel
        self.progress.setValue(0)
        self.progress.show()
        self.status.setText(tr('正在下载更新…'))
        self.status.show()
        self.help.hide()
        self.primary.hide()
        self.rebind(self.secondary, tr('取消'), cancel.set)
        shown = [-1]

        def progress(done, total):
            permille = int(done * 1000 / max(total, 1))
            if permille != shown[0]:
                shown[0] = permille
                try:
                    self.relay.done.emit(self.downloading, (done, total))
                except RuntimeError:
                    pass
        _run(self.relay, lambda: updates.download(self.update_info, progress, cancel.is_set), self.downloaded)

    def downloading(self, value):
        done, total = value
        if self.cancel is None:
            return
        self.progress.setValue(int(done * 1000 / max(total, 1)))
        self.status.setText(tr('正在下载更新… {done} / {total} MB', done=round(done / 1e6), total=megabytes(total)))

    def downloaded(self, result):
        self.cancel = None
        if isinstance(result, downloads.Cancelled):
            self.progress.hide()
            self.status.setText(tr('已取消下载，下次会从中断处继续。'))
            self.show_offer()
            return
        if isinstance(result, BaseException):
            self.progress.hide()
            self.status.setText(tr('更新下载失败：{error}', error=result))
            self.show_help((tr('也可以在发布页面手动下载完整安装包（已包含全部 AI 模型），直接覆盖安装：'),
                            [page_link(self.update_info.page), page_link(updates.MIRROR_PAGE)]))
            self.rebind(self.primary, tr('重试'), self.start)
            self.rebind(self.secondary, tr('关闭'), self.reject)
            return
        self.installer = result
        self.progress.setValue(1000)
        self.status.setText(tr('下载完成。安装时 LUMEN RAW 会先关闭（未保存的修改会先询问），装好后自动重新打开。'))
        self.rebind(self.primary, tr('重启并更新'), self.install)
        self.rebind(self.secondary, tr('稍后'), self.reject)

    def install(self):
        if self.window.request_update_install(self.installer):
            self.accept()
        else:
            self.status.setText(tr('没有关闭 LUMEN RAW，更新未安装。随时可以再点“重启并更新”。'))

    def reject(self):
        if self.cancel is not None:
            self.cancel.set()
        super().reject()


class ModelsDialog(_Card):
    """The AI models this version needs and lacks (after an update, or a program-only install)."""

    def __init__(self, window, records=None):
        super().__init__(window, tr('LUMEN RAW — AI 模型'), caption=tr('AI 模型'))
        self.cancel = None
        self.records = ai_models.missing() if records is None else records
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.root.addWidget(self.text)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.hide()
        self.root.addWidget(self.progress)
        self.status = QLabel()
        self.status.setObjectName('subtle')
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.root.addWidget(self.status)
        self.add_help()
        self.buttons()
        self.show_state()

    def show_state(self):
        if not self.records:
            self.text.setText(tr('这个版本需要的 AI 模型都已在本机，可以离线使用。'))
            self.status.setText(tr('模型文件夹：{folder}', folder=ai_models.download_folder()))
            self.rebind(self.primary, tr('关闭'), self.accept)
            self.secondary.hide()
            return
        features = '、'.join(dict.fromkeys(tr(ai_models.FEATURES.get(r['name'], r['name'])) for r in self.records))
        self.text.setText(tr('以下功能需要的 AI 模型还没有下载（约 {size} MB）：{features}。下载后保存在本机，之后离线可用，'
                             '以后的更新只会下载有变化的模型。', size=megabytes(ai_models.size(self.records)), features=features))
        self.status.setText(tr('保存到：{folder}', folder=ai_models.download_folder()))
        self.rebind(self.primary, tr('立即下载'), self.start)
        self.rebind(self.secondary, tr('稍后'), self.reject)

    def start(self):
        cancel = threading.Event()
        self.cancel = cancel
        self.progress.setValue(0)
        self.progress.show()
        self.help.hide()
        self.primary.hide()
        self.rebind(self.secondary, tr('取消'), cancel.set)
        shown = [-1]

        def progress(done, total):
            permille = int(done * 1000 / max(total, 1))
            if permille != shown[0]:
                shown[0] = permille
                try:
                    self.relay.done.emit(self.downloading, (done, total))
                except RuntimeError:
                    pass
        records = list(self.records)
        _run(self.relay, lambda: ai_models.download(records, progress, cancel.is_set), self.downloaded)

    def downloading(self, value):
        done, total = value
        if self.cancel is None:
            return
        self.progress.setValue(int(done * 1000 / max(total, 1)))
        self.status.setText(tr('正在下载 AI 模型… {done} / {total} MB', done=round(done / 1e6), total=megabytes(total)))

    def downloaded(self, result):
        self.cancel = None
        self.progress.hide()
        if isinstance(result, downloads.Cancelled):
            self.show_state()
            self.status.setText(tr('已取消下载，下次会从中断处继续。'))
            return
        if isinstance(result, BaseException):
            self.show_state()
            self.status.setText(str(result))
            self.show_help((tr('也可以手动下载：最新版本的完整安装包已包含全部 AI 模型，覆盖安装即可：'),
                            [page_link(updates.release_page()), page_link(updates.MIRROR_PAGE)]),
                           (tr('或者只下载缺少的模型文件（不要改文件名），放进模型文件夹后点“重试”：'),
                            [page_link(ai_models.PAGE), folder_link(ai_models.download_folder(), tr('打开模型文件夹'))]))
            self.rebind(self.primary, tr('重试'), self.start)
            return
        self.records = ai_models.missing()
        self.window.models_changed()
        self.show_state()
        if not self.records:
            self.text.setText(tr('AI 模型已下载完成，可以使用全部 AI 功能。'))

    def reject(self):
        if self.cancel is not None:
            self.cancel.set()
        super().reject()


class CheckResult(_Card):
    """“Already the newest” or why the check failed, in the same style as the update window."""

    def __init__(self, window, error=None):
        latest = error is None
        super().__init__(window, tr('LUMEN RAW — 软件更新'), tr('已是最新版本') if latest else tr('检查失败'), tr('检查更新'),
                         'okBadge' if latest else 'warnBadge')
        text = QLabel(tr('LUMEN RAW {version} 已是最新版本。', version=__version__) if latest
                      else tr('无法检查更新：{error}', error=error))
        text.setWordWrap(True)
        text.setTextFormat(Qt.TextFormat.PlainText)
        self.root.addWidget(text)
        if latest:
            hint = QLabel(tr('有新版本时，启动 LUMEN RAW 后会自动提示（可在「帮助 → 启动时检查更新」中关闭）。'))
            hint.setWordWrap(True)
            hint.setObjectName('subtle')
            self.root.addWidget(hint)
        else:
            self.add_help()
            self.show_help((tr('请检查网络连接后重试，或在发布页面手动下载最新版本：'),
                            [page_link(updates.release_page()), page_link(updates.MIRROR_PAGE)]))
        self.buttons()
        if latest:
            self.rebind(self.primary, tr('关闭'), self.accept)
            self.secondary.hide()
        else:
            self.rebind(self.primary, tr('重试'), self.retry)
            self.rebind(self.secondary, tr('关闭'), self.reject)

    def retry(self):
        self.accept()
        self.window.check_for_updates(manual=True)


class UpdateMixin:
    """MainWindow side: start-up checks, the Help menu entries and installing on close."""

    def init_updates(self):
        self.update_relay = _Relay(self)
        self.update_dialog = None
        self.models_dialog = None
        self.update_installer = None
        if updates.interactive() and (updates.automatic() or ai_models.packaged()):
            QTimer.singleShot(2500, self, self.startup_checks)

    def build_update_menu(self, menu):
        menu.addAction(tr('检查更新…'), lambda: self.check_for_updates(manual=True))
        menu.addAction(tr('AI 模型…'), self.open_models_dialog)
        automatic = menu.addAction(tr('启动时检查更新'))
        automatic.setCheckable(True)
        automatic.setChecked(updates.setting(updates.SETTING, '1') != '0')
        automatic.toggled.connect(self.set_update_check)

    def set_update_check(self, on):
        try:
            updates.set_automatic(on)
        except OSError as exc:
            self.statusBar().showMessage(tr('无法保存设置：') + str(exc))

    def startup_checks(self):
        """Tidy the models and downloads, offer missing models, then look for a new version."""
        def work():
            ai_models.prepare(keep=updates.keep_models())
            updates.tidy()
            return ai_models.missing()
        _run(self.update_relay, work, self.startup_models)

    def startup_models(self, result):
        if isinstance(result, list) and result and self.isVisible():
            self.open_models_dialog(result)
        elif isinstance(result, BaseException):
            self.statusBar().showMessage(str(result))
        if updates.automatic():
            self.check_for_updates(manual=False)

    def check_for_updates(self, manual=False):
        if manual:
            self.statusBar().showMessage(tr('正在检查更新…'))
        _run(self.update_relay, updates.check, lambda result: self.update_checked(result, manual))

    def update_checked(self, result, manual):
        if isinstance(result, BaseException) or result is None:
            if manual:
                self.statusBar().clearMessage()
                self.check_result = CheckResult(self, result)
                self.check_result.show()
            return
        if self.update_dialog is not None and self.update_dialog.isVisible():
            self.update_dialog.raise_()
            return
        self.update_dialog = UpdateDialog(self, result)
        self.update_dialog.show()

    def open_models_dialog(self, records=None):
        if self.models_dialog is not None and self.models_dialog.isVisible():
            self.models_dialog.raise_()
            return
        self.models_dialog = ModelsDialog(self, records)
        self.models_dialog.show()

    def models_changed(self):
        """Models arrived; the face analyser, given up on without them, is tried again, and the
        photos still lacking a part of the analysis are queued (2.1.1: the closed eyes wait for
        the face models)."""
        from . import culling
        if culling._faces is False:
            culling._faces = None
        worker = getattr(self, 'library_worker', None)
        if worker is not None and worker.parts:
            from .library_tasks import possible
            worker.want_analysis(self.library_db.unanalysed(limit=5000, parts=possible(worker.parts)))
        self.statusBar().showMessage(tr('AI 模型已就绪'))

    def request_update_install(self, installer):
        """Close the window (asking about unsaved edits); ``main`` starts the installer after."""
        self.update_installer = installer
        self.close()
        if not self.closing:
            self.update_installer = None
            return False
        return True
