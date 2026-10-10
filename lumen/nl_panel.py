"""自然语言输入 (1.5.0): the left-panel page between 预设 and 快照.

Type or say what the photo should look like; the chosen local or cloud language model
returns slider values (``nl_edit``) that are applied as one undo step, with an
adjustable strength. Speech is recognized offline (``speech``) from the default
microphone through Qt Multimedia.
"""
from __future__ import annotations
import copy
import json
import sys
import threading
import time
import numpy as np
from PySide6.QtCore import Qt, QObject, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QPlainTextEdit,
    QComboBox, QListWidget, QDialog, QFormLayout, QLineEdit, QCheckBox, QSpinBox, QDialogButtonBox,
    QToolButton, QSizePolicy, QAbstractItemView, QProgressBar, QScrollArea, QFrame)
from . import nl_edit, speech, engine, host
from .scheduler import Activity as A
from .widgets import AdjustSlider
from .i18n import tr

EXAMPLES = [
    (tr('示例指令…'), ''),
    (tr('通透自然'), tr('整体通透自然一些，压一点高光、提亮暗部，颜色干净')),
    (tr('天空更蓝'), tr('天空更蓝更有层次，地面保持自然')),
    (tr('温暖日落'), tr('做成温暖的日落氛围，高光偏金色，暗部略带青蓝')),
    (tr('日系清新'), tr('日系清新风格：明亮、低对比、略偏青、饱和度低一点')),
    (tr('电影青橙'), tr('电影感青橙色调，暗部偏青，高光偏橙，加一点暗角')),
    (tr('复古胶片'), tr('复古胶片感：黑位抬起、轻微褪色、加颗粒')),
    (tr('黑白高反差'), tr('高反差黑白，强调纹理和光影')),
    (tr('夜景降噪'), tr('夜景：压暗高光防止过曝，提亮暗部并降噪')),
    (tr('人像柔肤'), tr('人像：人物提亮一点，肤色自然通透，纹理柔和')),
]
RECORD_LIMIT = 30
VOICE = tr('语音')


class _Relay(QObject):
    """Delivers results from worker threads on the GUI thread."""
    done = Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.done.connect(lambda callback, value: callback(value))


def run_async(relay, work, success, failure):
    def target():
        try:
            result, callback = work(), success
        except Exception as error:  # reported in the panel
            result, callback = error, failure
        try:
            relay.done.emit(callback, result)
        except RuntimeError:
            pass  # the window closed while the request was running
    threading.Thread(target=target, daemon=True, name='lumen-natural-language').start()


class InstructionEdit(QPlainTextEdit):
    """Enter sends; Shift+Enter starts a new line."""
    submitted = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.submitted.emit()
            return
        super().keyPressEvent(event)


class VoiceRecorder(QObject):
    """Default microphone -> mono int16 at 16 kHz; stops on a pause after speech."""
    level = Signal(float)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = None
        self.device = None

    @property
    def recording(self):
        return self.source is not None

    def start(self):
        from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices
        device = QMediaDevices.defaultAudioInput()
        if device.isNull():
            self.failed.emit(tr('没有找到麦克风。请连接麦克风，并在系统隐私设置中允许应用使用麦克风。'))
            return False
        wanted = QAudioFormat()
        wanted.setSampleRate(speech.RATE)
        wanted.setChannelCount(1)
        wanted.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        self.format = wanted if device.isFormatSupported(wanted) else device.preferredFormat()
        self.source = QAudioSource(device, self.format, self)
        self.source.setBufferSize(self.format.bytesForDuration(200000))
        self.chunks, self.started = [], time.monotonic()
        self.heard, self.quiet_since, self.floor = False, None, None
        self.device = self.source.start()
        if self.device is None:
            self.source = None
            self.failed.emit(tr('无法打开麦克风。'))
            return False
        self.device.readyRead.connect(self._read)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(100)
        return True

    def _samples(self, data):
        from PySide6.QtMultimedia import QAudioFormat
        kind = self.format.sampleFormat()
        if kind == QAudioFormat.SampleFormat.Int16:
            x = np.frombuffer(data, np.int16).astype(np.float32) / 32768
        elif kind == QAudioFormat.SampleFormat.Int32:
            x = np.frombuffer(data, np.int32).astype(np.float32) / 2147483648
        elif kind == QAudioFormat.SampleFormat.Float:
            x = np.frombuffer(data, np.float32).copy()
        else:
            x = (np.frombuffer(data, np.uint8).astype(np.float32) - 128) / 128
        channels = max(1, self.format.channelCount())
        x = x[:len(x) // channels * channels]
        return x.reshape(-1, channels).mean(axis=1)

    def _read(self):
        if self.device is None:
            return
        data = bytes(self.device.readAll())
        if not data:
            return
        x = self._samples(data)
        self.chunks.append(x)
        rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.
        self.level.emit(min(1., rms * 8))
        elapsed = time.monotonic() - self.started
        if self.floor is None or (elapsed < .4 and not self.heard):
            self.floor = rms if self.floor is None else min(self.floor * .8 + rms * .2, max(self.floor, rms))
        threshold = max(.012, self.floor * 3)
        if rms > threshold:
            self.heard, self.quiet_since = True, None
        elif self.heard and self.quiet_since is None:
            self.quiet_since = time.monotonic()

    def _tick(self):
        if self.source is None:
            return
        # QtAudio.Error since Qt 6.7 (QAudio.Error before); compare by name.
        if getattr(self.source.error(), 'name', 'NoError') != 'NoError':
            self.cancel()
            self.failed.emit(tr('麦克风录音出错。请在系统隐私设置中允许应用使用麦克风。'))
            return
        elapsed = time.monotonic() - self.started
        if self.heard and self.quiet_since is not None and time.monotonic() - self.quiet_since > 1.2:
            self.stop()
        elif not self.heard and elapsed > 8:
            self.cancel()
            self.failed.emit(tr('没有听到声音。请检查麦克风是否静音，或靠近一些再说。'))
        elif elapsed > RECORD_LIMIT:
            self.stop()

    def _close(self):
        self.timer.stop()
        if self.device is not None:
            self._read()
        source, self.source, self.device = self.source, None, None
        source.stop()
        source.deleteLater()

    def cancel(self):
        if self.source is not None:
            self._close()

    def stop(self):
        if self.source is None:
            return
        self._close()
        x = np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)
        rate = self.format.sampleRate()
        if rate != speech.RATE and len(x):
            count = int(round(len(x) * speech.RATE / rate))
            spectrum = np.fft.rfft(x)[:count // 2 + 1]
            x = np.fft.irfft(spectrum, count) * (count / len(x))
        self.finished.emit(np.clip(x * 32768, -32768, 32767).astype(np.int16))


class SettingsDialog(QDialog):
    def __init__(self, parent, settings):
        super().__init__(parent)
        self.setWindowTitle(tr('自然语言修图 · 模型设置'))
        self.setMinimumWidth(560)
        self.settings = copy.deepcopy(settings)
        self.relay = _Relay(self)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.provider = QComboBox()
        for title, keys in ((tr('本地模型（不需要 API Key）'), nl_edit.LOCAL_PROVIDERS), (tr('云端 API'), nl_edit.CLOUD_PROVIDERS)):
            self.provider.addItem(f'—— {title} ——', None)
            self.provider.model().item(self.provider.count() - 1).setEnabled(False)
            for key in keys:
                self.provider.addItem(nl_edit.provider_label(key), key)
        form.addRow(tr('服务'), self.provider)
        url_row = QHBoxLayout()
        self.base_url = QLineEdit()
        url_row.addWidget(self.base_url, 1)
        default = QPushButton(tr('默认地址'))
        default.clicked.connect(lambda: self.base_url.setText(nl_edit.PROVIDERS[self.current][2]))
        url_row.addWidget(default)
        form.addRow(tr('接口地址'), url_row)
        key_row = QHBoxLayout()
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        key_row.addWidget(self.key, 1)
        show = QCheckBox(tr('显示'))
        show.toggled.connect(lambda on: self.key.setEchoMode(QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password))
        key_row.addWidget(show)
        form.addRow('API Key', key_row)
        model_row = QHBoxLayout()
        self.model_name = QComboBox()
        self.model_name.setEditable(True)
        self.model_name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        model_row.addWidget(self.model_name, 1)
        self.refresh_button = QPushButton(tr('获取模型列表'))
        self.refresh_button.clicked.connect(self.fetch_models)
        model_row.addWidget(self.refresh_button)
        form.addRow(tr('模型'), model_row)
        self.timeout = QSpinBox()
        self.timeout.setRange(10, 900)
        self.timeout.setSuffix(tr(' 秒'))
        self.timeout.setValue(settings['timeout'])
        form.addRow(tr('超时'), self.timeout)
        self.reasoning = QComboBox()
        for code, title in ((code, tr(title)) for code, title in nl_edit.REASONING.items()):
            self.reasoning.addItem(title, code)
        self.reasoning.setCurrentIndex(max(0, self.reasoning.findData(settings.get('reasoning', 'default'))))
        self.reasoning.setToolTip(tr('思考越深，结果可能越细致，但更慢、消耗的 token 更多；修图参数通常“低”或“关闭”就够用。'))
        self.reasoning.currentIndexChanged.connect(self.show_thinking_note)
        form.addRow(tr('思考强度'), self.reasoning)
        self.thinking_note = QLabel('')
        self.thinking_note.setWordWrap(True)
        self.thinking_note.setObjectName('subtle')
        form.addRow('', self.thinking_note)
        self.attach = QCheckBox(tr('附带当前预览图（长边 768 像素；模型需支持图像输入）'))
        self.attach.setChecked(settings['attach_image'])
        form.addRow('', self.attach)
        self.voice = QComboBox()
        for title, code in ((tr('自动（中文 / English）'), 'auto'), (tr('中文'), 'zh'), ('English', 'en')):
            self.voice.addItem(title, code)
        self.voice.setCurrentIndex(max(0, self.voice.findData(settings['voice_language'])))
        form.addRow(tr('语音识别'), self.voice)
        self.auto_send = QCheckBox(tr('说完后自动修图'))
        self.auto_send.setChecked(settings['auto_send'])
        form.addRow('', self.auto_send)
        layout.addLayout(form)
        test_row = QHBoxLayout()
        self.test_button = QPushButton(tr('测试连接'))
        self.test_button.clicked.connect(self.test)
        test_row.addWidget(self.test_button)
        self.status = QLabel('')
        self.status.setWordWrap(True)
        self.status.setObjectName('subtle')
        test_row.addWidget(self.status, 1)
        layout.addLayout(test_row)
        self.hint = QLabel('')
        self.hint.setWordWrap(True)
        self.hint.setObjectName('subtle')
        layout.addWidget(self.hint)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.current = None
        self.provider.currentIndexChanged.connect(self.provider_changed)
        self.provider.setCurrentIndex(self.provider.findData(settings['provider']))

    def store(self):
        if self.current:
            entry = self.settings['providers'][self.current]
            entry.update(base_url=self.base_url.text().strip(), key=self.key.text().strip(),
                         model=self.model_name.currentText().strip())

    def provider_changed(self, _=None):
        key = self.provider.currentData()
        if key is None:
            return
        self.store()
        self.current = key
        entry = self.settings['providers'][key]
        label, api, url, suggested, local, _, _ = nl_edit.PROVIDERS[key]
        self.base_url.setText(entry['base_url'])
        self.key.setText(entry['key'])
        self.key.setPlaceholderText(tr('本地服务通常留空') if local else tr('在服务商控制台创建'))
        self.model_name.clear()
        if entry['model']:
            self.model_name.addItem(entry['model'])
        self.model_name.setCurrentText(entry['model'])
        self.model_name.lineEdit().setPlaceholderText(suggested or tr('点“获取模型列表”或直接填写'))
        self.status.setText('')
        if local:
            self.hint.setText({
                'ollama': tr('Ollama：安装后保持运行即可（默认端口 11434）。用 ollama pull qwen3:8b 等下载模型；带图像输入可选 qwen2.5vl、gemma3 等。'),
                'lmstudio': tr('LM Studio：在“开发者”页面加载模型并启动本地服务（默认端口 1234）。'),
                'llamacpp': tr('llama.cpp：运行 llama-server -m 模型.gguf --port 8080；图像输入需要同时加载 --mmproj。'),
            }.get(key, tr('任何提供 /v1/chat/completions 的本地服务都可以使用。')) +
                tr('\n指令、参数和照片统计只发送到本机服务。'))
        else:
            self.hint.setText(tr('云端服务会收到：你的指令、当前调色参数、照片统计信息和拍摄参数（不含 GPS）')
                              + (tr('，以及一张预览图') if self.attach.isChecked() else '') +
                              tr('。费用按服务商计费。API Key 仅保存在本机') + (tr('，使用 Windows 数据保护加密。') if host.WINDOWS else
                                                       tr('，存放在 macOS 钥匙串中。') if host.MACOS else tr('。')))
        if not local and key == 'custom':
            self.key.setPlaceholderText(tr('如果接口需要，填写 Key'))
        self.show_thinking_note()

    def show_thinking_note(self, *_):
        if self.current:
            self.thinking_note.setText(tr(nl_edit.THINKING_NOTES.get(nl_edit.PROVIDERS[self.current][6], '')))

    def service(self):
        self.store()
        settings = dict(self.settings, provider=self.current, timeout=self.timeout.value(), reasoning=self.reasoning.currentData())
        return nl_edit.active(settings)

    def fetch_models(self):
        service = self.service()
        self.refresh_button.setEnabled(False)
        self.status.setText(tr('正在获取模型列表…'))
        def ok(names):
            self.refresh_button.setEnabled(True)
            current = self.model_name.currentText()
            self.model_name.clear()
            self.model_name.addItems(names)
            self.model_name.setCurrentText(current if current else (names[0] if names else ''))
            self.status.setText(tr('找到 {n_names} 个模型。', n_names=len(names)) if names else tr('服务没有返回模型；可直接填写模型名称。'))
        def fail(error):
            self.refresh_button.setEnabled(True)
            self.status.setText(str(error))
        run_async(self.relay, lambda: nl_edit.list_models(service), ok, fail)

    def test(self):
        service = self.service()
        self.test_button.setEnabled(False)
        self.status.setText(tr('正在测试…'))
        started = time.monotonic()
        messages = [dict(role='system', content='只输出 JSON。'), dict(role='user', content='请回复 {"ok": true}')]
        usage = {}
        def ok(text):
            self.test_button.setEnabled(True)
            try:
                nl_edit.parse_reply(text)
                tokens = f' · {usage["total"]:,} token' if usage else ''
                self.status.setText(tr('连接正常 · {model} · {v:.1f} 秒{tokens}', model=service["model"], v=time.monotonic() - started, tokens=tokens))
            except nl_edit.ServiceError:
                self.status.setText(tr('已连接，但模型没有按要求返回 JSON：') + text.strip()[:120])
        def fail(error):
            self.test_button.setEnabled(True)
            self.status.setText(str(error))
        run_async(self.relay, lambda: nl_edit.complete(service, messages, usage=usage), ok, fail)

    def values(self):
        self.store()
        self.settings.update(provider=self.current, timeout=self.timeout.value(), attach_image=self.attach.isChecked(),
                             voice_language=self.voice.currentData(), auto_send=self.auto_send.isChecked(),
                             reasoning=self.reasoning.currentData())
        return self.settings


class NaturalLanguageMixin:
    def init_natural_language(self):
        self.nl_settings = nl_edit.load_settings()
        self.nl_relay = _Relay(self)
        self.nl_request = 0
        self.nl_busy = False
        self.nl_transcribing = False
        self.nl_plan = None
        self.nl_applied = None
        self.nl_history = []
        self.nl_document = None
        self.nl_started = 0.
        self.nl_usage = {}
        self.nl_recorder = VoiceRecorder(self)
        self.nl_recorder.level.connect(self.nl_level)
        self.nl_recorder.finished.connect(self.nl_recorded)
        self.nl_recorder.failed.connect(self.nl_voice_failed)

    def build_natural_language(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        page = QWidget()
        scroll.setWidget(page)
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 12, 4, 0)
        v.setSpacing(7)
        row = QHBoxLayout()
        self.nl_service = QLabel('')
        self.nl_service.setObjectName('subtle')
        self.nl_service.setWordWrap(True)
        row.addWidget(self.nl_service, 1)
        settings = QToolButton()
        settings.setText(tr('设置'))
        settings.setToolTip(tr('选择本地模型或云端 API'))
        settings.clicked.connect(self.nl_configure)
        row.addWidget(settings)
        v.addLayout(row)
        self.nl_input = InstructionEdit()
        self.nl_input.setPlaceholderText(tr('一句话描述想要的效果，例如：\n天空更蓝，整体暖一点，人物提亮'))
        self.nl_input.setFixedHeight(78)
        self.nl_input.submitted.connect(lambda: None if self.nl_busy else self.nl_send())
        v.addWidget(self.nl_input)
        self.nl_examples = QComboBox()
        for title, text in EXAMPLES:
            self.nl_examples.addItem(title, text)
        self.nl_examples.activated.connect(self.nl_example)
        v.addWidget(self.nl_examples)
        row = QHBoxLayout()
        self.nl_voice = QPushButton(VOICE)
        self.nl_voice.setToolTip(tr('离线语音识别（中文 / English），说完停顿约 1 秒自动结束 · {v}', v=host.keys("Ctrl+Shift+Space")))
        self.nl_voice.clicked.connect(self.nl_toggle_voice)
        row.addWidget(self.nl_voice)
        self.nl_go = self.button(tr('一键修图'), self.nl_send, True)
        # Takes the rest of the row and never widens the narrow left panel (longer languages).
        self.nl_go.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.nl_go.setToolTip(tr('Enter 发送，Shift+Enter 换行'))
        row.addWidget(self.nl_go, 1)
        v.addLayout(row)
        self.nl_meter = QProgressBar()
        self.nl_meter.setRange(0, 100)
        self.nl_meter.setTextVisible(False)
        self.nl_meter.setFixedHeight(4)
        self.nl_meter.setStyleSheet('QProgressBar { background: #171d18; border: 0; border-radius: 2px; }'
                                    'QProgressBar::chunk { background: #b1c498; border-radius: 2px; }')
        self.nl_meter.hide()
        v.addWidget(self.nl_meter)
        self.nl_status = QLabel('')
        self.nl_status.setObjectName('subtle')
        self.nl_status.setWordWrap(True)
        v.addWidget(self.nl_status)
        self.nl_explanation = QLabel('')
        self.nl_explanation.setWordWrap(True)
        self.nl_explanation.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.nl_explanation)
        self.nl_changes = QListWidget()
        self.nl_changes.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.nl_changes.setWordWrap(True)
        self.nl_changes.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.nl_changes.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.nl_changes.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nl_changes.setStyleSheet('QListWidget::item { padding: 2px 4px; }')
        font = QFont(self.nl_changes.font())
        font.setPointSizeF(max(7.5, font.pointSizeF() - .5))
        self.nl_changes.setFont(font)
        self.nl_changes.setMinimumHeight(84)
        v.addWidget(self.nl_changes, 1)
        self.nl_amount = AdjustSlider(tr('应用强度 %'), 0, 150)
        self.nl_amount.default_value = 100
        self.nl_amount.setValue(100)
        self.nl_amount.changed.connect(self.nl_strength)
        self.nl_amount.committed.connect(self.commit)
        v.addWidget(self.nl_amount)
        row = QHBoxLayout()
        self.nl_revert = self.button(tr('撤销本次'), self.nl_undo)
        self.nl_again = self.button(tr('重新生成'), self.nl_regenerate)
        row.addWidget(self.nl_revert)
        row.addWidget(self.nl_again)
        v.addLayout(row)
        self.nl_timer = QTimer(self)
        self.nl_timer.timeout.connect(self.nl_progress)
        from PySide6.QtGui import QAction, QKeySequence
        action = QAction(self)
        action.setShortcut(QKeySequence('Ctrl+Shift+Space'))
        action.triggered.connect(self.nl_toggle_voice)
        self.addAction(action)
        self.nl_show_result(None)
        self.nl_update_service()
        return scroll

    # ----------------------------------------------------------------- state

    def nl_update_service(self):
        service = nl_edit.active(self.nl_settings)
        name = service['model'] or tr('未选择模型')
        self.nl_service.setText(f'{tr("本地") if service["local"] else tr("云端")} · {service["label"].split("（")[0]} · {name}')
        self.nl_service.setToolTip(service['base_url'])

    def refresh_natural_language(self):
        if not hasattr(self, 'nl_go'):
            return
        if self.nl_document != (self.document_token, self.source_path):
            # Another photo: its own conversation and no strength handle on the previous result.
            self.nl_document = (self.document_token, self.source_path)
            self.nl_history = []
            self.nl_plan = self.nl_applied = None
            self.nl_request += 1
            self.nl_set_busy(False)
            self.nl_show_result(None)
        ready = self.source is not None and not self.loading
        self.nl_go.setEnabled(ready or self.nl_busy)
        self.nl_voice.setEnabled(speech.available() and (ready or self.nl_recorder.recording) and not self.nl_transcribing)
        if not speech.available():
            self.nl_voice.setToolTip(tr('未安装语音识别模型（assets/models/sensevoice）'))
        live = self.nl_plan is not None and self.edits == self.nl_applied
        self.nl_amount.setEnabled(live and not self.nl_busy)
        self.nl_revert.setEnabled(live and not self.nl_busy)
        self.nl_again.setEnabled(self.nl_plan is not None and live and not self.nl_busy and bool(self.nl_history))

    def nl_set_busy(self, busy):
        self.nl_busy = busy
        self.nl_go.setText(tr('取消') if busy else tr('一键修图'))
        self.nl_go.setObjectName('' if busy else 'primary')
        self.nl_go.style().unpolish(self.nl_go)
        self.nl_go.style().polish(self.nl_go)
        if busy:
            self.nl_started = time.monotonic()
            self.nl_timer.start(500)
        else:
            self.nl_timer.stop()
        self.refresh_natural_language()

    def nl_progress(self):
        service = nl_edit.active(self.nl_settings)
        self.nl_status.setText(tr('{v} · {model} 正在思考… {v2:.0f} 秒', v=service["label"].split("（")[0], model=service["model"], v2=time.monotonic() - self.nl_started))

    def nl_show_result(self, plan, lines=None):
        self.nl_changes.clear()
        if plan is None:
            self.nl_explanation.setText('')
            self.nl_explanation.hide()
            self.nl_changes.hide()
            self.nl_amount.hide()
            self.nl_revert.hide()
            self.nl_again.hide()
            return
        text = plan.explanation or tr('已按指令调整。')
        if plan.notes:
            text += tr('\n注意：') + tr('；').join(plan.notes)
        self.nl_explanation.setText(text)
        self.nl_changes.addItems(lines or [tr('没有参数变化')])
        for widget in (self.nl_explanation, self.nl_changes, self.nl_amount, self.nl_revert, self.nl_again):
            widget.show()

    def nl_example(self, index):
        text = self.nl_examples.itemData(index)
        if text:
            self.nl_input.setPlainText(text)
            self.nl_input.setFocus()
        self.nl_examples.setCurrentIndex(0)

    def nl_configure(self):
        dialog = SettingsDialog(self, self.nl_settings)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.nl_settings = dialog.values()
            try:
                nl_edit.save_settings(self.nl_settings)
            except OSError as error:
                self.error(tr('无法保存模型设置：') + str(error))
            self.nl_update_service()

    # ----------------------------------------------------------------- requests

    def nl_send(self, instruction=None, base=None, history=None):
        if self.nl_busy:
            self.nl_request += 1
            self.nl_set_busy(False)
            self.nl_status.setText(tr('已取消。'))
            return
        instruction = (instruction if isinstance(instruction, str) else self.nl_input.toPlainText()).strip()
        if self.source is None or self.loading:
            return
        if not instruction:
            self.nl_status.setText(tr('请先输入或说出想要的效果。'))
            return
        self.commit()
        service = nl_edit.active(self.nl_settings)
        base = copy.deepcopy(base if base is not None else self.edits)
        history = list(self.nl_history if history is None else history)
        stats = image = None
        if self.rendered is not None:
            shown = engine.crop_rotate(self.rendered, self.edits)
            stats = nl_edit.image_statistics(shown)
            if self.nl_settings['attach_image']:
                image = nl_edit.preview_jpeg(shown)
        messages = nl_edit.build_messages(instruction, base, self.info, stats, history)
        self.nl_request += 1
        request, sent = self.nl_request, copy.deepcopy(self.edits)
        usage = {}
        self.nl_set_busy(True)
        self.nl_progress()
        def ok(text):
            if request != self.nl_request:
                return
            self.nl_set_busy(False)
            try:
                reply = nl_edit.parse_reply(text)
                # Edited meanwhile: the model's final values apply to what is on screen now.
                plan = nl_edit.plan(self.edits if self.edits != sent else base, reply)
            except nl_edit.ServiceError as error:
                self.nl_status.setText(str(error))
                return
            self.nl_history = history + [(instruction, json.dumps(reply, ensure_ascii=False)[:4000])]
            self.nl_usage = dict(usage)
            self.nl_resolve(plan, instruction)
        def fail(error):
            if request != self.nl_request:
                return
            self.nl_set_busy(False)
            self.nl_status.setText(str(error) if isinstance(error, nl_edit.ServiceError) else tr('请求失败：{error}', error=error))
        run_async(self.nl_relay, lambda: nl_edit.complete(service, messages, image, usage), ok, fail)

    def nl_resolve(self, plan, instruction):
        """Compute AI region masks locally, then apply everything as one step."""
        if not plan.regions:
            return self.nl_finish(nl_edit.attach_regions(plan, []))
        if not self.work.can_start(A.SELECTION):
            plan.notes.append(tr('正在进行其他识别，区域蒙版未创建'))
            plan.regions = []
            return self.nl_finish(nl_edit.attach_regions(plan, []))
        from . import selection
        self.work.begin(A.SELECTION)
        token, source, profile = self.document_token, self.source, copy.deepcopy(self.edits)
        cuda = self.backend_combo.currentIndex() == 0
        kinds = [kind for kind, _ in plan.regions]
        self.nl_status.setText(tr('正在本地识别') + tr('、').join(tr(nl_edit.REGIONS[k][0]) for k in kinds) + '…')
        def work():
            rgb = engine.develop_view(source, profile)
            alphas = []
            for kind in kinds:
                alpha, _ = selection.automatic(rgb, kind, cuda)
                alphas.append(selection.encode(alpha) if float((alpha > .5).mean()) >= .0003 else None)
            return alphas
        def ready(alphas):
            self.work.end(A.SELECTION)
            if token == self.document_token:
                self.nl_finish(nl_edit.attach_regions(plan, alphas))
        def failed(text):
            self.work.end(A.SELECTION)
            if token == self.document_token:
                plan.notes.append(tr('区域识别失败：') + text.strip().splitlines()[-1][:120])
                plan.regions = []
                self.nl_finish(nl_edit.attach_regions(plan, []))
        self.job(work, ready, failed)

    def nl_finish(self, plan):
        lines = nl_edit.changes(plan.base, plan.target)
        self.commit()
        self.nl_plan = plan
        self.edits = copy.deepcopy(plan.target)
        self.nl_applied = copy.deepcopy(self.edits)
        self.clear_preset_selection()
        self.current_mask = min(self.current_mask, len(self.edits['masks']) - 1)
        self.nl_amount.blockSignals(True)
        self.nl_amount.setValue(100)
        self.nl_amount.blockSignals(False)
        self.refresh()
        self.changed()
        self.commit()
        self.nl_show_result(plan, lines)
        tokens = f' · {self.nl_usage["total"]:,} token' if self.nl_usage.get('total') else ''
        self.nl_status.setText((tr('已应用 {n_lines} 项调整', n_lines=len(lines)) if lines else tr('模型认为不需要调整')) + tokens +
                               (tr(' · {v} 可撤销', v=host.keys("Ctrl+Z")) if lines else tr('。')))
        self.nl_status.setToolTip(tr('输入 {v:,} · 输出 {v2:,} token', v=self.nl_usage.get("input", 0), v2=self.nl_usage.get("output", 0)) if tokens else '')
        self.statusBar().showMessage(tr('自然语言修图：') + (plan.explanation[:80] or tr('{n_lines} 项调整', n_lines=len(lines))))
        self.refresh_natural_language()

    def nl_strength(self, value):
        if self.refreshing or self.nl_plan is None:
            return
        if self.edits != self.nl_applied:
            # Edited elsewhere since: the strength handle no longer describes this photo.
            self.refresh_natural_language()
            return
        self.edits = nl_edit.blend(self.nl_plan.base, self.nl_plan.target, value / 100)
        self.nl_applied = copy.deepcopy(self.edits)
        self.refresh()
        self.changed()

    def nl_undo(self):
        if self.nl_plan is None or self.edits != self.nl_applied:
            return
        self.commit()
        self.edits = copy.deepcopy(self.nl_plan.base)
        self.current_mask = min(self.current_mask, len(self.edits['masks']) - 1)
        self.nl_plan = self.nl_applied = None
        self.nl_history = self.nl_history[:-1]
        self.refresh()
        self.changed()
        self.commit()
        self.nl_show_result(None)
        self.nl_status.setText(tr('已撤销本次修图。'))
        self.refresh_natural_language()

    def nl_regenerate(self):
        if self.nl_plan is None or not self.nl_history or self.edits != self.nl_applied:
            return
        instruction = self.nl_history[-1][0]
        base, history = copy.deepcopy(self.nl_plan.base), self.nl_history[:-1]
        self.nl_undo()
        self.nl_history = history
        self.nl_status.setText(tr('正在重新生成…'))
        # Wait for the restored preview so the statistics describe the starting point.
        deadline = time.monotonic() + 10
        def attempt():
            settled = not self.timer.isActive() and not self.render_running
            if settled or time.monotonic() > deadline:
                self.nl_send(instruction, base, history)
            else:
                QTimer.singleShot(60, attempt)
        QTimer.singleShot(60, attempt)

    # ----------------------------------------------------------------- voice

    def nl_toggle_voice(self):
        if self.nl_recorder.recording:
            self.nl_recorder.stop()
            return
        if self.source is None or self.nl_transcribing or not speech.available() or not self.nl_voice.isEnabled():
            return
        if host.MACOS and not self.nl_microphone_permission():
            return
        if self.nl_recorder.start():
            self.nl_voice.setText(tr('停止'))
            self.nl_meter.setValue(0)
            self.nl_meter.show()
            self.nl_status.setText(tr('正在听…说完停顿一下即可'))
            # Load the recognizer while the user speaks.
            threading.Thread(target=lambda: self.nl_preload(), daemon=True).start()

    def nl_preload(self):
        try:
            speech.recognizer()
        except Exception:
            pass

    def nl_microphone_permission(self):
        # Run from source, Python has no Info.plist for Qt to check; macOS asks on behalf of
        # the terminal when the microphone opens.
        if not getattr(sys, 'frozen', False):
            return True
        from PySide6.QtCore import QMicrophonePermission
        from PySide6.QtWidgets import QApplication
        permission = QMicrophonePermission()
        status = QApplication.instance().checkPermission(permission)
        if status == Qt.PermissionStatus.Granted:
            return True
        if status == Qt.PermissionStatus.Undetermined:
            QApplication.instance().requestPermission(permission, self, lambda *_: self.nl_toggle_voice())
            return False
        self.nl_status.setText(tr('没有麦克风权限：请在“系统设置 → 隐私与安全性 → 麦克风”中允许 LUMEN RAW。'))
        return False

    def nl_level(self, value):
        self.nl_meter.setValue(int(value * 100))

    def nl_voice_failed(self, text):
        self.nl_voice.setText(VOICE)
        self.nl_meter.hide()
        self.nl_status.setText(text)
        self.refresh_natural_language()

    def nl_recorded(self, samples):
        self.nl_voice.setText(VOICE)
        self.nl_meter.hide()
        if len(samples) < speech.RATE * .3:
            self.nl_status.setText(tr('录音太短，请再说一次。'))
            return
        self.nl_transcribing = True
        self.nl_status.setText(tr('正在识别语音…'))
        self.refresh_natural_language()
        language = self.nl_settings['voice_language']
        def ok(text):
            self.nl_transcribing = False
            self.refresh_natural_language()
            if not speech.meaningful(text):
                self.nl_status.setText(tr('没有识别到文字，请再说一次。'))
                return
            self.nl_input.setPlainText(text)
            if self.nl_settings['auto_send'] and not self.nl_busy:
                self.nl_send(text)
            else:
                self.nl_status.setText(tr('已识别，按 Enter 或“一键修图”发送。'))
        def fail(error):
            self.nl_transcribing = False
            self.refresh_natural_language()
            self.nl_status.setText(tr('语音识别失败：{error}', error=error))
        run_async(self.nl_relay, lambda: speech.transcribe(samples, language), ok, fail)
