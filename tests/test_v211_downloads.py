"""2.1.1 (fix after release): downloads that survive dropped and stalled connections, short failure
reasons, and links for downloading by hand when a check or a download fails."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import hashlib
import socket
import ssl
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from lumen import ai_models, downloads, speech, updates
from test_v211 import _Files, feed, feed_server, packaged, publish, record, server  # noqa: F401
from test_ui import app, window, wait_until  # noqa: E402,F401


@pytest.fixture(autouse=True)
def no_pauses(monkeypatch):
    monkeypatch.setattr(downloads, 'PAUSES', (0, 0))


# ----------------------------------------------------------------------------- a flaky server

class _Flaky(BaseHTTPRequestHandler):
    """Serves ``data``; the first ``drops`` responses end after ``cut`` bytes, ``stall`` seconds
    of silence come before the end of the first response."""
    data, cut, drops, stall, requests = b'', 0, 0, 0, []

    def log_message(self, *_):
        pass

    def do_GET(self):
        _Flaky.requests.append(self.headers.get('Range'))
        start = int(self.headers['Range'].split('=')[1].rstrip('-')) if self.headers.get('Range') else 0
        self.send_response(206 if start else 200)
        self.send_header('Content-Length', str(len(_Flaky.data) - start))
        self.end_headers()
        body = _Flaky.data[start:]
        if _Flaky.stall and len(_Flaky.requests) == 1:
            self.wfile.write(body[:len(body) // 2])
            self.wfile.flush()
            time.sleep(_Flaky.stall)
            return
        if _Flaky.drops:
            _Flaky.drops -= 1
            self.wfile.write(body[:_Flaky.cut])
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(body)


@pytest.fixture
def flaky():
    _Flaky.data, _Flaky.cut, _Flaky.drops, _Flaky.stall, _Flaky.requests = os.urandom(400_000), 0, 0, 0, []
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), _Flaky)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{httpd.server_address[1]}/file.bin'
    httpd.shutdown()
    httpd.server_close()


def test_a_connection_dropped_again_and_again_still_finishes(flaky, tmp_path):
    _Flaky.cut, _Flaky.drops = 60_000, 5
    data = _Flaky.data
    target = downloads.fetch_verified(flaky, tmp_path / 'file.bin', len(data), hashlib.sha256(data).hexdigest())
    assert target.read_bytes() == data
    # Every retry continued where the previous connection stopped.
    assert _Flaky.requests == [None] + [f'bytes={60_000 * n}-' for n in range(1, 6)]


def test_a_stalled_connection_is_resumed(flaky, tmp_path):
    _Flaky.stall = 2
    data = _Flaky.data
    part = tmp_path / 'file.bin.part'
    downloads.fetch(flaky, part, len(data), lambda have: None, lambda: False, timeout=.5)
    assert part.read_bytes() == data and _Flaky.requests[1] == f'bytes={len(data) // 2}-'


def test_without_progress_the_download_gives_up_with_a_short_reason(flaky, tmp_path):
    _Flaky.cut, _Flaky.drops = 0, 10
    part = tmp_path / 'file.bin.part'
    with pytest.raises(OSError) as caught:
        downloads.fetch(flaky, part, len(_Flaky.data), lambda have: None, lambda: False)
    assert len(_Flaky.requests) == downloads.ATTEMPTS
    assert '下载不完整' in downloads.describe(caught.value)


def test_a_missing_file_is_not_asked_for_again(server, tmp_path):
    with pytest.raises(urllib.error.HTTPError) as caught:
        downloads.fetch(server + 'nothing.bin', tmp_path / 'x.part', 10, lambda have: None, lambda: False)
    assert len(_Files.requests) == 1 and downloads.describe(caught.value) == '服务器返回 HTTP 404'


def test_a_cancel_during_the_pause_before_a_retry_stops_at_once(flaky, tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, 'PAUSES', (30, 30))
    _Flaky.cut, _Flaky.drops = 0, 10
    stop = threading.Timer(.3, lambda: flag.append(True))
    flag = []
    stop.start()
    started = time.monotonic()
    with pytest.raises(downloads.Cancelled):
        downloads.fetch(flaky, tmp_path / 'f.part', len(_Flaky.data), lambda have: None, lambda: bool(flag))
    assert time.monotonic() - started < 5


def test_failure_reasons_are_short_and_readable():
    describe = downloads.describe
    assert describe(urllib.error.URLError(TimeoutError('timed out'))) == '连接超时'
    assert describe(TimeoutError()) == '连接超时'
    assert describe(urllib.error.URLError(ConnectionRefusedError(10061, 'x'))) == '连接被拒绝（请检查代理设置）'
    assert describe(urllib.error.URLError(socket.gaierror(11001, 'getaddrinfo failed'))) == '无法解析服务器地址（DNS）'
    assert describe(urllib.error.URLError(ssl.SSLCertVerificationError('bad'))) == '安全连接（TLS）失败'
    assert describe(ConnectionResetError(10054, 'reset')) == '连接被中断'
    assert describe(urllib.error.HTTPError('u', 503, 'busy', {}, None)) == '服务器返回 HTTP 503'
    assert downloads.host('https://github.com/hety0211/lumen-raw/releases/latest') == 'github.com'
    # Local files that cannot be written are not network trouble.
    assert not downloads.transient(PermissionError()) and downloads.transient(ConnectionResetError())
    assert downloads.transient(urllib.error.HTTPError('u', 503, 'busy', {}, None))
    assert not downloads.transient(urllib.error.HTTPError('u', 404, 'gone', {}, None))


def test_an_unreachable_feed_gives_a_short_reason(monkeypatch, tmp_path):
    monkeypatch.setenv('LUMEN_UPDATE_FEED', 'http://127.0.0.1:9/lumen-update.json')
    with pytest.raises(updates.UpdateError) as caught:
        updates.check(timeout=3)
    assert str(caught.value) == '127.0.0.1: 连接被拒绝（请检查代理设置）'


def test_a_model_download_failure_names_the_server_and_the_file(packaged):
    _Files.files.clear()
    with pytest.raises(OSError) as caught:
        ai_models.download()
    name = packaged['models']['inpaint-lama.onnx']['file']
    assert str(caught.value) == f'AI 模型下载失败：127.0.0.1: 服务器返回 HTTP 404 ({name})'


def test_the_manual_download_pages():
    assert updates.release_page() == 'https://github.com/hety0211/lumen-raw/releases/latest'
    assert ai_models.PAGE == 'https://github.com/hety0211/lumen-raw/releases/tag/ai-models'
    assert speech.release_page() == 'https://github.com/hety0211/lumen-raw/releases/tag/speech-model-1'


# ----------------------------------------------------------------------------- the windows

def _parent():
    from PySide6.QtWidgets import QMainWindow
    parent = QMainWindow()
    parent.closing = False
    parent.models_changed = lambda: None
    parent.check_for_updates = lambda manual=False: None
    return parent


def _links(label):
    import re
    return re.findall(r'href="([^"]+)"', label.text())


def test_a_failed_check_links_the_latest_release(app, monkeypatch):
    from lumen import update_dialog
    opened = []
    monkeypatch.setattr(update_dialog.QDesktopServices, 'openUrl', lambda url: opened.append(url.toString()))
    parent = _parent()
    failed = update_dialog.CheckResult(parent, updates.UpdateError('github.com: 连接超时'))
    assert failed.help.isVisibleTo(failed) and '手动下载最新版本' in failed.help.text()
    assert _links(failed.help) == [updates.LATEST] and 'github.com/hety0211/lumen-raw/releases/latest</a>' in failed.help.text()
    failed.help.linkActivated.emit(updates.LATEST)
    assert opened == [updates.LATEST]
    latest = update_dialog.CheckResult(parent)
    assert not hasattr(latest, 'help')
    parent.close()


def test_a_failed_update_download_links_its_release_and_retries(app, feed_server, packaged, monkeypatch):
    from lumen import update_dialog
    payload = os.urandom(50_000)
    publish(feed_server, feed(feed_server, '2.9.0', dict(format=1, models={}), payload))
    monkeypatch.setattr(updates, 'mode', lambda: 'installer')
    parent = _parent()
    dialog = update_dialog.UpdateDialog(parent, updates.check())
    dialog.show()
    assert not dialog.help.isVisible()
    dialog.primary.click()                                     # the installer is not on the server
    wait_until(lambda: dialog.primary.isVisible() and dialog.primary.text() == '重试', 15)
    assert dialog.status.text() == '更新下载失败：127.0.0.1: 服务器返回 HTTP 404'
    assert dialog.help.isVisible() and _links(dialog.help) == [feed_server + 'page']
    assert '已包含全部 AI 模型' in dialog.help.text()
    _Files.files['/v2.9.0/LumenRAW-2.9.0-Update.exe'] = payload
    dialog.primary.click()
    assert not dialog.help.isVisible()
    wait_until(lambda: dialog.primary.isVisible() and dialog.primary.text() == '重启并更新', 15)
    assert not dialog.help.isVisible()
    parent.close()


def test_a_failed_model_download_offers_the_installer_the_files_and_the_folder(app, packaged, monkeypatch):
    from lumen import update_dialog
    opened = []
    monkeypatch.setattr(update_dialog.QDesktopServices, 'openUrl', lambda url: opened.append(url))
    blobs = dict(packaged['blobs'])
    _Files.files.clear()
    parent = _parent()
    dialog = update_dialog.ModelsDialog(parent)
    dialog.show()
    dialog.primary.click()
    wait_until(lambda: dialog.primary.isVisible() and dialog.primary.text() == '重试', 15)
    assert '服务器返回 HTTP 404' in dialog.status.text() and dialog.help.isVisible()
    links = _links(dialog.help)
    assert links[:2] == [updates.release_page(), ai_models.PAGE] and links[2].startswith('file:')
    assert '打开模型文件夹' in dialog.help.text()
    folder = ai_models.download_folder()
    import shutil
    shutil.rmtree(folder, ignore_errors=True)
    dialog.help.linkActivated.emit(links[2])
    assert folder.is_dir() and opened[-1].isLocalFile() and Path(opened[-1].toLocalFile()) == folder
    # Files downloaded in a browser, put into the folder: “Retry” finds them.
    for name, data in blobs.items():
        (folder / packaged['models'][name]['file']).write_bytes(data)
    _Files.requests.clear()
    dialog.primary.click()
    wait_until(lambda: dialog.primary.isVisible() and dialog.primary.text() == '关闭', 15)
    assert ai_models.missing() == [] and not dialog.help.isVisible()
    assert not [r for r in _Files.requests if r[0].startswith('/ai-models/')]
    parent.close()


def test_a_failed_speech_download_links_the_release_and_the_folder(window, monkeypatch, tmp_path):
    import numpy as np
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(speech, 'BUNDLED', tmp_path / 'none')
    monkeypatch.setattr(speech, 'user_folder', lambda: tmp_path / 'user')
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.StandardButton.Yes)

    def failing(progress, cancelled):
        raise OSError('语音识别模型下载失败：hf-mirror.com: 连接超时')
    monkeypatch.setattr(speech, 'download', failing)
    window.source = np.zeros((4, 4, 3), np.float32)
    try:
        window.refresh_natural_language()
        window.nl_toggle_voice()
        wait_until(lambda: window.nl_download is None, 10)
        assert '连接超时' in window.nl_status.text() and not window.nl_help.isHidden()
        links = _links(window.nl_help)
        assert links[0] == speech.release_page() and links[1].startswith('file:')
        assert 'model.int8.onnx' in window.nl_help.text() and '打开语音模型文件夹' in window.nl_help.text()
        monkeypatch.setattr(speech, 'download', lambda progress, cancelled: tmp_path)
        window.nl_toggle_voice()
        wait_until(lambda: window.nl_download is None, 10)
        assert window.nl_help.isHidden()
    finally:
        window.source = None
        window.saved_edits = window.edits
