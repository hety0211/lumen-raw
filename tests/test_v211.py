"""2.1.1: AI models kept apart from the program, updates from inside the editor, DeepSeek Harness
(MCP) and the natural-language providers checked again (MiMo in; Bailian, Ark, SiliconFlow out)."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from lumen import __version__, ai_models, cli, downloads, nl_edit, updates

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- a file server

class _Files(BaseHTTPRequestHandler):
    files, requests = {}, []

    def log_message(self, *_):
        pass

    def do_GET(self):
        _Files.requests.append((self.path, self.headers.get('Range')))
        data = _Files.files.get(self.path)
        if data is None:
            self.send_error(404)
            return
        start = int(self.headers['Range'].split('=')[1].rstrip('-')) if self.headers.get('Range') else 0
        self.send_response(206 if start else 200)
        self.send_header('Content-Length', str(len(data) - start))
        self.end_headers()
        self.wfile.write(data[start:])


@pytest.fixture
def server():
    _Files.files, _Files.requests = {}, []
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), _Files)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{httpd.server_address[1]}/'
    httpd.shutdown()
    httpd.server_close()


def record(data, name):
    digest = hashlib.sha256(data).hexdigest()
    stem, suffix = name.rsplit('.', 1)
    return dict(file=f'{stem}.{digest[:12]}.{suffix}', bytes=len(data), sha256=digest)


@pytest.fixture
def packaged(tmp_path, monkeypatch, server):
    """A package's model manifest with two small models, served by ``server``."""
    blobs = {'skyseg.onnx': os.urandom(200_000), 'inpaint-lama.onnx': os.urandom(150_000)}
    models = {name: record(data, name) for name, data in blobs.items()}
    manifest = tmp_path / 'ai-models.json'
    manifest.write_text(json.dumps(dict(format=1, release=server + 'ai-models/', models=models)), encoding='utf-8')
    monkeypatch.setattr(ai_models, 'MANIFEST', manifest)
    folder = tmp_path / 'models'
    monkeypatch.setenv('LUMEN_MODELS', str(folder))
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'appdata'))
    for name, data in blobs.items():
        _Files.files[f'/ai-models/{models[name]["file"]}'] = data
    return dict(blobs=blobs, models=models, folder=folder, base=server)


# ----------------------------------------------------------------------------- AI models

def test_a_source_checkout_uses_the_pinned_models_of_assets():
    assert not ai_models.packaged() and ai_models.manifest() is None
    assert ai_models.path('skyseg.onnx') == ROOT / 'assets' / 'models' / 'skyseg.onnx'
    assert ai_models.missing() == []


def test_a_package_finds_models_by_content_name_and_names_the_missing_features(packaged):
    with pytest.raises(ai_models.Missing) as caught:
        ai_models.path('skyseg.onnx')
    assert '天空蒙版' in str(caught.value) and isinstance(caught.value, FileNotFoundError)
    assert not ai_models.available('inpaint-lama.onnx')
    lacking = ai_models.missing()
    assert [r['name'] for r in lacking] == ['inpaint-lama.onnx', 'skyseg.onnx']
    assert lacking[0]['url'] == packaged['base'] + 'ai-models/' + packaged['models']['inpaint-lama.onnx']['file']
    reports = []
    folder = ai_models.download(progress=lambda done, total: reports.append((done, total)))
    assert folder == packaged['folder'] and ai_models.missing() == []
    path = ai_models.path('skyseg.onnx')
    assert path.name == packaged['models']['skyseg.onnx']['file'] and path.read_bytes() == packaged['blobs']['skyseg.onnx']
    assert reports[-1] == (350_000, 350_000) and not list(folder.glob('*.part'))
    # A truncated copy is not a model.
    path.write_bytes(b'x')
    assert not ai_models.available('skyseg.onnx')


def test_a_damaged_download_is_rejected_and_a_cancelled_one_resumes(packaged, monkeypatch):
    monkeypatch.setattr(downloads, 'CHUNK', 16384)
    name = packaged['models']['skyseg.onnx']['file']
    good = packaged['blobs']['skyseg.onnx']
    _Files.files[f'/ai-models/{name}'] = good[:-1] + bytes([good[-1] ^ 1])
    with pytest.raises(OSError, match='AI 模型下载失败'):
        ai_models.download([r for r in ai_models.missing() if r['name'] == 'skyseg.onnx'])
    assert not (packaged['folder'] / name).exists()
    _Files.files[f'/ai-models/{name}'] = good
    seen = []
    with pytest.raises(downloads.Cancelled):
        ai_models.download(progress=lambda done, total: seen.append(done), cancelled=lambda: bool(seen) and seen[-1] >= 60_000 and seen[-1] < 150_000)
    part = next(packaged['folder'].glob('*.part'))
    have = part.stat().st_size
    ai_models.download()
    assert ai_models.missing() == []
    assert any(r == f'bytes={have}-' for _, r in _Files.requests)


def test_start_up_adopts_the_models_of_2_1_0_and_drops_old_versions(packaged):
    folder = packaged['folder']
    imports = folder / ai_models.IMPORT
    imports.mkdir(parents=True)
    (imports / 'skyseg.onnx').write_bytes(packaged['blobs']['skyseg.onnx'])     # still needed
    (imports / 'midas-small.onnx').write_bytes(os.urandom(1000))                # no longer listed
    stale = folder / 'skyseg.0123456789ab.onnx'
    stale.write_bytes(b'old')
    pending = folder / 'landscape-seg.abcdefabcdef.onnx'                       # a downloaded update's
    pending.write_bytes(b'next')
    partial = folder / (packaged['models']['inpaint-lama.onnx']['file'] + '.part')
    partial.write_bytes(b'half')
    (folder / 'notes.txt').write_text('mine')
    result = ai_models.prepare(keep={pending.name})
    assert result['adopted'] == [packaged['models']['skyseg.onnx']['file']]
    assert ai_models.available('skyseg.onnx') and not imports.exists()
    assert not stale.exists() and pending.exists() and partial.exists() and (folder / 'notes.txt').exists()
    assert ai_models.status()['missing'] == ['inpaint-lama.onnx']


def test_the_build_names_models_after_their_content_and_lists_them(tmp_path, monkeypatch):
    from tools import bundle
    compact = tmp_path / 'compact'

    def fake(source, destination, log=print):
        compact.mkdir(exist_ok=True)
        (compact / 'skyseg.onnx').write_bytes(b'sky' * 1000)
        (compact / 'u2netp.onnx').write_bytes(b'u2' * 1000)
        (compact / 'model-info.json').write_text('{}')
        return compact
    monkeypatch.setattr(bundle, 'compact_folder', fake)
    build = tmp_path / 'build'
    (build / 'model-pack').mkdir(parents=True)
    (build / 'model-pack' / 'skyseg.000000000000.onnx').write_bytes(b'old')
    pack, manifest = bundle.model_pack(ROOT, build)
    data = json.loads(manifest.read_text(encoding='utf-8'))
    digest = hashlib.sha256(b'sky' * 1000).hexdigest()
    assert data['release'] == ai_models.RELEASE and data['models']['skyseg.onnx'] == dict(
        file=f'skyseg.{digest[:12]}.onnx', bytes=3000, sha256=digest)
    assert sorted(p.name for p in pack.iterdir()) == sorted(r['file'] for r in data['models'].values())
    program = tmp_path / 'dist' / 'LumenRAW'
    placed = bundle.place_models(ROOT, program, build)
    assert placed == program / 'models' and (placed / f'skyseg.{digest[:12]}.onnx').read_bytes() == b'sky' * 1000
    assert all(ai_models.NAME.match(r['file']) for r in data['models'].values())


def test_the_packaging_tools_keep_models_out_of_the_program(tmp_path):
    from tools import package_windows
    program = tmp_path / 'LumenRAW'
    (program / '_internal' / 'assets').mkdir(parents=True)
    (program / 'models').mkdir()
    models = {}
    for name in package_windows.MODELS:
        data = name.encode() * (1000 if name != 'skyseg.onnx' else 2000)
        models[name] = record(data, name)
        (program / 'models' / models[name]['file']).write_bytes(data)
    (program / '_internal' / 'assets' / 'ai-models.json').write_text(json.dumps(dict(models=models)), encoding='utf-8')
    assert package_windows.check_models(program)['models'] == models
    (program / '_internal' / 'assets' / 'skyseg.onnx').write_bytes(b'x')
    with pytest.raises(SystemExit, match='inside the program'):
        package_windows.check_models(program)
    (program / '_internal' / 'assets' / 'skyseg.onnx').unlink()
    next((program / 'models').glob('u2netp.*')).unlink()
    with pytest.raises(SystemExit, match='u2netp'):
        package_windows.check_models(program)


def test_the_installer_builds_an_update_package_without_models():
    iss = (ROOT / 'installer.iss').read_text(encoding='utf-8')
    assert '#ifdef UpdatePackage' in iss and 'Excludes: "\\models"' in iss
    assert 'OutputBaseFilename=LumenRAW-{#AppVersion}-{#PackageKind}' in iss
    assert '{param:WAITPID|0}' in iss and 'WaitForSingleObject' in iss
    assert 'Check: RelaunchRequested' in iss and '{param:RELAUNCH|0}' in iss
    assert 'Type: filesandordirs; Name: "{app}\\models"' in iss
    assert "KeepModels(ExpandConstant('{app}\\_internal\\assets\\models'), ExpandConstant('{app}\\models\\import'))" in iss
    build = (ROOT / 'tools' / 'build_release.ps1').read_text(encoding='utf-8')
    assert '/DUpdatePackage' in build and 'check_update_package.py' in build and 'make_update_feed.py' in build
    spec = (ROOT / 'LumenRAW.spec').read_text(encoding='utf-8')
    assert "bundle.place_models(root, Path(DISTPATH) / 'LumenRAW')" in spec


# ----------------------------------------------------------------------------- updates

def feed(base, version, models=None, payload=b''):
    package = dict(url=base + f'v{version}/LumenRAW-{version}-Update.exe', bytes=len(payload),
                   sha256=hashlib.sha256(payload).hexdigest())
    entry = dict(version=version, update=package, models=models)
    return dict(format=1, version=version, page=base + 'page', notes={'zh_CN': '新版本的说明', 'en': 'What changed'},
                windows=entry, macos=dict(entry))


@pytest.fixture
def feed_server(server, monkeypatch, tmp_path):
    monkeypatch.setenv('LUMEN_UPDATE_FEED', server + 'latest/lumen-update.json')
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'appdata'))
    return server


def publish(base, data):
    _Files.files['/latest/lumen-update.json'] = json.dumps(data).encode('utf-8')


def test_versions_compare_as_numbers():
    assert updates.newer('2.1.10', '2.1.9') and updates.newer('2.2.0', '2.1.1') and not updates.newer('2.1.1', '2.1.1')
    assert not updates.newer('2.1.0') and updates.newer('99.0.0')


def test_the_feed_offers_only_a_newer_version_for_this_platform(feed_server):
    publish(feed_server, feed(feed_server, __version__))
    assert updates.check() is None
    publish(feed_server, feed(feed_server, '2.9.0'))
    update = updates.check()
    assert update.version == '2.9.0' and update.notes == '新版本的说明' and update.page == feed_server + 'page'
    data = feed(feed_server, '2.9.0')
    data.pop('windows'), data.pop('macos')
    publish(feed_server, data)
    assert updates.check() is None
    bad = feed(feed_server, '2.9.0')
    bad['windows']['update']['sha256'] = 'nope'
    bad['macos'] = bad['windows']
    publish(feed_server, bad)
    with pytest.raises(updates.UpdateError):
        updates.check()
    _Files.files['/latest/lumen-update.json'] = json.dumps(dict(format=99)).encode()
    with pytest.raises(updates.UpdateError):
        updates.check()


def test_only_lumen_raw_releases_are_trusted(monkeypatch):
    monkeypatch.delenv('LUMEN_UPDATE_FEED', raising=False)
    assert updates.trusted('https://github.com/hety0211/lumen-raw/releases/download/v2.1.2/LumenRAW-2.1.2-Update.exe')
    assert not updates.trusted('https://github.com/someone/else/releases/download/v1/x.exe')
    assert not updates.trusted('http://github.com/hety0211/lumen-raw/releases/download/v1/x.exe')
    monkeypatch.setenv('LUMEN_UPDATE_FEED', 'http://127.0.0.1:9/feed.json')
    assert updates.trusted('http://127.0.0.1:9/v2/x.exe') and not updates.trusted('http://127.0.0.1:10/x.exe')


def test_an_installed_copy_downloads_the_program_and_only_changed_models(feed_server, packaged, monkeypatch):
    payload = os.urandom(250_000)
    # This version's models are here; the new version keeps inpaint-lama and changes skyseg.
    ai_models.download()
    new_sky = os.urandom(120_000)
    models = dict(packaged['models'], **{'skyseg.onnx': record(new_sky, 'skyseg.onnx')})
    manifest = dict(format=1, release=feed_server + 'ai-models/', models=models)
    _Files.files['/ai-models/' + models['skyseg.onnx']['file']] = new_sky
    publish(feed_server, feed(feed_server, '2.9.0', manifest, payload))
    _Files.files['/v2.9.0/LumenRAW-2.9.0-Update.exe'] = payload
    monkeypatch.setattr(updates, 'mode', lambda: 'manual')
    update = updates.check()
    assert not update.installable()
    with pytest.raises(updates.UpdateError):
        updates.download(update)
    monkeypatch.setattr(updates, 'mode', lambda: 'installer')
    assert update.installable() and [r['name'] for r in update.missing_models()] == ['skyseg.onnx']
    assert update.download_bytes() == 250_000 + 120_000
    reports = []
    installer = updates.download(update, lambda done, total: reports.append((done, total)))
    assert installer.read_bytes() == payload and installer.parent == updates.updates_folder()
    assert reports[-1] == (370_000, 370_000)
    assert (packaged['folder'] / models['skyseg.onnx']['file']).read_bytes() == new_sky
    # The current version's skyseg stays until the update is installed; the new one is kept.
    assert ai_models.available('skyseg.onnx')
    assert updates.pending()['version'] == '2.9.0' and models['skyseg.onnx']['file'] in updates.keep_models()
    ai_models.prepare(keep=updates.keep_models())
    assert (packaged['folder'] / models['skyseg.onnx']['file']).exists() and ai_models.available('skyseg.onnx')
    assert updates.check().installer_ready() and updates.check().download_bytes() == 0
    command = updates.command(installer)
    assert command[0] == str(installer) and '/SILENT' in command and '/RELAUNCH=1' in command
    assert f'/WAITPID={os.getpid()}' in command
    started = []
    monkeypatch.setattr(updates.subprocess, 'Popen', lambda *a, **k: started.append((a, k)))
    updates.launch(installer)
    assert started and started[0][0][0] == command
    installer.write_bytes(b'damaged')
    with pytest.raises(updates.UpdateError, match='damaged'):
        updates.launch(installer)


def test_downloaded_installers_are_removed_once_the_new_version_runs(feed_server, monkeypatch):
    folder = updates.updates_folder()
    folder.mkdir(parents=True)
    (folder / 'LumenRAW-2.1.1-Update.exe').write_bytes(b'x')
    (folder / 'pending.json').write_text(json.dumps(dict(version=__version__, models=['a'])))
    assert updates.pending() is None and updates.keep_models() == set()
    updates.tidy()
    assert not any(folder.iterdir())


def test_update_checks_never_run_in_tests_and_follow_the_setting(monkeypatch):
    assert not updates.interactive() and not updates.automatic()
    monkeypatch.delenv('QT_QPA_PLATFORM', raising=False)
    monkeypatch.delenv('LUMEN_NO_UPDATE_CHECK', raising=False)
    assert updates.automatic()
    updates.set_automatic(False)
    assert not updates.automatic()
    updates.set_automatic(True)
    assert updates.automatic()


def test_notes_follow_the_interface_language(monkeypatch):
    texts = {'zh_CN': '中文', 'en': 'English'}
    for code, expected in (('zh_CN', '中文'), ('zh_TW', '中文'), ('ja', 'English'), ('en', 'English')):
        monkeypatch.setattr(updates, 'language', lambda c=code: c)
        assert updates.localized(texts) == expected
    assert updates.localized('plain') == 'plain' and updates.localized({}) == ''


def test_the_update_window_offers_downloads_and_installs_on_close(app, feed_server, packaged, monkeypatch):
    from PySide6.QtWidgets import QMainWindow
    from lumen import update_dialog
    payload = os.urandom(90_000)
    manifest = dict(format=1, release=feed_server + 'ai-models/', models=packaged['models'])
    publish(feed_server, feed(feed_server, '2.9.0', manifest, payload))
    _Files.files['/v2.9.0/LumenRAW-2.9.0-Update.exe'] = payload
    monkeypatch.setattr(updates, 'mode', lambda: 'installer')
    asked = []
    parent = QMainWindow()
    parent.closing = False
    parent.models_changed = lambda: None
    parent.request_update_install = lambda installer: asked.append(installer) or False
    dialog = update_dialog.UpdateDialog(parent, updates.check())
    dialog.show()
    assert dialog.primary.text() == '立即更新' and dialog.secondary.text() == '下次提醒'
    assert '程序' in dialog.size_label.text() and 'v2.9.0' in [l.text() for l in dialog.findChildren(type(dialog.size_label))]
    dialog.primary.click()
    wait_until(lambda: dialog.primary.isVisible() and dialog.primary.text() == '重启并更新', 15)
    assert dialog.secondary.text() == '稍后'
    dialog.primary.click()
    assert asked and Path(asked[0]).read_bytes() == payload
    assert '更新未安装' in dialog.status.text()
    monkeypatch.setattr(updates, 'mode', lambda: 'manual')
    opened = []
    monkeypatch.setattr(update_dialog.QDesktopServices, 'openUrl', lambda url: opened.append(url.toString()))
    manual = update_dialog.UpdateDialog(parent, updates.check())
    assert manual.primary.text() == '打开下载页面'
    manual.primary.click()
    assert opened == [feed_server + 'page']
    for path in packaged['folder'].glob('skyseg.*'):        # the update fetched the models; lose one
        path.unlink()
    models = update_dialog.ModelsDialog(parent)
    assert models.primary.text() == '立即下载' and '天空蒙版' in models.text.text()
    models.primary.click()
    wait_until(lambda: not ai_models.missing() and models.primary.text() == '关闭', 15)
    parent.close()


@pytest.fixture
def own_library(tmp_path_factory, monkeypatch):
    """A library and preview cache of the test's own: the window below really closes, and a closed
    window's database must not be the one the next test's window opens."""
    folder = tmp_path_factory.mktemp('own-library')
    monkeypatch.setenv('LUMEN_LIBRARY', str(folder / 'library.db'))
    monkeypatch.setenv('LUMEN_PREVIEW_CACHE', str(folder / 'previews'))


def test_the_window_installs_an_update_only_when_it_closes(own_library, window, monkeypatch, tmp_path):
    titles = [a.text() for a in window.menuBar().actions()]
    help_menu = window.menuBar().actions()[titles.index('帮助')].menu()
    entries = [a.text() for a in help_menu.actions()]
    assert '检查更新…' in entries and 'AI 模型…' in entries and '启动时检查更新' in entries
    installer = tmp_path / 'LumenRAW-2.9.0-Update.exe'
    installer.write_bytes(b'x')
    monkeypatch.setattr(window, 'confirm_close_library', lambda: False)        # the user cancels
    assert not window.request_update_install(installer) and window.update_installer is None
    monkeypatch.setattr(window, 'confirm_close_library', lambda: True)
    assert window.request_update_install(installer) and window.update_installer == installer


from test_ui import app, window, wait_until  # noqa: E402,F401


# ----------------------------------------------------------------------------- DeepSeek Harness

def test_the_dsh_entry_is_a_dsh_mcp_client_row_with_literal_windows_paths(monkeypatch):
    monkeypatch.setattr(cli, 'executable', lambda: [r"C:\Users\O'Neil\AppData\Local\Programs\LUMEN RAW\lumen-cli.exe"])
    entry = cli.dsh_entry()
    assert entry.splitlines()[:4] == ['- insert:', '    - id: mcp-lumen-raw', "      name: '@deepseek-ai/dsh-mcp-client'", '      config:']
    assert "command: 'C:\\Users\\O''Neil\\AppData\\Local\\Programs\\LUMEN RAW\\lumen-cli.exe'" in entry
    assert "args: ['mcp']" in entry and 'serverName: lumen-raw' in entry and 'transport: stdio' in entry
    assert 'toolCallTimeoutMs: 600000' in entry
    assert cli.mcp_config('dsh').endswith(entry) and 'cordis.patch.yml' in cli.mcp_config('dsh')


def test_writing_the_dsh_patch_file_merges_and_keeps_the_users_entries(tmp_path, monkeypatch):
    monkeypatch.setenv('DSH_HOME', str(tmp_path / 'dsh'))
    path = cli.dsh_patch_file()
    assert path == tmp_path / 'dsh' / 'cordis.patch.yml'
    assert cli.install_dsh() == (path, 'created')
    first = path.read_text(encoding='utf-8')
    assert first.startswith(cli.DSH_BEGIN) and first.rstrip().endswith(cli.DSH_END)
    assert cli.install_dsh() == (path, 'updated') and path.read_text(encoding='utf-8') == first
    # A file with the user's own patches: the block is appended, the old file kept once.
    mine = "# my patches\n- insert:\n    - id: memory-engram\n      name: '@deepseek-ai/dsh-mcp-client'\n"
    path.write_text(mine, encoding='utf-8')
    assert cli.install_dsh() == (path, 'added')
    text = path.read_text(encoding='utf-8')
    assert text.startswith(mine) and text.count('mcp-lumen-raw') == 1
    assert path.with_name('cordis.patch.yml.lumen-backup').read_text(encoding='utf-8') == mine
    # An entry the user wrote by hand is left alone; a mapping cannot be merged automatically.
    path.write_text("- insert:\n    - config:\n        serverName: 'lumen-raw'\n", encoding='utf-8')
    assert cli.install_dsh()[1] == 'present'
    path.write_text('plugins:\n  a: 1\n', encoding='utf-8')
    with pytest.raises(ValueError):
        cli.install_dsh()
    path.write_text('# nothing yet\n[]\n', encoding='utf-8')
    assert cli.install_dsh()[1] == 'added' and '[]' not in path.read_text(encoding='utf-8')
    assert cli.main(['mcp-config', 'dsh', '--install']) == 0


def test_the_ai_assistant_window_has_a_deepseek_harness_page(window, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QMessageBox, QPushButton, QTabWidget
    monkeypatch.setenv('DSH_HOME', str(tmp_path / 'dsh'))
    shown = []
    monkeypatch.setattr(QMessageBox, 'information', lambda *a: shown.append(a[2]))
    window.open_agent_dialog()
    tabs = window.agent_dialog.findChild(QTabWidget)
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    assert 'DeepSeek Harness' in titles
    page = tabs.widget(titles.index('DeepSeek Harness'))
    button = next(b for b in page.findChildren(QPushButton) if b.text() == '写入 DSH 配置')
    button.click()
    assert (tmp_path / 'dsh' / 'cordis.patch.yml').is_file() and '重启 DeepSeek Harness' in shown[-1]
    window.agent_dialog.close()


# ----------------------------------------------------------------------------- language models

def test_bailian_ark_and_siliconflow_are_gone_and_mimo_is_offered():
    for removed in ('dashscope', 'siliconflow', 'volcengine'):
        assert removed not in nl_edit.PROVIDERS
    label, api, base, suggested, local, json_mode, style = nl_edit.PROVIDERS['mimo']
    assert (api, base, suggested, local, json_mode) == ('openai', 'https://api.xiaomimimo.com/v1', 'mimo-v2.6-flash', False, 'json_object')
    assert style in nl_edit.THINKING_NOTES and 'mimo' in nl_edit.CLOUD_PROVIDERS


def test_a_setup_on_a_removed_provider_moves_to_the_custom_entry(tmp_path):
    path = tmp_path / 'nl.json'
    stored = dict(provider='dashscope', providers=dict(
        dashscope=dict(base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', model='qwen3.7-flash',
                       key=nl_edit.protect_key('sk-qwen'))))
    path.write_text(json.dumps(stored), encoding='utf-8')
    settings = nl_edit.load_settings(path)
    assert settings['provider'] == 'custom'
    assert settings['providers']['custom'] == dict(base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
                                                  model='qwen3.7-flash', key='sk-qwen')
    assert 'dashscope' not in settings['providers']
    # A custom entry already in use is not overwritten.
    stored['providers']['custom'] = dict(base_url='http://my-proxy/v1', model='x', key='')
    path.write_text(json.dumps(stored), encoding='utf-8')
    settings = nl_edit.load_settings(path)
    assert settings['provider'] == 'ollama' and settings['providers']['custom']['base_url'] == 'http://my-proxy/v1'


def test_claude_uses_api_version_2026_10_21_and_openrouter_its_title_header():
    service = nl_edit.active(dict(nl_edit.default_settings(), provider='anthropic'))
    assert nl_edit._anthropic_headers(service)['anthropic-version'] == '2026-10-21'
    settings = nl_edit.default_settings()
    settings['provider'] = 'openrouter'
    settings['providers']['openrouter']['key'] = 'sk-or'
    headers = nl_edit._openai_headers(nl_edit.active(settings))
    assert headers['X-OpenRouter-Title'] == 'LUMEN RAW' and headers['Authorization'] == 'Bearer sk-or'
    assert 'api-key' not in headers


# ----------------------------------------------------------------------------- sharpness and eyes apart

from test_v200 import photo, scene  # noqa: E402


@pytest.fixture
def library(tmp_path):
    from lumen import library_db
    database = library_db.LibraryDB(tmp_path / 'library.db')
    yield database
    database.close()


def test_sharpness_and_closed_eyes_are_stored_and_requested_apart(tmp_path, library):
    from lumen import culling, library_db
    path = photo(tmp_path / 'a.jpg')
    library.add([str(path)])
    pid = library.id_of(str(path))
    sharp, eyes, both = {library_db.SHARPNESS}, {library_db.EYES}, library_db.BOTH
    assert [r['id'] for r in library.unanalysed(parts=sharp)] == [pid]
    result = culling.analyse(scene(640, 420, seed=2), sharp=True, eyes=False)
    assert result['sharpness'] is not None and result['faces'] is None and result['signature']
    library.set_analysis(pid, result, sharp)
    row = library.record(str(path))
    assert library_db.analysed_parts(row) == sharp and row['analysis'] == 2 and row['sharpness'] == result['sharpness']
    assert library.unanalysed(parts=sharp) == [] and [r['id'] for r in library.unanalysed(parts=eyes)] == [pid]
    assert library_db.lacks(row, both) and not library_db.lacks(row, sharp)
    # The eyes later: the sharpness stays, both parts are there (the value of 2.1.0, so it reads them too).
    library.set_analysis(pid, dict(faces=2, eyes_closed=1, boxes=[[.1, .1, .2, .2, 1]]), eyes)
    row = library.record(str(path))
    assert row['analysis'] == library_db.ANALYSIS and row['sharpness'] == result['sharpness'] and row['eyes_closed'] == 1
    assert library.unanalysed(parts=both) == [] and not library_db.lacks(dict(row, missing=1), both)
    # Eyes only, then sharpness.
    library.set_analysis(pid, {}, ())
    library._db().execute('UPDATE photos SET analysis=0 WHERE id=?', (pid,))
    library.set_analysis(pid, dict(faces=0, eyes_closed=0), eyes)
    assert library.record(str(path))['analysis'] == 3 and library.analysed_parts_of(pid) == eyes


def test_the_background_worker_analyses_only_the_parts_switched_on(tmp_path, library, monkeypatch):
    from lumen import culling, library_db, library_tasks
    path = photo(tmp_path / 'a.jpg')
    library.add([str(path)])
    pid = library.id_of(str(path))
    calls = []
    monkeypatch.setattr(culling, 'analyse', lambda rgb, sharp=True, eyes=True: calls.append((sharp, eyes)) or
                        dict(sharpness=40. if sharp else None, faces=1 if eyes else None, eyes_closed=0 if eyes else None,
                             signature=b'x', boxes=[]))

    class Cache:
        def view(self, p):
            return scene(320, 200)

        def has_thumbnail(self, p):
            return True
    worker = library_tasks.LibraryWorker(library, Cache(), paused=lambda: True)    # no runner threads
    worker.parts = {library_db.EYES}
    worker._run_analysis(pid, str(path))
    assert calls == [(False, True)] and library.analysed_parts_of(pid) == {library_db.EYES}
    worker._run_analysis(pid, str(path), library.analysed_parts_of(pid))     # nothing left for the eyes
    assert len(calls) == 1 and worker.done == 2
    worker.parts = set(library_db.BOTH)
    worker._run_analysis(pid, str(path), library.analysed_parts_of(pid))
    assert calls[-1] == (True, False) and library.analysed_parts_of(pid) == library_db.BOTH
    # Queued rows carry what they have; nothing is queued for work while both parts are off.
    worker.want_analysis([dict(id=pid, path=str(path), analysis=2)])
    assert worker._take() == ('analyse', (pid, str(path), {library_db.SHARPNESS}))
    worker.parts = set()
    assert not worker.analysis_enabled
    worker.want_analysis([dict(id=pid, path=str(path))])
    assert worker._take() == (None, None)
    worker.shutdown()


def test_the_two_settings_start_from_the_old_one_and_show_in_the_hints(monkeypatch):
    from lumen import i18n, library_db, library_view
    saved = {}
    monkeypatch.setattr(i18n, 'setting', lambda key, default=None, path=None: saved.get(key, default))
    assert library_view.analysis_parts() == set(library_db.BOTH)
    saved['library_analysis'] = '0'                         # switched off in 2.0 / 2.1.0
    assert library_view.analysis_parts() == set()
    saved['library_analysis_eyes'] = '1'
    assert library_view.analysis_parts() == {library_db.EYES}
    hint = library_view.hint_text
    assert hint(dict(analysis=0)) == '清晰度与闭眼提示：分析中…'
    assert hint(dict(analysis=0), {library_db.EYES}) == '闭眼提示：分析中…'
    assert hint(dict(analysis=0), set()) == ''
    sharp_only = dict(analysis=2, sharpness=63.)
    assert hint(sharp_only, {library_db.SHARPNESS}) == '清晰度 63'
    assert hint(sharp_only) == '清晰度 63  ·  闭眼分析中…'
    assert '2 张人脸' in hint(dict(analysis=3, faces=2, eyes_closed=1), {library_db.EYES})


def test_the_library_menu_has_two_analysis_switches(window, monkeypatch):
    from lumen import i18n, library_db
    saved = {}
    monkeypatch.setattr(i18n, 'save_setting', lambda key, value, path=None: saved.__setitem__(key, value))
    actions = window.analysis_actions
    titles = {part: action.text() for part, action in actions.items()}
    assert titles == {library_db.SHARPNESS: '后台分析清晰度', library_db.EYES: '后台分析闭眼'}
    actions[library_db.EYES].setChecked(False)
    assert saved == {'library_analysis_eyes': '0'} and window.library_worker.parts == {library_db.SHARPNESS}
    actions[library_db.EYES].setChecked(True)
    assert saved['library_analysis_eyes'] == '1' and window.library_worker.parts == set(library_db.BOTH)


def test_update_windows_use_the_apps_orange_main_button(app, monkeypatch):
    from PySide6.QtWidgets import QMainWindow
    from lumen import update_dialog
    parent = QMainWindow()
    parent.closing = False
    parent.models_changed = lambda: None
    retried = []
    parent.check_for_updates = lambda manual=False: retried.append(manual)
    latest = update_dialog.CheckResult(parent)
    assert latest.windowTitle() == '检查更新' and latest.badge.text() == '已是最新版本' and latest.badge.objectName() == 'okBadge'
    assert latest.primary.objectName() == 'primary' and latest.primary.text() == '关闭' and latest.secondary.isHidden()
    failed = update_dialog.CheckResult(parent, OSError('timed out'))
    assert failed.badge.objectName() == 'warnBadge' and failed.primary.text() == '重试' and failed.secondary.text() == '关闭'
    failed.primary.click()
    assert retried == [True]
    models = update_dialog.ModelsDialog(parent, [])
    assert models.windowTitle() == 'AI 模型' and models.primary.objectName() == 'primary' and models.secondary.objectName() == ''
    from lumen import theme
    assert 'pillPrimary' not in theme.stylesheet('dark') and 'QPushButton#primary { background: #f0a530' in theme.stylesheet('dark')
    parent.close()


# ----------------------------------------------------------------------------- review of 2.1.1

from test_v210 import _patch, _pending_removal  # noqa: E402


def test_review_1_a_removal_finishing_while_switching_away_stays_on_its_photo(window, monkeypatch, tmp_path):
    w = window
    monkeypatch.setattr(w, 'confirm_close_library', lambda: True)   # no "save the album?" at teardown
    first = w.source_path
    _, ready, _ = _pending_removal(w, monkeypatch)
    other = tmp_path / 'other.png'
    other.write_bytes(Path(first).read_bytes())
    w.open_path(str(other))
    assert w.loading and w.source_path == first               # still decoding the other photo
    ready(_patch())                                           # the removal finishes now
    wait_until(lambda: not w.loading and w.source_path == str(other.resolve()) and not w.render_running)
    assert not w.edits['retouch'] and len(w.documents[first]['edits']['retouch']) == 1
    w.open_path(first)
    wait_until(lambda: not w.loading and w.source_path == first and not w.render_running)
    assert len(w.edits['retouch']) == 1
    w.undo(-1)
    assert not w.edits['retouch']


def test_review_1_a_removal_finishing_while_switching_back_is_kept(window, monkeypatch, tmp_path):
    w = window
    monkeypatch.setattr(w, 'confirm_close_library', lambda: True)
    first = w.source_path
    _, ready, _ = _pending_removal(w, monkeypatch)
    other = tmp_path / 'other.png'
    other.write_bytes(Path(first).read_bytes())
    w.open_path(str(other))
    wait_until(lambda: not w.loading and w.source_path == str(other.resolve()) and not w.render_running)
    w.open_path(first)
    assert w.loading                                          # the old copy of its edits was taken
    ready(_patch())
    wait_until(lambda: not w.loading and w.source_path == first and not w.render_running)
    assert len(w.edits['retouch']) == 1
    w.open_path(str(other))                                   # stashing it again keeps the removal
    wait_until(lambda: not w.loading and not w.render_running)
    assert len(w.documents[first]['edits']['retouch']) == 1


def test_review_2_keychain_access_allowed_while_saving_keeps_the_key(monkeypatch, tmp_path):
    monkeypatch.setattr(nl_edit, 'KEYCHAIN', True)
    monkeypatch.setattr(nl_edit, '_keychain_cache', None)
    item = {'value': json.dumps({'openai': 'FAKE-REVIEW-KEY'}).encode(), 'refuse': True}

    def read():
        if item['refuse']:
            raise OSError('User denied macOS Keychain access')
        return item['value']

    def write(value):
        item.update(value=value)
    monkeypatch.setattr(nl_edit, '_keychain_read', read)
    monkeypatch.setattr(nl_edit, '_keychain_write', write)
    settings = nl_edit.default_settings()
    settings['providers']['openai']['key'] = 'keychain:'
    path = tmp_path / 'natural-language.json'
    path.write_text(json.dumps(settings), encoding='utf-8')
    loaded = nl_edit.load_settings(path)
    assert loaded['providers']['openai']['key'] == '' and loaded['keychain_unread'] == ['openai']
    item['refuse'] = False                                    # allowed while saving
    nl_edit.save_settings(loaded, path)
    assert json.loads(path.read_text(encoding='utf-8'))['providers']['openai']['key'] == 'keychain:'
    assert loaded['providers']['openai']['key'] == 'FAKE-REVIEW-KEY' and 'keychain_unread' not in loaded
    monkeypatch.setattr(nl_edit, '_keychain_cache', None)
    again = nl_edit.load_settings(path)
    assert again['providers']['openai']['key'] == 'FAKE-REVIEW-KEY'
    nl_edit.save_settings(again, path)
    assert json.loads(item['value']) == {'openai': 'FAKE-REVIEW-KEY'}


def test_review_2_a_removed_providers_key_moves_to_the_custom_entry(monkeypatch, tmp_path):
    monkeypatch.setattr(nl_edit, 'KEYCHAIN', True)
    monkeypatch.setattr(nl_edit, '_keychain_cache', None)
    item = {'value': json.dumps({'dashscope': 'FAKE-QWEN-KEY'}).encode(), 'refuse': True}

    def read():
        if item['refuse']:
            raise OSError('denied')
        return item['value']
    monkeypatch.setattr(nl_edit, '_keychain_read', read)
    monkeypatch.setattr(nl_edit, '_keychain_write', lambda value: item.update(value=value))
    path = tmp_path / 'natural-language.json'
    path.write_text(json.dumps(dict(provider='dashscope', providers=dict(dashscope=dict(
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', model='qwen3.7-flash', key='keychain:')))),
        encoding='utf-8')
    loaded = nl_edit.load_settings(path)
    assert loaded['provider'] == 'custom' and loaded['keychain_unread'] == ['dashscope']
    item['refuse'] = False
    nl_edit.save_settings(loaded, path)
    assert loaded['providers']['custom']['key'] == 'FAKE-QWEN-KEY'
    assert json.loads(item['value']) == {'custom': 'FAKE-QWEN-KEY'}
    assert json.loads(path.read_text(encoding='utf-8'))['providers']['custom']['key'] == 'keychain:'


class _Cache:
    def view(self, p):
        return scene(320, 200)

    def has_thumbnail(self, p):
        return True


def _record_analyse(monkeypatch, calls):
    from lumen import culling
    monkeypatch.setattr(culling, 'analyse', lambda rgb, sharp=True, eyes=True: calls.append((sharp, eyes)) or
                        dict(sharpness=40. if sharp else None, faces=1 if eyes else None, eyes_closed=0 if eyes else None,
                             signature=b'x', boxes=[]))


def test_review_3_closed_eyes_wait_for_the_face_models(tmp_path, library, monkeypatch):
    from lumen import ai_models, culling, library_db, library_tasks
    path = photo(tmp_path / 'a.jpg')
    library.add([str(path)])
    pid = library.id_of(str(path))
    calls = []
    _record_analyse(monkeypatch, calls)
    monkeypatch.setattr(culling, 'faces', lambda: None)       # models not downloaded yet
    worker = library_tasks.LibraryWorker(library, _Cache(), paused=lambda: True)
    worker._run_analysis(pid, str(path))
    assert calls == [(True, False)] and library.analysed_parts_of(pid) == {library_db.SHARPNESS}
    assert [r['id'] for r in library.unanalysed(parts={library_db.EYES})] == [pid]
    worker._run_analysis(pid, str(path), library.analysed_parts_of(pid))      # nothing to decode for
    assert len(calls) == 1
    monkeypatch.setattr(culling, 'faces', lambda: object())   # the models arrived
    worker._run_analysis(pid, str(path), library.analysed_parts_of(pid))
    assert calls[-1] == (False, True) and library.analysed_parts_of(pid) == library_db.BOTH
    worker.shutdown()
    # The views do not queue photos for the eyes alone while the face model files are missing.
    assert library_tasks.possible(library_db.BOTH) == set(library_db.BOTH)
    monkeypatch.setattr(ai_models, 'available', lambda name: False)
    assert library_tasks.possible(library_db.BOTH) == {library_db.SHARPNESS}


def test_review_3_and_5_the_window_queues_only_what_is_missing(own_library, window, monkeypatch, tmp_path):
    """From the window's entry points to the parts actually computed."""
    from lumen import culling, library_db, library_view
    calls = []
    _record_analyse(monkeypatch, calls)
    monkeypatch.setattr(culling, 'faces', lambda: object())
    w = window
    worker = w.library_worker
    running = worker.paused
    worker.paused = lambda: True                              # set the photo up before it is analysed
    path = photo(tmp_path / 'lib' / 'a.jpg')
    w.library_db.add([str(path)])
    pid = w.library_db.id_of(str(path))
    w.library_db.set_analysis(pid, dict(sharpness=55.), {library_db.SHARPNESS})
    for row in w.library_db.rows():                           # the fixture's own photo is not part of this
        if row['id'] != pid:
            w.library_db.set_analysis(row['id'], {}, library_db.BOTH)
    worker._analyse.clear()
    worker.paused = running
    worker.parts = set(library_db.BOTH)
    w.want_library_work(w.library_db.rows())
    wait_until(lambda: w.library_db.analysed_parts_of(pid) == library_db.BOTH, 20)
    assert calls == [(False, True)] and w.library_db.record(str(path))['sharpness'] == 55.
    # Models downloaded later: what still lacks a part is queued again, with what it has.
    w.library_db._db().execute('UPDATE photos SET analysis=2 WHERE id=?', (pid,))
    w.models_changed()
    wait_until(lambda: w.library_db.analysed_parts_of(pid) == library_db.BOTH, 20)
    assert calls[-1] == (False, True) and len(calls) == 2
    assert library_view.analysis_requests([dict(id=1, path='p', analysis=3), dict(id=2, path='q', analysis=1)],
                                          library_db.BOTH) == [dict(id=1, path='p', analysis=3)]


def test_review_4_a_partial_update_download_survives_a_restart(feed_server, packaged, monkeypatch):
    monkeypatch.setattr(downloads, 'CHUNK', 16384)
    monkeypatch.setattr(updates, 'mode', lambda: 'installer')
    payload = os.urandom(200_000)
    manifest = dict(format=1, release=feed_server + 'ai-models/', models={})
    publish(feed_server, feed(feed_server, '2.9.0', manifest, payload))
    _Files.files['/v2.9.0/LumenRAW-2.9.0-Update.exe'] = payload
    folder = updates.updates_folder()
    folder.mkdir(parents=True)
    (folder / 'LumenRAW-2.8.0-Update.exe.part').write_bytes(b'older offer')
    update = updates.check()
    seen = []
    with pytest.raises(downloads.Cancelled):
        updates.download(update, lambda done, total: seen.append(done), lambda: bool(seen) and seen[-1] >= 50_000)
    part = update.installer_path().with_name(update.installer_path().name + '.part')
    have = part.stat().st_size
    assert 0 < have < len(payload) and updates.pending()['version'] == '2.9.0'
    assert not (folder / 'LumenRAW-2.8.0-Update.exe.part').exists()
    updates.tidy()                                            # the next start
    assert part.stat().st_size == have
    installer = updates.download(updates.check())
    assert installer.read_bytes() == payload
    assert any(r == f'bytes={have}-' for p, r in _Files.requests if p.endswith('Update.exe'))
