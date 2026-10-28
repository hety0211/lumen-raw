"""Checking for and installing new versions from inside the editor (2.1.1).

Each Windows release on GitHub carries ``lumen-update.json`` (tools/make_update_feed.py): the
version, short notes per language, the update installer (``LumenRAW-<version>-Update.exe``, the
program without the AI models) and the AI-model manifest of that version.  The editor reads the
copy of the newest release, ``…/releases/latest/download/lumen-update.json``, which needs no API
call and sends nothing but the program version in its User-Agent.

An installed copy (Inno Setup's ``unins000.exe`` next to it) downloads the update installer and
only the models that changed (``ai_models``), checks every file's size and SHA-256, and after the
window has closed runs the installer silently; the installer waits for this process to exit
(``/WAITPID``) and opens the new version (``/RELAUNCH=1``).  Portable copies, source runs and the
macOS app open the release page instead.

2.1.2: when GitHub cannot be reached the feed is read from LUMEN RAW's mirrors, Singapore first,
then Hong Kong (``downloads.alternatives``), and the files are downloaded from the mirror that
answered.  A feed read from a mirror counts only with a valid Ed25519 signature
(``lumen-update.json.sig``, ``signing``) by a key in ``FEED_KEYS``; its SHA-256 sums then vouch
for every file, wherever it comes from.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import __version__, ai_models, downloads, host, signing
from .i18n import language, setting, save_setting, tr

FEED = 'https://github.com/hety0211/lumen-raw/releases/latest/download/lumen-update.json'
RELEASES = 'https://github.com/hety0211/lumen-raw/releases'
#: 2.1.1: the newest Windows release, offered for downloading by hand when a check or download fails.
LATEST = RELEASES + '/latest'
#: Files are only downloaded from LUMEN RAW's own releases (a test feed may serve its own).
TRUSTED = ('https://github.com/hety0211/lumen-raw/releases/download/',)
#: settings.ini key: check when the editor starts ('1', the default) or only from the Help menu.
SETTING = 'check_updates'
FEED_FORMAT = 1
#: 2.1.2: public keys of the release signing key (its private half stays on the publishing computer).
FEED_KEYS = ('7093f7608ed06406192a688b698084c78f11aa309b68d74ea7caf857d7646c19',)
#: The feed's detached signature: hex, next to the feed on the release.
SIGNATURE = '.sig'
#: The mirrors' download page (Hong Kong), for downloading by hand.
MIRROR_PAGE = 'https://lumenraw.com/download/'


class UpdateError(Exception):
    pass


def feed_url():
    return os.environ.get('LUMEN_UPDATE_FEED') or FEED


def interactive():
    """False in tests, the smoke test and other windowless runs: no start-up checks or windows."""
    return not (os.environ.get('LUMEN_NO_UPDATE_CHECK') or os.environ.get('QT_QPA_PLATFORM') == 'offscreen')


def automatic():
    """Whether the editor checks for a new version when it starts."""
    return interactive() and setting(SETTING, '1') != '0'


def set_automatic(on):
    save_setting(SETTING, '1' if on else '0')


def version_key(text):
    return tuple(int(part) for part in re.findall(r'\d+', str(text))[:4])


def newer(version, than=__version__):
    return version_key(version) > version_key(than)


def platform_key():
    return 'macos' if host.MACOS else 'windows'


def release_page():
    """Where to download the newest version by hand: the latest release (Windows; the macOS
    releases are not marked latest, so macOS gets the list of releases)."""
    return RELEASES if host.MACOS else LATEST


def mode():
    """``'installer'``: this copy was installed by the Windows installer and updates itself;
    ``'manual'``: portable, source or macOS, the user downloads the new version."""
    if os.name == 'nt' and getattr(sys, 'frozen', False):
        if (Path(sys.executable).resolve().parent / 'unins000.exe').is_file():
            return 'installer'
    return 'manual'


def trusted(url):
    if url.startswith(TRUSTED):
        return True
    override = os.environ.get('LUMEN_UPDATE_FEED')
    if override:
        base = urllib.parse.urlsplit(override)
        return urllib.parse.urlsplit(url)[:2] == base[:2]
    return False


def localized(texts):
    """The text of a ``{language: text}`` mapping in the interface language."""
    if isinstance(texts, str):
        return texts
    if not isinstance(texts, dict) or not texts:
        return ''
    code = language()
    order = [code, 'zh_CN' if code.startswith('zh') else 'en', 'en', 'zh_CN']
    return next((texts[key] for key in order if texts.get(key)), next(iter(texts.values())))


class Update:
    """A newer version for this platform, read from the feed."""

    def __init__(self, data, entry):
        self.data, self.entry = data, entry
        self.version = str(entry.get('version') or data['version'])
        self.date = str(data.get('date', ''))
        self.page = str(entry.get('page') or data.get('page') or RELEASES)
        #: 2.1.2: the server the feed came from (a mirror when GitHub could not be reached).
        self.source = str(data.get('_source') or '')
        self.notes = localized(entry.get('notes') or data.get('notes'))
        self.package = entry.get('update')
        self.models = entry.get('models')
        for record in [self.package] + list((self.models or {}).get('models', {}).values()):
            if record and not (isinstance(record.get('bytes'), int) and re.fullmatch(r'[0-9a-f]{64}', str(record.get('sha256')))):
                raise UpdateError('invalid record in the update feed')

    def installable(self):
        return mode() == 'installer' and bool(self.package) and trusted(self.package['url'])

    def missing_models(self):
        return ai_models.missing(self.models) if self.models else []

    def download_bytes(self):
        """What ``download`` still fetches: the installer unless it is here, and changed models."""
        package = 0 if self.installer_ready() else self.package['bytes'] if self.package else 0
        return package + ai_models.size(self.missing_models())

    def installer_path(self):
        name = Path(urllib.parse.urlsplit(self.package['url']).path).name
        return updates_folder() / name

    def installer_ready(self):
        if not self.package:
            return False
        path = self.installer_path()
        return path.is_file() and path.stat().st_size == self.package['bytes']


def updates_folder():
    return host.data_folder() / 'updates'


def _get(url, timeout, limit=1 << 20):
    request = urllib.request.Request(url, headers={'User-Agent': f'LUMEN-RAW/{__version__}', 'Cache-Control': 'no-cache'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(limit)


def signed(body, signature):
    """Whether ``signature`` (the hex text of ``lumen-update.json.sig``) signs ``body`` with a
    key of ``FEED_KEYS``."""
    try:
        raw = bytes.fromhex(signature.decode('ascii').strip())
    except (UnicodeDecodeError, ValueError):
        return False
    return any(signing.verify(bytes.fromhex(key), body, raw) for key in FEED_KEYS)


def fetch_feed(timeout=15):
    """The update feed: from GitHub, or (2.1.2) when that fails from the Singapore mirror, then the
    Hong Kong one, whose copy must be signed.  ``UpdateError`` gives every server's reason."""
    reasons = []
    for url in downloads.alternatives(feed_url()):
        where = downloads.host(url)
        try:
            body = _get(url, timeout)
            if downloads.mirrored(url):
                try:
                    signature = _get(url + SIGNATURE, timeout, 4096)
                except urllib.error.HTTPError:
                    signature = b''                 # a feed published without one
                if not signed(body, signature):
                    raise UpdateError(tr('镜像上的更新信息没有有效签名'))
            data = json.loads(body.decode('utf-8'))
            if not isinstance(data, dict) or not isinstance(data.get('format'), int) or data['format'] > FEED_FORMAT:
                raise UpdateError('unsupported update feed')
        except UpdateError as exc:
            reasons.append((where, str(exc)))
        except ValueError:
            reasons.append((where, tr('服务器返回的更新信息无法读取')))
        except (OSError, http.client.HTTPException) as exc:
            reasons.append((where, downloads.describe(exc)))
        else:
            downloads.remember(url)
            data['_source'] = where
            return data
    raise UpdateError('；'.join(f'{where}: {reason}' for where, reason in reasons))


def check(current=__version__, timeout=15):
    """The newer ``Update`` for this platform, or None when this is the newest version."""
    data = fetch_feed(timeout)
    entry = data.get(platform_key())
    if not isinstance(entry, dict):
        return None
    update = Update(data, entry)
    return update if newer(update.version, current) else None


def download(update, progress=lambda done, total: None, cancelled=lambda: False):
    """Fetch the update installer and the models the new version adds or changes; returns the
    installer.  A cancelled download continues where it stopped."""
    if not update.installable():
        raise UpdateError('this copy cannot install updates')
    models = update.missing_models()
    for record in models:
        if not trusted(record['url']):
            raise UpdateError(f'untrusted model source {record["url"]}')
    total = update.download_bytes()
    folder = updates_folder()
    folder.mkdir(parents=True, exist_ok=True)
    installer = update.installer_path()
    # 2.1.1: recorded before anything is fetched, so start-up housekeeping (tidy, keep_models) keeps
    # the partial files of this version; what an older offer left is deleted now.
    _write_pending(update, installer, models)
    for item in folder.iterdir():
        if item.is_file() and item.name not in (installer.name, installer.name + '.part', 'pending.json'):
            try:
                item.unlink()
            except OSError:
                pass
    done = 0
    if not update.installer_ready():
        size = update.package['bytes']
        try:
            downloads.fetch_verified(update.package['url'], installer, size, update.package['sha256'],
                                     lambda have: progress(have, total), cancelled)
        except downloads.Cancelled:
            raise
        except Exception as exc:
            raise UpdateError(downloads.explain(update.package['url'], exc)) from exc
        done = size
    elif downloads.sha256(installer) != update.package['sha256']:
        installer.unlink()
        return download(update, progress, cancelled)
    if models:
        ai_models.download(models, lambda have, _: progress(done + have, total), cancelled)
    progress(total, total)
    return installer


def _write_pending(update, installer, models):
    record = dict(version=update.version, installer=str(installer), sha256=update.package['sha256'],
                  bytes=update.package['bytes'], models=sorted(r['file'] for r in (update.models or {}).get('models', {}).values()))
    path = updates_folder() / 'pending.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=1), encoding='utf-8')
    temporary.replace(path)


def pending():
    """The downloaded update still to be installed, or None."""
    try:
        record = json.loads((updates_folder() / 'pending.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or not newer(record.get('version', '0'), __version__):
        return None
    return record


def keep_models():
    """Model files of a downloaded update that start-up housekeeping must not delete."""
    record = pending()
    return set(record.get('models', [])) if record else set()


def tidy():
    """Delete downloaded installers once their version (or a newer one) is running; a download of
    a newer version, finished or not, stays."""
    folder = updates_folder()
    if not folder.is_dir() or pending() is not None:
        return
    for item in folder.iterdir():
        if item.is_file():
            try:
                item.unlink()
            except OSError:
                pass


def command(installer):
    return [str(installer), '/SILENT', '/SUPPRESSMSGBOXES', '/NORESTART', f'/WAITPID={os.getpid()}', '/RELAUNCH=1']


def launch(installer, expected=None):
    """Start the update installer, detached; it waits for this process to exit."""
    installer = Path(installer)
    record = pending()
    digest = expected or (record or {}).get('sha256')
    if digest and downloads.sha256(installer) != digest:
        raise UpdateError('the downloaded update is damaged')
    flags = getattr(subprocess, 'DETACHED_PROCESS', 0) | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
    return subprocess.Popen(command(installer), creationflags=flags, close_fds=True)
