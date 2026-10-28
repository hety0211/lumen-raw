"""Resumable, verified downloads (2.1.1), shared by the speech model, the AI models and updates.

``fetch`` continues a ``.part`` file with an HTTP Range request when the server allows it; the
caller checks the finished file's SHA-256 (``sha256``) before it is used.  Standard library only.

2.1.1: a dropped or stalled connection (common for GitHub's release files on some networks) is
resumed by ``fetch`` itself, again and again while bytes keep arriving, and ``describe`` turns
the operating system's long messages into a short reason the windows can show.

2.1.2: LUMEN RAW's release files (update installers, AI and speech models) are mirrored on the
author's servers, copied from GitHub every hour.  A file GitHub does not deliver is fetched from
Singapore, then from Hong Kong (``alternatives``); every caller checks the SHA-256 it got from the
program itself or from the signed update feed (``signing``), so a mirror can never change a file.
"""
from __future__ import annotations

import errno
import hashlib
import http.client
import os
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

from . import __version__
from .i18n import tr

CHUNK = 1 << 20
#: Attempts in a row that bring no new bytes before ``fetch`` gives up (2.1.1).
ATTEMPTS = 3
#: Seconds to wait before the n-th retry in a row without progress.
PAUSES = (1, 3)
#: 2.1.2: GitHub's release files and their mirrors (same paths after the prefix), tried in this order.
GITHUB = 'https://github.com/hety0211/lumen-raw/releases/'
MIRRORS = ('https://103.195.190.106/lumen-raw/releases/',      # Singapore (an IP-address certificate)
           'https://lumenraw.com/lumen-raw/releases/')         # Hong Kong (HK1)
#: The mirror that answered while GitHub did not: tried first for the rest of the session.
_preferred = None


class Cancelled(Exception):
    pass


class Failed(OSError):
    """Every source of a file failed (2.1.2); ``reasons`` holds ``(host, reason)`` pairs."""

    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__('；'.join(f'{host}: {reason}' for host, reason in self.reasons))


def mirrors():
    """Mirror bases; ``LUMEN_MIRRORS`` (comma separated, empty for none) replaces them."""
    override = os.environ.get('LUMEN_MIRRORS')
    if override is not None:
        return tuple(base.strip().rstrip('/') + '/' for base in override.split(',') if base.strip())
    return MIRRORS


def alternatives(url):
    """``url`` and its mirror copies in the order to try them: GitHub, Singapore, Hong Kong (a
    mirror that answered earlier while GitHub did not comes first).  Other URLs have none."""
    if not url.startswith(GITHUB):
        return [url]
    bases = [GITHUB, *mirrors()]
    if _preferred in bases:
        bases.remove(_preferred)
        bases.insert(0, _preferred)
    return [base + url[len(GITHUB):] for base in bases]


def remember(url):
    """Note the source that worked: a mirror is tried first from now on, GitHub resets that."""
    global _preferred
    for base in (GITHUB, *mirrors()):
        if url.startswith(base):
            _preferred = None if base == GITHUB else base
            return


def mirrored(url):
    """Whether ``url`` is a mirror's copy of a GitHub release file."""
    return any(url.startswith(base) for base in mirrors())


def explain(url, exc):
    """``host: reason`` of a failed download, or every host's reason when several were tried."""
    return str(exc) if isinstance(exc, Failed) else f'{host(url)}: {describe(exc)}'


def local(exc):
    """A failure of this computer (folder not writable, disk full): another source cannot help."""
    if isinstance(exc, urllib.error.URLError):
        return False
    return isinstance(exc, (PermissionError, FileNotFoundError, IsADirectoryError, NotADirectoryError)) \
        or getattr(exc, 'errno', None) == errno.ENOSPC


def sha256(path, chunk=None):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(chunk or CHUNK), b''):
            digest.update(block)
    return digest.hexdigest()


def transient(exc):
    """Whether trying again may help: network trouble and server errors, not a file missing on
    the server or a local file that cannot be written."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in (408, 425, 429) or exc.code >= 500
    if isinstance(exc, (PermissionError, FileNotFoundError, IsADirectoryError, NotADirectoryError)) \
            or getattr(exc, 'errno', None) == errno.ENOSPC:
        return False
    return isinstance(exc, (OSError, http.client.HTTPException))


def describe(exc):
    """A short reason for a failed connection or download, in the interface language."""
    if isinstance(exc, urllib.error.HTTPError):
        return tr('服务器返回 HTTP {code}', code=exc.code)
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(reason, socket.gaierror):
        return tr('无法解析服务器地址（DNS）')
    if isinstance(reason, ssl.SSLError):
        return tr('安全连接（TLS）失败')
    if isinstance(reason, TimeoutError):
        return tr('连接超时')
    if isinstance(reason, ConnectionRefusedError):
        return tr('连接被拒绝（请检查代理设置）')
    if isinstance(reason, (ConnectionError, http.client.HTTPException)):
        return tr('连接被中断')
    return str(reason or exc)


def host(url):
    return urllib.parse.urlsplit(url).hostname or url


def _size(path):
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _pause(seconds, cancelled):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if cancelled():
            raise Cancelled()
        time.sleep(min(.2, max(0., end - time.monotonic())))


def _fetch_once(url, part, size, progress, cancelled, chunk, timeout):
    have = part.stat().st_size if part.is_file() else 0
    if have > size:
        part.unlink()
        have = 0
    if have == size:
        return  # complete before an interruption; the caller checks its SHA-256
    headers = {'User-Agent': f'LUMEN-RAW/{__version__}'}
    if have:
        headers['Range'] = f'bytes={have}-'
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
        if have and response.status != 206:
            have = 0
        # read1 returns what has arrived instead of waiting for a whole chunk, so a connection that
        # stalls keeps (and reports) every byte it brought.
        read = getattr(response, 'read1', response.read)
        with open(part, 'ab' if have else 'wb') as stream:
            while have < size:
                if cancelled():
                    raise Cancelled()
                block = read(chunk or CHUNK)
                if not block:
                    break
                stream.write(block)
                have += len(block)
                progress(have)
    if have != size:
        raise OSError(tr('下载不完整（{have} / {size} 字节）', have=have, size=size))


def fetch(url, part, size, progress, cancelled, chunk=None, timeout=30, attempts=None):
    """Download ``url`` into ``part``, continuing a partial file when the server allows it.
    2.1.1: an interrupted transfer is resumed at once while it keeps bringing bytes; after
    ``attempts`` tries in a row without any, the last error is raised."""
    attempts = ATTEMPTS if attempts is None else attempts
    idle = 0
    while True:
        before = _size(part)
        try:
            return _fetch_once(url, part, size, progress, cancelled, chunk, timeout)
        except Cancelled:
            raise
        except Exception as exc:
            if not transient(exc):
                raise
            idle = 0 if _size(part) > before else idle + 1
            if idle >= attempts:
                raise
            _pause(PAUSES[min(idle, len(PAUSES)) - 1] if idle else 0, cancelled)


def fetch_verified(url, target, size, digest, progress=lambda have: None, cancelled=lambda: False):
    """``fetch`` into ``target.part``, check size and SHA-256, then move it onto ``target``.

    2.1.2: a GitHub release file that cannot be fetched is tried from the mirrors
    (``alternatives``); the partial file carries over (the copies are identical), and a source
    whose file fails the check is followed by the next one.  ``Failed`` lists every reason."""
    part = target.with_name(target.name + '.part')
    sources = alternatives(url)
    reasons, last = [], None
    for index, source in enumerate(sources):
        try:
            # With other sources left, a source that brings nothing is given up on sooner.
            fetch(source, part, size, progress, cancelled, attempts=2 if index < len(sources) - 1 else None)
            if sha256(part) != digest:
                part.unlink()
                raise OSError(tr('校验失败'))
            part.replace(target)
            remember(source)
            return target
        except Cancelled:
            raise
        except Exception as exc:
            if local(exc):
                raise
            reasons.append((host(source), describe(exc)))
            last = exc
    if len(reasons) == 1:
        raise last
    raise Failed(reasons)
