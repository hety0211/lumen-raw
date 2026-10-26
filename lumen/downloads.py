"""Resumable, verified downloads (2.1.1), shared by the speech model, the AI models and updates.

``fetch`` continues a ``.part`` file with an HTTP Range request when the server allows it; the
caller checks the finished file's SHA-256 (``sha256``) before it is used.  Standard library only.

2.1.1: a dropped or stalled connection (common for GitHub's release files on some networks) is
resumed by ``fetch`` itself, again and again while bytes keep arriving, and ``describe`` turns
the operating system's long messages into a short reason the windows can show.
"""
from __future__ import annotations

import errno
import hashlib
import http.client
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


class Cancelled(Exception):
    pass


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
    """``fetch`` into ``target.part``, check size and SHA-256, then move it onto ``target``."""
    part = target.with_name(target.name + '.part')
    fetch(url, part, size, progress, cancelled)
    if sha256(part) != digest:
        part.unlink()
        raise OSError(tr('校验失败'))
    part.replace(target)
    return target
