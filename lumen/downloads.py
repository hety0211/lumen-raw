"""Resumable, verified downloads (2.1.1), shared by the speech model, the AI models and updates.

``fetch`` continues a ``.part`` file with an HTTP Range request when the server allows it; the
caller checks the finished file's SHA-256 (``sha256``) before it is used.  Standard library only.
"""
from __future__ import annotations

import hashlib
import urllib.request

from . import __version__
from .i18n import tr

CHUNK = 1 << 20


class Cancelled(Exception):
    pass


def sha256(path, chunk=None):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(chunk or CHUNK), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, part, size, progress, cancelled, chunk=None, timeout=30):
    """Download ``url`` into ``part``, continuing a partial file when the server allows it."""
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
        with open(part, 'ab' if have else 'wb') as stream:
            while have < size:
                if cancelled():
                    raise Cancelled()
                block = response.read(chunk or CHUNK)
                if not block:
                    break
                stream.write(block)
                have += len(block)
                progress(have)
    if have != size:
        raise OSError(tr('下载不完整（{have} / {size} 字节）', have=have, size=size))


def fetch_verified(url, target, size, digest, progress=lambda have: None, cancelled=lambda: False):
    """``fetch`` into ``target.part``, check size and SHA-256, then move it onto ``target``."""
    part = target.with_name(target.name + '.part')
    fetch(url, part, size, progress, cancelled)
    if sha256(part) != digest:
        part.unlink()
        raise OSError(tr('校验失败'))
    part.replace(target)
    return target
