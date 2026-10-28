"""Write ``lumen-update.json``, the update feed the editor reads (2.1.1, lumen/updates.py).

    python tools/make_update_feed.py

Reads the update installer of this version from ``.publish/vXYZ/packages``, the model manifest
the build bundled (``build/ai-models.json``) and the short notes ``.publish/vXYZ/update-notes.json``
(``{"zh_CN": "…", "en": "…", …}``), and writes ``.publish/vXYZ/packages/lumen-update.json``.  It is
published on the Windows release; the editor reads it through ``releases/latest/download/``.
A macOS build adds its own ``macos`` entry when it is published.

2.1.2: the feed is signed (Ed25519, ``lumen/signing.py``) into ``lumen-update.json.sig`` with the
release key, ``~/.lumen-release/update-signing.key`` (``LUMEN_SIGNING_KEY`` overrides
the path).  Editors accept a feed from the download mirrors only with this signature; without the
key the feed is written unsigned with a warning, and ``publish_release.ps1`` refuses to publish it.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from lumen import __version__, signing, updates  # noqa: E402

REPOSITORY = 'https://github.com/hety0211/lumen-raw'
PUBLISH = PROJECT / '.publish' / ('v' + __version__.replace('.', ''))
PACKAGES = PUBLISH / 'packages'
KEY = Path(os.environ.get('LUMEN_SIGNING_KEY') or Path.home() / '.lumen-release' / 'update-signing.key')


def sign(feed):
    """Write ``<feed>.sig`` (hex Ed25519 signature of the feed's bytes); False without the key."""
    signature = feed.with_name(feed.name + updates.SIGNATURE)
    if not KEY.is_file():
        signature.unlink(missing_ok=True)
        print(f'warning: {KEY} is missing; the feed is NOT signed and the mirrors cannot serve it', file=sys.stderr)
        return False
    seed = bytes.fromhex(KEY.read_text(encoding='ascii').strip())
    if signing.public_key(seed).hex() not in updates.FEED_KEYS:
        raise SystemExit(f'{KEY} is not a key the editor knows (lumen/updates.py FEED_KEYS)')
    body = feed.read_bytes()
    value = signing.sign(seed, body)
    assert updates.signed(body, value.hex().encode('ascii'))
    signature.write_text(value.hex() + '\n', encoding='ascii')
    return True


def record(path, tag):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(url=f'{REPOSITORY}/releases/download/{tag}/{path.name}', bytes=path.stat().st_size, sha256=digest)


def main():
    tag = f'v{__version__}'
    update = PACKAGES / f'LumenRAW-{__version__}-Update.exe'
    setup = PACKAGES / f'LumenRAW-{__version__}-Setup.exe'
    for path in (update, setup):
        if not path.is_file():
            raise SystemExit(f'missing {path}')
    models = json.loads((PROJECT / 'build' / 'ai-models.json').read_text(encoding='utf-8'))
    notes_file = PUBLISH / 'update-notes.json'
    if notes_file.is_file():
        notes = json.loads(notes_file.read_text(encoding='utf-8'))
    else:
        print(f'warning: {notes_file} is missing; the feed carries a generic note', file=sys.stderr)
        notes = {'zh_CN': f'LUMEN RAW {__version__} 已发布，详见发布页面。', 'en': f'LUMEN RAW {__version__} is out; see the release page.'}
    page = f'{REPOSITORY}/releases/tag/{tag}'
    feed = dict(format=1, name='LUMEN RAW', version=__version__, date=time.strftime('%Y-%m-%d'), page=page, notes=notes,
                windows=dict(version=__version__, page=page, update=record(update, tag), setup=record(setup, tag),
                             models=models))
    output = PACKAGES / 'lumen-update.json'
    output.write_bytes((json.dumps(feed, ensure_ascii=False, indent=1) + '\n').encode('utf-8'))
    signed = sign(output)
    print(json.dumps(dict(feed=str(output), signed=signed, update_bytes=feed['windows']['update']['bytes'],
                          setup_bytes=feed['windows']['setup']['bytes'], models=len(models['models'])), indent=2))


if __name__ == '__main__':
    main()
