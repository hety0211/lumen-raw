"""Write ``lumen-update.json``, the update feed the editor reads (2.1.1, lumen/updates.py).

    python tools/make_update_feed.py

Reads the update installer of this version from ``.publish/vXYZ/packages``, the model manifest
the build bundled (``build/ai-models.json``) and the short notes ``.publish/vXYZ/update-notes.json``
(``{"zh_CN": "…", "en": "…", …}``), and writes ``.publish/vXYZ/packages/lumen-update.json``.  It is
published on the Windows release; the editor reads it through ``releases/latest/download/``.
A macOS build adds its own ``macos`` entry when it is published.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from lumen import __version__  # noqa: E402

REPOSITORY = 'https://github.com/hety0211/lumen-raw'
PUBLISH = PROJECT / '.publish' / ('v' + __version__.replace('.', ''))
PACKAGES = PUBLISH / 'packages'


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
    output.write_text(json.dumps(feed, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(dict(feed=str(output), update_bytes=feed['windows']['update']['bytes'],
                          setup_bytes=feed['windows']['setup']['bytes'], models=len(models['models'])), indent=2))


if __name__ == '__main__':
    main()
