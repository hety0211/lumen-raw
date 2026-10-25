"""The AI models, kept apart from the program (2.1.1).

Up to 2.1.0 the models (about 520 MB of the 600 MB installer) were part of the program files, so
every new version downloaded all of them again.  The packages now keep them in their own folder,
``models`` next to LumenRAW.exe (the macOS app carries them inside its bundle), which an update
leaves alone: an update downloads the program (about 150 MB) and only the models that changed.

Each model file is named after its content, ``<name>.<first 12 hex digits of its SHA-256>.onnx``.
``assets/ai-models.json`` (written by tools/bundle.py) lists the files this version needs; a file
is only ever written under that name after its size and SHA-256 were checked, so a new version of
a model never replaces a file a running program has open, and finding a model needs no hashing.
Missing files are downloaded from LUMEN RAW's own GitHub release (``download``), into the models
folder when it is writable and into the per-user data folder otherwise.

A source checkout has no manifest and uses the pinned float32 files of ``assets/models``.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from . import downloads, host
from .i18n import N_, language, tr

ASSETS = Path(__file__).resolve().parents[1] / 'assets'
SOURCE = ASSETS / 'models'
MANIFEST = ASSETS / 'ai-models.json'
RELEASE = 'https://github.com/hety0211/lumen-raw/releases/download/ai-models/'
#: The update installer moves the models of a 2.1.0 installation here; ``prepare`` adopts them.
IMPORT = 'import'
NAME = re.compile(r'^.+\.[0-9a-f]{12}\.onnx$')
#: What each model is for, as the download dialog lists it.
FEATURES = {
    'skyseg.onnx': N_('天空蒙版'), 'person-deeplab.onnx': N_('人物蒙版'), 'u2netp.onnx': N_('主体 / 背景蒙版'),
    'midas-small.onnx': N_('前景 / 景深蒙版'), 'landscape-seg.onnx': N_('风景蒙版（水面、植被、山体、建筑、地面）'),
    'object-encoder.onnx': N_('点选物体'), 'object-decoder.onnx': N_('点选物体'), 'inpaint-lama.onnx': N_('AI 移除'),
    'realesrgan-x4plus.onnx': N_('AI 超分辨率'), 'realesr-general-x4v3.onnx': N_('AI 超分辨率'),
    'nafnet-sidd.onnx': N_('AI 去杂色'), 'drunet-color.onnx': N_('AI 去杂色'), 'ffdnet-color.onnx': N_('AI 去杂色'),
    'face-yunet.onnx': N_('选片的人脸与闭眼提示'), 'face-landmarks.onnx': N_('选片的人脸与闭眼提示'),
}


class Missing(FileNotFoundError):
    """Models this version needs that are not on this computer yet."""

    def __init__(self, names):
        self.names = list(names)
        comma = '、' if language().startswith('zh') else ', '
        features = comma.join(dict.fromkeys(tr(FEATURES.get(name, name)) for name in self.names))
        super().__init__(tr('{features} 需要的 AI 模型尚未下载，请在“帮助 → AI 模型…”中下载。', features=features))


def manifest():
    """``{'release': …, 'models': {name: {file, bytes, sha256}}}`` in a package, None in a checkout."""
    try:
        return json.loads(MANIFEST.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None


def packaged():
    return MANIFEST.is_file()


def _program_folder():
    """``models`` next to the program (Windows and Linux packages); None for source runs and on macOS."""
    if not getattr(sys, 'frozen', False) or host.MACOS:
        return None
    return Path(sys.executable).resolve().parent / 'models'


def folders():
    """Folders that may hold model files, in the order they are looked in."""
    override = os.environ.get('LUMEN_MODELS')
    result = [Path(override)] if override else []
    program = _program_folder()
    if program is not None and not override:
        result.append(program)
    bundled = getattr(sys, '_MEIPASS', None)
    if bundled and host.MACOS and not override:
        result.append(Path(bundled) / 'models')
    result.append(host.data_folder() / 'models')
    return list(dict.fromkeys(result))


def _writable(folder):
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / f'.write-test-{os.getpid()}'
        probe.write_bytes(b'')
        probe.unlink()
        return True
    except OSError:
        return False


def download_folder():
    """Where downloaded models go: the program's models folder when it is writable."""
    override = os.environ.get('LUMEN_MODELS')
    if override:
        return Path(override)
    program = _program_folder()
    if program is not None and _writable(program):
        return program
    return host.data_folder() / 'models'


def _find(record):
    for folder in folders():
        candidate = folder / record['file']
        try:
            if candidate.stat().st_size == record['bytes']:
                return candidate
        except OSError:
            continue
    return None


def path(name):
    """The file of model ``name`` (e.g. ``'skyseg.onnx'``); ``Missing`` when it is not here yet."""
    data = manifest()
    if data is None:
        return SOURCE / name
    record = data['models'].get(name)
    found = _find(record) if record else None
    if found is None:
        raise Missing([name])
    return found


def available(name):
    try:
        return path(name).is_file()
    except Missing:
        return False


def url(record, data=None):
    """Where a model file is downloaded from; ``LUMEN_MODEL_RELEASE`` serves them locally in the
    package checks (tools/check_update_package.py)."""
    data = data if data is not None else (manifest() or {})
    release = os.environ.get('LUMEN_MODEL_RELEASE') or data.get('release', RELEASE)
    return record.get('url') or release + record['file']


def missing(data=None):
    """Records (with ``name`` and ``url``) of the models in ``data`` (this version's manifest by
    default) that are not on this computer."""
    data = data if data is not None else manifest()
    if not data:
        return []
    return [dict(record, name=name, url=url(record, data)) for name, record in sorted(data['models'].items())
            if _find(record) is None]


def size(records):
    return sum(record['bytes'] for record in records)


def download(records=None, progress=lambda done, total: None, cancelled=lambda: False):
    """Fetch ``records`` (the missing models by default); returns the folder they went to.
    A cancelled download (``downloads.Cancelled``) continues where it stopped next time."""
    records = missing() if records is None else records
    folder = download_folder()
    folder.mkdir(parents=True, exist_ok=True)
    total, done = size(records), 0
    for record in records:
        if _find(record) is None:
            source = record.get('url') or url(record)
            try:
                downloads.fetch_verified(source, folder / record['file'], record['bytes'], record['sha256'],
                                         lambda have: progress(done + have, total), cancelled)
            except downloads.Cancelled:
                raise
            except Exception as exc:
                raise OSError(tr('AI 模型下载失败：') + f'{record["file"]}: {exc}') from exc
        done += record['bytes']
        progress(done, total)
    return folder


def prepare(keep=()):
    """Start-up housekeeping in a package: adopt the models a 2.1.0 installation left in
    ``models/import`` (moved there by the update installer) when they are files this version
    needs, and delete model files no longer needed (``keep``: files of a downloaded update)."""
    data = manifest()
    if data is None:
        return dict(adopted=[], removed=[])
    records = list(data['models'].values())
    wanted = {record['file'] for record in records} | set(keep)
    adopted, removed = [], []
    for folder in folders():
        imports = folder / IMPORT
        if imports.is_dir():
            sizes = {record['bytes'] for record in records}
            by_digest = {record['sha256']: record for record in records}
            for item in sorted(imports.iterdir()):
                try:
                    record = by_digest.get(downloads.sha256(item)) if item.is_file() and item.stat().st_size in sizes else None
                    if record is not None and _find(record) is None:
                        item.replace(folder / record['file'])
                        adopted.append(record['file'])
                    else:
                        item.unlink()
                except OSError:
                    continue
            try:
                imports.rmdir()
            except OSError:
                pass
        try:
            items = list(folder.iterdir())
        except OSError:
            continue
        for item in items:
            name = item.name[:-5] if item.name.endswith('.part') else item.name
            if NAME.match(name) and name not in wanted:
                try:
                    item.unlink()
                    removed.append(item.name)
                except OSError:
                    pass
    return dict(adopted=adopted, removed=removed)


def status():
    """For ``lumen-cli models`` and the dialog: where the models are and what is missing."""
    data = manifest()
    lacking = missing(data)
    return dict(packaged=data is not None, folders=[str(f) for f in folders()], download_folder=str(download_folder()),
                models=len(data['models']) if data else len(list(SOURCE.glob('*.onnx'))),
                missing=[record['name'] for record in lacking], missing_bytes=size(lacking))
