"""What the PyInstaller specs put into the packages (1.8.0), shared by LumenRAW.spec and
LumenRAW-macOS.spec.

* The AI models are bundled as the float16-weight copies of ``tools/compact_models.py``.
  2.1.1: they are kept apart from the program (``lumen/ai_models.py``): ``model_pack`` names each
  copy after its SHA-256 in ``build/model-pack`` and lists them in ``assets/ai-models.json``; the
  Windows package keeps them in ``models`` next to LumenRAW.exe, which updates leave alone.
* The SenseVoice speech model is not bundled: ``lumen.speech`` downloads it on first use.
* Packages that only reach the build through optional imports (pandas via
  onnxruntime.transformers, Tcl/Tk via tifffile, lxml, …) are excluded, and so are the Qt pieces
  the app never loads (software OpenGL, the virtual keyboard with Qt Quick / QML, the PDF image
  plugin, Qt Multimedia's FFmpeg backend, TLS backends, unused platform plugins and translations).
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

from tools.compact_models import compact_folder

#: Modules no code path of the app needs; excluding them keeps their hooks from collecting data.
EXCLUDES = ['cupy', 'torch', 'torchvision', 'onnx', 'sympy', 'tkinter', '_tkinter', 'pandas', 'lxml', 'matplotlib',
            'IPython', 'scipy', 'dateutil', 'pytest', 'onnxruntime.transformers', 'onnxruntime.tools',
            'onnxruntime.quantization', 'onnxruntime.training', 'numpy.f2py', 'PySide6.QtQuick', 'PySide6.QtQml',
            'PySide6.QtOpenGL', 'PySide6.QtPdf']
#: ONNX Runtime's Python tooling (model conversion, quantization, benchmarks) and its data files.
ORT_TOOLING = ('onnxruntime.transformers', 'onnxruntime.tools', 'onnxruntime.quantization', 'onnxruntime.training')
ORT_EXCLUDED_DATAS = ['**/transformers/**', '**/tools/**', '**/quantization/**', '**/training/**']
#: Assets read only by the build tools, or fetched at run time.
SKIP_ASSETS = {'runtime-assets.json', 'culling-assets.json', 'editing-assets.json', 'models'}
#: Where packaged programs download model files they do not have (lumen/ai_models.py).
MODEL_RELEASE = 'https://github.com/hety0211/lumen-raw/releases/download/ai-models/'
#: Interface languages (lumen/i18n.py); Qt's own strings (dialog buttons) come from qtbase_<code>.qm.
QT_LANGUAGES = ('zh_CN', 'zh_TW', 'ja', 'ko', 'de', 'fr', 'es', 'ru')
#: Bundled Qt files the app never loads (matched against the destination path, lower case).
QT_UNUSED = re.compile('|'.join([
    r'(^|/)(lib)?qt6(quick|qml|qmlmodels|qmlmeta|qmlworkerscript|virtualkeyboard|pdf)[^/]*\.(dll|dylib|so[.0-9]*)$',
    r'(^|/)qt(quick|qml|qmlmodels|qmlmeta|qmlworkerscript|virtualkeyboard|pdf)\.framework/',
    r'/plugins/platforminputcontexts/',
    r'/plugins/imageformats/(lib)?qpdf\.',
    r'/plugins/platforms/(lib)?q(direct2d|minimal)\.',
    r'/plugins/tls/',
]))
#: Windows only (checked there against every bundled DLL's imports): the Mesa software OpenGL
#: fallback and Qt OpenGL, and Qt Multimedia's FFmpeg backend: the microphone is recorded
#: through the Windows Media Foundation backend (``QT_MEDIA_BACKEND=windows``, lumen/__init__.py).
QT_UNUSED_WINDOWS = re.compile('|'.join([
    r'(^|/)opengl32sw\.dll$',
    r'(^|/)qt6opengl\.dll$',
    r'/plugins/multimedia/ffmpegmediaplugin\.dll$',
    r'(^|/)(avcodec|avformat|avutil|swscale|swresample)-[0-9]+\.dll$',
]))


def runtime_module(name):
    """collect_all filter: ONNX Runtime modules the app imports (not its tooling)."""
    return not any(name == tool or name.startswith(tool + '.') for tool in ORT_TOOLING)


def _sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def model_pack(root, build=None, log=print):
    """2.1.1: the compact models under their content names in ``build/model-pack`` and the
    manifest ``build/ai-models.json`` listing them (bundled as ``assets/ai-models.json``).
    Returns ``(pack folder, manifest path)``; tests pass their own ``build`` folder."""
    root = Path(root)
    build = Path(build) if build else root / 'build'
    compact = compact_folder(root / 'assets' / 'models', build / 'compact-models', log)
    pack = build / 'model-pack'
    pack.mkdir(parents=True, exist_ok=True)
    models = {}
    for path in sorted(compact.glob('*.onnx')):
        digest = _sha256(path)
        name = f'{path.stem}.{digest[:12]}{path.suffix}'
        target = pack / name
        # Copies, not links: compact_folder rewrites its files in place.
        if not (target.is_file() and target.stat().st_size == path.stat().st_size):
            shutil.copyfile(path, target)
        models[path.name] = dict(file=name, bytes=path.stat().st_size, sha256=digest)
    files = {record['file'] for record in models.values()}
    for stale in pack.iterdir():
        if stale.name not in files:
            stale.unlink()
    manifest = build / 'ai-models.json'
    manifest.write_text(json.dumps(dict(format=1, release=MODEL_RELEASE, models=models), indent=1) + '\n',
                        encoding='utf-8')
    return pack, manifest


def model_files(root, build=None):
    """The model files of ``model_pack`` (the macOS bundle carries them in ``models``)."""
    pack, _ = model_pack(root, build)
    return sorted(path for path in pack.iterdir() if path.is_file())


def place_models(root, program, build=None):
    """Copy the model pack into ``<program>/models`` (the Windows onedir build)."""
    pack, _ = model_pack(root, build)
    target = Path(program) / 'models'
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for path in sorted(pack.iterdir()):
        shutil.copy2(path, target / path.name)
    return target


def asset_datas(root, platform='windows', build=None):
    """(source, destination folder) pairs for ``assets``; 2.1.1: the models themselves are not
    part of it, only their manifest (``model_pack``)."""
    root = Path(root)
    pairs = []
    for item in sorted((root / 'assets').iterdir()):
        if item.name in SKIP_ASSETS:
            continue
        if item.name == 'exiftool' and platform in ('macos', 'linux'):
            # Windows' exiftool.exe with its Strawberry Perl is not shipped on macOS or Linux;
            # the Perl ExifTool from tools/fetch_exiftool.py runs with the system perl instead.
            pairs.append((str(item / 'unix'), 'assets/exiftool/unix'))
        elif item.is_dir():
            pairs.append((str(item), f'assets/{item.name}'))
        else:
            pairs.append((str(item), 'assets'))
    _, manifest = model_pack(root, build)
    pairs.append((str(manifest), 'assets'))
    return pairs


def unused(destination, platform='windows'):
    """True for a bundled file (its destination path) that the app never loads."""
    path = '/' + str(destination).replace('\\', '/').lower()
    if '/translations/' in path and path.endswith('.qm'):
        name = path.rsplit('/', 1)[1]
        return not any(name == f'qtbase_{code.lower()}.qm' for code in QT_LANGUAGES)
    return bool(QT_UNUSED.search(path) or (platform == 'windows' and QT_UNUSED_WINDOWS.search(path)))


def trim(entries, platform='windows'):
    """A PyInstaller TOC without the unused Qt files; entries are (destination, source, kind).

    2.1.0: a symbolic link goes with the file it points to.  On macOS PyInstaller adds one per
    framework at the top of the bundle (``('QtQuick', '…/QtQuick.framework/Versions/A/QtQuick',
    'SYMLINK')``); left pointing at a trimmed framework, ``codesign --verify`` rejected the app."""
    return [entry for entry in entries if not unused(entry[0], platform)
            and not (len(entry) > 2 and entry[2] == 'SYMLINK' and unused(entry[1], platform))]
