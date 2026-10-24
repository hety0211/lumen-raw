"""What the PyInstaller specs put into the packages (1.8.0), shared by LumenRAW.spec and
LumenRAW-macOS.spec.

* The AI models are bundled as the float16-weight copies of ``tools/compact_models.py``.
* The SenseVoice speech model is not bundled: ``lumen.speech`` downloads it on first use.
* Packages that only reach the build through optional imports (pandas via
  onnxruntime.transformers, Tcl/Tk via tifffile, lxml, …) are excluded, and so are the Qt pieces
  the app never loads (software OpenGL, the virtual keyboard with Qt Quick / QML, the PDF image
  plugin, Qt Multimedia's FFmpeg backend, TLS backends, unused platform plugins and translations).
"""
from __future__ import annotations

import re
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


def asset_datas(root, platform='windows'):
    """(source, destination folder) pairs for ``assets``; the models come from the compact copies."""
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
    models = compact_folder(root / 'assets' / 'models', root / 'build' / 'compact-models')
    pairs += [(str(path), 'assets/models') for path in sorted(models.iterdir()) if path.is_file()]
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
