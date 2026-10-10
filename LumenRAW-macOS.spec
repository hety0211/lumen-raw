# macOS Apple silicon build (1.3.1): ONNX Runtime with Core ML, PyObjC Metal and the
# pure-Perl ExifTool.  Produces "dist/LUMEN RAW.app"; run it through tools/build_macos.sh,
# which draws build/macos/LumenRAW.icns first and signs the bundle afterwards.
import os
import re
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files

root = Path(SPECPATH)
version = re.search(r"__version__ = '([0-9.]+)'",
                    (root / 'lumen' / '__init__.py').read_text(encoding='utf-8')).group(1)
art = root / 'build' / 'macos'
minimum = os.environ.get('LUMEN_MACOS_MINIMUM', '15.0')

raw_data, raw_binaries, raw_hidden = collect_all('rawpy')
ort_data, ort_binaries, ort_hidden = collect_all('onnxruntime')

# Windows-only runtime files (exiftool.exe with its Strawberry Perl) are not shipped on macOS;
# the Perl ExifTool from tools/fetch_exiftool.py runs with /usr/bin/perl instead.
assets = []
for item in sorted((root / 'assets').iterdir()):
    if item.name == 'exiftool':
        assets.append((str(item / 'unix'), 'assets/exiftool/unix'))
    elif item.is_dir():
        assets.append((str(item), f'assets/{item.name}'))
    else:
        assets.append((str(item), 'assets'))

a = Analysis(
    [str(root / 'main.py')], pathex=[str(root)],
    binaries=raw_binaries + ort_binaries,
    datas=assets + [(str(root / 'lumen' / 'locales'), 'lumen/locales')] + raw_data + ort_data + collect_data_files('tifffile'),
    hiddenimports=raw_hidden + ort_hidden + ['PIL.ImageCms', 'objc', 'Foundation', 'Metal', 'PySide6.QtMultimedia',
                                             'PySide6.QtNetwork'],
    excludes=['cupy', 'torch', 'torchvision', 'onnx', 'sympy', 'windowsml', 'tkinter'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='LumenRAW',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, disable_windowed_traceback=False, argv_emulation=False,
          target_arch='arm64', codesign_identity=None, entitlements_file=None)
# 1.6.0: Contents/MacOS/lumen-cli, the command line and the MCP server (stdio).
cli = EXE(pyz, a.scripts, [], exclude_binaries=True, name='lumen-cli',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=True, argv_emulation=False, target_arch='arm64', codesign_identity=None, entitlements_file=None)
coll = COLLECT(exe, cli, a.binaries, a.datas, strip=False, upx=False, name='LumenRAW')

document = lambda name, types, rank='Alternate': {
    'CFBundleTypeName': name, 'CFBundleTypeRole': 'Editor', 'LSHandlerRank': rank, 'LSItemContentTypes': types}
exported = lambda identifier, description, extension: {
    'UTTypeIdentifier': identifier, 'UTTypeDescription': description, 'UTTypeConformsTo': ['public.json'],
    'UTTypeIconFile': 'LumenRAW.icns', 'UTTypeTagSpecification': {'public.filename-extension': [extension]}}
app = BUNDLE(
    coll, name='LUMEN RAW.app', icon=str(art / 'LumenRAW.icns'),
    bundle_identifier='io.github.hety0211.lumenraw', version=version,
    info_plist={
        'CFBundleName': 'LUMEN RAW',
        'CFBundleDisplayName': 'LUMEN RAW',
        'CFBundleShortVersionString': version,
        'CFBundleVersion': version,
        'CFBundleDevelopmentRegion': 'zh_CN',
        # 1.5.1: the interface languages, so native panels follow the same language list.
        'CFBundleLocalizations': ['zh_CN', 'zh_TW', 'en', 'ja', 'ko', 'de', 'fr', 'es', 'ru'],
        'CFBundleAllowMixedLocalizations': True,
        'LSMinimumSystemVersion': minimum,
        'LSArchitecturePriority': ['arm64'],
        'LSApplicationCategoryType': 'public.app-category.photography',
        'NSHighResolutionCapable': True,
        'NSHumanReadableCopyright': 'LUMEN RAW · MIT License',
        # 1.5.0: voice instructions are recognized offline on this Mac.
        'NSMicrophoneUsageDescription': 'LUMEN RAW 在本机离线识别你的语音修图指令，录音不会上传。',
        # Every RAW type macOS knows (ARW, CR2 / CR3, NEF, RAF, RW2, DNG, ...) conforms to public.camera-raw-image.
        'CFBundleDocumentTypes': [
            document('Camera RAW', ['public.camera-raw-image']),
            document('Image', ['public.jpeg', 'public.png', 'public.tiff']),
            document('LUMEN RAW 工程', ['io.github.hety0211.lumenraw.project'], 'Owner'),
            document('LUMEN RAW 选片集', ['io.github.hety0211.lumenraw.album'], 'Owner'),
        ],
        'UTExportedTypeDeclarations': [
            exported('io.github.hety0211.lumenraw.project', 'LUMEN RAW 工程', 'lumen'),
            exported('io.github.hety0211.lumenraw.album', 'LUMEN RAW 选片集', 'lumenalbum'),
        ],
    },
)
