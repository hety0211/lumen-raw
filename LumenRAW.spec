# Windows x64 build; collects the active ONNX Runtime (Windows ML build with DirectML) and windowsml.
# 1.8.0: float16-weight models, no bundled speech model, unused Qt pieces dropped (tools/bundle.py).
# 2.1.1: the models go to dist/LumenRAW/models, outside _internal (see the end).
from pathlib import Path
import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_data_files

root = Path(SPECPATH)
sys.path.insert(0, str(root))
from tools import bundle

raw_data, raw_binaries, raw_hidden = collect_all('rawpy')
ort_data, ort_binaries, ort_hidden = collect_all('onnxruntime', filter_submodules=bundle.runtime_module,
                                                 exclude_datas=bundle.ORT_EXCLUDED_DATAS)
try:
    winml_data, winml_binaries, winml_hidden = collect_all('windowsml')
except Exception:
    winml_data, winml_binaries, winml_hidden = [], [], []
a = Analysis(
    [str(root / 'main.py')], pathex=[str(root)],
    binaries=raw_binaries + ort_binaries + winml_binaries,
    datas=bundle.asset_datas(root) + [(str(root / 'lumen' / 'locales'), 'lumen/locales')] + raw_data + ort_data + winml_data + collect_data_files('tifffile'),
    hiddenimports=raw_hidden + ort_hidden + winml_hidden + ['PIL.ImageCms', 'PySide6.QtMultimedia', 'PySide6.QtNetwork'],
    excludes=bundle.EXCLUDES, noarchive=False,
)
if os.name == 'nt':
    # Qt 6.11 uses the Windows ICU shim. A different ICU on PATH must not shadow
    # System32/icuuc.dll: it exports versioned symbols and prevents Qt loading.
    a.binaries = [item for item in a.binaries
                  if Path(item[0]).name.lower() != 'icuuc.dll'
                  and not Path(item[0]).name.lower().startswith('icudt')]
a.binaries = bundle.trim(a.binaries)
a.datas = bundle.trim(a.datas)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='LumenRAW',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, disable_windowed_traceback=False, icon=str(root/'assets/lumen.ico'))
# 1.6.0: the same program as a console executable: the command line and the MCP server (stdio).
cli = EXE(pyz, a.scripts, [], exclude_binaries=True, name='lumen-cli',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=True, icon=str(root/'assets/lumen.ico'))
coll = COLLECT(exe, cli, a.binaries, a.datas, strip=False, upx=False, name='LumenRAW')
# 2.1.1: the AI models sit beside the program in models/, apart from _internal, which an update
# replaces (lumen/ai_models.py).
bundle.place_models(root, Path(DISTPATH) / 'LumenRAW')
