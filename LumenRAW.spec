# Windows x64 build; collects the active ONNX Runtime (Windows ML build with DirectML) and windowsml.
from pathlib import Path
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files

root = Path(SPECPATH)
raw_data, raw_binaries, raw_hidden = collect_all('rawpy')
ort_data, ort_binaries, ort_hidden = collect_all('onnxruntime')
try:
    winml_data, winml_binaries, winml_hidden = collect_all('windowsml')
except Exception:
    winml_data, winml_binaries, winml_hidden = [], [], []
a = Analysis(
    [str(root / 'main.py')], pathex=[str(root)],
    binaries=raw_binaries + ort_binaries + winml_binaries,
    datas=[(str(root / 'assets'), 'assets'), (str(root / 'lumen' / 'locales'), 'lumen/locales')] + raw_data + ort_data + winml_data + collect_data_files('tifffile'),
    hiddenimports=raw_hidden + ort_hidden + winml_hidden + ['PIL.ImageCms', 'PySide6.QtMultimedia'],
    excludes=['cupy', 'torch', 'torchvision', 'onnx', 'sympy'], noarchive=False,
)
if os.name == 'nt':
    # Qt 6.11 uses the Windows ICU shim. A different ICU on PATH must not shadow
    # System32/icuuc.dll: it exports versioned symbols and prevents Qt loading.
    a.binaries = [item for item in a.binaries
                  if Path(item[0]).name.lower() != 'icuuc.dll'
                  and not Path(item[0]).name.lower().startswith('icudt')]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='LumenRAW',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, disable_windowed_traceback=False, icon=str(root/'assets/lumen.ico'))
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='LumenRAW')
