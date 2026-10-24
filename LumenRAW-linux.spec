# Linux x86_64 build (2.1.0): ONNX Runtime's CUDA provider, CuPy native pixel kernels, the Perl
# ExifTool and the Qt FFmpeg multimedia backend.  Run tools/build_linux.sh, which installs the
# build requirements and packages dist/LumenRAW as a portable .tar.gz.
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files

root = Path(SPECPATH)
sys.path.insert(0, str(root))
from tools import bundle

version = re.search(r"__version__ = '([0-9.]+)'",
                    (root / 'lumen' / '__init__.py').read_text(encoding='utf-8')).group(1)

raw_data, raw_binaries, raw_hidden = collect_all('rawpy')
ort_data, ort_binaries, ort_hidden = collect_all('onnxruntime', filter_submodules=bundle.runtime_module,
                                                 exclude_datas=bundle.ORT_EXCLUDED_DATAS)
try:
    cupy_data, cupy_binaries, cupy_hidden = collect_all('cupy')
except Exception:
    cupy_data, cupy_binaries, cupy_hidden = [], [], []
# CuPy 14 asks ``cuda.pathfinder`` (a separate wheel) for the pip CUDA libraries; it is
# imported dynamically, so PyInstaller's graph cannot see it and the frozen app would report
# "缺少 CuPy".  Collect the package explicitly.  ``cupy_backends.cuda._softlink`` is also
# imported from a Cython initialiser rather than a normal ``import`` statement.
try:
    cuda_py_data, cuda_py_binaries, cuda_py_hidden = collect_all('cuda')
except Exception:
    cuda_py_data, cuda_py_binaries, cuda_py_hidden = [], [], []
try:
    cpb_data, cpb_binaries, cpb_hidden = collect_all('cupy_backends')
except Exception:
    cpb_data, cpb_binaries, cpb_hidden = [], [], []

# The pip CUDA/cuDNN wheels live outside onnxruntime; put them next to it so lumen.cuda_env can
# find them in the frozen runtime folder and add unversioned symlinks before ONNX Runtime loads.
cuda_binaries = []
try:
    from lumen import cuda_env
    for folder in cuda_env.library_folders():
        package = folder.parent.name
        for library in sorted(folder.glob('*.so*')):
            cuda_binaries.append((str(library), f'nvidia/{package}/lib'))
except Exception:
    pass

assets = bundle.asset_datas(root, 'linux')

a = Analysis(
    [str(root / 'main.py')], pathex=[str(root)],
    binaries=raw_binaries + ort_binaries + cupy_binaries + cuda_py_binaries + cpb_binaries + cuda_binaries,
    datas=(assets + [(str(root / 'lumen' / 'locales'), 'lumen/locales')]
           + raw_data + ort_data + cupy_data + cuda_py_data + cpb_data
           + collect_data_files('tifffile')),
    hiddenimports=(raw_hidden + ort_hidden + cupy_hidden + cuda_py_hidden + cpb_hidden
                   + ['graphlib', 'PIL.ImageCms', 'PySide6.QtMultimedia', 'PySide6.QtNetwork']),
    excludes=[name for name in bundle.EXCLUDES if name != 'cupy'],
    noarchive=False,
)
a.binaries = bundle.trim(a.binaries, 'linux')
a.datas = bundle.trim(a.datas, 'linux')
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='LumenRAW',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, disable_windowed_traceback=False)
# 1.6.0: the same program as a console executable: the command line and the MCP server (stdio).
cli = EXE(pyz, a.scripts, [], exclude_binaries=True, name='lumen-cli',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=True)
coll = COLLECT(exe, cli, a.binaries, a.datas, strip=False, upx=False, name='LumenRAW')
