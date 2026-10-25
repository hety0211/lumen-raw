"""Load the CUDA libraries that ONNX Runtime's Linux wheel needs (Linux only).

``onnxruntime-gpu[cuda,cudnn]`` installs CUDA 13 and cuDNN 9 into
``site-packages/nvidia/.../lib``.  The loader is not told about those folders and cuDNN only
ships versioned files (``libcudnn.so.9``), while ONNX Runtime calls ``dlopen("libcudnn.so")``.
This module prepares a small folder of unversioned symlinks and the launcher exports both
folders through ``LD_LIBRARY_PATH`` before the process starts; the dynamic loader only reads
that variable at startup, so ``main.py`` re-executes itself once when needed.

Nothing here imports Qt or ONNX Runtime: it must stay usable before the application starts.
"""
from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

MARKER = 'LUMEN_CUDA_LIBS_READY'


def _roots():
    """Folders that may contain the pip ``nvidia`` packages (source run and frozen app)."""
    roots = []
    with_version = sysconfig.get_paths().get('purelib')
    if with_version:
        roots.append(Path(with_version))
    if getattr(sys, 'frozen', False):
        roots.append(Path(getattr(sys, '_MEIPASS', '')))
    # A virtual environment can be started through a symlinked interpreter; ask it directly.
    for entry in sys.path:
        if entry and (Path(entry) / 'nvidia').is_dir():
            roots.append(Path(entry))
    return roots


def library_folders():
    """Every ``nvidia/*/lib`` folder that exists, newest package layout first."""
    seen, folders = set(), []
    for root in _roots():
        base = root / 'nvidia'
        if not base.is_dir():
            continue
        candidates = [base / 'cu13' / 'lib', base / 'cudnn' / 'lib']
        candidates += sorted(path / 'lib' for path in base.iterdir() if path.is_dir())
        for folder in candidates:
            if folder.is_dir():
                resolved = str(folder.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    folders.append(folder)
    return folders


def cache_folder():
    base = os.environ.get('XDG_CACHE_HOME') or str(Path.home() / '.cache')
    return Path(base) / 'lumen-raw' / 'cuda-libs'


def prepare(create=True):
    """Return the ``LD_LIBRARY_PATH`` value for this run, or '' when no CUDA wheel is installed.

    Unversioned symlinks are created in a cache folder; the original folders stay on the path
    because cuDNN's own libraries refer to each other by their versioned names.
    """
    if not sys.platform.startswith('linux'):
        return ''
    folders = library_folders()
    if not folders:
        return ''
    target = cache_folder()
    if create:
        try:
            target.mkdir(parents=True, exist_ok=True)
            for folder in folders:
                for library in folder.glob('*.so.*'):
                    base = library.name.split('.so.', 1)[0]
                    link = target / f'{base}.so'
                    desired = library.resolve()
                    # The cache is shared by source runs and portable builds.  Re-point a link
                    # left by another install instead of keeping a path that may no longer exist.
                    if link.is_symlink() and link.resolve() == desired:
                        continue
                    if link.exists() or link.is_symlink():
                        link.unlink()
                    link.symlink_to(desired)
        except OSError:
            pass
    parts = [str(target)] + [str(folder) for folder in folders]
    current = os.environ.get('LD_LIBRARY_PATH', '')
    if current:
        parts.append(current)
    return os.pathsep.join(parts)


def ensure_process():
    """Re-execute a source or frozen run once with the CUDA folders on ``LD_LIBRARY_PATH``.

    Returns normally when no CUDA wheel is present, when the variable is already set up, or
    when re-execution is not possible.  The frozen build also has the libraries under
    ``_MEIPASS``; re-execution is still used so the loader path includes every cuDNN
    sub-library, not only the lone top-level ``libcudnn.so`` symlink.
    """
    if not sys.platform.startswith('linux') or os.environ.get(MARKER) == '1':
        return
    path = prepare()
    if not path:
        return
    # Do not stop when ``ctypes.CDLL('libcudnn.so')`` happens to succeed.  A PyInstaller
    # runtime puts a lone ``libcudnn.so`` symlink at the top level, and a system cuDNN may
    # exist too; both can load the top library while its versioned sub-libraries (which cuDNN
    # loads by name) remain undiscoverable.  The extra re-exec costs a few milliseconds.
    environment = dict(os.environ)
    environment[MARKER] = '1'
    environment['LD_LIBRARY_PATH'] = path
    arguments = [sys.executable]
    if not getattr(sys, 'frozen', False):
        arguments.append(os.path.abspath(sys.argv[0]))
    arguments += sys.argv[1:]
    try:
        os.execve(sys.executable, arguments, environment)
    except OSError:
        # execv is unavailable in some embedded runtimes; ONNX Runtime will fall back to CPU.
        return
