"""Standard output handles seen by native libraries with their own C runtime (2.1.2).

OpenCV (``cv2.pyd``) and ONNX Runtime's Python module (``onnxruntime_pybind11_state.pyd``) are
linked with a static C runtime on Windows.  When such a module loads, its runtime copies the
process's standard handles (``GetStdHandle``) and writes its warnings to that copy for the rest
of the process's life, whatever Python does with ``sys.stderr`` or file descriptor 2 later.

If the handle is closed afterwards, Windows gives the same value to the next file the process
opens.  That is what damaged the photo library in the 2.1.1 test runs ("database disk image is
malformed"): pytest points fd 2 at a fresh capture file for every test and closes the previous
handle, OpenCV and ONNX Runtime kept the closed one, SQLite's next ``library.db`` connection got
its value, and the next warning (OpenCV's "Targets are not supported by the new graph engine"
for the face detector, ONNX Runtime's "Config with key … already exists") was written into the
database over page 4, the ``folders`` table.

* ``repair`` (every process, at ``import lumen``): a standard handle whose value is not a valid
  handle of this process (a parent that passed values without letting them be inherited) is
  replaced by a handle to ``NUL`` before any native library copies it.
* ``pin`` loads the native modules while standard output and error are a ``NUL`` handle that is
  never closed; the tests do this before pytest's output capture swaps handles around.

The application itself never closes or swaps its standard handles.  On macOS and Linux the
libraries share the C library's descriptors and SQLite never uses descriptors 0–2.
"""
from __future__ import annotations

import os

STD_INPUT, STD_OUTPUT, STD_ERROR = -10, -11, -12
_INVALID = (1 << 64) - 1
_nul = None


def _kernel32():
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
    kernel32.GetStdHandle.restype = ctypes.c_void_p
    kernel32.SetStdHandle.argtypes = [wintypes.DWORD, ctypes.c_void_p]
    kernel32.SetStdHandle.restype = wintypes.BOOL
    kernel32.GetHandleInformation.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetHandleInformation.restype = wintypes.BOOL
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                     wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    kernel32.CreateFileW.restype = ctypes.c_void_p
    return kernel32


def nul_handle():
    """A write handle to ``NUL`` that stays open for the life of the process."""
    global _nul
    if _nul is None:
        kernel32 = _kernel32()
        # GENERIC_READ | GENERIC_WRITE, shared, OPEN_EXISTING
        handle = kernel32.CreateFileW('NUL', 0xC0000000, 0x3, None, 3, 0, None)
        if handle in (None, 0, _INVALID):
            raise OSError('NUL could not be opened')
        _nul = handle
    return _nul


def valid(handle, kernel32=None):
    """Whether ``handle`` is an open handle of this process."""
    import ctypes
    from ctypes import wintypes
    kernel32 = kernel32 or _kernel32()
    flags = wintypes.DWORD()
    return bool(kernel32.GetHandleInformation(handle, ctypes.byref(flags)))


def repair():
    """Point standard handles that are not open in this process at ``NUL``; returns the numbers of
    the ones replaced.  Missing handles (a windowed program) stay missing: a C runtime then has
    no standard output at all, which is safe."""
    if os.name != 'nt':
        return []
    try:
        kernel32 = _kernel32()
        replaced = []
        for which in (STD_INPUT, STD_OUTPUT, STD_ERROR):
            handle = kernel32.GetStdHandle(which)
            if handle in (None, 0, _INVALID) or valid(handle, kernel32):
                continue
            kernel32.SetStdHandle(which, nul_handle())
            replaced.append(which)
        return replaced
    except (OSError, AttributeError):
        return []


def pin(modules=('cv2', 'onnxruntime')):
    """Import ``modules`` while standard output and error are the permanent ``NUL`` handle, so
    the C runtimes linked into them never keep a handle that is closed later.  Must run before
    anything else imports them; modules that are not installed are skipped."""
    if os.name != 'nt':
        return
    import importlib
    kernel32 = _kernel32()
    saved = {which: kernel32.GetStdHandle(which) for which in (STD_OUTPUT, STD_ERROR)}
    nul = nul_handle()
    try:
        for which in saved:
            kernel32.SetStdHandle(which, nul)
        for name in modules:
            try:
                if name == 'onnxruntime':
                    from . import performance
                    performance.onnxruntime()
                else:
                    importlib.import_module(name)
            except ImportError:
                pass
    finally:
        for which, handle in saved.items():
            kernel32.SetStdHandle(which, handle)
