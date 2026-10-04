"""Host platform: per-user folders and the few interface words that differ on macOS (1.3.1).

Windows keeps its 1.3 locations (``%LOCALAPPDATA%\\LUMEN RAW``).  macOS uses the
standard ``~/Library`` folders.  Qt maps ``Ctrl`` shortcuts to the Command key
and ``Alt`` to Option on macOS, so only the displayed names change.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from .i18n import tr

MACOS = sys.platform == 'darwin'
WINDOWS = os.name == 'nt'
#: CFBundleIdentifier of the macOS app (LumenRAW-macOS.spec); names its ~/Library/Caches folder.
BUNDLE_ID = 'io.github.hety0211.lumenraw'

#: Key names shown in hints; the shortcuts themselves stay ``Ctrl+…`` in code.
COMMAND = '⌘' if MACOS else 'Ctrl'
OPTION = 'Option' if MACOS else 'Alt'
#: Canvas navigation hint: trackpads pan with two fingers and zoom with a pinch.
NAVIGATION = tr('双指滑动平移，捏合或滚轮缩放') if MACOS else tr('中键平移，滚轮放大')
#: Acceleration choice in the right panel and the enhancement dialog note.
ACCELERATION = tr('自动加速（Metal / Core ML）') if MACOS else tr('自动加速（DirectML / TensorRT for RTX）')
GPU_NOTE = (tr('自动模式在 Apple 芯片的 GPU 上运行（Metal / Core ML），失败则使用 CPU。') if MACOS
            else tr('自动模式会尝试 AMD DirectML 或 NVIDIA CUDA，失败则使用 CPU。'))


def keys(text):
    """``'Ctrl+O'`` as the user presses it: ``'⌘O'`` on macOS, unchanged elsewhere."""
    return text.replace('Ctrl+', '⌘').replace('Shift+', '⇧').replace('Alt+', '⌥') if MACOS else text


def data_folder():
    """Per-user application data (``gpu-compat.json``)."""
    if MACOS:
        return Path.home() / 'Library' / 'Application Support' / 'LUMEN RAW'
    base = os.environ.get('LOCALAPPDATA') or str(Path.home() / '.local' / 'state')
    return Path(base) / 'LUMEN RAW'


def log_folder():
    if MACOS:
        return Path.home() / 'Library' / 'Logs' / 'LUMEN RAW'
    return data_folder() / 'logs'


def cache_folder():
    """Rebuildable caches such as compiled Core ML models."""
    if MACOS:
        return Path.home() / 'Library' / 'Caches' / 'LUMEN RAW'
    return data_folder() / 'cache'
