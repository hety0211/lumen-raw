"""Windows ML execution-provider catalog (1.3.1).

On Windows 11 24H2+ (build 26100) Windows ML can download vendor execution
providers on demand.  LUMEN RAW uses it for NVIDIA GeForce RTX 30-series and
newer: the ``NvTensorRtRtxExecutionProvider`` (TensorRT for RTX) is fetched
through the catalog, registered with ONNX Runtime as a plugin library and then
used for the AI models.  Everything here is best effort: any failure leaves the
existing DirectML / CPU path in place.

``LUMEN_WINML=0`` disables the catalog entirely.
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
from .i18n import tr

log = logging.getLogger(__name__)

TENSORRT_RTX = 'NvTensorRtRtxExecutionProvider'
NVIDIA = 0x10DE
MIN_BUILD = 26100

_lock = threading.Lock()
_devices = {}
_errors = {}


def windows_build():
    if os.name != 'nt':
        return 0
    try:
        return sys.getwindowsversion().build
    except AttributeError:
        return 0


def supported():
    return windows_build() >= MIN_BUILD and os.environ.get('LUMEN_WINML', '1') != '0'


def preload_system_runtime():
    """Load System32's MSVCP140.dll before bundled copies (microsoft/WindowsML#22).

    Catalog execution providers are built against the current VC++ runtime; an
    older MSVCP140.dll loaded first by another package makes them crash.
    """
    if os.name != 'nt':
        return False
    try:
        ctypes.WinDLL(os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32', 'MSVCP140.dll'))
        return True
    except OSError:
        return False


def _matches(name, wanted):
    return name.lower() == wanted.lower()


def ensure(ep_name=TENSORRT_RTX, vendor=NVIDIA, allow_download=True, status=None):
    """Download (if needed) and register a catalog EP; returns its ONNX Runtime devices.

    Results are cached per process; an empty list means "not available".
    """
    with _lock:
        if ep_name in _devices:
            return _devices[ep_name]
        devices = []
        try:
            if not supported():
                raise RuntimeError(f'Windows ML execution providers need Windows 11 24H2 (build {MIN_BUILD}+); '
                                   f'this is build {windows_build()}')
            import onnxruntime as ort
            if not hasattr(ort, 'register_execution_provider_library'):
                raise RuntimeError(f'onnxruntime {ort.__version__} cannot load plugin execution providers')
            from windowsml import EpCatalog
            with EpCatalog() as catalog:
                providers = catalog.find_all_providers()
                log.info('Windows ML catalog: %s', ', '.join(f'{p.name} {p.version} ({p.ready_state.name})' for p in providers))
                match = next((p for p in providers if _matches(p.name, ep_name)), None)
                if match is None:
                    raise RuntimeError(f'{ep_name} is not offered for this PC')
                if match.ready_state.name.lower() != 'ready':
                    if not allow_download:
                        raise RuntimeError(f'{ep_name} is not installed')
                    if status:
                        status(tr('首次使用：正在通过 Windows 下载 NVIDIA TensorRT for RTX 加速组件…'))
                    log.info('Downloading %s %s through Windows ML', match.name, match.version)
                    match.ensure_ready()
                ort.register_execution_provider_library(match.name, match.library_path)
                devices = [d for d in ort.get_ep_devices()
                           if _matches(d.ep_name, ep_name) and getattr(d.device, 'vendor_id', vendor) == vendor]
                log.info('%s %s registered from %s: %d device(s)', match.name, match.version,
                         match.library_path, len(devices))
        except Exception as exc:
            _errors[ep_name] = f'{type(exc).__name__}: {exc}'
            log.info('Windows ML %s unavailable: %s', ep_name, exc, exc_info=log.isEnabledFor(logging.DEBUG))
        _devices[ep_name] = devices
        return devices


def last_error(ep_name=TENSORRT_RTX):
    return _errors.get(ep_name, '')
