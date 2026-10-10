"""Rotating diagnostic log for the desktop application (1.3.0).

Windowed builds have no console, so GPU fallbacks, job failures and state
transitions are written to %LOCALAPPDATA%\\LUMEN RAW\\logs\\lumen.log
(macOS: ~/Library/Logs/LUMEN RAW/lumen.log).
Set LUMEN_LOG_LEVEL=DEBUG to include scheduler transitions.
"""
import faulthandler
import logging
import logging.handlers
import os
import sys

_crash_stream = None


def folder():
    from .host import log_folder
    return log_folder()


def configure(level=None, filename='lumen.log'):
    """Rotating log plus faulthandler output; the AI worker process uses its own files."""
    global _crash_stream
    root = logging.getLogger()
    if any(getattr(h, '_lumen', False) for h in root.handlers):
        return
    level = level or os.environ.get('LUMEN_LOG_LEVEL', 'INFO').upper()
    try:
        target = folder()
        target.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(target / filename, maxBytes=2 * 2**20,
                                                       backupCount=3, encoding='utf-8')
        # Native crashes (e.g. a GPU driver fault) are appended by faulthandler.
        crash = 'crash.log' if filename == 'lumen.log' else 'crash-' + filename.replace('lumen-', '')
        _crash_stream = open(target / crash, 'a', encoding='utf-8', buffering=1)
        faulthandler.enable(_crash_stream)
    except OSError:
        handler = logging.StreamHandler(sys.stderr)
    handler._lumen = True
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s'))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level, logging.INFO))
    from . import __version__
    logging.getLogger('lumen').info('LUMEN RAW %s %s · Python %s · pid %d', __version__,
                                    {'lumen.log': 'starting', 'lumen-cli.log': 'command line'}.get(filename, 'AI worker'),
                                    sys.version.split()[0], os.getpid())


def describe_system():
    """One log line per GPU with its driver, plus the ONNX Runtime build (1.3.1)."""
    log = logging.getLogger('lumen')
    if sys.platform == 'darwin':
        return _describe_macos(log)
    try:
        from . import compute, winml
        import onnxruntime as ort
        log.info('Windows build %s · onnxruntime %s · providers %s', winml.windows_build(), ort.__version__,
                 ', '.join(ort.get_available_providers()))
        for index, name, dedicated, vendor, *driver in compute.dxgi_adapters():
            log.info('GPU %d: %s · vendor 0x%04X · %d MiB · driver %s', index, name, vendor,
                     dedicated // 2**20, driver[0] if driver else '?')
    except Exception:
        log.warning('system description failed', exc_info=True)


def _describe_macos(log):
    """macOS version, chip and Metal device, plus the ONNX Runtime build (1.3.1)."""
    try:
        import platform
        import onnxruntime as ort
        from . import metal
        log.info('macOS %s %s · onnxruntime %s · providers %s', metal.macos_version(), platform.machine(),
                 ort.__version__, ', '.join(ort.get_available_providers()))
        info = metal.device_info()
        if info:
            log.info('Metal device: %s · recommended working set %d MiB', info[1], info[2] // 2**20)
        else:
            log.info('Metal device: none')
    except Exception:
        log.warning('system description failed', exc_info=True)
