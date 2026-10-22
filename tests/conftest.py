"""Tests run with the Simplified Chinese source texts, whatever the system language (1.5.1).

1.6.0: tests use their own rating catalog and control-channel name, so they never touch the
user's library and never talk to a LUMEN RAW window the user has open.  2.0.0: likewise the photo
library database and the preview cache."""
import os
import tempfile

os.environ['LUMEN_LANGUAGE'] = os.environ.get('LUMEN_TEST_LANGUAGE', 'zh_CN')
os.environ['LUMEN_CATALOG'] = os.path.join(tempfile.mkdtemp(prefix='lumen-test-catalog-'), 'catalog.jsonl')
os.environ['LUMEN_CONTROL_NAME'] = f'lumen-raw-test-{os.getpid()}'
# 1.7.0: imported presets go to a folder of the test run, never the user's preset library.
os.environ['LUMEN_PRESETS'] = tempfile.mkdtemp(prefix='lumen-test-presets-')
# 2.0.0: the photo library (next to the test catalog) and the preview cache are the test run's own.
os.environ['LUMEN_PREVIEW_CACHE'] = tempfile.mkdtemp(prefix='lumen-test-previews-')
# 2.1.0: settings.ini, natural-language.json, logs and the GPU notes of the test run are its own as
# well (on macOS too: host.data_folder() follows LOCALAPPDATA when it is set); tests that need a
# fresh folder still set their own.
os.environ['LOCALAPPDATA'] = tempfile.mkdtemp(prefix='lumen-test-data-')


import sys  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _release_closed_windows():
    """2.1.0: a closed main window is never freed by itself (the lambdas connected to its child
    widgets keep it, through Qt), with its render, tile and detail caches and its widgets' images:
    the whole suite grew to 13 GB and ran out of memory on a machine with other programs open.
    After each test, closed main windows are deleted."""
    yield
    if 'lumen.app' not in sys.modules:
        return
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        return
    from lumen.app import MainWindow
    closed = [w for w in app.topLevelWidgets() if isinstance(w, MainWindow) and not w.isVisible()]
    for widget in closed:
        widget.deleteLater()
    if closed:
        del closed, widget
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        import gc
        gc.collect()
