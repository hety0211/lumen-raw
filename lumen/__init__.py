import os
# 2.1.0: ONNX Runtime's own telemetry stays off in every process (read when it starts; inherited by
# the AI worker): on macOS its uploader threads aborted the app as it quit (lumen/performance.py).
os.environ.setdefault('ORT_DISABLE_TELEMETRY', '1')
from . import performance
performance.configure()
__version__ = '2.1.0'
# 1.8.0: Windows records the microphone through Qt Multimedia's Media Foundation backend; the
# packages leave out Qt's FFmpeg backend, so source runs use the same one.
if os.name == 'nt':
    os.environ.setdefault('QT_MEDIA_BACKEND', 'windows')
