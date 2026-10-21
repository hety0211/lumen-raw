import os
from . import performance
performance.configure()
__version__ = '2.0.1'
# 1.8.0: Windows records the microphone through Qt Multimedia's Media Foundation backend; the
# packages leave out Qt's FFmpeg backend, so source runs use the same one.
if os.name == 'nt':
    os.environ.setdefault('QT_MEDIA_BACKEND', 'windows')
