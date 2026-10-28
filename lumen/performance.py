"""Shared compute ceiling. Idle workers are not simultaneous image computations."""
import os

MAX_THREADS=32
THREADS=min(MAX_THREADS,max(1,os.cpu_count() or 1))

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]=str(THREADS)

def configure():
    import cv2
    cv2.setNumThreads(THREADS)

_QUIET=False

def onnxruntime():
    """``onnxruntime`` with its own telemetry turned off (2.1.0).  On macOS ONNX Runtime's
    Microsoft telemetry (1DS) posts events over HTTP from background threads: a reply arriving
    while the process exited locked a destroyed mutex and aborted the app as it quit (2 of 4
    smoke runs).  Only ``ORT_DISABLE_TELEMETRY`` keeps those threads from starting;
    ``disable_telemetry_events`` stops the events.  LUMEN RAW works offline: off everywhere."""
    global _QUIET
    os.environ.setdefault('ORT_DISABLE_TELEMETRY','1')   # its uploader threads (see lumen/__init__.py)
    import onnxruntime as ort
    if not _QUIET:
        if hasattr(ort, 'disable_telemetry_events'):
            ort.disable_telemetry_events()
        # 2.1.2: errors only; its warnings went to a stale copy of stderr (lumen/native_stdio.py).
        if hasattr(ort, 'set_default_logger_severity'):
            ort.set_default_logger_severity(3)
        _QUIET=True
    return ort

def session_options():
    ort=onnxruntime()
    options=ort.SessionOptions()
    options.intra_op_num_threads=THREADS
    options.inter_op_num_threads=1
    options.add_session_config_entry('session.intra_op.allow_spinning','0')
    return options
