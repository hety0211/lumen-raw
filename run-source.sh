#!/bin/bash
# Run LUMEN RAW from source on Linux (Python 3.12).  The first run creates .venv-linux,
# installs the runtime requirements (plus the CUDA build on an NVIDIA machine) and restores
# the runtime assets and the Perl ExifTool.
cd "$(dirname "$0")" || exit 1
PYTHON_BASE="${LUMEN_PYTHON:-python3.12}"
if [ ! -x .venv-linux/bin/python ]; then
    if command -v uv >/dev/null 2>&1; then
        uv venv -p "$PYTHON_BASE" .venv-linux || exit 1
    else
        command -v "$PYTHON_BASE" >/dev/null ||
            { echo "Install Python 3.12 (python3.12, python.org, or uv) and run again."; exit 1; }
        "$PYTHON_BASE" -m venv .venv-linux || exit 1
    fi
fi
PY=.venv-linux/bin/python

# The CUDA/cuDNN wheels live under site-packages/nvidia.  Put unversioned symlinks and those
# folders on LD_LIBRARY_PATH before any CUDA import, exactly as main.py does when it re-execs.
cuda_library_path() {
    "$PY" -c 'from lumen.cuda_env import prepare; print(prepare())' 2>/dev/null
}
CUDA_LIBS=$(cuda_library_path)
if [ -n "$CUDA_LIBS" ]; then
    export LD_LIBRARY_PATH="$CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if command -v uv >/dev/null 2>&1; then
    pip_install() { uv pip install --python "$PY" "$@"; }
    pip_uninstall() { uv pip uninstall --python "$PY" "$@"; }
else
    if ! "$PY" -m pip --version >/dev/null 2>&1; then
        "$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
    fi
    pip_install() { "$PY" -m pip install --disable-pip-version-check --no-cache-dir "$@"; }
    pip_uninstall() { "$PY" -m pip uninstall -y "$@"; }
fi

# Reinstall whenever a requirements file changed, an import is missing, or an NVIDIA GPU is
# present but the environment still has the CPU ONNX Runtime wheel.
STAMP=.venv-linux/lumen-requirements.txt
cat requirements.txt requirements-linux.txt requirements-linux-gpu.txt > "$STAMP.new"
need_gpu=false
if command -v nvidia-smi >/dev/null 2>&1; then
    "$PY" -c "import cupy, onnxruntime; assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()" >/dev/null 2>&1 ||
        need_gpu=true
fi
if ! cmp -s "$STAMP.new" "$STAMP" || [ "$need_gpu" = true ] ||
   ! "$PY" -c "import numpy, cv2, PIL, rawpy, PySide6.QtMultimedia, tifffile, onnxruntime, av" >/dev/null 2>&1; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        # Never install the CPU wheel next to the CUDA wheel: both distributions own the same
        # ``onnxruntime`` modules, and removing the CPU one deletes files the GPU one needs.
        grep -v '^onnxruntime' requirements.txt > "$STAMP.base"
        pip_install -r "$STAMP.base" || exit 1
        rm -f "$STAMP.base"
        pip_uninstall onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
        pip_install -r requirements-linux-gpu.txt || exit 1
    else
        pip_install -r requirements.txt || exit 1
    fi
    mv "$STAMP.new" "$STAMP"
fi
rm -f "$STAMP.new"

# A first install has just added the CUDA wheels; refresh the loader path for this run.
CUDA_LIBS=$(cuda_library_path)
if [ -n "$CUDA_LIBS" ]; then
    export LD_LIBRARY_PATH="$CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

"$PY" tools/fetch_assets.py --verify-only >/dev/null 2>&1 || "$PY" tools/fetch_assets.py || exit 1
"$PY" tools/fetch_exiftool.py || exit 1
exec "$PY" main.py "$@"
