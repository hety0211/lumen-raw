#!/bin/bash
# Run LUMEN RAW from source on an Apple silicon Mac (Python 3.12). First run creates .venv-macos,
# installs requirements-macos.txt and restores the runtime assets and the Perl ExifTool.
cd "$(dirname "$0")" || exit 1
PYTHON_BASE="${LUMEN_PYTHON:-python3.12}"
if [ ! -x .venv-macos/bin/python ]; then
    command -v "$PYTHON_BASE" >/dev/null || { echo "Install Python 3.12 (python.org, Homebrew or uv) and run again."; exit 1; }
    "$PYTHON_BASE" -m venv .venv-macos || exit 1
fi
PY=.venv-macos/bin/python
# 2.1.0: install again whenever requirements-macos.txt changed since the last install (e.g. PyAV for
# time-lapse videos, new in 2.0), not only when an older module is missing.
if ! cmp -s requirements-macos.txt .venv-macos/lumen-requirements.txt || \
   ! "$PY" -c "import numpy, cv2, PIL, rawpy, PySide6.QtMultimedia, tifffile, onnxruntime, Metal, av" >/dev/null 2>&1; then
    "$PY" -m pip install --disable-pip-version-check --no-cache-dir -r requirements-macos.txt || exit 1
    cp requirements-macos.txt .venv-macos/lumen-requirements.txt
fi
"$PY" tools/fetch_assets.py --verify-only >/dev/null 2>&1 || "$PY" tools/fetch_assets.py || exit 1
"$PY" tools/fetch_exiftool.py || exit 1
exec "$PY" main.py "$@"
