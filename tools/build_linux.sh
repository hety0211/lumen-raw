#!/bin/bash
# LUMEN RAW Linux release build:
#   environment -> runtime assets + Perl ExifTool -> tests -> source smoke test -> PyInstaller
#   folder -> portable .tar.gz (+ optional frozen smoke test).
# Usage: tools/build_linux.sh [--python .venv-linux/bin/python] [--sample photo.DNG] [--skip-tests]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
PYTHON_BASE="python3.12"
SAMPLE=""
SKIP_TESTS=0
while [ $# -gt 0 ]; do
    case "$1" in
        --python) PYTHON_BASE="$2"; shift 2 ;;
        --sample) SAMPLE="$2"; shift 2 ;;
        --skip-tests) SKIP_TESTS=1; shift ;;
        *) echo "unknown option $1"; exit 2 ;;
    esac
done

VERSION="$(sed -n "s/^__version__ = '\([0-9.]*\)'/\1/p" lumen/__init__.py)"
TAG="v${VERSION//./}"
PUBLISH="$ROOT/.publish/$TAG/linux"
LOGS="$PUBLISH/logs"
mkdir -p "$LOGS"
BUILD_LOG="$LOGS/build.log"
echo "LUMEN RAW $VERSION Linux build $(date '+%Y-%m-%dT%H:%M:%S')" > "$BUILD_LOG"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

say() { echo "$*" | tee -a "$BUILD_LOG"; }
step() { say ""; say "==== $1  [$(date '+%H:%M:%S')] ===="; }
fail() { say "FAILED: $1"; exit 1; }
run() {
    local what="$1" log="$2"; shift 2
    "$@" 2>&1 | tee -a "$log"
    local code=${PIPESTATUS[0]}
    [ "$code" -eq 0 ] || fail "$what (exit $code)"
}

[ "$(uname -s)" = "Linux" ] || fail "this script builds the Linux app"
[ "$(uname -m)" = "x86_64" ] || fail "x86_64 Linux is required for the published build"

step 'Python environment'
if [ ! -x "$PYTHON_BASE" ]; then
    [ -x ".venv-linux/bin/python" ] || {
        command -v python3.12 >/dev/null || fail "Python 3.12 not found; pass --python /path/to/python3.12"
        python3.12 -m venv .venv-linux || fail 'create .venv-linux'
    }
    PYTHON_BASE=".venv-linux/bin/python"
fi
PY="$PYTHON_BASE"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || fail 'Python 3.12 is required'
if "$PY" -m pip --version >/dev/null 2>&1; then
    pip_install() { "$PY" -m pip install --disable-pip-version-check --no-cache-dir "$@"; }
    pip_freeze() { "$PY" -m pip freeze; }
elif command -v uv >/dev/null 2>&1; then
    pip_install() { uv pip install --python "$PY" "$@"; }
    pip_freeze() { uv pip freeze --python "$PY"; }
else
    "$PY" -m ensurepip --upgrade >/dev/null 2>&1 || fail 'pip or uv is required to install build tools'
    pip_install() { "$PY" -m pip install --disable-pip-version-check --no-cache-dir "$@"; }
    pip_freeze() { "$PY" -m pip freeze; }
fi
if ! "$PY" -c 'import PyInstaller, pytest' >/dev/null 2>&1; then
    run 'install build tools' "$BUILD_LOG" pip_install -r requirements-linux-dev.txt
fi
run 'probe runtime' "$BUILD_LOG" "$PY" -c "import onnxruntime as o, PySide6.QtMultimedia, rawpy, cv2, PyInstaller; print('onnxruntime', o.__version__, o.get_available_providers())"
pip_freeze > "$LOGS/pip-freeze.txt"
# Put the pip CUDA/cuDNN libraries on the loader path for pytest and both smoke tests; main.py
# also re-execs itself when it is started without this, but tests import ONNX Runtime directly.
CUDA_LIBS=$("$PY" -c 'from lumen.cuda_env import prepare; print(prepare())' 2>/dev/null)
if [ -n "$CUDA_LIBS" ]; then
    export LD_LIBRARY_PATH="$CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
uname -a | tee -a "$BUILD_LOG"

step 'Runtime assets'
run 'runtime assets' "$BUILD_LOG" "$PY" tools/fetch_assets.py
run 'Perl ExifTool' "$BUILD_LOG" "$PY" tools/fetch_exiftool.py

if [ "$SKIP_TESTS" -eq 0 ]; then
    step 'Regression tests'
    rm -rf "$PUBLISH/pytest-temp"
    run 'pytest' "$LOGS/pytest.log" env QT_QPA_PLATFORM=offscreen "$PY" -m pytest tests -q -p no:cacheprovider --basetemp="$PUBLISH/pytest-temp"
fi

step 'Sample photo'
if [ -z "$SAMPLE" ] || [ ! -f "$SAMPLE" ]; then
    SAMPLE="$PUBLISH/synthetic-sample.png"
    run 'synthetic sample' "$BUILD_LOG" "$PY" -c "import sys; sys.path.insert(0, '.'); import numpy as np, cv2; from tools.bench_pipeline import sample; cv2.imwrite('$SAMPLE', (np.clip(sample(None), 0, 1) ** (1 / 2.2) * 65535).astype(np.uint16)[..., ::-1])"
    say 'No camera RAW given; using a synthetic 16-bit PNG.'
fi
say "Sample: $SAMPLE"
rm -rf "$PUBLISH/source-smoke"
run 'source smoke test' "$BUILD_LOG" env QT_QPA_PLATFORM=offscreen "$PY" main.py --smoke-test "$SAMPLE" "$PUBLISH/source-smoke"

step 'PyInstaller (LumenRAW)'
rm -rf dist/LumenRAW build/LumenRAW-linux
run 'PyInstaller' "$LOGS/pyinstaller.log" "$PY" -m PyInstaller --noconfirm --clean LumenRAW-linux.spec
[ -x dist/LumenRAW/LumenRAW ] || fail 'app executable missing'

step 'Portable archive and frozen smoke test'
run 'package Linux app' "$LOGS/package.log" "$PY" tools/package_linux.py dist/LumenRAW --output "$PUBLISH" --sample "$SAMPLE"
say ""
say "SUCCESS: LUMEN RAW $VERSION for Linux is in $PUBLISH"
ls -l "$PUBLISH" | tee -a "$BUILD_LOG"
