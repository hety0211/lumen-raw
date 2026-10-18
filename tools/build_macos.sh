#!/bin/bash
# LUMEN RAW macOS (Apple silicon) release build:
#   environment -> runtime assets + Perl ExifTool -> Metal / Core ML check -> tests ->
#   source smoke test -> PyInstaller .app -> ad-hoc signature -> frozen smoke test -> DMG.
# Usage: tools/build_macos.sh [--python python3.12] [--sample photo.ARW] [--skip-tests]
# Output: .publish/v<version>/macos/  (logs/, LumenRAW-<version>-macOS-arm64.dmg, SHA256SUMS-macOS.txt)
set -uo pipefail
# Keep the Mac awake for the whole build (idle sleep would stop tests and packaging).
if [ -z "${LUMEN_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null; then
    LUMEN_CAFFEINATED=1 exec caffeinate -i /bin/bash "$0" "$@"
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
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
PUBLISH="$ROOT/.publish/$TAG/macos"
LOGS="$PUBLISH/logs"
mkdir -p "$LOGS"
BUILD_LOG="$LOGS/build.log"
STATUS="$LOGS/status.txt"
echo "LUMEN RAW $VERSION macOS build $(date '+%Y-%m-%dT%H:%M:%S')" > "$BUILD_LOG"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

say() { echo "$*" | tee -a "$BUILD_LOG"; }
step() { say ""; say "==== $1  [$(date '+%H:%M:%S')] ===="; echo "running: $1" > "$STATUS"; }
fail() { say "FAILED: $1"; echo "FAILED: $1" > "$STATUS"; exit 1; }
run() {  # run <what> <log> <command...>
    local what="$1" log="$2"; shift 2
    "$@" 2>&1 | tee -a "$log"
    local code=${PIPESTATUS[0]}
    [ "$code" -eq 0 ] || fail "$what (exit $code)"
}

[ "$(uname -s)" = "Darwin" ] || fail "this script builds the macOS app and must run on macOS"
[ "$(uname -m)" = "arm64" ] || fail "an Apple silicon (arm64) Mac is required"

step 'Python environment'
VENV="$ROOT/.venv-macos"
if [ ! -x "$VENV/bin/python" ]; then
    command -v "$PYTHON_BASE" >/dev/null || fail "$PYTHON_BASE not found; pass --python /path/to/python3.12"
    run 'create .venv-macos' "$BUILD_LOG" "$PYTHON_BASE" -m venv "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || fail 'Python 3.12 is required'
run 'install locked dependencies' "$BUILD_LOG" "$PY" -m pip install --disable-pip-version-check --no-cache-dir -q -r requirements-macos.txt
run 'probe runtime' "$BUILD_LOG" "$PY" -c "import onnxruntime as o, PySide6.QtMultimedia, rawpy, cv2, Metal, PyInstaller; assert 'CoreMLExecutionProvider' in o.get_available_providers(), o.get_available_providers(); print('onnxruntime', o.__version__, o.get_available_providers())"
"$PY" -m pip freeze > "$LOGS/pip-freeze.txt"
sw_vers | tee -a "$BUILD_LOG"
sysctl -n machdep.cpu.brand_string | tee -a "$BUILD_LOG"

step 'Runtime assets'
run 'runtime assets' "$BUILD_LOG" "$PY" tools/fetch_assets.py
run 'Perl ExifTool' "$BUILD_LOG" "$PY" tools/fetch_exiftool.py

step 'Metal pixel kernels and Core ML models'
run 'Metal / Core ML check' "$LOGS/check-metal.log" "$PY" tools/check_metal.py "$LOGS/check-metal.json" --models

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

step 'App icon and DMG artwork'
run 'artwork' "$BUILD_LOG" "$PY" tools/make_macos_art.py build/macos
tiffutil -cathidpicheck build/macos/dmg-background.png build/macos/dmg-background@2x.png \
    -out build/macos/dmg-background.tiff >> "$BUILD_LOG" 2>&1 || fail 'DMG background'

step 'PyInstaller (LUMEN RAW.app)'
rm -rf "dist/LUMEN RAW.app" dist/LumenRAW build/LumenRAW-macOS
run 'PyInstaller' "$LOGS/pyinstaller.log" "$PY" -m PyInstaller --noconfirm --clean LumenRAW-macOS.spec
APP="$ROOT/dist/LUMEN RAW.app"
[ -x "$APP/Contents/MacOS/LumenRAW" ] || fail 'app executable missing'

step 'Minimum macOS version and ad-hoc signature'
MINIMUM="$("$PY" tools/package_macos.py --minimum "$APP")" || fail 'minimum macOS scan'
say "Bundled binaries require macOS $MINIMUM"
plutil -replace LSMinimumSystemVersion -string "$MINIMUM" "$APP/Contents/Info.plist" || fail 'Info.plist'
xattr -cr "$APP"
run 'codesign' "$BUILD_LOG" codesign --force --deep --sign - --timestamp=none "$APP"
run 'codesign verify' "$BUILD_LOG" codesign --verify --deep --strict --verbose=2 "$APP"

step 'Frozen smoke test'
rm -rf "$PUBLISH/app-smoke"
QT_QPA_PLATFORM=offscreen "$APP/Contents/MacOS/LumenRAW" --smoke-test "$SAMPLE" "$PUBLISH/app-smoke" >> "$BUILD_LOG" 2>&1 \
    || fail "frozen smoke test; see $PUBLISH/app-smoke/report.json"
run 'smoke report' "$BUILD_LOG" "$PY" tools/package_macos.py --check-smoke "$PUBLISH/app-smoke/report.json"
run 'frozen lumen-cli and MCP check' "$LOGS/cli-smoke.log" "$PY" tools/check_cli.py "$APP/Contents/MacOS/lumen-cli" "$SAMPLE" "$PUBLISH/cli-smoke"

step 'DMG'
run 'package DMG' "$LOGS/package.log" "$PY" tools/package_macos.py --dmg "$APP" "$PUBLISH" --sample "$SAMPLE"
echo "SUCCESS $VERSION $(date '+%Y-%m-%dT%H:%M:%S')" > "$STATUS"
say ""
say "SUCCESS: LUMEN RAW $VERSION for macOS is in $PUBLISH"
ls -l "$PUBLISH" | tee -a "$BUILD_LOG"
