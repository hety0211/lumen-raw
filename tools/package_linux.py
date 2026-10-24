"""Package and verify the Linux PyInstaller build (2.1.0).

  python tools/package_linux.py dist/LumenRAW --output .publish/v210/linux [--sample photo.DNG]
      add a .desktop entry and a PNG icon, write a portable .tar.gz, and optionally run the
      frozen smoke test against ``--sample``.

The application folder stays usable in place (``dist/LumenRAW/LumenRAW``); the archive is the
Linux counterpart of the portable Windows zip and the macOS DMG.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from lumen import __version__  # noqa: E402

DESKTOP = """[Desktop Entry]
Type=Application
Name=LUMEN RAW
GenericName=RAW Photo Editor
Comment=Non-destructive RAW development for landscape and travel photography
Exec=LumenRAW %F
Icon=lumen
Terminal=false
Categories=Graphics;Photography;
MimeType=image/x-dcraw;image/x-adobe-dng;image/tiff;image/jpeg;image/png;
StartupWMClass=LumenRAW
"""


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_launcher(folder):
    icon = folder / 'lumen.png'
    try:
        from PIL import Image
        with Image.open(PROJECT / 'assets' / 'lumen.ico') as image:
            image.save(icon)
    except Exception:
        pass
    (folder / 'LumenRAW.desktop').write_text(DESKTOP, encoding='utf-8')
    # A one-command installation into the current user's menu; no root is needed.
    install = folder / 'install-desktop.sh'
    install.write_text("""#!/bin/sh
# Install the menu entry for the current user.  Run this script from the unpacked folder.
set -e
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BIN="$HOME/.local/bin"
APPS="$HOME/.local/share/applications"
ICONS="$HOME/.local/share/icons/hicolor/256x256/apps"
mkdir -p "$BIN" "$APPS" "$ICONS"
ln -sf "$ROOT/LumenRAW" "$BIN/LumenRAW"
ln -sf "$ROOT/lumen-cli" "$BIN/lumen-cli"
cp "$ROOT/lumen.png" "$ICONS/lumen.png"
cp "$ROOT/LumenRAW.desktop" "$APPS/LumenRAW.desktop"
echo "Installed LUMEN RAW. Start it from the application menu or run $BIN/LumenRAW."
""", encoding='utf-8')
    install.chmod(0o755)


def smoke(folder, sample, destination):
    smoke_dir = Path(destination)
    smoke_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    code = subprocess.run([str(folder / 'LumenRAW'), '--smoke-test', str(sample), str(smoke_dir)],
                          env=environment).returncode
    if code:
        raise SystemExit(f'frozen smoke test failed ({code}); see {smoke_dir / "report.json"}')
    report = json.loads((smoke_dir / 'report.json').read_text(encoding='utf-8'))
    if not report.get('ok'):
        raise SystemExit('frozen smoke test failed:\n' + report.get('error', '(no error text)'))
    backend = report.get('backend', '')
    print(f'backend: {backend}\ncompute: {report.get("compute")}')
    if 'CUDA' not in backend and os.environ.get('LUMEN_ALLOW_CPU_PIXELS') != '1':
        raise SystemExit(f'pixel development did not run on CUDA: {backend}')
    return smoke_dir


def package(folder, output, sample=None):
    folder, output = Path(folder), Path(output)
    if not (folder / 'LumenRAW').is_file():
        raise SystemExit(f'{folder} does not contain the LumenRAW executable')
    output.mkdir(parents=True, exist_ok=True)
    write_launcher(folder)
    archive = output / f'LumenRAW-{__version__}-Linux-x86_64.tar.gz'
    archive.unlink(missing_ok=True)
    subprocess.run(['tar', '-C', str(folder.parent), '-czf', str(archive), folder.name], check=True)
    report = dict(version=__version__, archive=archive.name, bytes=archive.stat().st_size,
                  sha256=sha256(archive), desktop_entry='LumenRAW.desktop')
    if sample:
        with tempfile.TemporaryDirectory(prefix='lumen-linux-smoke-') as temporary:
            smoke(folder, sample, temporary)
            report['frozen_smoke_test'] = 'passed'
    (output / 'SHA256SUMS-Linux.txt').write_text(f'{report["sha256"]}  {archive.name}\n', encoding='utf-8')
    archive.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


def main(arguments):
    if not arguments:
        raise SystemExit(__doc__)
    sample = arguments[arguments.index('--sample') + 1] if '--sample' in arguments else None
    output = arguments[arguments.index('--output') + 1] if '--output' in arguments else '.publish/linux'
    package(arguments[0], output, sample)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
