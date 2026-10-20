"""Package and verify the macOS app (1.3.1).

  --minimum APP            print the highest macOS version any bundled arm64 binary requires
  --check-smoke REPORT     fail unless a smoke-test report shows Metal development and a real AI preview
  --dmg APP OUT [--sample] build LumenRAW-<version>-macOS-arm64.dmg in OUT, then verify it:
                           checksum, read-only mount, code signature, smoke test from the image
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from lumen import __version__  # noqa: E402

ARM64 = 0x0100000C
LC_VERSION_MIN_MACOSX, LC_BUILD_VERSION = 0x24, 0x32


def _version(encoded):
    return (encoded >> 16, (encoded >> 8) & 0xFF, encoded & 0xFF)


def _thin_minimum(data, offset):
    """Minimum macOS of one 64-bit Mach-O image at ``offset`` (LC_BUILD_VERSION / LC_VERSION_MIN)."""
    magic, cpu, _, _, count, _, _, _ = struct.unpack_from('<8I', data, offset)
    if magic != 0xFEEDFACF or cpu != ARM64:
        return None
    position, found = offset + 32, None
    for _ in range(count):
        command, size = struct.unpack_from('<2I', data, position)
        if command == LC_BUILD_VERSION:
            platform, minos = struct.unpack_from('<2I', data, position + 8)
            if platform == 1:                      # PLATFORM_MACOS
                found = _version(minos)
        elif command == LC_VERSION_MIN_MACOSX:
            found = _version(struct.unpack_from('<I', data, position + 8)[0])
        position += size
    return found


def macho_minimum(path):
    with open(path, 'rb') as stream:
        head = stream.read(4)
        if head not in (b'\xcf\xfa\xed\xfe', b'\xca\xfe\xba\xbe'):
            return None
        data = head + stream.read()
    if data[:4] == b'\xcf\xfa\xed\xfe':
        return _thin_minimum(data, 0)
    count = struct.unpack_from('>I', data, 4)[0]          # universal binary: use the arm64 slice
    for index in range(count):
        cpu, _, offset, _, _ = struct.unpack_from('>5I', data, 8 + index * 20)
        if cpu == ARM64:
            return _thin_minimum(data, offset)
    return None


def minimum(app):
    versions = {}
    for path in Path(app).rglob('*'):
        if path.is_file() and not path.is_symlink():
            found = macho_minimum(path)
            if found:
                versions[path] = found
    highest = max(versions.values()) if versions else (11, 0, 0)
    return f'{highest[0]}.{highest[1]}', versions


def check_smoke(report_path):
    report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    if not report.get('ok'):
        raise SystemExit('Smoke test failed:\n' + report.get('error', '(no error text)'))
    backend, enhance = report.get('backend', ''), report.get('enhance', '')
    print(f'backend: {backend}\nenhance: {enhance}\ncompute: {report.get("compute")}')
    if not backend.startswith('Metal'):
        raise SystemExit(f'Pixel development did not run on Metal: {backend}')
    if 'CPUExecutionProvider' in enhance and os.environ.get('LUMEN_ALLOW_CPU_AI') != '1':
        raise SystemExit(f'AI preview fell back to the CPU: {enhance}')
    # 1.5.0: Qt Multimedia and one natural-language edit (diagnostics.natural_language); since 1.8.0
    # the speech model is downloaded on first use, so the package must not contain it.
    language = report.get('natural_language') or {}
    print(f'natural language: {language}')
    if not language.get('speech_download_mb'):
        raise SystemExit('The natural-language smoke test did not report the speech-model download')


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build_dmg(app, output):
    """dmgbuild lays out the Finder window; plain hdiutil is the fallback."""
    art = PROJECT / 'build' / 'macos'
    volume = f'LUMEN RAW {__version__}'
    try:
        import dmgbuild
        settings = {
            'format': 'UDZO', 'filesystem': 'HFS+', 'compression-level': 9,
            'files': [str(app)], 'symlinks': {'Applications': '/Applications'},
            'icon': str(art / 'LumenRAW.icns'),
            'icon_locations': {app.name: (180, 200), 'Applications': (480, 200)},
            'background': str(art / 'dmg-background.tiff'),
            'window_rect': ((200, 140), (660, 400)), 'default_view': 'icon-view',
            'show_status_bar': False, 'show_tab_view': False, 'show_toolbar': False,
            'show_pathbar': False, 'show_sidebar': False, 'icon_size': 128, 'text_size': 13,
            'hide_extension': [app.name],
        }
        dmgbuild.build_dmg(str(output), volume, settings=settings)
        return 'dmgbuild'
    except Exception as exc:
        print(f'dmgbuild failed ({type(exc).__name__}: {exc}); using hdiutil', flush=True)
    with tempfile.TemporaryDirectory(prefix='lumen-dmg-') as staging:
        subprocess.run(['ditto', str(app), str(Path(staging) / app.name)], check=True)
        os.symlink('/Applications', Path(staging) / 'Applications')
        subprocess.run(['hdiutil', 'create', '-volname', volume, '-srcfolder', staging, '-ov',
                        '-fs', 'HFS+', '-format', 'UDZO', '-imagekey', 'zlib-level=9', str(output)], check=True)
    return 'hdiutil'


def verify_dmg(output, app_name, sample):
    subprocess.run(['hdiutil', 'verify', str(output)], check=True)
    mount = Path(tempfile.mkdtemp(prefix='lumen-dmg-mount-'))
    subprocess.run(['hdiutil', 'attach', str(output), '-readonly', '-nobrowse', '-noautoopen',
                    '-mountpoint', str(mount)], check=True, stdout=subprocess.DEVNULL)
    try:
        app = mount / app_name
        subprocess.run(['codesign', '--verify', '--deep', '--strict', str(app)], check=True)
        link = mount / 'Applications'
        if not link.is_symlink() or os.readlink(link) != '/Applications':
            raise SystemExit('DMG is missing the Applications link')
        result = {'mounted_signature': 'valid'}
        if sample:
            smoke = Path(tempfile.mkdtemp(prefix='lumen-dmg-smoke-'))
            environment = dict(os.environ, QT_QPA_PLATFORM='offscreen')
            code = subprocess.run([str(app / 'Contents' / 'MacOS' / 'LumenRAW'), '--smoke-test', str(sample), str(smoke)],
                                  env=environment).returncode
            if code:
                raise SystemExit(f'smoke test from the mounted DMG failed ({code}); see {smoke}/report.json')
            check_smoke(smoke / 'report.json')
            if list(app.rglob('model.int8.onnx')):
                raise SystemExit('The speech model must not be packaged (it is downloaded on first use)')
            result['mounted_smoke_test'] = 'passed'
            shutil.rmtree(smoke, ignore_errors=True)
        return result
    finally:
        subprocess.run(['hdiutil', 'detach', str(mount)], check=False, stdout=subprocess.DEVNULL)
        try:
            mount.rmdir()
        except OSError:
            pass


def package(app, folder, sample=None):
    app, folder = Path(app), Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f'LumenRAW-{__version__}-macOS-arm64.dmg'
    output.unlink(missing_ok=True)
    tool = build_dmg(app, output)
    report = dict(version=__version__, dmg=output.name, built_with=tool, bytes=output.stat().st_size)
    report.update(verify_dmg(output, app.name, sample))
    report['sha256'] = sha256(output)
    report['minimum_macos'] = minimum(app)[0]
    (folder / 'SHA256SUMS-macOS.txt').write_text(f'{report["sha256"]}  {output.name}\n', encoding='utf-8')
    output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def main(arguments):
    if arguments[:1] == ['--minimum']:
        print(minimum(arguments[1])[0])
    elif arguments[:1] == ['--check-smoke']:
        check_smoke(arguments[1])
    elif arguments[:1] == ['--dmg']:
        sample = arguments[arguments.index('--sample') + 1] if '--sample' in arguments else None
        package(arguments[1], arguments[2], sample)
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
