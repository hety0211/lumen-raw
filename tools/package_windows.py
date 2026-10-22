"""Package the verified Windows onedir build as a portable ZIP."""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from lumen import __version__  # noqa: E402

SOURCE = PROJECT / 'dist' / 'LumenRAW'
PACKAGES = PROJECT / '.publish' / ('v' + __version__.replace('.', '')) / 'packages'
OUTPUT = PACKAGES / f'LumenRAW-{__version__}-Windows.zip'


def write_checksums():
    """SHA256SUMS.txt for the portable ZIP and installer, as published on Releases."""
    lines = []
    for path in sorted(PACKAGES.glob(f'LumenRAW-{__version__}-*')):
        if path.suffix.lower() in ('.zip', '.exe'):
            with path.open('rb') as stream:
                lines.append(f'{hashlib.file_digest(stream, "sha256").hexdigest()}  {path.name}')
    (PACKAGES / 'SHA256SUMS.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


def main():
    if not (SOURCE / 'LumenRAW.exe').is_file():
        raise SystemExit('Missing frozen editor executable.')
    files = sorted(path for path in SOURCE.rglob('*') if path.is_file())
    names = {path.name.lower() for path in files}
    required = {'lumen.ico', 'notosanssc.ttf', 'exiftool.exe',
                'realesrgan-x4plus.onnx', 'realesr-general-x4v3.onnx',
                'nafnet-sidd.onnx', 'drunet-color.onnx', 'ffdnet-color.onnx',
                'skyseg.onnx', 'person-deeplab.onnx', 'u2netp.onnx', 'midas-small.onnx',
                # 2.1.0: AI removal, picked objects and landscape masks.
                'inpaint-lama.onnx', 'object-encoder.onnx', 'object-decoder.onnx', 'landscape-seg.onnx'}
    missing = required - names
    if missing:
        raise SystemExit('Frozen app is missing runtime resources: ' + ', '.join(sorted(missing)))
    # 1.8.0: the speech model is downloaded on first use, the AI models carry float16 weights.
    if 'model.int8.onnx' in names:
        raise SystemExit('The speech model must not be packaged (it is downloaded on first use).')
    sky = next(path for path in files if path.name.lower() == 'skyseg.onnx')
    if sky.stat().st_size > 120 * 2**20:
        raise SystemExit('The packaged models are not the float16-weight copies (tools/compact_models.py).')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise SystemExit(f'Preserving existing package: {OUTPUT}')
    temporary = OUTPUT.with_suffix('.zip.tmp')
    if temporary.exists():
        raise SystemExit(f'Remove the interrupted temporary archive after inspecting it: {temporary}')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=4, allowZip64=True) as archive:
            for path in files:
                archive.write(path, 'LumenRAW-Windows/' + path.relative_to(SOURCE).as_posix())
        with zipfile.ZipFile(temporary) as archive:
            corrupt = archive.testzip()
            if corrupt:
                raise RuntimeError(f'Portable ZIP failed CRC verification: {corrupt}')
        temporary.replace(OUTPUT)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    with OUTPUT.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = {'version': __version__, 'archive': OUTPUT.name, 'sha256': digest,
              'bytes': OUTPUT.stat().st_size, 'files': len(files),
              'zip_crc_verified': True, 'distribution': 'Windows DirectML'}
    OUTPUT.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(write_checksums() if '--checksums' in sys.argv else main())
