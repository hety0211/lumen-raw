"""Package the verified Windows onedir build as a portable ZIP.

2.1.1: the AI models are in ``dist/LumenRAW/models`` under their content names, apart from the
program (``_internal``), and ``_internal/assets/ai-models.json`` lists them; both are checked here.
The ZIP carries the models (a portable copy is complete); the update installer does not."""
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


#: Every model 2.1.0 bundled; a manifest without one of them is a packaging error.
MODELS = {'realesrgan-x4plus.onnx', 'realesr-general-x4v3.onnx', 'nafnet-sidd.onnx', 'drunet-color.onnx',
          'ffdnet-color.onnx', 'skyseg.onnx', 'person-deeplab.onnx', 'u2netp.onnx', 'midas-small.onnx',
          'inpaint-lama.onnx', 'object-encoder.onnx', 'object-decoder.onnx', 'landscape-seg.onnx',
          'face-yunet.onnx', 'face-landmarks.onnx'}


def check_models(program=SOURCE):
    """2.1.1: the models of the manifest are in ``models`` (and nowhere in ``_internal``)."""
    manifest = json.loads((program / '_internal' / 'assets' / 'ai-models.json').read_text(encoding='utf-8'))
    listed = manifest['models']
    if MODELS - set(listed):
        raise SystemExit('The model manifest lacks: ' + ', '.join(sorted(MODELS - set(listed))))
    folder = program / 'models'
    for name, record in listed.items():
        path = folder / record['file']
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise SystemExit(f'models/{record["file"]} ({name}) is missing or has the wrong size')
    extra = sorted(p.name for p in folder.iterdir() if p.name not in {r['file'] for r in listed.values()})
    if extra:
        raise SystemExit('Unlisted files in models/: ' + ', '.join(extra))
    # (ONNX Runtime brings a few tiny test graphs of its own, e.g. datasets/logreg_iris.onnx.)
    files = {record['file'] for record in listed.values()}
    inside = sorted(p.name for p in (program / '_internal').rglob('*.onnx') if p.name in MODELS or p.name in files)
    if inside:
        raise SystemExit('Models inside the program files (they belong in models/): ' + ', '.join(inside[:10]))
    # 1.8.0: the packaged models carry float16 weights (tools/compact_models.py).
    if listed['skyseg.onnx']['bytes'] > 120 * 2**20:
        raise SystemExit('The packaged models are not the float16-weight copies (tools/compact_models.py).')
    return manifest


def main():
    if not (SOURCE / 'LumenRAW.exe').is_file():
        raise SystemExit('Missing frozen editor executable.')
    files = sorted(path for path in SOURCE.rglob('*') if path.is_file())
    names = {path.name.lower() for path in files}
    required = {'lumen.ico', 'notosanssc.ttf', 'exiftool.exe', 'ai-models.json'}
    missing = required - names
    if missing:
        raise SystemExit('Frozen app is missing runtime resources: ' + ', '.join(sorted(missing)))
    check_models()
    # 1.8.0: the speech model is downloaded on first use.
    if 'model.int8.onnx' in names:
        raise SystemExit('The speech model must not be packaged (it is downloaded on first use).')
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
    models = sum(p.stat().st_size for p in (SOURCE / 'models').iterdir())
    report = {'version': __version__, 'archive': OUTPUT.name, 'sha256': digest,
              'bytes': OUTPUT.stat().st_size, 'files': len(files),
              'program_bytes': sum(p.stat().st_size for p in files) - models, 'model_bytes': models,
              'zip_crc_verified': True, 'distribution': 'Windows DirectML'}
    OUTPUT.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(write_checksums() if '--checksums' in sys.argv else main())
