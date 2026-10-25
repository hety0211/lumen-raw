"""Compare an installer extracted with /PORTABLE=1 against the frozen build, file by file.

Usage: python tools/verify_installer.py <extracted-folder> [--update]

2.1.1: ``--update`` checks the update installer, which is the build without ``models/``.
"""
import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from lumen import __version__  # noqa: E402


def files(folder):
    return {p.relative_to(folder).as_posix(): p for p in folder.rglob('*') if p.is_file()}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').digest()


def main(extracted, update=False):
    source, target = files(PROJECT / 'dist' / 'LumenRAW'), files(Path(extracted))
    if update:
        source = {name: path for name, path in source.items() if not name.startswith('models/')}
    # Inno Setup adds nothing in portable mode; any difference is a packaging error.
    if source.keys() != target.keys():
        raise SystemExit(json.dumps({'missing': sorted(source.keys() - target.keys())[:20],
                                     'extra': sorted(target.keys() - source.keys())[:20]}, ensure_ascii=False))
    mismatched = [name for name in sorted(source) if digest(source[name]) != digest(target[name])]
    if mismatched:
        raise SystemExit('content mismatch: ' + ', '.join(mismatched[:20]))
    report = {'version': __version__, 'package': 'update' if update else 'setup', 'files': len(source),
              'bytes': sum(p.stat().st_size for p in source.values()),
              'installer_files_match_portable': True}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main(sys.argv[1], '--update' in sys.argv[2:])
