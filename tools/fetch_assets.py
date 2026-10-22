"""Download and verify the versioned runtime assets; uses only Python's stdlib."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def asset_path(name):
    value = PurePosixPath(name)
    if (value.is_absolute() or not value.parts or value.parts[0] != 'assets'
            or any(part in ('', '.', '..') or ':' in part or '\\' in part for part in value.parts)
            or len(value.parts) < 2):
        raise ValueError(f'Invalid asset path: {name}')
    return Path(*value.parts)


def verify(root, manifest):
    missing = []
    for record in manifest['files']:
        path = root / asset_path(record['path'])
        if not path.is_file() or path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            missing.append(record['path'])
    return missing


def install_archive(archive, root, manifest):
    # Validate the entire download and every member before changing any asset.
    if sha256(archive) != manifest['bundle']['sha256']:
        raise ValueError('Asset archive SHA-256 mismatch; nothing was installed.')
    expected = {record['path']: record for record in manifest['files']}
    if len(expected) != len(manifest['files']):
        raise ValueError('Duplicate asset paths in manifest.')
    for name in expected:
        relative = asset_path(name)
        if not (root / relative).resolve().is_relative_to(root.resolve()):
            raise ValueError(f'Asset destination escapes project: {name}')
    cache = root / '.asset-cache'
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='unpack-', dir=cache) as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive) as bundle:
            members = [member for member in bundle.infolist() if not member.is_dir()]
            if len(members) != len(expected) or {m.filename for m in members} != set(expected):
                raise ValueError('Unexpected archive members; nothing was installed.')
            for member in members:
                record = expected[member.filename]
                if member.file_size != record['bytes'] or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f'Invalid archive member: {member.filename}')
                destination = staging / asset_path(member.filename)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, destination.open('wb') as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
                if sha256(destination) != record['sha256']:
                    raise ValueError(f'Asset SHA-256 mismatch: {member.filename}')
        for name in expected:
            relative = asset_path(name)
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / relative, target)


#: 1.5.0: assets fetched file by file from a pinned upstream revision (not part of the bundle).
#: 2.0.0: the culling hints' face models (YuNet from OpenCV Zoo on GitHub, MediaPipe Face Mesh).
#: 2.1.0: AI removal, picked objects and landscape masks.
DIRECT_MANIFESTS = ('assets/speech-assets.json', 'assets/culling-assets.json', 'assets/editing-assets.json')
DIRECT_HOSTS = ('https://huggingface.co/', 'https://github.com/')


def install_direct(root, manifest):
    """Download each missing file of a per-file manifest, verifying size and SHA-256 first."""
    cache = root / '.asset-cache'
    cache.mkdir(exist_ok=True)
    for record in manifest['files']:
        target = root / asset_path(record['path'])
        if target.is_file() and target.stat().st_size == record['bytes'] and sha256(target) == record['sha256']:
            continue
        if not record['url'].startswith(DIRECT_HOSTS):
            raise ValueError(f'Unexpected download host: {record["url"]}')
        with tempfile.TemporaryDirectory(prefix='download-', dir=cache) as temporary:
            staging = Path(temporary) / 'file'
            print(f'Downloading {record["bytes"] / 1024**2:.1f} MiB {record["path"]}', flush=True)
            request = urllib.request.Request(record['url'], headers={'User-Agent': 'LUMEN-RAW-asset-bootstrap'})
            with urllib.request.urlopen(request, timeout=120) as response, staging.open('wb') as stream:
                shutil.copyfileobj(response, stream, 1024 * 1024)
            if staging.stat().st_size != record['bytes'] or sha256(staging) != record['sha256']:
                raise ValueError(f'Asset SHA-256 mismatch: {record["path"]}; nothing was installed.')
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, target)


def direct_manifests():
    return [json.loads((ROOT / name).read_text(encoding='utf8')) for name in DIRECT_MANIFESTS if (ROOT / name).is_file()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use a previously downloaded runtime asset ZIP.')
    parser.add_argument('--verify-only', action='store_true', help='Verify installed files without downloading.')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'assets/runtime-assets.json').read_text(encoding='utf8'))
    direct = direct_manifests()
    missing = verify(ROOT, manifest)
    direct_missing = [path for extra in direct for path in verify(ROOT, extra)]
    count = len(manifest['files']) + sum(len(extra['files']) for extra in direct)
    if not missing and not direct_missing:
        print(f'All {count} runtime assets verified.')
        return 0
    if args.verify_only:
        print('Missing or modified assets:\n' + '\n'.join(missing + direct_missing))
        return 1
    for extra in direct:
        install_direct(ROOT, extra)
    if missing and args.archive:
        install_archive(args.archive, ROOT, manifest)
    elif missing:
        url = manifest['bundle']['url']
        if not url.startswith('https://github.com/'):
            raise ValueError('Expected an HTTPS GitHub release URL.')
        cache = ROOT / '.asset-cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='download-', dir=cache) as temporary:
            target = Path(temporary) / 'assets.zip'
            print(f'Downloading {manifest["bundle"]["bytes"] / 1024**2:.0f} MiB from {url}', flush=True)
            request = urllib.request.Request(url, headers={'User-Agent': 'LUMEN-RAW-asset-bootstrap'})
            with urllib.request.urlopen(request, timeout=120) as response, target.open('wb') as stream:
                shutil.copyfileobj(response, stream, 1024 * 1024)
            install_archive(target, ROOT, manifest)
    missing = verify(ROOT, manifest) + [path for extra in direct for path in verify(ROOT, extra)]
    if missing:
        raise ValueError('Installed assets failed verification: ' + ', '.join(missing))
    print(f'Installed and verified {count} runtime assets.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f'Asset setup failed: {error}')
        raise SystemExit(1)
