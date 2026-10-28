"""2.1.2: the library database damaged by native libraries' stderr (root cause and fix), input colour
management (ICC profiles, camera Adobe RGB, Nikon "High Efficiency" NEF), update mirrors with a
signed feed, and the photo's GPS position: offline map, XMP sidecar, exports and the border pin."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import hashlib
import io
import json
import struct
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms

from lumen import (catalog, color, downloads, engine, geo, icc, icc_input, library_db, photo_metadata,
                   preview_cache, signing, updates, watermark, world_map)
from test_ui import app, window  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = Path(os.environ.get('LUMEN_SAMPLES', r'E:\LumenSamples'))
EXIFTOOL = ROOT / 'assets' / 'exiftool' / ('exiftool.exe' if os.name == 'nt' else 'exiftool')
windows = pytest.mark.skipif(os.name != 'nt', reason='Windows handles')


# ----------------------------------------------------------------------------- the damaged library

HANDLE_REUSE = textwrap.dedent(r'''
    import ctypes, os, sqlite3, sys, tempfile
    from ctypes import wintypes
    from pathlib import Path
    sys.path.insert(0, sys.argv[1])
    work = Path(sys.argv[2])
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.GetStdHandle.restype = ctypes.c_void_p
    kernel32.GetFinalPathNameByHandleW.argtypes = [ctypes.c_void_p, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    if sys.argv[3] == 'pin':
        from lumen import native_stdio
        native_stdio.pin(('cv2',))
    saved = os.dup(2)
    capture = open(work / 'capture.txt', 'w+b')
    os.dup2(capture.fileno(), 2)                    # pytest's output capture during a test
    stale = kernel32.GetStdHandle(-12)
    import cv2                                      # a static C runtime keeps that handle
    os.dup2(saved, 2)                               # the capture ends: the handle is closed
    received = False
    connections = []
    for attempt in range(200):
        db = work / f'library-{attempt}.db'
        c = sqlite3.connect(db, isolation_level=None)
        c.execute('PRAGMA journal_mode=WAL')
        c.execute('CREATE TABLE IF NOT EXISTS t(id INTEGER PRIMARY KEY, v TEXT)')
        c.executemany('INSERT INTO t(v) VALUES (?)', [('x' * 200,)] * 30)
        connections.append(c)
        buffer = ctypes.create_unicode_buffer(1024)
        if kernel32.GetFinalPathNameByHandleW(stale, buffer, 1024, 0) and buffer.value.endswith(db.name):
            received = True
            break
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_WARNING)
    cv2.imread(str(work / 'missing.png'))           # OpenCV warns on its stderr
    print(received, b'WARN' in db.read_bytes())
''')


def _handle_reuse(tmp_path, mode):
    script = tmp_path / 'reuse.py'
    script.write_text(HANDLE_REUSE, encoding='utf-8')
    work = tmp_path / mode
    work.mkdir()
    result = subprocess.run([sys.executable, str(script), str(ROOT), str(work), mode], capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr.decode('utf-8', 'replace')
    received, written = result.stdout.decode().split()
    return received == 'True', written == 'True'


@windows
def test_a_closed_stderr_handle_lets_opencv_write_into_the_library_database(tmp_path):
    """The 2.1.1 "database disk image is malformed": OpenCV's static C runtime keeps the stderr
    handle it saw when it loaded; once that is closed, SQLite gets the same handle value and
    OpenCV's next warning is written into the database file."""
    received, written = _handle_reuse(tmp_path, 'plain')
    if not received:
        pytest.skip('SQLite did not get the closed handle this time')
    assert written


@windows
def test_native_libraries_loaded_with_a_pinned_stderr_never_write_into_files(tmp_path):
    _, written = _handle_reuse(tmp_path, 'pin')
    assert not written


@windows
def test_an_invalid_standard_handle_is_replaced_by_nul(tmp_path):
    script = tmp_path / 'repair.py'
    script.write_text(textwrap.dedent(r'''
        import ctypes, sys
        sys.path.insert(0, sys.argv[1])
        from lumen import native_stdio
        k = native_stdio._kernel32()
        k.SetStdHandle(native_stdio.STD_ERROR, 0x7ffc)        # a value that is no handle here
        print(native_stdio.repair(), native_stdio.valid(k.GetStdHandle(native_stdio.STD_ERROR)),
              k.GetStdHandle(native_stdio.STD_ERROR) == native_stdio.nul_handle())
    '''), encoding='utf-8')
    result = subprocess.run([sys.executable, '-I', str(script), str(ROOT)], capture_output=True, timeout=60)
    assert result.stdout.decode().split() == ['[-12]', 'True', 'True'], result.stderr.decode('utf-8', 'replace')


def test_the_native_warnings_are_turned_down():
    from lumen import performance
    assert os.environ.get('OPENCV_LOG_LEVEL') == 'ERROR'
    ort = performance.onnxruntime()
    assert ort is not None


# ----------------------------------------------------------------------------- input colour management

def _para_profile(primaries, description='Display P3 para test'):
    """A version-4-like matrix / TRC profile with the sRGB curve as a ``para`` type-3 function."""
    space = color._space('x', 'x', primaries)
    adapt = np.linalg.inv(icc.BRADFORD) @ np.diag((icc.BRADFORD @ icc.D50) / (icc.BRADFORD @ icc.D65)) @ icc.BRADFORD
    colorants = adapt @ np.asarray(space.to_xyz, np.float64)
    params = (2.4, 1 / 1.055, .055 / 1.055, 1 / 12.92, .04045)
    trc = b'para' + bytes(4) + struct.pack('>HH', 3, 0) + b''.join(icc._fixed(v) for v in params)
    tags = [(b'desc', icc._description(description)), (b'wtpt', icc._xyz(icc.D65)),
            (b'rXYZ', icc._xyz(colorants[:, 0])), (b'gXYZ', icc._xyz(colorants[:, 1])), (b'bXYZ', icc._xyz(colorants[:, 2])),
            (b'rTRC', trc), (b'gTRC', trc), (b'bTRC', trc)]
    offset = 128 + 4 + 12 * len(tags)
    table, data = [], b''
    for signature, body in tags:
        table.append(struct.pack('>4sII', signature, offset + len(data), len(body)))
        data += body + bytes(-len(body) % 4)
    size = offset + len(data)
    header = (struct.pack('>I', size) + b'test' + struct.pack('>I', 0x04300000) + b'mntrRGB XYZ ' + struct.pack('>6H', 2026, 1, 1, 0, 0, 0)
              + b'acsp' + bytes(24) + struct.pack('>I', 0) + b''.join(icc._fixed(v) for v in icc.D50) + b'test' + bytes(44))
    return header + struct.pack('>I', len(tags)) + b''.join(table) + data


P3 = [(.680, .320), (.265, .690), (.150, .060)]


def test_matrix_profiles_decode_to_unclipped_linear_srgb():
    for key in ('p3', 'adobe-rgb'):
        source = icc_input.read(icc.profile(key))
        np.testing.assert_allclose(source.to_srgb, color.space(key).to_srgb, atol=5e-5)
    assert icc_input.is_identity(icc_input.read(icc.profile(None)))
    para = icc_input.read(_para_profile(P3))
    assert para.label == 'Display P3 para test' and icc.identify(_para_profile(P3)) is None
    np.testing.assert_allclose(para.to_srgb, color.space('p3').to_srgb, atol=5e-5)
    red = icc_input.to_linear(np.array([[[255, 0, 0]]], np.uint8), para)[0, 0]
    assert red[0] > 1.15 and red[1] < -.03                 # outside sRGB, kept
    x = np.linspace(0, 1, 11)
    np.testing.assert_allclose(para.curves[0](x), color.srgb_to_linear(x.astype(np.float32)), atol=2e-4)


def test_icc_input_agrees_with_littlecms():
    rng = np.random.default_rng(3)
    pixels = rng.integers(0, 256, (48, 48, 3), np.uint8)
    for data in (icc.profile('adobe-rgb'), _para_profile(P3)):
        ours = icc_input.to_linear(pixels, icc_input.read(data))
        reference = np.asarray(ImageCms.profileToProfile(Image.fromarray(pixels), ImageCms.ImageCmsProfile(io.BytesIO(data)),
                                                         ImageCms.createProfile('sRGB'), outputMode='RGB')) / 255
        inside = np.all((ours > .003) & (ours < .997), axis=2)
        assert inside.sum() > 1000
        assert np.abs(color.linear_to_srgb(ours[inside]) - reference[inside]).max() < 1.5 / 255


def test_lookup_table_and_broken_profiles_are_left_to_littlecms():
    assert icc_input.read(b'') is None and icc_input.read(b'x' * 200) is None
    data = bytearray(_para_profile(P3))
    data[16:20] = b'CMYK'
    assert icc_input.read(bytes(data)) is None


def test_camera_adobe_rgb_comes_from_the_metadata():
    assert icc_input.camera_space({'ColorSpace': 2}) == 'adobe-rgb'                        # Nikon / Canon maker notes
    assert icc_input.camera_space({'ColorSpace': 65535, 'InteropIndex': 'R03 - DCF option file (Adobe RGB)'}) == 'adobe-rgb'
    assert icc_input.camera_space({'ColorSpace': 65535}, '_DSC0001.JPG') == 'adobe-rgb'
    assert icc_input.camera_space({'ColorSpace': 65535}, 'DSC0001.JPG') is None
    assert icc_input.camera_space({'ColorSpace': 1, 'InteropIndex': 'R98'}) is None
    assert icc_input.camera_space({}) is None


def _exiftool(*arguments):
    return subprocess.run([str(EXIFTOOL), *map(str, arguments)], capture_output=True, check=True)


def test_an_adobe_rgb_camera_jpeg_keeps_its_colours(tmp_path):
    path = tmp_path / 'DSC0001.JPG'
    Image.fromarray(np.full((16, 16, 3), (0, 255, 0), np.uint8)).save(path, quality=100)
    plain, info = engine.load_image(path, None)
    assert 'Adobe' not in info['note'] and plain[8, 8, 0] >= 0
    _exiftool('-overwrite_original', '-ColorSpace=Uncalibrated', '-InteropIndex=R03', path)
    rgb, info = engine.load_image(path, None)
    assert 'Adobe RGB (1998)' in info['note']
    assert rgb[8, 8, 0] < -.1 and rgb[8, 8, 2] < -.01        # Adobe RGB green lies outside sRGB


def test_a_16_bit_tiff_with_another_profile_is_converted(tmp_path):
    import tifffile
    path = tmp_path / 'p3.tif'
    pixels = np.zeros((8, 8, 3), np.uint16)
    pixels[..., 0] = 65535
    tifffile.imwrite(path, pixels, photometric='rgb', extratags=[(34675, 7, None, _para_profile(P3), True)])
    rgb, info = engine.load_image(path, None)
    assert 'Display P3 para test' in info['note'] and '16-bit' in info['note']
    np.testing.assert_allclose(rgb[4, 4], color.space('p3').to_srgb[:, 0], atol=2e-4)


HE = SAMPLES / 'Z50_2' / 'Nikon_Z50_2_DSC_0061_DX_HE_Star.NEF'


@pytest.mark.skipif(not HE.is_file(), reason='Nikon Z50 II sample not available')
def test_a_high_efficiency_nef_opens_its_jpeg_in_the_cameras_colour_space(monkeypatch):
    from lumen import white_balance
    rgb, info = engine.load_image(HE, 800)
    assert info['embedded'] and info['input_space'] == 'srgb' and '8 位 sRGB' in info['note']
    real = white_balance.metadata

    def adobe(path):
        tags, warning = real(path)
        return dict(tags, ColorSpace=2), warning
    monkeypatch.setattr(white_balance, 'metadata', adobe)
    wide, info = engine.load_image(HE, 800)
    assert info['input_space'] == 'adobe-rgb' and '8 位 Adobe RGB (1998)' in info['note']
    assert wide.shape == rgb.shape and np.abs(wide - rgb).max() > .02


def test_library_thumbnails_of_adobe_rgb_jpegs_are_converted(tmp_path):
    from lumen import previews
    path = tmp_path / 'DSC0002.JPG'
    Image.fromarray(np.full((16, 16, 3), (40, 200, 60), np.uint8)).save(path, quality=100)
    before = previews.camera_view(path, 64)[8, 8]
    _exiftool('-overwrite_original', '-ColorSpace=Uncalibrated', '-InteropIndex=R03', path)
    after = previews.camera_view(path, 64)[8, 8]
    assert after[1] > before[1] and after[0] < before[0]     # converted to sRGB: a more saturated green


# ----------------------------------------------------------------------------- signatures and mirrors

def test_ed25519_matches_rfc_8032():
    seed = bytes.fromhex('9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60')
    public = signing.public_key(seed)
    assert public.hex() == 'd75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a'
    signature = signing.sign(seed, b'')
    assert signature.hex() == ('e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e3970'
                               '1cf9b46bd25bf5f0595bbe24655141438e7a100b')
    assert signing.verify(public, b'', signature)
    assert not signing.verify(public, b'x', signature)
    assert not signing.verify(public, b'', signature[:63] + bytes([signature[63] ^ 1]))
    seed = bytes.fromhex('4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb')
    assert signing.sign(seed, b'\x72').hex().startswith('92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da')


def test_the_release_key_is_built_in():
    assert len(updates.FEED_KEYS) >= 1 and all(len(bytes.fromhex(key)) == 32 for key in updates.FEED_KEYS)
    assert updates.MIRROR_PAGE.startswith('https://') and downloads.GITHUB == 'https://github.com/hety0211/lumen-raw/releases/'
    assert [downloads.host(m) for m in downloads.MIRRORS] == ['103.195.190.106', 'lumenraw.com']


def _handler():
    class Files(BaseHTTPRequestHandler):
        files, requests = {}, []

        def log_message(self, *_):
            pass

        def do_GET(self):
            Files.requests.append((self.path, self.headers.get('Range')))
            data = Files.files.get(self.path)
            if data is None:
                self.send_error(404)
                return
            start = int(self.headers['Range'].split('=')[1].rstrip('-')) if self.headers.get('Range') else 0
            self.send_response(206 if start else 200)
            self.send_header('Content-Length', str(len(data) - start))
            self.end_headers()
            self.wfile.write(data[start:])
    return Files


@pytest.fixture
def mirrors(monkeypatch, tmp_path):
    """GitHub (a port nobody listens on) and two mirrors, as local servers; the feed key is a test key."""
    servers = []
    for _ in range(2):
        handler = _handler()
        httpd = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append((httpd, handler, f'http://127.0.0.1:{httpd.server_address[1]}/lumen-raw/releases/'))
    closed = ThreadingHTTPServer(('127.0.0.1', 0), _handler())
    port = closed.server_address[1]
    closed.server_close()
    github = f'http://127.0.0.1:{port}/hety0211/lumen-raw/releases/'
    monkeypatch.setattr(downloads, 'GITHUB', github)
    monkeypatch.setattr(downloads, '_preferred', None)
    monkeypatch.setattr(downloads, 'PAUSES', (0, 0))
    monkeypatch.setenv('LUMEN_MIRRORS', ','.join(base for _, _, base in servers))
    monkeypatch.delenv('LUMEN_UPDATE_FEED', raising=False)
    monkeypatch.setattr(updates, 'FEED', github + 'latest/download/lumen-update.json')
    monkeypatch.setattr(updates, 'TRUSTED', (github + 'download/',))
    monkeypatch.setattr(updates, 'mode', lambda: 'installer')
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'appdata'))
    seed = bytes(range(32))
    monkeypatch.setattr(updates, 'FEED_KEYS', (signing.public_key(seed).hex(),))
    yield dict(github=github, seed=seed, servers=[(handler, base) for _, handler, base in servers])
    for httpd, _, _ in servers:
        httpd.shutdown()
        httpd.server_close()


def _feed(github, version, payload):
    package = dict(url=github + f'download/v{version}/LumenRAW-{version}-Update.exe', bytes=len(payload),
                   sha256=hashlib.sha256(payload).hexdigest())
    entry = dict(version=version, update=package, models=None)
    return json.dumps(dict(format=1, version=version, page=github + 'tag', notes={'zh_CN': '镜像测试'},
                           windows=entry, macos=dict(entry))).encode('utf-8')


def test_github_down_reads_the_signed_feed_from_singapore_then_downloads_there(mirrors):
    (sg, sg_base), (hk, hk_base) = mirrors['servers']
    payload = os.urandom(300_000)
    body = _feed(mirrors['github'], '2.9.0', payload)
    sg.files['/lumen-raw/releases/latest/download/lumen-update.json'] = body
    sg.files['/lumen-raw/releases/latest/download/lumen-update.json.sig'] = signing.sign(mirrors['seed'], body).hex().encode()
    sg.files['/lumen-raw/releases/download/v2.9.0/LumenRAW-2.9.0-Update.exe'] = payload
    update = updates.check()
    assert update.version == '2.9.0' and update.source == '127.0.0.1' and update.notes == '镜像测试'
    assert downloads.alternatives(update.package['url'])[0].startswith(sg_base)
    installer = updates.download(update)
    assert installer.read_bytes() == payload and not hk.requests


def test_an_unsigned_or_forged_mirror_feed_is_refused(mirrors):
    (sg, _), (hk, _) = mirrors['servers']
    body = _feed(mirrors['github'], '2.9.0', b'x')
    sg.files['/lumen-raw/releases/latest/download/lumen-update.json'] = body          # no signature
    forged = _feed(mirrors['github'], '2.9.1', b'y')
    hk.files['/lumen-raw/releases/latest/download/lumen-update.json'] = forged
    hk.files['/lumen-raw/releases/latest/download/lumen-update.json.sig'] = signing.sign(bytes(32), forged).hex().encode()
    with pytest.raises(updates.UpdateError) as failure:
        updates.check()
    text = str(failure.value)
    assert text.count('127.0.0.1') == 3 and text.count('镜像上的更新信息没有有效签名') == 2
    hk.files['/lumen-raw/releases/latest/download/lumen-update.json.sig'] = signing.sign(mirrors['seed'], forged).hex().encode()
    assert updates.check().version == '2.9.1'                     # Hong Kong after Singapore


def test_a_file_missing_or_damaged_on_one_mirror_comes_from_the_next(mirrors, tmp_path):
    (sg, _), (hk, _) = mirrors['servers']
    data = os.urandom(200_000)
    url = mirrors['github'] + 'download/ai-models/model.onnx'
    sg.files['/lumen-raw/releases/download/ai-models/model.onnx'] = data[:100_000] + bytes(100_000)   # damaged copy
    hk.files['/lumen-raw/releases/download/ai-models/model.onnx'] = data
    target = tmp_path / 'model.onnx'
    downloads.fetch_verified(url, target, len(data), hashlib.sha256(data).hexdigest())
    assert target.read_bytes() == data
    sg.files.clear()
    hk.files.clear()
    with pytest.raises(downloads.Failed) as failure:
        downloads.fetch_verified(url, tmp_path / 'other.onnx', len(data), hashlib.sha256(data).hexdigest())
    assert len(failure.value.reasons) == 3 and 'HTTP 404' in str(failure.value)


def test_a_partial_download_from_github_is_finished_by_a_mirror(mirrors, tmp_path):
    (sg, _), _ = mirrors['servers']
    data = os.urandom(150_000)
    target = tmp_path / 'setup.exe'
    target.with_name('setup.exe.part').write_bytes(data[:60_000])
    sg.files['/lumen-raw/releases/download/v9/setup.exe'] = data
    downloads.fetch_verified(mirrors['github'] + 'download/v9/setup.exe', target, len(data), hashlib.sha256(data).hexdigest())
    assert target.read_bytes() == data and sg.requests[-1][1] == 'bytes=60000-'


def test_other_downloads_have_no_mirrors():
    assert downloads.alternatives('https://huggingface.co/x/y.onnx') == ['https://huggingface.co/x/y.onnx']
    copies = downloads.alternatives(downloads.GITHUB + 'download/speech-model-1/model.onnx')
    assert copies == [downloads.GITHUB + 'download/speech-model-1/model.onnx',
                      'https://103.195.190.106/lumen-raw/releases/download/speech-model-1/model.onnx',
                      'https://lumenraw.com/lumen-raw/releases/download/speech-model-1/model.onnx']


def test_the_feed_tool_signs_with_the_release_key(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / 'tools'))
    import make_update_feed
    seed = bytes(range(1, 33))
    key = tmp_path / 'update-signing.key'
    key.write_text(seed.hex() + '\n', encoding='ascii')
    monkeypatch.setattr(make_update_feed, 'KEY', key)
    feed = tmp_path / 'lumen-update.json'
    feed.write_bytes(b'{"format": 1}\n')
    with pytest.raises(SystemExit):
        make_update_feed.sign(feed)                          # not a key the editor knows
    monkeypatch.setattr(updates, 'FEED_KEYS', (signing.public_key(seed).hex(),))
    assert make_update_feed.sign(feed)
    assert updates.signed(feed.read_bytes(), (tmp_path / 'lumen-update.json.sig').read_bytes())
    monkeypatch.setattr(make_update_feed, 'KEY', tmp_path / 'missing.key')
    assert not make_update_feed.sign(feed) and not (tmp_path / 'lumen-update.json.sig').exists()


# ----------------------------------------------------------------------------- GPS positions

def test_coordinates_are_read_in_the_usual_forms():
    shanghai = (31.230278, 121.473611)
    for text in ('31.230278, 121.473611', '31°13′49″N 121°28′25″E', "31°13'49\"N 121°28'25\"E", '北纬31°13′49″ 东经121°28′25″',
                 'E121°28′25″ N31°13′49″', '31 13 49 N 121 28 25 E'):
        assert geo.parse(text) == pytest.approx(shanghai, abs=1e-5), text
    assert geo.parse('33°52′08″S 151°12′33″E') == pytest.approx((-33.868889, 151.209167), abs=1e-5)
    assert geo.parse('40.7128° N, 74.0060° W') == (40.7128, -74.006)
    for bad in ('', 'Shanghai', '95, 10', '10'):
        with pytest.raises(ValueError):
            geo.parse(bad)
    assert geo.dms(shanghai) == '31°13′49″N 121°28′25″E' and geo.decimal(shanghai) == '31.230278, 121.473611'
    assert geo.from_world(*geo.to_world(*shanghai)) == pytest.approx(shanghai)


def test_a_position_goes_into_the_xmp_sidecar_and_back(tmp_path):
    photo = tmp_path / 'DSC0001.ARW'
    photo.write_bytes(b'raw')
    assert geo.write(photo, (-33.868889, 151.209167)) == []
    assert geo.sidecar_position(photo) == (-33.868889, 151.209167)
    catalog.SIDECARS.queue(ratings={str(photo): dict(rating=3)})
    assert catalog.SIDECARS.flush() and geo.sidecar_position(photo) == (-33.868889, 151.209167)
    assert geo.write(photo, None) == [] and geo.sidecar_position(photo) is None
    assert 'Rating="3"' in catalog.sidecar(photo).read_text(encoding='utf-8') or \
        '<xmp:Rating>3</xmp:Rating>' in catalog.sidecar(photo).read_text(encoding='utf-8')


def _png(path):
    Image.fromarray(np.full((24, 32, 3), 120, np.uint8)).save(path)
    return path


def test_the_sidecar_position_wins_over_the_cameras_and_reaches_cached_photos(tmp_path):
    camera = photo_metadata.from_tags({'GPSLatitude': 10.5, 'GPSLongitude': 20.25})
    assert photo_metadata.position(camera) == (10.5, 20.25) and camera['gps_source'] == 'camera'
    assert photo_metadata.position(photo_metadata.from_tags({'GPSLatitude': 0, 'GPSLongitude': 0})) is None
    path = _png(tmp_path / 'a.png')
    cache = preview_cache.PreviewCache(tmp_path / 'cache', limit=64)
    assert photo_metadata.position(cache.load(path)[1]['photo']) is None
    geo.write(path, (35.6762, 139.6503))
    _, info = cache.load(path)                                   # decoded before: from the cache
    assert photo_metadata.position(info['photo']) == (35.6762, 139.6503) and info['photo']['gps_source'] == 'xmp'
    _, info = engine.load_image(path, 64)
    assert photo_metadata.position(info['photo']) == (35.6762, 139.6503)
    cache.close()


def test_exports_carry_the_chosen_position(tmp_path):
    source = _png(tmp_path / 'b.png')
    photo = photo_metadata.with_position({}, (-33.868889, 151.209167))
    for location, expected in ((True, True), (False, False)):
        target = tmp_path / f'out-{location}.jpg'
        engine.export_image(target, np.full((16, 16, 3), .5, np.float32), photo=photo,
                            metadata=dict(source=source, location=location))
        tags = json.loads(_exiftool('-j', '-n', '-GPSLatitude', '-GPSLatitudeRef', '-GPSLongitude', '-GPSPosition', target).stdout)[0]
        assert ('GPSLatitude' in tags) == expected
        if expected:
            assert tags['GPSLatitudeRef'] == 'S' and abs(tags['GPSLatitude']) == pytest.approx(33.868889, abs=1e-5)
            assert tags['GPSLongitude'] == pytest.approx(151.209167, abs=1e-5)


def test_the_library_reads_and_stores_positions(tmp_path):
    path = _png(tmp_path / 'c.png')
    geo.write(path, (48.8566, 2.3522))
    db = library_db.LibraryDB(tmp_path / 'library.db')
    added, _ = db.add([path])
    db.index_all()
    row = db.record(str(path))
    assert row['gps_lat'] == pytest.approx(48.8566, abs=1e-4) and row['gps_lon'] == pytest.approx(2.3522, abs=1e-4)
    db.set_position(str(path), None)
    assert db.record(str(path))['gps_lat'] is None
    db.close()


# ----------------------------------------------------------------------------- the border

PHOTO = dict(date='2026-10-28 14:32:10', body='Z 8', latitude='31.230400', longitude='121.473700', gps_source='xmp')


def test_the_position_row_is_off_by_default_and_follows_the_time():
    assert 'location' in watermark.LABELS and 'location' not in watermark.defaults()['fields']
    settings = dict(watermark.defaults(), fields=['body', 'date', 'location'])
    assert watermark.validate(settings)['fields'] == ['body', 'date', 'location']
    assert watermark.rows(settings, PHOTO) == [('Z 8', False), ('2026-10-28 14:32:10  ·  31°13′49″N  121°28′25″E', True)]
    assert watermark.rows(dict(settings, fields=['location']), PHOTO) == [('31°13′49″N  121°28′25″E', True)]
    assert watermark.rows(settings, dict(PHOTO, latitude='')) == [('Z 8', False), ('2026-10-28 14:32:10', False)]
    assert watermark.lines(dict(watermark.defaults()), PHOTO)[-1] == '2026-10-28 14:32:10'


def test_the_border_draws_a_map_pin():
    def red(settings):
        strip = watermark.band(900, 120, settings, PHOTO)
        pixels = np.round(strip * 255).astype(int)
        return int(np.sum((np.abs(pixels - watermark.PIN) < 30).all(axis=2)))
    with_pin = red(dict(watermark.defaults(), fields=['date', 'location'], camera_logo='', lens_logo=''))
    without = red(dict(watermark.defaults(), fields=['date'], camera_logo='', lens_logo=''))
    assert with_pin > 60 and without == 0


# ----------------------------------------------------------------------------- the map

def test_the_offline_map_draws_land_water_and_finds_towns():
    assert world_map.available()
    renderer = world_map.Renderer('light')
    colours = world_map.COLORS['light']
    inland = world_map.clamp(world_map.View(geo.to_world(30.6, 104.0), 6, 320, 200))   # Chengdu
    ocean = world_map.clamp(world_map.View(geo.to_world(5., -145.), 4, 320, 200))     # the Pacific, off the grid lines
    assert tuple(renderer.render(inland)[100, 160]) == colours['land']
    assert tuple(renderer.render(ocean)[100, 160]) == colours['ocean']
    assert world_map.search('上海', 'zh_CN')[0][0] == '上海'
    assert world_map.search('Shanghai', 'ja')[0][0] == '上海市'
    assert world_map.search('東京', 'en')[0][0] == 'Tokyo'
    place, km = world_map.nearest((31.23, 121.47))
    assert world_map.names('zh_CN')[place.index] == '上海' and km < 10
    assert world_map.nearest((0., -150.)) is None
    view = world_map.clamp(world_map.View(geo.to_world(31.23, 121.47), 9, 400, 300))
    assert view.position(200, 150) == pytest.approx((31.23, 121.47), abs=1e-4)
    assert list(world_map.clamp(world_map.View((0., .5), 1, 2000, 300)).offsets()) == [-2, -1, 0, 1]
    labels = renderer.labels(world_map.clamp(world_map.View(geo.to_world(31.2, 121.5), 6, 400, 300)), 'zh_CN')
    assert '上海' in [name for _, _, name, _ in labels]


def test_the_map_asset_is_public_domain_and_borderless():
    readme = (ROOT / 'assets' / 'map' / 'README.txt').read_text(encoding='utf-8')
    assert 'public domain' in readme and 'no national or provincial boundaries' in readme
    assert not any('admin' in name for name in world_map.data())


# ----------------------------------------------------------------------------- the picker and the 水印 panel

def test_the_picker_places_the_pin_by_click_and_by_search(app):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from lumen.location_dialog import LocationDialog
    dialog = LocationDialog(None, 'a.png')
    dialog.resize(900, 640)
    dialog.show()
    app.processEvents()
    assert dialog.position is None and not dialog.ok_button.isEnabled() and not dialog.clear_button.isEnabled()
    center = QPoint(dialog.map.width() // 2, dialog.map.height() // 2)
    QTest.mouseClick(dialog.map, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, center)
    expected = dialog.map.view.position(center.x(), center.y())
    assert dialog.position == pytest.approx(expected) and dialog.ok_button.isEnabled() and '″' in dialog.info.text()
    dialog.query.setText('杭州')
    assert dialog.results.isVisible() and dialog.results.count() >= 1
    dialog.choose(dialog.results.item(0))
    assert dialog.position == pytest.approx((30.25, 120.17), abs=.1) and dialog.map.view.zoom >= 10
    dialog.query.setText('31.2304, 121.4737')
    dialog.take_first()
    assert dialog.position == (31.2304, 121.4737)
    zoom = dialog.map.view.zoom
    dialog.map.zoom_by(-1)
    assert dialog.map.view.zoom == zoom - 1
    image = dialog.map.grab().toImage()
    assert image.width() == dialog.map.width()
    dialog.close()


def test_the_watermark_panel_writes_the_chosen_position(window, monkeypatch):
    from PySide6.QtWidgets import QDialog
    from lumen import location_dialog
    w = window
    path = Path(w.source_path)
    editor = w.watermark_editor
    assert editor.location_label.text() == '没有 GPS 位置' and editor.location_button.isEnabled()
    choice = {'position': (39.9042, 116.4074), 'cleared': False}

    def run(dialog):
        dialog.position, dialog.cleared = choice['position'], choice['cleared']
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(location_dialog.LocationDialog, 'exec', run)
    editor.location_button.click()
    assert geo.sidecar_position(path) == (39.9042, 116.4074)
    assert photo_metadata.position(w.info['photo']) == (39.9042, 116.4074) and w.info['photo']['gps_source'] == 'xmp'
    assert editor.location_label.text().startswith('39°54′15″N 116°24′27″E') and '已写入 XMP' in editor.location_label.text()
    assert editor.fields['location'].isChecked() and 'location' in w.edits['watermark']['fields']   # the pin row is on
    assert watermark.rows(w.edits['watermark'], w.info['photo'])[-1][1]
    choice.update(position=None, cleared=True)
    editor.location_button.click()
    assert geo.sidecar_position(path) is None and photo_metadata.position(w.info['photo']) is None
    assert editor.location_label.text() == '没有 GPS 位置'
