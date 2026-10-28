"""Camera-rendered previews embedded in RAW files (1.4.1).

LibRaw hands over JPEG and bitmap previews.  Canon CR3 files shot with HDR PQ
(e.g. EOS R5 Mark II) embed HEVC previews instead, which LibRaw cannot return;
they are decoded with the FFmpeg build that ships with OpenCV and converted from
BT.2020 PQ to SDR sRGB.  Develop references and filmstrip thumbnails therefore
still follow the camera's own rendering.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import struct
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

#: HDR reference white (ITU-R BT.2408) shown as SDR white.
REFERENCE_WHITE = 203.
#: Box types of Canon's preview (``PRVW``, 1620 × 1080) and thumbnail (``THMB``, 320 × 214).
CANON_PREVIEWS = (b'PRVW', b'THMB')


def libraw_preview(raw):
    """Upright sRGB uint8 preview LibRaw can extract (JPEG or bitmap), or None.

    A JPEG's own EXIF orientation wins; otherwise the RAW's orientation is applied.
    """
    import rawpy
    try:
        thumb = raw.extract_thumb()
    except Exception:
        return None
    if thumb.format == rawpy.ThumbFormat.JPEG:
        try:
            with Image.open(io.BytesIO(thumb.data)) as image:
                if image.getexif().get(0x0112, 1) != 1:
                    return np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
                return orient(np.asarray(image.convert('RGB')), raw.sizes.flip)
        except Exception:
            return None
    data = np.asarray(thumb.data)
    return orient(data[..., :3], raw.sizes.flip) if data.ndim == 3 else None


def _boxes(buffer):
    position = 0
    while position + 8 <= len(buffer):
        size, kind = struct.unpack('>I4s', buffer[position:position + 8])
        if size < 8 or position + size > len(buffer):
            return
        yield kind, buffer[position + 8:position + size]
        position += size


def canon_hevc(path):
    """Largest HEVC preview of a CR3 as ``(payload, width, height)``, or None."""
    with open(path, 'rb') as stream:
        head = stream.read(4 * 2**20)  # previews precede the sensor data (mdat)
    for kind in CANON_PREVIEWS:
        start = head.find(kind)
        if start < 4:
            continue
        # [size][type][version u32][u16][width u16][height u16][u16][length u32][payload]
        width, height, length = struct.unpack('>HHxxI', head[start + 10:start + 20])
        payload = head[start + 20:start + 20 + length]
        if len(payload) == length and b'hvcC' in payload[:64] and width and height:
            return payload, width, height
    return None


def _annex_b(payload):
    items = dict(_boxes(payload))
    config, picture = items.get(b'hvcC'), items.get(b'IMGD')
    if config is None or picture is None:
        raise ValueError('no HEVC configuration or picture')
    stream = bytearray()
    size = (config[21] & 3) + 1
    offset = 23
    for _ in range(config[22]):  # parameter-set arrays: VPS, SPS, PPS, SEI
        count = struct.unpack('>H', config[offset + 1:offset + 3])[0]
        offset += 3
        for _ in range(count):
            length = struct.unpack('>H', config[offset:offset + 2])[0]
            stream += b'\0\0\0\1' + config[offset + 2:offset + 2 + length]
            offset += 2 + length
    if int.from_bytes(picture[:4], 'big') == len(picture) - 4:
        picture = picture[4:]
    offset = 0
    while offset + size <= len(picture):
        length = int.from_bytes(picture[offset:offset + size], 'big')
        offset += size
        stream += b'\0\0\0\1' + picture[offset:offset + length]
        offset += length
    colour = items.get(b'colr', b'')
    nclx = struct.unpack('>4sHHHB', colour[:11]) if len(colour) >= 11 else None
    return bytes(stream), nclx


def pq_to_linear(signal):
    """SMPTE ST 2084 EOTF: normalized PQ signal to absolute luminance in nits."""
    m1, m2, c1, c2, c3 = 2610 / 16384, 2523 / 4096 * 128, 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    power = np.power(np.clip(signal, 0, 1), 1 / m2)
    return 10000 * np.power(np.maximum(power - c1, 0) / (c2 - c3 * power), 1 / m1)


def _sdr(nits):
    # BT.2408 reference white becomes SDR white; brighter highlights roll off smoothly.
    x = nits / REFERENCE_WHITE
    return np.where(x <= .8, x, .8 + .2 * (1 - np.exp(-(x - .8) / .2)))


def decode_hevc(payload, width, height):
    """sRGB uint8 image of a Canon HEVC preview (PQ previews are mapped to SDR)."""
    stream, nclx = _annex_b(payload)
    handle, name = tempfile.mkstemp(suffix='.hevc')
    try:
        with os.fdopen(handle, 'wb') as file:
            file.write(stream)
        capture = cv2.VideoCapture(name, cv2.CAP_FFMPEG)
        try:
            ok, bgr = capture.read()
        finally:
            capture.release()
    finally:
        Path(name).unlink(missing_ok=True)
    if not ok or bgr is None:
        raise ValueError('HEVC preview could not be decoded')
    rgb = bgr[:height, :width, ::-1].astype(np.float32) / 255
    primaries, transfer, matrix, full_range = (nclx[1], nclx[2], nclx[3], nclx[4] >> 7) if nclx else (1, 1, 1, 0)
    if transfer != 16:
        return np.ascontiguousarray(bgr[:height, :width, ::-1])
    # OpenCV converts with BT.601 coefficients and limited range; recover Y'CbCr exactly,
    # then apply the signalled BT.2020 matrix and range.
    y = rgb @ np.array([.299, .587, .114], np.float32)
    cb, cr = (rgb[..., 2] - y) / 1.772, (rgb[..., 0] - y) / 1.402
    if full_range:
        y, cb, cr = (y * 876 + 64) / 1023, cb * 896 / 1023, cr * 896 / 1023
    if matrix == 9:
        kr, kb = .2627, .0593
    else:
        kr, kb = .2126, .0722
    r = y + 2 * (1 - kr) * cr
    b = y + 2 * (1 - kb) * cb
    g = (y - kr * r - kb * b) / (1 - kr - kb)
    linear = pq_to_linear(np.stack([r, g, b], axis=-1))
    if primaries == 9:  # BT.2020 to BT.709 / sRGB primaries
        linear = linear @ np.array([[1.6605, -.5876, -.0728], [-.1246, 1.1329, -.0083],
                                    [-.0182, -.1006, 1.1187]], np.float32).T
    from .engine import to_srgb
    return np.round(np.clip(to_srgb(_sdr(np.maximum(linear, 0))), 0, 1) * 255).astype(np.uint8)


def hevc_preview(path):
    """sRGB uint8 HEVC preview of a CR3 in sensor orientation, or None."""
    if Path(path).suffix.lower() != '.cr3':
        return None
    try:
        found = canon_hevc(path)
        return decode_hevc(*found) if found is not None else None
    except Exception:
        log.info('CR3 HEVC preview unavailable for %s', path, exc_info=True)
        return None


def orient(image, flip):
    """Apply LibRaw's ``sizes.flip`` (0, 3, 5, 6) to a sensor-oriented image."""
    return {3: lambda a: np.rot90(a, 2), 5: lambda a: np.rot90(a, 1), 6: lambda a: np.rot90(a, -1)}.get(flip, lambda a: a)(image)


# --------------------------------------------------------------------------- 2.0.0: camera views

#: Long edge of the camera views the library and culling show (and cache).
VIEW_EDGE = 2560
#: Formats Pillow decodes quickly enough that their views are not cached.
QUICK = ('.jpg', '.jpeg')


def _fit(image, long_edge):
    if long_edge and max(image.size) > long_edge:
        scale = long_edge / max(image.size)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                             Image.Resampling.BICUBIC, reducing_gap=2.0)
    return image


def _draft(image, long_edge):
    """Let the JPEG decoder skip detail the view does not need (DCT scaling: ½, ¼, ⅛)."""
    if long_edge and image.format == 'JPEG':
        scale = long_edge / max(image.size)
        if scale < 1:
            image.draft('RGB', (max(1, round(image.width * scale)), max(1, round(image.height * scale))))


_transforms = {}
_transforms_lock = threading.Lock()


def _transform(profile):
    """A cached LittleCMS transform from an embedded profile to sRGB, or None for sRGB itself.
    Building one takes up to seconds; without the pixel cache (NOCACHE) it may be shared by threads."""
    from PIL import ImageCms
    key = hashlib.sha1(profile).digest()
    with _transforms_lock:
        if key not in _transforms:
            try:
                source = ImageCms.ImageCmsProfile(io.BytesIO(profile))
                if 'srgb' in (ImageCms.getProfileDescription(source) or '').lower():
                    _transforms[key] = None
                else:
                    flags = ImageCms.Flags.NOCACHE if hasattr(ImageCms, 'Flags') else 0x0040
                    _transforms[key] = ImageCms.buildTransform(source, ImageCms.createProfile('sRGB'), 'RGB', 'RGB',
                                                               flags=flags)
            except Exception:
                log.debug('ICC profile not usable', exc_info=True)
                _transforms[key] = None
        return _transforms[key]


def _exif_adobe(image):
    """Whether a JPEG's own EXIF says Adobe RGB (``icc_input.camera_space``)."""
    try:
        exif = image.getexif()
        main = exif.get_ifd(0x8769)
        interop = exif.get_ifd(0xA005).get(0x0001, '') if 0xA005 in main else ''
        from . import icc_input
        return icc_input.camera_space(dict(ColorSpace=main.get(0xA001), InteropIndex=interop),
                                      getattr(image, 'filename', '') and Path(image.filename).name) is not None
    except Exception:
        return False


def _srgb(image):
    """``image`` in sRGB for display: an embedded ICC profile is converted (LUMEN's own wide-gamut
    exports and other profiles alike).  2.1.2: a camera JPEG without a profile that its EXIF marks
    as Adobe RGB is converted too."""
    profile = image.info.get('icc_profile')
    if not profile and _exif_adobe(image):
        from . import icc
        profile = icc.profile('adobe-rgb')
    image = image.convert('RGB')
    if profile:
        transform = _transform(profile)
        if transform is not None:
            try:
                from PIL import ImageCms
                image = ImageCms.applyTransform(image, transform)
            except Exception:
                log.debug('ICC conversion skipped', exc_info=True)
    return image


def jpeg_view(data, long_edge=VIEW_EDGE, flip=0):
    """Upright sRGB uint8 array of JPEG bytes (an embedded preview), at most ``long_edge``."""
    with Image.open(io.BytesIO(data)) as image:
        _draft(image, long_edge)
        oriented = image.getexif().get(0x0112, 1) != 1
        upright = ImageOps.exif_transpose(image) if oriented else image
        array = np.asarray(_fit(upright.convert('RGB'), long_edge))
    return array if oriented else orient(array, flip)


def embedded_view(path, long_edge=VIEW_EDGE):
    """The camera's own rendering of a RAW file (its largest embedded preview), upright, sRGB uint8,
    at most ``long_edge``; ``(array, full_size)`` where ``full_size`` is True when the preview has
    at least half the sensor's pixels (good enough to judge focus at 100 %).  None without one."""
    import rawpy
    with rawpy.imread(str(path)) as raw:
        flip, pixels = raw.sizes.flip, raw.sizes.width * raw.sizes.height
        try:
            thumb = raw.extract_thumb()
        except Exception:
            thumb = None
        if thumb is not None and thumb.format == rawpy.ThumbFormat.JPEG:
            try:
                with Image.open(io.BytesIO(thumb.data)) as probe:
                    size = probe.size
                return jpeg_view(thumb.data, long_edge, flip), size[0] * size[1] >= .5 * pixels
            except Exception:
                log.info('embedded JPEG of %s unreadable', path, exc_info=True)
                return None
        if thumb is not None and np.asarray(thumb.data).ndim == 3:
            data = np.asarray(thumb.data)[..., :3]
            array = orient(np.asarray(_fit(Image.fromarray(np.ascontiguousarray(data)), long_edge)), flip)
            return array, data.shape[0] * data.shape[1] >= .5 * pixels
    hevc = hevc_preview(path)
    if hevc is not None:
        array = orient(np.asarray(_fit(Image.fromarray(hevc), long_edge)), flip)
        return array, False
    return None


def camera_view(path, long_edge=VIEW_EDGE):
    """What the library and the culling view show of a photo: upright sRGB uint8 at most
    ``long_edge`` (None: full size).  RAW files show the camera's embedded preview; without one
    (or for LUMEN's linear DNG and 16-bit TIFF) the photo is decoded and developed."""
    from . import engine, dng
    path = Path(path)
    suffix = path.suffix.lower()
    raw = suffix in engine.RAW_EXTENSIONS and not (suffix == '.dng' and dng.is_rendered(path))
    if raw:
        found = embedded_view(path, long_edge)
        if found is not None:
            return found[0]
    elif suffix not in ('.tif', '.tiff', '.dng'):
        with Image.open(path) as image:
            _draft(image, long_edge)
            upright = ImageOps.exif_transpose(image)
            return np.asarray(_fit(_srgb(upright), long_edge))
    rgb, info = engine.load_image(path, long_edge or None)
    developed = engine.develop_view(rgb, dict(develop=info.get('develop', {}), process=2))
    return np.round(np.clip(developed, 0, 1) * 255).astype(np.uint8)


def encode_jpeg(array, quality=88):
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(array)).save(buffer, 'JPEG', quality=quality, subsampling=0 if quality >= 90 else 2,
                                                       optimize=False)
    return buffer.getvalue()


def thumbnail_of(array, long_edge):
    """A smaller copy of an upright uint8 view."""
    return np.asarray(_fit(Image.fromarray(np.ascontiguousarray(array)), long_edge))
