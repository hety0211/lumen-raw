"""Colour management of the photos LUMEN RAW opens (2.1.2).

Until 2.1.1 only LUMEN's own export profiles kept their colours on input: 8-bit JPEG / PNG / TIFF
with any other profile went through LittleCMS to 8-bit sRGB (colours outside sRGB clipped), 16-bit
files and camera files without a profile were read as sRGB.

Now an embedded ICC profile of the usual matrix / TRC kind (Display P3, Adobe RGB (1998), ProPhoto
RGB, Rec. 2020, camera and monitor profiles; version 2 and 4) is read here: each channel's tone
curve (``curv`` gamma or table, ``para`` functions 0–4) makes it linear, its colorants (D50 in the
connection space) go to XYZ, the Bradford transform to D65 and the result to linear sRGB primaries
without clipping, so wide-gamut colours reach process 2 intact, at 8 or 16 bits.  Grey profiles
(``kTRC``) work the same way.  Profiles built from lookup tables (printer, CMYK, Lab) still go
through LittleCMS.

A camera file without a profile is Adobe RGB when its metadata says so: EXIF ``ColorSpace`` 2
(written by some cameras, and in Nikon's and Canon's maker notes), or ``Uncalibrated`` (65535)
with the DCF interoperability index ``R03``.  This also covers the full-size JPEG inside a Nikon
"High Efficiency" NEF, which LUMEN opens because no open decoder reads that compression.
"""
from __future__ import annotations

import struct
from typing import Callable, NamedTuple

import numpy as np

from . import color

D50 = np.array([.9642, 1., .8249])
D65 = np.array([.95047, 1., 1.08883])
BRADFORD = np.array([[.8951, .2664, -.1614], [-.7502, 1.7135, .0367], [.0389, -.0685, 1.0296]])
#: Linear sRGB → XYZ (D65), as ``color`` builds it.
SRGB_TO_XYZ = color._rgb_to_xyz([(.64, .33), (.30, .60), (.15, .06)])
#: Rows of a strip converted at once (keeps the temporaries of a 100 MP photo small).
STRIP = 512


class Source(NamedTuple):
    """How to turn a profile's encoded values into linear sRGB."""
    label: str
    curves: tuple                  # one callable per channel (grey: one): encoded 0–1 → linear
    to_srgb: np.ndarray            # 3 × 3, linear profile RGB → linear sRGB (identity for grey)
    gray: bool = False


def _adapt(source, target):
    return np.linalg.inv(BRADFORD) @ np.diag((BRADFORD @ target) / (BRADFORD @ source)) @ BRADFORD


def _s15(data, offset):
    return struct.unpack_from('>i', data, offset)[0] / 65536


def _tags(data):
    count = struct.unpack_from('>I', data, 128)[0]
    if count > 200:
        raise ValueError('too many tags')
    tags = {}
    for i in range(count):
        signature, offset, size = struct.unpack_from('>4sII', data, 132 + 12 * i)
        if offset + size > len(data) or size < 8:
            raise ValueError('tag outside the profile')
        tags[signature] = data[offset:offset + size]
    return tags


def _xyz(tag):
    if tag[:4] != b'XYZ ':
        raise ValueError('not an XYZ tag')
    return np.array([_s15(tag, 8), _s15(tag, 12), _s15(tag, 16)])


def _curve(tag) -> Callable[[np.ndarray], np.ndarray]:
    kind = tag[:4]
    if kind == b'curv':
        count = struct.unpack_from('>I', tag, 8)[0]
        if count == 0:
            return lambda x: x
        if count == 1:
            gamma = struct.unpack_from('>H', tag, 12)[0] / 256
            return lambda x: np.power(x, gamma)
        table = np.frombuffer(tag, '>u2', count, 12).astype(np.float64) / 65535
        grid = np.linspace(0, 1, count)
        return lambda x: np.interp(x, grid, table)
    if kind == b'para':
        function = struct.unpack_from('>H', tag, 8)[0]
        n = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}.get(function)
        if n is None:
            raise ValueError('unknown parametric curve')
        g, a, b, c, d, e, f = (*[_s15(tag, 12 + 4 * i) for i in range(n)], 0, 0, 0, 0, 0, 0)[:7]
        if function == 0:
            return lambda x: np.power(x, g)
        if function in (1, 2):
            offset = c if function == 2 else 0.
            edge = -b / a if a else 0.
            return lambda x: np.where(x >= edge, np.power(np.maximum(a * x + b, 0), g) + offset, offset)
        if function == 3:
            return lambda x: np.where(x >= d, np.power(np.maximum(a * x + b, 0), g), c * x)
        return lambda x: np.where(x >= d, np.power(np.maximum(a * x + b, 0), g) + e, c * x + f)
    raise ValueError('lookup-table curves are not read here')


def _description(tags):
    tag = tags.get(b'desc', b'')
    try:
        if tag[:4] == b'desc':
            length = struct.unpack_from('>I', tag, 8)[0]
            return tag[12:12 + length].split(b'\0')[0].decode('latin-1').strip()
        if tag[:4] == b'mluc':
            count, size = struct.unpack_from('>II', tag, 8)
            records = [struct.unpack_from('>2s2sII', tag, 16 + i * size) for i in range(count)]
            language, country, length, offset = next((r for r in records if r[0] == b'en'), records[0])
            return tag[offset:offset + length].decode('utf-16-be').strip('\0 ')
    except (struct.error, StopIteration, UnicodeDecodeError, IndexError):
        pass
    return ''


def read(data):
    """A ``Source`` for ICC profile bytes of the matrix / TRC kind (RGB or grey), else None
    (missing, damaged, or built from lookup tables)."""
    if not data or len(data) < 132:
        return None
    data = bytes(data)
    try:
        if struct.unpack_from('>I', data, 0)[0] > len(data) + 4 or data[36:40] != b'acsp':
            return None
        space, connection = data[16:20], data[20:24]
        tags = _tags(data)
        label = _description(tags) or 'ICC'
        if space == b'GRAY' and b'kTRC' in tags:
            return Source(label[:60], (_curve(tags[b'kTRC']),), np.eye(3), True)
        if space != b'RGB ' or connection != b'XYZ ':
            return None
        if not all(key in tags for key in (b'rXYZ', b'gXYZ', b'bXYZ', b'rTRC', b'gTRC', b'bTRC')):
            return None
        colorants = np.stack([_xyz(tags[key]) for key in (b'rXYZ', b'gXYZ', b'bXYZ')], axis=1)
        curves = tuple(_curve(tags[key]) for key in (b'rTRC', b'gTRC', b'bTRC'))
    except (struct.error, ValueError):
        return None
    if not np.all(np.isfinite(colorants)) or abs(np.linalg.det(colorants)) < 1e-6:
        return None
    to_srgb = np.linalg.inv(SRGB_TO_XYZ) @ _adapt(D50, D65) @ colorants
    return Source(label[:60], curves, to_srgb, False)


def standard(key):
    """A ``Source`` for one of LUMEN's colour spaces (``'adobe-rgb'``, ``'p3'``, ``'srgb'``)."""
    space = color.space(key)
    curve = (lambda x: np.power(x, space.gamma)) if space.gamma else color.srgb_to_linear
    return Source(space.label, (curve,) * 3, np.asarray(space.to_srgb, np.float64))


def camera_space(tags, name=''):
    """``'adobe-rgb'`` when a file without a profile was encoded in Adobe RGB by the camera
    (metadata read with ``-ColorSpace#`` and ``-InteropIndex``), else None (sRGB)."""
    try:
        value = int((tags or {}).get('ColorSpace') or 0)
    except (TypeError, ValueError):
        value = 0
    interop = str((tags or {}).get('InteropIndex') or '').upper()
    if value == 2:
        return 'adobe-rgb'
    if value == 0xFFFF and (interop.startswith('R03') or (not interop and str(name).startswith('_'))):
        return 'adobe-rgb'      # DCF: uncalibrated + R03, or the "_DSC" names of Adobe RGB files
    return None


def is_identity(source, tolerance=2e-3):
    """True for sRGB itself (matrix and curves), which needs no work."""
    if source.gray or not np.allclose(source.to_srgb, np.eye(3), atol=tolerance):
        return False
    x = np.linspace(0, 1, 33)
    return all(np.allclose(curve(x), color.srgb_to_linear(x.astype(np.float32)), atol=tolerance) for curve in source.curves)


def _tables(source, levels):
    x = np.arange(levels, dtype=np.float64) / (levels - 1)
    return [np.asarray(curve(x), np.float32) for curve in source.curves]


def to_linear(pixels, source):
    """Encoded ``pixels`` (uint8 / uint16, H × W × 3, or H × W for grey) → linear sRGB float32,
    unclipped (colours outside sRGB keep negative or above-one values)."""
    import cv2
    pixels = np.asarray(pixels)
    if pixels.dtype not in (np.uint8, np.uint16):
        raise ValueError('8 or 16-bit pixels expected')
    levels = 256 if pixels.dtype == np.uint8 else 65536
    tables = _tables(source, levels)
    height, width = pixels.shape[:2]
    from . import large_image
    out = large_image.allocate((height, width, 3))
    matrix = np.asarray(source.to_srgb, np.float32)
    for y in range(0, height, STRIP):
        block = pixels[y:y + STRIP]
        if source.gray or block.ndim == 2:
            grey = tables[0][block if block.ndim == 2 else block[..., 0]]
            out[y:y + STRIP] = grey[..., None]
            continue
        linear = np.stack([tables[c][block[..., c]] for c in range(3)], axis=2)
        out[y:y + STRIP] = cv2.transform(linear, matrix)
    return out
