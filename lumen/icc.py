"""ICC profiles embedded in exports: sRGB (LittleCMS's built-in), Display P3 and Adobe RGB (1998).

The wide spaces are written here as ICC v2 matrix / TRC display profiles: the D65 primaries of
``color.SPACES`` adapted to the D50 connection space with the Bradford transform, and the space's
transfer curve (Display P3 uses the sRGB curve, sampled at 1024 points; Adobe RGB a 563/256
gamma).  Generating them avoids shipping third-party profile files; LittleCMS reads them back
in the tests and they convert exactly as the matrices in ``color``.
"""
from __future__ import annotations

import io
import struct
from functools import lru_cache

import numpy as np

from . import color

D50 = np.array([.9642, 1., .8249])
D65 = np.array([.95047, 1., 1.08883])
BRADFORD = np.array([[.8951, .2664, -.1614], [-.7502, 1.7135, .0367], [.0389, -.0685, 1.0296]])
DESCRIPTIONS = {'p3': 'Display P3', 'adobe-rgb': 'Adobe RGB (1998) compatible'}
COPYRIGHT = 'No copyright, use freely. Written by LUMEN RAW.'


def _fixed(value):
    return struct.pack('>i', int(round(value * 65536)))


def _xyz(values):
    return b'XYZ ' + bytes(4) + b''.join(_fixed(v) for v in values)


def _text(text):
    return b'text' + bytes(4) + text.encode('ascii') + b'\0'


def _description(text):
    ascii_ = text.encode('ascii') + b'\0'
    return (b'desc' + bytes(4) + struct.pack('>I', len(ascii_)) + ascii_ +
            struct.pack('>II', 0, 0) + struct.pack('>HB', 0, 0) + bytes(67))


def _curve(space):
    if space.gamma:
        return b'curv' + bytes(4) + struct.pack('>IH', 1, int(round(space.gamma * 256)))
    samples = color.srgb_to_linear(np.linspace(0, 1, 1024, dtype=np.float32))
    values = np.round(np.clip(samples, 0, 1) * 65535).astype('>u2')
    return b'curv' + bytes(4) + struct.pack('>I', len(values)) + values.tobytes()


def _profile(space):
    adapt = np.linalg.inv(BRADFORD) @ np.diag((BRADFORD @ D50) / (BRADFORD @ D65)) @ BRADFORD
    colorants = adapt @ np.asarray(space.to_xyz, np.float64)
    trc = _curve(space)
    tags = [(b'desc', _description(DESCRIPTIONS[space.key])), (b'cprt', _text(COPYRIGHT)), (b'wtpt', _xyz(D65)),
            (b'rXYZ', _xyz(colorants[:, 0])), (b'gXYZ', _xyz(colorants[:, 1])), (b'bXYZ', _xyz(colorants[:, 2])),
            (b'rTRC', trc), (b'gTRC', trc), (b'bTRC', trc)]
    offset = 128 + 4 + 12 * len(tags)
    table, data, placed = [], b'', {}
    for signature, body in tags:
        if body not in placed:                      # the three TRC tags share one curve
            placed[body] = offset + len(data)
            data += body + bytes(-len(body) % 4)
        table.append(struct.pack('>4sII', signature, placed[body], len(body)))
    size = offset + len(data)
    header = (struct.pack('>I', size) + b'lcms' + struct.pack('>I', 0x02100000) + b'mntrRGB XYZ ' +
              struct.pack('>6H', 2026, 1, 1, 0, 0, 0) + b'acsp' + bytes(4) + struct.pack('>I', 0) +
              bytes(4) + bytes(4) + bytes(8) + struct.pack('>I', 0) + b''.join(_fixed(v) for v in D50) +
              b'LUMN' + bytes(16) + bytes(28))
    assert len(header) == 128
    return header + struct.pack('>I', len(tags)) + b''.join(table) + data


@lru_cache(maxsize=4)
def profile(key=None):
    """ICC profile bytes for an output space key (None: sRGB)."""
    space = color.space(key)
    if space.key == 'srgb':
        from PIL import ImageCms
        return ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    return _profile(space)


def identify(data):
    """The output space key of a profile LUMEN RAW wrote (``'p3'``, ``'adobe-rgb'``), else None."""
    if not data:
        return None
    for key in ('p3', 'adobe-rgb'):
        if bytes(data) == profile(key):
            return key
    return None


def load(data):
    """A Pillow ``ImageCmsProfile`` for profile bytes."""
    from PIL import ImageCms
    return ImageCms.ImageCmsProfile(io.BytesIO(bytes(data)))
