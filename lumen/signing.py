"""Ed25519 signatures (RFC 8032) for the update feed (2.1.2), standard library only.

The feed ``lumen-update.json`` names the update installer and the AI models with their SHA-256.
When GitHub cannot be reached the editor reads it from LUMEN RAW's own mirrors (``updates``);
there the feed must carry a valid signature (``lumen-update.json.sig``) by the release key, whose
public half is built into the editor, so a mirror can serve the files but never change them.
``tools/make_update_feed.py`` signs; the private key never leaves the publishing computer.

Plain Python integers: verifying one signature takes a few milliseconds.
"""
from __future__ import annotations

import hashlib

P = 2 ** 255 - 19
L = 2 ** 252 + 27742317777372353535851937790883648493
D = -121665 * pow(121666, P - 2, P) % P
I = pow(2, (P - 1) // 4, P)


def _sha512(data):
    return hashlib.sha512(data).digest()


def _recover_x(y, sign):
    if y >= P:
        return None
    x2 = (y * y - 1) * pow(D * y * y + 1, P - 2, P) % P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (P + 3) // 8, P)
    if (x * x - x2) % P:
        x = x * I % P
    if (x * x - x2) % P:
        return None
    if x & 1 != sign:
        x = P - x
    return x


_BY = 4 * pow(5, P - 2, P) % P
_BX = _recover_x(_BY, 0)
BASE = (_BX, _BY, 1, _BX * _BY % P)
ZERO = (0, 1, 1, 0)


def _add(a, b):
    x1, y1, z1, t1 = a
    x2, y2, z2, t2 = b
    A = (y1 - x1) * (y2 - x2) % P
    B = (y1 + x1) * (y2 + x2) % P
    C = 2 * t1 * t2 * D % P
    Dd = 2 * z1 * z2 % P
    E, F, G, H = B - A, Dd - C, Dd + C, B + A
    return E * F % P, G * H % P, F * G % P, E * H % P


def _multiply(scalar, point):
    result = ZERO
    while scalar:
        if scalar & 1:
            result = _add(result, point)
        point = _add(point, point)
        scalar >>= 1
    return result


def _encode(point):
    x, y, z, _ = point
    inverse = pow(z, P - 2, P)
    x, y = x * inverse % P, y * inverse % P
    return int.to_bytes(y | ((x & 1) << 255), 32, 'little')


def _decode(data):
    if len(data) != 32:
        return None
    y = int.from_bytes(data, 'little')
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    return None if x is None else (x, y, 1, x * y % P)


def _secret(seed):
    digest = _sha512(seed)
    a = int.from_bytes(digest[:32], 'little')
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, digest[32:]


def public_key(seed):
    """The 32-byte public key of a 32-byte private seed."""
    return _encode(_multiply(_secret(seed)[0], BASE))


def sign(seed, message):
    """The 64-byte signature of ``message`` (bytes) with a 32-byte private seed."""
    if len(seed) != 32:
        raise ValueError('an Ed25519 seed has 32 bytes')
    a, prefix = _secret(seed)
    public = _encode(_multiply(a, BASE))
    r = int.from_bytes(_sha512(prefix + message), 'little') % L
    R = _encode(_multiply(r, BASE))
    k = int.from_bytes(_sha512(R + public + message), 'little') % L
    return R + int.to_bytes((r + k * a) % L, 32, 'little')


def verify(public, message, signature):
    """Whether ``signature`` (64 bytes) is ``public``'s (32 bytes) signature of ``message``."""
    if len(public) != 32 or len(signature) != 64:
        return False
    A, R = _decode(public), _decode(signature[:32])
    s = int.from_bytes(signature[32:], 'little')
    if A is None or R is None or s >= L:
        return False
    k = int.from_bytes(_sha512(signature[:32] + public + message), 'little') % L
    return _encode(_multiply(s, BASE)) == _encode(_add(R, _multiply(k, A)))
