"""Creative colour lookup tables (.cube) as an adjustment (1.7.0).

A LUT is part of a recipe (``edits['lut']``) with a strength, like the other look settings:

* Stage: after the global colour stage (curves, mixer, grading) and the local masks, before
  vignette, grain and the crop.  It acts on display-referred sRGB 0–1, what .cube files
  for photographs and video are made for.  Process 1 renders in sRGB anyway; process 2 with a
  wider output space (Display P3, Adobe RGB) evaluates the LUT on the sRGB encoding of the
  colour and keeps what lies outside sRGB as an offset (``apply``).
* Interpolation: tetrahedral (the neutral axis stays neutral, as in Resolve and OCIO).  The
  stage is pointwise and uses exact arithmetic only (floor, multiply, add, compare), so a
  block render matches the whole-frame render bit for bit.  It runs on the CPU, also after the
  GPU graphs.
* Strength: ``x + (LUT(x) − x) · amount / 100``.
* Storage: the table itself travels with the recipe (float32, zlib, base64), so a project opens
  on another computer without the .cube file; projects and albums store each table once
  (``pack`` / ``unpack``).  3D tables above ``MAX_3D`` points per axis are resampled to it.

The .cube format is Adobe's Cube LUT Specification 1.0 (``LUT_1D_SIZE`` / ``LUT_3D_SIZE``,
``DOMAIN_MIN`` / ``DOMAIN_MAX``, red changing fastest) plus Resolve's variant: a 1D shaper
before a 3D table in one file and ``LUT_1D_INPUT_RANGE`` / ``LUT_3D_INPUT_RANGE``.
"""
from __future__ import annotations

import base64
import hashlib
import math
import re
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import performance
from .i18n import tr

#: Largest 3D table kept as is; bigger ones are resampled (65³ is the common "fine" size).
MAX_3D = 65
MAX_1D = 65536
#: Largest .cube file read (a 256³ table as text is about 400 MB; 64 MB covers 129³).
MAX_FILE = 64 * 1024 * 1024
#: Output values beyond this are rejected as nonsense.
MAX_VALUE = 64.
#: Pixels per work item when a frame is split over threads.
CHUNK = 1 << 18
IDENTITY_DOMAIN = [[0., 0., 0.], [1., 1., 1.]]


class Table(NamedTuple):
    """Decoded tables of a LUT setting: ``shaper`` (n, 3) and ``cube`` (n³, 3), red fastest."""
    shaper: np.ndarray | None
    cube: np.ndarray | None
    size: int
    domain_1d: np.ndarray
    domain_3d: np.ndarray


# --------------------------------------------------------------------------- .cube files

def _numbers(words, count, line):
    if len(words) != count:
        raise ValueError(tr('LUT 第 {line} 行格式无效。', line=line))
    try:
        values = [float(w) for w in words]
    except ValueError:
        raise ValueError(tr('LUT 第 {line} 行格式无效。', line=line))
    if not all(math.isfinite(v) for v in values):
        raise ValueError(tr('LUT 第 {line} 行格式无效。', line=line))
    return values


def parse_cube(data, name='LUT'):
    """A ``.cube`` file (bytes or text) as a recipe LUT setting at full strength.

    Returns ``(setting, notes)``; ``notes`` lists what was ignored or changed (resampling)."""
    if isinstance(data, bytes):
        if len(data) > MAX_FILE:
            raise ValueError(tr('LUT 文件过大。'))
        data = data.decode('utf-8', 'replace')
    notes = []
    size_1d = size_3d = 0
    domain = None
    domain_1d = domain_3d = None
    title = ''
    rows = []
    first_row = 0
    for number, raw in enumerate(data.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line[0] in '+-.0123456789':
            if not rows:
                first_row = number
            rows.append(line)
            continue
        if rows:
            raise ValueError(tr('LUT 第 {line} 行格式无效。', line=number))
        keyword, _, rest = line.partition(' ')
        keyword = keyword.upper()
        words = rest.split()
        if keyword == 'TITLE':
            title = rest.strip().strip('"')[:80]
        elif keyword in ('LUT_1D_SIZE', 'LUT_3D_SIZE'):
            value = _numbers(words, 1, number)[0]
            if not value.is_integer():
                raise ValueError(tr('LUT 第 {line} 行格式无效。', line=number))
            value = int(value)
            if keyword == 'LUT_1D_SIZE':
                if not 2 <= value <= MAX_1D:
                    raise ValueError(tr('不支持的 1D LUT 尺寸 {size}（2–{limit}）。', size=value, limit=MAX_1D))
                size_1d = value
            else:
                if not 2 <= value <= 256:
                    raise ValueError(tr('不支持的 3D LUT 尺寸 {size}（2–256）。', size=value))
                size_3d = value
        elif keyword in ('DOMAIN_MIN', 'DOMAIN_MAX'):
            domain = domain or [[0., 0., 0.], [1., 1., 1.]]
            domain[keyword == 'DOMAIN_MAX'] = _numbers(words, 3, number)
        elif keyword in ('LUT_1D_INPUT_RANGE', 'LUT_3D_INPUT_RANGE'):
            low, high = _numbers(words, 2, number)
            pair = [[low] * 3, [high] * 3]
            if keyword == 'LUT_1D_INPUT_RANGE':
                domain_1d = pair
            else:
                domain_3d = pair
        else:
            notes.append(tr('已忽略 LUT 关键字 {keyword}', keyword=keyword))
    if not size_1d and not size_3d:
        raise ValueError(tr('LUT 缺少 LUT_1D_SIZE 或 LUT_3D_SIZE。'))
    # DOMAIN_MIN / MAX describe the only table of an Adobe .cube; with a Resolve shaper, the shaper.
    if domain is not None:
        if size_1d and domain_1d is None:
            domain_1d = domain
        elif domain_3d is None:
            domain_3d = domain
    expected = size_1d + size_3d ** 3
    if len(rows) != expected:
        raise ValueError(tr('LUT 数据行数 {count} 与尺寸不符（应为 {expected}）。', count=len(rows), expected=expected))
    try:
        values = np.array(' '.join(rows).split(), dtype=np.float64)
    except ValueError:
        raise ValueError(tr('LUT 第 {line} 行格式无效。', line=first_row))
    if values.size != expected * 3:
        raise ValueError(tr('LUT 每行需要三个数值。'))
    if not np.all(np.isfinite(values)) or np.abs(values).max(initial=0) > MAX_VALUE:
        raise ValueError(tr('LUT 含有无效数值。'))
    values = values.reshape(-1, 3).astype(np.float32)
    shaper, cube = (values[:size_1d], values[size_1d:]) if size_1d else (None, values)
    domain_1d = domain_1d or IDENTITY_DOMAIN
    domain_3d = domain_3d or IDENTITY_DOMAIN
    for low, high in (domain_1d, domain_3d):
        if any(not h > l for l, h in zip(low, high)):
            raise ValueError(tr('LUT 的 DOMAIN_MAX 必须大于 DOMAIN_MIN。'))
    if size_3d > MAX_3D:
        cube = resample(cube, size_3d, MAX_3D)
        notes.append(tr('{size}³ LUT 已重采样为 {limit}³', size=size_3d, limit=MAX_3D))
        size_3d = MAX_3D
    setting = dict(name=(title or str(name))[:80], amount=100., size_1d=size_1d if shaper is not None else 0,
                   size_3d=size_3d if cube is not None and size_3d else 0,
                   domain_1d=[list(map(float, p)) for p in domain_1d], domain_3d=[list(map(float, p)) for p in domain_3d],
                   data=encode(shaper, cube if size_3d else None))
    return validate(setting), notes


def load_cube(path):
    path = Path(path)
    if path.stat().st_size > MAX_FILE:
        raise ValueError(tr('LUT 文件过大。'))
    return parse_cube(path.read_bytes(), path.stem)


def to_cube(setting):
    """A LUT setting as .cube text (export, tests)."""
    t = table(setting)
    lines = [f'TITLE "{setting["name"]}"']
    if t.shaper is not None:
        lines.append(f'LUT_1D_SIZE {len(t.shaper)}')
    if t.cube is not None:
        lines.append(f'LUT_3D_SIZE {t.size}')
    if t.shaper is not None and t.cube is not None:
        lines.append('LUT_1D_INPUT_RANGE {} {}'.format(*t.domain_1d[:, 0]))
        lines.append('LUT_3D_INPUT_RANGE {} {}'.format(*t.domain_3d[:, 0]))
    else:
        domain = t.domain_1d if t.cube is None else t.domain_3d
        lines.append('DOMAIN_MIN {} {} {}'.format(*domain[0]))
        lines.append('DOMAIN_MAX {} {} {}'.format(*domain[1]))
    for values in (t.shaper, t.cube):
        if values is not None:
            lines += ['{:.7g} {:.7g} {:.7g}'.format(*row) for row in values]
    return '\n'.join(lines) + '\n'


# --------------------------------------------------------------------------- recipe setting

def encode(shaper, cube):
    parts = [np.asarray(v, '<f4').tobytes() for v in (shaper, cube) if v is not None]
    return base64.b64encode(zlib.compress(b''.join(parts), 6)).decode('ascii')


@lru_cache(maxsize=8)
def _decode(data, size_1d, size_3d):
    try:
        raw = zlib.decompress(base64.b64decode(data.encode('ascii'), validate=True))
    except (ValueError, zlib.error, UnicodeEncodeError):
        raise ValueError(tr('无效的 LUT 数据。'))
    count = size_1d + size_3d ** 3
    if len(raw) != count * 12:
        raise ValueError(tr('无效的 LUT 数据。'))
    values = np.frombuffer(raw, '<f4').reshape(count, 3).astype(np.float32)
    if not np.all(np.isfinite(values)) or np.abs(values).max(initial=0) > MAX_VALUE:
        raise ValueError(tr('无效的 LUT 数据。'))
    values.setflags(write=False)
    return (values[:size_1d] if size_1d else None), (values[size_1d:] if size_3d else None)


def _domain(value):
    if not isinstance(value, list) or len(value) != 2 or any(not isinstance(p, list) or len(p) != 3 for p in value):
        raise ValueError(tr('无效的 LUT 数据。'))
    result = []
    for p in value:
        row = []
        for v in p:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 1e4:
                raise ValueError(tr('无效的 LUT 数据。'))
            row.append(float(v))
        result.append(row)
    if any(not h > l for l, h in zip(*result)):
        raise ValueError(tr('无效的 LUT 数据。'))
    return result


def validate(value):
    """A clean LUT setting, or None for no LUT.  Raises ValueError."""
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get('data'), str):
        raise ValueError(tr('无效的 LUT 数据。'))
    def integer(key, low, high):
        v = value.get(key, 0)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not float(v).is_integer():
            raise ValueError(tr('无效的 LUT 数据。'))
        v = int(v)
        if v and not low <= v <= high:
            raise ValueError(tr('无效的 LUT 数据。'))
        return v
    size_1d, size_3d = integer('size_1d', 2, MAX_1D), integer('size_3d', 2, MAX_3D)
    if not size_1d and not size_3d:
        raise ValueError(tr('无效的 LUT 数据。'))
    amount = value.get('amount', 100)
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not 0 <= amount <= 100:
        raise ValueError(tr('LUT 强度需介于 0–100。'))
    _decode(value['data'], size_1d, size_3d)
    return dict(name=str(value.get('name', 'LUT'))[:80], amount=float(amount), size_1d=size_1d, size_3d=size_3d,
                domain_1d=_domain(value.get('domain_1d', IDENTITY_DOMAIN)),
                domain_3d=_domain(value.get('domain_3d', IDENTITY_DOMAIN)), data=value['data'])


def table(setting):
    shaper, cube = _decode(setting['data'], setting['size_1d'], setting['size_3d'])
    return Table(shaper, cube, setting['size_3d'], np.asarray(setting['domain_1d'], np.float32),
                 np.asarray(setting['domain_3d'], np.float32))


@lru_cache(maxsize=32)
def _digest(data):
    return hashlib.blake2b(data.encode('ascii'), digest_size=12).hexdigest()


def digest(setting):
    """Short content key of a LUT's tables (render cache keys, project storage)."""
    return _digest(setting['data']) if setting else ''


def key(setting):
    """What a render depends on: tables, domains and strength (None without a LUT)."""
    if not active(setting):
        return None
    return (digest(setting), setting['size_1d'], setting['size_3d'], setting['domain_1d'], setting['domain_3d'],
            setting['amount'])


def active(setting):
    return bool(setting) and setting.get('amount', 0) > 0


def summary(setting):
    """What the interface and the command layer show of a LUT (never its tables)."""
    if not setting:
        return None
    kind = f'{setting["size_3d"]}³' if setting['size_3d'] else f'1D {setting["size_1d"]}'
    return dict(name=setting['name'], amount=round(float(setting['amount']), 1), size=kind)


def with_amount(setting, amount):
    if not setting:
        return None
    return dict(setting, amount=float(min(100., max(0., amount))))


# --------------------------------------------------------------------------- evaluation

def _shaper(x, values, domain):
    size = len(values)
    out = np.empty_like(x)
    for c in range(3):
        position = (x[..., c] - domain[0, c]) * np.float32((size - 1) / (domain[1, c] - domain[0, c]))
        position = np.clip(position, 0, size - 1)
        start = np.minimum(np.floor(position), size - 2)
        index = start.astype(np.int32)
        fraction = position - start
        left = values[index, c]
        out[..., c] = left + (values[index + 1, c] - left) * fraction
    return out


def _tetrahedral(x, cube, size, domain):
    """Tetrahedral interpolation of an (n³, 3) table (red fastest) at (…, 3) points."""
    scale = (np.float32(size - 1) / (domain[1] - domain[0])).astype(np.float32)
    position = np.clip((x - domain[0]) * scale, 0, size - 1)
    start = np.minimum(np.floor(position), size - 2)
    index = start.astype(np.int32)
    f = position - start
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    n, nn = size, size * size
    base = index[..., 0] + index[..., 1] * n + index[..., 2] * nn
    # Axis order by fraction, ties broken red before green before blue, so the order is total.
    rg, gb, rb = r >= g, g >= b, r >= b
    first = np.where(rg & rb, 1, np.where(gb, n, nn))           # stride of the largest fraction
    last = np.where(rb & gb, nn, np.where(rg, n, 1))            # stride of the smallest fraction
    high = np.maximum(np.maximum(r, g), b)
    low = np.minimum(np.minimum(r, g), b)
    middle = np.maximum(np.minimum(r, g), np.minimum(np.maximum(r, g), b))
    corner = 1 + n + nn
    c0 = cube[base]
    c1 = cube[base + first]
    c2 = cube[base + (corner - last)]
    c3 = cube[base + corner]
    return (c0 * (1 - high)[..., None] + c1 * (high - middle)[..., None]
            + c2 * (middle - low)[..., None] + c3 * low[..., None])


def evaluate(x, setting):
    """The LUT's output at display values ``x`` (…, 3), full strength, unclipped."""
    t = table(setting)
    y = np.asarray(x, np.float32)
    if t.shaper is not None:
        y = _shaper(y, t.shaper, t.domain_1d)
    if t.cube is not None:
        y = _tetrahedral(y, t.cube, t.size, t.domain_3d)
    return y.astype(np.float32, copy=False)


def resample(cube, size, target):
    """An (n³, 3) table evaluated at a target³ grid (tetrahedral)."""
    axis = np.linspace(0, 1, target, dtype=np.float32)
    b, g, r = np.meshgrid(axis, axis, axis, indexing='ij')
    points = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    return _tetrahedral(points, np.asarray(cube, np.float32), size, np.asarray(IDENTITY_DOMAIN, np.float32))


_pool = None
_pool_lock = threading.Lock()


def _executor():
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=max(1, performance.THREADS), thread_name_prefix='lut')
        return _pool


def _blend(x, setting, t):
    y = evaluate(x, setting)
    if t < 1:
        y = x + (y - x) * np.float32(t)
    return y


def _display(x, setting, t):
    return np.clip(_blend(x, setting, t), 0, 1)


def _wide(x, setting, t, space):
    """Display values of a wide output space: the LUT reads the sRGB encoding of each colour; what
    lies outside sRGB is kept as an offset from the colour clipped into it."""
    from . import color
    from .tone import _signed_linear, _signed_srgb
    srgb = _signed_srgb(color.to_linear_srgb(x, space))
    inside = np.clip(srgb, 0, 1)
    srgb = srgb + (evaluate(inside, setting) - inside) * np.float32(t)
    return color.encode(color.gamut_map(_signed_linear(srgb), space), space)


def apply(x, setting, space=None):
    """The LUT stage: ``x`` (display values 0–1 of ``space``, sRGB by default) blended with the
    LUT's output at the setting's strength.  Returns ``x`` itself without an active LUT."""
    if not active(setting):
        return x
    from . import color
    t = min(1., setting['amount'] / 100)
    work = (lambda block: _wide(block, setting, t, space)) if color.wide(space) else \
        (lambda block: _display(block, setting, t))
    flat = np.asarray(x, np.float32)
    shape = flat.shape
    pixels = int(np.prod(shape[:-1]))
    if pixels <= CHUNK:
        return np.ascontiguousarray(work(flat), dtype=np.float32)
    from . import large_image
    out = large_image.allocate(shape)
    rows = max(1, CHUNK // max(1, shape[1]))
    def run(y):
        out[y:y + rows] = work(np.ascontiguousarray(flat[y:y + rows]))
    list(_executor().map(run, range(0, shape[0], rows)))
    return out


# --------------------------------------------------------------------------- files

def pack(recipes):
    """Recipes (``edits`` dicts) with their LUT tables moved into one ``{digest: data}`` table,
    as projects and albums store them; the recipes are copied, never changed."""
    tables, result = {}, []
    for edits in recipes:
        setting = edits.get('lut') if isinstance(edits, dict) else None
        if setting and 'data' in setting:
            ref = digest(setting)
            tables[ref] = setting['data']
            edits = dict(edits, lut={k: v for k, v in setting.items() if k != 'data'} | dict(ref=ref))
        result.append(edits)
    return result, tables


def unpack(edits, tables):
    """A stored recipe with its LUT table taken back from ``tables`` (see ``pack``)."""
    setting = edits.get('lut') if isinstance(edits, dict) else None
    if not setting or 'ref' not in setting:
        return edits
    data = (tables or {}).get(setting['ref'])
    if not isinstance(data, str) or _digest(data) != setting['ref']:
        raise ValueError(tr('工程中的 LUT 数据缺失或已损坏。'))
    return dict(edits, lut={k: v for k, v in setting.items() if k != 'ref'} | dict(data=data))


_NAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def safe_name(text):
    return _NAME.sub('_', str(text)).strip(' .')[:80] or 'LUT'
