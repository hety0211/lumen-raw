"""Pointwise development graphs for GPU execution providers (DirectML / CUDA).

Every graph takes and returns channel-last float32 tiles (1, H, W, 3), so the
NumPy image can be uploaded without a transpose.  Three graphs are built:

* ``tonal``: white-balance gains, exposure and the four tonal zones (1.2.2).
* ``color``: saturation / vibrance, 8-color HSL, RGB + channel curves,
  monochrome and three-way grading.
* ``fused``: ``tonal`` followed by ``color`` in one upload / download, used
  when no spatial detail tool (clarity, dehaze, ...) sits between them.

Process version 2 (1.6.0) adds ``tonal2``, ``color2`` and ``fused2``: the
scene-referred tone stage of ``tone`` (local tone mapping from per-pixel
guided-filter coefficients, develop curve as tone map, OkLab gamut mapping) and
the OkLCh colour stage of ``color``.

The CPU path in ``engine`` remains the reference implementation; the parity
tests compare both across non-default parameters.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .onnx_graph import Graph, INT64
from .model import COLORS
from . import color, tone

LUMA = np.array([.2126, .7152, .0722], np.float32).reshape(1, 1, 1, 3)
HUE_SAMPLES = 361          # 1 degree grid; every HSL center is an integer degree.
CURVE_SAMPLES = 4097       # identical axis to engine.apply_curves smooth mode.
EPS = float(np.finfo(np.float32).eps)
SHAPE = [1, 'height', 'width', 3]


def _scalar(g, value):
    return g.const(np.float32(value))


def _clip(g, x, low=0., high=1.):
    return g.op('Clip', x, _scalar(g, low), _scalar(g, high))


def _luma(g, x):
    return g.op('ReduceSum', g.op('Mul', x, LUMA), g.const([3], np.int64), keepdims=1)


def _channel(g, x, index):
    i64 = lambda v: g.const([v], np.int64)
    return g.op('Slice', x, i64(index), i64(index + 1), i64(3))


def _lookup(g, x, table, samples, low, high):
    """Linear interpolation into a 1-D table, matching numpy.interp on a grid."""
    position = g.op('Mul', _clip(g, x, low, high), _scalar(g, (samples - 1) / (high - low)))
    if low:
        position = g.op('Sub', position, _scalar(g, low * (samples - 1) / (high - low)))
    index = _clip(g, g.op('Floor', position), 0, samples - 2)
    fraction = g.op('Sub', position, index)
    index = g.op('Cast', index, to=INT64)
    left = g.op('Gather', table, index, axis=0)
    right = g.op('Gather', table, g.op('Add', index, g.const(1, np.int64)), axis=0)
    return g.op('Add', left, g.op('Mul', g.op('Sub', right, left), fraction))


def _tonal(g, x):
    gains = g.input('gains', [1, 1, 1, 3])
    exposure = g.input('exposure', [1])
    zones = {name: g.input(name, [1]) for name in ('shadows', 'highlights', 'blacks', 'whites')}
    contrast = g.input('contrast', [1])
    exposed = g.op('Mul', g.op('Mul', x, gains), exposure)
    p = g.op('Pow', _clip(g, _luma(g, exposed)), _scalar(g, .45))
    inv = g.op('Sub', _scalar(g, 1), p)
    middle = g.op('Add', g.op('Mul', zones['shadows'], g.op('Pow', inv, _scalar(g, 2))),
                  g.op('Mul', zones['highlights'], g.op('Pow', p, _scalar(g, 3))))
    ends = g.op('Add', g.op('Mul', zones['blacks'], g.op('Pow', inv, _scalar(g, 6))),
                g.op('Mul', zones['whites'], g.op('Pow', p, _scalar(g, 6))))
    stops = g.op('Add', g.op('Div', middle, _scalar(g, 65)), g.op('Div', ends, _scalar(g, 85)))
    linear = g.op('Max', g.op('Mul', exposed, g.op('Pow', _scalar(g, 2), stops)), _scalar(g, 0))
    srgb = g.op('Where', g.op('LessOrEqual', linear, _scalar(g, .0031308)),
                g.op('Mul', linear, _scalar(g, 12.92)),
                g.op('Sub', g.op('Mul', g.op('Pow', linear, _scalar(g, 1 / 2.4)), _scalar(g, 1.055)),
                     _scalar(g, .055)))
    centered = g.op('Sub', srgb, _scalar(g, .5))
    return _clip(g, g.op('Add', g.op('Mul', centered, contrast), _scalar(g, .5)))


def _hsl(g, x):
    tables = [g.input(name, [HUE_SAMPLES]) for name in ('hue_lut', 'sat_lut', 'val_lut')]
    r, gr, b = (_channel(g, x, i) for i in range(3))
    v = g.op('Max', r, gr, b)
    diff = g.op('Sub', v, g.op('Min', r, gr, b))
    s = g.op('Div', diff, g.op('Add', g.op('Abs', v), _scalar(g, EPS)))
    k = g.op('Div', _scalar(g, 60), g.op('Add', diff, _scalar(g, EPS)))
    h_r = g.op('Mul', g.op('Sub', gr, b), k)
    h_g = g.op('Add', g.op('Mul', g.op('Sub', b, r), k), _scalar(g, 120))
    h_b = g.op('Add', g.op('Mul', g.op('Sub', r, gr), k), _scalar(g, 240))
    h = g.op('Where', g.op('Equal', v, r), h_r, g.op('Where', g.op('Equal', v, gr), h_g, h_b))
    h = g.op('Where', g.op('Less', h, _scalar(g, 0)), g.op('Add', h, _scalar(g, 360)), h)
    dh, ds, dv = (_lookup(g, h, t, HUE_SAMPLES, 0., 360.) for t in tables)
    hue = g.op('Add', h, dh)
    hue = g.op('Sub', hue, g.op('Mul', g.op('Floor', g.op('Div', hue, _scalar(g, 360))), _scalar(g, 360)))
    s = _clip(g, g.op('Mul', s, g.op('Add', ds, _scalar(g, 1))))
    v = _clip(g, g.op('Mul', v, g.op('Pow', _scalar(g, 2), dv)))
    sector = g.op('Div', hue, _scalar(g, 60))
    vs = g.op('Mul', v, s)
    channels = []
    for n in (5., 3., 1.):
        t = g.op('Add', sector, _scalar(g, n))
        t = g.op('Sub', t, g.op('Mul', g.op('Floor', g.op('Div', t, _scalar(g, 6))), _scalar(g, 6)))
        weight = _clip(g, g.op('Min', t, g.op('Sub', _scalar(g, 4), t)))
        channels.append(g.op('Sub', v, g.op('Mul', vs, weight)))
    return g.op('Concat', *channels, axis=3)


def _color(g, x):
    saturation, vibrance = g.input('saturation', [1]), g.input('vibrance', [1])
    lum = _luma(g, x)
    spread = g.op('Sub', g.op('ReduceMax', x, axes=[3], keepdims=1), g.op('ReduceMin', x, axes=[3], keepdims=1))
    scale = g.op('Add', saturation, g.op('Mul', vibrance, g.op('Sub', _scalar(g, 1), spread)))
    x = _clip(g, g.op('Add', lum, g.op('Mul', g.op('Sub', x, lum), scale)))
    x = _hsl(g, x)
    curve = g.input('curve_rgb', [CURVE_SAMPLES])
    x = _lookup(g, x, curve, CURVE_SAMPLES, 0., 1.)
    tables = [g.input(f'curve_{c}', [CURVE_SAMPLES]) for c in 'rgb']
    x = g.op('Concat', *[_lookup(g, _channel(g, x, i), tables[i], CURVE_SAMPLES, 0., 1.) for i in range(3)], axis=3)
    mono = g.input('mono', [1])
    x = g.op('Add', x, g.op('Mul', mono, g.op('Sub', _luma(g, x), x)))
    balance = g.input('balance', [1])
    zones = [g.input(f'grade_{z}', [1, 1, 1, 3]) for z in ('shadows', 'midtones', 'highlights')]
    lum = _luma(g, x)
    tone = _clip(g, g.op('Add', lum, balance))
    inv = g.op('Sub', _scalar(g, 1), tone)
    weights = [g.op('Mul', inv, inv), g.op('Mul', g.op('Mul', tone, inv), _scalar(g, 2)), g.op('Mul', tone, tone)]
    protection = g.op('Add', _scalar(g, .2), g.op('Mul', _scalar(g, .8),
                      g.op('Sin', g.op('Mul', _clip(g, lum), _scalar(g, np.pi)))))
    tint = g.op('Add', g.op('Add', g.op('Mul', weights[0], zones[0]), g.op('Mul', weights[1], zones[1])),
                g.op('Mul', weights[2], zones[2]))
    return _clip(g, g.op('Add', x, g.op('Mul', protection, tint)))


# --------------------------------------------------------------------------- process 2

TONE_SAMPLES = tone.TABLE
LOCAL_SHAPE = [1, 'local_height', 'local_width', 2]


def _matrix(g, x, matrix):
    return g.op('MatMul', x, np.ascontiguousarray(np.asarray(matrix, np.float32).T))


def _cbrt(g, x):
    return g.op('Mul', g.op('Sign', x), g.op('Pow', g.op('Abs', x), _scalar(g, 1 / 3)))


def _to_oklab(g, linear):
    return _matrix(g, _cbrt(g, _matrix(g, linear, color.RGB_TO_LMS)), color.LMS_TO_LAB)


def _from_oklab(g, lab):
    lms = _matrix(g, lab, color.LAB_TO_LMS)
    return _matrix(g, g.op('Mul', g.op('Mul', lms, lms), lms), color.LMS_TO_RGB)


def _srgb_to_linear(g, x):
    return g.op('Where', g.op('LessOrEqual', x, _scalar(g, .04045)), g.op('Div', x, _scalar(g, 12.92)),
                g.op('Pow', g.op('Max', g.op('Div', g.op('Add', x, _scalar(g, .055)), _scalar(g, 1.055)), _scalar(g, 0)),
                     _scalar(g, 2.4)))


def _linear_to_srgb(g, x):
    x = g.op('Max', x, _scalar(g, 0))
    return g.op('Where', g.op('LessOrEqual', x, _scalar(g, .0031308)), g.op('Mul', x, _scalar(g, 12.92)),
                g.op('Sub', g.op('Mul', g.op('Pow', x, _scalar(g, 1 / 2.4)), _scalar(g, 1.055)), _scalar(g, .055)))


def _slice(g, x, start, stop):
    i64 = lambda v: g.const([v], np.int64)
    return g.op('Slice', x, i64(start), i64(stop), i64(3))


def _excess(g, rgb):
    """How far a colour lies outside the sRGB cube (≤ 0 inside), as color._excess."""
    over = g.op('Sub', g.op('ReduceMax', rgb, axes=[3], keepdims=1), _scalar(g, 1))
    under = g.op('Neg', g.op('ReduceMin', rgb, axes=[3], keepdims=1))
    return g.op('Max', over, under)


def _inside(g, rgb):
    low = g.op('GreaterOrEqual', g.op('ReduceMin', rgb, axes=[3], keepdims=1), _scalar(g, -color.GAMUT_EPS))
    high = g.op('LessOrEqual', g.op('ReduceMax', rgb, axes=[3], keepdims=1), _scalar(g, 1 + color.GAMUT_EPS))
    return g.op('And', low, high)


def _gamut(g, linear):
    """color.gamut_map: bisect the position of out-of-gamut colours on the line towards their
    grey anchor (constant hue), then interpolate between the bracket ends (one secant step)."""
    lab = _to_oklab(g, linear)
    lightness = _slice(g, lab, 0, 1)
    chroma = _slice(g, lab, 1, 3)
    size = g.op('Sqrt', g.op('ReduceSum', g.op('Mul', chroma, chroma), g.const([3], np.int64), keepdims=1))
    offset = g.op('Sub', lightness, _scalar(g, .5))
    distance = g.op('Abs', offset)
    e1 = g.op('Add', g.op('Add', _scalar(g, .5), distance), g.op('Mul', size, _scalar(g, color.GAMUT_ALPHA)))
    root = g.op('Sqrt', g.op('Max', g.op('Sub', g.op('Mul', e1, e1), g.op('Mul', distance, _scalar(g, 2))), _scalar(g, 0)))
    grey = g.op('Mul', _scalar(g, .5), g.op('Add', _scalar(g, 1), g.op('Mul', g.op('Sign', offset), g.op('Sub', e1, root))))
    span = g.op('Sub', lightness, grey)
    at = lambda t: _from_oklab(g, g.op('Concat', g.op('Add', grey, g.op('Mul', span, t)), g.op('Mul', chroma, t), axis=3))
    low = g.op('Mul', lightness, _scalar(g, 0))
    high = g.op('Add', low, _scalar(g, 1))
    excess_low, excess_high = _excess(g, at(low)), _excess(g, at(high))
    for _ in range(color.GAMUT_STEPS):
        middle = g.op('Mul', g.op('Add', low, high), _scalar(g, .5))
        excess = _excess(g, at(middle))
        fits = g.op('LessOrEqual', excess, _scalar(g, color.GAMUT_EPS))
        low, excess_low = g.op('Where', fits, middle, low), g.op('Where', fits, excess, excess_low)
        high, excess_high = g.op('Where', fits, high, middle), g.op('Where', fits, excess_high, excess)
    width = g.op('Max', g.op('Sub', excess_high, excess_low), _scalar(g, 1e-12))
    step = _clip(g, g.op('Div', g.op('Neg', excess_low), width))
    mapped = at(g.op('Add', low, g.op('Mul', g.op('Sub', high, low), step)))
    return _clip(g, g.op('Where', _inside(g, linear), linear, mapped))


def _degrees(g, b, a):
    """atan2(b, a) in degrees, 0…360 (ONNX has no Atan2)."""
    tiny = 1e-12
    safe = g.op('Where', g.op('GreaterOrEqual', a, _scalar(g, 0)), g.op('Max', a, _scalar(g, tiny)),
                g.op('Min', a, _scalar(g, -tiny)))
    angle = g.op('Atan', g.op('Div', b, safe))
    turn = g.op('Where', g.op('GreaterOrEqual', b, _scalar(g, 0)), _scalar(g, np.pi), _scalar(g, -np.pi))
    angle = g.op('Add', angle, g.op('Where', g.op('Less', safe, _scalar(g, 0)), turn, _scalar(g, 0)))
    degrees = g.op('Mul', angle, _scalar(g, 180 / np.pi))
    return g.op('Sub', degrees, g.op('Mul', g.op('Floor', g.op('Div', degrees, _scalar(g, 360))), _scalar(g, 360)))


def _tonal2(g, x):
    gains = g.input('gains', [1, 1, 1, 3])
    zones = {name: g.input(name, [1]) for name in ('shadows', 'highlights', 'blacks', 'whites')}
    contrast, contrast_on = g.input('contrast', [1]), g.input('contrast_on', [1])
    table = g.input('tone_lut', [TONE_SAMPLES])
    local = g.input('local_ab', LOCAL_SHAPE)
    lookup = lambda v: _lookup(g, v, table, TONE_SAMPLES, 0., tone.HEADROOM)
    x = g.op('Mul', x, gains)
    y = _luma(g, x)
    log = g.op('Div', g.op('Log', g.op('Max', y, _scalar(g, tone.FLOOR))), _scalar(g, np.log(2)))
    base = g.op('Pow', _scalar(g, 2), g.op('Add', g.op('Mul', _slice(g, local, 0, 1), log), _slice(g, local, 1, 2)))
    p = g.op('Pow', _clip(g, lookup(base)), _scalar(g, .45))
    inv = g.op('Sub', _scalar(g, 1), p)
    middle = g.op('Add', g.op('Mul', zones['shadows'], g.op('Mul', inv, inv)),
                  g.op('Mul', zones['highlights'], g.op('Pow', p, _scalar(g, 3))))
    q = g.op('Pow', _clip(g, lookup(y)), _scalar(g, .45))
    qinv = g.op('Sub', _scalar(g, 1), q)
    ends = g.op('Add', g.op('Mul', zones['blacks'], g.op('Pow', qinv, _scalar(g, 6))),
                g.op('Mul', zones['whites'], g.op('Pow', q, _scalar(g, 6))))
    stops = g.op('Add', g.op('Div', middle, _scalar(g, 65)), g.op('Div', ends, _scalar(g, 85)))
    x = g.op('Mul', x, g.op('Pow', _scalar(g, 2), stops))
    y = _luma(g, x)
    x = g.op('Mul', x, g.op('Div', lookup(y), g.op('Max', y, _scalar(g, 1e-7))))
    encoded = g.op('Mul', g.op('Sign', x), _linear_to_srgb(g, g.op('Abs', x)))
    graded = g.op('Add', g.op('Mul', g.op('Sub', encoded, _scalar(g, .5)), contrast), _scalar(g, .5))
    contrasted = g.op('Mul', g.op('Sign', graded), _srgb_to_linear(g, g.op('Abs', graded)))
    x = g.op('Where', g.op('Greater', contrast_on, _scalar(g, 0)), contrasted, x)
    return _clip(g, _linear_to_srgb(g, _gamut(g, x)))


def _color2(g, x, lab_stage):
    """Process 2 colour: curves, then (``lab_stage``) one OkLCh pass and gamut mapping."""
    curve = g.input('curve_rgb', [CURVE_SAMPLES])
    x = _lookup(g, x, curve, CURVE_SAMPLES, 0., 1.)
    curves = [g.input(f'curve_{c}', [CURVE_SAMPLES]) for c in 'rgb']
    x = g.op('Concat', *[_lookup(g, _channel(g, x, i), curves[i], CURVE_SAMPLES, 0., 1.) for i in range(3)], axis=3)
    if not lab_stage:
        return _clip(g, x)
    saturation, vibrance = g.input('saturation', [1]), g.input('vibrance', [1])
    keep = g.input('chroma_keep', [1])            # 0 for monochrome
    tables = {name: g.input(name, [HUE_SAMPLES]) for name in ('hue_shift', 'chroma_lut', 'light_lut', 'vibrance_lut')}
    hue_lookup = lambda name, v: _lookup(g, v, tables[name], HUE_SAMPLES, 0., 360.)
    lab = _to_oklab(g, _srgb_to_linear(g, x))
    lightness, a, b = (_slice(g, lab, i, i + 1) for i in range(3))
    chroma = g.op('Sqrt', g.op('Add', g.op('Mul', a, a), g.op('Mul', b, b)))
    hue = _degrees(g, b, a)
    fading = g.op('Sub', _scalar(g, 1), g.op('Min', g.op('Div', chroma, _scalar(g, color.CHROMA_FULL)), _scalar(g, 1)))
    scale = g.op('Add', saturation, g.op('Mul', g.op('Mul', vibrance, fading), hue_lookup('vibrance_lut', hue)))
    weight = g.op('Min', g.op('Div', chroma, _scalar(g, color.CHROMA_HUE)), _scalar(g, 1))
    scale = g.op('Mul', scale, g.op('Add', hue_lookup('chroma_lut', hue), _scalar(g, 1)))
    lightness = g.op('Mul', lightness, g.op('Pow', _scalar(g, 2), g.op('Mul', hue_lookup('light_lut', hue), weight)))
    hue = g.op('Add', hue, hue_lookup('hue_shift', hue))
    chroma = g.op('Mul', g.op('Mul', chroma, g.op('Max', scale, _scalar(g, 0))), keep)
    radians = g.op('Mul', hue, _scalar(g, np.pi / 180))
    ab = g.op('Concat', g.op('Mul', chroma, g.op('Cos', radians)), g.op('Mul', chroma, g.op('Sin', radians)), axis=3)
    # Grading: OkLab (a, b) tints by tonal zone (zero vectors when off).
    balance = g.input('balance', [1])
    vectors = [g.input(f'grade_{z}', [1, 1, 1, 2]) for z in ('shadows', 'midtones', 'highlights')]
    level = _clip(g, g.op('Add', lightness, balance))
    inv = g.op('Sub', _scalar(g, 1), level)
    weights = [g.op('Mul', inv, inv), g.op('Mul', g.op('Mul', level, inv), _scalar(g, 2)), g.op('Mul', level, level)]
    protection = g.op('Add', _scalar(g, .2), g.op('Mul', _scalar(g, .8),
                      g.op('Sin', g.op('Mul', _clip(g, lightness), _scalar(g, np.pi)))))
    tint = g.op('Mul', protection, g.op('Add', g.op('Add', g.op('Mul', weights[0], vectors[0]), g.op('Mul', weights[1], vectors[1])),
                                        g.op('Mul', weights[2], vectors[2])))
    lab = g.op('Concat', lightness, g.op('Add', ab, tint), axis=3)
    return _clip(g, _linear_to_srgb(g, _gamut(g, _from_oklab(g, lab))))


#: ``color2`` / ``fused2`` skip the OkLCh pass (neutral saturation, mixer, monochrome
#: and grading); ``color2lab`` / ``fused2lab`` include it.
KINDS = ('tonal', 'color', 'fused', 'tonal2', 'color2', 'fused2', 'color2lab', 'fused2lab')


@lru_cache(maxsize=None)
def model(kind):
    if kind not in KINDS:
        raise ValueError(kind)
    g = Graph(f'LUMEN RAW {kind} pipeline')
    x = g.input('image', SHAPE)
    if kind in ('tonal', 'fused'):
        x = _tonal(g, x)
    if kind in ('color', 'fused'):
        x = _color(g, x)
    if kind.startswith(('tonal2', 'fused2')):
        x = _tonal2(g, x)
    if kind.startswith(('color2', 'fused2')):
        x = _color2(g, x, kind.endswith('lab'))
    g.output(x, 'output', SHAPE)
    return g.serialize()


def tonal_inputs(a):
    temp, tint = a['temperature'] / 100, a['tint'] / 100
    gains = np.array([2 ** (.4 * temp + .15 * tint), 2 ** (-.15 * tint),
                      2 ** (-.4 * temp + .15 * tint)], np.float32).reshape(1, 1, 1, 3)
    scalar = lambda number: np.array([number], np.float32)
    return dict(gains=gains, exposure=scalar(2 ** a['exposure']),
                shadows=scalar(a['shadows']), highlights=scalar(a['highlights']),
                blacks=scalar(a['blacks']), whites=scalar(a['whites']),
                contrast=scalar(1 + a['contrast'] / 125))


def color_inputs(edits):
    import colorsys
    from . import curves as tone_curves
    a = edits['adjustments']
    scalar = lambda number: np.array([number], np.float32)
    hue_axis = np.arange(HUE_SAMPLES, dtype=np.float64)
    centers = [c[1] for c in COLORS] + [360]
    controls = np.asarray(list(edits['hsl']) + [edits['hsl'][0]], dtype=np.float64)
    inputs = dict(saturation=scalar(1 + a['saturation'] / 100), vibrance=scalar(a['vibrance'] / 100),
                  hue_lut=np.interp(hue_axis, centers, controls[:, 0] * .45).astype(np.float32),
                  sat_lut=np.interp(hue_axis, centers, controls[:, 1] / 100).astype(np.float32),
                  val_lut=np.interp(hue_axis, centers, controls[:, 2] / 100).astype(np.float32))
    axis = np.linspace(0, 1, CURVE_SAMPLES)
    mode = edits.get('curve_mode', 'linear')
    for channel, name in (('RGB', 'curve_rgb'), ('R', 'curve_r'), ('G', 'curve_g'), ('B', 'curve_b')):
        points = edits['curves'][channel]
        if channel == 'RGB':
            values = tone_curves.rgb_table(edits, axis)  # with the exposure curve's fine tone curve (1.4.1)
        else:
            values = axis if points == [[0., 0.], [1., 1.]] else tone_curves.evaluate(points, axis, mode)
        inputs[name] = np.asarray(values, np.float32)
    inputs['mono'] = scalar(1. if edits.get('monochrome', False) else 0.)
    grading = edits.get('grading', {})
    inputs['balance'] = scalar(grading.get('balance', 0) / 300)
    for zone in ('shadows', 'midtones', 'highlights'):
        hue, strength = grading.get(zone, [0, 0])
        color = np.asarray(colorsys.hsv_to_rgb((hue % 360) / 360, 1, 1), np.float32)
        color -= np.dot(color, [.2126, .7152, .0722])
        inputs[f'grade_{zone}'] = (color * np.float32(strength / 350)).astype(np.float32).reshape(1, 1, 1, 3)
    return inputs


def tonal2_inputs(a, spec, shape):
    """Inputs of ``tonal2`` for an image of ``shape`` (the local coefficients are per pixel)."""
    scalar = lambda number: np.array([number], np.float32)
    gains = (tone.white_balance_gains(a) * np.float32(2 ** a['exposure'])).reshape(1, 1, 1, 3)
    if spec.context is not None and tone.needs_context(a):
        local = tone.maps(spec, shape, a['exposure'])[None]
    else:
        local = np.array([1., 0.], np.float32).reshape(1, 1, 1, 2)
    return dict(gains=gains.astype(np.float32), shadows=scalar(a['shadows']), highlights=scalar(a['highlights']),
                blacks=scalar(a['blacks']), whites=scalar(a['whites']),
                contrast=scalar(1 + a['contrast'] / 125), contrast_on=scalar(1. if a['contrast'] else 0.),
                tone_lut=np.asarray(spec.table, np.float32), local_ab=np.ascontiguousarray(local, np.float32))


def lab_stage(edits):
    """True when process 2's colour stage needs its OkLCh pass."""
    a = edits['adjustments']
    return bool(a['saturation'] or a['vibrance'] or np.any(edits['hsl']) or edits.get('monochrome', False)
                or color.graded(edits.get('grading', {})))


def color2_inputs(edits):
    from . import curves as tone_curves
    a = edits['adjustments']
    scalar = lambda number: np.array([number], np.float32)
    axis = np.linspace(0, 1, CURVE_SAMPLES)
    mode = edits.get('curve_mode', 'linear')
    inputs = {}
    for channel, name in (('RGB', 'curve_rgb'), ('R', 'curve_r'), ('G', 'curve_g'), ('B', 'curve_b')):
        points = edits['curves'][channel]
        if channel == 'RGB':
            values = tone_curves.rgb_table(edits, axis)
        else:
            values = axis if points == [[0., 0.], [1., 1.]] else tone_curves.evaluate(points, axis, mode)
        inputs[name] = np.asarray(values, np.float32)
    if not lab_stage(edits):
        return inputs
    shift, chroma, light = color.mixer_tables(edits['hsl'])
    grading = edits.get('grading', {})
    inputs.update(saturation=scalar(1 + a['saturation'] / 100), vibrance=scalar(a['vibrance'] / 100),
                  chroma_keep=scalar(0. if edits.get('monochrome', False) else 1.),
                  hue_shift=shift, chroma_lut=chroma, light_lut=light, vibrance_lut=color.vibrance_weights(),
                  balance=scalar(grading.get('balance', 0) / 300))
    for zone, vector in zip(('shadows', 'midtones', 'highlights'), color.grade_vectors(grading)):
        inputs[f'grade_{zone}'] = vector.reshape(1, 1, 1, 2).astype(np.float32)
    return inputs
