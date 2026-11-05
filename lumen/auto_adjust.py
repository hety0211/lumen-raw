"""Automatic exposure and colour (2.3.0): the 自动 button, as ordinary slider values to edit.

The photo is developed small (``SAMPLE_EDGE`` pixels) through the real pipeline, so the result
follows the photo's process version and develop starting point, and the sliders are settled in
a few measured steps:

1. **White balance** (冷暖 / 色调).  The light's colour is estimated from *grey pixels* (after
   Yang, Gao & Li, "Efficient illuminant estimation for color constancy using grey pixels",
   CVPR 2015): where the local contrast of log R, log G and log B is the same, the surface is
   neutral and its colour is the light's.  With few such pixels the estimate falls back to
   shades of grey (Finlayson & Trezzi 2004, p = 6) and is applied more weakly, so a sunset
   is warmed less away than a grey street.
2. **Exposure** brings the developed median to a target that follows the scene's key
   (Reinhard's automatic key, 2002: where the log-average luminance lies between the darkest
   and brightest percent), so night stays dark; a very bright, unclipped scene such as snow keeps
   its brightness (``_keep_high_key``, for the fitted model too).
3. **Highlights / shadows** recover the ends when many pixels crowd them; **whites / blacks**
   then place the end points (99.7 % near white, 0.3 % near black) without clipping more.
4. **Contrast** from the spread of the midtones, **vibrance** from the colourfulness of the
   result (Hasler & Süsstrunk 2003).

``measure`` returns the numbers each step looks at.  When LUMEN RAW ships a fitted model
(``assets/auto-adjust-model.json``, made by ``tools/train_auto_adjust.py`` from photos edited by
hand), ``suggest`` uses it instead of the rules: ``describe`` measures the photo, one linear model
per slider predicts the sliders and the median the edit reaches, and the exposure that reaches
that median is solved through the pipeline.  The rules remain the fallback.
"""
from __future__ import annotations

import copy

import cv2
import numpy as np

#: Long edge of the copy the photo is measured on.
SAMPLE_EDGE = 320
#: Sliders ``suggest`` sets (the others keep their values).
KEYS = ('temperature', 'tint', 'exposure', 'contrast', 'highlights', 'shadows', 'whites', 'blacks',
        'vibrance', 'saturation')
#: Display median for a scene of average key.
MID = .44


def _luma(rgb):
    return rgb[..., 0] * .2126 + rgb[..., 1] * .7152 + rgb[..., 2] * .0722


def illuminant(linear):
    """``(rgb, reliability)``: the light's colour (G = 1) in a white-balanced linear frame, and how
    much of the frame supported it (0–1)."""
    x = np.asarray(linear, np.float32)
    x = np.maximum(x, 0)
    luma = _luma(x)
    white = np.percentile(luma, 99.5) if luma.size else 1
    valid = (x.max(axis=2) < max(white, 1e-4) * .98) & (luma > max(white, 1e-4) * .02)
    if valid.sum() < 100:
        return np.ones(3, np.float32), 0.
    logs = np.log(x + 1e-4)
    contrast = []
    for c in range(3):
        blurred = cv2.GaussianBlur(logs[..., c], (0, 0), 1.)
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        contrast.append(np.sqrt(gx * gx + gy * gy))
    contrast = np.stack(contrast, axis=2)
    mean = contrast.mean(axis=2)
    spread = contrast.std(axis=2)
    textured = valid & (mean > .05)
    grey = np.zeros_like(valid)
    if textured.sum() > 50:
        index = spread[textured] / (mean[textured] + 1e-4)
        limit = np.percentile(index, 2)
        grey = textured & (spread / (mean + 1e-4) <= limit)
        if grey.any():
            # Grey pixels of a coloured surface still differ; only the most consistent count.
            ratio = np.log((x[..., 0] + 1e-4) / (x[..., 1] + 1e-4))
            grey &= np.abs(ratio - np.median(ratio[grey])) < .6
    if grey.sum() >= 30:
        colour = np.median(x[grey], axis=0)
        reliability = float(np.clip(grey.sum() / valid.sum() / .01, .35, 1))
    else:
        colour = np.mean(x[valid] ** 6, axis=0) ** (1 / 6)
        reliability = .3
    colour = np.maximum(colour, 1e-6)
    return (colour / colour[1]).astype(np.float32), reliability


def white_balance_sliders(light):
    """冷暖 and 色调 that make the light ``(r, 1, b)`` neutral (see ``tone`` / ``engine``: the
    sliders scale R by 2^(.4t + .15n), G by 2^(-.15n) and B by 2^(-.4t + .15n))."""
    red, blue = np.log2(1 / max(light[0], 1e-6)), np.log2(1 / max(light[2], 1e-6))
    tint = (red + blue) / .6
    temperature = (red - blue) / .8
    return float(temperature * 100), float(tint * 100)


def measure(display):
    """Statistics of a developed display RGB image (0–1) the steps look at."""
    luma = _luma(np.clip(display, 0, 1))
    q = np.percentile(luma, [.3, 1, 5, 10, 25, 50, 75, 95, 99, 99.7])
    rg = display[..., 0] - display[..., 1]
    yb = (display[..., 0] + display[..., 1]) / 2 - display[..., 2]
    colourful = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + .3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    # Share of the frame in strong warm colour (sunsets, golden light, lamplight).
    top, bottom = display.max(axis=2), display.min(axis=2)
    lit = top > .2
    warm = lit & (display[..., 0] >= display[..., 2]) & (top - bottom > .25 * np.maximum(top, 1e-3)) & \
        (display[..., 1] < display[..., 0] * .95)
    # Clipping by luminance: a blue sky fills the blue channel long before it is overexposed.
    return dict(q=q, median=float(q[5]), high_clip=float((luma >= .965).mean()),
                low_clip=float((display.max(axis=2) <= .01).mean()), colourful=colourful,
                warm=float(warm.sum() / max(1, lit.sum())))


def scene_key(linear):
    """Reinhard's automatic key, relative to an average scene (0 = average, +1 brighter key)."""
    luma = _luma(np.maximum(np.asarray(linear, np.float32), 0))
    low, high = np.percentile(luma, [1, 99])
    low, high = max(float(low), 1e-5), max(float(high), 2e-5)
    average = float(np.exp(np.mean(np.log(luma + 1e-5))))
    if high / low < 1.05:
        return 0.
    span = np.log2(high) - np.log2(low)
    return float(np.clip((2 * np.log2(max(average, 1e-5)) - np.log2(low) - np.log2(high)) / span, -1, 1))


def features(source, edits):
    """``describe`` of a photo with its camera white balance and develop reference (for fitting)."""
    from . import engine
    return describe(engine.resize_limit(source, SAMPLE_EDGE), _neutral(edits))


#: What ``describe`` measures, in order (the model file names them, so a retrained model can add
#: or drop features without the code and the file getting out of step).
FEATURES = ('p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'mean', 'spread', 'high_clip',
            'low_clip', 'colourful', 'warm', 'red_green', 'yellow_blue', 'saturation', 'key', 'light_red',
            'light_blue', 'grey_share', 'centre', 'top_bottom', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7',
            'scene_median', 'scene_range')


def describe(small, base):
    """The photo as numbers (``FEATURES``): its unedited development ``base`` at the measuring size
    and its scene-linear light.  What the automatic adjustment model reads."""
    from . import engine
    display = np.clip(engine.process(small, base), 0, 1)
    stats = measure(display)
    luma = _luma(display)
    q = np.percentile(luma, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    top, bottom = display.max(axis=2), display.min(axis=2)
    h, w = luma.shape
    centre = luma[h // 4:3 * h // 4, w // 4:3 * w // 4].mean()
    third = max(1, h // 3)
    histogram = np.histogram(luma, bins=8, range=(0, 1))[0] / luma.size
    balanced = np.maximum(_balanced(small, base), 0)
    scene = _luma(balanced)
    low, middle, high = np.percentile(scene, [1, 50, 99])
    light, reliability = illuminant(balanced)
    values = dict(zip(('p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99'), q))
    values.update(mean=luma.mean(), spread=luma.std(), high_clip=stats['high_clip'], low_clip=stats['low_clip'],
                  colourful=stats['colourful'], warm=stats['warm'],
                  red_green=(display[..., 0] - display[..., 1]).mean(),
                  yellow_blue=((display[..., 0] + display[..., 1]) / 2 - display[..., 2]).mean(),
                  saturation=((top - bottom) / np.maximum(top, 1e-3)).mean(), key=scene_key(balanced),
                  light_red=np.log2(light[0]), light_blue=np.log2(light[2]), grey_share=reliability,
                  centre=np.log2(max(centre, 1e-3) / max(luma.mean(), 1e-3)),
                  top_bottom=luma[:third].mean() - luma[-third:].mean(),
                  scene_median=np.log2(max(middle, 1e-6)),
                  scene_range=np.log2(max(high, 1e-6) / max(low, 1e-6)))
    values.update({f'h{i}': v for i, v in enumerate(histogram)})
    return np.array([float(values[name]) for name in FEATURES])


#: The fitted model (``tools/train_auto_adjust.py``) shipped with LUMEN RAW.
MODEL = 'auto-adjust-model.json'
_model = None


def model():
    """The fitted automatic adjustment model, or None (then ``suggest`` uses its rules)."""
    global _model
    if _model is None:
        import json
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / 'assets' / MODEL
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            # A model made for other measurements (an older or newer LUMEN RAW) is not used.
            count = len(data.get('features', []))
            if data.get('format') != 1 or not set(data['features']) <= set(FEATURES) or                     len(data['mean']) != count or len(data['scale']) != count or 'median' not in data['targets'] or                     any(len(t['coef']) != count for t in data['targets'].values()):
                raise ValueError('format')
            _model = data
        except (OSError, ValueError, KeyError, TypeError):
            _model = False
    return _model or None


def predict(model, features):
    """Slider values and the display median the model expects for a photo's ``features``."""
    index = {name: i for i, name in enumerate(FEATURES)}
    x = np.array([features[index[name]] for name in model['features']], np.float64)
    x = (x - np.asarray(model['mean'])) / np.asarray(model['scale'])
    x = np.clip(x, -6, 6)                      # a photo far outside the training set stays sane
    out = {}
    for target, fit in model['targets'].items():
        value = float(np.dot(fit['coef'], x) + fit['intercept'])
        low, high = fit.get('range', (-100, 100))
        out[target] = float(np.clip(value, low, high))
    return out


def _neutral(edits):
    from . import model
    base = model.recipe()
    for key in ('develop', 'white_balance', 'wb_gain', 'lens', 'process', 'monochrome'):
        if key in edits:
            base[key] = copy.deepcopy(edits[key])
    return base


def _balanced(small, base):
    from . import engine, lens
    return lens.correct(small, base) * engine._gain(base)


def _solve(test, low, high, steps=7):
    """The largest value in [low, high] for which ``test`` (true below some value) holds."""
    if not test(low):
        return low
    if test(high):
        return high
    for _ in range(steps):
        middle = (low + high) / 2
        low, high = (middle, high) if test(middle) else (low, middle)
    return low


def _exposure(a, render, target, start, bright=.975, clip=.01):
    """Exposure that brings the median to ``target`` (bisection; the median rises with exposure),
    but never so far that the bright end (99th percentile) passes ``bright`` or more than ``clip``
    of the frame clips: a dark foreground under a bright sky stays a dark foreground."""
    def at(value):
        a['exposure'] = float(value)
        return render()
    low, high = max(-3., start - 2.5), min(3., start + 2.5)
    median = _solve(lambda v: at(v)['median'] <= target, low, high)
    def safe(v):
        stats = at(v)
        return stats['q'][8] <= bright and stats['high_clip'] <= clip
    ceiling = _solve(safe, low, high)
    return float(min(median, max(ceiling, low)))


def _smoothstep(x, low, high):
    t = min(max((x - low) / (high - low), 0.), 1.)
    return t * t * (3 - 2 * t)


def _keep_high_key(target, stats, key):
    """The median to aim for when the unedited photo reads ``stats``.  A very bright scene with a
    high key that is not clipped (snow, a white studio) is bright on purpose: it keeps its
    brightness instead of being pulled to the median of ordinary photos, which turns snow grey.
    A photo that is simply overexposed has clipped highlights and is not held back."""
    share = _smoothstep(stats['median'], .72, .85) * _smoothstep(key, .3, .6) * (1 - _smoothstep(stats['high_clip'], .01, .05))
    return target + max(0., stats['median'] - target) * share


def _rounded(a):
    return {k: round(a[k], 2) if k == 'exposure' else float(round(a[k])) for k in KEYS}


def suggest(source, edits, use_model=True):
    """Slider values (``KEYS``) for the photo ``source`` with its ``edits``: from the fitted model
    (``model``) when LUMEN RAW has one, else from the rules below."""
    from . import engine, tone
    small = engine.resize_limit(source, SAMPLE_EDGE)
    base = _neutral(edits)
    a = base['adjustments']
    balanced = _balanced(small, base)

    def render():
        return measure(engine.process(small, base))

    unedited = render()
    scene = scene_key(balanced)
    fitted = model() if use_model else None
    if fitted is not None:
        # The model gives every slider but exposure, and the median the edit should reach; the
        # exposure that reaches it is solved through the pipeline, so it holds for any camera.
        guess = predict(fitted, describe(small, base))
        for key in KEYS:
            if key != 'exposure' and key in guess:
                a[key] = guess[key]
        if base.get('monochrome'):
            a['vibrance'] = a['saturation'] = 0.
        start = tone.auto_tone(small, base.get('develop', {}), engine._gain(base)) if engine.process_version(base) == 2             else engine.auto_tone(small, base.get('wb_gain'))
        target = _keep_high_key(guess['median'], unedited, scene)
        a['exposure'] = _exposure(a, render, target, float(start['exposure']), bright=1.01, clip=.03)
        return _rounded(a)

    # 1. White balance: a nudge, never a new mood.  The camera's own balance is usually close,
    # and people keep warm light warm: cooling is held back more than warming.
    light, reliability = illuminant(balanced)
    temperature, tint = white_balance_sliders(light)
    cooling = .45 * (.25 if render()['warm'] > .3 else 1)
    a['temperature'] = float(np.clip(temperature * reliability * (cooling if temperature < 0 else .6), -25, 30))
    a['tint'] = float(np.clip(tint * reliability * .5, -20, 20))
    # 2. Exposure towards a median that follows the scene's key (see ``_exposure``).
    target = _keep_high_key(MID * 2 ** (.45 * scene), unedited, scene)
    start = tone.auto_tone(small, base.get('develop', {}), engine._gain(base)) if engine.process_version(base) == 2 \
        else engine.auto_tone(small, base.get('wb_gain'))
    a['exposure'] = _exposure(a, render, target, float(start['exposure']))
    # 3. Recover crowded ends, then place the end points.
    stats = render()
    q = stats['q']
    a['highlights'] = -float(np.clip(140 * max(0., q[8] - .9) + 500 * stats['high_clip'], 0, 55))
    a['shadows'] = float(np.clip(220 * max(0., .1 - q[3]), 0, 35))
    for _ in range(2):
        stats = render()
        q = stats['q']
        if stats['high_clip'] < .003:
            a['whites'] = float(np.clip(a['whites'] + 140 * (.965 - q[9]), -30, 40))
        else:
            # Light sources and the sun clip anyway; only a large clipped area pulls whites down.
            a['whites'] = float(np.clip(a['whites'] - 1200 * (stats['high_clip'] - .003), -30, 40))
        a['blacks'] = float(np.clip(a['blacks'] - 220 * (q[0] - .02), -25, 25))
    # 4. Midtone contrast (flat photos get some, contrasty ones keep theirs), the median once
    # more, and colourfulness.
    stats = render()
    q = stats['q']
    a['contrast'] = float(np.clip(200 * (.24 - (q[6] - q[4])), -6, 25))
    a['exposure'] = round(_exposure(a, render, target, a['exposure']), 2)
    stats = render()
    a['vibrance'] = 0. if base.get('monochrome') else float(np.clip((.15 - stats['colourful']) * 260, 0, 35))
    a['saturation'] = 0.
    return _rounded(a)
