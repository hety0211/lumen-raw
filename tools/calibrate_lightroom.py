"""Fit the Lightroom → LUMEN slider factors (``lumen.lightroom.SCALE``) on test photographs.

    python tools/calibrate_lightroom.py [photo or folder …] [--size 1200] [--process 2] [--json out.json]

Adobe does not publish the curves of its sliders and no Adobe software is used here.  The
targets below are how far a Lightroom Classic slider (process version 2012 and later) moves a
typical photograph, estimated from Adobe's documentation and widely published comparisons —
approximations, stated so they can be checked and refined.  For every slider this script renders
the test photos with LUMEN at a range of slider values, measures the same statistic, and finds the
LUMEN value that reaches each target; the factor is LUMEN value / Adobe value, averaged over the
photos and over both directions.  It measures what LUMEN's own scale does on real photos (local
tone mapping, gamut mapping, OkLCh colour of process version 2), not Adobe's rendering.

Photos default to ``LUMEN_SAMPLES/playraw`` (``E:/LumenSamples`` on the build machine); slider
families that need a controlled input (colour grading, grain, noise reduction, sharpening, the
colour mixer) use synthetic images.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('LUMEN_LANGUAGE', 'en')

from lumen import color, engine, lightroom, model  # noqa: E402

RAW = {'.arw', '.cr2', '.cr3', '.nef', '.raf', '.rw2', '.dng', '.orf'}

#: Adobe value → response, per LUMEN slider (estimates; see the module docstring).
#: ev_p: change in stops of the p-th luminance percentile; iqr: spread of the middle half of the
#: tones (sRGB encoded) relative to the unedited photo; local / fine: amplitude of large / fine
#: detail relative to unedited; chroma: mean OkLCh chroma relative to unedited; warmth / magenta:
#: change in stops of red ÷ blue / of green against red and blue; corner: change in stops at the
#: frame corners.
TARGETS = {
    'contrast':    ('iqr', {-100: .72, -50: .86, 50: 1.15, 100: 1.30}),
    'highlights':  ('ev90', {-100: -.85, -50: -.45, 50: .30, 100: .55}),
    'shadows':     ('ev10', {-100: -.90, -50: -.45, 50: .70, 100: 1.30}),
    'whites':      ('ev99', {-100: -.60, -50: -.30, 50: .25, 100: .45}),
    'blacks':      ('ev2', {-100: -1.40, -50: -.70, 50: .60, 100: 1.10}),
    'clarity':     ('local', {-100: .55, -50: .78, 50: 1.22, 100: 1.45}),
    'texture':     ('fine', {-100: .62, -50: .80, 50: 1.20, 100: 1.40}),
    'dehaze':      ('spread', {-100: .70, -50: .85, 50: 1.12, 100: 1.25}),
    'saturation':  ('chroma', {-100: 0., -50: .50, 50: 1.45, 100: 1.85}),
    'vibrance':    ('chroma_low', {-100: .40, -50: .70, 50: 1.30, 100: 1.60}),
    'temperature': ('warmth', {-100: -.80, -50: -.40, 50: .40, 100: .80}),
    'tint':        ('magenta', {-100: -.40, -50: -.20, 50: .20, 100: .40}),
}
#: Settings measured on synthetic images: (statistic, {Adobe value: response}).
SYNTHETIC = {
    'hue':            ('band_hue', {-100: -24., -50: -12., 50: 12., 100: 24.}),     # OkLCh degrees, blue band
    'hsl_saturation': ('band_chroma', {-100: .08, -50: .55, 50: 1.40, 100: 1.75}),
    'hsl_luminance':  ('band_ev', {-100: -1.20, -50: -.60, 50: .50, 100: .90}),
    'grading':        ('tint', {25: .015, 50: .030, 100: .060}),                     # OkLab chroma on grey
    'vignette':       ('corner', {-100: -1.60, -50: -.80, 50: .50, 100: .90}),
    'grain':          ('grain', {25: .010, 50: .020, 100: .040}),                    # std on mid grey
    'sharpness':      ('acutance', {40: 1.12, 80: 1.25, 150: 1.45}),
    'denoise':        ('noise', {25: .80, 50: .62, 100: .40}),
    'color_noise':    ('chroma_noise', {25: .55, 50: .40, 100: .25}),
}
GRID = (10, 20, 35, 50, 70, 100)


def photos(paths):
    found = []
    for name in paths:
        p = Path(name)
        if p.is_dir():
            found += sorted(f for f in p.iterdir() if f.suffix.lower() in RAW)
        elif p.is_file():
            found.append(p)
    return found


def load(path, size, process):
    rgb, info = engine.load_image(path, preview_limit=size)
    edits = model.recipe()
    edits['process'] = process
    if info.get('develop'):
        edits['develop'] = info['develop']
    if info.get('white_balance'):
        edits['white_balance'] = info['white_balance']
    return rgb, edits


def render(source, edits):
    return engine.process(source, edits, engine.Backend('cpu'), apply_crop=False)


def linear_y(x):
    return np.maximum(color.luminance(color.srgb_to_linear(x)), 1e-6)


def statistic(kind, x, reference):
    y, y0 = linear_y(x), linear_y(reference)
    if kind.startswith('ev'):
        p = float(kind[2:])
        return float(np.log2(np.percentile(y, p) / np.percentile(y0, p)))
    if kind == 'iqr':
        l, l0 = color.linear_to_srgb(y), color.linear_to_srgb(y0)
        return float(np.subtract(*np.percentile(l, [75, 25])) / max(np.subtract(*np.percentile(l0, [75, 25])), 1e-6))
    if kind == 'spread':
        l, l0 = color.linear_to_srgb(y), color.linear_to_srgb(y0)
        return float(np.subtract(*np.percentile(l, [95, 5])) / max(np.subtract(*np.percentile(l0, [95, 5])), 1e-6))
    if kind in ('local', 'fine'):
        l, l0 = color.linear_to_srgb(y), color.linear_to_srgb(y0)
        sigma = max(2., min(x.shape[:2]) / 40) if kind == 'local' else 1.2
        band = lambda v: v - cv2.GaussianBlur(v.astype(np.float32), (0, 0), sigma)
        return float(np.std(band(l)) / max(np.std(band(l0)), 1e-6))
    lab, lab0 = color.to_oklab(color.srgb_to_linear(x)), color.to_oklab(color.srgb_to_linear(reference))
    chroma, chroma0 = np.hypot(lab[..., 1], lab[..., 2]), np.hypot(lab0[..., 1], lab0[..., 2])
    if kind == 'chroma':
        return float(chroma.mean() / max(chroma0.mean(), 1e-6))
    if kind == 'chroma_low':
        low = chroma0 < np.percentile(chroma0, 40)
        return float(chroma[low].mean() / max(chroma0[low].mean(), 1e-6))
    lin, lin0 = color.srgb_to_linear(x).reshape(-1, 3), color.srgb_to_linear(reference).reshape(-1, 3)
    mid = (y0.ravel() > np.percentile(y0, 30)) & (y0.ravel() < np.percentile(y0, 80))
    m, m0 = lin[mid].mean(axis=0) + 1e-6, lin0[mid].mean(axis=0) + 1e-6
    if kind == 'warmth':
        return float(np.log2(m[0] / m[2]) - np.log2(m0[0] / m0[2]))
    if kind == 'magenta':
        return float(-(np.log2(m[1] / np.sqrt(m[0] * m[2])) - np.log2(m0[1] / np.sqrt(m0[0] * m0[2]))))
    raise ValueError(kind)


def fit(responses, target):
    """The LUMEN value reaching ``target`` from (value, response) pairs (linear interpolation; the
    end of the range when the target is beyond it)."""
    values, measured = zip(*sorted(responses))
    measured = np.asarray(measured)
    order = np.argsort(measured)
    if not (measured.min() <= target <= measured.max()):
        return float(values[int(np.argmin(np.abs(measured - target)))]), True
    return float(np.interp(target, measured[order], np.asarray(values)[order])), False


def calibrate_photo(source, edits, keys):
    reference = render(source, edits)
    result = {}
    for key in keys:
        kind, targets = TARGETS[key]
        factors = []
        for sign in (-1, 1):
            responses = [(0., 0. if kind.startswith('ev') or kind in ('warmth', 'magenta') else 1.)]
            for value in GRID:
                e = model.validate(edits)
                e['adjustments'][key] = sign * value
                responses.append((sign * value, statistic(kind, render(source, e), reference)))
            for adobe, target in targets.items():
                if np.sign(adobe) != sign:
                    continue
                ours, clipped = fit(responses, target)
                factors.append((ours / adobe, clipped, adobe))
        result[key] = factors
    return result


# --------------------------------------------------------------------------- synthetic tests

def _scene(h=360, w=540, seed=7):
    """A synthetic outdoor-like frame: sky gradient, foliage and skin-like patches, some texture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = np.zeros((h, w, 3), np.float32)
    sky = yy < h * .45
    x[sky] = np.stack([.18 + .1 * yy[sky] / h, .32 + .1 * yy[sky] / h, .62 + .05 * yy[sky] / h], -1)
    x[~sky] = np.stack([.12, .22, .06], -1) + .05 * rng.random((int((~sky).sum()), 1)).astype(np.float32)
    x[int(h * .6):int(h * .8), int(w * .1):int(w * .3)] = [.55, .34, .24]
    x *= (1 + .08 * rng.standard_normal((h, w, 1))).astype(np.float32)
    return np.clip(x, 0, 2)


def _linear_edits(**values):
    e = model.recipe()
    e['develop'] = dict(mode='linear', curve=[[0., 0.], [1., 1.]], source='')
    for key, value in values.items():
        target = e
        *path, last = key.split('.')
        for part in path:
            target = target[part] if not part.isdigit() else target[int(part)]
        target[last if not last.isdigit() else int(last)] = value
    return model.validate(e)


def synthetic(key, value):
    """The response of one LUMEN value for the synthetic families."""
    if key in ('hue', 'hsl_saturation', 'hsl_luminance'):
        part = ('hue', 'hsl_saturation', 'hsl_luminance').index(key)
        x = _scene()
        e = _linear_edits()
        base = render(x, e)
        e['hsl'][5][part] = value
        out = render(x, e)
        lab, lab0 = color.to_oklab(color.srgb_to_linear(out)), color.to_oklab(color.srgb_to_linear(base))
        sky = np.zeros(x.shape[:2], bool)
        sky[:int(x.shape[0] * .45)] = True
        if key == 'hue':
            h = np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))[sky]
            h0 = np.degrees(np.arctan2(lab0[..., 2], lab0[..., 1]))[sky]
            return float(np.mean((h - h0 + 180) % 360 - 180))
        if key == 'hsl_saturation':
            return float(np.hypot(lab[..., 1], lab[..., 2])[sky].mean() / np.hypot(lab0[..., 1], lab0[..., 2])[sky].mean())
        return float(np.log2(linear_y(out)[sky].mean() / linear_y(base)[sky].mean()))
    if key == 'grading':
        ramp = np.repeat(np.linspace(.02, .9, 256, dtype=np.float32)[None, :, None], 32, axis=0).repeat(3, axis=2)
        e = _linear_edits()
        e['grading']['midtones'] = [30., float(value)]
        lab = color.to_oklab(color.srgb_to_linear(render(ramp, e)))
        return float(np.hypot(lab[..., 1], lab[..., 2])[:, 96:160].mean())
    if key == 'vignette':
        flat = np.full((300, 450, 3), .18, np.float32)
        e = _linear_edits(**{'effects.vignette': float(value)})
        out = linear_y(render(flat, e))
        return float(np.log2(out[:6, :6].mean() / out[140:160, 215:235].mean()))
    if key == 'grain':
        flat = np.full((300, 450, 3), .2, np.float32)
        out = render(flat, _linear_edits(**{'effects.grain': float(value)}))
        return float(color.luminance(out).std())
    if key == 'sharpness':
        rng = np.random.default_rng(3)
        x = cv2.GaussianBlur(rng.random((300, 450, 1)).astype(np.float32).repeat(3, axis=2) * .5 + .1, (0, 0), 1.5)
        base = render(x, _linear_edits())
        out = render(x, _linear_edits(**{'adjustments.sharpness': float(value)}))
        detail = lambda v: np.std(color.luminance(v) - cv2.GaussianBlur(color.luminance(v), (0, 0), 1.))
        return float(detail(out) / detail(base))
    if key in ('denoise', 'color_noise'):
        rng = np.random.default_rng(5)
        flat = np.full((300, 450, 3), .18, np.float32)
        if key == 'denoise':
            noisy = flat * (1 + .12 * rng.standard_normal((300, 450, 1))).astype(np.float32)
        else:
            noisy = flat * (1 + .15 * rng.standard_normal((300, 450, 3))).astype(np.float32)
        base = render(np.clip(noisy, 0, 1), _linear_edits())
        out = render(np.clip(noisy, 0, 1), _linear_edits(**{f'adjustments.{key}': float(value)}))
        if key == 'denoise':
            return float(color.luminance(out).std() / color.luminance(base).std())
        lab, lab0 = color.to_oklab(color.srgb_to_linear(out)), color.to_oklab(color.srgb_to_linear(base))
        return float(np.hypot(lab[..., 1], lab[..., 2]).std() / np.hypot(lab0[..., 1], lab0[..., 2]).std())
    raise ValueError(key)


def calibrate_synthetic(key):
    kind, targets = SYNTHETIC[key]
    ranged = key in ('hue', 'hsl_saturation', 'hsl_luminance', 'vignette')
    neutral = synthetic(key, 0)
    factors = []
    for sign in ((-1, 1) if ranged else (1,)):
        responses = [(0., neutral)] + [(sign * v, synthetic(key, sign * v)) for v in GRID]
        for adobe, target in targets.items():
            if np.sign(adobe) != sign:
                continue
            ours, clipped = fit(responses, target)
            factors.append((ours / adobe, clipped, adobe))
    return factors


def summarize(factors):
    """Median factor of the targets LUMEN reaches.  A target beyond LUMEN's ±100 says LUMEN's range
    is narrower than Adobe's there; the factor then matches the reachable (usually moderate)
    values and the strongest Adobe values clip at ±100.  Without any reachable target it is 1."""
    reached = [f for f, clipped, _ in factors if not clipped]
    clipped = len(factors) - len(reached)
    factor = float(np.median(reached)) if reached else 1.
    by_target = {}
    for f, c, adobe in factors:
        by_target.setdefault(adobe, []).append((f, c))
    detail = {str(int(a)): dict(factor=round(float(np.median([f for f, _ in v])), 2), clipped=sum(c for _, c in v),
                                samples=len(v)) for a, v in sorted(by_target.items())}
    return dict(factor=round(factor, 2), low=round(float(min(reached)), 2) if reached else None,
                high=round(float(max(reached)), 2) if reached else None, clipped=clipped, samples=len(factors),
                targets=detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('photos', nargs='*')
    parser.add_argument('--size', type=int, default=1200)
    parser.add_argument('--process', type=int, default=2, choices=(1, 2))
    parser.add_argument('--limit', type=int, default=8, help='at most this many photos')
    parser.add_argument('--json', help='write the results here')
    args = parser.parse_args()
    samples = Path(os.environ.get('LUMEN_SAMPLES', 'E:/LumenSamples'))
    files = photos(args.photos or [samples / 'playraw'])[:args.limit]
    per_key = {k: [] for k in TARGETS}
    for path in files:
        source, edits = load(path, args.size, args.process)
        print(f'{path.name}: {source.shape[1]} × {source.shape[0]}', flush=True)
        for key, factors in calibrate_photo(source, edits, list(TARGETS)).items():
            per_key[key] += factors
    results = {k: summarize(v) for k, v in per_key.items() if v}
    for key in SYNTHETIC:
        results[key] = summarize(calibrate_synthetic(key))
    print(f'\n{"slider":16} {"fitted":>7} {"min":>6} {"max":>6}  current  clipped  by Adobe value: factor (clipped/n)')
    for key, r in results.items():
        current = lightroom.SCALE.get(key)
        targets = '  '.join(f'{a}:{t["factor"]:.2f}({t["clipped"]}/{t["samples"]})' for a, t in r['targets'].items())
        low, high = (f'{v:6.2f}' if v is not None else '     -' for v in (r['low'], r['high']))
        print(f'{key:16} {r["factor"]:7.2f} {low} {high}  {current!s:>7}  {r["clipped"]:>3}/{r["samples"]:<3}  {targets}')
    if args.json:
        Path(args.json).write_text(json.dumps(dict(photos=[p.name for p in files], process=args.process, size=args.size,
                                                   results=results), indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
