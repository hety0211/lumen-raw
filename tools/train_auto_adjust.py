"""Fit the automatic adjustment model (2.3.0) from photos edited by hand.

    python tools/train_auto_adjust.py FOLDER [FOLDER …] [--out assets/auto-adjust-model.json]
                                      [--cache DIR] [--report report.json] [--folds 10]

Every RAW (or JPEG / TIFF) in the folders with a ``.lumen`` project next to it is one example:
the photo is developed small with its camera white balance and develop reference only, measured
(``auto_adjust.describe``), and paired with the ten sliders the project sets (``auto_adjust.KEYS``)
and the display median the edit reaches.  Projects that changed the Kelvin temperature or used
the white-balance eyedropper do not teach temperature and tint (those two targets are left out
for them).  Edit only those ten sliders, from the default, when making training projects.

One ridge regression per target over standardised features; the strength is chosen per target by
k-fold cross-validation, and the report compares the fit with the built-in rules
(``auto_adjust.suggest`` without a model) and with always answering the average.  Exposure is
not regressed directly: the model predicts the median, and ``auto_adjust.suggest`` solves the
exposure that reaches it with the other predicted sliders in place.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Never the user's library, settings or preview cache.
for key, name in (('LUMEN_CATALOG', 'catalog.jsonl'), ('LUMEN_PREVIEW_CACHE', 'previews'), ('LOCALAPPDATA', 'data')):
    os.environ.setdefault(key, os.path.join(tempfile.gettempdir(), 'lumen-train-auto', name))
os.makedirs(os.environ['LOCALAPPDATA'], exist_ok=True)

from lumen import auto_adjust, engine, model  # noqa: E402

SUFFIXES = engine.RAW_EXTENSIONS | {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
#: Targets the model fits: the median for the exposure solve, then the other sliders.
TARGETS = ('median', 'contrast', 'highlights', 'shadows', 'whites', 'blacks', 'temperature', 'tint', 'vibrance',
           'saturation')
RANGES = dict(median=(.04, .8), contrast=(-30, 40), highlights=(-80, 30), shadows=(-30, 70), whites=(-40, 40),
              blacks=(-40, 40), temperature=(-100, 100), tint=(-100, 100), vibrance=(-20, 50), saturation=(-30, 30))
ALPHAS = (.1, .3, 1, 3, 10, 30, 100, 300)


def pairs(folders):
    for folder in folders:
        for photo in sorted(Path(folder).iterdir()):
            project = photo.with_suffix('.lumen')
            if photo.suffix.lower() in SUFFIXES and project.is_file():
                yield photo, project


def example(job):
    """Features, targets and target weights of one edited photo."""
    photo, project, cache = job
    cached = Path(cache) / (photo.stem + '.npz') if cache else None
    if cached is not None and cached.is_file():
        data = np.load(cached)
        source, info = data['source'].astype(np.float32), json.loads(str(data['info']))
    else:
        source, full = engine.load_image(str(photo), preview_limit=1000)
        info = dict(white_balance=full.get('white_balance'), develop=full.get('develop'))
        if cached is not None:
            np.savez_compressed(cached, source=np.asarray(source, np.float16), info=json.dumps(info))
    _, edits = model.load_project(project)
    base = model.recipe()
    base['process'] = edits.get('process', model.PROCESS)
    base['develop'] = info.get('develop') or edits['develop']
    base['white_balance'] = copy.deepcopy(info.get('white_balance') or edits['white_balance'])
    base['lens'] = copy.deepcopy(edits['lens'])
    small = engine.resize_limit(source, auto_adjust.SAMPLE_EDGE)
    features = auto_adjust.describe(small, base)
    edited = copy.deepcopy(base)
    a = edits['adjustments']
    for key in auto_adjust.KEYS:
        edited['adjustments'][key] = a[key]
    wb = edits.get('white_balance', {})
    colour_known = wb.get('kelvin') == wb.get('camera_kelvin') and edits.get('wb_gain', [1, 1, 1]) == [1, 1, 1]
    if not colour_known:
        # The user's own white balance stays in the picture the median is measured on.
        edited['white_balance'], edited['wb_gain'] = copy.deepcopy(wb), list(edits.get('wb_gain', [1, 1, 1]))
    median = auto_adjust.measure(engine.process(small, edited))['median']
    targets = dict(median=median, **{k: a[k] for k in TARGETS if k != 'median'})
    weights = {k: 0. if k in ('temperature', 'tint') and not colour_known else 1. for k in TARGETS}
    rules = auto_adjust.suggest(source, base, use_model=False)
    return dict(name=photo.name, features=features.tolist(), targets=targets, weights=weights, rules=rules,
                exposure=a['exposure'])


def ridge(x, y, w, alpha):
    """Weighted ridge regression on standardised ``x`` (no penalty on the intercept)."""
    sw = np.sqrt(w)[:, None]
    design = np.hstack([x, np.ones((len(x), 1))])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[-1, -1] = 0
    solution = np.linalg.solve((design * sw).T @ (design * sw) + penalty, (design * sw).T @ (y * sw[:, 0]))
    return solution[:-1], solution[-1]


def folds(n, k, seed=7):
    order = np.random.default_rng(seed).permutation(n)
    return [order[i::k] for i in range(k)]


def cross_validate(x, y, w, alpha, k):
    predictions = np.zeros_like(y)
    for test in folds(len(y), k):
        train = np.setdiff1d(np.arange(len(y)), test)
        coef, intercept = ridge(x[train], y[train], w[train], alpha)
        predictions[test] = x[test] @ coef + intercept
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folders', nargs='+')
    parser.add_argument('--out', default=str(ROOT / 'assets' / auto_adjust.MODEL))
    parser.add_argument('--cache', default='')
    parser.add_argument('--report', default='')
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args()
    if args.cache:
        Path(args.cache).mkdir(parents=True, exist_ok=True)
    jobs = [(photo, project, args.cache) for photo, project in pairs(args.folders)]
    if len(jobs) < 30:
        raise SystemExit(f'only {len(jobs)} edited photos; at least 30 are needed')
    started = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        examples = list(pool.map(example, jobs))
    print(f'{len(examples)} photos measured in {time.time() - started:.0f} s')
    features = np.array([e['features'] for e in examples])
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale[scale < 1e-9] = 1
    x = (features - mean) / scale
    fitted, report = {}, dict(photos=len(examples), folds=args.folds, targets={})
    for target in TARGETS:
        y = np.array([e['targets'][target] for e in examples], np.float64)
        w = np.array([e['weights'][target] for e in examples], np.float64)
        used = w > 0
        best = min(ALPHAS, key=lambda alpha: np.average(np.abs(cross_validate(x[used], y[used], w[used], alpha, args.folds)
                                                              - y[used])))
        cv = cross_validate(x[used], y[used], w[used], best, args.folds)
        coef, intercept = ridge(x[used], y[used], w[used], best)
        fitted[target] = dict(coef=[round(float(c), 6) for c in coef], intercept=round(float(intercept), 6),
                              alpha=best, range=list(RANGES[target]))
        entry = dict(alpha=best, model_mae=round(float(np.abs(cv - y[used]).mean()), 4),
                     mean_mae=round(float(np.abs(y[used].mean() - y[used]).mean()), 4), examples=int(used.sum()))
        if target != 'median':
            rules = np.array([e['rules'][target] for e in examples])[used]
            entry['rules_mae'] = round(float(np.abs(rules - y[used]).mean()), 4)
        report['targets'][target] = entry
        print(f"{target:12s} alpha {best:<5} CV MAE model {entry['model_mae']:<8} rules {entry.get('rules_mae', '-'):<8} "
              f"average {entry['mean_mae']}")
    data = {'format': 1, 'name': 'LUMEN RAW automatic adjustment', 'photos': len(examples),
            'trained': time.strftime('%Y-%m-%d'), 'features': list(auto_adjust.FEATURES),
            'mean': [round(float(v), 6) for v in mean], 'scale': [round(float(v), 6) for v in scale],
            'targets': fitted}
    Path(args.out).write_text(json.dumps(data, indent=1) + '\n', encoding='utf-8')
    print('model written to', args.out)
    if args.report:
        report['examples'] = examples
        Path(args.report).write_text(json.dumps(report, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
