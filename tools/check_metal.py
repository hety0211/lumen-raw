"""macOS hardware check (1.3.1): Metal pixel kernels and Core ML for every model.

Returns nonzero unless the Metal pipeline compiles, matches the NumPy reference
for all three stages and actually runs.  For each ONNX model it reports the
provider ONNX Runtime used, the first-run (Core ML compile) and steady-state
time, and the CPU time for comparison.  Usage:
    python tools/check_metal.py [report.json] [--models]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lumen import compute, engine, gpu_graphs, metal, model  # noqa: E402

TILES = {'realesr-general-x4v3.onnx': 192, 'realesrgan-x4plus.onnx': 192, 'nafnet-sidd.onnx': 320,
         'drunet-color.onnx': 320, 'ffdnet-color.onnx': 320, 'skyseg.onnx': 320,
         'person-deeplab.onnx': 384, 'u2netp.onnx': 320, 'midas-small.onnx': 256}


def edits():
    e = model.recipe()
    e['process'] = 1                     # these are the process-1 stages; 1.6.0 recipes default to 2
    e['adjustments'].update(exposure=.35, contrast=14, shadows=28, highlights=-30, blacks=-8, whites=10,
                            temperature=-12, tint=6, saturation=8, vibrance=15)
    e['hsl'] = [[10, -20, 5], [0, 15, 0], [-5, 0, 10], [0, -10, 0], [15, 0, -5], [0, 20, 0], [0, 0, 0], [-10, 5, 0]]
    e['curves']['RGB'] = [[0., 0.], [.3, .27], [.7, .76], [1., 1.]]
    e['curves']['R'] = [[0., .02], [1., 1.]]
    e['grading'].update(shadows=[210., 25.], midtones=[35., 10.], highlights=[45., 30.], balance=-10.)
    return e


def check_pixels(report):
    pipeline = metal.pipeline()
    if pipeline is None:
        raise SystemExit('Metal pixel pipeline unavailable: ' + metal.last_error())
    e = edits()
    rng = np.random.default_rng(7)
    image = (rng.random((1067, 1600, 3), dtype=np.float32) ** 2.2) * 1.1
    tonal_inputs = gpu_graphs.tonal_inputs(e['adjustments'])
    color_inputs = gpu_graphs.color_inputs(e)
    tonal = engine.Backend('cpu').tonal(image, e['adjustments'])
    expected = {'tonal': tonal, 'color': engine.color_stage(tonal, e)}
    expected['fused'] = expected['color']
    sources = {'tonal': image, 'color': tonal, 'fused': image}
    inputs = {'tonal': tonal_inputs, 'color': color_inputs, 'fused': dict(tonal_inputs, **color_inputs)}
    for kind in ('tonal', 'color', 'fused'):
        pipeline.run(kind, sources[kind], inputs[kind])          # warm-up
        started = time.perf_counter()
        result = pipeline.run(kind, sources[kind], inputs[kind])
        elapsed = (time.perf_counter() - started) * 1000
        error = float(np.max(np.abs(result - expected[kind])))
        report['pixels'][kind] = dict(max_error=error, metal_ms=round(elapsed, 2))
        print(f'{kind}: Metal {elapsed:.1f} ms on 1600 x 1067 · max error vs NumPy {error:.2e}')
        if not np.isfinite(result).all() or error > 1e-4:
            raise SystemExit(f'Metal {kind} kernel differs from the NumPy reference ({error:.2e})')
    # Process 2 (1.6.0): pipeline() already ran its self-test; a failure only disables process 2
    # in the app, but a release must have it.
    if not pipeline.version2:
        raise SystemExit('Metal process-2 kernel unavailable: ' + pipeline.error2)
    report['pixels']['process2'] = dict(max_error=pipeline.self_test2())
    print(f'process 2 fused: max error vs NumPy {report["pixels"]["process2"]["max_error"]:.2e}')
    started = time.perf_counter()
    engine.color_stage(engine.Backend('cpu').tonal(image, e['adjustments']), e)
    report['pixels']['numpy_fused_ms'] = round((time.perf_counter() - started) * 1000, 1)
    print(f'NumPy tonal + color: {report["pixels"]["numpy_fused_ms"]:.0f} ms')
    report['metal'] = dict(device=pipeline.name, threadgroup=pipeline.group)


def inputs_for(session, size):
    feeds = {}
    for item in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else size for d in item.shape]
        if item.name == 'sigma':
            feeds[item.name] = np.full([1 if not isinstance(d, int) else d for d in item.shape], .05, np.float32)
        else:
            feeds[item.name] = np.random.default_rng(1).random(shape, dtype=np.float32)
    return feeds


def timed(session, feeds, repeat=3):
    started = time.perf_counter()
    first = session.run(None, feeds)
    first_ms = (time.perf_counter() - started) * 1000
    times = []
    for _ in range(repeat):
        started = time.perf_counter()
        session.run(None, feeds)
        times.append((time.perf_counter() - started) * 1000)
    return first, first_ms, min(times)


def check_models(report):
    folder = ROOT / 'assets' / 'models'
    for name, size in TILES.items():
        path = folder / name
        if not path.exists():
            print(f'{name}: missing')
            continue
        started = time.perf_counter()
        accelerated = compute.Session(path, True)
        open_ms = (time.perf_counter() - started) * 1000
        feeds = inputs_for(accelerated, size)
        result, first_ms, steady_ms = timed(accelerated, feeds)
        cpu = compute.Session(path, False)
        reference, _, cpu_ms = timed(cpu, feeds)
        error = float(np.max(np.abs(result[0] - reference[0])))
        row = dict(provider=accelerated.provider, device=accelerated.device, warning=accelerated.warning,
                   open_ms=round(open_ms), first_ms=round(first_ms), steady_ms=round(steady_ms, 1),
                   cpu_ms=round(cpu_ms, 1), max_error=error, tile=size)
        report['models'][name] = row
        print(f'{name}: {accelerated.provider} · open {open_ms:.0f} ms · first {first_ms:.0f} ms · '
              f'{steady_ms:.0f} ms vs CPU {cpu_ms:.0f} ms · max error {error:.1e}'
              + (f' · {accelerated.warning}' if accelerated.warning else ''))


def main():
    arguments = [a for a in sys.argv[1:] if not a.startswith('--')]
    report = dict(pixels={}, models={}, providers=__import__('onnxruntime').get_available_providers(),
                  macos=metal.macos_version())
    check_pixels(report)
    if '--models' in sys.argv:
        compute.coreml_housekeeping()           # remember the Core ML cache entries that are not ours
        check_models(report)
        compute.coreml_housekeeping()           # and drop the ones these sessions added
    if arguments:
        Path(arguments[0]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
