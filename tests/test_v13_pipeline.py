"""1.3.0: stage cache, mask bounding-box rendering and fused GPU pixel graphs."""
import copy
import numpy as np
import onnxruntime as ort
import pytest
from lumen import compute, engine, gpu_graphs, model


def source(seed=7, shape=(150, 220, 3)):
    rng = np.random.default_rng(seed)
    image = rng.random(shape, dtype=np.float32) ** 2 * .8
    image += np.linspace(0, .2, shape[1], dtype=np.float32)[None, :, None]
    return np.ascontiguousarray(np.clip(image, 0, 1))


def edits_with_masks():
    e = model.recipe()
    e['process'] = 1  # the 1.x pipeline these 1.3.0 checks describe (process 2: test_v160)
    e['adjustments'].update(exposure=.4, shadows=35, contrast=12, clarity=30, texture=20,
                            sharpness=40, denoise=15, saturation=10, vibrance=-15)
    e['develop'] = dict(mode='camera', curve=[[0, 0], [.1, .3], [1, 1]], source='test')
    e['hsl'][4] = [20., -30., 10.]
    e['curves']['RGB'] = [[0, 0], [.4, .45], [1, 1]]
    e['curve_mode'] = 'smooth'
    e['grading']['highlights'] = [45., 40.]
    radial = model.new_mask('radial', 1)
    radial.update(start=[.1, .15], end=[.35, .5])
    radial['adjustments'].update(exposure=.6, clarity=40, sharpness=30)
    brush = model.new_mask('brush', 2)
    brush['strokes'] = [dict(points=[[.6, .6], [.7, .65]], radius=.05, erase=False)]
    brush['adjustments'].update(highlights=-40, texture=25, denoise=20)
    luminance = model.new_mask('luminance', 3)
    luminance['luminance_range'] = [60., 100.]
    luminance['adjustments'].update(saturation=-30, dehaze=25)
    e['masks'] += [radial, brush, luminance]
    return e


def reference_process(src, e, detail_scale=1.):
    """The 1.2.2 algorithm: every stage and every mask over the whole frame."""
    a = e['adjustments']
    from lumen import develop, retouch, white_balance
    gain = np.asarray(e['wb_gain'], np.float32) * white_balance.gains(e['white_balance'])
    x = develop.apply(retouch.apply(src, e['retouch']), e['develop']) * gain
    x = engine.Backend('cpu').tonal(x, a)
    x = engine.details(x, a, detail_scale)
    x = engine.apply_hsl(x, e['hsl'])
    x = engine.apply_curves(x, e['curves'], e['curve_mode'])
    x = engine.color_grade(x, e['grading'])
    reference = np.clip(engine.to_srgb(develop.apply(src, e['develop'])), 0, 1)
    for m in e['masks']:
        alpha = engine.mask_alpha(m, x.shape, reference)[..., None]
        local = engine.details(engine.Backend('cpu').tonal(engine.to_linear(x), m['adjustments']), m['adjustments'], detail_scale)
        x = x * (1 - alpha) + local * alpha
    return np.clip(engine.finishing(x, e['effects'], e['crop']), 0, 1)


@pytest.mark.parametrize('detail_scale', [1., .35])
def test_mask_bounding_boxes_match_whole_frame_rendering(detail_scale):
    src, e = source(), edits_with_masks()
    expected = reference_process(src, e, detail_scale)
    actual = engine.process(src, e, engine.Backend('cpu'), apply_crop=False, detail_scale=detail_scale)
    np.testing.assert_allclose(actual, expected, atol=2e-6)


def test_stage_cache_reuses_upstream_results_and_matches_uncached():
    src, e = source(), edits_with_masks()
    cache = engine.RenderCache()
    backend = engine.Backend('cpu')
    first = engine.process(src, e, backend, apply_crop=False, cache=cache)
    misses = cache.misses
    changed = copy.deepcopy(e)
    changed['hsl'][4][1] = -55.
    cached = engine.process(src, changed, backend, apply_crop=False, cache=cache)
    # Base, tonal, spatial detail, luminance reference and all three mask alphas are reused.
    assert cache.misses == misses + 1 and cache.hits >= 6
    np.testing.assert_array_equal(cached, engine.process(src, changed, backend, apply_crop=False))
    np.testing.assert_array_equal(first, engine.process(src, e, backend, apply_crop=False))


def test_stage_cache_never_leaks_between_sources_or_mutates_results():
    e = edits_with_masks()
    cache = engine.RenderCache()
    one, two = source(1), source(2)
    a = engine.process(one, e, apply_crop=False, cache=cache)
    b = engine.process(two, e, apply_crop=False, cache=cache)
    np.testing.assert_array_equal(b, engine.process(two, e, apply_crop=False))
    again = engine.process(one, e, apply_crop=False, cache=cache)
    np.testing.assert_array_equal(a, again)
    snapshot = {k: v[1].copy() for k, v in cache._stages.items()}
    engine.process(one, e, apply_crop=False, cache=cache)
    for key, value in snapshot.items():
        np.testing.assert_array_equal(cache._stages[key][1], value)


def test_stage_cache_respects_its_memory_budget():
    cache = engine.RenderCache(max_bytes=source().nbytes + 1)
    e = edits_with_masks()
    engine.process(source(), e, apply_crop=False, cache=cache)
    assert cache._stored() <= cache.max_bytes


def run_graph(kind, image, inputs):
    session = ort.InferenceSession(gpu_graphs.model(kind), providers=['CPUExecutionProvider'])
    return session.run(None, dict(inputs, image=image[None]))[0][0]


@pytest.mark.parametrize('mode,monochrome', [('smooth', False), ('linear', False), ('smooth', True)])
def test_color_and_fused_graphs_match_cpu_reference(mode, monochrome):
    image = source(5, (47, 63, 3)) * 1.2
    image[0, :4] = [0, 1, .5, .02][0]
    e = edits_with_masks()
    e['masks'] = []
    for k in engine.SPATIAL_KEYS:
        e['adjustments'][k] = 0.
    e['hsl'] = [[20, -30, 15], [0, 40, -20], [-50, 10, 0], [10, 0, 30],
                [0, -60, 0], [35, 20, -10], [0, 0, 0], [-15, 5, 25]]
    e['curves']['G'] = [[0, .05], [.5, .55], [1, .95]]
    e['curve_mode'], e['monochrome'] = mode, monochrome
    e['grading'] = dict(shadows=[220., 40.], midtones=[30., 25.], highlights=[45., 60.], balance=20.)
    tonal = engine.Backend('cpu').tonal(image, e['adjustments'])
    expected = engine.color_stage(tonal, e)
    # Linear curves are resampled on a 4097 grid; float32 hue branches may flip at exact ties.
    tolerance = 3e-5 if mode == 'linear' else 2e-5
    np.testing.assert_allclose(run_graph('color', tonal, gpu_graphs.color_inputs(e)), expected, atol=tolerance)
    fused = dict(gpu_graphs.tonal_inputs(e['adjustments']), **gpu_graphs.color_inputs(e))
    np.testing.assert_allclose(run_graph('fused', image.astype(np.float32), fused), expected, atol=tolerance)


def test_gpu_backend_graph_path_matches_cpu_pipeline(monkeypatch):
    """Force the graph path on the CPU provider to check the tiled fused / color routes."""
    src = source(9, (90, 130, 3))
    for spatial in (False, True):
        e = edits_with_masks()
        e['masks'] = e['masks'][:2]
        if not spatial:
            for k in engine.SPATIAL_KEYS:
                e['adjustments'][k] = 0.
        backend = engine.Backend('cpu')
        backend.use_dml = True
        # The CPU provider stands in for DirectML here; keep the graph route active.
        monkeypatch.setattr(backend, '_disable_gpu', lambda warning: None)
        calls = []
        run = backend._run_graph
        monkeypatch.setattr(backend, '_run_graph', lambda kind, *a: calls.append(kind) or run(kind, *a))
        monkeypatch.setattr(engine, 'GPU_TILE', 64)
        graph = engine.process(src, e, backend, apply_crop=False)
        cpu = engine.process(src, e, engine.Backend('cpu'), apply_crop=False)
        np.testing.assert_allclose(graph, cpu, atol=2e-5)
        assert calls[0] == ('tonal' if spatial else 'fused') and ('color' in calls) == spatial


def test_identity_color_graph_is_transparent():
    image = source(3, (32, 40, 3))
    np.testing.assert_allclose(run_graph('color', image, gpu_graphs.color_inputs(model.recipe())), image, atol=2e-6)


def test_compute_mode_override_and_experimental_cupy(monkeypatch):
    monkeypatch.delenv('LUMEN_EXPERIMENTAL_CUPY', raising=False)
    monkeypatch.setenv('LUMEN_COMPUTE', 'cpu')
    assert compute.provider_plan(['DmlExecutionProvider', 'CPUExecutionProvider'], (0, 'GPU', 1, 1)) == []
    assert engine.Backend('auto').gpu_pointwise is False
    monkeypatch.setenv('LUMEN_COMPUTE', 'dml')
    plan = compute.provider_plan(['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider'], (1, 'GPU', 1, 1))
    assert [p[1] for p in plan] == ['DmlExecutionProvider'] and plan[0][0][0][1] == {'device_id': 1}
    monkeypatch.setenv('LUMEN_COMPUTE', 'bogus')
    assert compute.requested_mode() == 'auto' and not compute.cupy_enabled()
    monkeypatch.setenv('LUMEN_EXPERIMENTAL_CUPY', '1')
    assert compute.cupy_enabled()


def test_large_sources_keep_the_stage_cache_when_it_fits(monkeypatch):
    """1.3.1: images above the streaming threshold (24 MP+) used to bypass the cache."""
    from lumen import large_image
    src = source(4, (120, 160, 3))
    e = model.recipe()
    e['adjustments'].update(exposure=.3, shadows=20, saturation=10)
    e['hsl'][3] = [10., 20., 0.]
    monkeypatch.setattr(large_image, 'MAP_BYTES', src.nbytes // 4)
    streamed = engine.process(src, e, engine.Backend('cpu'), apply_crop=False)
    cache = engine.RenderCache(max_bytes=src.nbytes * 8)
    first = engine.process(src, e, engine.Backend('cpu'), apply_crop=False, cache=cache)
    changed = copy.deepcopy(e)
    changed['hsl'][3][0] = -15.
    engine.process(src, changed, engine.Backend('cpu'), apply_crop=False, cache=cache)
    assert cache.hits >= 2  # base and tonal reused at "full" size
    np.testing.assert_allclose(first, streamed, atol=1e-6)
    small = engine.RenderCache(max_bytes=src.nbytes)
    engine.process(src, e, engine.Backend('cpu'), apply_crop=False, cache=small)
    assert small.misses == 0  # too small a budget: the streaming path is used, as before
