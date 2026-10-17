"""1.7.0: fine AI mask edges refined on the original photograph."""
import copy
import json

import cv2
import numpy as np
import pytest
from PIL import Image

from lumen import catalog, commands, engine, model, selection


def branches(width=2000, height=1334, seed=3):
    """Dark twigs against a blue sky gradient and the true sky coverage (anti-aliased)."""
    rng = np.random.default_rng(seed)
    scale = 4
    big = np.zeros((height * scale, width * scale), np.uint8)
    for _ in range(70):
        x, y = int(rng.integers(0, width * scale)), int(height * scale * .55)
        angle = rng.uniform(-2.4, -.7)
        for _ in range(6):
            length = rng.uniform(120, 420) * scale
            nx, ny = int(x + length * np.cos(angle)), int(y + length * np.sin(angle))
            cv2.line(big, (x, y), (nx, ny), 255, int(rng.integers(1, 4)) * scale)
            x, y, angle = nx, ny, angle + rng.uniform(-.5, .5)
    big[int(height * scale * .55):] = 255
    tree = cv2.resize(big.astype(np.float32) / 255, (width, height), interpolation=cv2.INTER_AREA)
    sky = 1 - tree
    yy = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    colour = np.concatenate([.25 + .1 * yy, .45 + .1 * yy, .85 - .05 * yy], axis=2) * np.ones((1, width, 1), np.float32)
    dark = np.array([.08, .1, .05], np.float32)
    rgb = colour * sky[..., None] + dark * tree[..., None]
    return np.clip(rgb + rng.normal(0, .004, rgb.shape).astype(np.float32), 0, 1), sky


def quick(sky, rgb, network=320, preview=800):
    """What the quick pass gives: a network-sized mask aligned on a preview."""
    h, w = sky.shape
    net = cv2.resize(sky, (network, network), interpolation=cv2.INTER_AREA)
    small = cv2.resize(rgb, (preview, round(preview * h / w)), interpolation=cv2.INTER_AREA)
    return selection.refine(net, small)


def test_fine_pass_recovers_twigs_the_quick_mask_misses():
    rgb, truth = branches()
    coarse = quick(truth, rgb)
    up = cv2.resize(coarse, truth.shape[::-1], interpolation=cv2.INTER_LINEAR)
    refined = selection.refine_fine('sky', coarse, rgb)
    assert refined.shape == truth.shape and refined.dtype == np.float32
    edge = (cv2.dilate((truth < .98).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & (truth > .02)
    edge |= (truth > .02) & (truth < .98)
    before, after = np.abs(up - truth)[edge].mean(), np.abs(refined - truth)[edge].mean()
    assert after < before * .5, (before, after)
    # Thin twigs: pixels mostly covered by a branch are recognised as not-sky.
    twigs = (truth < .5) & (cv2.erode((truth < .5).astype(np.uint8), np.ones((9, 9), np.uint8)) == 0)
    twigs[int(truth.shape[0] * .5):] = False
    assert (refined[twigs] < .5).mean() > .6 > (up[twigs] < .5).mean()


def test_fine_pass_keeps_broad_ramps_and_known_regions():
    h, w = 900, 1350
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    haze = np.repeat(np.stack([.9 - .5 * yy, .7 - .4 * yy, .5 - .3 * yy], -1), w, axis=1).astype(np.float32)
    ramp = np.clip(1.6 - 2 * np.linspace(0, 1, 400, dtype=np.float32)[:, None], 0, 1) * np.ones((1, 600), np.float32)
    refined = selection.fine(ramp, haze, 320)
    expected = cv2.resize(ramp, (w, h), interpolation=cv2.INTER_LINEAR)
    assert np.abs(refined - expected).mean() < .03
    rgb, truth = branches(1200, 800)
    coarse = quick(truth, rgb, preview=600)
    refined = selection.fine(coarse, rgb, 320)
    up = cv2.resize(coarse, (1200, 800), interpolation=cv2.INTER_LINEAR)
    far = cv2.erode((up > .999).astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
    assert far.any() and np.all(refined[far] == 1)
    assert np.all(refined[cv2.erode((up < 1e-4).astype(np.uint8), np.ones((61, 61), np.uint8)) > 0] == 0)


def test_kinds_and_guide_size():
    alpha = np.full((80, 120), .5, np.float32)
    guide = np.full((400, 600, 3), .5, np.float32)
    assert selection.refine_fine('foreground', alpha, guide) is None and selection.refine_fine('color', alpha, guide) is None
    assert selection.refine_fine('background', alpha, guide).shape == (400, 600)
    assert selection.fine_size(9504, 6336) == (4096, 2731) and selection.fine_size(3000, 2000) == (3000, 2000)
    source = np.random.default_rng(1).random((5000, 3000, 3), dtype=np.float32) * .1
    assert selection.fine_guide(source, model.recipe()).shape == (4096, 2458, 3)


# ----------------------------------------------------------------------------- storage and rendering

def test_large_rasters_render_the_same_in_blocks_and_whole_frames():
    rgb, truth = branches(1600, 1067)
    mask = model.new_mask('sky', 1)
    mask.update(raster=selection.encode(truth), feather=20.)
    shape = (533, 800)
    sigma = 20 / 100 * 533 * .008
    whole = selection.raster_alpha(mask, shape)
    reference = cv2.GaussianBlur(cv2.resize(selection.decode(mask['raster']), (800, 533), interpolation=cv2.INTER_AREA),
                                 (0, 0), sigma)
    np.testing.assert_allclose(whole, reference, atol=1e-6)
    sharp = dict(mask, feather=0.)
    sharp_whole = selection.raster_alpha(sharp, shape)
    for rect in ((0, 0, 800, 533), (101, 37, 455, 300), (700, 500, 800, 533)):
        # The averaged-down raster is shared, so blocks read exactly the frame's pixels…
        block = selection.raster_alpha(sharp, shape, engine.Area(800, 533, *rect))
        np.testing.assert_array_equal(block, sharp_whole[rect[1]:rect[3], rect[0]:rect[2]])
        # …and feathering differs only by the blur's float rounding at block borders, as before.
        block = selection.raster_alpha(mask, shape, engine.Area(800, 533, *rect))
        np.testing.assert_allclose(block, whole[rect[1]:rect[3], rect[0]:rect[2]], atol=1e-6)
    # A raster smaller than the frame is still interpolated as before.
    small = dict(mask, raster=selection.encode(cv2.resize(truth, (400, 267), interpolation=cv2.INTER_AREA)))
    expected = cv2.GaussianBlur(cv2.resize(selection.decode(small['raster']), (800, 533), interpolation=cv2.INTER_LINEAR),
                                (0, 0), sigma)
    np.testing.assert_array_equal(selection.raster_alpha(small, shape), expected)


def test_mask_size_limits():
    fine = np.zeros((2731, 4096), np.float32)
    fine[:1000] = 1
    assert selection.decode(selection.encode(fine)).shape == (2731, 4096)
    with pytest.raises(ValueError):
        selection.decode(selection.encode(np.zeros((10, 9000), np.float32)))
    with pytest.raises(ValueError):
        selection.decode(selection.encode(np.zeros((7000, 7000), np.float32)))


def test_projects_and_albums_store_each_mask_raster_once(tmp_path):
    rgb, truth = branches(2400, 1600)
    edits = model.recipe()
    mask = model.new_mask('sky', 1)
    mask.update(raster=selection.encode(truth))
    mask['adjustments']['exposure'] = -.5
    edits['masks'].append(mask)
    snapshots = [dict(name=f'v{i}', edits=copy.deepcopy(edits)) for i in range(20)]
    project = tmp_path / 'fine.lumen'
    model.save_project(project, tmp_path / 'a.arw', edits, snapshots)
    raw = json.loads(project.read_text(encoding='utf-8'))
    assert len(raw['rasters']) == 1 and 'raster' not in raw['edits']['masks'][0]
    assert project.stat().st_size < 2.5 * len(mask['raster'])
    _, loaded, snaps = model.load_project(project, include_snapshots=True)
    assert loaded['masks'][0]['raster'] == mask['raster'] and all(s['edits'] == loaded for s in snaps)
    raw['rasters'] = {k: v[:-40] + 'A' * 40 for k, v in raw['rasters'].items()}
    project.write_text(json.dumps(raw), encoding='utf-8')
    with pytest.raises(ValueError, match='蒙版'):
        model.load_project(project)
    # 1.6 projects keep their rasters inline and still open.
    old = model.recipe()
    old['version'] = 6
    old.pop('lut')
    old['masks'] = [dict(mask, raster=selection.encode(cv2.resize(truth, (800, 533))))]
    legacy = tmp_path / 'old.lumen'
    legacy.write_text(json.dumps(dict(application='LUMEN RAW', source='a.png', edits=old)), encoding='utf-8')
    assert model.load_project(legacy)[1]['masks'][0]['raster'] == old['masks'][0]['raster']


# ----------------------------------------------------------------------------- commands and window

@pytest.fixture
def scene(tmp_path):
    rgb, truth = branches(2400, 1600)
    path = tmp_path / 'branches.png'
    Image.fromarray((rgb * 255 + .5).astype(np.uint8)).save(path)
    return str(path), truth


@pytest.fixture
def fake_network(monkeypatch):
    """The sky network replaced by the true sky at network resolution (no model files needed)."""
    calls = []

    def automatic(rgb, kind, cuda=False):
        calls.append(rgb.shape)
        h, w = rgb.shape[:2]
        sky = (rgb[..., 2] > rgb[..., 0] + .2).astype(np.float32)
        net = cv2.resize(sky, (320, 320), interpolation=cv2.INTER_AREA)
        alpha = selection.refine(net, rgb)
        return (1 - alpha if kind == 'background' else alpha), 'test'
    monkeypatch.setattr(selection, 'automatic', automatic)
    return calls


def test_edit_apply_refines_ai_regions_on_the_original(tmp_path, scene, fake_network):
    from lumen.session import Session
    path, _ = scene
    session = Session(catalog.Catalog(tmp_path / 'catalog.jsonl'), engine.Backend('cpu'))
    run = lambda command, **args: commands.run(session, command, args)
    run('photo.open', path=path)
    run('edit.apply', changes=dict(masks=[dict(region='sky', adjustments=dict(exposure=-.5))]))
    fine = selection.decode(session.edits(session.current())['masks'][0]['raster'])
    assert fine.shape == (1600, 2400)
    run('edit.apply', changes=dict(masks=[dict(region='sky', adjustments=dict(exposure=-.3))]), fine=False)
    quick_mask = session.edits(session.current())['masks'][1]
    assert selection.decode(quick_mask['raster']).shape == (1067, 1600)
    result = run('mask.refine', index=1)
    assert result['refined'] == [dict(index=1, name=quick_mask['name'], size=[2400, 1600])]
    assert run('edit.undo')['path'] and selection.decode(session.edits(session.current())['masks'][1]['raster']).shape == (1067, 1600)
    with pytest.raises(commands.CommandError):
        run('mask.refine', index=7)
    from lumen import mcp
    assert 'refine_masks' in {t['name'] for t in mcp.tools()}
    assert 'fine' in commands.REGISTRY['edit.apply'].params


from test_ui import app, window, wait_until  # noqa: E402,F401


def test_window_replaces_quick_masks_with_fine_ones(window, scene, fake_network, monkeypatch):
    from lumen import auto_masks
    monkeypatch.setattr(auto_masks, 'save_setting', lambda *a: None)
    w = window
    path, truth = scene
    w.open_path(path)
    wait_until(lambda: w.rendered is not None and not w.render_running and w.source_path.endswith('branches.png'))
    w.fine_mask_check.setChecked(True)
    w.create_auto_mask('sky')
    wait_until(lambda: w.edits['masks'] and not w.selection_busy and
               selection.decode(w.edits['masks'][-1]['raster']).shape == (1600, 2400), timeout=60)
    assert '精细化' in w.statusBar().currentMessage()
    fine = selection.decode(w.edits['masks'][-1]['raster'])
    sky = cv2.erode((truth > .99).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    assert (fine[sky] > .95).mean() > .99
    assert not w.fine_mask_button.isEnabled()                     # already fine
    w.undo(-1)
    assert selection.decode(w.edits['masks'][-1]['raster']).shape[1] == 1600
    w.current_mask = len(w.edits['masks']) - 1
    w.refresh_mask_controls()
    assert w.fine_mask_button.isEnabled()
    w.refine_selected_mask()
    wait_until(lambda: not w.selection_busy and selection.decode(w.edits['masks'][-1]['raster']).shape[1] == 2400, timeout=60)
    # With the option off, only the quick mask is made.
    w.fine_mask_check.setChecked(False)
    w.create_auto_mask('subject')
    wait_until(lambda: len(w.edits['masks']) == 2 and not w.selection_busy, timeout=60)
    assert selection.decode(w.edits['masks'][-1]['raster']).shape[1] == 1600
