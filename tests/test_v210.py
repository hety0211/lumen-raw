"""2.1.0: AI removal (LaMa), landscape masks (EfficientViT-Seg) and picked objects (EfficientViT-SAM)."""
import copy
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from lumen import engine, model, nl_edit, remove, retouch, selection

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'assets' / 'models'
SAMPLES = Path(os.environ.get('LUMEN_SAMPLES', 'E:/LumenSamples'))
needs = lambda *names: pytest.mark.skipif(not all((MODELS / n).is_file() for n in names), reason='model files not fetched')


def textured(h=900, w=1350, seed=5):
    """Linear values: a soft gradient with fine texture and sensor-like noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([.18 + .08 * xx / w, .22 + .05 * yy / h, .30 - .06 * yy / h], axis=2)
    texture = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 3)[..., None] * .02
    noise = rng.normal(0, 1, (h, w, 3)).astype(np.float32) * np.sqrt(base) * .02
    return np.clip(base + texture + noise, 0, None).astype(np.float32)


def with_object(image, box=(560, 330, 700, 520)):
    x0, y0, x1, y1 = box
    out = image.copy()
    out[y0:y1, x0:x1] = [.9, .05, .04]
    return out


# ----------------------------------------------------------------------------- AI removal

@needs('inpaint-lama.onnx')
def test_removal_fills_from_surroundings_and_renders_the_same_everywhere():
    clean = textured()
    photo = with_object(clean)
    h, w = photo.shape[:2]
    stroke = dict(kind='remove', enabled=True, points=[[630 / (w - 1), 380 / (h - 1)], [630 / (w - 1), 470 / (h - 1)]],
                  radius=.1, opacity=100., feather=20.)
    coverage, area = remove.brush_coverage(stroke, h, w)
    assert coverage.max() == 1 and np.all(coverage[retouch.stroke(stroke, h, w)[1] > 0] == 1)    # all of the stroke
    patch = remove.fill(photo, [], coverage, area)
    op = dict(stroke, patch=patch)
    result = retouch.apply(photo, [op])
    inside = (slice(345, 505), slice(575, 685))
    assert np.abs(result[inside] - clean[inside]).mean(axis=(0, 1)).max() < .03     # the red block is gone
    assert np.abs(result[inside].mean(axis=(0, 1)) - clean[inside].mean(axis=(0, 1))).max() < .02
    x0, y0, x1, y1 = remove.window(patch, h, w)
    outside = np.ones((h, w), bool)
    outside[y0:y1, x0:x1] = False
    assert np.array_equal(result[outside], photo[outside])                            # nothing else moves
    # Blocks of the frame read exactly what the whole frame shows (tiles, strips).
    for rect in ((x0 + 13, y0 + 7, x1 - 20, y1 - 3), (0, 0, w, (y0 + y1) // 2), (x0 - 40, y0 - 40, x0 + 30, y0 + 30)):
        np.testing.assert_array_equal(retouch.apply(photo, [op], rect), result[rect[1]:rect[3], rect[0]:rect[2]])
    # A smaller frame (the preview) shows the same fill, scaled.
    half = cv2.resize(photo, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    small = retouch.apply(half, [op])
    expected = cv2.resize(result, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    assert np.abs(small[175:250, 290:340] - expected[175:250, 290:340]).mean() < .01
    # Hidden or transparent removals leave the photo alone; opacity blends.
    assert np.array_equal(retouch.apply(photo, [dict(op, enabled=False)]), photo)
    blend = retouch.apply(photo, [dict(op, opacity=50.)])
    np.testing.assert_allclose(blend[inside], (photo[inside] + result[inside]) / 2, atol=.02)


@needs('inpaint-lama.onnx')
def test_removal_patch_is_compact_and_large_fills_get_the_photos_grain():
    clean = textured(1800, 2700, seed=9)
    photo = with_object(clean, (900, 500, 1700, 1300))
    h, w = photo.shape[:2]
    coverage = np.zeros((900, 900), np.float32)
    coverage[30:870, 30:870] = 1
    patch = remove.fill(photo, [], coverage, (850, 450, 1750, 1350))
    colour, alpha = remove.decode(patch['png'], patch['scale'])
    assert max(alpha.shape) <= remove.WORK_EDGE and len(patch['png']) < 4_000_000
    assert remove.session().arena is False                       # its 2-3 GB go back after each fill
    assert patch['frame'] == [w, h] and min(patch['grain']) > 0
    result = retouch.apply(photo, [dict(kind='remove', enabled=True, opacity=100., patch=patch)])
    fill = result[700:1100, 1100:1500]
    around = photo[200:450, 300:800]
    residual = lambda x: (x - cv2.GaussianBlur(x, (0, 0), 1.)).std()
    # The fill was made at about a third of the size; without grain it would be far smoother.
    assert .6 < residual(fill) / residual(around) < 1.6
    assert np.isfinite(result).all()


@needs('inpaint-lama.onnx')
def test_removal_reads_the_retouching_before_it_and_object_coverage():
    photo = with_object(textured())
    h, w = photo.shape[:2]
    alpha = np.zeros((450, 675), np.float32)
    alpha[165:260, 280:350] = 1                                   # the object on a half-size preview
    coverage, area = remove.object_coverage(alpha, h, w)
    assert coverage.shape == (area[3] - area[1], area[2] - area[0])
    assert area[0] < 560 and area[2] > 700 and coverage.max() == 1 and coverage[0, 0] == 0
    # A clone stamp already over the object: the removal sees the stamped photo.
    clone = dict(kind='clone', enabled=True, points=[[630 / (w - 1), 425 / (h - 1)]], radius=.2, opacity=100., feather=0.,
                 offset=[-.3, 0])
    stamped = retouch.apply(photo, [clone])
    patch = remove.fill(photo, [clone], coverage, area)
    x0, y0, x1, y1 = remove.window(patch, h, w)
    region = (slice(y0, y1), slice(x0, x1))
    local = remove.composite(stamped[region], patch, 100, h, w)
    assert np.abs(local - stamped[region]).mean() < .02


def test_removal_recipes_validate_pack_and_reject_damaged_patches(tmp_path):
    alpha = np.zeros((40, 60), np.float32)
    alpha[5:35, 10:50] = 1
    patch = dict(box=[.4, .3, .5, .45], scale=.8, frame=[3000, 2000], grain=[.004, .003, .003], kept=.5, seed=7,
                 png=remove._encode(np.full((40, 60, 3), .2, np.float32), alpha, .8))
    edits = model.recipe()
    edits['retouch'] = [dict(kind='remove', enabled=True, points=[[.45, .4]], radius=.02, opacity=80., feather=0.,
                             picked=True, patch=patch)]
    checked = model.validate(copy.deepcopy(edits))
    assert checked['version'] == model.VERSION == 8 and checked['retouch'][0]['patch']['png'] == patch['png']
    colour, decoded = remove.decode(patch['png'], .8)
    assert np.allclose(colour[10, 20], .2, atol=1e-3) and decoded[0, 0] == 0
    # Projects and snapshots store each patch once.
    path = tmp_path / 'a.lumen'
    model.save_project(path, tmp_path / 'a.png', checked, [dict(name='x', edits=checked)])
    data = json.loads(path.read_text(encoding='utf-8'))
    assert len(data['rasters']) == 1 and 'png' not in data['edits']['retouch'][0]['patch']
    assert model.load_project(path)[1]['retouch'][0]['patch']['png'] == patch['png']
    for damage in (dict(png='AAAA'), dict(box=[.5, .3, .4, .45]), dict(box=[.4, .3, 1.2, .45]), dict(frame=[0, 10]),
                   dict(grain=[.1, .1]), dict(scale=float('nan'))):
        broken = copy.deepcopy(edits)
        broken['retouch'][0]['patch'].update(damage)
        with pytest.raises(ValueError):
            model.validate(broken)
    broken = copy.deepcopy(edits)
    del broken['retouch'][0]['patch']
    with pytest.raises(ValueError):
        model.validate(broken)
    # 2.0 projects (format 7) still open.
    old = model.recipe()
    old['version'] = 7
    assert model.validate(old)['retouch'] == []


def test_masks_see_the_retouched_photo():
    """A removed object is neither selected again nor left in a luminance mask as a ghost."""
    photo = textured(300, 450)
    photo[100:200, 200:260] = 4.                                  # a bright object
    h, w = photo.shape[:2]
    x0, y0, x1, y1 = 190, 90, 270, 210
    alpha = np.zeros((y1 - y0, x1 - x0), np.float32)
    alpha[5:115, 5:75] = 1
    fill = textured(300, 450, seed=8)[y0:y1, x0:x1]
    patch = dict(box=[x0 / w, y0 / h, x1 / w, y1 / h], scale=float(fill.max()), frame=[w, h], grain=[0., 0., 0.], kept=1.,
                 seed=1, png=remove._encode(fill, alpha, float(fill.max())))
    edits = model.recipe()
    edits['retouch'] = [dict(kind='remove', enabled=True, points=[[.5, .5]], radius=.01, opacity=100., feather=0., patch=patch)]
    edits = model.validate(edits)
    assert engine.selection_view(photo, edits)[150, 230].max() < .9
    bright = model.new_mask('luminance', 1)
    bright.update(luminance_range=[80., 100.], range_falloff=5.)
    bright['adjustments']['exposure'] = -2.
    masked = dict(edits, masks=[bright])
    whole = engine.process(photo, masked, apply_crop=False)
    np.testing.assert_allclose(whole[150, 230], engine.process(photo, edits, apply_crop=False)[150, 230], atol=1e-4)
    block = engine.process_region(photo, masked, rect=(180, 80, 300, 220))
    np.testing.assert_allclose(block, whole[80:220, 180:300], atol=2e-5)


# ----------------------------------------------------------------------------- landscape and object masks

@needs('landscape-seg.onnx')
@pytest.mark.skipif(not (SAMPLES / 'playraw').is_dir(), reason='sample RAW files not present')
def test_landscape_groups_on_real_photos():
    def develop(name):
        source, info = engine.load_image(SAMPLES / 'playraw' / name, 1200)
        edits = model.recipe()
        edits['develop'] = info.get('develop', edits['develop'])
        return engine.develop_view(source, edits)
    beach = selection.landscape(develop('Beach.CR2'))
    h = beach.shape[1]
    assert beach[0, int(h * .7):int(h * .8)].mean() > .6          # the sea in the middle band
    assert beach[4, int(h * .85):].mean() > .5                    # the sand at the bottom
    assert beach[:, :int(h * .4)].sum(axis=0).mean() < .3         # the sky is none of them
    taipei = selection.landscape(develop('Taipei101.ARW'))
    assert taipei[3, int(taipei.shape[1] * .85):].mean() > .6     # buildings along the bottom
    alpha, provider = selection.automatic(develop('Waterfall.ARW'), 'water')
    assert provider == 'CPUExecutionProvider' and .1 < (alpha > .5).mean() < .6


@needs('object-encoder.onnx', 'object-decoder.onnx')
def test_object_mask_follows_clicks():
    rng = np.random.default_rng(2)
    h, w = 800, 1200
    rgb = np.full((h, w, 3), .55, np.float32) + rng.normal(0, .02, (h, w, 3)).astype(np.float32)
    cv2.circle(rgb, (420, 400), 150, (.85, .15, .1), -1)
    cv2.rectangle(rgb, (760, 250), (1000, 600), (.1, .3, .8), -1)
    rgb = np.clip(rgb, 0, 1)
    disk = np.zeros((h, w), np.uint8)
    cv2.circle(disk, (420, 400), 150, 1, -1)
    alpha = selection.object_mask(rgb, [[420 / w, 400 / h, 1]], key='scene')
    picked = alpha > .5
    assert alpha.shape == (h, w) and (picked & (disk > 0)).sum() / (picked | (disk > 0)).sum() > .9
    both = selection.object_mask(rgb, [[420 / w, 400 / h, 1], [880 / w, 420 / h, 1]], key='scene') > .5
    assert both[420, 880] and both[400, 420]
    # A click labelled 0 keeps what it touches out of the mask.
    fewer = selection.object_mask(rgb, [[420 / w, 400 / h, 1], [880 / w, 420 / h, 0]], key='scene') > .5
    assert not fewer[420, 880] and fewer[400, 420]


def test_a_subtracting_click_takes_away_its_whole_object(monkeypatch):
    # Given only points the decoder can leave an island of the object under an Alt click; the
    # object that click alone selects is taken away too, unless it holds an adding click.
    h, w = 400, 600
    left = np.zeros((h, w), np.float32); left[100:300, 50:250] = 1
    right = np.zeros((h, w), np.float32); right[100:300, 350:550] = 1
    island = right.copy(); island[:, 420:] = 0

    def probability(embedding, points, h_, w_):
        if [p[2] for p in points] == [1, 0]:
            return np.maximum(left, island)
        return right if points[0][0] > .5 else left
    monkeypatch.setattr(selection, 'object_embedding', lambda rgb, key=None: None)
    monkeypatch.setattr(selection, '_object_probability', probability)
    rgb = np.full((h, w, 3), .5, np.float32)
    alpha = selection.object_mask(rgb, [[.25, .5, 1], [.75, .5, 0]])
    assert alpha[200, 150] > .5 and alpha[100:300, 350:550].max() < .5
    # An Alt click on a part of the selected object (its own mask holds the adding click): the
    # decoder's answer stands.
    alpha = selection.object_mask(rgb, [[.25, .5, 1], [.2, .5, 0]])
    assert alpha[200, 150] > .5 and alpha[200, 385] > .5


@needs('object-encoder.onnx', 'object-decoder.onnx')
@pytest.mark.skipif(not (SAMPLES / 'playraw' / 'HotBread.NEF').is_file(), reason='sample RAW files not present')
def test_alt_click_takes_away_a_whole_loaf():
    source, info = engine.load_image(SAMPLES / 'playraw' / 'HotBread.NEF', 1600)
    edits = model.recipe()
    edits['develop'] = info.get('develop', edits['develop'])
    rgb = engine.selection_view(source, edits)
    h, w = rgb.shape[:2]
    first, second, third = (.31, .6), (.51, .6), (.76, .62)
    # Shift on the second loaf selects all three; Alt on the third leaves the first two.
    picked = selection.object_mask(rgb, [[*first, 1], [*second, 1], [*third, 0]], key='bread') > .5
    assert picked[int(.48 * h):int(.74 * h), int(.64 * w):int(.9 * w)].mean() < .002
    assert all(picked[int(y * h), int(x * w)] for x, y in (first, second))
    # Alt on the middle loaf between two selected ones takes all of it away.
    outer = selection.object_mask(rgb, [[*first, 1], [*third, 1], [*second, 0]], key='bread') > .5
    assert outer[int(.45 * h):int(.72 * h), int(.45 * w):int(.57 * w)].mean() < .01
    assert all(outer[int(y * h), int(x * w)] for x, y in (first, third))


def test_mask_kinds_render_validate_and_reach_natural_language():
    edits = model.recipe()
    raster = selection.encode(np.tile(np.linspace(0, 1, 64, dtype=np.float32), (40, 1)))
    for kind in model.LANDSCAPE_KINDS + ('object',):
        mask = model.new_mask(kind, 1)
        mask.update(raster=raster, feather=3.)
        if kind == 'object':
            mask['points'] = [[.5, .5, 1], [.2, .2, 0]]
        edits['masks'].append(mask)
    checked = model.validate(copy.deepcopy(edits))
    assert [m['kind'] for m in checked['masks']] == list(model.LANDSCAPE_KINDS) + ['object']
    assert checked['masks'][-1]['points'] == [[.5, .5, 1], [.2, .2, 0]]
    alpha = engine.mask_alpha(checked['masks'][0], (40, 64, 3))
    assert alpha.shape == (40, 64) and alpha[0, -1] > .9 > .1 > alpha[0, 0]
    for points in ([], [[.5, .5, 2]], [[.5, .5]], [[.5, .5, 1]] * 33, [[1.5, .5, 1]]):
        broken = copy.deepcopy(edits)
        broken['masks'][-1]['points'] = points
        with pytest.raises(ValueError):
            model.validate(broken)
    for kind in model.LANDSCAPE_KINDS + ('object',):
        assert kind in selection.FINE_KINDS and kind in selection.NETWORK
    reply = dict(masks=[dict(region='水面', adjustments=dict(exposure=-.3)),
                        dict(region='trees', adjustments=dict(saturation=10)),
                        dict(region='building', adjustments=dict(clarity=12))])
    plan = nl_edit.plan(model.recipe(), reply)
    assert [kind for kind, _ in plan.regions] == ['water', 'vegetation', 'building']
    assert all(kind in nl_edit.REGIONS for kind in model.LANDSCAPE_KINDS)


def test_assets_manifest_and_bundle():
    from tools import bundle, fetch_assets
    data = json.loads((ROOT / 'assets/editing-assets.json').read_text(encoding='utf-8'))
    names = {Path(f['path']).name for f in data['files']}
    assert names == {'inpaint-lama.onnx', 'object-encoder.onnx', 'object-decoder.onnx', 'landscape-seg.onnx'}
    assert 'assets/editing-assets.json' in fetch_assets.DIRECT_MANIFESTS and 'editing-assets.json' in bundle.SKIP_ASSETS
    assert all(f['url'].startswith(fetch_assets.DIRECT_HOSTS) and len(f['sha256']) == 64 for f in data['files'])
    for name in ('LaMa-LICENSE.txt', 'EfficientViT-LICENSE.txt'):
        assert 'Apache License' in (ROOT / 'assets' / name).read_text(encoding='utf-8')
    present = [f for f in data['files'] if (ROOT / f['path']).is_file()]
    for record in present:
        assert (ROOT / record['path']).stat().st_size == record['bytes']


# ----------------------------------------------------------------------------- exposure curve ends

def test_black_and_white_points_stretch_the_whole_curve():
    from lumen import exposure_curve as ec
    a = dict(exposure=0., contrast=0., blacks=0., shadows=0., highlights=0., whites=0.)
    plain = [[0., 0.], [1., 1.]]
    assert ec.ends(a, plain) == ((0., 0.), (1., pytest.approx(1.)))
    # A flat (log-like) photo: pull the white point left and the black point right and up.
    points = ec.stretch(a, ec.stretch(a, plain, 'white', .7, 1.), 'black', .15, .05)
    (bx, by), (wx, wy) = ec.ends(a, points)
    assert abs(bx - .15) < .004 and abs(by - .05) < .002 and abs(wx - .7) < .004 and wy == pytest.approx(1)
    expected = .05 + np.clip((ec.SAMPLES - .15) / .55, 0, 1) * .95
    assert np.abs(ec.display(a, points, ec.SAMPLES) - expected).max() < .002
    checked = model.validate(dict(model.recipe(), tone_curve=points))
    assert checked['tone_curve'] == points
    # The shape between the ends is kept: a bent curve stretched keeps its bend.
    sliders, bent = ec.drag(a, .3, .45)
    before = ec.display(sliders, bent, ec.SAMPLES)
    moved = ec.stretch(sliders, bent, 'white', .8, .9)
    after = ec.display(sliders, moved, ec.SAMPLES)
    np.testing.assert_allclose(np.interp(ec.SAMPLES * .8, ec.SAMPLES, after) / .9, before, atol=.01)
    # Resetting both ends brings the plain curve back; the ends can not cross.
    back = ec.stretch(a, ec.stretch(a, points, 'black', 0., 0.), 'white', 1., 1.)
    assert np.abs(ec.display(a, back, ec.SAMPLES) - ec.SAMPLES).max() < .003
    crossed = ec.stretch(a, points, 'black', .95, .99)
    (bx, by), (wx, wy) = ec.ends(a, crossed)
    assert bx <= wx - ec.GAP + .004 and by <= wy - ec.GAP + .002
    # The renderer follows: tones below the black point become black.
    ramp = np.repeat(ec.linear(np.linspace(0, 1, 256)).astype(np.float32)[None, :, None], 3, axis=2) * np.ones((8, 1, 1), np.float32)
    edits = model.recipe()
    plain_out = engine.process(ramp, edits)
    crushed = engine.process(ramp, dict(edits, tone_curve=ec.stretch(a, plain, 'black', .3, 0.)))
    assert (crushed[4, :, 1] < 1 / 255).sum() > (plain_out[4, :, 1] < 1 / 255).sum() + 40


# ----------------------------------------------------------------------------- window

from test_ui import app, window, wait_until  # noqa: E402,F401


@pytest.fixture
def scene_window(window, tmp_path):
    w = window
    image = with_object(textured(800, 1200))
    path = tmp_path / 'scene.png'
    Image.fromarray((np.clip(engine.to_srgb(np.clip(image, 0, 1)), 0, 1) * 255 + .5).astype(np.uint8)).save(path)
    w.open_path(str(path))
    wait_until(lambda: w.rendered is not None and not w.render_running and w.source_path.endswith('scene.png'))
    return w


@needs('inpaint-lama.onnx')
def test_window_paint_removal_queue_list_and_undo(scene_window):
    w = scene_window
    w.tabs.setCurrentIndex(8)
    w.retouch_tool.setCurrentIndex(2)
    assert w.canvas.tool == 'remove' and 'LaMa' in w.retouch_tip.text()
    h, wd = 800, 1200
    first = dict(kind='remove', points=[[630 / wd, 380 / h], [630 / wd, 470 / h]], radius=.1, erase=False)
    second = dict(kind='remove', points=[[.15, .15]], radius=.03, erase=False)
    w.canvas.stroke_finished.emit(first)
    w.canvas.stroke_finished.emit(second)
    assert len(w.canvas.pending) == 2 and w.removal_running
    w.export()
    assert '移除' in w.statusBar().currentMessage() and not w.exporting
    wait_until(lambda: len(w.edits['retouch']) == 2 and not w.removal_running, timeout=120)
    assert not w.canvas.pending and all(op['kind'] == 'remove' and 'patch' in op for op in w.edits['retouch'])
    assert w.retouch_list.item(0).text().endswith('AI 移除')
    wait_until(lambda: not w.timer.isActive() and not w.render_running)
    shown = w.rendered[300:430, 470:620]
    assert shown[..., 0].mean() < shown[..., 2].mean() + .25                           # no red block in the preview
    w.undo(-1)
    assert len(w.edits['retouch']) == 1
    model.validate(copy.deepcopy(w.edits))


@needs('inpaint-lama.onnx', 'object-encoder.onnx', 'object-decoder.onnx')
def test_window_click_removal_and_object_masks(scene_window):
    w = scene_window
    w.tabs.setCurrentIndex(8)
    w.retouch_tool.setCurrentIndex(3)
    assert w.canvas.tool == 'object'
    w.canvas.object_picked.emit([630 / 1200, 425 / 800], 'new')
    wait_until(lambda: len(w.edits['retouch']) == 1 and not w.removal_running, timeout=120)
    assert w.edits['retouch'][0]['picked'] and w.retouch_list.item(0).text().endswith('AI 移除 · 物体')
    assert w.removal_overlay is None
    # Masks: a click makes an object mask, Shift adds a point, Alt takes one away.
    w.tabs.setCurrentIndex(4)
    w.object_button.setChecked(True)
    assert w.canvas.tool == 'object'
    w.edits['retouch'] = []
    w.changed()
    w.canvas.object_picked.emit([630 / 1200, 425 / 800], 'new')
    wait_until(lambda: w.edits['masks'] and not w.selection_busy, timeout=60)
    mask = w.edits['masks'][-1]
    assert mask['kind'] == 'object' and mask['points'] == [[630 / 1200, 425 / 800, 1]] and mask['name'].startswith('物体')
    w.canvas.object_picked.emit([.1, .1], 'subtract')
    wait_until(lambda: len(w.edits['masks'][-1]['points']) == 2 and not w.selection_busy, timeout=60)
    assert w.edits['masks'][-1]['points'][-1][2] == 0 and len(w.edits['masks']) == 1
    w.object_button.setChecked(False)
    assert w.canvas.tool != 'object'
    model.validate(copy.deepcopy(w.edits))


def test_window_landscape_buttons(scene_window, monkeypatch):
    from lumen import auto_masks
    monkeypatch.setattr(auto_masks, 'save_setting', lambda *a: None)   # never the user's settings.ini
    w = scene_window
    seen = []

    def automatic(rgb, kind, cuda=False):
        seen.append(kind)
        return np.ones(rgb.shape[:2], np.float32) * (np.arange(rgb.shape[1]) > rgb.shape[1] / 2), 'test'
    monkeypatch.setattr(selection, 'automatic', automatic)
    w.tabs.setCurrentIndex(4)
    w.fine_mask_check.setChecked(False)
    labels = {b.text(): b for b in w.ai_mask_buttons}
    for text in ('水面', '植被', '山体', '建筑', '地面'):
        labels[text].click()
        wait_until(lambda: not w.selection_busy, timeout=30)
    assert seen == list(model.LANDSCAPE_KINDS)
    assert [m['kind'] for m in w.edits['masks']] == list(model.LANDSCAPE_KINDS)


def test_window_drags_the_exposure_curve_ends(scene_window):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from lumen import exposure_curve as ec
    w = scene_window
    w.tabs.setCurrentIndex(0)
    curve = w.exposure_curve
    wait_until(lambda: curve.isVisible() and curve.histogram is not None)
    assert 0 < curve.histogram.max() <= 1
    sliders = copy.deepcopy(w.edits['adjustments'])
    history = len(w.history.items)
    white = curve.screen(1, 1).toPoint()
    QTest.mouseMove(curve, white)
    assert curve.hover_end == 'white'
    QTest.mousePress(curve, Qt.MouseButton.LeftButton, pos=white)
    target = curve.screen(.75, .95).toPoint()
    QTest.mouseMove(curve, target)
    QTest.mouseRelease(curve, Qt.MouseButton.LeftButton, pos=target)
    (_, _), (wx, wy) = ec.ends(w.edits['adjustments'], w.edits['tone_curve'])
    assert abs(wx - .75) < .02 and abs(wy - .95) < .02
    assert w.edits['adjustments'] == sliders and len(w.history.items) > history
    model.validate(copy.deepcopy(w.edits))
    QTest.mouseDClick(curve, Qt.MouseButton.LeftButton, pos=curve.screen(wx, wy).toPoint())
    assert np.abs(ec.display(sliders, w.edits['tone_curve'], ec.SAMPLES) - ec.display(sliders, None, ec.SAMPLES)).max() < .004


def test_finished_jobs_release_what_they_captured(app):
    """Each job's signals held its callbacks and they held the job: a cycle through Qt that kept
    every finished job's closures, e.g. the original-resolution image of each photo zoomed into."""
    import gc
    import weakref
    from PySide6.QtCore import QThreadPool
    from lumen.scheduler import JobScheduler

    class Big:
        def __init__(self):
            self.pixels = np.ones(2_000_000)
    scheduler, refs, results = JobScheduler(QThreadPool()), [], []
    for _ in range(3):
        big = Big()
        refs.append(weakref.ref(big))
        scheduler.submit(lambda b=big: float(b.pixels.sum()), lambda r, b=big: results.append(r), results.append)
        del big
        wait_until(lambda: len(results) == len(refs) and scheduler.active is None)
    wait_until(lambda: (gc.collect(), all(r() is None for r in refs))[1])
    assert results == [2_000_000.] * 3


def test_onnx_runtime_telemetry_is_turned_off_once(monkeypatch):
    """ONNX Runtime's own telemetry posted events from a thread that aborted the Mac app at quit."""
    import sys
    import types
    from lumen import performance
    calls = []
    fake = types.SimpleNamespace(disable_telemetry_events=lambda: calls.append(1), SessionOptions=lambda: None)
    monkeypatch.setitem(sys.modules, 'onnxruntime', fake)
    monkeypatch.setattr(performance, '_QUIET', False)
    assert performance.onnxruntime() is fake and performance.onnxruntime() is fake
    assert calls == [1] and os.environ['ORT_DISABLE_TELEMETRY'] == '1'
    import pathlib
    sources = [p.read_text(encoding='utf-8') for p in pathlib.Path(performance.__file__).parent.glob('*.py')]
    assert sum(s.count('import onnxruntime') for s in sources) == 1          # only through the helper


def test_library_shows_raw_and_jpeg_together_or_one_kind(app, tmp_path, monkeypatch):
    """2.1.0: RAW / JPEG check boxes in the library bar; at least one stays on; culling follows."""
    import shutil
    from test_v200 import library_window, photo  # noqa: F401
    from lumen import app as app_module, library_view
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setenv('LUMEN_LIBRARY', str(tmp_path / 'formats.db'))
    monkeypatch.setattr(QMessageBox, 'information', lambda *a, **k: None)
    trip = tmp_path / 'trip'
    for i in range(3):
        jpeg = photo(trip / f'DSC_{i}.JPG', seed=i, when=f'2026:10:05 08:30:{10 + 5 * i:02d}.00')
        shutil.copyfile(jpeg, trip / f'DSC_{i}.ARW')        # a RAW + JPEG pair (the copy only needs its name)
    shutil.copyfile(jpeg, trip / 'PANO.dng')
    shutil.copyfile(jpeg, trip / 'NOTE.png')
    assert [library_view.file_format(dict(path=p)) for p in ('a.ARW', 'b.jpeg', 'c.DNG', 'd.tif')] == ['raw', 'jpeg', 'raw', 'other']
    w = app_module.MainWindow()
    w.resize(1500, 950)
    w.show()
    try:
        raw, jpeg = w.format_checks['raw'], w.format_checks['jpeg']
        assert raw.isChecked() and jpeg.isChecked() and raw.isEnabled() and jpeg.isEnabled()
        w.import_folder(str(trip))
        wait_until(lambda: w.library_db.count() == 8 and len(w.library_model.rows) == 8, 30)
        names = lambda: sorted(r['name'] for r in w.library_model.rows)
        jpeg.setChecked(False)
        assert names() == ['DSC_0.ARW', 'DSC_1.ARW', 'DSC_2.ARW', 'NOTE.png', 'PANO.dng'] and not raw.isEnabled()
        assert '5' in w.library_summary.text()
        jpeg.setChecked(True)
        raw.setChecked(False)
        assert names() == ['DSC_0.JPG', 'DSC_1.JPG', 'DSC_2.JPG', 'NOTE.png'] and not jpeg.isEnabled() and raw.isEnabled()
        w.start_culling()
        wait_until(lambda: w.cull_window is not None)
        assert sorted(Path(p).name for p in w.cull_window.paths) == names()
        w.cull_window.close()
        raw.setChecked(True)
        assert len(w.library_model.rows) == 8 and raw.isEnabled() and jpeg.isEnabled()
    finally:
        if w.cull_window is not None:
            w.cull_window.close()
        w.saved_edits, w.saved_snapshots = w.edits, w.snapshots
        for document in w.documents.values():
            document['saved_edits'], document['saved_snapshots'] = document['edits'], document['snapshots']
        w.library_structure_dirty = False
        w.close()
        w.library_db.close()


# ----------------------------------------------------------------------------- review fixes (F1–F7)

def _pending_removal(w, monkeypatch):
    """Start an AI removal whose fill is held back; returns its (work, ready, fail)."""
    from lumen import nl_panel
    captured = []
    monkeypatch.setattr(remove, 'available', lambda: True)
    monkeypatch.setattr(nl_panel, 'run_async', lambda relay, work, ok, fail: captured.append((work, ok, fail)))
    w.start_removal(dict(kind='remove', points=[[.5, .5]], radius=.03, erase=False))
    assert w.removal_running and len(captured) == 1
    return captured[0]


def _patch():
    alpha = np.zeros((20, 30), np.float32)
    alpha[4:16, 6:24] = 1
    return dict(box=[.4, .4, .5, .5], scale=.5, frame=[360, 240], grain=[0., 0., 0.], kept=1., seed=3,
                png=remove._encode(np.full((20, 30, 3), .2, np.float32), alpha, .5))


def test_a_removal_finishing_after_switching_photos_lands_on_its_photo(window, monkeypatch, tmp_path):
    from lumen.scheduler import Activity as A
    w = window
    first = w.source_path
    _, ready, _ = _pending_removal(w, monkeypatch)
    assert w.work.active(A.REMOVAL) and w.canvas.pending
    other = tmp_path / 'other.png'
    other.write_bytes(Path(first).read_bytes())
    w.open_path(str(other))
    wait_until(lambda: not w.loading and w.source_path == str(other.resolve()) and not w.render_running)
    assert not w.canvas.pending                                       # its stroke is not drawn here
    ready(_patch())
    assert w.documents[first]['edits']['retouch'][0]['kind'] == 'remove' and not w.edits['retouch']
    assert w.film_item(first).text().endswith(' •') and not w.work.active(A.REMOVAL)
    w.open_path(first)
    wait_until(lambda: not w.loading and w.source_path == first and not w.render_running)
    assert len(w.edits['retouch']) == 1
    w.undo(-1)
    assert not w.edits['retouch']                                     # undoable on its photo


def test_close_save_and_every_export_wait_for_a_pending_removal(window, monkeypatch):
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import QDialog
    from lumen import batch_export
    w = window
    wait_until(lambda: not w.timer.isActive() and not w.render_running)
    _, ready, _ = _pending_removal(w, monkeypatch)
    monkeypatch.setattr(w, 'confirm_close_library', lambda: True)
    event = QCloseEvent()
    w.closeEvent(event)
    assert not event.isAccepted()
    entered = []
    monkeypatch.setattr(batch_export.BatchExportDialog, 'exec', lambda self: (entered.append(1), QDialog.DialogCode.Rejected)[1])
    w.filmstrip.selectAll()
    w.batch_export()
    w.make_timelapse(paths=[w.source_path, w.source_path])
    assert not entered and '移除' in w.statusBar().currentMessage()
    assert w.save() is False and w.save_album(path=str(Path(w.source_path).with_suffix('.lumenalbum'))) is False
    ready(_patch())
    assert not w.removal_running and w.edits['retouch']


def test_landscape_kinds_share_one_session(monkeypatch):
    from types import SimpleNamespace
    from lumen import compute
    calls = []
    monkeypatch.setattr(compute, 'session', lambda *a, **k: (calls.append(a[0]), SimpleNamespace())[1])
    selection.session.cache_clear()
    try:
        for kind in ('landscape',) + selection.LANDSCAPE:
            selection.session(kind)
            selection.session(kind, True)
        assert len(calls) == 1
    finally:
        selection.session.cache_clear()


def test_a_locked_library_is_never_treated_as_damaged(monkeypatch, tmp_path):
    import sqlite3
    from lumen import library_db
    path = tmp_path / 'library.db'
    first = library_db.LibraryDB(path)
    holder = sqlite3.connect(path, isolation_level=None)
    holder.execute('BEGIN IMMEDIATE')
    original = sqlite3.connect
    monkeypatch.setattr(sqlite3, 'connect', lambda *a, **kw: original(*a, **(kw | {'timeout': .05})))
    moved = []
    monkeypatch.setattr(library_db.os, 'replace', lambda *a: moved.append(a))
    try:
        with pytest.raises(sqlite3.OperationalError):
            library_db.LibraryDB(path)
        assert not moved and path.exists()
    finally:
        holder.execute('ROLLBACK')
        holder.close()
        first.close()


def test_a_damaged_library_is_set_aside_with_its_journal(tmp_path, monkeypatch):
    import sqlite3
    from lumen import library_db
    path = tmp_path / 'library.db'
    path.write_bytes(b'this is not an SQLite database' * 200)
    db = library_db.LibraryDB(path)                                   # a real damaged file: a new library
    try:
        assert db.count() == 0 and len(list(tmp_path.glob('library.damaged-*.db'))) == 1
    finally:
        db.close()
    # The journal files go along with it, untouched (SQLite may still hold committed pages there).
    other = tmp_path / 'other' / 'library.db'
    other.parent.mkdir()
    for suffix, data in (('', b'main'), ('-wal', b'journal'), ('-shm', b'shared')):
        Path(str(other) + suffix).write_bytes(data)
    error = sqlite3.DatabaseError('database disk image is malformed')
    error.sqlite_errorname = 'SQLITE_CORRUPT'
    assert library_db.damaged(error) and not library_db.damaged(sqlite3.OperationalError('database is locked'))
    db = library_db.LibraryDB.__new__(library_db.LibraryDB)
    db.path = other
    db._set_aside(error)
    aside = next(other.parent.glob('library.damaged-*.db'))
    assert [Path(str(aside) + s).read_bytes() for s in ('', '-wal', '-shm')] == [b'main', b'journal', b'shared']
    assert not other.exists() and not Path(str(other) + '-wal').exists()


def test_projects_over_the_limit_are_refused_before_the_file_is_replaced(tmp_path, monkeypatch):
    from lumen import library
    assert model.PROJECT_BYTES == library.ALBUM_BYTES == 512 * 2 ** 20
    edits = model.recipe()
    edits['retouch'] = [dict(kind='remove', enabled=True, points=[[.5, .5]], radius=.05, opacity=100., feather=0.,
                             patch=_patch() | dict(seed=i)) for i in range(3)]
    target = tmp_path / 'a.lumen'
    model.save_project(target, tmp_path / 'a.png', model.recipe())
    before = target.read_bytes()
    monkeypatch.setattr(model, 'PROJECT_BYTES', len(before) + 1000)
    with pytest.raises(ValueError, match='MB'):
        model.save_project(target, tmp_path / 'a.png', edits)
    assert target.read_bytes() == before                              # the old project stays readable
    monkeypatch.setattr(model, 'PROJECT_BYTES', 512 * 2 ** 20)
    model.save_project(target, tmp_path / 'a.png', edits)
    assert len(model.load_project(target)[1]['retouch']) == 3


def test_photo_paths_in_projects_and_albums_work_on_both_systems(tmp_path, monkeypatch):
    folder = tmp_path / 'trip'
    (folder / 'RAW').mkdir(parents=True)
    photo = folder / 'RAW' / 'IMG.ARW'
    photo.write_bytes(b'x')
    assert model.stored_source(photo, folder) == 'RAW/IMG.ARW'
    model.save_project(folder / 'a.lumen', photo, model.recipe())
    assert json.loads((folder / 'a.lumen').read_text(encoding='utf-8'))['source'] == 'RAW/IMG.ARW'
    assert model.load_project(folder / 'a.lumen')[0] == str(photo.resolve())
    # A project or album written by an older Windows version (backslashes), read on macOS / Linux.
    monkeypatch.setattr(os, 'sep', '/')
    assert model.source_path('RAW\\IMG.ARW', folder) == photo.resolve()
    assert model.source_path('../trip/RAW/IMG.ARW', folder) == photo.resolve()


def test_a_refused_keychain_read_never_replaces_the_stored_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(nl_edit, 'KEYCHAIN', True)
    monkeypatch.setattr(nl_edit, '_keychain_cache', None)
    item = {'value': json.dumps({'openai': 'sk-old'}).encode(), 'refuse': True}

    def read():
        if item['refuse']:
            raise OSError('User denied macOS Keychain access')
        return item['value']
    written = []

    def write(value):
        written.append(json.loads(value))
        item.update(value=value)
    monkeypatch.setattr(nl_edit, '_keychain_read', read)
    monkeypatch.setattr(nl_edit, '_keychain_write', write)
    settings = nl_edit.default_settings()
    settings['providers']['openai']['key'] = 'keychain:'
    path = tmp_path / 'natural-language.json'
    path.write_text(json.dumps(settings), encoding='utf-8')
    loaded = nl_edit.load_settings(path)
    assert loaded['providers']['openai']['key'] == '' and loaded['keychain_unread'] == ['openai']
    loaded['providers']['anthropic']['key'] = 'sk-new'
    nl_edit.save_settings(loaded, path)
    assert not written                                                # the item it could not read is kept
    stored = json.loads(path.read_text(encoding='utf-8'))['providers']
    assert stored['openai']['key'] == 'keychain:' and stored['anthropic']['key'] not in ('', 'keychain:')
    item['refuse'] = False                                            # access allowed later: merged
    again = nl_edit.load_settings(path)
    assert again['providers']['openai']['key'] == 'sk-old' and again['providers']['anthropic']['key'] == 'sk-new'
    nl_edit.save_settings(again, path)
    assert written[-1] == {'openai': 'sk-old', 'anthropic': 'sk-new'}


def test_source_launchers_install_again_when_the_requirements_change():
    root = Path(__file__).resolve().parents[1]
    cmd = (root / 'run-source.cmd').read_bytes().decode('ascii')
    command = (root / 'run-source.command').read_text(encoding='utf-8')
    assert 'onnxruntime, av"' in cmd and 'lumen-requirements.txt' in cmd and '\r\n' in cmd
    assert 'Metal, av"' in command and 'cmp -s requirements-macos.txt .venv-macos/lumen-requirements.txt' in command
