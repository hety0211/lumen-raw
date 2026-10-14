"""Export colour spaces (sRGB, Display P3, Adobe RGB) and the JPEG file-size limit."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms

from lumen import color, engine, icc, model
from test_v14_viewport import RECTS, frame, recipes


def scene():
    """Ordinary colours on the left, scene reds and greens beyond sRGB on the right."""
    rng = np.random.default_rng(1)
    x = np.zeros((96, 128, 3), np.float32)
    x[:, :64] = rng.uniform(.04, .5, (96, 64, 3))
    x[:48, 64:] = np.float32([.9, -.05, -.04]) * rng.uniform(.3, 1, (48, 64, 1))
    x[48:, 64:] = np.float32([-.06, .7, .1]) * rng.uniform(.3, 1, (48, 64, 1))
    return x


def edits(version=2, **adjustments):
    e = model.recipe()
    e['process'] = version
    e['develop'] = dict(mode='linear', curve=[[0., 0.], [1., 1.]])
    e['adjustments'].update(adjustments)
    return e


def test_srgb_output_is_unchanged():
    x, e = scene(), edits(saturation=20, shadows=30)
    np.testing.assert_array_equal(engine.process(x, e), engine.process(x, e, space='srgb'))
    rgb = np.float32([[1.2, -.1, .3], [.2, .4, .6]])
    np.testing.assert_array_equal(color.gamut_map(rgb), color.gamut_map(rgb, 'srgb'))


@pytest.mark.parametrize('key', ['p3', 'adobe-rgb'])
def test_wide_spaces_keep_colours_beyond_srgb_and_the_look_of_the_rest(key):
    x, e = scene(), edits()
    srgb, wide = engine.process(x, e), engine.process(x, e, space=key)
    # Colours inside sRGB: exactly the sRGB render, re-encoded.
    np.testing.assert_allclose(wide[:, :64], engine.convert_display(srgb, key)[:, :64], atol=1e-5)
    # Beyond sRGB: more chroma than the sRGB render could keep.
    linear = color.to_linear_srgb(wide[:, 64:], key)
    chroma = lambda rgb: np.hypot(*color.to_oklab(rgb)[..., 1:].transpose(2, 0, 1)).mean()
    assert linear.min() < -.005
    assert chroma(linear) > chroma(color.srgb_to_linear(srgb[:, 64:])) + .005
    # Display-space tools (clarity) change the look of a photograph's colours by well under 1/255.
    import cv2
    smooth = cv2.GaussianBlur(np.random.default_rng(2).uniform(.03, .6, (120, 160, 3)).astype(np.float32), (0, 0), 2)
    e = edits(clarity=40, saturation=20, highlights=-40)
    difference = np.abs(engine.process(smooth, e, space=key) - engine.convert_display(engine.process(smooth, e), key))
    assert difference.mean() < .5 / 255
    # Process 1 renders in sRGB and is re-encoded.
    e = edits(version=1, saturation=20)
    np.testing.assert_allclose(engine.process(x, e, space=key), engine.convert_display(engine.process(x, e), key),
                               atol=1e-6)


@pytest.mark.parametrize('name', ['spatial', 'masked'])
def test_wide_block_render_matches_the_whole_frame(name):
    source, e = frame(), recipes()[name]
    whole = engine.process(source, e, engine.Backend('cpu'), apply_crop=False, detail_scale=.7, space='p3')
    for x0, y0, x1, y1 in RECTS:
        block = engine.process_region(source, e, engine.Backend('cpu'), (x0, y0, x1, y1), detail_scale=.7, space='p3')
        difference = np.abs(block - whole[y0:y1, x0:x1])
        assert difference.max() < 2e-4 and difference.mean() < 1e-5


@pytest.mark.parametrize('key', ['p3', 'adobe-rgb'])
def test_generated_icc_profiles_convert_like_the_matrices(key):
    data = icc.profile(key)
    profile = icc.load(data)
    assert ImageCms.getProfileDescription(profile).strip() == icc.DESCRIPTIONS[key]
    assert icc.identify(data) == key and icc.identify(icc.profile()) is None
    assert icc.profile() == ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    patch = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB'))
    converted = np.asarray(ImageCms.profileToProfile(Image.fromarray(patch), srgb, profile, outputMode='RGB'))
    ours = color.encode(color.srgb_to_linear(patch / 255) @ color.space(key).from_srgb.T, key) * 255
    assert np.abs(converted - np.round(np.clip(ours, 0, 255))).max() <= 1


@pytest.mark.parametrize('key', ['p3', 'adobe-rgb'])
def test_wide_exports_embed_the_profile_and_reopen_with_their_colours(tmp_path, key):
    x = scene()
    rendered = engine.process(x, edits(), space=key)
    linear = color.to_linear_srgb(rendered, key)
    for ext in ('.tif', '.dng', '.png', '.jpg'):
        path = tmp_path / f'成片{ext}'
        assert engine.export_image(path, rendered, 95, space=key).notes == ()
        if ext != '.dng':
            assert icc.identify(engine.embedded_profile(path)) == key
        back, info = engine.load_image(path, preview_limit=None)
        tolerance = {'.tif': 1e-4, '.dng': 1e-4, '.png': 1e-2}.get(ext, .25)
        assert np.abs(back - linear).max() < tolerance, ext
        if ext in ('.tif', '.dng'):
            assert back.min() < -.005                        # colours beyond sRGB survive the round trip
        if ext != '.dng':
            assert color.space(key).label in info['note']


def test_jpeg_size_limit_lowers_the_quality_until_the_file_fits(tmp_path):
    rng = np.random.default_rng(3)
    picture = np.clip(.5 + .2 * rng.standard_normal((600, 900, 3)), 0, 1).astype(np.float32)
    free = engine.export_image(tmp_path / 'free.jpg', picture, 95)
    assert free.quality == 95 and free.size > 300 * 1024
    limited = engine.export_image(tmp_path / 'limited.jpg', picture, 95, max_kb=300)
    assert limited.notes == () and 10 <= limited.quality < 95
    assert (tmp_path / 'limited.jpg').stat().st_size == limited.size <= 300 * 1024
    # Quality one step higher would not have fitted.
    larger = engine.export_image(tmp_path / 'next.jpg', picture, limited.quality + 1)
    assert larger.size > 300 * 1024
    impossible = engine.export_image(tmp_path / 'tiny.jpg', picture, 95, max_kb=5)
    assert impossible.quality == engine.JPEG_MIN_QUALITY and len(impossible.notes) == 1
    assert '5 KB' in impossible.notes[0] and (tmp_path / 'tiny.jpg').exists()
    # Other formats ignore the limit.
    assert engine.export_image(tmp_path / 'free.png', picture, max_kb=5).notes == ()


def test_size_limit_counts_the_copied_metadata(tmp_path):
    from test_v161 import _tagged_original
    from lumen.white_balance import exiftool
    if exiftool() is None:
        pytest.skip('ExifTool is not installed')
    original = _tagged_original(tmp_path)
    rng = np.random.default_rng(4)
    picture = np.clip(.5 + .2 * rng.standard_normal((400, 600, 3)), 0, 1).astype(np.float32)
    plain = engine.export_image(tmp_path / 'plain.jpg', picture, 95, max_kb=120)
    tagged = engine.export_image(tmp_path / 'tagged.jpg', picture, 95, max_kb=120, metadata=dict(source=original))
    assert tagged.notes == () and tagged.size <= 120 * 1024 and tagged.quality <= plain.quality


def test_batch_export_takes_the_space_and_the_limit(tmp_path):
    from lumen import batch_export
    path = tmp_path / '原片.png'
    rng = np.random.default_rng(5)
    Image.fromarray((rng.uniform(0, 1, (300, 450, 3)) * 255).astype(np.uint8)).save(path)
    out = tmp_path / 'out'
    out.mkdir()

    class Never:
        def is_set(self):
            return False
    options = dict(extension='.jpg', long_edge=None, quality=95, folder=str(out), metadata=False, space='p3', max_kb=40)
    (_, target, note), = batch_export.run([(str(path), model.recipe(), False)], options, engine.Backend('cpu'),
                                          lambda *_: None, Never())
    assert note == '' and icc.identify(engine.embedded_profile(target)) == 'p3'
    assert Path(target).stat().st_size <= 40 * 1024


# ----------------------------------------------------------------------------- dialogs

from test_ui import app, window, wait_until  # noqa: E402,F401


def test_export_dialogs_offer_space_and_size_limit(window):
    from lumen.batch_export import BatchExportDialog
    from lumen.enhance_dialog import ExportDialog
    w = window
    for dialog in (ExportDialog(w, w.source_path), BatchExportDialog(w, 2, '.')):
        output = dialog.output
        assert [output.space.itemText(i) for i in range(output.space.count())] == ['sRGB', 'Display P3', 'Adobe RGB (1998)']
        dialog.format.setCurrentIndex(0)
        assert output.limit.isEnabled() and dialog.quality.isEnabled()
        dialog.format.setCurrentIndex(2)                          # TIFF
        assert not output.limit.isEnabled() and not dialog.quality.isEnabled()
        output.space.setCurrentIndex(1)
        output.limit.setValue(800)
        assert output.options() == dict(space='p3', max_kb=800)
        output.limit.setValue(0)
        assert output.options()['max_kb'] is None
        dialog.reject()


# ----------------------------------------------------------------------------- follow-ups

def test_limited_jpeg_subsamples_chroma_only_when_the_limit_binds(tmp_path):
    from PIL import JpegImagePlugin
    rng = np.random.default_rng(6)
    picture = np.clip(.5 + .2 * rng.standard_normal((500, 700, 3)), 0, 1).astype(np.float32)
    plain = engine.export_image(tmp_path / 'plain.jpg', picture, 90)
    roomy = engine.export_image(tmp_path / 'roomy.jpg', picture, 90, max_kb=plain.size // 1024 + 50)
    assert roomy.quality == 90 and (tmp_path / 'roomy.jpg').read_bytes() == (tmp_path / 'plain.jpg').read_bytes()
    tight = engine.export_image(tmp_path / 'tight.jpg', picture, 90, max_kb=plain.size // 2048)
    with Image.open(tmp_path / 'tight.jpg') as im:
        assert JpegImagePlugin.get_sampling(im) == 2            # 4:2:0
    with Image.open(tmp_path / 'plain.jpg') as im:
        assert JpegImagePlugin.get_sampling(im) == 0            # 4:4:4
    assert tight.notes == () and tight.size <= plain.size // 2


def test_watermark_borders_keep_their_colour_in_wide_spaces():
    from lumen import watermark
    settings = dict(watermark.defaults(), enabled=True, preset='travel', camera_logo='', lens_logo='')
    picture = np.full((60, 80, 3), .5, np.float32)
    srgb = watermark.apply(picture, settings, {})
    for key in ('p3', 'adobe-rgb'):
        wide = watermark.apply(picture, settings, {}, key)
        np.testing.assert_allclose(wide[-2, -2], color.reencode(srgb[-2:, -2:], key)[-1, -1], atol=1e-6)
        assert np.abs(wide[-2, -2] - srgb[-2, -2]).max() > 1e-3


def test_dng_names_the_current_version(tmp_path):
    import tifffile
    from lumen import __version__
    engine.export_image(tmp_path / 'x.dng', np.full((8, 8, 3), .5, np.float32))
    with tifffile.TiffFile(tmp_path / 'x.dng') as f:
        assert f.pages[0].tags[305].value == f'LUMEN RAW {__version__}'


def test_command_results_keep_the_metadata_error_field():
    from lumen import commands
    from lumen.i18n import tr
    text = tr('未能写入原片元数据：') + 'ExifTool missing'
    assert commands._notes(text) == dict(note=text, metadata_error='ExifTool missing')
    assert commands._notes('over the limit') == dict(note='over the limit') and commands._notes('') == {}


def test_single_export_offers_a_long_edge(window, tmp_path):
    from lumen.enhance_dialog import ExportDialog
    w = window
    dialog = ExportDialog(w, w.source_path)
    assert dialog.long_edge() is None
    dialog.size.setCurrentIndex(3)
    assert dialog.long_edge() == 1600
    dialog.reject()
    assert ExportDialog(w, w.source_path, True).long_edge() is None    # enhance mode has its own scale
    output = tmp_path / 'small.png'
    w.start_export(str(output), long_edge=200, metadata=False)
    wait_until(lambda: not w.exporting)
    with Image.open(output) as im:
        assert max(im.size) == 200


def test_disabled_check_boxes_look_disabled(window):
    from lumen.widgets import MetadataOptions
    options = MetadataOptions()
    options.keep.setChecked(True)
    options.location.setChecked(True)
    options.resize(320, 60)
    options.show()
    _process_events()
    enabled = options.location.grab().toImage().pixelColor(8, options.location.height() // 2)
    options.keep.setChecked(False)
    _process_events()
    disabled = options.location.grab().toImage().pixelColor(8, options.location.height() // 2)
    assert not options.location.isEnabled() and enabled != disabled
    options.deleteLater()


def _process_events():
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()


def test_cli_render_works_without_optional_options(tmp_path):
    # lumen-cli passes unset options (--size, --max-kb) as None; 1.6.0 – 1.6.1 rejected them.
    from lumen import cli, commands
    source = tmp_path / 'photo.png'
    Image.new('RGB', (120, 80), (120, 140, 160)).save(source)
    output = tmp_path / 'out.jpg'
    assert cli.main(['render', str(source), str(output), '--space', 'p3']) == 0
    assert output.exists() and icc.identify(engine.embedded_profile(output)) == 'p3'
    cmd = commands.REGISTRY['photo.export']
    assert 'long_edge' not in commands.arguments(cmd, dict(output='x.jpg', long_edge=None))
    with pytest.raises(commands.CommandError):
        commands.arguments(cmd, dict(output=None))
