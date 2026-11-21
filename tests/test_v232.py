"""2.3.2: Sony ILCE-7M5 (A7 V) RAW opened at the wrong size and aspect ratio.

Sony's lossless compressed RAW stores the sensor in 512-pixel tiles; the bundled LibRaw 0.22 has
no size entry for the A7 V (nor the A1 II, RX1R III), so it showed the whole tile grid — 7168 × 5120
instead of 7008 × 4672, black bands at the right and bottom.  The file's DefaultCrop is used for
these.  The A7 V's lossy "Compressed RAW 2" (ARW 6.0) is a format LibRaw cannot open at all; those
open from the camera's full-size embedded JPEG.  Samples from raw.pixls.us (CC0) in
``LUMEN_SAMPLES/A7M5``.
"""
import io
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lumen import engine, previews

SAMPLES = Path(os.environ.get('LUMEN_SAMPLES', 'E:/LumenSamples'))
A7V = SAMPLES / 'A7M5'
LOSSLESS = [(A7V / 'full_compressed_lossless.ARW', (7008, 4672)), (A7V / 'apsc_compressed_lossless.ARW', (4608, 3072))]
LOSSY = [(A7V / 'full_compressed.ARW', (7008, 4672)), (A7V / 'full_compressed_HQ.ARW', (7008, 4672)),
         (A7V / 'apsc_compressed.ARW', (4608, 3072)), (A7V / 'apcs_compressed_hq.ARW', (4608, 3072))]
A7RV = SAMPLES / '7RM5-LosslessCompressedLarge.ARW'


def need(path):
    if not path.exists():
        pytest.skip(f'{path.name} not available')


def sizes(width, height, crop, margins=(0, 0), flip=0):
    x, y, w, h = crop
    return SimpleNamespace(sizes=SimpleNamespace(width=width, height=height, left_margin=margins[0],
                                                 top_margin=margins[1], crop_left_margin=x, crop_top_margin=y,
                                                 crop_width=w, crop_height=h, flip=flip))


def test_picture_area_only_for_sony_tile_padding():
    a7v = sizes(7168, 5120, (12, 8, 7008, 4672))
    assert engine.picture_area(a7v, 'x.ARW') == (12, 8, 7008, 4672)
    assert engine.raw_size(a7v, 'x.arw') == (7008, 4672)
    # Other makes, bodies LibRaw trimmed itself, a few edge pixels only, invalid or missing crops: unchanged.
    assert engine.picture_area(a7v, 'x.NEF') is None
    assert engine.picture_area(sizes(9566, 6374, (32, 20, 9504, 6336)), 'x.ARW') is None      # A7R V
    assert engine.picture_area(sizes(4608, 3072, (0, 0, 4592, 3056)), 'x.ARW') is None        # NEX-5: 0.35 %
    assert engine.picture_area(sizes(6024, 4024, (348, 12, 5328, 4000)), 'x.ARW') is None     # not tiled
    assert engine.picture_area(sizes(7168, 5120, (0, 0, 0, 0)), 'x.ARW') is None
    assert engine.picture_area(sizes(7168, 5120, (200, 8, 7008, 4672)), 'x.ARW') is None
    assert engine.picture_area(sizes(7168, 5120, (12, 8, 7008, 4672), margins=(8, 0)), 'x.ARW') is None
    assert engine.raw_size(sizes(7168, 5120, (12, 8, 7008, 4672), flip=6), 'x.ARW') == (4672, 7008)


@pytest.mark.parametrize('flip', [0, 3, 5, 6])
def test_crop_follows_the_orientation(flip):
    sensor = np.arange(40 * 64).reshape(40, 64)
    area = (6, 4, 50, 30)
    expected = previews.orient(sensor[4:34, 6:56], flip)
    np.testing.assert_array_equal(engine.crop_picture(previews.orient(sensor, flip), area, flip), expected)
    half = sensor[::2, ::2]
    np.testing.assert_array_equal(engine.crop_picture(previews.orient(half, flip), area, flip, 2),
                                  previews.orient(half[2:17, 3:28], flip))
    assert engine.crop_picture(sensor, None, flip) is sensor


def test_crop_of_rotated_libraw_decodes():
    """Portrait shots: LibRaw's own rotation and ``crop_picture`` agree for every orientation."""
    path = LOSSLESS[1][0]
    need(path)
    import rawpy

    def decode(flip):
        with rawpy.imread(str(path)) as raw:
            raw.unpack()
            area = engine.picture_area(raw, path)
            return raw.postprocess(half_size=True, user_flip=flip), area

    upright, area = decode(0)
    picture = engine.crop_picture(upright, area, 0, 2)
    for flip in (3, 5, 6):
        rotated, _ = decode(flip)
        np.testing.assert_array_equal(engine.crop_picture(rotated, area, flip, 2), previews.orient(picture, flip))


@pytest.mark.parametrize('path, size', LOSSLESS)
def test_lossless_a7v_decodes_at_the_camera_size(path, size):
    need(path)
    preview, info = engine.load_image(path, 1600)
    assert (info['width'], info['height']) == size and info['raw']
    assert preview.shape[1] == 1600 and abs(preview.shape[1] / preview.shape[0] - 1.5) < .01
    # No black bands left at the right or bottom.
    assert preview[:, -4:].mean() > .005 and preview[-4:].mean() > .005
    full, _ = engine.load_image(path, 0)
    assert full.shape[:2] == size[::-1]
    # Library / culling view and sizes read without decoding agree.
    from lumen import timelapse
    assert timelapse.photo_size(path) == size
    view, full_size = previews.embedded_view(path, 800)
    assert full_size and abs(view.shape[1] / view.shape[0] - 1.5) < .01


def test_lossless_a7v_lines_up_with_the_camera_jpeg():
    path = LOSSLESS[0][0]
    need(path)
    import cv2
    import rawpy
    from PIL import Image
    rgb, info = engine.load_image(path, 1600)
    with rawpy.imread(str(path)) as raw:
        jpeg = np.asarray(Image.open(io.BytesIO(raw.extract_thumb().data)).convert('L'), np.float64)
    ours = engine.develop_view(rgb, dict(develop=info['develop'], process=2)).mean(axis=2)
    camera = cv2.resize(jpeg, (ours.shape[1], ours.shape[0]), interpolation=cv2.INTER_AREA)
    (dx, dy), response = cv2.phaseCorrelate(camera, ours.astype(np.float64) * 255)
    assert abs(dx) < .5 and abs(dy) < .5 and response > .8


@pytest.mark.parametrize('path, size', LOSSY)
def test_compressed_raw_2_opens_from_the_embedded_jpeg(path, size):
    need(path)
    import rawpy
    with pytest.raises(rawpy.LibRawFileUnsupportedError):
        rawpy.imread(str(path))
    preview, info = engine.load_image(path, 1600)
    assert info['embedded'] and not info['raw']
    assert (info['width'], info['height']) == size and preview.shape[:2] == (1067, 1600)
    assert info['photo'] and info.get('camera_model') == 'ILCE-7M5'
    view, full_size = previews.embedded_view(path, 800)
    assert full_size and view.shape[:2] == (533, 800)
    from lumen import timelapse
    assert timelapse.photo_size(path) == size


def test_bodies_libraw_knows_keep_their_frame():
    need(A7RV)
    _, info = engine.load_image(A7RV, 1600)
    assert (info['width'], info['height']) == (9566, 6374)
