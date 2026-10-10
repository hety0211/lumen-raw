"""1.5.2: Nikon Z50 II / Z5 II NEFs, which the bundled LibRaw 0.22 does not list.

Lossless files decode but LibRaw has no colour matrix for these bodies; "High Efficiency"
files fail with a data error instead of "unsupported", so 1.4.1's embedded-JPEG fallback
never ran. Samples from raw.pixls.us (CC0) in ``LUMEN_SAMPLES/Z50_2`` and ``Z5_2``.
"""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lumen import engine, white_balance

SAMPLES = Path(os.environ.get('LUMEN_SAMPLES', 'E:/LumenSamples'))
Z50II = SAMPLES / 'Z50_2'
Z50II_LOSSLESS = Z50II / 'Nikon_Z50_2_DSC_0060_DX_Lossless.NEF'
Z5II_LOSSLESS = SAMPLES / 'Z5_2' / 'Nikon_Z5_2_Lossless_compression_FX_1.NEF'
Z8_LOSSLESS = SAMPLES / 'Nikon_Z8_raw_14_bit_lossless_compression.NEF'
HIGH_EFFICIENCY = [Z50II / 'Nikon_Z50_2_DSC_0061_DX_HE_Star.NEF', Z50II / 'Nikon_Z50_2_DSC_0062_DX_HE.NEF',
                   Z50II / 'Nikon_Z50_2_DSC_0064_1x1_HE_Star.NEF', Z50II / 'Nikon_Z50_2_DSC_0068_16x9_HE.NEF',
                   SAMPLES / 'Z5_2' / 'Nikon_Z5_2_High_Efficiency_Star_FX_1.NEF']


def need(path):
    if not path.exists():
        pytest.skip(f'{path.name} not available')


def test_camera_matrix_only_fills_in_where_libraw_has_none():
    known = SimpleNamespace(rgb_xyz_matrix=np.eye(4, 3))
    unknown = SimpleNamespace(rgb_xyz_matrix=np.zeros((4, 3)))
    assert engine.camera_matrix(known, 'NIKON Z50_2') is None
    assert engine.camera_matrix(unknown, 'NIKON Z9_9') is None
    matrix = engine.camera_matrix(unknown, 'NIKON Z50_2')
    np.testing.assert_allclose(matrix[0], [1.164, -.4829, -.1079])
    # White (any white-balanced neutral) stays neutral.
    rgb = np.full((3, 5, 3), .4, np.float32)
    np.testing.assert_allclose(engine.camera_to_srgb(rgb, matrix), .4, atol=1e-6)


@pytest.mark.parametrize('sample', [Z8_LOSSLESS, SAMPLES / '7RM5-LosslessCompressedLarge.ARW'])
def test_camera_to_srgb_matches_libraws_own_conversion(sample):
    import rawpy
    need(sample)
    with rawpy.imread(str(sample)) as raw:
        options = dict(use_camera_wb=True, no_auto_bright=True, output_bps=16, gamma=(1, 1), half_size=True,
                       user_flip=None, highlight_mode=rawpy.HighlightMode.Blend)
        reference = raw.postprocess(output_color=rawpy.ColorSpace.sRGB, **options).astype(np.float32) / 65535
        camera = raw.postprocess(output_color=rawpy.ColorSpace.raw, **options).astype(np.float32) / 65535
        engine.camera_to_srgb(camera, np.asarray(raw.rgb_xyz_matrix[:3], float))
    assert np.abs(camera - reference).max() < 4 / 65535


def _chromaticity(rgb):
    import cv2
    small = cv2.resize(np.ascontiguousarray(rgb, np.float32), (240, 160), interpolation=cv2.INTER_AREA)
    total = small.sum(-1, keepdims=True)
    keep = (total[..., 0] > .05) & (small.max(-1) < .95)
    return (small / np.maximum(total, 1e-6))[keep][:, [0, 2]], keep


@pytest.mark.parametrize('sample,body', [(Z50II_LOSSLESS, 'NIKON Z50_2'), (Z5II_LOSSLESS, 'NIKON Z5_2')])
def test_lossless_nef_decodes_as_raw_in_camera_colours(sample, body):
    import rawpy
    from lumen import previews
    need(sample)
    rgb, info = engine.load_image(sample, 240)
    assert info['raw'] and not info.get('embedded') and info['photo']['body'] == body
    assert 4000 < info['white_balance']['camera_kelvin'] < 8000  # estimated from the supplied matrix
    with rawpy.imread(str(sample)) as raw:
        jpeg = engine.to_linear(previews.libraw_preview(raw).astype(np.float32) / 255)
        plain = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16, gamma=(1, 1), half_size=True,
                                user_flip=None).astype(np.float32) / 65535
    # Chromaticities are compared, so the camera's tone curve does not matter: with the matrix the
    # RAW is much closer to the camera's own JPEG than LibRaw's unconverted camera RGB.
    target, _ = _chromaticity(jpeg)
    error = lambda image: np.abs(np.median(_chromaticity(image)[0], 0) - np.median(target, 0)).sum()
    exposure = np.median(jpeg) / np.median(rgb)
    assert error(rgb * exposure) < .5 * error(plain * np.median(jpeg) / np.median(plain))


@pytest.mark.parametrize('sample', HIGH_EFFICIENCY, ids=lambda p: p.stem)
def test_high_efficiency_nef_of_bodies_libraw_does_not_list_opens_from_embedded_jpeg(sample):
    need(sample)
    preview, info = engine.load_image(sample)
    assert not info['raw'] and info['embedded'] and '内嵌 JPEG' in info['format']
    assert '高效率' in info['note'] and info['photo']['body'].startswith('NIKON Z5')
    assert max(preview.shape[:2]) == 1600 and info['width'] > 3700


def test_undecodable_sensor_data_without_metadata_still_opens_with_a_neutral_note(monkeypatch):
    sample = HIGH_EFFICIENCY[0]
    need(sample)
    monkeypatch.setattr(white_balance, 'metadata', lambda path: ({}, '未安装元数据读取器'))
    preview, info = engine.load_image(sample, 400)
    assert info['embedded'] and '数据不完整' in info['note'] and max(preview.shape[:2]) == 400


def test_full_size_decode_and_thumbnail_use_the_same_paths():
    from lumen import library
    need(Z50II_LOSSLESS)
    need(HIGH_EFFICIENCY[1])
    full, info = engine.load_image(Z50II_LOSSLESS, None, clip=True)
    assert full.shape == (3728, 5600, 3) and full.dtype == np.float32 and 0 <= full.min() and full.max() <= 1
    # 1.6.0: without clip, colours outside sRGB stay (process 2 maps them later).
    unclipped, _ = engine.load_image(Z50II_LOSSLESS, None)
    assert np.isfinite(unclipped).all() and unclipped.min() < 0
    np.testing.assert_allclose(np.clip(unclipped, 0, 1), full, atol=1e-6)
    embedded, _ = engine.load_image(HIGH_EFFICIENCY[1], None)
    assert embedded.shape == (3712, 5568, 3)
    assert library.thumbnail(HIGH_EFFICIENCY[1]).shape == (80, 120, 3)
