"""Built-in camera and lens logos in watermark borders, picked from the EXIF."""
import base64
import hashlib
import io
import json

import numpy as np
import pytest
from PIL import Image

from lumen import model, watermark


def photo(brand, body, lens='', lens_brand=''):
    return dict(brand=brand, body=body, lens=lens, lens_brand=lens_brand, date='2026-10-01 08:30:00',
                aperture='f/2.8', shutter='1/500 s', iso='ISO 100', focal='35 mm')


@pytest.mark.parametrize('exif,expected', [
    (('SONY', 'ILCE-7M3', 'E 28-200mm F2.8-5.6 A071', 'TAMRON'), ('sony', 'tamron')),
    (('SONY', 'ILCE-1', 'FE 24-70mm F2.8 GM II', 'SONY'), ('sony', None)),
    (('Canon', 'Canon EOS R5m2', 'RF135mm F1.8 L IS USM', 'CANON'), ('canon', None)),
    (('Nikon', 'NIKON Z 8', 'NIKKOR Z 14-24mm f/2.8 S', 'NIKON'), ('nikon', 'nikkor')),
    (('FUJIFILM', 'X-T5', 'XF33mmF1.4 R LM WR', 'FUJIFILM'), ('fujifilm', None)),
    (('Panasonic', 'DC-S5M2', 'LUMIX S 24-105/F4', 'PANASONIC'), ('lumix', None)),
    (('OM Digital Solutions', 'OM-1', 'OLYMPUS M.12-40mm F2.8', 'OLYMPUS'), ('om-system', 'olympus')),
    (('RICOH IMAGING COMPANY, LTD.', 'PENTAX K-3 Mark III', 'HD PENTAX-DA 16-50mm', 'PENTAX'), ('pentax', None)),
    (('RICOH IMAGING COMPANY, LTD.', 'RICOH GR III', '', ''), ('ricoh', None)),
    (('SONY', 'ILCE-7RM5', 'AF 75mm F1.2 Pro', 'VILTROX'), ('sony', None)),
    (('SONY', 'ILCE-1', 'Batis 2/40 CF', 'ZEISS'), ('sony', 'zeiss')),
    (('Apple', 'iPhone 15 Pro', 'iPhone 15 Pro back camera', 'Apple'), ('apple', None)),
    (('', '', '', ''), (None, None)),
])
def test_logos_follow_the_exif(exif, expected):
    assert watermark.brand_logos(photo(*exif)) == expected


def test_every_logo_has_its_source_and_a_rendered_image():
    entries = watermark.builtin()
    assert len(entries) >= 19
    for key, entry in entries.items():
        svg = (watermark.LOGOS / 'src' / f'{key}.svg').read_bytes()
        assert hashlib.sha256(svg).hexdigest() == entry['sha256'], key
        assert entry['source'].startswith('https://commons.wikimedia.org/wiki/File:')
        with Image.open(watermark.LOGOS / f'{key}.png') as im:
            assert im.mode == 'RGBA' and im.height == 512
        assert entry['colour'] in ('ink', 'colour', 'badge') and entry['role'] in ('camera', 'lens', 'both')
    readme = (watermark.LOGOS / 'README.txt').read_text(encoding='utf-8')
    assert 'trademarks of their respective owners' in readme and all(e['source'] in readme for e in entries.values())


def _ink(key, preset):
    _, paper, ink = watermark.PRESETS[preset]
    image = np.asarray(watermark.builtin_logo(key, tuple(ink), tuple(paper)))
    solid = image[..., 3] > 200
    return image[solid][:, :3], ink


def test_logo_colours_stay_readable_on_every_border():
    pixels, ink = _ink('sony', 'noir')                 # black wordmark → the light border ink
    assert (pixels == ink).all()
    pixels, ink = _ink('fujifilm', 'noir')             # black letters → ink, red mark kept
    assert (pixels == ink).mean() > .5 and ((pixels[:, 0] > 180) & (pixels[:, 1] < 80)).any()
    pixels, ink = _ink('olympus', 'noir')              # dark blue on black: drawn in ink
    assert (pixels == ink).all()
    pixels, _ = _ink('canon', 'noir')                  # red reads on black: kept
    assert pixels[:, 0].mean() > 150 and pixels[:, 1].mean() < 80
    pixels, _ = _ink('nikon', 'noir')                  # badge: yellow kept
    assert ((pixels[:, 0] > 200) & (pixels[:, 1] > 200) & (pixels[:, 2] < 80)).mean() > .3


def test_border_draws_both_logos_and_leaves_out_their_brand_names():
    settings = dict(watermark.defaults(), enabled=True)
    tamron = photo('SONY', 'ILCE-7M3', 'E 28-200mm F2.8-5.6 A071', 'TAMRON')
    strip = watermark.band(1400, 150, settings, tamron)
    paper = np.array(watermark.PRESETS['gallery'][1]) / 255
    ink_left = np.abs(strip[:, :250] - paper).max(axis=2) > .2
    ink_right = np.abs(strip[:, -250:] - paper).max(axis=2) > .2
    assert ink_left.sum() > 500 and ink_right.sum() > 500
    rows = watermark.lines(settings, tamron, {'SONY', 'TAMRON'})
    assert not any('SONY' in r or 'TAMRON' in r for r in rows)
    # A lens make without a logo stays as text; a maker's own lens adds no second logo or name.
    assert watermark.lines(settings, photo('SONY', 'ILCE-7RM5', 'AF 75mm F1.2 Pro', 'VILTROX'), {'SONY'})[0] == 'VILTROX'
    without = dict(settings, camera_logo='', lens_logo='')
    plain = watermark.band(1400, 150, without, tamron)
    assert (np.abs(plain[:, :250] - paper).max(axis=2) > .2).sum() < ink_left.sum() / 4


def test_settings_default_to_automatic_and_keep_older_choices():
    assert watermark.defaults()['camera_logo'] == watermark.AUTO == model.recipe()['watermark']['lens_logo']
    older = dict(watermark.defaults(), camera_logo='', lens_logo='')
    assert watermark.validate(older)['camera_logo'] == ''           # projects without logos stay so
    buffer = io.BytesIO()
    Image.new('RGBA', (40, 20), (200, 0, 0, 255)).save(buffer, 'PNG')
    custom = base64.b64encode(buffer.getvalue()).decode('ascii')
    settings = watermark.validate(dict(watermark.defaults(), enabled=True, camera_logo=custom))
    strip = watermark.band(800, 120, settings, photo('SONY', 'ILCE-7M3'))
    assert ((strip[..., 0] > .7) & (strip[..., 1] < .1)).any()      # an imported PNG still draws
    with pytest.raises(ValueError):
        watermark.validate(dict(watermark.defaults(), camera_logo='broken'))


# ----------------------------------------------------------------------------- panel

from test_ui import app, window, wait_until  # noqa: E402,F401


def test_watermark_panel_switches_the_automatic_logos(window):
    from lumen.watermark_dialog import WatermarkEditor
    w = window
    w.edits['watermark'] = dict(watermark.defaults(), enabled=True)
    editor = WatermarkEditor(w)
    assert all(c.isChecked() for c in editor.logos.values())
    editor.logos['lens_logo'].setChecked(False)
    values = editor.values()
    assert values['camera_logo'] == watermark.AUTO and values['lens_logo'] == ''
    editor.set_settings(dict(values, lens_logo=watermark.AUTO))
    assert editor.logos['lens_logo'].isChecked()
    editor.deleteLater()
