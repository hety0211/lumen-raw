"""Capture metadata used by borders, including metadata embedded in AI copies."""
import json
import math
import re
from pathlib import Path

FIELDS = ('date', 'body', 'lens', 'aperture', 'shutter', 'iso', 'focal', 'brand', 'lens_brand',
          # 2.1.2: where it was taken (decimal degrees as text; ``gps_source`` 'camera' or 'xmp').
          'latitude', 'longitude', 'gps_source')


def clean(record):
    if not isinstance(record, dict): return {}
    return {key: re.sub(r'[\x00-\x1f]', ' ', str(record.get(key, '')))[:200] for key in FIELDS}


def from_tags(tags):
    def text(key):
        value = tags.get(key, '')
        return str(value).strip() if value is not None else ''
    def numeric(key):
        try:
            value = float(tags.get(key, 0))
            return value if math.isfinite(value) and value > 0 else None
        except (ValueError, TypeError): return None
    date = text('DateTimeOriginal') or text('CreateDate')
    if re.match(r'^\d{4}:\d{2}:\d{2}', date): date = date[:10].replace(':', '-') + date[10:]
    aperture, exposure, focal, iso = (numeric(k) for k in ('FNumber','ExposureTime','FocalLength','ISO'))
    shutter = ''
    if exposure:
        reciprocal=1/exposure
        shutter=(f'1/{round(reciprocal)} s' if exposure<1 and abs(reciprocal-round(reciprocal))<.02*reciprocal else f'{exposure:.3g} s')
    lens = text('LensModel') or text('Lens')
    if not lens and isinstance(tags.get('LensID'), str): lens = tags['LensID']
    brand = text('Make')
    if brand.upper().startswith('NIKON'):brand='Nikon'
    lens_brand = text('LensMake')
    if not lens_brand:
        identity=lens
        if isinstance(tags.get('LensID'),str) and ' or ' not in tags['LensID'].lower():identity+=' '+tags['LensID']
        for name in ('SIGMA','TAMRON','ZEISS','VOIGTLANDER','SAMYANG','VILTROX','CANON','NIKON','NIKKOR','FUJIFILM','FUJINON','PANASONIC','LUMIX','SONY','LEICA'):
            if name in identity.upper(): lens_brand = name; break
    record = dict(date=date,body=text('Model'),lens=lens,brand=brand,lens_brand=lens_brand,
                  aperture=f'f/{aperture:g}' if aperture else '',shutter=shutter,
                  iso=f'ISO {iso:g}' if iso else '',focal=f'{focal:g} mm' if focal else '')
    from . import geo
    try:
        found = (float(tags['GPSLatitude']), float(tags['GPSLongitude']))
    except (KeyError, TypeError, ValueError):
        found = None
    if found is not None and geo.valid(found) and found != (0., 0.):
        record = with_position(record, found, 'camera')
    return clean(record)


def position(photo):
    """``(lat, lon)`` of a photo record, or None (2.1.2)."""
    from . import geo
    try:
        found = (float(photo['latitude']), float(photo['longitude']))
    except (KeyError, TypeError, ValueError):
        return None
    return found if geo.valid(found) else None


def with_position(photo, found, source='xmp'):
    """A copy of ``photo`` with ``found`` (``(lat, lon)``; None removes it) as its position."""
    record = dict(photo or {})
    if found is None:
        record.update(latitude='', longitude='', gps_source='')
    else:
        record.update(latitude=f'{found[0]:.6f}', longitude=f'{found[1]:.6f}', gps_source=source)
    return record


def description(photo, provenance=None):
    return json.dumps({'application':'Lumen photo','version':1,'photo':clean(photo),
                       'processing':provenance or {}},ensure_ascii=True)


#: Tags of the original that still describe an exported picture (1.6.1, GitHub issue #5).
#: Left out: the original's pixel size, orientation, colour space, subject area and thumbnail
#: (the export is cropped, rotated, resized and sRGB), Camera Raw develop settings (Lightroom
#: would apply them again), and maker notes (they can overflow a JPEG's 64 KB EXIF segment).
COPIED = ('-ExifIFD:all', '--ExifIFD:ExifImageWidth', '--ExifIFD:ExifImageHeight', '--ExifIFD:ColorSpace',
          '--ExifIFD:CompressedBitsPerPixel', '--ExifIFD:SubjectArea', '--ExifIFD:SubjectLocation',
          '-IFD0:Make', '-IFD0:Model', '-IFD0:Artist', '-IFD0:Copyright', '-IFD0:Rating', '-IFD0:RatingPercent',
          '-IFD0:XPTitle', '-IFD0:XPComment', '-IFD0:XPAuthor', '-IFD0:XPKeywords', '-IFD0:XPSubject',
          '-IPTC:all')
XMP = ('-XMP:all', '--XMP-crs:all', '--XMP-crss:all', '--XMP-crd:all', '--XMP-crlcp:all', '--XMP-tiff:Orientation', '--XMP-tiff:ImageWidth',
       '--XMP-tiff:ImageLength', '--XMP-exif:PixelXDimension', '--XMP-exif:PixelYDimension', '--XMP-exif:SubjectArea')


def copy_from(source, target, location=True, position=None):
    """Copy the original's shooting data, copyright, keywords and (optionally) GPS position into
    the exported file ``target`` with ExifTool, in place.  Metadata in a ``<photo>.xmp``
    sidecar (Lightroom's) is included and wins over the photo's own XMP.  Returns '' or why
    nothing was written; the picture itself is never touched.

    2.1.2: ``position`` (``(lat, lon)``, e.g. chosen on the map) is written as the export's EXIF
    and XMP GPS position, over the original's, when ``location`` is on."""
    import os
    import subprocess
    import tempfile
    from . import __version__
    from .i18n import tr
    from .white_balance import exiftool
    command = exiftool()
    if command is None:
        return tr('没有找到 ExifTool')
    source, target = Path(source).resolve(), Path(target).resolve()
    ext = target.suffix.lower()
    xmp = XMP if location else XMP + ('--XMP-exif:GPS*',)
    lines = ['-charset', 'filename=UTF8', '-overwrite_original', '-m', '-TagsFromFile', str(source), *COPIED, *xmp]
    if location:
        lines.append('-GPS:all')
    if location and position is not None:
        lat, lon = position
        lines += [f'-GPS:GPSLatitude={abs(lat):.6f}', f'-GPS:GPSLatitudeRef={"N" if lat >= 0 else "S"}',
                  f'-GPS:GPSLongitude={abs(lon):.6f}', f'-GPS:GPSLongitudeRef={"E" if lon >= 0 else "W"}',
                  '-GPS:GPSMapDatum=WGS-84', f'-XMP-exif:GPSLatitude={lat:.6f}', f'-XMP-exif:GPSLongitude={lon:.6f}']
    if ext in ('.jpg', '.jpeg', '.png'):
        lines.append('-IFD0:ImageDescription')          # TIFF / DNG keep LUMEN's own description
    for sidecar in (source.with_suffix('.xmp'), source.with_name(source.name + '.xmp')):
        if sidecar.is_file():
            lines += ['-TagsFromFile', str(sidecar), *xmp]
            break
    lines.append(f'-Software=LUMEN RAW {__version__}')
    if ext != '.dng':
        lines.append('-ExifIFD:ColorSpace=sRGB')
    lines += ['--', str(target)]
    with tempfile.NamedTemporaryFile('w', suffix='.args', delete=False, encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
        arguments = handle.name
    try:
        env = dict(os.environ, LC_ALL='C', LANG='C', LC_CTYPE='C')
        result = subprocess.run([*command, '-config', '', '-charset', 'filename=UTF8', '-@', arguments],
                                capture_output=True, timeout=600, env=env,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc) or type(exc).__name__
    finally:
        os.unlink(arguments)
    if result.returncode:
        errors = [line for line in result.stderr.decode('utf-8', 'replace').splitlines() if line.strip()]
        return (errors[0] if errors else f'ExifTool {result.returncode}')[:200]
    return ''


def embedded(path):
    if Path(path).suffix.lower() not in ('.dng','.tif','.tiff'): return {}
    try:
        import tifffile
        with tifffile.TiffFile(path) as f:
            text = f.pages[0].description or ''
            if len(text)>16384: return {}
            data = json.loads(text)
            if data.get('application')=='Lumen photo' and data.get('version')==1:
                return clean(data.get('photo'))
    except (OSError, ValueError, AttributeError): pass
    return {}
