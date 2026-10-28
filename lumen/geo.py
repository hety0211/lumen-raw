"""Where a photo was taken (2.1.2): GPS positions, their text forms and the XMP sidecar.

A position is ``(latitude, longitude)`` in decimal degrees, WGS-84, north and east positive (what
cameras and phones record).  The camera's own GPS comes from the photo's EXIF; a position chosen
on LUMEN RAW's map is written into the photo's XMP sidecar (``exif:GPSLatitude`` and
``exif:GPSLongitude``, as Lightroom does), never into the photo itself, and wins over the
camera's when the photo is opened again.  Exports carry it in their EXIF unless location
metadata is switched off.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

#: Web Mercator's latitude limit: the map is square.
MAX_LATITUDE = 85.05112878


def valid(position):
    try:
        lat, lon = (float(v) for v in position)
    except (TypeError, ValueError):
        return False
    return math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180


def normalized(position):
    """``(lat, lon)`` rounded to 6 decimals (about 0.1 m) with the longitude in −180…180."""
    lat, lon = (float(v) for v in position)
    lon = (lon + 180) % 360 - 180 if not -180 <= lon <= 180 else lon
    return round(max(-90., min(90., lat)), 6), round(lon, 6)


def _dms(value, positive, negative):
    hemisphere = positive if value >= 0 else negative
    seconds = round(abs(value) * 3600)
    degrees, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f'{degrees}°{minutes:02d}′{seconds:02d}″{hemisphere}'


def dms(position):
    """``31°13′49″N 121°28′25″E`` (to the second, about 30 m)."""
    lat, lon = position
    return f'{_dms(lat, "N", "S")} {_dms(lon, "E", "W")}'


def decimal(position):
    """``31.230400, 121.473700``, the form map applications accept."""
    lat, lon = position
    return f'{lat:.6f}, {lon:.6f}'


#: One coordinate: optional hemisphere, degrees, optional minutes and seconds, optional hemisphere.
_PART = re.compile(r'([NSEW])?\s*([-+]?\d+(?:\.\d+)?)\s*°?\s*(?:(\d+(?:\.\d+)?)\s*\'?\s*)?'
                   r'(?:(\d+(?:\.\d+)?)\s*"?\s*)?([NSEW])?', re.I)
_SYMBOLS = str.maketrans({'北': 'N', '南': 'S', '东': 'E', '西': 'W', '東': 'E', 'º': '°', '度': '°', '′': "'",
                          '’': "'", '分': "'", '″': '"', '”': '"', '秒': '"', '纬': None, '經': None, '经': None,
                          '緯': None})


def _degrees(match):
    lead, d, m, s, trail = match.groups()
    value = float(d)
    if m:
        value = math.copysign(abs(value) + float(m) / 60, value)
    if s:
        value = math.copysign(abs(value) + float(s) / 3600, value)
    hemisphere = (lead or trail or '').upper()
    if hemisphere in ('S', 'W'):
        value = -abs(value)
    return value, hemisphere


def parse(text):
    """``(lat, lon)`` from text: ``31.2304, 121.4737``, ``31.2304 N 121.4737 E``,
    ``31°13′49″N 121°28′25″E``, ``N 31 13.82 E 121 28.42``…  Raises ValueError."""
    text = str(text or '').strip().replace('，', ',').replace('；', ';').replace("''", '"').translate(_SYMBOLS)
    if not text:
        raise ValueError('empty')
    # "31.2304, 121.4737": a comma between two plain numbers separates them (not a decimal comma).
    plain = re.fullmatch(r'\s*([-+]?\d+(?:\.\d+)?)\s*[,;\s]\s*([-+]?\d+(?:\.\d+)?)\s*', text)
    if plain:
        position = float(plain.group(1)), float(plain.group(2))
    else:
        parts = [m for m in _PART.finditer(text.replace(',', ' ').replace(';', ' ')) if m.group(2)]
        if len(parts) != 2:
            raise ValueError('two coordinates expected')
        (a, ha), (b, hb) = _degrees(parts[0]), _degrees(parts[1])
        if ha in ('E', 'W') and hb in ('N', 'S', ''):
            a, b = b, a                      # longitude first ("E121 N31")
        position = a, b
    if not valid(position):
        raise ValueError('out of range')
    return normalized(position)


# -- Web Mercator: the map's world square, 0…1 from the west / north edge

def to_world(lat, lon):
    lat = max(-MAX_LATITUDE, min(MAX_LATITUDE, lat))
    x = (lon + 180) / 360
    s = math.sin(math.radians(lat))
    y = .5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)
    return x, y


def from_world(x, y):
    lon = x * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y))))
    return lat, lon


# -- the XMP sidecar

def _xmp_coordinate(text):
    """XMP's ``DDD,MM.mmk`` / ``DDD,MM,SSk`` (or a signed decimal) → degrees."""
    text = str(text).strip()
    match = re.fullmatch(r'(\d+),(\d+(?:\.\d+)?)(?:,(\d+(?:\.\d+)?))?\s*([NSEW])', text, re.I)
    if match:
        degrees, minutes, seconds, hemisphere = match.groups()
        value = int(degrees) + float(minutes) / 60 + (float(seconds) / 3600 if seconds else 0)
        return -value if hemisphere.upper() in 'SW' else value
    return float(text)


def sidecar_position(photo):
    """The position in the photo's XMP sidecar (``name.xmp`` or ``name.ext.xmp``), or None."""
    from . import catalog
    for path in catalog._sidecars(photo):
        try:
            text = path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        values = {}
        for name in ('GPSLatitude', 'GPSLongitude'):
            match = re.search(rf'exif:{name}\s*=\s*"([^"]+)"', text) or \
                re.search(rf'<exif:{name}>([^<]+)</exif:{name}>', text)
            if match:
                try:
                    values[name] = _xmp_coordinate(match.group(1))
                except ValueError:
                    pass
        if len(values) == 2 and valid((values['GPSLatitude'], values['GPSLongitude'])):
            return normalized((values['GPSLatitude'], values['GPSLongitude']))
    return None


def write(photo, position, timeout=60):
    """Write ``position`` (None: remove it) into the photo's XMP sidecar through the one sidecar
    writer, and wait.  Returns the error lines ([] when written)."""
    from . import catalog
    if position is not None:
        position = normalized(position)
    writer = catalog.SIDECARS
    errors = writer.wait(writer.queue(locations={str(Path(photo)): position}), timeout)
    return ['XMP sidecar: timed out'] if errors is None else errors
