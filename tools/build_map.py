"""Build the offline world map of the location picker (2.1.2): ``assets/map/world.npz``.

    python tools/build_map.py [--source FOLDER]

Reads Natural Earth's GeoJSON layers (public domain, https://www.naturalearthdata.com/) from
``--source`` or downloads them from the natural-earth-vector repository at a fixed commit, checks
their SHA-256, simplifies the lines (Douglas–Peucker) and writes one compressed NumPy archive:

* ``land`` at three levels of detail (1:110m, 1:50m, 1:10m), ``lakes`` and ``rivers`` at two;
  ``coast``: the land's outlines without the edges Natural Earth cut along the 180° meridian
  and the South Pole (they are not coast);
  each layer is ``<layer>_<lod>_xy`` (float32 longitude, latitude) and ``<layer>_<lod>_parts``
  (int32 start of every ring or line, plus the end);
* ``places``: the 7,342 populated places with their position, the zoom level at which Natural
  Earth labels them, population and names in LUMEN RAW's nine interface languages.

The map is physical on purpose: coastlines, lakes, rivers and towns, no national or provincial
boundaries (and no country names), which are disputed in places and drawn differently by
different countries.  ``assets/map/README.txt`` records the source.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / 'assets' / 'map'
COMMIT = 'ca96624a56bd078437bca8184e78163e5039ad19'
BASE = f'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/{COMMIT}/geojson/'
#: Source files and their SHA-256 at ``COMMIT``.
SOURCES = {
    'ne_110m_land': '9e0729ee253ca7d7a5c4ae9395fb1902264c5377c52e224d13dd85010e2835d9',
    'ne_50m_land': 'e874b27a51d146452be360cafb3cc50c86001074a67d534113e6534682f9826b',
    'ne_10m_land': '1ac90796408bc6ad6911d69448485d3c4dbf2190370080368a09976e1c9f7416',
    'ne_50m_lakes': 'd350b75978b26fe839b797c2c529b2fb8f47fb3983c03f4964e36d5df9378a52',
    'ne_10m_lakes': '2d036f53dedec578001c5c30c2959ee7d4eebc1306900fa4367c49929ec8f2d9',
    'ne_50m_rivers_lake_centerlines': 'f286e0ce978fde999ca2d7a78c764be08542e19b63cded52b05c12d5173ccc51',
    'ne_10m_rivers_lake_centerlines': 'bb854a900ecbd3b408df46d5e16e3e0f974ba55993f9d8b5c26e855273c0905a',
    'ne_10m_populated_places': '9b8e3de09048ef00dfc70357dbb9fa324493f214b5e0ae4daf1aa79a8d10116b',
}
#: Layer → [(level of detail, source, Douglas–Peucker tolerance in degrees)].
LAYERS = {
    'land': [('110m', 'ne_110m_land', 0.), ('50m', 'ne_50m_land', .01), ('10m', 'ne_10m_land', .0012)],
    'lakes': [('50m', 'ne_50m_lakes', .01), ('10m', 'ne_10m_lakes', .0012)],
    'rivers': [('50m', 'ne_50m_rivers_lake_centerlines', .01), ('10m', 'ne_10m_rivers_lake_centerlines', .0015)],
}
#: Interface language → Natural Earth name field.
NAMES = {'zh_CN': 'NAME_ZH', 'zh_TW': 'NAME_ZHT', 'en': 'NAME_EN', 'ja': 'NAME_JA', 'ko': 'NAME_KO',
         'de': 'NAME_DE', 'fr': 'NAME_FR', 'es': 'NAME_ES', 'ru': 'NAME_RU'}


def load(name, source):
    if source:
        data = (Path(source) / f'{name}.geojson').read_bytes()
    else:
        with urllib.request.urlopen(BASE + f'{name}.geojson', timeout=300) as response:
            data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if SOURCES.get(name) and SOURCES[name] != digest:
        raise SystemExit(f'{name}: SHA-256 {digest} differs from the expected {SOURCES[name]}')
    return json.loads(data.decode('utf-8')), digest


def rings(geometry):
    """Every ring of a (multi)polygon, every line of a (multi)line string."""
    if geometry is None:
        return
    kind, coordinates = geometry['type'], geometry['coordinates']
    if kind == 'Polygon':
        yield from coordinates
    elif kind == 'MultiPolygon':
        for polygon in coordinates:
            yield from polygon
    elif kind == 'LineString':
        yield coordinates
    elif kind == 'MultiLineString':
        yield from coordinates


def simplify(points, tolerance, closed):
    if tolerance <= 0 or len(points) < 4:
        return points
    curve = points.astype(np.float32).reshape(-1, 1, 2)
    kept = cv2.approxPolyDP(curve, tolerance, closed).reshape(-1, 2)
    return kept.astype(np.float64) if len(kept) >= (3 if closed else 2) else points[[0, -1]]


def pack(features, tolerance, closed):
    parts, chunks, total = [0], [], 0
    for feature in features:
        for ring in rings(feature['geometry']):
            points = np.asarray(ring, np.float64)[:, :2]
            points = simplify(points, tolerance, closed)
            if len(points) < (3 if closed else 2):
                continue
            chunks.append(points)
            total += len(points)
            parts.append(total)
    return np.concatenate(chunks).astype(np.float32), np.asarray(parts, np.int32)


def coast(xy, parts):
    """Land outlines as open lines, broken where an edge runs along the map's border."""
    lines, total, out = [0], 0, []
    for start, end in zip(parts[:-1], parts[1:]):
        ring = np.concatenate([xy[start:end], xy[start:start + 1]])
        border = (np.abs(ring[:, 0]) >= 179.9999) | (ring[:, 1] <= -89.9999)
        cut = border[:-1] & border[1:]               # an edge along the border
        piece = [ring[0]]
        for i in range(1, len(ring)):
            if cut[i - 1]:
                if len(piece) >= 2:
                    out.append(np.asarray(piece))
                    total += len(piece)
                    lines.append(total)
                piece = [ring[i]]
            else:
                piece.append(ring[i])
        if len(piece) >= 2:
            out.append(np.asarray(piece))
            total += len(piece)
            lines.append(total)
    return np.concatenate(out).astype(np.float32), np.asarray(lines, np.int32)


def places(collection):
    xy, zoom, population, names = [], [], [], {code: [] for code in NAMES}
    for feature in collection['features']:
        p = {k.upper(): v for k, v in feature['properties'].items()}
        lon, lat = feature['geometry']['coordinates'][:2]
        xy.append((lon, lat))
        zoom.append(float(p.get('MIN_ZOOM') or 10))
        population.append(int(max(p.get('POP_MAX') or 0, 0)))
        fallback = str(p.get('NAME_EN') or p.get('NAME') or '').strip()
        for code, field in NAMES.items():
            names[code].append(str(p.get(field) or '').strip() or fallback)
    return (np.asarray(xy, np.float32), np.asarray(zoom, np.float32), np.asarray(population, np.int64),
            json.dumps(names, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--source', help='folder with the Natural Earth .geojson files (else downloaded)')
    args = parser.parse_args()
    arrays, digests, counts = {}, {}, {}
    for layer, levels in LAYERS.items():
        for lod, name, tolerance in levels:
            collection, digests[name] = load(name, args.source)
            xy, parts = pack(collection['features'], tolerance, layer != 'rivers')
            arrays[f'{layer}_{lod}_xy'], arrays[f'{layer}_{lod}_parts'] = xy, parts
            counts[f'{layer}_{lod}'] = (len(parts) - 1, len(xy))
            if layer == 'land':
                xy, parts = coast(xy, parts)
                arrays[f'coast_{lod}_xy'], arrays[f'coast_{lod}_parts'] = xy, parts
                counts[f'coast_{lod}'] = (len(parts) - 1, len(xy))
    collection, digests['ne_10m_populated_places'] = load('ne_10m_populated_places', args.source)
    xy, zoom, population, names = places(collection)
    arrays.update(places_xy=xy, places_zoom=zoom, places_population=population,
                  places_names=np.frombuffer(names.encode('utf-8'), np.uint8))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    (OUTPUT / 'world.npz').write_bytes(buffer.getvalue())
    lines = ['Offline world map of the location picker (LUMEN RAW 2.1.2), built by tools/build_map.py.', '',
             'Source: Natural Earth (https://www.naturalearthdata.com/), public domain:',
             '"All versions of Natural Earth raster + vector map data found on this website are in the public domain."',
             f'Files from https://github.com/nvkelso/natural-earth-vector at commit {COMMIT}:', '']
    lines += [f'  {name}.geojson  sha256 {digest}' for name, digest in sorted(digests.items())]
    lines += ['', 'Simplified with Douglas-Peucker; coastlines, lakes, rivers and populated places only:',
              'no national or provincial boundaries and no country names.', '',
              'Rings / lines and points per layer:']
    lines += [f'  {key}: {parts} parts, {points} points' for key, (parts, points) in counts.items()]
    lines += [f'  places: {len(xy)} towns and cities', '', f'world.npz: {len(buffer.getvalue())} bytes']
    (OUTPUT / 'README.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[-(len(counts) + 3):]))


if __name__ == '__main__':
    sys.exit(main())
