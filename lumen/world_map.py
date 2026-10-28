"""The offline world map of the location picker (2.1.2): Natural Earth data, drawn with OpenCV.

``assets/map/world.npz`` (``tools/build_map.py``) holds land, lakes and rivers at up to three
levels of detail and 7,342 towns with names in every interface language; it is public domain.
``Renderer`` draws the Web Mercator view of a widget into an RGB array: rings that cannot be seen
are skipped, the level of detail follows the zoom and the world repeats east and west.  Town
labels, the pin and the controls are the widget's (``location_dialog``).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import geo

ASSET = Path(__file__).resolve().parents[1] / 'assets' / 'map' / 'world.npz'
#: Size of the world square at zoom 0, in pixels (as web maps).
TILE = 256
MIN_ZOOM, MAX_ZOOM = 1., 16.
#: Fixed-point bits of the coordinates handed to OpenCV (sub-pixel accuracy).
SHIFT = 4
#: Level of detail per zoom: below 3.4 the 1:110m land, below 5.6 the 1:50m layers, then 1:10m.
LEVELS = ((3.4, '110m'), (5.6, '50m'), (99., '10m'))
COLORS = {
    'light': dict(ocean=(197, 219, 233), land=(242, 239, 231), coast=(141, 156, 168), river=(150, 189, 214),
                  grid=(214, 224, 231), urban=(229, 223, 210)),
    'dark': dict(ocean=(19, 31, 41), land=(48, 48, 44), coast=(92, 104, 112), river=(46, 78, 101),
                 grid=(31, 43, 53), urban=(60, 58, 52)),
}


class Place(NamedTuple):
    index: int
    lat: float
    lon: float
    zoom: float           # Natural Earth's label zoom: big cities from 2, villages up to 10
    population: int


@lru_cache(maxsize=1)
def data():
    with np.load(ASSET) as archive:
        return {name: archive[name] for name in archive.files}


def available():
    return ASSET.is_file()


@lru_cache(maxsize=1)
def _names():
    return json.loads(data()['places_names'].tobytes().decode('utf-8'))


def names(language):
    """Every town's name in ``language`` (English for languages without names)."""
    table = _names()
    return table.get(language) or table.get('zh_CN' if language.startswith('zh') else 'en') or table['en']


@lru_cache(maxsize=1)
def places():
    archive = data()
    return [Place(i, float(lat), float(lon), float(zoom), int(population)) for i, ((lon, lat), zoom, population)
            in enumerate(zip(archive['places_xy'], archive['places_zoom'], archive['places_population']))]


@lru_cache(maxsize=1)
def _place_world():
    xy = data()['places_xy'].astype(np.float64)
    lat = np.clip(xy[:, 1], -geo.MAX_LATITUDE, geo.MAX_LATITUDE)
    s = np.sin(np.radians(lat))
    return np.stack([(xy[:, 0] + 180) / 360, .5 - np.log((1 + s) / (1 - s)) / (4 * math.pi)], axis=1)


def search(text, language, limit=12):
    """Towns whose name (in any language) matches ``text``: exact names first, then names that
    start with it, then the rest; bigger towns first within each.  ``[(name, Place)]``."""
    text = str(text or '').strip().casefold()
    if not text:
        return []
    shown = names(language)
    table = _names()
    found = []
    for place in places():
        best = 9
        for code in table:
            name = table[code][place.index].casefold()
            if not name:
                continue
            if name == text:
                best = 0
                break
            if name.startswith(text):
                best = min(best, 1)
            elif text in name:
                best = min(best, 2)
        if best < 9:
            found.append((best, -place.population, place))
    found.sort(key=lambda item: item[:2])
    return [(shown[place.index], place) for _, _, place in found[:limit]]


def nearest(position, within=60.):
    """``(Place, km)`` of the town nearest to ``position`` within ``within`` km, or None."""
    lat, lon = position
    xy = data()['places_xy'].astype(np.float64)
    la1, lo1 = math.radians(lat), math.radians(lon)
    la2, lo2 = np.radians(xy[:, 1]), np.radians(xy[:, 0])
    a = np.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    km = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    index = int(np.argmin(km))
    return (places()[index], float(km[index])) if km[index] <= within else None


class Layer(NamedTuple):
    world: np.ndarray       # float64 (N, 2) Web Mercator 0…1
    parts: np.ndarray       # int32 start of each part, plus the end
    boxes: np.ndarray       # float64 (M, 4) x0, y0, x1, y1 per part


@lru_cache(maxsize=None)
def layer(name, level):
    archive = data()
    key = f'{name}_{level}'
    if f'{key}_xy' not in archive:
        return None
    xy = archive[f'{key}_xy'].astype(np.float64)
    parts = archive[f'{key}_parts'].astype(np.int64)
    lat = np.clip(xy[:, 1], -geo.MAX_LATITUDE, geo.MAX_LATITUDE)
    s = np.sin(np.radians(lat))
    world = np.stack([(xy[:, 0] + 180) / 360, .5 - np.log((1 + s) / (1 - s)) / (4 * math.pi)], axis=1)
    starts = parts[:-1]
    boxes = np.stack([np.minimum.reduceat(world[:, 0], starts), np.minimum.reduceat(world[:, 1], starts),
                      np.maximum.reduceat(world[:, 0], starts), np.maximum.reduceat(world[:, 1], starts)], axis=1)
    return Layer(world, parts, boxes)


def level_for(zoom):
    return next(level for limit, level in LEVELS if zoom < limit)


class View(NamedTuple):
    """A map view: ``center`` in world coordinates (0…1), ``zoom`` (world = TILE · 2^zoom px)."""
    center: tuple
    zoom: float
    width: int
    height: int

    @property
    def scale(self):
        return TILE * 2 ** self.zoom

    def to_pixel(self, x, y):
        return (x - self.center[0]) * self.scale + self.width / 2, (y - self.center[1]) * self.scale + self.height / 2

    def to_world(self, px, py):
        return (px - self.width / 2) / self.scale + self.center[0], (py - self.height / 2) / self.scale + self.center[1]

    def position(self, px, py):
        """``(lat, lon)`` under a pixel (longitude wrapped into −180…180)."""
        x, y = self.to_world(px, py)
        lat, lon = geo.from_world(x, min(1., max(0., y)))
        return geo.normalized((lat, lon))

    def offsets(self):
        """Whole-world shifts east and west that are on screen (the world repeats)."""
        x0, _ = self.to_world(0, 0)
        x1, _ = self.to_world(self.width, 0)
        return range(math.floor(x0), math.floor(x1) + 1)


def clamp(view):
    """``view`` with its zoom and its centre kept where the map fills the height."""
    zoom = min(MAX_ZOOM, max(MIN_ZOOM, view.zoom, math.log2(max(1, view.height) / TILE)))
    half = view.height / 2 / (TILE * 2 ** zoom)
    cx, cy = view.center
    cy = min(1 - half, max(half, cy)) if half < .5 else .5
    return View((cx - math.floor(cx), cy), zoom, view.width, view.height)


class Renderer:
    """Draws views of the map into RGB arrays."""

    def __init__(self, look='light'):
        self.colors = COLORS.get(look, COLORS['light'])

    def _shapes(self, name, level, view, offset):
        found = layer(name, level)
        if found is None:
            return []
        x0, y0 = view.to_world(0, 0)
        x1, y1 = view.to_world(view.width, view.height)
        x0, x1 = x0 - offset, x1 - offset
        boxes = found.boxes
        visible = np.nonzero((boxes[:, 2] >= x0) & (boxes[:, 0] <= x1) & (boxes[:, 3] >= y0) & (boxes[:, 1] <= y1))[0]
        if not len(visible):
            return []
        # Parts smaller than a pixel are left out (islets and ponds of a zoomed-out view).
        size = np.maximum(boxes[visible, 2] - boxes[visible, 0], boxes[visible, 3] - boxes[visible, 1]) * view.scale
        visible = visible[size >= .75]
        factor = float(1 << SHIFT)
        shapes = []
        starts, ends = found.parts[visible], found.parts[visible + 1]
        for start, end in zip(starts, ends):
            points = found.world[start:end]
            px = ((points[:, 0] + offset - view.center[0]) * view.scale + view.width / 2) * factor
            py = ((points[:, 1] - view.center[1]) * view.scale + view.height / 2) * factor
            shapes.append(np.stack([px, py], axis=1).round().astype(np.int32))
        return shapes

    def render(self, view):
        import cv2
        c = self.colors
        image = np.empty((view.height, view.width, 3), np.uint8)
        image[:] = c['ocean']
        level = level_for(view.zoom)
        water = '50m' if level == '110m' else level
        line = cv2.LINE_AA
        self._grid(image, view)
        for offset in view.offsets():
            land = self._shapes('land', level, view, offset)
            if land:
                cv2.fillPoly(image, land, c['land'], line, SHIFT)
                coast = self._shapes('coast', level, view, offset)
                if coast:
                    cv2.polylines(image, coast, False, c['coast'], 1, line, SHIFT)
            if view.zoom >= 2.5:
                lakes = self._shapes('lakes', water, view, offset)
                if lakes:
                    cv2.fillPoly(image, lakes, c['ocean'], line, SHIFT)
                    cv2.polylines(image, lakes, True, c['river'], 1, line, SHIFT)
                rivers = self._shapes('rivers', water, view, offset)
                if rivers:
                    cv2.polylines(image, rivers, False, c['river'], 1, line, SHIFT)
        return image

    def _grid(self, image, view):
        """Latitude and longitude lines every 30°, 10°, 5°, 1°, 0.5° or 0.1° by zoom."""
        import cv2
        step = next(s for z, s in ((3, 30), (5, 10), (6.5, 5), (9, 1), (11, .5), (99, .1)) if view.zoom < z)
        top, bottom = (geo.from_world(0, min(1., max(0., view.to_world(0, y)[1])))[0] for y in (0, view.height))
        x0, _ = view.to_world(0, 0)
        x1, _ = view.to_world(view.width, 0)
        lon = math.floor((x0 * 360 - 180) / step) * step
        while lon <= x1 * 360 - 180:
            px, _ = view.to_pixel((lon + 180) / 360, 0)
            cv2.line(image, (int(round(px)), 0), (int(round(px)), view.height), self.colors['grid'], 1)
            lon += step
        lat = math.floor(bottom / step) * step
        while lat <= top:
            _, py = view.to_pixel(0, geo.to_world(lat, 0)[1])
            cv2.line(image, (0, int(round(py))), (view.width, int(round(py))), self.colors['grid'], 1)
            lat += step

    def labels(self, view, language, limit=160):
        """Towns to label: ``[(px, py, name, Place)]``, most important first."""
        world = _place_world()
        shown = names(language)
        found = []
        for offset in view.offsets():
            px = (world[:, 0] + offset - view.center[0]) * view.scale + view.width / 2
            py = (world[:, 1] - view.center[1]) * view.scale + view.height / 2
            inside = np.nonzero((px >= -20) & (px <= view.width + 20) & (py >= -10) & (py <= view.height + 10))[0]
            for i in inside:
                place = places()[i]
                if place.zoom <= view.zoom + 1.2:
                    found.append((place.zoom, -place.population, float(px[i]), float(py[i]), i))
        found.sort()
        return [(x, y, shown[i], places()[i]) for _, _, x, y, i in found[:limit * 3]]
