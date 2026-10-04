"""Lens profile corrections from the open lensfun database (1.5.1).

Distortion, lateral chromatic aberration (colour fringes towards the corners) and
vignetting are corrected with the calibration data of the lensfun project
(https://lensfun.github.io/).  Its database is vendored unmodified in
``assets/lensfun/db`` (CC BY-SA 3.0); the published models are evaluated here with
NumPy and OpenCV, no lensfun library is linked.

* ``identify`` matches a photo's EXIF (camera, lens, focal length, aperture, focus
  distance) to a lens entry once, on the loading thread.
* ``solve`` interpolates that lens' calibrations for the photo and converts them to a
  resolution-independent form kept in the recipe (``edits['lens']['profile']``), so
  previews, zoomed tiles, export and batch export render without the database.
* ``apply`` renders the correction for the whole frame or for one block of it.

Frame coordinates: ρ is the distance from the frame centre in units of the frame's half
diagonal.  An output pixel at ρ samples the source at

    ρ_u = z·ρ                                   z keeps the corrected frame filled
    ρ_d = ρ_u·(1 + c1·ρ_u + c2·ρ_u² + c3·ρ_u³ + c4·ρ_u⁴)      distortion
    ρ_R, ρ_B = ρ_d·(t0 + t1·ρ_d + t2·ρ_d²)                 lateral chromatic aberration

and is divided by 1 + v1·ρ_d² + v2·ρ_d⁴ + v3·ρ_d⁶ (vignetting, linear light).

lensfun's calibrations use other units: distortion and aberration models put r = 1 at
half the short edge of the calibration sensor, the vignetting model at its half diagonal
(both scale with the crop factor of the calibration camera).  Distortion is normalised
so the frame centre keeps its scale.
"""
from __future__ import annotations

import copy
import functools
import logging
import math
import re
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

DB_FOLDER = Path(__file__).resolve().parents[1] / 'assets' / 'lensfun' / 'db'
DIAGONAL_35 = math.hypot(36., 24.)
#: Focus distance used when the camera records none: lensfun's "infinity".
FAR = 1000.
#: cv2.remap interpolates at 1/32 pixel; maps are rounded to that grid first so every
#: block of the frame samples exactly like the whole frame.
SUBPIXEL = 32.


def defaults():
    """Recipe entry: user intent plus the profile solved for the current photo."""
    return dict(enabled=False, distortion=100., vignetting=100., chromatic=True, manual=None, profile=None)


# ---------------------------------------------------------------- database

@dataclass
class Camera:
    maker: str
    model: str
    names: list
    makers: list
    mount: str
    crop: float


@dataclass
class Lens:
    maker: str
    model: str
    names: list
    makers: list
    mounts: list
    crop: float
    aspect: float
    kind: str
    center: tuple
    focal: tuple
    aperture: tuple
    distortion: list = field(default_factory=list)
    tca: list = field(default_factory=list)
    vignetting: list = field(default_factory=list)

    @property
    def label(self):
        return self.model if canonical_maker(self.maker) in words(self.model) else f'{self.maker} {self.model}'

    @property
    def ident(self):
        return dict(maker=self.maker, model=self.model)

    def has(self):
        return dict(distortion=bool(self.distortion), tca=bool(self.tca), vignetting=bool(self.vignetting))


_CORPORATE = {'corporation', 'corp', 'co', 'ltd', 'inc', 'ag', 'company', 'imaging', 'camera', 'cameras',
              'gmbh', 'optical', 'computer', 'techwin', 'kk', 'kabushiki', 'kaisha'}
_MAKER_ALIASES = {'carl zeiss': 'zeiss', 'eastman kodak': 'kodak', 'voigtlander': 'voigtländer',
                  'konica': 'konica minolta', 'om system': 'om digital solutions', 'ricoh pentax': 'pentax'}


def canonical_maker(text):
    words_ = [w for w in re.findall(r'[0-9a-zà-ÿ]+', str(text or '').lower()) if w not in _CORPORATE]
    name = ' '.join(words_)
    return _MAKER_ALIASES.get(name, name)


def words(text):
    return set(re.findall(r'[0-9a-zà-ÿ]+', str(text or '').lower()))


def _compact(text):
    return re.sub(r'[^0-9a-z]', '', str(text or '').lower())


def _float(value, default=None):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _aspect(text):
    if not text:
        return 1.5
    if ':' in text:
        a, b = text.split(':', 1)
        return max(_float(a, 3.), _float(b, 2.)) / max(min(_float(a, 3.), _float(b, 2.)), 1e-6)
    return _float(text, 1.5) or 1.5


class Database:
    def __init__(self, folder=DB_FOLDER):
        self.cameras, self.lenses, self.compat = [], [], {}
        self.folder = Path(folder)
        for path in sorted(self.folder.glob('*.xml')):
            try:
                self._read(ET.parse(path).getroot())
            except (ET.ParseError, OSError) as exc:
                log.warning('lens database %s skipped: %s', path.name, exc)
        self.makers = sorted({lens.maker for lens in self.lenses}, key=str.lower)
        self._camera_index = {}
        for camera in self.cameras:
            for maker in camera.makers:
                for name in camera.names:
                    self._camera_index.setdefault((canonical_maker(maker), _compact(name)), camera)
        log.info('lens database: %d cameras, %d lenses', len(self.cameras), len(self.lenses))

    def _read(self, root):
        for mount in root.findall('mount'):
            name = (mount.findtext('name') or '').strip()
            if name:
                self.compat.setdefault(name, set()).update((c.text or '').strip() for c in mount.findall('compat'))
        for node in root.findall('camera'):
            makers = [(m.text or '').strip() for m in node.findall('maker')]
            names = [(m.text or '').strip() for m in node.findall('model')]
            if makers and names:
                self.cameras.append(Camera(makers[0], names[0], names, makers, (node.findtext('mount') or '').strip(),
                                           _float(node.findtext('cropfactor'), 1.) or 1.))
        for node in root.findall('lens'):
            lens = self._lens(node)
            if lens is not None:
                self.lenses.append(lens)

    @staticmethod
    def _lens(node):
        makers = [(m.text or '').strip() for m in node.findall('maker')]
        names = [(m.text or '').strip() for m in node.findall('model')]
        if not makers or not names:
            return None
        calibration = node.find('calibration')
        distortion, tca, vignetting = [], [], []
        if calibration is not None:
            for item in calibration.findall('distortion'):
                model, focal = item.get('model'), _float(item.get('focal'))
                if focal and model in ('poly3', 'poly5', 'ptlens'):
                    keys = dict(poly3=('k1',), poly5=('k1', 'k2'), ptlens=('a', 'b', 'c'))[model]
                    distortion.append((focal, model, tuple(_float(item.get(k), 0.) for k in keys)))
            for item in calibration.findall('tca'):
                model, focal = item.get('model'), _float(item.get('focal'))
                if focal and model == 'linear':
                    tca.append((focal, 'poly3', (_float(item.get('kr'), 1.), _float(item.get('kb'), 1.), 0., 0., 0., 0.)))
                elif focal and model == 'poly3':
                    tca.append((focal, 'poly3', tuple(_float(item.get(k), d) for k, d in
                                                      (('vr', 1.), ('vb', 1.), ('cr', 0.), ('cb', 0.), ('br', 0.), ('bb', 0.)))))
            for item in calibration.findall('vignetting'):
                focal, aperture = _float(item.get('focal')), _float(item.get('aperture'))
                if item.get('model') == 'pa' and focal and aperture:
                    vignetting.append((focal, aperture, _float(item.get('distance'), FAR) or FAR,
                                       tuple(_float(item.get(k), 0.) for k in ('k1', 'k2', 'k3'))))
        focal_node, aperture_node, center = node.find('focal'), node.find('aperture'), node.find('center')
        focals = [f for f, _, _ in distortion] + [f for f, _, _ in tca] + [v[0] for v in vignetting]
        parsed = parse_name(names[0])
        if focal_node is not None:
            low = _float(focal_node.get('min')) or _float(focal_node.get('value'))
            high = _float(focal_node.get('max')) or low
        else:
            low, high = parsed['focal'] or ((min(focals), max(focals)) if focals else (None, None))
        if aperture_node is not None:
            aperture = (_float(aperture_node.get('min')) or _float(aperture_node.get('value')),
                        _float(aperture_node.get('max')))
        else:
            aperture = (parsed['aperture'], None)
        return Lens(makers[0], names[0], names, makers, [(m.text or '').strip() for m in node.findall('mount')],
                    _float(node.findtext('cropfactor'), 1.) or 1., _aspect(node.findtext('aspect-ratio')),
                    (node.findtext('type') or 'rectilinear').strip(),
                    ((_float(center.get('x'), 0.), _float(center.get('y'), 0.)) if center is not None else (0., 0.)),
                    (low, high), aperture, sorted(distortion), sorted(tca), vignetting)

    def camera(self, maker, model):
        if not model:
            return None
        key = canonical_maker(maker)
        compact = _compact(model)
        found = self._camera_index.get((key, compact))
        if found is None and key:
            # "Canon EOS R5m2" is listed as such, "NIKON Z 8" as "Nikon Z 8": compare without the maker too.
            stripped = _compact(re.sub(re.escape(key), '', str(model), flags=re.I))
            for (maker_key, name), camera in self._camera_index.items():
                if maker_key == key and (name == stripped or _compact(re.sub(re.escape(key), '', name)) == stripped):
                    return camera
        return found

    def find(self, maker, model):
        for lens in self.lenses:
            if lens.maker == maker and lens.model == model:
                return lens
        return None

    def mounts_for(self, camera):
        if camera is None or not camera.mount:
            return None
        return {camera.mount} | self.compat.get(camera.mount, set())


_database = None
_lock = threading.Lock()


def database():
    global _database
    with _lock:
        if _database is None:
            _database = Database()
        return _database


# ---------------------------------------------------------------- matching

_FOCAL = re.compile(r'(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*mm')
_APERTURE = re.compile(r'(?<![a-z0-9])(?:f\s*/?\s*|1\s*:\s*)(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?')
#: Words that say nothing about which lens it is.
_NOISE = {'mm', 'lens', 'zoom', 'f', 'and', 'or', 'compatibles', 'the'}
_THIRD_PARTY = {'sigma', 'tamron', 'tokina', 'samyang', 'rokinon', 'zeiss', 'viltrox', 'ttartisan', '7artisans',
                'laowa', 'venus', 'voigtländer', 'voigtlander', 'yongnuo', 'meike', 'sirui', 'irix', 'kipon'}


@functools.lru_cache(maxsize=8192)
def parse_name(text):
    """Focal range, widest aperture and identifying words of a lens name."""
    lowered = str(text or '').lower().replace('ƒ', 'f').replace('mm', 'mm ')
    focal = _FOCAL.search(lowered)
    rest = _FOCAL.sub(' ', lowered) if focal else lowered
    aperture = _APERTURE.search(rest)
    rest = _APERTURE.sub(' ', rest) if aperture else rest
    focal_range = None
    if focal:
        low = float(focal.group(1))
        focal_range = (low, float(focal.group(2)) if focal.group(2) else low)
    return dict(focal=focal_range, aperture=float(aperture.group(1)) if aperture else None,
                words=words(rest) - _NOISE)


def _close(a, b, tolerance=.04):
    return abs(a - b) <= tolerance * max(a, b)


def _score(query, lens, makers, focal):
    best = 0.
    lens_makers = {canonical_maker(m) for m in lens.makers}
    maker_words = set().union(*(words(m) for m in lens.makers), *(set(m.split()) for m in lens_makers))
    for name in lens.names:
        target = parse_name(name)
        span = target['focal'] or (lens.focal if lens.focal[0] else None)
        if query['focal'] and span:
            if not (_close(query['focal'][0], span[0]) and _close(query['focal'][1], span[1] or span[0])):
                continue
        elif focal and lens.focal[0]:
            if not lens.focal[0] * .97 <= focal <= (lens.focal[1] or lens.focal[0]) * 1.03:
                continue
        if query['aperture'] and target['aperture'] and not _close(query['aperture'], target['aperture'], .06):
            continue
        a = query['words'] - maker_words - _MAKER_WORDS
        b = target['words'] - maker_words - _MAKER_WORDS
        if not a and not b:
            score = .55
        else:
            score = 2 * len(a & b) / max(1, len(a) + len(b))
        if makers:
            score += .15 if makers & lens_makers else -.3
        best = max(best, score)
    return best


_MAKER_WORDS = {'canon', 'nikon', 'nikkor', 'sony', 'fujifilm', 'fujinon', 'panasonic', 'lumix', 'olympus', 'pentax',
                'leica', 'ricoh', 'samsung', 'hasselblad', 'sigma', 'tamron', 'tokina', 'samyang', 'zeiss', 'carl',
                'viltrox', 'voigtländer', 'voigtlander', 'minolta', 'konica', 'om', 'system'}


def exif_from_tags(tags):
    """The capture facts lens matching needs, from the ExifTool tags read at loading."""
    def number(*keys):
        for key in keys:
            value = _float(tags.get(key))
            if value and value > 0:
                return value
        return None
    def text(key):
        value = tags.get(key)
        return str(value).strip() if value not in (None, '') else ''
    focal = number('FocalLength')
    crop35 = None
    focal35 = number('FocalLengthIn35mmFormat')
    if focal and focal35:
        crop35 = focal35 / focal
    elif number('ScaleFactor35efl'):
        crop35 = number('ScaleFactor35efl')
    lens_id = text('LensID')
    return dict(make=text('Make'), model=text('Model'), lens_make=text('LensMake'), lens_model=text('LensModel'),
                lens=text('Lens'), lens_id='' if ' or ' in lens_id.lower() else lens_id, focal=focal,
                aperture=number('FNumber'),
                distance=number('FocusDistance', 'FocusDistance2', 'ApproximateFocusDistance', 'SubjectDistance'),
                crop35=crop35 if crop35 and .3 <= crop35 <= 15 else None)


def identify(exif, db=None):
    """Camera, lens and crop factor for a photo: a JSON-friendly match kept in ``info['lens_match']``."""
    if not exif or not (exif.get('lens_model') or exif.get('lens') or exif.get('lens_id') or exif.get('model')):
        return None
    db = db or database()
    camera = db.camera(exif.get('make'), exif.get('model'))
    crop = camera.crop if camera else None
    if exif.get('crop35') and (crop is None or abs(exif['crop35'] / crop - 1) > .08):
        crop = exif['crop35']  # in-camera crop modes (APS-C on a full-frame body) or unknown bodies
    match = dict(camera=(f'{camera.maker} {camera.model}' if camera else exif.get('model', '')).strip(),
                 mount=camera.mount if camera else None, crop=crop, focal=exif.get('focal'), aperture=exif.get('aperture'),
                 distance=exif.get('distance'), query=exif.get('lens_model') or exif.get('lens_id') or exif.get('lens', ''),
                 lens=None)
    lens = None
    if camera and camera.mount and camera.mount[:1].islower():
        # Fixed-lens cameras: lensfun gives their built-in lens a mount of its own.
        fixed = sorted((l for l in db.lenses if camera.mount in l.mounts), key=lambda l: len(l.model))
        if fixed:
            lens = fixed[0]  # converter variants have longer names
    if lens is None:
        lens = _match_lens(db, exif, camera)
    if lens is not None:
        match['lens'] = lens.ident
        match['label'] = lens.label
    return match


def _match_lens(db, exif, camera):
    queries = [q for q in (exif.get('lens_model'), exif.get('lens_id'), exif.get('lens')) if q]
    if not queries:
        return None
    focal = exif.get('focal')
    mounts = db.mounts_for(camera)
    stated = canonical_maker(exif.get('lens_make'))
    best, best_score = None, 0.
    for query_text in queries:
        query = parse_name(query_text)
        named = {m for m in _THIRD_PARTY if m in query['words'] or m in str(query_text).lower()}
        makers = {canonical_maker(m) for m in named} or ({stated} if stated else set())
        for lens in db.lenses:
            score = _score(query, lens, makers, focal)
            if score <= 0:
                continue
            if mounts and set(lens.mounts) & mounts:
                score += .05
            if not makers and camera and canonical_maker(camera.maker) in {canonical_maker(m) for m in lens.makers}:
                score += .1
            if score > best_score + 1e-9:
                best, best_score = lens, score
    return best if best_score >= .6 else None


def candidates(match=None, db=None):
    """Lenses for the manual choice: those that fit the camera first."""
    db = db or database()
    mount = (match or {}).get('mount')
    mounts = {mount} | db.compat.get(mount, set()) if mount else None
    lenses = sorted(db.lenses, key=lambda l: (l.maker.lower(), l.model.lower()))
    if mounts:
        return [l for l in lenses if set(l.mounts) & mounts], lenses
    return lenses, lenses


# ---------------------------------------------------------------- solving

def _hermite(xs, ys, x):
    """Cubic Hermite (Catmull-Rom tangents) through ``(xs[i], ys[i])``; the nearest end outside."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = int(np.searchsorted(xs, x, side='right')) - 1
    i = min(max(i, 0), len(xs) - 2)
    y0, y1 = ys[i], ys[i + 1]
    m0 = (y1 - ys[i - 1]) / 2 if i > 0 else y1 - y0
    m1 = (ys[i + 2] - y0) / 2 if i + 2 < len(xs) else y1 - y0
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    t2, t3 = t * t, t * t * t
    return (2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * m1


def _along_focal(entries, focal, scaled):
    """Interpolate calibration terms over focal length; terms with ``scaled`` behave like 1/f."""
    unique = {}
    for f, model, terms in entries:
        unique.setdefault(f, (model, terms))
    xs = np.array(sorted(unique), float)
    models = {unique[f][0] for f in xs}
    model = unique[xs[0]][0]
    if len(models) > 1:
        xs = np.array([f for f in xs if unique[f][0] == model], float)
    scale = np.array([[f if s else 1. for s in scaled] for f in xs])
    ys = np.array([unique[f][1] for f in xs], float) * scale
    target = np.array([focal if s else 1. for s in scaled])
    return model, _hermite(xs, ys, focal) / target


def _vignetting_terms(lens, focal, aperture, distance):
    """Inverse distance weighting over (focal, 1/aperture, 1/distance), as calibrations are sparse."""
    span = (lens.focal[1] or focal) - (lens.focal[0] or focal)
    low = lens.focal[0] or focal
    def position(f, a, d):
        return np.array([(f - low) / span if span else 0., 4. / a, .1 / d])
    target = position(focal, aperture, distance)
    weights, values, nearest = [], [], math.inf
    for f, a, d, terms in lens.vignetting:
        gap = float(np.linalg.norm(position(f, a, d) - target))
        if gap < 1e-4:
            return np.array(terms, float)
        nearest = min(nearest, gap)
        weights.append(gap ** -3.5)
        values.append(terms)
    if not weights or nearest > 1:
        return None
    weights = np.asarray(weights)
    return (np.asarray(values, float) * weights[:, None]).sum(axis=0) / weights.sum()


def solve(lens, focal=None, aperture=None, distance=None, crop=None):
    """Profile of ``lens`` for one photo in frame units (see module docstring), or ``(None, reason)``."""
    if focal is None:
        if lens.focal[0] and (lens.focal[1] is None or _close(lens.focal[0], lens.focal[1], .001)):
            focal = lens.focal[0]
        else:
            return None, 'focal'
    crop = crop or lens.crop
    if crop / lens.crop < .96:
        # The calibration covers a smaller image circle than this sensor (lensfun's rule).
        return None, 'crop'
    distance = distance if distance and distance > 0 else FAR
    k = lens.crop * math.hypot(lens.aspect, 1.) / crop   # frame half diagonal / calibration half short edge
    q = lens.crop / crop                                   # frame half diagonal / calibration half diagonal
    profile = dict(maker=lens.maker, model=lens.model, label=lens.label, focal=float(focal),
                   aperture=float(aperture) if aperture else None, distance=float(distance), crop=float(crop),
                   center=[float(lens.center[0]), float(lens.center[1])], distortion=None, tca=None, vignetting=None)
    if lens.distortion:
        model, terms = _along_focal(lens.distortion, focal, (True,) * len(lens.distortion[0][2]))
        if model == 'ptlens':
            a, b, c = terms
        elif model == 'poly3':
            a, b, c = 0., terms[0], 0.
        else:
            a = b = c = None
        if model in ('ptlens', 'poly3'):
            d = 1 - a - b - c
            if abs(d) > .2:
                profile['distortion'] = [c * k / d ** 2, b * k ** 2 / d ** 3, a * k ** 3 / d ** 4, 0.]
        else:
            k1, k2 = terms
            profile['distortion'] = [0., k1 * k ** 2, 0., k2 * k ** 4]
    if lens.tca:
        _, terms = _along_focal(lens.tca, focal, (False, False, True, True, True, True))
        vr, vb, cr, cb, br, bb = terms
        profile['tca'] = [[vr, cr * k, br * k * k], [vb, cb * k, bb * k * k]]
    if lens.vignetting and aperture:
        terms = _vignetting_terms(lens, focal, aperture, distance)
        if terms is not None:
            k1, k2, k3 = terms
            profile['vignetting'] = [k1 * q ** 2, k2 * q ** 4, k3 * q ** 6]
    for key in ('distortion', 'tca', 'vignetting'):
        if profile[key] is not None:
            profile[key] = np.round(np.asarray(profile[key], float), 10).tolist()
    return profile, ''


def _signature(profile):
    return None if not profile else (profile.get('maker'), profile.get('model'), profile.get('focal'),
                                      profile.get('aperture'), profile.get('distance'), profile.get('crop'))


def prepare(settings, match, db=None):
    """``settings`` with its profile solved for the photo described by ``match`` (``identify``).

    Returns ``(settings, reason)``; the profile is kept when it already fits this photo, so
    reopening a project or undoing never changes it.  ``reason`` explains a missing profile."""
    result = {**defaults(), **copy.deepcopy(settings or {})}
    wanted = result['manual'] or (match or {}).get('lens')
    if not wanted:
        result['profile'] = None
        return result, 'nolens' if match and match.get('query') else 'noexif'
    match = match or {}
    lens = (db or database()).find(wanted.get('maker'), wanted.get('model'))
    if lens is None:
        result['profile'] = None
        return result, 'unknown'
    profile, reason = solve(lens, match.get('focal'), match.get('aperture'), match.get('distance'), match.get('crop'))
    if profile is None:
        result['profile'] = None
        return result, reason
    if _signature(result['profile']) != _signature(profile):
        result['profile'] = profile
    return result, ''


# ---------------------------------------------------------------- rendering

def active(settings):
    if not settings or not settings.get('enabled') or not settings.get('profile'):
        return False
    p = settings['profile']
    return bool((p.get('distortion') and settings.get('distortion', 100)) or (p.get('tca') and settings.get('chromatic', True))
                or (p.get('vignetting') and settings.get('vignetting', 100)))


def effective(settings):
    """The terms that actually render (amounts applied), or None."""
    if not active(settings):
        return None
    p = settings['profile']
    amount = settings.get('distortion', 100.) / 100
    distortion = [c * amount for c in p['distortion']] if p.get('distortion') and amount else None
    tca = p['tca'] if p.get('tca') and settings.get('chromatic', True) else None
    strength = settings.get('vignetting', 100.) / 100
    vignetting = (p['vignetting'], strength) if p.get('vignetting') and strength else None
    return dict(distortion=distortion, tca=tca, vignetting=vignetting, center=p.get('center', [0., 0.]))


def key(settings):
    terms = effective(settings)
    return None if terms is None else repr(sorted(terms.items()))


def _radial(rho, coefficients):
    c1, c2, c3, c4 = coefficients
    return 1 + rho * (c1 + rho * (c2 + rho * (c3 + rho * c4)))


def _frame(terms, width, height):
    cx = (width - 1) / 2 + terms['center'][0] * min(width, height) / 2
    cy = (height - 1) / 2 + terms['center'][1] * min(width, height) / 2
    return cx, cy, math.hypot(width, height) / 2


def _sample(terms, x, y, width, height, zoom):
    """Source positions (one per channel, or one shared) and the vignetting gain for output pixels.

    ``x`` and ``y`` broadcast: a row and a column of pixel coordinates give the whole grid.
    Float32 keeps 1/32-pixel accuracy up to 400-megapixel frames and doubles the speed."""
    cx, cy, radius = _frame(terms, width, height)
    u = ((np.asarray(x, np.float64) - cx) * (zoom / radius)).astype(np.float32)
    v = ((np.asarray(y, np.float64) - cy) * (zoom / radius)).astype(np.float32)
    rho = np.sqrt(u * u + v * v)
    if terms['distortion']:
        factor = _radial(rho, np.float32(terms['distortion']))
        u, v, rho = u * factor, v * factor, rho * np.abs(factor)
    scale = np.float32(radius)
    positions = [(cx + u * scale, cy + v * scale)]
    if terms['tca']:
        red, blue = ((t0 + rho * (t1 + rho * t2)) * scale for t0, t1, t2 in np.float32(terms['tca']))
        positions = [(cx + u * red, cy + v * red), positions[0], (cx + u * blue, cy + v * blue)]
    gain = None
    if terms['vignetting']:
        (v1, v2, v3), strength = terms['vignetting']
        r2 = rho * rho
        falloff = np.maximum(1 + r2 * (np.float32(v1) + r2 * (np.float32(v2) + r2 * np.float32(v3))), np.float32(.05))
        gain = falloff ** np.float32(-strength)
    return positions, gain


def fill_zoom(terms, width, height):
    """Largest zoom whose output frame still samples only inside the source frame."""
    if not terms['distortion'] and not terms['tca']:
        return 1.
    n = 129  # odd: the edge midpoints, where barrel corrections reach furthest, are sampled exactly
    t = np.linspace(0, 1, n)
    xs = np.concatenate([t * (width - 1), t * (width - 1), np.zeros(n), np.full(n, width - 1.)])
    ys = np.concatenate([np.zeros(n), np.full(n, height - 1.), t * (height - 1), t * (height - 1)])

    def inside(zoom):
        positions, _ = _sample(dict(terms, vignetting=None), xs, ys, width, height, zoom)
        return all(px.min() >= -.5 and px.max() <= width - .5 and py.min() >= -.5 and py.max() <= height - .5
                   for px, py in positions)
    low, high = .3, 2.
    if not inside(low):
        return low
    for _ in range(40):
        middle = (low + high) / 2
        low, high = (middle, high) if inside(middle) else (low, middle)
    return low


def apply(source, settings, area=None, out=None):
    """Corrected frame, or its block ``area`` (``engine.Area`` of the whole frame)."""
    terms = effective(settings)
    if terms is None:
        if area is None:
            return source
        return source[area.y0:area.y1, area.x0:area.x1]
    height, width = source.shape[:2]
    if area is None:
        x0, y0, x1, y1 = 0, 0, width, height
    else:
        x0, y0, x1, y1 = area.x0, area.y0, area.x1, area.y1
    if out is None:
        from . import large_image
        out = large_image.allocate((y1 - y0, x1 - x0, source.shape[2]), np.float32)
    zoom = fill_zoom(terms, width, height)
    rows = max(8, 2 ** 20 // max(1, x1 - x0))
    xs = np.arange(x0, x1, dtype=np.float64)[None, :]
    geometric = bool(terms['distortion'] or terms['tca'])
    for top in range(y0, y1, rows):
        bottom = min(y1, top + rows)
        positions, gain = _sample(terms, xs, np.arange(top, bottom, dtype=np.float64)[:, None], width, height, zoom)
        if geometric:
            maps = []
            for px, py in positions:
                px, py = np.broadcast_to(px, (bottom - top, x1 - x0)), np.broadcast_to(py, (bottom - top, x1 - x0))
                maps.append((np.round(px * SUBPIXEL) / SUBPIXEL, np.round(py * SUBPIXEL) / SUBPIXEL))
            sx0 = int(np.clip(min(np.floor(px.min()) for px, _ in maps) - 2, 0, width - 1))
            sy0 = int(np.clip(min(np.floor(py.min()) for _, py in maps) - 2, 0, height - 1))
            sx1 = int(np.clip(max(np.floor(px.max()) for px, _ in maps) + 4, sx0 + 1, width))
            sy1 = int(np.clip(max(np.floor(py.max()) for _, py in maps) + 4, sy0 + 1, height))
            block = np.asarray(source[sy0:sy1, sx0:sx1], np.float32)
            if len(maps) == 1:
                mx, my = maps[0]
                result = cv2.remap(block, (mx - sx0).astype(np.float32), (my - sy0).astype(np.float32),
                                   cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            else:
                result = np.stack([cv2.remap(np.ascontiguousarray(block[..., c]), (mx - sx0).astype(np.float32),
                                             (my - sy0).astype(np.float32), cv2.INTER_CUBIC,
                                             borderMode=cv2.BORDER_REPLICATE)
                                   for c, (mx, my) in enumerate(maps)], axis=2)
            np.maximum(result, 0, out=result)
        else:
            result = np.array(source[top:bottom, x0:x1], np.float32)
        if gain is not None:
            result *= np.broadcast_to(gain, result.shape[:2])[..., None]
        out[top - y0:bottom - y0] = result
    return out


class CorrectedView:
    """Blocks of the corrected frame on demand, for block renders that read around their area
    (retouching, neighbourhood tools) without correcting the whole original."""

    def __init__(self, source, settings):
        self.source, self.settings = source, settings
        self.shape, self.dtype = source.shape, np.float32

    def __getitem__(self, index):
        rows, columns = index
        height, width = self.shape[:2]
        y0, y1, _ = rows.indices(height)
        x0, x1, _ = columns.indices(width)
        from .engine import Area
        return apply(self.source, self.settings, Area(width, height, x0, y0, x1, y1))


def correct(source, edits):
    """Whole corrected frame for ``edits`` (``source`` itself when no correction is active)."""
    settings = edits.get('lens')
    return apply(source, settings) if active(settings) else source


def view(source, edits):
    settings = edits.get('lens')
    return CorrectedView(source, settings) if active(settings) else source


def validate(data):
    """Checked recipe entry; raises ValueError for malformed project data."""
    result = defaults()
    if data is None:
        return result
    if not isinstance(data, dict):
        raise ValueError('lens')

    def number(value, low, high):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('lens')
        return float(value)

    def text(value):
        if not isinstance(value, str) or len(value) > 200:
            raise ValueError('lens')
        return value
    result['enabled'] = bool(data.get('enabled', False))
    result['chromatic'] = bool(data.get('chromatic', True))
    result['distortion'] = number(data.get('distortion', 100.), 0, 200)
    result['vignetting'] = number(data.get('vignetting', 100.), 0, 200)
    manual = data.get('manual')
    if manual is not None:
        if not isinstance(manual, dict):
            raise ValueError('lens')
        result['manual'] = dict(maker=text(manual.get('maker')), model=text(manual.get('model')))
    p = data.get('profile')
    if p is not None:
        if not isinstance(p, dict):
            raise ValueError('lens')
        profile = dict(maker=text(p.get('maker')), model=text(p.get('model')), label=text(p.get('label', p.get('model'))),
                       focal=number(p.get('focal'), .1, 5000), crop=number(p.get('crop'), .05, 20),
                       aperture=None if p.get('aperture') is None else number(p['aperture'], .3, 256),
                       distance=number(p.get('distance', FAR), 1e-3, 1e6),
                       center=[number(v, -1, 1) for v in (p.get('center') or [0., 0.])][:2],
                       distortion=None, tca=None, vignetting=None)
        if len(profile['center']) != 2:
            raise ValueError('lens')
        if p.get('distortion') is not None:
            if len(p['distortion']) != 4:
                raise ValueError('lens')
            profile['distortion'] = [number(v, -20, 20) for v in p['distortion']]
        if p.get('tca') is not None:
            if len(p['tca']) != 2 or any(len(row) != 3 for row in p['tca']):
                raise ValueError('lens')
            profile['tca'] = [[number(row[0], .8, 1.2), number(row[1], -2, 2), number(row[2], -2, 2)] for row in p['tca']]
        if p.get('vignetting') is not None:
            if len(p['vignetting']) != 3:
                raise ValueError('lens')
            profile['vignetting'] = [number(v, -100, 100) for v in p['vignetting']]
        result['profile'] = profile
    return result
