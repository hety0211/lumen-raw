"""Lightroom Classic / Camera Raw develop presets as LUMEN looks (1.7.0).

Reads ``.xmp`` presets (the ``crs:`` Camera Raw settings namespace, attribute or element form)
and the older ``.lrtemplate`` files (a Lua table, read by a small parser here — never run).
``convert`` maps the settings to LUMEN's recipe; the result is a *partial* look: only the
settings the preset holds are applied, as in Lightroom, and its strength can be set.

The mapping is an approximation (``docs/LIGHTROOM_PRESETS.md``).  LUMEN's sliders are not
Adobe's: process version 2 tone-maps scene-referred light with its own curves, colour works in
OkLCh, and Adobe's exact slider curves are not published.  The factors in ``SCALE`` were fitted
on test photographs (``tools/calibrate_lightroom.py``) so that a slider moves a photo about as
far as Adobe's does.  Settings LUMEN has no counterpart for — Adobe camera profiles and creative
looks, the calibration panel, masks and local adjustments, lens and transform settings … — are
listed in the import report (``notes``), never dropped silently.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import NamedTuple

import numpy as np

from . import model
from .i18n import tr, N_

CRS = 'http://ns.adobe.com/camera-raw-settings/1.0/'
RDF = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
LANG = '{http://www.w3.org/XML/1998/namespace}lang'
#: Largest preset file read; a preset with an embedded Adobe look table is a few hundred KB.
MAX_BYTES = 8 * 1024 * 1024

#: Adobe slider → LUMEN slider factors, fitted by ``tools/calibrate_lightroom.py`` on test photos
#: (process version 2).  See the module docstring.
#: Fitted 2026-10-17 on 8 raw.pixls.us PlayRaw photos at 1200 px, rounded to 0.05.  Highlights,
#: whites, blacks and grading are weaker in LUMEN than the Adobe targets: their factors match
#: moderate values and the strongest Adobe values clip at ±100.  ``gray`` (the black & white mix)
#: is not measured separately and uses the mixer's lightness factor.
SCALE = dict(contrast=.35, highlights=1.8, shadows=.75, whites=1.55, blacks=1.6, texture=.95, clarity=.45, dehaze=.65,
             vibrance=.7, saturation=.95, temperature=1., tint=1.35, hue=.5, hsl_saturation=.9, hsl_luminance=.6,
             gray=.6, grading=2.3, vignette=.45, grain=1.3, sharpness=.35, denoise=.85, color_noise=1.35)
#: Lift of a parametric tone-curve region at ±100, in units of the region's width.
PARAMETRIC_GAIN = .4
COLORS = ('Red', 'Orange', 'Yellow', 'Green', 'Aqua', 'Blue', 'Purple', 'Magenta')


class Preset(NamedTuple):
    name: str
    group: str
    settings: dict
    format: str             # 'xmp' or 'lrtemplate'


# --------------------------------------------------------------------------- reading

def _text(data):
    if isinstance(data, str):
        return data
    for encoding in ('utf-8-sig', 'gb18030'):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode('latin-1')


def _local(tag):
    return tag.rsplit('}', 1)[-1]


def _ns(tag):
    return tag[1:].split('}', 1)[0] if tag.startswith('{') else ''


def _struct(element):
    """crs: attributes and child elements of an rdf:Description / parseType=Resource element."""
    values = {_local(k): v for k, v in element.attrib.items() if _ns(k) == CRS}
    for child in element:
        if _ns(child.tag) == CRS:
            values[_local(child.tag)] = _value(child)
    return values


def _value(element):
    """The value of a crs: property element: text, a list (rdf:Seq / Bag), the default of an
    rdf:Alt, or a structure."""
    if element.get(f'{{{RDF}}}parseType') == 'Resource':
        return _struct(element)
    for child in element:
        kind = _local(child.tag) if _ns(child.tag) == RDF else ''
        if kind in ('Seq', 'Bag'):
            items = []
            for li in child:
                if li.get(f'{{{RDF}}}parseType') == 'Resource':
                    items.append(_struct(li))
                elif len(li) and _local(li[0].tag) == 'Description':
                    items.append(_struct(li[0]))
                elif any(_ns(k) == CRS for k in li.attrib):
                    items.append(_struct(li))
                else:
                    items.append((li.text or '').strip())
            return items
        if kind == 'Alt':
            items = list(child)
            chosen = next((li for li in items if li.get(LANG) == 'x-default'), items[0] if items else None)
            return '' if chosen is None else (chosen.text or '').strip()
        if kind == 'Description':
            return _struct(child)
    return (element.text or '').strip()


def read_xmp(data, fallback_name='', group=''):
    """A Preset from the bytes of a Camera Raw / Lightroom .xmp preset."""
    if len(data) > MAX_BYTES:
        raise ValueError(tr('预设文件过大。'))
    head = data if isinstance(data, str) else data.decode('latin-1')
    if re.search(r'<!(DOCTYPE|ENTITY)', head, re.IGNORECASE):
        raise ValueError(tr('无法读取这个 XMP 预设。'))
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        raise ValueError(tr('无法读取这个 XMP 预设。'))
    settings = {}
    # Only top-level descriptions: a look's or a mask's own settings are structures inside them.
    containers = [root] if root.tag == f'{{{RDF}}}RDF' else list(root.iter(f'{{{RDF}}}RDF'))
    for container in containers:
        for description in container:
            if description.tag == f'{{{RDF}}}Description':
                settings.update(_struct(description))
    if not settings:
        raise ValueError(tr('这个 XMP 文件中没有 Camera Raw 修图设置。'))
    if str(settings.get('PresetType', '')).lower() in ('look', 'profile'):
        raise ValueError(tr('这是 Adobe 配置文件（Profile），不是修图预设，无法导入。'))
    name = _alt(settings.get('Name')) or fallback_name or tr('Lightroom 预设')
    return Preset(name[:80], (_alt(settings.get('Group')) or group)[:80], settings, 'xmp')


def _alt(value):
    if isinstance(value, dict):
        return ''
    if isinstance(value, list):
        return str(value[0]) if value else ''
    return str(value or '').strip()


# --------------------------------------------------------------------------- .lrtemplate (Lua)

_TOKEN = re.compile(r'''
    (?P<space>\s+|--\[(?P<eq0>=*)\[.*?\](?P=eq0)\]|--[^\n]*)
  | (?P<long>\[(?P<eq>=*)\[(?P<body>.*?)\](?P=eq)\])
  | (?P<string>"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')
  | (?P<number>0[xX][0-9a-fA-F]+|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op>[{}\[\]=,;.()-])
''', re.VERBOSE | re.DOTALL)
_ESCAPES = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '"': '"', "'": "'", '\n': '\n', 'a': '\a', 'b': '\b',
            'f': '\f', 'v': '\v'}


def _unescape(body):
    out, i = [], 0
    while i < len(body):
        c = body[i]
        if c != '\\' or i + 1 >= len(body):
            out.append(c)
            i += 1
            continue
        nxt = body[i + 1]
        if nxt.isdigit():
            digits = re.match(r'\d{1,3}', body[i + 1:]).group()
            out.append(chr(int(digits) % 256))
            i += 1 + len(digits)
        else:
            out.append(_ESCAPES.get(nxt, nxt))
            i += 2
    text = ''.join(out)
    # Lua strings are bytes: \ddd escapes of UTF-8 text become characters again.
    try:
        return text.encode('latin-1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


class _Lua:
    """Recursive-descent reader of the Lua table syntax Lightroom writes (no code is executed)."""

    def __init__(self, text):
        self.tokens = []
        position = 0
        while position < len(text):
            match = _TOKEN.match(text, position)
            if not match:
                raise ValueError(tr('无法读取这个 Lightroom 模板。'))
            position = match.end()
            if match.group('space') is not None:
                continue
            if match.group('long') is not None:
                self.tokens.append(('string', match.group('body').removeprefix('\n')))
            elif match.group('string') is not None:
                self.tokens.append(('string', _unescape(match.group('string')[1:-1])))
            elif match.group('number') is not None:
                raw = match.group('number')
                self.tokens.append(('number', float(int(raw, 16)) if raw[:2].lower() == '0x' else float(raw)))
            elif match.group('name') is not None:
                self.tokens.append(('name', match.group('name')))
            else:
                self.tokens.append(('op', match.group('op')))
            if len(self.tokens) > 2_000_000:
                raise ValueError(tr('无法读取这个 Lightroom 模板。'))
        self.index = 0

    def peek(self, offset=0):
        i = self.index + offset
        return self.tokens[i] if i < len(self.tokens) else ('eof', None)

    def take(self, kind=None, value=None):
        token = self.peek()
        if (kind and token[0] != kind) or (value is not None and token[1] != value):
            raise ValueError(tr('无法读取这个 Lightroom 模板。'))
        self.index += 1
        return token

    def value(self, depth=0):
        if depth > 40:
            raise ValueError(tr('无法读取这个 Lightroom 模板。'))
        kind, token = self.peek()
        if kind == 'op' and token == '{':
            return self.table(depth)
        if kind == 'op' and token == '-':
            self.take()
            return -self.take('number')[1]
        if kind == 'op' and token == '(':
            self.take()
            result = self.value(depth + 1)
            self.take('op', ')')
            return result
        if kind in ('string', 'number'):
            self.take()
            return token
        if kind == 'name':
            self.take()
            if token == 'true':
                return True
            if token == 'false':
                return False
            if token == 'nil':
                return None
            # A call such as ZSTR "…" or LOC "…": its argument is the value.
            following = self.peek()
            if following[0] == 'string' or following == ('op', '{'):
                return self.value(depth + 1)
            if following == ('op', '('):
                self.take()
                result = self.value(depth + 1) if self.peek() != ('op', ')') else None
                while self.peek() == ('op', ','):
                    self.take()
                    self.value(depth + 1)
                self.take('op', ')')
                return result
            return token
        raise ValueError(tr('无法读取这个 Lightroom 模板。'))

    def table(self, depth):
        self.take('op', '{')
        items, fields = [], {}
        while self.peek() != ('op', '}'):
            if self.peek() == ('op', '['):
                self.take()
                key = self.value(depth + 1)
                self.take('op', ']')
                self.take('op', '=')
                fields[str(key) if not isinstance(key, float) else key] = self.value(depth + 1)
            elif self.peek()[0] == 'name' and self.peek(1) == ('op', '='):
                key = self.take()[1]
                self.take('op', '=')
                fields[key] = self.value(depth + 1)
            else:
                items.append(self.value(depth + 1))
            if self.peek()[0] == 'op' and self.peek()[1] in (',', ';'):
                self.take()
            elif self.peek() != ('op', '}'):
                raise ValueError(tr('无法读取这个 Lightroom 模板。'))
        self.take('op', '}')
        if fields and items:
            fields.update({float(i + 1): v for i, v in enumerate(items)})
            return fields
        return fields if fields else items


def parse_lua(text):
    """The first table of a .lrtemplate (``s = { … }`` or ``return { … }``)."""
    reader = _Lua(text)
    while reader.peek()[0] != 'eof' and reader.peek() != ('op', '{'):
        reader.take()
    if reader.peek()[0] == 'eof':
        raise ValueError(tr('无法读取这个 Lightroom 模板。'))
    return reader.value()


def _title(value):
    text = str(value or '').strip()
    if text.startswith('$$$/'):
        text = text.split('=', 1)[1] if '=' in text else text.rsplit('/', 1)[-1]
    return text


def read_lrtemplate(data, fallback_name='', group=''):
    """A Preset from a Lightroom 1–7 develop preset (.lrtemplate)."""
    if len(data) > MAX_BYTES:
        raise ValueError(tr('预设文件过大。'))
    table = parse_lua(_text(data))
    if not isinstance(table, dict):
        raise ValueError(tr('无法读取这个 Lightroom 模板。'))
    kind = str(table.get('type', 'Develop'))
    value = table.get('value') if isinstance(table.get('value'), dict) else {}
    settings = value.get('settings')
    if kind != 'Develop' or not isinstance(settings, dict):
        raise ValueError(tr('这不是 Lightroom 修图预设（类型：{kind}）。', kind=kind))
    name = _title(table.get('title')) or str(table.get('internalName') or '') or fallback_name or tr('Lightroom 预设')
    return Preset(name[:80], group[:80], {str(k): v for k, v in settings.items()}, 'lrtemplate')


# --------------------------------------------------------------------------- values

def _number(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    try:
        number = float(str(value).strip().replace('+', ''))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or '').strip().lower()
    return True if text in ('true', '1', 'yes') else False if text in ('false', '0', 'no') else None


def _points(value):
    """A point curve as [[x, y] …] in 0–1, from "x, y" strings (XMP) or a flat list (Lua)."""
    numbers = []
    if isinstance(value, dict):
        value = [value[k] for k in sorted(k for k in value if isinstance(k, float))]
    if not isinstance(value, list):
        return None
    for item in value:
        if isinstance(item, str):
            numbers += [_number(v) for v in item.split(',')]
        else:
            numbers.append(_number(item))
    if len(numbers) < 4 or len(numbers) % 2 or any(v is None for v in numbers):
        return None
    pairs = sorted((min(1., max(0., x / 255)), min(1., max(0., y / 255))) for x, y in zip(numbers[::2], numbers[1::2]))
    result = []
    for x, y in pairs:
        if not result or x - result[-1][0] >= .002:
            result.append([round(x, 5), round(y, 5)])
    if result[0][0] > 0:
        result.insert(0, [0., result[0][1]])
    if result[-1][0] < 1:
        result.append([1., result[-1][1]])
    result[0][0], result[-1][0] = 0., 1.
    return result if 2 <= len(result) <= 256 else None


def _clip(value, low=-100., high=100.):
    return float(min(high, max(low, value)))


def _signed(value, digits=0):
    text = f'{value:+.{digits}f}'
    return '0' if float(text) == 0 else text


def parametric_curve(shadows, darks, lights, highlights, splits=(25., 50., 75.), knots=65):
    """Lightroom's parametric region curve as dense knots (an approximation: each region bends the
    curve with a smooth bump over the region and half a region beyond it; ends stay fixed)."""
    s = sorted(min(95., max(5., float(v))) / 100 for v in splits)
    s[1] = max(s[1], s[0] + .02)
    s[2] = max(s[2], s[1] + .02)
    bounds = [0., *s, 1.]
    x = np.linspace(0, 1, knots)
    y = x.copy()
    for i, amount in enumerate((shadows, darks, lights, highlights)):
        if not amount:
            continue
        low, high = bounds[i], bounds[i + 1]
        width, centre = high - low, (low + high) / 2
        start, end = max(0., low - width / 2), min(1., high + width / 2)
        rise = np.clip((x - start) / max(centre - start, 1e-6), 0, 1)
        fall = np.clip((end - x) / max(end - centre, 1e-6), 0, 1)
        bump = np.where(x <= centre, .5 - .5 * np.cos(np.pi * rise), .5 - .5 * np.cos(np.pi * fall))
        y += amount / 100 * PARAMETRIC_GAIN * width * bump
    y = np.maximum.accumulate(np.clip(y, 0, 1))
    y[0], y[-1] = 0., 1.
    y = np.maximum.accumulate(y)
    return [[round(float(a), 5), round(float(b), 5)] for a, b in zip(x, y)]


@lru_cache(maxsize=1)
def _hue_table():
    from . import color
    hsv = np.arange(0, 360.25, .25)
    ok = np.unwrap(np.radians([color.ok_hue(h) for h in hsv]))
    return hsv, np.degrees(ok)


def _ok_angle(hsv_hue):
    hsv, ok = _hue_table()
    return float(np.interp(hsv_hue % 360, hsv, ok))


def _hsv_hue(ok_angle):
    hsv, ok = _hue_table()
    angle = (ok_angle - ok[0]) % 360 + ok[0]
    return float(np.interp(angle, ok, hsv) % 360)


def _add_tint(zone, extra):
    """Two colour-grading tints added as OkLab vectors: (hue, strength) of the sum."""
    vectors = [s * np.array([math.cos(math.radians(_ok_angle(h))), math.sin(math.radians(_ok_angle(h)))])
               for h, s in (zone, extra) if s > 0]
    if not vectors:
        return zone
    total = sum(vectors)
    strength = float(np.hypot(*total))
    if strength < 1e-6:
        return [zone[0], 0.]
    return [round(_hsv_hue(math.degrees(math.atan2(total[1], total[0]))), 2), round(min(100., strength), 2)]


# --------------------------------------------------------------------------- mapping

#: Description and file metadata: never reported.
METADATA = {
    'Version', 'ProcessVersion', 'PresetType', 'UUID', 'Name', 'ShortName', 'SortName', 'Group', 'Description',
    'Cluster', 'SupportsAmount', 'SupportsAmount2', 'SupportsColor', 'SupportsMonochrome', 'SupportsHighDynamicRange',
    'SupportsNormalDynamicRange', 'SupportsSceneReferred', 'SupportsOutputReferred', 'RequiresRGBTables',
    'CameraModelRestriction', 'Copyright', 'ContactInfo', 'HasSettings', 'AlreadyApplied', 'ToneCurveName',
    'ToneCurveName2012', 'ToneCurveName2010', 'CompatibleVersion', 'RawFileName', 'CameraProfileDigest',
    'OverrideLookVignette', 'LensProfileDigest', 'LensProfileFilename', 'LensProfileName', 'CropUnit',
    'CropWidth', 'CropHeight', 'Amount', 'AutoGrayscaleMix', 'ToneMapStrength', 'LookTable', 'RGBTable',
    'Table', 'UprightFourSegmentsCount', 'UprightPreview', 'UprightCenterMode', 'UprightFocalMode',
    'UprightTransformCount', 'UprightVersion', 'UprightDependentDigest', 'UprightCenterNormX', 'UprightCenterNormY',
    'UprightFocalLength35mm', 'UprightGuidedDependentDigest', 'CropConstrainToWarp', 'CropConstrainToUnitSquare',
    'LensProfileSetup', 'LensProfileIsEmbedded', 'DefringePurpleHueLo', 'DefringePurpleHueHi', 'DefringeGreenHueLo',
    'DefringeGreenHueHi', 'PresetVersion', 'EnableGrayscaleMix',
}

#: Settings with no LUMEN counterpart, by group (label), with Adobe's neutral value: reported
#: when a preset changes them.
UNMAPPED = {
    N_('校准面板'): dict(ShadowTint=0, RedHue=0, RedSaturation=0, GreenHue=0, GreenSaturation=0, BlueHue=0,
                      BlueSaturation=0),
    N_('镜头校正与去边'): dict(LensProfileEnable=0, AutoLateralCA=0, LensManualDistortionAmount=0, VignetteAmount=0,
                          VignetteMidpoint=50, DefringePurpleAmount=0, DefringeGreenAmount=0,
                          LensProfileDistortionScale=100, LensProfileVignettingScale=100,
                          LensProfileChromaticAberrationScale=100, ChromaticAberrationR=0, ChromaticAberrationB=0,
                          Defringe=0),
    N_('变换与几何'): dict(PerspectiveUpright=0, PerspectiveVertical=0, PerspectiveHorizontal=0, PerspectiveRotate=0,
                       PerspectiveScale=100, PerspectiveAspect=0, PerspectiveX=0, PerspectiveY=0),
    N_('裁切与方向'): dict(HasCrop=False, CropTop=0, CropLeft=0, CropBottom=1, CropRight=1, CropAngle=0, Orientation=1),
    N_('锐化半径、细节与蒙版'): dict(SharpenRadius=1, SharpenDetail=25, SharpenEdgeMasking=0),
    N_('降噪细节'): dict(LuminanceNoiseReductionDetail=50, LuminanceNoiseReductionContrast=0,
                     ColorNoiseReductionDetail=50, ColorNoiseReductionSmoothness=50),
    N_('暗角圆度、样式与高光'): dict(PostCropVignetteRoundness=0, PostCropVignetteStyle=1,
                           PostCropVignetteHighlightContrast=0),
    N_('颗粒粗糙度'): dict(GrainFrequency=50, GrainSeed=None),
    N_('色彩分级明亮度与混合'): dict(ColorGradeShadowLum=0, ColorGradeMidtoneLum=0, ColorGradeHighlightLum=0,
                           ColorGradeGlobalLum=0, ColorGradeBlending=50),
    N_('自动设置'): dict(AutoTone=False, AutoExposure=False, AutoContrast=False, AutoBrightness=False, AutoShadows=False,
                     AutoWhiteVersion=None),
    N_('AI 增强与降噪'): dict(EnhanceDetails=False, EnhanceDetailsVersion=None, EnhanceSuperResolution=False,
                          EnhanceDenoise=False, EnhanceDenoiseAmount=None),
    N_('旧版处理版本设置'): dict(Brightness=50),
}
#: Local adjustments and masks: reported with how many there are.
LOCAL = ('MaskGroupBasedCorrections', 'PaintBasedCorrections', 'GradientBasedCorrections',
         'CircularGradientBasedCorrections', 'RetouchAreas', 'RetouchInfo', 'RedEyeInfo', 'SpotRemoval')


def _neutral(value, default):
    if default is None:
        return False
    if isinstance(default, bool):
        return _bool(value) == default
    number = _number(value)
    return number is not None and abs(number - default) < 1e-6


def _zero(value):
    """A setting at a neutral value: zero, false or empty."""
    if value in (None, '', [], {}):
        return True
    if isinstance(value, bool):
        return not value
    number = _number(value)
    if number is not None:
        return number == 0
    return str(value).strip().lower() in ('false', 'none')


class _Reader:
    def __init__(self, settings):
        self.s, self.used = settings, set()

    def has(self, key):
        return key in self.s

    def raw(self, key):
        self.used.add(key)
        return self.s.get(key)

    def number(self, key):
        return _number(self.raw(key)) if key in self.s else None


def convert(settings):
    """``(look, notes, mapped)``: a partial LUMEN look, the (label, detail) pairs of settings that
    could not be carried over, and how many LUMEN settings the preset sets."""
    r = _Reader(settings)
    look, fields, notes = {}, set(), []
    adjust = model.adjustments()

    def slider(key, value):
        adjust[key] = round(_clip(value, -5, 5), 2) if key == 'exposure' else \
            round(_clip(value, 0 if key in ('sharpness', 'denoise', 'color_noise') else -100), 1)
        fields.add(f'adjustments.{key}')

    # Basic panel (process 2012 and later).
    for adobe, ours in (('Exposure2012', 'exposure'), ('Contrast2012', 'contrast'), ('Highlights2012', 'highlights'),
                        ('Shadows2012', 'shadows'), ('Whites2012', 'whites'), ('Blacks2012', 'blacks'),
                        ('Texture', 'texture'), ('Clarity2012', 'clarity'), ('Dehaze', 'dehaze'),
                        ('Vibrance', 'vibrance'), ('Saturation', 'saturation')):
        value = r.number(adobe)
        if value is not None:
            slider(ours, value * SCALE.get(ours, 1.))
    # Process 2003 / 2010 presets (Lightroom 1–3).
    legacy = (('Exposure', 'exposure', lambda v: v), ('Contrast', 'contrast', lambda v: (v - 25) * SCALE['contrast']),
              ('FillLight', 'shadows', lambda v: v * SCALE['shadows']),
              ('HighlightRecovery', 'highlights', lambda v: -v * SCALE['highlights']),
              ('Shadows', 'blacks', lambda v: -(v - 5) * 2 * SCALE['blacks']), ('Clarity', 'clarity', lambda v: v * SCALE['clarity']))
    for adobe, ours, mapping in legacy:
        value = r.number(adobe)
        if value is not None and f'adjustments.{ours}' not in fields:
            slider(ours, mapping(value))

    # White balance: Lightroom's Temperature / Tint are absolute for raw files (Kelvin), the
    # incremental ones apply to JPEG / TIFF.
    balance = str(r.raw('WhiteBalance') or '') if r.has('WhiteBalance') else ''
    kelvin, tint = r.number('Temperature'), r.number('Tint')
    step, step_tint = r.number('IncrementalTemperature'), r.number('IncrementalTint')
    if balance.lower() == 'auto':
        notes.append((N_('自动白平衡'), 'Auto'))
    elif balance.lower() == 'as shot' and kelvin is None and step is None:
        slider('temperature', 0)
        slider('tint', 0)
    if kelvin is not None and balance.lower() not in ('auto', 'as shot'):
        look['white_balance'] = dict(kelvin=float(min(25000., max(1500., kelvin))))
        fields.add('white_balance')
        notes.append((N_('绝对色温只对有相机色温的 RAW 照片生效'), f'{kelvin:.0f} K'))
    if step is not None:
        slider('temperature', step * SCALE['temperature'])
    if step_tint is not None:
        slider('tint', step_tint * SCALE['tint'])
    elif tint is not None and balance.lower() != 'auto':
        slider('tint', tint * SCALE['tint'])

    # Colour mixer, black & white mixer and treatment.
    gray = _bool(r.raw('ConvertToGrayscale')) if r.has('ConvertToGrayscale') else None
    if gray is not None:
        look['monochrome'] = gray
        fields.add('monochrome')
    mixer = [[0., 0., 0.] for _ in COLORS]
    found = False
    for index, name in enumerate(COLORS):
        hue, sat, lum = (r.number(f'{part}Adjustment{name}') for part in ('Hue', 'Saturation', 'Luminance'))
        grey = r.number(f'GrayMixer{name}')
        if gray:
            found |= grey is not None
            mixer[index][2] = round(_clip((grey or 0) * SCALE['gray']), 1)
        else:
            found |= any(v is not None for v in (hue, sat, lum))
            mixer[index] = [round(_clip((hue or 0) * SCALE['hue']), 1), round(_clip((sat or 0) * SCALE['hsl_saturation']), 1),
                            round(_clip((lum or 0) * SCALE['hsl_luminance']), 1)]
    if found:
        look['hsl'] = mixer
        fields.add('hsl')

    # Colour grading (Lightroom 10 keeps shadows and highlights in the split-toning fields).
    zones = dict(shadows=('SplitToningShadowHue', 'SplitToningShadowSaturation'),
                 midtones=('ColorGradeMidtoneHue', 'ColorGradeMidtoneSat'),
                 highlights=('SplitToningHighlightHue', 'SplitToningHighlightSaturation'))
    grading = model.grading()
    graded = False
    for zone, (hue_key, sat_key) in zones.items():
        hue, sat = r.number(hue_key), r.number(sat_key)
        for alternative in (f'ColorGrade{zone[:-1].capitalize()}Hue', f'ColorGrade{zone[:-1].capitalize()}Sat'):
            if r.has(alternative):
                value = r.number(alternative)
                if alternative.endswith('Hue'):
                    hue = value
                else:
                    sat = value
        if hue is not None or sat is not None:
            graded = True
            strength = round(_clip((sat or 0) * SCALE['grading'], 0, 100), 1)
            grading[zone] = [float((hue or 0) % 360) if strength else grading[zone][0], strength]
    global_hue, global_sat = r.number('ColorGradeGlobalHue'), r.number('ColorGradeGlobalSat')
    if global_sat:
        graded = True
        extra = [float((global_hue or 0) % 360), _clip(global_sat * SCALE['grading'], 0, 100)]
        for zone in zones:
            grading[zone] = _add_tint(grading[zone], extra)
    elif global_hue is not None:
        graded = True
    balance_value = r.number('SplitToningBalance')
    if balance_value is not None:
        graded = True
        grading['balance'] = _clip(balance_value)
    if graded:
        look['grading'] = grading
        fields.add('grading')

    # Tone curves: parametric regions as the fine tone curve, point curves as the RGB / channel curves.
    point = {}
    for suffix, channel in (('', 'RGB'), ('Red', 'R'), ('Green', 'G'), ('Blue', 'B')):
        for key in (f'ToneCurvePV2012{suffix}', f'ToneCurve{suffix}'):
            if r.has(key) and channel not in point:
                curve = _points(r.raw(key))
                if curve is None:
                    notes.append((N_('无法读取的色调曲线'), key))
                else:
                    point[channel] = curve
            elif r.has(key):
                r.raw(key)
    regions = [r.number(f'Parametric{k}') for k in ('Shadows', 'Darks', 'Lights', 'Highlights')]
    splits = [r.number(f'Parametric{k}Split') for k in ('Shadow', 'Midtone', 'Highlight')]
    if point or any(v is not None for v in regions + splits):
        curves = {c: point.get(c, [[0., 0.], [1., 1.]]) for c in ('RGB', 'R', 'G', 'B')}
        look['curves'] = curves
        look['curve_mode'] = 'smooth'
        look['tone_curve'] = parametric_curve(*[v or 0. for v in regions],
                                              splits=[s if s is not None else d for s, d in zip(splits, (25, 50, 75))]) \
            if any(regions) else [[0., 0.], [1., 1.]]
        fields.update(('curves', 'tone_curve'))

    # Detail.
    for adobe, ours in (('Sharpness', 'sharpness'), ('LuminanceSmoothing', 'denoise'), ('ColorNoiseReduction', 'color_noise')):
        value = r.number(adobe)
        if value is not None:
            slider(ours, value * SCALE[ours])

    # Effects.
    effects = model.effects()
    amount = r.number('PostCropVignetteAmount')
    if amount is not None:
        effects['vignette'] = _clip(amount * SCALE['vignette'])
        fields.add('effects.vignette')
        midpoint, feather = r.number('PostCropVignetteMidpoint'), r.number('PostCropVignetteFeather')
        if midpoint is not None:
            effects['midpoint'] = _clip(midpoint, 0)
            fields.add('effects.midpoint')
        if feather is not None:
            effects['feather'] = _clip(70 + (feather - 50) * .6, 0)
            fields.add('effects.feather')
    grain = r.number('GrainAmount')
    if grain is not None:
        effects['grain'] = _clip(grain * SCALE['grain'], 0)
        fields.add('effects.grain')
        size = r.number('GrainSize')
        if size is not None:
            effects['grain_size'] = _clip(size * 1.2, 0)
            fields.add('effects.grain_size')
    if any(f.startswith('adjustments.') for f in fields):
        look['adjustments'] = adjust
    if any(f.startswith('effects.') for f in fields):
        look['effects'] = effects

    # What is not carried over.
    profile = str(r.raw('CameraProfile') or '') if r.has('CameraProfile') else ''
    if profile:
        notes.append((N_('Adobe 相机配置文件'), profile))
    if r.has('Look'):
        value = r.raw('Look')
        name = _alt(value.get('Name')) if isinstance(value, dict) else _alt(value)
        # Lightroom names the profile twice: as CameraProfile and as the look that applies it.
        if (name or isinstance(value, dict)) and name != profile:
            notes.append((N_('Adobe 创意配置文件（Look）'), name or '?'))
    for key in LOCAL:
        if r.has(key):
            value = r.raw(key)
            count = len(value) if isinstance(value, (list, dict)) else (0 if _zero(value) else 1)
            if count:
                notes.append((N_('局部调整与蒙版'), f'{key} × {count}'))
    point_colors = r.raw('PointColors') if r.has('PointColors') else None
    if point_colors and not _zero(point_colors):
        notes.append((N_('点颜色'), str(len(point_colors) if isinstance(point_colors, list) else 1)))
    for label, defaults in UNMAPPED.items():
        changed = []
        for key, default in defaults.items():
            if r.has(key):
                value = r.raw(key)
                if not _neutral(value, default) and not (default is None and _zero(value)):
                    number = _number(value)
                    changed.append(f'{key} {_signed(number, 2 if abs(number) < 10 and number % 1 else 0)}'
                                   if number is not None else f'{key} {value}')
        if changed:
            notes.append((label, ', '.join(changed)))
    other = sorted(k for k in settings if k not in r.used and k not in METADATA and not k.startswith('Table_')
                   and not _zero(settings[k]))
    if other:
        notes.append((N_('其他未映射的设置'), ', '.join(other[:40]) + (' …' if len(other) > 40 else '')))
    if not fields:
        return None, notes, 0
    look['fields'] = sorted(fields)
    return model.clean_look(look), notes, len(fields)


def describe(notes):
    """Interface lines of an import report."""
    return [f'{tr(label)}：{detail}' for label, detail in notes]
