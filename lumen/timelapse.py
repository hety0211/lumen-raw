"""Time-lapse videos from interval shots (2.0.0).

A sequence of photos (in capture order) becomes a video, one photo per frame (or several).  A few
**keyframes** carry the look: each is a photo of the 选片集 with its own edits.  Every other frame
gets a blend of the two keyframes around it, so a sunset can be brightened step by step as it
gets dark, the white balance can drift with the light, and a crop that differs between keyframes
pans and zooms smoothly.  Numbers are interpolated (hue on the colour wheel the short way round);
whatever cannot be blended (masks of another shape, a LUT file, curves with a different number of
points) comes from the nearer keyframe.  Before the first keyframe and after the last the look
stays as it is.

**Deflicker** evens out the brightness changes between neighbouring frames that interval shooting
leaves (aperture, auto exposure, light).  The brightness of every frame is measured on the
camera's rendering (the library thumbnail), smoothed over a few frames, and the difference is
added to the frame's exposure; slow changes such as a sunset are kept.  The camera's automatic
white balance jumps the same way: each RAW's white-balance multipliers (read without decoding)
are smoothed over the same frames and every frame is corrected towards them.

Frames are decoded no larger than the video needs (LibRaw's half-size decode when that is enough,
a few frames ahead in the background), rendered with the same pipeline as an export, centre-
cropped to the chosen aspect ratio and encoded with PyAV (FFmpeg): H.264, H.265 and AV1 in MP4
or ProRes 422 HQ in MOV, at a capped bit rate, BT.709 tagged.  The hardware encoders of the
graphics card are tried first (AMD AMF, NVIDIA NVENC, Intel Quick Sync, Windows Media Foundation)
and the CPU encoders (x264, x265, SVT-AV1) take over when none opens.
"""
from __future__ import annotations

import copy
import io
import logging
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .i18n import N_, tr

log = logging.getLogger(__name__)

#: Output formats: key → title, file suffix, container, hardware encoders, CPU encoder, pixel
#: formats (8-bit, 10-bit) and how the bit rate is relative to H.264 at the same size.
FORMATS = {
    'h264': dict(title='H.264 · MP4', suffix='.mp4', container='mp4',
                 hardware=('h264_amf', 'h264_nvenc', 'h264_qsv', 'h264_mf', 'h264_videotoolbox'),
                 software='libx264', pixel=('yuv420p', None), rate=1.),
    'hevc': dict(title='H.265 / HEVC · MP4', suffix='.mp4', container='mp4',
                 hardware=('hevc_amf', 'hevc_nvenc', 'hevc_qsv', 'hevc_mf', 'hevc_videotoolbox'),
                 software='libx265', pixel=('yuv420p', 'yuv420p10le'), rate=.6),
    'av1': dict(title='AV1 · MP4', suffix='.mp4', container='mp4', hardware=('av1_amf', 'av1_nvenc', 'av1_qsv'),
                software='libsvtav1', pixel=('yuv420p', 'yuv420p10le'), rate=.5),
    'prores': dict(title='Apple ProRes 422 HQ · MOV', suffix='.mov', container='mov', hardware=(), software='prores_ks',
                   pixel=(None, 'yuv422p10le'), rate=None),
}
#: Output sizes: key → title, long edge (None: the photos' own size).
RESOLUTIONS = (('original', N_('原始'), None), ('4k', '4K', 3840), ('2k', '2K', 2560), ('fhd', 'FHD', 1920))
#: Aspect ratios (centre crop): key → title, width / height (None: as the photos are, the
#: sensor's own ratio — the default).
ASPECTS = (('original', N_('原始比例（与传感器相同）'), None), ('16:9', '16:9', 16 / 9), ('4:3', '4:3', 4 / 3),
           ('1:1', '1:1', 1.), ('9:16', N_('9:16 竖屏'), 9 / 16))
FRAME_RATES = (24, 25, 30, 50, 60)
#: Deflicker strength: key → title, smoothing window in frames (0: off).
DEFLICKER = (('off', N_('关'), 0), ('light', N_('轻'), 5), ('medium', N_('中'), 11), ('strong', N_('强'), 25))
#: Largest frames the H.264 / H.265 levels allow (8K); ProRes and AV1 may be larger.
LEVEL_PIXELS = 8192 * 4320


def platform_hardware(names):
    """Hardware encoders worth probing on this host (2.1.0 Linux).

    AMF and Media Foundation only exist on Windows; VideoToolbox only on macOS.  Trying the
    others on Linux makes FFmpeg print load errors into the diagnostic log for no benefit.
    QSV is kept for Intel GPUs with the oneVPL runtime installed.
    """
    if os.name == 'nt':
        return list(names)
    if sys.platform == 'darwin':
        return [name for name in names if name.endswith('_videotoolbox')]
    return [name for name in names if name.endswith('_nvenc') or name.endswith('_qsv')]


#: Recipe keys that are never blended: they switch at the middle between keyframes.
DISCRETE = {'version', 'process', 'rotation', 'mode', 'kind', 'name', 'source', 'raster', 'enabled', 'invert',
            'monochrome', 'curve_mode', 'estimated', 'erase', 'manual', 'profile', 'auto'}
#: Exposure corrections of the deflicker never exceed this many stops.
DEFLICKER_LIMIT = 1.5


def default_mbps(fmt, width, height):
    """A maximum bit rate that keeps detailed landscapes clean (Mbit/s), or None (ProRes)."""
    rate = FORMATS[fmt]['rate']
    if rate is None:
        return None
    # 80 Mbit/s for 3840 × 2160 in H.264, in proportion to the pixels (FHD 20, 3:2 4K 95).
    h264 = 80 * width * height / (3840 * 2160)
    return int(min(400, max(8, round(h264 * rate / 5) * 5)))


# --------------------------------------------------------------------------- order and keyframes

def ordered(paths, records=None):
    """``paths`` in capture order (sub-second capture times of the library catalog when every
    photo has one, else by file name)."""
    records = records or {}
    times = [records.get(p, {}).get('captured_ts') for p in paths]
    if all(t is not None for t in times):
        return [p for _, p in sorted(zip(times, paths), key=lambda item: (item[0], Path(item[1]).name.lower()))]
    return sorted(paths, key=lambda p: (Path(p).name.lower(), p))


def _hue(path):
    return len(path) == 3 and path[0] == 'grading' and path[2] == 0


def mix(a, b, t, path=()):
    """``a`` blended towards ``b`` by ``t`` (0–1): numbers interpolate, everything else switches
    at ``t = 0.5``."""
    if t <= 0:
        return copy.deepcopy(a)
    if t >= 1:
        return copy.deepcopy(b)
    key = path[-1] if path else None
    if isinstance(a, bool) or isinstance(b, bool) or key in DISCRETE:
        return copy.deepcopy(a if t < .5 else b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if _hue(path):
            delta = (b - a + 180) % 360 - 180
            return float((a + delta * t) % 360)
        return float(a + (b - a) * t)
    if isinstance(a, dict) and isinstance(b, dict):
        result = {}
        for name in a:
            result[name] = mix(a[name], b[name], t, path + (name,)) if name in b else copy.deepcopy(a[name])
        for name in b:
            if name not in a and t >= .5:
                result[name] = copy.deepcopy(b[name])
        return result
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [mix(x, y, t, path + (i,)) for i, (x, y) in enumerate(zip(a, b))]
    return copy.deepcopy(a if t < .5 else b)


def recipe_at(index, keyframes):
    """The recipe of frame ``index`` from ``keyframes`` = ``{frame index: recipe}``."""
    from . import model
    if not keyframes:
        return model.recipe()
    marks = sorted(keyframes)
    if index <= marks[0]:
        return copy.deepcopy(keyframes[marks[0]])
    if index >= marks[-1]:
        return copy.deepcopy(keyframes[marks[-1]])
    after = next(m for m in marks if m >= index)
    before = max(m for m in marks if m <= index)
    if before == after:
        return copy.deepcopy(keyframes[before])
    t = (index - before) / (after - before)
    blended = mix(keyframes[before], keyframes[after], t)
    try:
        return model.validate(blended)
    except (ValueError, TypeError, KeyError):
        log.info('blend of keyframes %s and %s not valid at %s; using the nearer one', before, after, index)
        return copy.deepcopy(keyframes[before if t < .5 else after])


def default_keyframes(frames):
    """Indices of the keyframes when the user has not chosen: the first and last frame and every
    photo that has edits of its own."""
    if not frames:
        return []
    marks = {0, len(frames) - 1}
    marks.update(i for i, frame in enumerate(frames) if frame.get('edited'))
    return sorted(marks)


# --------------------------------------------------------------------------- deflicker

def brightness(rgb):
    """Log-mean linear luminance of an sRGB uint8 image (its "key"), in stops."""
    from .engine import to_linear
    small = np.asarray(rgb, np.float32) / 255
    linear = to_linear(small)
    y = linear @ np.array([.2126, .7152, .0722], np.float32)
    return float(np.mean(np.log2(np.clip(y, 1e-4, 1))))


def _smooth(values, window):
    half = window // 2
    return np.array([values[max(0, i - half):i + half + 1].mean(axis=0) for i in range(len(values))])


def deflicker(levels, window):
    """Exposure corrections (stops) that replace each frame's brightness by the average of its
    ``window`` neighbours: flicker goes, slow changes (a sunset) stay."""
    levels = np.asarray(levels, np.float64)
    if window <= 1 or len(levels) < 3:
        return [0.] * len(levels)
    smooth = _smooth(levels, window)
    return [float(np.clip(s - v, -DEFLICKER_LIMIT, DEFLICKER_LIMIT)) for s, v in zip(smooth, levels)]


#: Largest white-balance correction (per channel) the stabilisation applies.
WB_LIMIT = 1.25


def camera_multipliers(path):
    """The camera's white-balance multipliers of a RAW, ``[R, G, B]`` with G = 1, or None."""
    from . import engine, dng
    suffix = Path(path).suffix.lower()
    if suffix not in engine.RAW_EXTENSIONS or (suffix == '.dng' and dng.is_rendered(path)):
        return None
    try:
        import rawpy
        with rawpy.imread(str(path)) as raw:
            r, g, b = [float(v) for v in raw.camera_whitebalance[:3]]
    except Exception:
        return None
    if min(r, g, b) <= 0:
        return None
    return [r / g, 1., b / g]


def stabilise_white_balance(multipliers, window):
    """Per-frame ``wb_gain`` factors that take every frame's camera white balance to the average of
    its ``window`` neighbours (automatic white balance jumps go, slow drifts stay); 1 where a frame
    has no multipliers."""
    count = len(multipliers)
    factors = [[1., 1., 1.] for _ in range(count)]
    known = [i for i, m in enumerate(multipliers) if m]
    if window <= 1 or len(known) < 3:
        return factors
    logs = np.log(np.array([multipliers[i] for i in known], np.float64))
    smooth = _smooth(logs, window)
    for position, index in enumerate(known):
        ratio = np.exp(smooth[position] - logs[position])
        factors[index] = [float(np.clip(v, 1 / WB_LIMIT, WB_LIMIT)) for v in ratio]
    return factors


# --------------------------------------------------------------------------- sizes

def photo_size(path):
    """``(width, height)`` of a photo as displayed (orientation applied), without decoding it."""
    from . import engine, preview_cache
    found = preview_cache.default().cached(path)
    if found is not None:
        return int(found[1]['width']), int(found[1]['height'])
    suffix = Path(path).suffix.lower()
    if suffix in engine.RAW_EXTENSIONS:
        from . import dng
        if not (suffix == '.dng' and dng.is_rendered(path)):
            import rawpy
            with rawpy.imread(str(path)) as raw:
                w, h = raw.sizes.width, raw.sizes.height
                return (h, w) if raw.sizes.flip in (5, 6) else (w, h)
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        upright = ImageOps.exif_transpose(image)
        return upright.size


def frame_size(edits, size):
    """Size of a photo's rendered frame (crop, straightening and rotation applied)."""
    from . import engine
    width, height = engine.display_size(edits, size, final=True)
    return max(1, int(width)), max(1, int(height))


def aspect_box(width, height, aspect):
    """The largest ``aspect`` (w / h) rectangle centred in ``width × height``."""
    if not aspect:
        return width, height
    if width / height > aspect:
        return max(1, round(height * aspect)), height
    return width, max(1, round(width / aspect))


def output_size(width, height, long_edge=None, aspect=None, fmt='h264'):
    """Video frame size for frames of ``width × height``: cropped to ``aspect``, the long edge
    brought to ``long_edge`` (None: kept), even numbers, within what the codec's levels allow."""
    w, h = aspect_box(width, height, aspect)
    if aspect and long_edge:
        # Exact standard sizes: 16:9 at 4K is 3840 × 2160.
        w, h = (long_edge, long_edge / aspect) if aspect >= 1 else (long_edge * aspect, long_edge)
        scale = 1.
    else:
        scale = 1. if not long_edge else long_edge / max(w, h)
    if fmt in ('h264', 'hevc') and w * h * scale * scale > LEVEL_PIXELS:
        scale *= math.sqrt(LEVEL_PIXELS / (w * h * scale * scale))
        return max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)     # never above the level
    return max(2, round(w * scale / 2) * 2), max(2, round(h * scale / 2) * 2)


# --------------------------------------------------------------------------- rendering

def prepare_keyframe(path, edits, initialized):
    """A keyframe's recipe with the photo's own white balance, develop reference and lens profile
    when it was never opened (as opening it in the editor would set them)."""
    from . import lens, model, preview_cache
    edits = copy.deepcopy(edits) if edits else model.recipe()
    _, info = preview_cache.default().load(path)
    if not initialized:
        edits['white_balance'] = copy.deepcopy(info.get('white_balance', edits['white_balance']))
        edits['develop'] = copy.deepcopy(info.get('develop', edits['develop']))
    edits = model.validate(edits)
    edits['lens'] = lens.prepare(edits['lens'], info.get('lens_match'))[0]
    return edits


def decode(path, need):
    """``(rgb, info)`` with the long edge at least ``need`` pixels (or the full size): LibRaw's
    half-size decode when that is enough, which is about twice as fast."""
    from . import engine
    full = max(photo_size(path))
    if need * 2 <= full:
        return engine.load_image(path, int(math.ceil(need)))
    rgb, info = engine.load_image(path, None)
    if need < full:
        rgb = engine.resize_limit(rgb, int(math.ceil(need)))
    return rgb, info


def load(path, edits, size, aspect):
    """The decoded photo, no larger than frame ``size`` needs (runs ahead in the background)."""
    width, height = size
    full = photo_size(path)
    fw, fh = frame_size(edits, full)
    bw, bh = aspect_box(fw, fh, aspect)
    scale = max(width / bw, height / bh)
    return decode(path, max(full) * min(1., scale * 1.05))


def render(path, edits, size, aspect, backend, loaded=None):
    """Frame ``path`` rendered with ``edits`` at exactly ``size`` (sRGB 0–1); ``loaded`` is the
    result of ``load`` when it was decoded ahead."""
    from . import engine, lens
    width, height = size
    source, info = loaded or load(path, edits, size, aspect)
    edits = copy.deepcopy(edits)
    edits['lens'] = lens.prepare(edits.get('lens'), info.get('lens_match'))[0]
    detail = max(.05, source.shape[1] / max(1, info.get('width', source.shape[1])))
    result = engine.process(source, edits, backend, detail_scale=detail)
    del source
    rh, rw = result.shape[:2]
    cw, ch = aspect_box(rw, rh, aspect)
    x0, y0 = (rw - cw) // 2, (rh - ch) // 2
    result = result[y0:y0 + ch, x0:x0 + cw]
    shrink = width < cw
    result = cv2.resize(np.ascontiguousarray(result), (width, height),
                        interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC)
    return np.clip(result, 0, 1)


# --------------------------------------------------------------------------- encoding

class EncoderError(RuntimeError):
    pass


def _settings(codec, fmt, mbps, ten_bit, fps):
    """``(pixel format, bit rate, options)`` of an encoder."""
    eight, ten = FORMATS[fmt]['pixel']
    pixel = ten if (ten_bit and ten) or not eight else eight
    if codec.endswith('_qsv') or codec.endswith('_mf') or codec.endswith('_videotoolbox'):
        pixel = 'p010le' if pixel.endswith('10le') else 'nv12'
    elif codec.endswith('_amf') or codec.endswith('_nvenc'):
        pixel = 'p010le' if pixel.endswith('10le') else 'yuv420p'
    gop = str(max(1, int(fps * 2)))
    maximum = f'{mbps}M' if mbps else None
    # Hardware encoders aim at 60 % of the cap; the CPU encoders below are quality based (CRF).
    average = int(mbps * 1e6 * .6) if mbps else None
    if codec == 'libx264':
        options = dict(crf='17', preset='slow', profile='high', g=gop)
        if maximum:
            options.update(maxrate=maximum, bufsize=f'{mbps * 2}M')
        return pixel, None, options
    if codec == 'libx265':
        params = ['log-level=error', f'keyint={gop}']
        if maximum:
            params += [f'vbv-maxrate={mbps * 1000}', f'vbv-bufsize={mbps * 2000}']
        return pixel, None, {'crf': '19', 'preset': 'medium', 'x265-params': ':'.join(params)}
    if codec == 'libsvtav1':
        options = dict(crf='27', preset='8', g=gop)
        if maximum:
            options['maxrate'] = maximum
        return pixel, None, options
    if codec == 'prores_ks':
        return pixel, None, dict(profile='3', vendor='apl0')
    options = dict(g=gop)
    if codec.endswith('_amf'):
        options.update(rc='vbr_peak', quality='quality')
        if pixel == 'p010le' and codec.startswith('hevc'):
            options['profile'] = 'main10'
    elif codec.endswith('_nvenc'):
        options.update(rc='vbr', cq='19', preset='p6')
    elif codec.endswith('_mf'):
        options.update(rate_control='u_vbr' if maximum else 'quality')
    if maximum:
        options.update(maxrate=maximum, bufsize=f'{mbps * 2}M')
    return pixel, average, options


class Encoder:
    """Frames (float sRGB 0–1, ``size``) into a video file through PyAV."""

    def __init__(self, path, fmt, size, fps, mbps=None, ten_bit=False, hardware=True, metadata=None):
        os.environ.setdefault('SVT_LOG', '1')          # SVT-AV1: errors only
        import av
        import av.logging
        av.logging.set_level(av.logging.ERROR)
        self.av = av
        self.path, self.fmt, self.size, self.fps = Path(path), fmt, tuple(size), fps
        self.mbps, self.ten_bit = mbps, ten_bit
        spec = FORMATS[fmt]
        candidates = (platform_hardware(spec['hardware']) if hardware else []) + [spec['software']]
        errors = []
        for codec in candidates:
            try:
                self._probe(codec)
            except Exception as exc:
                errors.append(f'{codec}: {exc}')
                continue
            self.codec = codec
            break
        else:
            raise EncoderError(tr('没有可用的视频编码器：') + '；'.join(errors))
        self.part = self.path.with_name(self.path.stem + '.part' + self.path.suffix)
        self.container = av.open(str(self.part), 'w', format=spec['container'],
                                 options={'movflags': '+faststart'} if spec['container'] in ('mp4', 'mov') else {})
        for key, value in (metadata or {}).items():
            self.container.metadata[key] = value
        self.stream = self._stream(self.container, codec)
        self.frames = 0

    def _stream(self, container, codec):
        pixel, rate, options = _settings(codec, self.fmt, self.mbps, self.ten_bit, self.fps)
        stream = container.add_stream(codec, rate=self.fps, options=options)
        stream.width, stream.height = self.size
        stream.pix_fmt = pixel
        if rate:
            stream.bit_rate = rate
        context = stream.codec_context
        for attribute in ('color_primaries', 'color_trc', 'colorspace'):
            try:
                setattr(context, attribute, 1)            # BT.709
            except (AttributeError, ValueError, TypeError):
                pass
        if codec == 'libx265' or codec.startswith('hevc'):
            try:
                stream.codec_tag = 'hvc1'                 # plays in QuickTime and on Apple devices
            except Exception:
                pass
        self.pixel = pixel
        return stream

    def _probe(self, codec):
        """Open ``codec`` with the real settings and encode two frames into memory."""
        probe = self.av.open(io.BytesIO(), 'w', format=FORMATS[self.fmt]['container'])
        try:
            stream = self._stream(probe, codec)
            frame = np.zeros((self.size[1], self.size[0], 3), np.float32)
            for i in range(2):
                for packet in stream.encode(self._frame(frame, i)):
                    probe.mux(packet)
            for packet in stream.encode():
                probe.mux(packet)
        finally:
            try:
                probe.close()
            except Exception:
                pass

    def _frame(self, rgb, pts):
        deep = self.pixel.endswith('10le') or self.pixel == 'p010le'
        if deep:
            data = np.ascontiguousarray(np.round(np.clip(rgb, 0, 1) * 65535).astype(np.uint16))
            frame = self.av.VideoFrame.from_ndarray(data, format='rgb48le')
        else:
            data = np.ascontiguousarray(np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8))
            frame = self.av.VideoFrame.from_ndarray(data, format='rgb24')
        frame = frame.reformat(format=self.pixel, dst_colorspace='ITU709', interpolation='BICUBIC')
        frame.pts = pts
        return frame

    def write(self, rgb, repeat=1):
        for _ in range(max(1, repeat)):
            for packet in self.stream.encode(self._frame(rgb, self.frames)):
                self.container.mux(packet)
            self.frames += 1

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        os.replace(self.part, self.path)
        return self.path

    def abort(self):
        try:
            self.container.close()
        except Exception:
            pass
        try:
            self.part.unlink(missing_ok=True)
        except OSError:
            pass


# --------------------------------------------------------------------------- the whole job

def run(frames, options, backend, progress=lambda done, total, text: None, cancel=None):
    """Make the video.  ``frames`` = ``[{path, edits, initialized}]`` in order; ``options``:
    output, format, resolution (key), aspect (key), fps, repeat, mbps (None: default), ten_bit,
    hardware, deflicker (key), keyframes (frame indices; default ``default_keyframes``).

    Returns ``{output, frames, seconds, codec, size, mbps, keyframes}``."""
    cancel = cancel or threading.Event()
    started = time.perf_counter()
    total = len(frames)
    if total < 2:
        raise ValueError(tr('延时视频至少需要 2 张照片。'))
    fmt = options.get('format', 'h264')
    marks = sorted(set(options.get('keyframes') or default_keyframes(frames)))
    marks = [m for m in marks if 0 <= m < total] or [0]
    progress(0, total, tr('正在准备关键帧…'))
    keyframes = {m: prepare_keyframe(frames[m]['path'], frames[m].get('edits'), frames[m].get('initialized', False))
                 for m in marks}
    first = keyframes[marks[0]]
    width, height = frame_size(first, photo_size(frames[marks[0]]['path']))
    long_edge = dict((k, v) for k, _, v in RESOLUTIONS).get(options.get('resolution', '4k'))
    aspect = dict((k, v) for k, _, v in ASPECTS).get(options.get('aspect', 'original'))
    size = output_size(width, height, long_edge, aspect, fmt)
    mbps = options.get('mbps') or default_mbps(fmt, *size)
    if fmt == 'prores':
        mbps = None
    window = dict((k, v) for k, _, v in DEFLICKER).get(options.get('deflicker', 'off'), 0)
    offsets = [0.] * total
    balance = [[1., 1., 1.] for _ in range(total)]
    if window:
        from . import preview_cache
        cache = preview_cache.default()
        levels = []
        for index, frame in enumerate(frames):
            if cancel.is_set():
                raise InterruptedError(tr('已取消'))
            progress(0, total, tr('正在测量亮度（去闪烁）{done} / {total}', done=index + 1, total=total))
            from PIL import Image
            with Image.open(io.BytesIO(cache.thumbnail(frame['path']))) as thumb:
                levels.append(brightness(np.asarray(thumb.convert('RGB'))))
        offsets = deflicker(levels, window)
        balance = stabilise_white_balance([camera_multipliers(frame['path']) for frame in frames], window)
    repeat = max(1, int(options.get('repeat', 1)))
    from . import __version__
    encoder = Encoder(options['output'], fmt, size, options.get('fps', 30), mbps, bool(options.get('ten_bit')),
                      options.get('hardware', True), metadata=dict(comment=f'LUMEN RAW {__version__} time-lapse',
                                                                   encoder=f'LUMEN RAW {__version__}'))

    def prepare(index):
        edits = recipe_at(index, keyframes)
        if offsets[index]:
            exposure = edits['adjustments']['exposure'] + offsets[index]
            edits['adjustments']['exposure'] = float(np.clip(exposure, -5, 5))
        if balance[index] != [1., 1., 1.]:
            edits['wb_gain'] = [float(np.clip(g * f, .125, 8)) for g, f in zip(edits['wb_gain'], balance[index])]
        return edits

    # Decoding (LibRaw, about half the time of a frame) runs two frames ahead in the background;
    # every frame is rendered here, in order, on one backend, so frames never differ by device.
    pool = ThreadPoolExecutor(2, thread_name_prefix='timelapse-decode')
    try:
        ahead = {}

        def schedule(index):
            if index < total and index not in ahead and not cancel.is_set():
                edits = prepare(index)
                ahead[index] = (edits, pool.submit(load, frames[index]['path'], edits, size, aspect))
        for index in range(total):
            if cancel.is_set():
                raise InterruptedError(tr('已取消'))
            for following in (index, index + 1, index + 2):
                schedule(following)
            edits, decoded = ahead.pop(index)
            image = render(frames[index]['path'], edits, size, aspect, backend, decoded.result())
            encoder.write(image, repeat)
            elapsed = time.perf_counter() - started
            remaining = elapsed / (index + 1) * (total - index - 1)
            progress(index + 1, total, tr('正在生成视频 {done} / {total} · 编码器 {codec} · 剩余约 {minutes} 分钟',
                                          done=index + 1, total=total, codec=encoder.codec,
                                          minutes=max(1, round(remaining / 60))))
        path = encoder.close()
    except BaseException:
        encoder.abort()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return dict(output=str(path), frames=encoder.frames, seconds=round(encoder.frames / options.get('fps', 30), 2),
                codec=encoder.codec, size=list(size), mbps=mbps, keyframes=marks,
                elapsed=round(time.perf_counter() - started, 1))
