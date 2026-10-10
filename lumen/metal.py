"""Apple silicon GPU development through Metal (macOS, 1.3.1).

On macOS the DirectML pixel graphs of ``gpu_graphs`` are replaced by one Metal
compute kernel.  It implements the same three stages (``tonal``, ``color`` and
``fused``) and consumes exactly the inputs produced by
``gpu_graphs.tonal_inputs`` / ``color_inputs``, so the parameter derivation is
shared with DirectML and the NumPy reference.  Like the CPU path it skips the
color mixer, identity curves and grading when they are neutral.

The kernel is compiled from source at first use with fast math disabled and is
checked once against the NumPy reference before it is trusted; any failure
leaves the CPU path in place.  Process version 2 (1.6.0) has its own kernel,
``develop2`` (``tone`` and ``color``), compiled and checked separately: if it
fails, only process-2 photos fall back to the CPU.  Apple silicon has unified memory: tiles are
copied into shared buffers without a PCIe transfer.  ``LUMEN_COMPUTE=cpu``
disables Metal.
"""
from __future__ import annotations

import functools
import logging
import platform
import struct
import sys
import threading

import numpy as np

from . import large_image
from .i18n import tr

log = logging.getLogger(__name__)

PROVIDER = 'MetalExecutionProvider'      # compute.state label of the Metal pixel kernels
APPLE = 0x106B                           # PCI vendor id, as in compute.dxgi_adapters()
STRIP_BYTES = 48 * 2**20                 # input bytes per dispatch; bounds the shared buffers
HUE_SAMPLES, CURVE_SAMPLES = 361, 4097
STAGES = {'tonal': 1, 'color': 2, 'fused': 3}
CURVES = ('curve_rgb', 'curve_r', 'curve_g', 'curve_b')
GRADES = ('grade_shadows', 'grade_midtones', 'grade_highlights')
IDENTITY = np.linspace(0, 1, CURVE_SAMPLES).astype(np.float32)   # gpu_graphs.color_inputs identity curve

SOURCE = r'''
#include <metal_stdlib>
using namespace metal;

// Mirrors gpu_graphs._tonal / _color / _hsl; see LUMEN RAW lumen/metal.py.
struct Params {
    float gain_r, gain_g, gain_b, exposure;
    float shadows, highlights, blacks, whites, contrast;
    float saturation, vibrance, mono, balance;
    float grade[9];          // shadows, midtones, highlights tint vectors (RGB)
    uint count;              // pixels in this dispatch
    uint stages;             // bit 0 tonal, bit 1 color
    uint hsl;                // apply the 8-color mixer
    uint curves;             // bit 0 RGB, 1 R, 2 G, 3 B
    uint grading;            // three-way grading active
};

constant float EPS = 1.1920928955078125e-07f;
constant uint HUE = 0, SAT = 361, VAL = 722, CURVE = 1083, CURVE_SAMPLES = 4097;

static inline float luma(float3 x) {
    return x.r * 0.2126f + x.g * 0.7152f + x.b * 0.0722f;
}

// Linear interpolation into a table on [0, high], as gpu_graphs._lookup.
static inline float lookup(device const float* table, float x, float high, float scale, uint samples) {
    float position = clamp(x, 0.0f, high) * scale;
    float index = clamp(floor(position), 0.0f, float(samples - 2));
    float fraction = position - index;
    uint i = uint(index);
    float left = table[i];
    return left + (table[i + 1] - left) * fraction;
}

static inline float3 tonal(float3 x, constant Params& p) {
    float3 exposed = x * float3(p.gain_r, p.gain_g, p.gain_b) * p.exposure;
    float q = precise::pow(clamp(luma(exposed), 0.0f, 1.0f), 0.45f);
    float inv = 1.0f - q;
    float inv2 = inv * inv, q3 = q * q * q;
    float middle = p.shadows * inv2 + p.highlights * q3;
    float ends = p.blacks * (inv2 * inv2 * inv2) + p.whites * (q3 * q3);
    float stops = middle / 65.0f + ends / 85.0f;
    float3 linear = max(exposed * precise::exp2(stops), float3(0.0f));
    float3 srgb;
    for (int c = 0; c < 3; ++c)
        srgb[c] = linear[c] <= 0.0031308f ? linear[c] * 12.92f
                                          : precise::pow(linear[c], 1.0f / 2.4f) * 1.055f - 0.055f;
    return clamp((srgb - 0.5f) * p.contrast + 0.5f, float3(0.0f), float3(1.0f));
}

static inline float3 mixer(float3 x, device const float* luts) {
    float r = x.r, g = x.g, b = x.b;
    float v = max(max(r, g), b);
    float diff = v - min(min(r, g), b);
    float s = diff / (fabs(v) + EPS);
    float k = 60.0f / (diff + EPS);
    float h = v == r ? (g - b) * k : (v == g ? (b - r) * k + 120.0f : (r - g) * k + 240.0f);
    if (h < 0.0f) h += 360.0f;
    float dh = lookup(luts + HUE, h, 360.0f, 1.0f, 361);
    float ds = lookup(luts + SAT, h, 360.0f, 1.0f, 361);
    float dv = lookup(luts + VAL, h, 360.0f, 1.0f, 361);
    float hue = h + dh;
    hue = hue - floor(hue / 360.0f) * 360.0f;
    s = clamp(s * (ds + 1.0f), 0.0f, 1.0f);
    v = clamp(v * precise::exp2(dv), 0.0f, 1.0f);
    float sector = hue / 60.0f, vs = v * s;
    float3 out;
    const float shift[3] = {5.0f, 3.0f, 1.0f};
    for (int c = 0; c < 3; ++c) {
        float t = sector + shift[c];
        t = t - floor(t / 6.0f) * 6.0f;
        out[c] = v - vs * clamp(min(t, 4.0f - t), 0.0f, 1.0f);
    }
    return out;
}

static inline float3 color(float3 x, constant Params& p, device const float* luts) {
    float lum = luma(x);
    float spread = max(max(x.r, x.g), x.b) - min(min(x.r, x.g), x.b);
    float scale = p.saturation + p.vibrance * (1.0f - spread);
    x = clamp(lum + (x - lum) * scale, float3(0.0f), float3(1.0f));
    if (p.hsl) x = mixer(x, luts);
    const float s = float(CURVE_SAMPLES - 1);
    if (p.curves & 1u)
        for (int c = 0; c < 3; ++c) x[c] = lookup(luts + CURVE, x[c], 1.0f, s, CURVE_SAMPLES);
    for (uint c = 0; c < 3; ++c)
        if (p.curves & (2u << c)) x[c] = lookup(luts + CURVE + CURVE_SAMPLES * (c + 1), x[c], 1.0f, s, CURVE_SAMPLES);
    if (p.mono != 0.0f) x = x + p.mono * (luma(x) - x);
    if (p.grading) {
        lum = luma(x);
        float tone = clamp(lum + p.balance, 0.0f, 1.0f);
        float inv = 1.0f - tone;
        float protection = 0.2f + 0.8f * precise::sin(clamp(lum, 0.0f, 1.0f) * M_PI_F);
        float3 tint = inv * inv * float3(p.grade[0], p.grade[1], p.grade[2])
                    + tone * inv * 2.0f * float3(p.grade[3], p.grade[4], p.grade[5])
                    + tone * tone * float3(p.grade[6], p.grade[7], p.grade[8]);
        x = x + protection * tint;
    }
    return clamp(x, float3(0.0f), float3(1.0f));
}

kernel void develop(device const packed_float3* source [[buffer(0)]],
                    device packed_float3* target [[buffer(1)]],
                    constant Params& p [[buffer(2)]],
                    device const float* luts [[buffer(3)]],
                    uint index [[thread_position_in_grid]]) {
    if (index >= p.count) return;
    float3 x = float3(source[index]);
    if (p.stages & 1u) x = tonal(x, p);
    if (p.stages & 2u) x = color(x, p, luts);
    target[index] = packed_float3(x);
}
'''

_PARAMS = struct.Struct('<13f9f5I')

SOURCE2 = r'''
#include <metal_stdlib>
using namespace metal;

// Process version 2: mirrors gpu_graphs._tonal2 / _color2; see LUMEN RAW lumen/metal.py.
struct Params2 {
    float gain_r, gain_g, gain_b;
    float shadows, highlights, blacks, whites, contrast, contrast_on;
    float saturation, vibrance, chroma_keep, balance;
    float grade[6];          // shadows, midtones, highlights OkLab (a, b) tints
    uint count;              // pixels in this dispatch
    uint stages;             // bit 0 tonal, bit 1 color
    uint local;              // per-pixel local tone coefficients present
    uint curves;             // bit 0 RGB, 1 R, 2 G, 3 B
    uint lab;                // OkLCh pass (saturation, mixer, monochrome, grading)
};

constant float EPS = 1.1920928955078125e-07f;
constant float GAMUT_EPS = 1e-6f;
constant float FLOOR = 1.52587890625e-05f;
constant uint TONE_SAMPLES = 65537, CURVE_SAMPLES = 4097, HUE_SAMPLES = 361;
constant float HEADROOM = 4.0f;
constant uint CURVE = 65537, SHIFT = 65537 + 4 * 4097, CHROMA = SHIFT + 361, LIGHT = SHIFT + 722, VIBRANCE = SHIFT + 1083;
constant int GAMUT_STEPS = 10;

static inline float luma(float3 x) {
    return x.r * 0.2126f + x.g * 0.7152f + x.b * 0.0722f;
}

static inline float lookup(device const float* table, float x, float high, float scale, uint samples) {
    float position = clamp(x, 0.0f, high) * scale;
    float index = clamp(floor(position), 0.0f, float(samples - 2));
    float fraction = position - index;
    uint i = uint(index);
    float left = table[i];
    return left + (table[i + 1] - left) * fraction;
}

static inline float to_linear(float x) {
    return x <= 0.04045f ? x / 12.92f : precise::pow(max((x + 0.055f) / 1.055f, 0.0f), 2.4f);
}

static inline float to_srgb(float x) {
    x = max(x, 0.0f);
    return x <= 0.0031308f ? x * 12.92f : precise::pow(x, 1.0f / 2.4f) * 1.055f - 0.055f;
}

static inline float signed_cbrt(float x) {
    return x < 0.0f ? -precise::pow(-x, 1.0f / 3.0f) : precise::pow(x, 1.0f / 3.0f);
}

static inline float3 to_lab(float3 c) {
    float l = 0.4122214708f * c.r + 0.5363325363f * c.g + 0.0514459929f * c.b;
    float m = 0.2119034982f * c.r + 0.6806995451f * c.g + 0.1073969566f * c.b;
    float s = 0.0883024619f * c.r + 0.2817188376f * c.g + 0.6299787005f * c.b;
    l = signed_cbrt(l); m = signed_cbrt(m); s = signed_cbrt(s);
    return float3(0.2104542553f * l + 0.7936177850f * m - 0.0040720468f * s,
                  1.9779984951f * l - 2.4285922050f * m + 0.4505937099f * s,
                  0.0259040371f * l + 0.7827717662f * m - 0.8086757660f * s);
}

static inline float3 from_lab(float3 lab) {
    float l = lab.x + 0.3963377774f * lab.y + 0.2158037573f * lab.z;
    float m = lab.x - 0.1055613458f * lab.y - 0.0638541728f * lab.z;
    float s = lab.x - 0.0894841775f * lab.y - 1.2914855480f * lab.z;
    l = l * l * l; m = m * m * m; s = s * s * s;
    return float3(4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s,
                  -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s,
                  -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s);
}

static inline bool inside(float3 c) {
    return min(min(c.r, c.g), c.b) >= -GAMUT_EPS && max(max(c.r, c.g), c.b) <= 1.0f + GAMUT_EPS;
}

static inline float excess(float3 c) {
    return max(max(max(c.r, c.g), c.b) - 1.0f, -min(min(c.r, c.g), c.b));
}

// Constant hue; move along the line towards the grey anchor (adaptive L0, alpha 0.4) until the
// colour fits in sRGB, then interpolate between the bracket ends (one secant step).
static inline float3 gamut(float3 c) {
    if (inside(c)) return clamp(c, float3(0.0f), float3(1.0f));
    float3 lab = to_lab(c);
    float2 ab = lab.yz;
    float offset = lab.x - 0.5f;
    float e1 = 0.5f + fabs(offset) + 0.4f * precise::sqrt(dot(ab, ab));
    float grey = 0.5f * (1.0f + sign(offset) * (e1 - precise::sqrt(max(e1 * e1 - 2.0f * fabs(offset), 0.0f))));
    float span = lab.x - grey;
    float low = 0.0f, high = 1.0f;
    float excess_low = excess(from_lab(float3(grey, 0.0f, 0.0f)));
    float excess_high = excess(from_lab(lab));
    for (int i = 0; i < GAMUT_STEPS; ++i) {
        float middle = (low + high) * 0.5f;
        float e = excess(from_lab(float3(grey + span * middle, ab * middle)));
        if (e <= GAMUT_EPS) { low = middle; excess_low = e; } else { high = middle; excess_high = e; }
    }
    float t = low + (high - low) * clamp(-excess_low / max(excess_high - excess_low, 1e-12f), 0.0f, 1.0f);
    return clamp(from_lab(float3(grey + span * t, ab * t)), float3(0.0f), float3(1.0f));
}

static inline float3 tonal2(float3 x, constant Params2& p, device const float* luts, float2 local) {
    const float scale = float(TONE_SAMPLES - 1) / HEADROOM;
    x = x * float3(p.gain_r, p.gain_g, p.gain_b);
    float y = luma(x);
    float base = precise::exp2(local.x * precise::log2(max(y, FLOOR)) + local.y);
    float q = precise::pow(clamp(lookup(luts, base, HEADROOM, scale, TONE_SAMPLES), 0.0f, 1.0f), 0.45f);
    float inv = 1.0f - q;
    float middle = p.shadows * inv * inv + p.highlights * q * q * q;
    float w = precise::pow(clamp(lookup(luts, y, HEADROOM, scale, TONE_SAMPLES), 0.0f, 1.0f), 0.45f);
    float wi = 1.0f - w;
    float ends = p.blacks * (wi * wi * wi) * (wi * wi * wi) + p.whites * (w * w * w) * (w * w * w);
    x = x * precise::exp2(middle / 65.0f + ends / 85.0f);
    y = luma(x);
    x = x * (lookup(luts, y, HEADROOM, scale, TONE_SAMPLES) / max(y, 1e-7f));
    if (p.contrast_on != 0.0f) {
        for (int c = 0; c < 3; ++c) {
            float encoded = sign(x[c]) * to_srgb(fabs(x[c]));
            float graded = (encoded - 0.5f) * p.contrast + 0.5f;
            x[c] = sign(graded) * to_linear(fabs(graded));
        }
    }
    x = gamut(x);
    return clamp(float3(to_srgb(x.r), to_srgb(x.g), to_srgb(x.b)), float3(0.0f), float3(1.0f));
}

static inline float3 color2(float3 x, constant Params2& p, device const float* luts) {
    const float s = float(CURVE_SAMPLES - 1);
    if (p.curves & 1u)
        for (int c = 0; c < 3; ++c) x[c] = lookup(luts + CURVE, x[c], 1.0f, s, CURVE_SAMPLES);
    for (uint c = 0; c < 3; ++c)
        if (p.curves & (2u << c)) x[c] = lookup(luts + CURVE + CURVE_SAMPLES * (c + 1), x[c], 1.0f, s, CURVE_SAMPLES);
    if (p.lab) {
        float3 lab = to_lab(float3(to_linear(x.r), to_linear(x.g), to_linear(x.b)));
        float lightness = lab.x;
        float chroma = precise::sqrt(lab.y * lab.y + lab.z * lab.z);
        float hue = precise::atan2(lab.z, lab.y) * (180.0f / M_PI_F);
        hue = hue - floor(hue / 360.0f) * 360.0f;
        float fading = 1.0f - min(chroma / 0.3f, 1.0f);
        float scale = p.saturation + p.vibrance * fading * lookup(luts + VIBRANCE, hue, 360.0f, 1.0f, HUE_SAMPLES);
        float weight = min(chroma / 0.04f, 1.0f);
        scale *= lookup(luts + CHROMA, hue, 360.0f, 1.0f, HUE_SAMPLES) + 1.0f;
        lightness *= precise::exp2(lookup(luts + LIGHT, hue, 360.0f, 1.0f, HUE_SAMPLES) * weight);
        hue += lookup(luts + SHIFT, hue, 360.0f, 1.0f, HUE_SAMPLES);
        chroma = chroma * max(scale, 0.0f) * p.chroma_keep;
        float radians = hue * (M_PI_F / 180.0f);
        float2 ab = float2(chroma * precise::cos(radians), chroma * precise::sin(radians));
        float level = clamp(lightness + p.balance, 0.0f, 1.0f);
        float inv = 1.0f - level;
        float protection = 0.2f + 0.8f * precise::sin(clamp(lightness, 0.0f, 1.0f) * M_PI_F);
        ab += protection * (inv * inv * float2(p.grade[0], p.grade[1])
                            + level * inv * 2.0f * float2(p.grade[2], p.grade[3])
                            + level * level * float2(p.grade[4], p.grade[5]));
        float3 mapped = gamut(from_lab(float3(lightness, ab)));
        x = float3(to_srgb(mapped.r), to_srgb(mapped.g), to_srgb(mapped.b));
    }
    return clamp(x, float3(0.0f), float3(1.0f));
}

kernel void develop2(device const packed_float3* source [[buffer(0)]],
                     device packed_float3* target [[buffer(1)]],
                     constant Params2& p [[buffer(2)]],
                     device const float* luts [[buffer(3)]],
                     device const float2* local [[buffer(4)]],
                     uint index [[thread_position_in_grid]]) {
    if (index >= p.count) return;
    float3 x = float3(source[index]);
    if (p.stages & 1u) x = tonal2(x, p, luts, p.local ? local[index] : float2(1.0f, 0.0f));
    if (p.stages & 2u) x = color2(x, p, luts);
    target[index] = packed_float3(x);
}
'''

_PARAMS2 = struct.Struct('<13f6f5I')
STAGES2 = {'tonal2': 1, 'color2': 2, 'fused2': 3, 'color2lab': 2, 'fused2lab': 3}
TONE_SAMPLES = 65537                     # tone.TABLE


class Unavailable(RuntimeError):
    pass


def macos_version():
    return platform.mac_ver()[0] or platform.release()


@functools.lru_cache(maxsize=1)
def device_info():
    """``(0, name, working-set bytes, vendor, macOS version)``, shaped like a DXGI adapter."""
    if sys.platform != 'darwin':
        return None
    try:
        import Metal
        device = Metal.MTLCreateSystemDefaultDevice()
    except Exception:
        log.debug('Metal device query failed', exc_info=True)
        return None
    if device is None:
        return None
    return (0, str(device.name()), int(device.recommendedMaxWorkingSetSize()), APPLE, macos_version())


def _scalar(inputs, name, default=0.):
    return float(np.asarray(inputs.get(name, default), np.float64).ravel()[0])


def parameters2(kind, inputs):
    """Pack the process-2 ``gpu_graphs`` inputs into ``develop2``'s parameter block and table."""
    stages = STAGES2[kind]
    color = bool(stages & 2)
    lab = color and kind.endswith('lab')
    gains = np.asarray(inputs.get('gains', np.ones(3)), np.float32).ravel()
    tone = np.asarray(inputs['tone_lut'], np.float32).ravel() if stages & 1 else np.zeros(TONE_SAMPLES, np.float32)
    curves = [np.asarray(inputs[k], np.float32).ravel() for k in CURVES] if color else [IDENTITY] * 4
    hues = [np.asarray(inputs[k], np.float32).ravel() for k in ('hue_shift', 'chroma_lut', 'light_lut', 'vibrance_lut')] \
        if lab else [np.zeros(HUE_SAMPLES, np.float32)] * 4
    grades = np.concatenate([np.asarray(inputs[f'grade_{z}'], np.float32).ravel()
                             for z in ('shadows', 'midtones', 'highlights')]) if lab else np.zeros(6, np.float32)
    if tone.size != TONE_SAMPLES or any(c.size != CURVE_SAMPLES for c in curves) or any(h.size != HUE_SAMPLES for h in hues):
        raise ValueError('unexpected lookup table size')
    local = np.asarray(inputs.get('local_ab', np.zeros((1, 1, 1, 2), np.float32)), np.float32)
    curve_bits = sum(1 << i for i, c in enumerate(curves) if not np.array_equal(c, IDENTITY))
    values = [*map(float, gains[:3]), _scalar(inputs, 'shadows'), _scalar(inputs, 'highlights'),
              _scalar(inputs, 'blacks'), _scalar(inputs, 'whites'), _scalar(inputs, 'contrast', 1.),
              _scalar(inputs, 'contrast_on'), _scalar(inputs, 'saturation', 1.), _scalar(inputs, 'vibrance'),
              _scalar(inputs, 'chroma_keep', 1.), _scalar(inputs, 'balance'), *map(float, grades),
              0, stages, int(local.shape[1:3] != (1, 1)), curve_bits, int(lab)]
    return values, np.ascontiguousarray(np.concatenate([tone] + curves + hues), np.float32), local


def parameters(kind, inputs, count=0):
    """Pack the ``gpu_graphs`` inputs of one stage into the kernel's parameter block and table."""
    stages = STAGES[kind]
    gains = np.asarray(inputs.get('gains', np.ones(3)), np.float32).ravel()
    color = bool(stages & 2)
    tables = [np.asarray(inputs[k], np.float32).ravel() for k in ('hue_lut', 'sat_lut', 'val_lut')] if color else []
    curves = [np.asarray(inputs[k], np.float32).ravel() for k in CURVES] if color else []
    grades = [np.asarray(inputs[k], np.float32).ravel() for k in GRADES] if color else [np.zeros(3, np.float32)] * 3
    if any(t.size != HUE_SAMPLES for t in tables) or any(c.size != CURVE_SAMPLES for c in curves):
        raise ValueError('unexpected lookup table size')
    hsl = int(any(np.any(t) for t in tables))
    curve_bits = sum(1 << i for i, c in enumerate(curves) if not np.array_equal(c, IDENTITY))
    grading = int(any(np.any(g) for g in grades))
    values = [*map(float, gains[:3]), _scalar(inputs, 'exposure', 1.),
              _scalar(inputs, 'shadows'), _scalar(inputs, 'highlights'), _scalar(inputs, 'blacks'),
              _scalar(inputs, 'whites'), _scalar(inputs, 'contrast', 1.),
              _scalar(inputs, 'saturation', 1.), _scalar(inputs, 'vibrance'), _scalar(inputs, 'mono'),
              _scalar(inputs, 'balance'), *map(float, np.concatenate(grades)),
              count, stages, hsl, curve_bits, grading]
    luts = np.concatenate(tables + curves) if color else np.zeros(4, np.float32)
    return values, np.ascontiguousarray(luts, np.float32)


class Pipeline:
    """The compiled ``develop`` kernel on the system default Metal device."""

    def __init__(self):
        if sys.platform != 'darwin':
            raise Unavailable(tr('Metal 仅在 macOS 上可用'))
        try:
            import Metal
            import objc
        except ImportError as exc:
            raise Unavailable(tr('缺少 PyObjC Metal 组件（pyobjc-framework-Metal）')) from exc
        self.Metal, self.objc = Metal, objc
        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            raise Unavailable(tr('系统没有可用的 Metal 设备'))
        options = Metal.MTLCompileOptions.alloc().init()
        options.setFastMathEnabled_(False)
        library, error = device.newLibraryWithSource_options_error_(SOURCE, options, None)
        if library is None:
            raise Unavailable(tr('Metal 着色器编译失败：{error}', error=error))
        function = library.newFunctionWithName_('develop')
        state, error = device.newComputePipelineStateWithFunction_error_(function, None)
        if state is None:
            raise Unavailable(tr('Metal 管线创建失败：{error}', error=error))
        self.device, self.state, self.queue = device, state, device.newCommandQueue()
        self.name = str(device.name())
        self.group = int(min(256, state.maxTotalThreadsPerThreadgroup()))
        self._lock = threading.Lock()
        self._buffers = (0, None, None)
        self._local = (0, None)
        # Process 2 (1.6.0): a separate library, so a fault there never disables process 1.
        self.state2, self.version2, self.error2 = None, False, ''
        library2, error = device.newLibraryWithSource_options_error_(SOURCE2, options, None)
        function2 = library2.newFunctionWithName_('develop2') if library2 is not None else None
        if function2 is not None:
            self.state2, error = device.newComputePipelineStateWithFunction_error_(function2, None)
        if self.state2 is None:
            self.error2 = str(error)
            log.warning('Metal process-2 kernel unavailable: %s', error)

    def _shared(self, length):
        buffer = self.device.newBufferWithLength_options_(length, self.Metal.MTLResourceStorageModeShared)
        if buffer is None:
            raise MemoryError(tr('Metal 无法分配 {v:.0f} MiB 共享内存', v=length / 2**20))
        return buffer

    def _strip_buffers(self, length):
        """Reuse the input / output pair between calls; previews keep the same size."""
        if self._buffers[0] < length:
            self._buffers = (length, self._shared(length), self._shared(length))
        return self._buffers[1], self._buffers[2]

    @staticmethod
    def _view(buffer, length):
        return np.frombuffer(buffer.contents().as_buffer(length), np.float32)

    def _dispatch(self, source, target, values, luts, count, local=None):
        values[-5] = count
        # Worker threads have no run loop: drain the command buffer and encoder every dispatch.
        with self.objc.autorelease_pool():
            command = self.queue.commandBuffer()
            encoder = command.computeCommandEncoder()
            encoder.setComputePipelineState_(self.state if local is None else self.state2)
            encoder.setBuffer_offset_atIndex_(source, 0, 0)
            encoder.setBuffer_offset_atIndex_(target, 0, 1)
            block = _PARAMS if local is None else _PARAMS2
            encoder.setBytes_length_atIndex_(block.pack(*values), block.size, 2)
            encoder.setBuffer_offset_atIndex_(luts, 0, 3)
            if local is not None:
                encoder.setBuffer_offset_atIndex_(local, 0, 4)
            groups = (count + self.group - 1) // self.group
            encoder.dispatchThreadgroups_threadsPerThreadgroup_((groups, 1, 1), (self.group, 1, 1))
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            if command.status() != self.Metal.MTLCommandBufferStatusCompleted:
                raise RuntimeError(tr('Metal 运算失败：{v}', v=command.error()))

    def run(self, kind, image, inputs):
        """Apply one stage to an (H, W, 3) image; large images run strip by strip."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(tr('Metal 显影需要 RGB 图像'))
        if kind in STAGES2:
            return self._run2(kind, image, inputs)
        values, table = parameters(kind, inputs)
        height, width = image.shape[:2]
        out = large_image.allocate(image.shape)
        rows = max(1, min(height, STRIP_BYTES // (width * 12)))
        length = rows * width * 12
        with self._lock, self.objc.autorelease_pool():
            source, target = self._strip_buffers(length)
            luts = self.device.newBufferWithBytes_length_options_(
                table.tobytes(), table.nbytes, self.Metal.MTLResourceStorageModeShared)
            source_view, target_view = self._view(source, length), self._view(target, length)
            for y in range(0, height, rows):
                block = image[y:y + rows]
                count = block.shape[0] * width
                np.copyto(source_view[:count * 3].reshape(block.shape), block, casting='same_kind')
                self._dispatch(source, target, values, luts, count)
                out[y:y + len(block)] = target_view[:count * 3].reshape(block.shape)
        return out

    def _run2(self, kind, image, inputs):
        if self.state2 is None:
            raise Unavailable(tr('Metal 着色器编译失败：{error}', error=self.error2))
        values, table, local = parameters2(kind, inputs)
        height, width = image.shape[:2]
        out = large_image.allocate(image.shape)
        rows = max(1, min(height, STRIP_BYTES // (width * 12)))
        length = rows * width * 12
        with self._lock, self.objc.autorelease_pool():
            source, target = self._strip_buffers(length)
            if self._local[0] < rows * width * 8:
                self._local = (rows * width * 8, self._shared(rows * width * 8))
            local_buffer = self._local[1]
            local_view = np.frombuffer(local_buffer.contents().as_buffer(self._local[0]), np.float32)
            luts = self.device.newBufferWithBytes_length_options_(
                table.tobytes(), table.nbytes, self.Metal.MTLResourceStorageModeShared)
            source_view, target_view = self._view(source, length), self._view(target, length)
            for y in range(0, height, rows):
                block = image[y:y + rows]
                count = block.shape[0] * width
                np.copyto(source_view[:count * 3].reshape(block.shape), block, casting='same_kind')
                if values[-3]:
                    np.copyto(local_view[:count * 2].reshape(block.shape[0], width, 2), local[0, y:y + rows])
                self._dispatch(source, target, values, luts, count, local_buffer)
                out[y:y + len(block)] = target_view[:count * 3].reshape(block.shape)
        return out

    def self_test2(self):
        """Process 2: the fused kernel with local tone mapping and every colour tool against NumPy."""
        from . import engine, gpu_graphs, model, tone
        rng = np.random.default_rng(160)
        image = (rng.random((43, 71, 3), dtype=np.float32) * 1.6 - .15).astype(np.float32)
        image[0, :4] = [[0, 0, 0], [1, 1, 1], [.5, .5, .5], [1.4, -.1, .05]]
        edits = model.recipe()
        edits['develop'] = dict(mode='camera', curve=[[0., 0.], [.2, .45], [.6, .9], [1., 1.]], source='')
        edits['adjustments'].update(exposure=.35, contrast=14, shadows=28, highlights=-33, blacks=-9,
                                    whites=11, temperature=-8, tint=5, saturation=12, vibrance=18)
        edits['hsl'][5] = [15., 25., -20.]
        edits['curves']['RGB'] = [[0., 0.], [.5, .55], [1., 1.]]
        edits['curve_mode'] = 'smooth'
        edits['grading'].update(shadows=[210., 25.], highlights=[45., 30.], balance=-12.)
        a = edits['adjustments']
        spec = tone.spec(edits, tone.context(image, a))
        inputs = dict(gpu_graphs.tonal2_inputs(a, spec, image.shape), **gpu_graphs.color2_inputs(edits))
        expected = engine.color_stage(tone.tonal(image, a, spec), edits)
        actual = self.run('fused2lab', image, inputs)
        error = np.abs(actual - expected) if np.isfinite(actual).all() else np.full(1, np.inf)
        if float(error.max()) > 5e-3 or float(error.mean()) > 1e-4:
            raise Unavailable(tr('Metal 自检结果与 CPU 参考不一致（最大误差 {error:.2g}）', error=float(error.max())))
        log.info('Metal process-2 kernel on %s verified (max error %.2g)', self.name, float(error.max()))
        self.version2 = True
        return float(error.max())

    def self_test(self):
        """Compare the fused kernel with the NumPy reference before trusting the device."""
        from . import engine, gpu_graphs, model
        rng = np.random.default_rng(131)
        image = rng.random((41, 67, 3), dtype=np.float32) * 1.25
        image[0, :4] = [[0, 0, 0], [1, 1, 1], [.5, .5, .5], [1, 0, 0]]
        edits = model.recipe()
        edits['process'] = 1
        edits['adjustments'].update(exposure=.45, contrast=18, shadows=32, highlights=-27, blacks=-12,
                                    whites=14, temperature=12, tint=-6, saturation=14, vibrance=-9)
        edits['hsl'][3] = [12., -25., 18.]
        edits['curves']['RGB'] = [[0., 0.], [.45, .52], [1., 1.]]
        edits['curves']['B'] = [[0., .03], [1., .97]]
        edits['curve_mode'] = 'smooth'
        edits['grading'].update(shadows=[215., 30.], highlights=[40., 45.], balance=10.)
        inputs = dict(gpu_graphs.tonal_inputs(edits['adjustments']), **gpu_graphs.color_inputs(edits))
        expected = engine.color_stage(engine.Backend._tonal(image, edits['adjustments'], np), edits)
        actual = self.run('fused', image, inputs)
        error = float(np.max(np.abs(actual - expected))) if np.isfinite(actual).all() else float('inf')
        if error > 1e-4:
            raise Unavailable(tr('Metal 自检结果与 CPU 参考不一致（最大误差 {error:.2g}）', error=error))
        log.info('Metal pipeline on %s verified (max error %.2g)', self.name, error)
        return error


_lock = threading.Lock()
_pipeline = None
_error = ''


def pipeline():
    """The shared, self-tested pipeline, or ``None`` when Metal cannot be used here."""
    global _pipeline, _error
    with _lock:
        if _pipeline is None and not _error:
            try:
                candidate = Pipeline()
                candidate.self_test()
                try:
                    candidate.self_test2()
                except Exception as exc:
                    candidate.error2 = str(exc) or type(exc).__name__
                    log.warning('Metal process-2 kernel disabled: %s', candidate.error2,
                                exc_info=not isinstance(exc, Unavailable))
                _pipeline = candidate
            except Exception as exc:
                _error = str(exc) or type(exc).__name__
                log.warning('Metal pixel pipeline unavailable: %s', _error,
                            exc_info=not isinstance(exc, Unavailable))
        return _pipeline


def last_error():
    return _error
