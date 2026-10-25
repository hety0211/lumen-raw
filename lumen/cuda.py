"""NVIDIA GPU development through CuPy (Linux, 2.1.0).

The Metal kernels of ``lumen.metal`` are the model for this module: the same three stages
(``tonal``, ``color`` and ``fused``) and the process-2 ``develop2`` kernel, fed by exactly the
inputs produced by ``gpu_graphs.tonal_inputs`` / ``color_inputs``.  Keeping the parameter
derivation in one place means the CUDA and Metal paths cannot drift apart.

The kernels are compiled with NVRTC at first use and checked against the NumPy reference before
they are trusted; any failure leaves the CPU path (or the ONNX CUDA graph path) in place.
Large frames run strip by strip so a small laptop GPU can develop a 60 MP photo without
allocating the whole float32 frame twice.  ``LUMEN_COMPUTE=cpu`` disables the pipeline, and
``LUMEN_COMPUTE=cuda`` makes a failed CUDA initialisation visible instead of silently using
the CPU.
"""
from __future__ import annotations

import logging
import struct
import sys
import threading

import numpy as np

from . import large_image
from .i18n import tr

log = logging.getLogger(__name__)

PROVIDER = 'CUDAExecutionProvider'
STRIP_BYTES = 48 * 2**20
THREADS = 256

SOURCE = r'''
struct F2 { float x, y; };
struct F3 { float x, y, z; };

__device__ __forceinline__ F2 f2(float x, float y) { F2 v = {x, y}; return v; }
__device__ __forceinline__ F3 f3(float x, float y, float z) { F3 v = {x, y, z}; return v; }
__device__ __forceinline__ F3 operator+(F3 a, F3 b) { return f3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ F3 operator-(F3 a, F3 b) { return f3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ F3 operator*(F3 a, F3 b) { return f3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ __forceinline__ F3 operator*(F3 a, float s) { return f3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ F3 operator/(F3 a, float s) { return f3(a.x / s, a.y / s, a.z / s); }
__device__ __forceinline__ float dot2(F2 a, F2 b) { return a.x * b.x + a.y * b.y; }
__device__ __forceinline__ float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
__device__ __forceinline__ F3 clamp3(F3 x, float lo, float hi) { return f3(clampf(x.x, lo, hi), clampf(x.y, lo, hi), clampf(x.z, lo, hi)); }
__device__ __forceinline__ float signf(float x) { return (float)((x > 0.0f) - (x < 0.0f)); }

struct Params {
    float gain_r, gain_g, gain_b, exposure;
    float shadows, highlights, blacks, whites, contrast;
    float saturation, vibrance, mono, balance;
    float grade[9];
    unsigned count, stages, hsl, curves, grading;
};

struct Params2 {
    float gain_r, gain_g, gain_b;
    float shadows, highlights, blacks, whites, contrast, contrast_on;
    float saturation, vibrance, chroma_keep, balance;
    float grade[6];
    unsigned count, stages, local, curves, lab;
};

#define EPS 1.1920928955078125e-07f
#define GAMUT_EPS 1e-6f
#define FLOOR 1.52587890625e-05f
#define HEADROOM 4.0f
#define PI_F 3.14159265358979323846f
#define C_HUE 0
#define C_SAT 361
#define C_VAL 722
#define C_CURVE 1083
#define CURVE_SAMPLES 4097
#define HUE_SAMPLES 361
#define TONE_SAMPLES 65537
#define CURVE2 65537
#define SHIFT2 (CURVE2 + 4 * 4097)
#define CHROMA2 (SHIFT2 + 361)
#define LIGHT2 (SHIFT2 + 722)
#define VIBRANCE2 (SHIFT2 + 1083)
#define GAMUT_STEPS 10

__device__ __forceinline__ float luma(F3 x) {
    return x.x * 0.2126f + x.y * 0.7152f + x.z * 0.0722f;
}

__device__ __forceinline__ float lookup(const float* table, float x, float high, float scale, unsigned samples) {
    float position = clampf(x, 0.0f, high) * scale;
    float index = clampf(floorf(position), 0.0f, (float)(samples - 2));
    float fraction = position - index;
    unsigned i = (unsigned)index;
    float left = table[i];
    return left + (table[i + 1] - left) * fraction;
}

__device__ __forceinline__ F3 tonal(F3 x, const Params& p) {
    F3 exposed = x * f3(p.gain_r, p.gain_g, p.gain_b) * p.exposure;
    float q = powf(clampf(luma(exposed), 0.0f, 1.0f), 0.45f);
    float inv = 1.0f - q;
    float inv2 = inv * inv, q3 = q * q * q;
    float middle = p.shadows * inv2 + p.highlights * q3;
    float ends = p.blacks * (inv2 * inv2 * inv2) + p.whites * (q3 * q3);
    float stops = middle / 65.0f + ends / 85.0f;
    F3 linear = exposed * exp2f(stops);
    linear.x = fmaxf(linear.x, 0.0f);
    linear.y = fmaxf(linear.y, 0.0f);
    linear.z = fmaxf(linear.z, 0.0f);
    float c[3] = {linear.x, linear.y, linear.z};
    for (int i = 0; i < 3; ++i)
        c[i] = c[i] <= 0.0031308f ? c[i] * 12.92f : powf(c[i], 1.0f / 2.4f) * 1.055f - 0.055f;
    return clamp3(f3((c[0] - 0.5f) * p.contrast + 0.5f,
                     (c[1] - 0.5f) * p.contrast + 0.5f,
                     (c[2] - 0.5f) * p.contrast + 0.5f), 0.0f, 1.0f);
}

__device__ __forceinline__ F3 mixer(F3 x, const float* luts) {
    float r = x.x, g = x.y, b = x.z;
    float v = fmaxf(fmaxf(r, g), b);
    float diff = v - fminf(fminf(r, g), b);
    float s = diff / (fabsf(v) + EPS);
    float k = 60.0f / (diff + EPS);
    float h = v == r ? (g - b) * k : (v == g ? (b - r) * k + 120.0f : (r - g) * k + 240.0f);
    if (h < 0.0f) h += 360.0f;
    float dh = lookup(luts + C_HUE, h, 360.0f, 1.0f, 361);
    float ds = lookup(luts + C_SAT, h, 360.0f, 1.0f, 361);
    float dv = lookup(luts + C_VAL, h, 360.0f, 1.0f, 361);
    float hue = h + dh;
    hue = hue - floorf(hue / 360.0f) * 360.0f;
    s = clampf(s * (ds + 1.0f), 0.0f, 1.0f);
    v = clampf(v * exp2f(dv), 0.0f, 1.0f);
    float sector = hue / 60.0f, vs = v * s;
    const float shift[3] = {5.0f, 3.0f, 1.0f};
    float out[3];
    for (int i = 0; i < 3; ++i) {
        float t = sector + shift[i];
        t = t - floorf(t / 6.0f) * 6.0f;
        out[i] = v - vs * clampf(fminf(t, 4.0f - t), 0.0f, 1.0f);
    }
    return f3(out[0], out[1], out[2]);
}

__device__ __forceinline__ F3 color(F3 x, const Params& p, const float* luts) {
    float lum = luma(x);
    float spread = fmaxf(fmaxf(x.x, x.y), x.z) - fminf(fminf(x.x, x.y), x.z);
    float scale = p.saturation + p.vibrance * (1.0f - spread);
    x = clamp3(f3(lum, lum, lum) + (x - f3(lum, lum, lum)) * scale, 0.0f, 1.0f);
    if (p.hsl) x = mixer(x, luts);
    const float s = (float)(CURVE_SAMPLES - 1);
    if (p.curves & 1u) {
        x.x = lookup(luts + C_CURVE, x.x, 1.0f, s, CURVE_SAMPLES);
        x.y = lookup(luts + C_CURVE, x.y, 1.0f, s, CURVE_SAMPLES);
        x.z = lookup(luts + C_CURVE, x.z, 1.0f, s, CURVE_SAMPLES);
    }
    if (p.curves & 2u) x.x = lookup(luts + C_CURVE + CURVE_SAMPLES, x.x, 1.0f, s, CURVE_SAMPLES);
    if (p.curves & 4u) x.y = lookup(luts + C_CURVE + CURVE_SAMPLES * 2, x.y, 1.0f, s, CURVE_SAMPLES);
    if (p.curves & 8u) x.z = lookup(luts + C_CURVE + CURVE_SAMPLES * 3, x.z, 1.0f, s, CURVE_SAMPLES);
    if (p.mono != 0.0f) x = x + (f3(luma(x), luma(x), luma(x)) - x) * p.mono;
    if (p.grading) {
        lum = luma(x);
        float tone = clampf(lum + p.balance, 0.0f, 1.0f);
        float inv = 1.0f - tone;
        float protection = 0.2f + 0.8f * sinf(clampf(lum, 0.0f, 1.0f) * PI_F);
        F3 tint = f3(p.grade[0], p.grade[1], p.grade[2]) * (inv * inv)
                + f3(p.grade[3], p.grade[4], p.grade[5]) * (tone * inv * 2.0f)
                + f3(p.grade[6], p.grade[7], p.grade[8]) * (tone * tone);
        x = x + tint * protection;
    }
    return clamp3(x, 0.0f, 1.0f);
}

extern "C" __global__ void develop(const float* source, float* target, const Params* p,
                                   const float* luts, unsigned total) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    float* s = (float*)source + index * 3;
    F3 x = f3(s[0], s[1], s[2]);
    if (p->stages & 1u) x = tonal(x, *p);
    if (p->stages & 2u) x = color(x, *p, luts);
    float* t = target + index * 3;
    t[0] = x.x; t[1] = x.y; t[2] = x.z;
}

__device__ __forceinline__ float to_linear(float x) {
    return x <= 0.04045f ? x / 12.92f : powf(fmaxf((x + 0.055f) / 1.055f, 0.0f), 2.4f);
}

__device__ __forceinline__ float to_srgb(float x) {
    x = fmaxf(x, 0.0f);
    return x <= 0.0031308f ? x * 12.92f : powf(x, 1.0f / 2.4f) * 1.055f - 0.055f;
}

__device__ __forceinline__ float signed_cbrt(float x) {
    return x < 0.0f ? -powf(-x, 1.0f / 3.0f) : powf(x, 1.0f / 3.0f);
}

__device__ __forceinline__ F3 to_lab(F3 c) {
    float l = 0.4122214708f * c.x + 0.5363325363f * c.y + 0.0514459929f * c.z;
    float m = 0.2119034982f * c.x + 0.6806995451f * c.y + 0.1073969566f * c.z;
    float s = 0.0883024619f * c.x + 0.2817188376f * c.y + 0.6299787005f * c.z;
    l = signed_cbrt(l); m = signed_cbrt(m); s = signed_cbrt(s);
    return f3(0.2104542553f * l + 0.7936177850f * m - 0.0040720468f * s,
              1.9779984951f * l - 2.4285922050f * m + 0.4505937099f * s,
              0.0259040371f * l + 0.7827717662f * m - 0.8086757660f * s);
}

__device__ __forceinline__ F3 from_lab(F3 lab) {
    float l = lab.x + 0.3963377774f * lab.y + 0.2158037573f * lab.z;
    float m = lab.x - 0.1055613458f * lab.y - 0.0638541728f * lab.z;
    float s = lab.x - 0.0894841775f * lab.y - 1.2914855480f * lab.z;
    l = l * l * l; m = m * m * m; s = s * s * s;
    return f3(4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s,
              -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s,
              -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s);
}

__device__ __forceinline__ bool inside(F3 c) {
    return fminf(fminf(c.x, c.y), c.z) >= -GAMUT_EPS && fmaxf(fmaxf(c.x, c.y), c.z) <= 1.0f + GAMUT_EPS;
}

__device__ __forceinline__ float excess(F3 c) {
    return fmaxf(fmaxf(fmaxf(c.x, c.y), c.z) - 1.0f, -fminf(fminf(c.x, c.y), c.z));
}

__device__ __forceinline__ F3 gamut(F3 c) {
    if (inside(c)) return clamp3(c, 0.0f, 1.0f);
    F3 lab = to_lab(c);
    F2 ab = f2(lab.y, lab.z);
    float offset = lab.x - 0.5f;
    float e1 = 0.5f + fabsf(offset) + 0.4f * sqrtf(dot2(ab, ab));
    float grey = 0.5f * (1.0f + signf(offset) * (e1 - sqrtf(fmaxf(e1 * e1 - 2.0f * fabsf(offset), 0.0f))));
    float span = lab.x - grey;
    float low = 0.0f, high = 1.0f;
    float excess_low = excess(from_lab(f3(grey, 0.0f, 0.0f)));
    float excess_high = excess(from_lab(lab));
    for (int i = 0; i < GAMUT_STEPS; ++i) {
        float middle = (low + high) * 0.5f;
        float e = excess(from_lab(f3(grey + span * middle, ab.x * middle, ab.y * middle)));
        if (e <= GAMUT_EPS) { low = middle; excess_low = e; } else { high = middle; excess_high = e; }
    }
    float t = low + (high - low) * clampf(-excess_low / fmaxf(excess_high - excess_low, 1e-12f), 0.0f, 1.0f);
    return clamp3(from_lab(f3(grey + span * t, ab.x * t, ab.y * t)), 0.0f, 1.0f);
}

__device__ __forceinline__ F3 tonal2(F3 x, const Params2& p, const float* luts, F2 local) {
    const float scale = (float)(TONE_SAMPLES - 1) / HEADROOM;
    x = x * f3(p.gain_r, p.gain_g, p.gain_b);
    float y = luma(x);
    float base = exp2f(local.x * log2f(fmaxf(y, FLOOR)) + local.y);
    float q = powf(clampf(lookup(luts, base, HEADROOM, scale, TONE_SAMPLES), 0.0f, 1.0f), 0.45f);
    float inv = 1.0f - q;
    float middle = p.shadows * inv * inv + p.highlights * q * q * q;
    float w = powf(clampf(lookup(luts, y, HEADROOM, scale, TONE_SAMPLES), 0.0f, 1.0f), 0.45f);
    float wi = 1.0f - w;
    float ends = p.blacks * (wi * wi * wi) * (wi * wi * wi) + p.whites * (w * w * w) * (w * w * w);
    x = x * exp2f(middle / 65.0f + ends / 85.0f);
    y = luma(x);
    x = x * (lookup(luts, y, HEADROOM, scale, TONE_SAMPLES) / fmaxf(y, 1e-7f));
    if (p.contrast_on != 0.0f) {
        float c[3] = {x.x, x.y, x.z};
        for (int i = 0; i < 3; ++i) {
            float encoded = signf(c[i]) * to_srgb(fabsf(c[i]));
            float graded = (encoded - 0.5f) * p.contrast + 0.5f;
            c[i] = signf(graded) * to_linear(fabsf(graded));
        }
        x = f3(c[0], c[1], c[2]);
    }
    x = gamut(x);
    return clamp3(f3(to_srgb(x.x), to_srgb(x.y), to_srgb(x.z)), 0.0f, 1.0f);
}

__device__ __forceinline__ F3 color2(F3 x, const Params2& p, const float* luts) {
    const float s = (float)(CURVE_SAMPLES - 1);
    if (p.curves & 1u) {
        x.x = lookup(luts + CURVE2, x.x, 1.0f, s, CURVE_SAMPLES);
        x.y = lookup(luts + CURVE2, x.y, 1.0f, s, CURVE_SAMPLES);
        x.z = lookup(luts + CURVE2, x.z, 1.0f, s, CURVE_SAMPLES);
    }
    if (p.curves & 2u) x.x = lookup(luts + CURVE2 + CURVE_SAMPLES, x.x, 1.0f, s, CURVE_SAMPLES);
    if (p.curves & 4u) x.y = lookup(luts + CURVE2 + CURVE_SAMPLES * 2, x.y, 1.0f, s, CURVE_SAMPLES);
    if (p.curves & 8u) x.z = lookup(luts + CURVE2 + CURVE_SAMPLES * 3, x.z, 1.0f, s, CURVE_SAMPLES);
    if (p.lab) {
        F3 lab = to_lab(f3(to_linear(x.x), to_linear(x.y), to_linear(x.z)));
        float lightness = lab.x;
        float chroma = sqrtf(lab.y * lab.y + lab.z * lab.z);
        float hue = atan2f(lab.z, lab.y) * (180.0f / PI_F);
        hue = hue - floorf(hue / 360.0f) * 360.0f;
        float fading = 1.0f - fminf(chroma / 0.3f, 1.0f);
        float scale = p.saturation + p.vibrance * fading * lookup(luts + VIBRANCE2, hue, 360.0f, 1.0f, HUE_SAMPLES);
        float weight = fminf(chroma / 0.04f, 1.0f);
        scale *= lookup(luts + CHROMA2, hue, 360.0f, 1.0f, HUE_SAMPLES) + 1.0f;
        lightness *= exp2f(lookup(luts + LIGHT2, hue, 360.0f, 1.0f, HUE_SAMPLES) * weight);
        hue += lookup(luts + SHIFT2, hue, 360.0f, 1.0f, HUE_SAMPLES);
        chroma = chroma * fmaxf(scale, 0.0f) * p.chroma_keep;
        float radians = hue * (PI_F / 180.0f);
        F2 ab = f2(chroma * cosf(radians), chroma * sinf(radians));
        float level = clampf(lightness + p.balance, 0.0f, 1.0f);
        float inv = 1.0f - level;
        float protection = 0.2f + 0.8f * sinf(clampf(lightness, 0.0f, 1.0f) * PI_F);
        ab.x += protection * (inv * inv * p.grade[0] + level * inv * 2.0f * p.grade[2] + level * level * p.grade[4]);
        ab.y += protection * (inv * inv * p.grade[1] + level * inv * 2.0f * p.grade[3] + level * level * p.grade[5]);
        F3 mapped = gamut(from_lab(f3(lightness, ab.x, ab.y)));
        x = f3(to_srgb(mapped.x), to_srgb(mapped.y), to_srgb(mapped.z));
    }
    return clamp3(x, 0.0f, 1.0f);
}

extern "C" __global__ void develop2(const float* source, float* target, const Params2* p,
                                    const float* luts, const float* local, unsigned total) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const float* s = source + index * 3;
    F3 x = f3(s[0], s[1], s[2]);
    F2 l = p->local ? f2(local[index * 2], local[index * 2 + 1]) : f2(1.0f, 0.0f);
    if (p->stages & 1u) x = tonal2(x, *p, luts, l);
    if (p->stages & 2u) x = color2(x, *p, luts);
    float* t = target + index * 3;
    t[0] = x.x; t[1] = x.y; t[2] = x.z;
}
'''


class Unavailable(RuntimeError):
    pass


def device_info():
    """``(0, name, total bytes, vendor, driver)`` shaped like a DXGI adapter."""
    try:
        import cupy as cp
        if not cp.cuda.runtime.getDeviceCount():
            return None
        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties['name']
        name = name.decode() if isinstance(name, bytes) else str(name)
        total = int(properties['totalGlobalMem'])
        return (0, name, total, 0x10DE, '')
    except Exception:
        log.debug('CUDA device query failed', exc_info=True)
        return None


class Pipeline:
    """The compiled ``develop`` / ``develop2`` kernels on CUDA device 0."""

    def __init__(self):
        if not sys.platform.startswith('linux'):
            raise Unavailable(tr('CUDA 内核仅在 Linux 上启用'))
        try:
            import cupy as cp
        except ImportError as exc:
            log.warning('CuPy import failed: %s', exc, exc_info=True)
            raise Unavailable(tr('缺少 CuPy（cupy-cuda13x）')) from exc
        if not cp.cuda.runtime.getDeviceCount():
            raise Unavailable(tr('系统没有可用的 CUDA 设备'))
        self.cp = cp
        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties['name']
        self.name = name.decode() if isinstance(name, bytes) else str(name)
        try:
            module = cp.RawModule(code=SOURCE, options=('-std=c++14',))
            self.develop = module.get_function('develop')
            self.develop2 = module.get_function('develop2')
        except Exception as exc:
            raise Unavailable(tr('CUDA 内核编译失败：{error}', error=exc)) from exc
        self._module = module
        self.group = THREADS
        self._lock = threading.Lock()
        self.version2 = True
        self.error2 = ''

    def _apply(self, kernel, image, values, table, local=None):
        cp = self.cp
        from . import metal
        height, width = image.shape[:2]
        out = large_image.allocate(image.shape)
        rows = max(1, min(height, STRIP_BYTES // max(1, width * 12)))
        block_struct = metal._PARAMS if local is None else metal._PARAMS2
        table_gpu = cp.asarray(table, dtype=cp.float32)
        params_gpu = cp.asarray(np.frombuffer(block_struct.pack(*values), dtype=np.uint8))
        with self._lock:
            for y in range(0, height, rows):
                block = np.ascontiguousarray(image[y:y + rows], dtype=np.float32)
                source = cp.asarray(block)
                target = cp.empty_like(source)
                count = block.shape[0] * width
                grid = ((count + self.group - 1) // self.group,)
                if local is None:
                    kernel(grid, (self.group,), (source, target, params_gpu, table_gpu, count))
                else:
                    local_block = np.ascontiguousarray(local[0, y:y + rows], dtype=np.float32)
                    local_gpu = cp.asarray(local_block)
                    kernel(grid, (self.group,), (source, target, params_gpu, table_gpu, local_gpu, count))
                out[y:y + len(block)] = cp.asnumpy(target)
        return out

    def run(self, kind, image, inputs):
        """Apply one stage to an (H, W, 3) image; large images run strip by strip."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(tr('CUDA 显影需要 RGB 图像'))
        from . import metal
        if kind in metal.STAGES2:
            return self._run2(kind, image, inputs)
        values, table = metal.parameters(kind, inputs)
        return self._apply(self.develop, image, values, table)

    def _run2(self, kind, image, inputs):
        if not self.version2:
            raise Unavailable(tr('CUDA 第二阶段内核不可用：{error}', error=self.error2))
        from . import metal
        values, table, local = metal.parameters2(kind, inputs)
        return self._apply(self.develop2, image, values, table, local)

    def self_test2(self):
        """Process 2: local tone mapping and every colour tool against the NumPy reference."""
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
            raise Unavailable(tr('CUDA 自检结果与 CPU 参考不一致（最大误差 {error:.2g}）', error=float(error.max())))
        log.info('CUDA process-2 kernel on %s verified (max error %.2g)', self.name, float(error.max()))
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
            raise Unavailable(tr('CUDA 自检结果与 CPU 参考不一致（最大误差 {error:.2g}）', error=error))
        log.info('CUDA pipeline on %s verified (max error %.2g)', self.name, error)
        return error


_lock = threading.Lock()
_pipeline = None
_error = ''


def pipeline():
    """The shared, self-tested CUDA pipeline, or ``None`` when CUDA cannot be used here."""
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
                    candidate.version2 = False
                    log.warning('CUDA process-2 kernel disabled: %s', candidate.error2,
                                exc_info=not isinstance(exc, Unavailable))
                _pipeline = candidate
            except Exception as exc:
                _error = str(exc) or type(exc).__name__
                log.debug('CUDA pipeline initialisation failed', exc_info=True)
                log.warning('CUDA pixel pipeline unavailable: %s', _error,
                            exc_info=not isinstance(exc, Unavailable))
        return _pipeline


def last_error():
    return _error
