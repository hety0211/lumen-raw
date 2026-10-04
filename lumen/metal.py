"""Apple silicon GPU development through Metal (macOS, 1.3.1).

On macOS the DirectML pixel graphs of ``gpu_graphs`` are replaced by one Metal
compute kernel.  It implements the same three stages (``tonal``, ``color`` and
``fused``) and consumes exactly the inputs produced by
``gpu_graphs.tonal_inputs`` / ``color_inputs``, so the parameter derivation is
shared with DirectML and the NumPy reference.  Like the CPU path it skips the
color mixer, identity curves and grading when they are neutral.

The kernel is compiled from source at first use with fast math disabled and is
checked once against the NumPy reference before it is trusted; any failure
leaves the CPU path in place.  Apple silicon has unified memory: tiles are
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

    def _dispatch(self, source, target, values, luts, count):
        values[-5] = count
        # Worker threads have no run loop: drain the command buffer and encoder every dispatch.
        with self.objc.autorelease_pool():
            command = self.queue.commandBuffer()
            encoder = command.computeCommandEncoder()
            encoder.setComputePipelineState_(self.state)
            encoder.setBuffer_offset_atIndex_(source, 0, 0)
            encoder.setBuffer_offset_atIndex_(target, 0, 1)
            encoder.setBytes_length_atIndex_(_PARAMS.pack(*values), _PARAMS.size, 2)
            encoder.setBuffer_offset_atIndex_(luts, 0, 3)
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

    def self_test(self):
        """Compare the fused kernel with the NumPy reference before trusting the device."""
        from . import engine, gpu_graphs, model
        rng = np.random.default_rng(131)
        image = rng.random((41, 67, 3), dtype=np.float32) * 1.25
        image[0, :4] = [[0, 0, 0], [1, 1, 1], [.5, .5, .5], [1, 0, 0]]
        edits = model.recipe()
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
                _pipeline = candidate
            except Exception as exc:
                _error = str(exc) or type(exc).__name__
                log.warning('Metal pixel pipeline unavailable: %s', _error,
                            exc_info=not isinstance(exc, Unavailable))
        return _pipeline


def last_error():
    return _error
