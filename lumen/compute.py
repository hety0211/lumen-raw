"""ONNX execution on a real GPU (Windows DirectML / TensorRT, macOS Core ML), with a CPU fallback."""
from __future__ import annotations

import collections
import ctypes
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import types
from pathlib import Path

import numpy as np

from . import performance
from .i18n import tr

log = logging.getLogger(__name__)


class _Luid(ctypes.Structure):
    _fields_ = [('low', ctypes.c_uint32), ('high', ctypes.c_int32)]


class _Guid(ctypes.Structure):
    _fields_ = [('data1', ctypes.c_uint32), ('data2', ctypes.c_uint16),
                ('data3', ctypes.c_uint16), ('data4', ctypes.c_ubyte * 8)]


class _AdapterDesc(ctypes.Structure):
    _fields_ = [('name', ctypes.c_wchar * 128), ('vendor', ctypes.c_uint32),
                ('device', ctypes.c_uint32), ('subsystem', ctypes.c_uint32),
                ('revision', ctypes.c_uint32), ('dedicated', ctypes.c_size_t),
                ('system', ctypes.c_size_t), ('shared', ctypes.c_size_t),
                ('luid', _Luid), ('flags', ctypes.c_uint32)]


_IID_DXGI_DEVICE = (0x54ec77fa, 0x1377, 0x44e6, (0x8c, 0x32, 0x88, 0xfd, 0x5f, 0x44, 0xc8, 0x4c))


def format_driver(value):
    """UMD version from IDXGIAdapter::CheckInterfaceSupport, e.g. 32.0.16.1088."""
    value &= (1 << 64) - 1
    return '.'.join(str(value >> shift & 0xFFFF) for shift in (48, 32, 16, 0))


def dxgi_adapters():
    """(index, name, dedicated bytes, vendor id, driver version) for hardware adapters.

    DirectML device_id is the DXGI adapter index; prefer dedicated VRAM.
    """
    if os.name != 'nt':
        return []
    factory = ctypes.c_void_p()
    iid = _Guid(0x770aae78, 0xf26f, 0x4dba,
                (ctypes.c_ubyte * 8)(0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87))
    dll = ctypes.WinDLL('dxgi.dll')
    create = dll.CreateDXGIFactory1
    create.argtypes = (ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p))
    create.restype = ctypes.c_long
    if create(ctypes.byref(iid), ctypes.byref(factory)) < 0:
        return []

    def method(ptr, index, result, *arguments):
        vtable = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *arguments)(vtable[index])

    adapters = []
    try:
        enum = method(factory, 12, ctypes.c_long, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
        for index in range(32):
            adapter = ctypes.c_void_p()
            if enum(factory, index, ctypes.byref(adapter)) < 0:
                break
            try:
                desc = _AdapterDesc()
                get_desc = method(adapter, 10, ctypes.c_long, ctypes.POINTER(_AdapterDesc))
                if get_desc(adapter, ctypes.byref(desc)) >= 0 and not desc.flags & 2:
                    driver = ''
                    try:
                        version = ctypes.c_longlong()
                        check = method(adapter, 9, ctypes.c_long, ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_longlong))
                        device_iid = _Guid(*_IID_DXGI_DEVICE[:3], (ctypes.c_ubyte * 8)(*_IID_DXGI_DEVICE[3]))
                        if check(adapter, ctypes.byref(device_iid), ctypes.byref(version)) >= 0:
                            driver = format_driver(version.value)
                    except Exception:
                        log.debug('driver version unavailable for %s', desc.name, exc_info=True)
                    adapters.append((index, desc.name.rstrip('\0'), desc.dedicated, desc.vendor, driver))
            finally:
                method(adapter, 2, ctypes.c_ulong)(adapter)
    finally:
        method(factory, 2, ctypes.c_ulong)(factory)
    return adapters


def preferred_adapter():
    """Most dedicated VRAM wins; ``LUMEN_DML_DEVICE=<DXGI index>`` forces one adapter.

    On macOS the single Metal device is returned in the same shape (1.3.1).
    """
    if sys.platform == 'darwin':
        from . import metal
        return metal.device_info()
    adapters = dxgi_adapters()
    forced = os.environ.get('LUMEN_DML_DEVICE', '').strip()
    if forced:
        for adapter in adapters:
            if str(adapter[0]) == forced:
                return adapter
        log.warning('LUMEN_DML_DEVICE=%s does not match any DXGI adapter %s', forced, adapters)
    return max(adapters, key=lambda item: item[2]) if adapters else None


MODES = ('auto', 'cpu', 'dml', 'cuda', 'trt', 'metal')
NVIDIA = 0x10DE
COREML = 'CoreMLExecutionProvider'
COREML_UNITS = ('CPUAndGPU', 'ALL', 'CPUAndNeuralEngine', 'CPUOnly')


def adapter_driver(adapter):
    return adapter[4] if adapter is not None and len(adapter) > 4 else ''


def adapter_key(adapter):
    """Identifies a GPU + driver pair for the crash compatibility record."""
    if adapter is None:
        return 'no-gpu'
    return f'{adapter[1]}|{adapter_driver(adapter) or "unknown driver"}'


class PluginDevices(list):
    """ONNX Runtime EP devices of a plugin execution provider (Windows ML catalog)."""


def requested_mode():
    """``LUMEN_COMPUTE`` = auto (default) | cpu | dml | cuda | trt | metal, for troubleshooting and tests."""
    mode = os.environ.get('LUMEN_COMPUTE', 'auto').strip().lower()
    return mode if mode in MODES else 'auto'


def cupy_enabled():
    """CuPy has never been validated on hardware; since 1.3.0 it is opt-in only."""
    return os.environ.get('LUMEN_EXPERIMENTAL_CUPY') == '1' and requested_mode() in ('auto', 'cuda')


def coreml_options():
    """Core ML provider options: the Apple GPU through Metal by default (macOS, 1.3.1).

    ``LUMEN_COREML_UNITS=ALL`` also allows the Neural Engine.  Models compile in a
    temporary folder at session creation (0.1–2 s); no persistent cache is kept.
    """
    units = os.environ.get('LUMEN_COREML_UNITS', 'CPUAndGPU')
    return {'ModelFormat': 'MLProgram', 'MLComputeUnits': units if units in COREML_UNITS else 'CPUAndGPU',
            'RequireStaticInputShapes': '0', 'EnableOnSubgraphs': '0'}


E5_CACHE = 'com.apple.e5rt.e5bundlecache'
_E5_BEFORE = None           # source runs: runtime cache entries that existed when the worker started


def _process_alive(pid):
    if os.name == 'nt':
        return True             # os.kill(pid, 0) would terminate the process on Windows
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def coreml_housekeeping():
    """Delete Core ML files that are never read again (macOS, 1.3.1).

    ONNX Runtime compiles every Core ML session into a new temporary
    ``onnxruntime-<uuid>-<pid>-<n>`` folder and ``.mlmodelc`` bundle, which a
    killed or crashed worker leaves behind, and Core ML's runtime caches one
    compiled bundle per such path in ~/Library/Caches/<app>/com.apple.e5rt.e5bundlecache.
    The paths never repeat, so in testing both only grew, by hundreds of MB per
    session.  The AI worker calls this when it starts and stops.  The bundled app's
    runtime cache is cleared; a source run shares Python's cache folder with other
    programs, so only the entries added since the worker started are removed.
    """
    global _E5_BEFORE
    if sys.platform != 'darwin':
        return
    for path in Path(tempfile.gettempdir()).glob('onnxruntime-*'):
        fields = path.name.split('.')[0].split('-')
        if len(fields) != 8 or not fields[6].isdigit():
            continue
        pid = int(fields[6])
        if pid == os.getpid() or _process_alive(pid):
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    if getattr(sys, 'frozen', False):
        from . import host
        shutil.rmtree(Path.home() / 'Library' / 'Caches' / host.BUNDLE_ID / E5_CACHE, ignore_errors=True)
        return
    folder = Path.home() / 'Library' / 'Caches' / Path(sys.executable).name / E5_CACHE
    entries = set(folder.glob('*/*'))
    if _E5_BEFORE is None:
        _E5_BEFORE = entries
        return
    for path in entries - _E5_BEFORE:
        shutil.rmtree(path, ignore_errors=True)


def signature(ort, path):
    """Inputs, outputs and the named dynamic input dimensions ``{symbol: (input, axis)}`` of a model."""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    model = ort.InferenceSession(path, sess_options=options, providers=['CPUExecutionProvider'])
    describe = lambda values: [types.SimpleNamespace(name=v.name, type=v.type, shape=list(v.shape)) for v in values]
    inputs, outputs = describe(model.get_inputs()), describe(model.get_outputs())
    symbols = {}
    for item in inputs:
        for axis, dim in enumerate(item.shape):
            if dim is None:                  # an unnamed dynamic dimension cannot be fixed
                return inputs, outputs, None
            if isinstance(dim, str):
                symbols.setdefault(dim, (item.name, axis))
    return inputs, outputs, symbols


class ShapedSessions:
    """Core ML sessions compiled per input size for models with dynamic height / width (macOS, 1.3.1).

    With dynamic shapes Core ML cannot plan DRUNet and splits NAFNet into 80
    partitions (slower than the CPU); with the symbolic dimensions fixed to the
    tile size the restoration models run 8–10x faster than the CPU on an M1 Pro.
    Tiling produces only a few distinct sizes, so the most recent ``LIMIT``
    sessions are kept.  Results match the dynamic model: nothing is padded.
    """
    LIMIT = 4

    def __init__(self, ort, path, providers, described, profiled):
        self.ort, self.path, self.providers = ort, path, providers
        self.inputs, self.outputs, self.symbols = described
        self.profiled = profiled              # options of the first session, which is verified
        self.sessions = collections.OrderedDict()
        self.current = None

    def _session(self, feeds):
        key = tuple((name, tuple(np.shape(feeds[name]))) for name in sorted({n for n, _ in self.symbols.values()}))
        session = self.sessions.pop(key, None)
        if session is None:
            options, self.profiled = self.profiled or performance.session_options(), None
            for symbol, (name, axis) in self.symbols.items():
                options.add_free_dimension_override_by_name(symbol, int(np.shape(feeds[name])[axis]))
            session = self.ort.InferenceSession(self.path, sess_options=options, providers=self.providers)
            if COREML not in session.get_providers():
                raise RuntimeError(tr('Core ML 未接管该尺寸的模型'))
            log.info('%s: Core ML session for %s', Path(self.path).name, dict(key))
        self.sessions[key] = session
        while len(self.sessions) > self.LIMIT:
            self.sessions.popitem(last=False)
        self.current = session
        return session

    def run(self, output_names, inputs):
        return self._session(inputs).run(output_names, inputs)

    def end_profiling(self):
        return self.current.end_profiling()

    def get_providers(self):
        return [COREML, 'CPUExecutionProvider']

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs


def provider_plan(available, adapter, accelerated=True, models=False, excluded=(), status=None):
    """Ordered (providers, provider, device) attempts before the CPU fallback.

    Neural models (``models=True``) on an NVIDIA GeForce RTX GPU first try TensorRT
    for RTX from the Windows ML catalog (Windows 11 24H2+); DirectML itself is in
    maintenance mode upstream.  On macOS they run through Core ML on the Apple GPU
    (Metal).  Providers in ``excluded`` crashed before on this GPU and driver and
    are skipped (see ai_worker.CompatRecord).
    """
    mode = requested_mode()
    if not accelerated or mode == 'cpu':
        return []
    attempts = []
    if models and mode in ('auto', 'metal') and COREML in available and COREML not in excluded:
        device = adapter[1] if adapter is not None and len(adapter) > 1 else 'Apple GPU'
        attempts.append(([(COREML, coreml_options()), 'CPUExecutionProvider'], COREML, device))
    if (models and mode in ('auto', 'trt') and adapter is not None and len(adapter) > 3
            and adapter[3] == NVIDIA and 'NvTensorRtRtxExecutionProvider' not in excluded):
        from . import winml
        devices = winml.ensure(status=status)
        if devices:
            attempts.append((PluginDevices(devices), 'NvTensorRtRtxExecutionProvider', adapter[1]))
    if mode in ('auto', 'cuda') and 'CUDAExecutionProvider' in available:
        attempts.append((['CUDAExecutionProvider', 'CPUExecutionProvider'], 'CUDAExecutionProvider', 'CUDA GPU'))
    if mode in ('auto', 'dml') and adapter is not None and 'DmlExecutionProvider' in available:
        attempts.append(([('DmlExecutionProvider', {'device_id': adapter[0]}), 'CPUExecutionProvider'],
                         'DmlExecutionProvider', adapter[1]))
    return [a for a in attempts if a[1] not in excluded]


class ComputeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.provider = 'CPUExecutionProvider'
        self.device = ''
        self.warning = ''
        self.detail = tr('{threads} 线程 · NumPy / OpenCV', threads=performance.THREADS)
        self.warning = ''

    def report(self, provider, device='', detail='', warning=''):
        with self.lock:
            self.provider, self.device = provider, device
            self.detail = detail or (tr('{threads} 线程 · NumPy / OpenCV', threads=performance.THREADS) if provider == 'CPUExecutionProvider' else provider)
            self.warning = warning

    def snapshot(self):
        with self.lock:
            return self.provider, self.device, self.detail, self.warning


state = ComputeState()


def _gpu_kernels_in_profile(path, provider):
    try:
        events = json.loads(Path(path).read_text(encoding='utf-8'))
        return any(event.get('cat') == 'Node' and event.get('args', {}).get('provider') == provider
                   for event in events)
    finally:
        Path(path).unlink(missing_ok=True)


class Session:
    """Session facade: first inference verifies GPU kernels; failures retry on CPU."""
    def __init__(self, path, accelerated=True, excluded=(), on_attempt=None, status=None):
        import onnxruntime as ort
        self.path = path if isinstance(path, bytes) else str(path)
        self.label = '<in-memory graph>' if isinstance(path, bytes) else Path(self.path).name
        self.ort = ort
        self.provider = 'CPUExecutionProvider'
        self.device = ''
        self.warning = ''
        self._verified = False
        self._lock = threading.Lock()
        available = ort.get_available_providers()
        adapter = preferred_adapter() if accelerated and (os.name == 'nt' or sys.platform == 'darwin') else None
        models = not isinstance(path, bytes)
        attempts = provider_plan(available, adapter, accelerated, models, excluded, status)
        last_error = ''
        for providers, provider, device in attempts:
            try:
                if on_attempt:
                    on_attempt(provider)
                options = performance.session_options()
                plugin = isinstance(providers, PluginDevices)
                if plugin:
                    # Plugin EPs claim whole subgraphs; the provider list is the verification.
                    options.add_provider_for_devices(list(providers), {})
                    candidate = ort.InferenceSession(self.path, sess_options=options)
                    names = [p for p in candidate.get_providers() if p != 'CPUExecutionProvider']
                    if names:
                        self._session, self.provider, self.device = candidate, names[0], device
                        self._verified = True
                        break
                    continue
                options.enable_profiling = True
                options.profile_file_prefix = str(Path(tempfile.gettempdir()) / 'lumen-dml-profile')
                if provider == 'DmlExecutionProvider':
                    options.enable_mem_pattern = False
                    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                if provider == COREML:
                    described = signature(ort, self.path)
                    if described[2]:
                        # Compiled per tile size at first use; run() verifies and falls back.
                        self._session = ShapedSessions(ort, self.path, providers, described, options)
                        self.provider, self.device = provider, device
                        break
                candidate = ort.InferenceSession(self.path, sess_options=options, providers=providers)
                if provider in candidate.get_providers():
                    self._session, self.provider, self.device = candidate, provider, device
                    break
                Path(candidate.end_profiling()).unlink(missing_ok=True)
            except Exception as exc:
                log.warning('%s session failed for %s', provider, self.label, exc_info=True)
                last_error = str(exc)[:120]
                continue
        else:
            if on_attempt:
                on_attempt('CPUExecutionProvider')
            self._session = ort.InferenceSession(self.path,
                sess_options=performance.session_options(), providers=['CPUExecutionProvider'])
            if accelerated:
                self.warning = (tr('GPU 初始化失败：') + last_error) if last_error else tr('当前 ONNX 环境没有可用的 GPU 执行设备。')

    def _cpu_fallback(self, warning):
        log.warning('ONNX CPU fallback for %s: %s', getattr(self, 'label', self.path), warning)
        if self.provider != 'CPUExecutionProvider' and not self._verified:
            try:
                Path(self._session.end_profiling()).unlink(missing_ok=True)
            except Exception:
                pass
        self._session = self.ort.InferenceSession(self.path,
            sess_options=performance.session_options(), providers=['CPUExecutionProvider'])
        self.provider, self.device, self._verified = 'CPUExecutionProvider', '', True
        self.warning = warning
        state.report(self.provider, detail=tr('{threads} 线程 · ONNX CPU', threads=performance.THREADS), warning=self.warning)

    def run(self, output_names, inputs):
        with self._lock:
            if self.provider == 'CPUExecutionProvider':
                result = self._session.run(output_names, inputs)
                state.report(self.provider, detail=tr('{threads} 线程 · ONNX CPU', threads=performance.THREADS),warning=self.warning)
                return result
            if not self._verified:
                try:
                    result = self._session.run(output_names, inputs)
                    path = self._session.end_profiling()
                    if not _gpu_kernels_in_profile(path, self.provider):
                        self._cpu_fallback(tr('此模型没有在 GPU 上执行节点，已切换 CPU。'))
                        return self._session.run(output_names, inputs)
                    self._verified = True
                except Exception as exc:
                    self._cpu_fallback(tr('GPU 运算失败，已回退 CPU：{v}', v=str(exc)[:120]))
                    return self._session.run(output_names, inputs)
            else:
                try:
                    result = self._session.run(output_names, inputs)
                except Exception as exc:
                    self._cpu_fallback(tr('GPU 运算失败，已回退 CPU：{v}', v=str(exc)[:120]))
                    return self._session.run(output_names, inputs)
            if self.provider not in self._session.get_providers():
                self._cpu_fallback(tr('运行库已将模型切换到 CPU。'))
                return result
            state.report(self.provider, self.device, f'{self.provider} · {self.device}')
            return result

    def get_providers(self):
        return [self.provider]

    def get_inputs(self):
        return self._session.get_inputs()

    def get_outputs(self):
        return self._session.get_outputs()


def isolation_enabled():
    """AI models run in a worker process unless ``LUMEN_AI_ISOLATION=0`` (1.3.1)."""
    return os.environ.get('LUMEN_AI_ISOLATION', '1') != '0' and requested_mode() != 'cpu'


def session(path, accelerated=True):
    # Preserve the tiny ONNX stand-in used by the tiling contract test.
    import onnxruntime as ort
    if not hasattr(ort, 'SessionOptions'):
        return ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    if accelerated and not isinstance(path, bytes) and isolation_enabled() and not in_worker():
        # A GPU driver fault in a neural model must not take the editor down.
        from .ai_worker import RemoteSession
        return RemoteSession(path)
    return Session(path, accelerated)


_IN_WORKER = False


def in_worker():
    return _IN_WORKER
