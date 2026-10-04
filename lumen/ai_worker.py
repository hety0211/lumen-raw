"""Out-of-process inference for the neural models (1.3.1).

A GPU driver or execution-provider fault inside ONNX Runtime is a native crash
that no Python code can catch.  AI super-resolution, denoising and automatic
masks therefore run their ONNX sessions in one long-lived worker process.  When
that process dies, the provider it was using is recorded for this GPU + driver
(``%LOCALAPPDATA%\\LUMEN RAW\\gpu-compat.json``; macOS: ``~/Library/Application
Support/LUMEN RAW``), the worker restarts, and the same tile is retried on the next
provider (TensorRT for RTX -> DirectML -> CPU; macOS: Core ML -> CPU).
The editor itself keeps running and later sessions skip the crashing provider.
"""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import signal
import sys
import threading
import time
import types
from pathlib import Path

from . import compute
from .i18n import tr

log = logging.getLogger(__name__)

OPEN_TIMEOUT = 900      # first use may download TensorRT for RTX and build kernels
RUN_TIMEOUT = 600
MAX_RETRIES = 4


class WorkerCrashed(RuntimeError):
    pass


# ----------------------------------------------------------------------------- worker side

def _serve(conn, log_level):
    compute._IN_WORKER = True
    from . import logs, winml
    winml.preload_system_runtime()
    logs.configure(level=log_level, filename='lumen-worker.log')
    if os.name != 'nt':
        # Daemonic workers are stopped with SIGTERM: unwind so the sessions delete their files.
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    compute.coreml_housekeeping()
    sessions = {}
    try:
        _requests(conn, sessions)
    finally:
        sessions.clear()
        compute.coreml_housekeeping()


def _requests(conn, sessions):
    crash_model = os.environ.get('LUMEN_TEST_WORKER_CRASH', '')

    def send(*message):
        conn.send(message)

    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            return
        kind = message[0]
        try:
            if kind == 'open':
                _, key, path, excluded = message
                if key not in sessions:
                    if crash_model and Path(path).name == crash_model and 'SimulatedGPU' not in excluded:
                        send('trying', 'SimulatedGPU')     # test hook: a provider that faults natively
                        os._exit(9)
                    sessions[key] = compute.Session(path, True, excluded=set(excluded),
                                                    on_attempt=lambda p: send('trying', p),
                                                    status=lambda text: send('status', text))
                s = sessions[key]
                describe = lambda values: [(v.name, v.type, list(v.shape)) for v in values]
                send('ok', dict(inputs=describe(s.get_inputs()), outputs=describe(s.get_outputs()),
                                provider=s.provider, device=s.device, warning=s.warning))
            elif kind == 'run':
                _, key, names, inputs = message
                s = sessions[key]
                send('trying', s.provider)
                result = s.run(names, inputs)
                send('ok', dict(result=result, provider=s.provider, device=s.device, warning=s.warning))
            elif kind == 'ping':
                send('ok', os.getpid())
            elif kind == 'stop':
                send('ok', None)
                return
        except Exception as exc:
            log.exception('worker request %s failed', kind)
            send('error', f'{type(exc).__name__}: {exc}')


# ----------------------------------------------------------------------------- crash record

#: Warnings that report a worker crash (the AI dialog shows them with the result).
CRASH_NOTICES = set()

def _record_path():
    from .host import data_folder
    return data_folder() / 'gpu-compat.json'


class CompatRecord:
    """Providers that crashed natively, per GPU + driver version and model file."""

    def __init__(self, path=None):
        self.path = Path(path) if path else _record_path()
        self._lock = threading.Lock()

    def _load(self):
        try:
            return json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {}

    def excluded(self, adapter, model):
        entry = self._load().get(compute.adapter_key(adapter), {})
        return sorted(p for p, info in entry.items() if info.get('all') or model in info.get('models', []))

    def record(self, adapter, model, provider, detail=''):
        if provider in (None, '', 'CPUExecutionProvider'):
            return
        with self._lock:
            data = self._load()
            info = data.setdefault(compute.adapter_key(adapter), {}).setdefault(provider, {'models': []})
            if model not in info['models']:
                info['models'].append(model)
            # Two different models crashing means the provider itself is unusable here.
            info['all'] = len(info['models']) >= 2
            info['last'] = dict(model=model, detail=str(detail)[:300], time=time.strftime('%Y-%m-%d %H:%M:%S'))
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix('.tmp')
                tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
                tmp.replace(self.path)
            except OSError:
                log.warning('cannot write %s', self.path, exc_info=True)
        log.warning('%s crashed natively on %s for %s; excluded from now on (%s)',
                    provider, compute.adapter_key(adapter), model, detail)


# ----------------------------------------------------------------------------- parent side

class Worker:
    def __init__(self):
        self.lock = threading.RLock()
        self.process = None
        self.conn = None
        self.generation = 0
        self.trying = None
        self.status = None

    def alive(self):
        return self.process is not None and self.process.is_alive()

    def start(self):
        context = multiprocessing.get_context('spawn')
        parent, child = context.Pipe()
        level = logging.getLevelName(logging.getLogger().getEffectiveLevel())
        self.process = context.Process(target=_serve, args=(child, level), name='LumenAIWorker', daemon=True)
        self.process.start()
        child.close()
        self.conn = parent
        self.generation += 1
        log.info('AI worker started (pid %s, generation %d)', self.process.pid, self.generation)

    def stop(self, timeout=5):
        with self.lock:
            if self.alive():
                try:
                    self.conn.send(('stop',))
                    self.process.join(timeout)
                except (OSError, EOFError):
                    pass
                if self.process.is_alive():
                    self.process.kill()
            self.process = self.conn = None

    def kill(self):
        if self.process is not None and self.process.is_alive():
            self.process.kill()
            self.process.join(5)
        self.process = self.conn = None

    def request(self, message, timeout):
        if not self.alive():
            self.start()
        self.trying = None
        try:
            self.conn.send(message)
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.kill()
                    raise WorkerCrashed(f'no response within {timeout} s')
                if self.conn.poll(min(remaining, 1.)):
                    reply = self.conn.recv()
                    if reply[0] == 'trying':
                        self.trying = reply[1]
                    elif reply[0] == 'status':
                        if self.status:
                            self.status(reply[1])
                    else:
                        return reply
                elif not self.process.is_alive():
                    raise WorkerCrashed(f'exit code {self.process.exitcode}')
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError) as exc:
            code = self.process.exitcode if self.process is not None else None
            self.kill()
            raise WorkerCrashed(f'{type(exc).__name__}, exit code {code}') from exc


_worker = Worker()
_record = None


def worker():
    return _worker


def record():
    global _record
    if _record is None:
        _record = CompatRecord()
    return _record


def shutdown():
    _worker.stop()


def set_status_listener(callback):
    """Receives messages such as the first-use TensorRT for RTX download."""
    _worker.status = callback


class RemoteSession:
    """Drop-in for compute.Session whose ONNX session lives in the worker process."""

    def __init__(self, path):
        self.path = str(path)
        self.model = Path(self.path).name
        self.provider = 'CPUExecutionProvider'
        self.device = ''
        self.warning = ''
        self._notice = ''
        self._meta = None
        self._key = None
        self._generation = -1

    def _open(self):
        adapter = compute.preferred_adapter()
        excluded = record().excluded(adapter, self.model)
        self._key = (self.path, tuple(excluded))
        status, reply = _worker.request(('open', self._key, self.path, tuple(excluded)), OPEN_TIMEOUT)
        if status != 'ok':
            raise RuntimeError(reply)
        self._meta, self._generation = reply, _worker.generation
        self.provider, self.device, self.warning = reply['provider'], reply['device'], reply['warning']
        log.info('%s runs on %s %s (excluded: %s)', self.model, self.provider, self.device, excluded or '-')

    def _call(self, fn):
        with _worker.lock:
            for _ in range(MAX_RETRIES):
                try:
                    if self._meta is None or self._generation != _worker.generation or not _worker.alive():
                        self._open()
                    return fn()
                except WorkerCrashed as crash:
                    culprit = _worker.trying or self.provider
                    adapter = compute.preferred_adapter()
                    if culprit in (None, 'CPUExecutionProvider'):
                        raise RuntimeError(tr('AI 推理进程在 CPU 上异常退出：{crash}', crash=crash)) from crash
                    record().record(adapter, self.model, culprit, crash)
                    self._notice = self.warning = tr('{culprit} 在此显卡与驱动上运行 {model} 时崩溃，已自动改用其他设备重试。', culprit=culprit, model=self.model)
                    CRASH_NOTICES.add(self._notice)
                    compute.state.report('CPUExecutionProvider', warning=self.warning)
                    self._meta = None
            raise RuntimeError(tr('AI 推理进程多次异常退出，已停止。'))

    def run(self, output_names, inputs):
        def go():
            status, reply = _worker.request(('run', self._key, output_names, inputs), RUN_TIMEOUT)
            if status != 'ok':
                raise RuntimeError(reply)
            self.provider, self.device = reply['provider'], reply['device']
            self.warning = self._notice or reply['warning']
            if self.provider == 'CPUExecutionProvider':
                compute.state.report(self.provider, detail=tr('{threads} 线程 · ONNX CPU（独立进程）', threads=compute.performance.THREADS),
                                     warning=self.warning)
            else:
                compute.state.report(self.provider, self.device, f'{self.provider} · {self.device}', self.warning)
            return reply['result']
        return self._call(go)

    def _values(self, kind):
        self._call(lambda: None)
        return [types.SimpleNamespace(name=n, type=t, shape=s) for n, t, s in self._meta[kind]]

    def get_inputs(self):
        return self._values('inputs')

    def get_outputs(self):
        return self._values('outputs')

    def get_providers(self):
        return [self.provider]
