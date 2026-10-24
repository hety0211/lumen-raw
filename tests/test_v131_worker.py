"""1.3.1: AI models run in a worker process; native crashes fall back to the next provider."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from lumen import ai_worker, compute, winml

MODEL = Path(__file__).resolve().parents[1] / 'assets' / 'models' / 'realesr-general-x4v3.onnx'


# The worker may run the model on a GPU (DirectML / TensorRT / CUDA); allow GPU rounding.
# CUDA's convolution kernels differ from the CPU by about 0.3 % on this model, which is
# float rounding rather than a functional difference.
GPU_TOLERANCE = 3e-3 if sys.platform.startswith('linux') else 1e-4


def direct(image):
    session = ort.InferenceSession(str(MODEL), providers=['CPUExecutionProvider'])
    return session.run(None, {'image': image})[0]


def test_remote_session_matches_in_process_inference():
    image = np.random.default_rng(1).random((1, 3, 24, 31), dtype=np.float32)
    remote = compute.session(MODEL, True)
    try:
        assert isinstance(remote, ai_worker.RemoteSession)
        assert [i.name for i in remote.get_inputs()] == ['image']
        np.testing.assert_allclose(remote.run(None, {'image': image})[0], direct(image), atol=GPU_TOLERANCE)
        assert remote.get_providers()[0] in ort.get_available_providers()
    finally:
        ai_worker.shutdown()


def test_native_crash_is_recorded_and_retried_on_the_next_provider(monkeypatch, tmp_path):
    ai_worker.shutdown()
    monkeypatch.setenv('LUMEN_TEST_WORKER_CRASH', MODEL.name)
    monkeypatch.setattr(ai_worker, '_record', ai_worker.CompatRecord(tmp_path / 'gpu-compat.json'))
    image = np.random.default_rng(2).random((1, 3, 20, 20), dtype=np.float32)
    try:
        remote = ai_worker.RemoteSession(MODEL)
        result = remote.run(None, {'image': image})[0]
        np.testing.assert_allclose(result, direct(image), atol=GPU_TOLERANCE)
        record = json.loads((tmp_path / 'gpu-compat.json').read_text(encoding='utf-8'))
        entry = next(iter(record.values()))
        assert entry['SimulatedGPU']['models'] == [MODEL.name] and not entry['SimulatedGPU']['all']
        assert '崩溃' in remote.warning and 'SimulatedGPU' in compute.state.snapshot()[3]
        # A new session skips the crashing provider straight away (no second crash).
        generation = ai_worker.worker().generation
        again = ai_worker.RemoteSession(MODEL)
        again.run(None, {'image': image})
        assert ai_worker.worker().generation == generation
    finally:
        ai_worker.shutdown()


def test_compat_record_blocks_a_provider_after_two_models(tmp_path):
    record = ai_worker.CompatRecord(tmp_path / 'compat.json')
    gpu = (0, 'GeForce RTX 5060', 8 << 30, 0x10DE, '32.0.16.1088')
    other_driver = gpu[:4] + ('32.0.16.2000',)
    record.record(gpu, 'drunet-color.onnx', 'DmlExecutionProvider', 'exit code 3221225477')
    assert record.excluded(gpu, 'drunet-color.onnx') == ['DmlExecutionProvider']
    assert record.excluded(gpu, 'nafnet-sidd.onnx') == []
    record.record(gpu, 'nafnet-sidd.onnx', 'DmlExecutionProvider')
    assert record.excluded(gpu, 'u2netp.onnx') == ['DmlExecutionProvider']
    assert record.excluded(other_driver, 'drunet-color.onnx') == []      # a new driver gets a fresh try
    record.record(gpu, 'u2netp.onnx', 'CPUExecutionProvider')
    assert 'CPUExecutionProvider' not in record.excluded(gpu, 'u2netp.onnx')


def test_session_routing(monkeypatch):
    assert isinstance(compute.session(MODEL, False), compute.Session)
    assert isinstance(compute.session(MODEL, True), ai_worker.RemoteSession)
    monkeypatch.setenv('LUMEN_AI_ISOLATION', '0')
    assert isinstance(compute.session(MODEL, True), compute.Session)
    monkeypatch.delenv('LUMEN_AI_ISOLATION')
    from lumen import gpu_graphs
    assert isinstance(compute.session(gpu_graphs.model('tonal'), True), compute.Session)


@pytest.mark.skipif(os.name != 'nt', reason='TensorRT for RTX comes from the Windows ML catalog')
def test_nvidia_models_try_tensorrt_for_rtx_first(monkeypatch):
    monkeypatch.setenv('LUMEN_COMPUTE', 'auto')
    monkeypatch.setattr(winml, 'ensure', lambda **_: ['rtx-device'])
    available = ['DmlExecutionProvider', 'CPUExecutionProvider']
    nvidia = (0, 'NVIDIA GeForce RTX 5060', 8 << 30, 0x10DE, '32.0.16.1088')
    plan = compute.provider_plan(available, nvidia, True, models=True)
    assert [p[1] for p in plan] == ['NvTensorRtRtxExecutionProvider', 'DmlExecutionProvider']
    assert isinstance(plan[0][0], compute.PluginDevices) and list(plan[0][0]) == ['rtx-device']
    # Pixel graphs, AMD GPUs and recorded crashes keep the DirectML / CPU route.
    assert [p[1] for p in compute.provider_plan(available, nvidia, True, models=False)] == ['DmlExecutionProvider']
    amd = (0, 'AMD Radeon RX 9070 XT', 16 << 30, 0x1002, '32.0.31041.1004')
    assert [p[1] for p in compute.provider_plan(available, amd, True, models=True)] == ['DmlExecutionProvider']
    plan = compute.provider_plan(available, nvidia, True, models=True,
                                 excluded={'NvTensorRtRtxExecutionProvider', 'DmlExecutionProvider'})
    assert plan == []
    monkeypatch.setenv('LUMEN_COMPUTE', 'dml')
    assert [p[1] for p in compute.provider_plan(available, nvidia, True, models=True)] == ['DmlExecutionProvider']


def test_driver_version_format_and_windows_ml_guards(monkeypatch):
    assert compute.format_driver((32 << 48) | (0 << 32) | (16 << 16) | 1088) == '32.0.16.1088'
    assert compute.adapter_key((0, 'GPU', 1, 0x10DE, '32.0.16.1088')) == 'GPU|32.0.16.1088'
    assert compute.adapter_key((0, 'GPU', 1, 0x10DE)) == 'GPU|unknown driver'
    monkeypatch.setattr(winml, 'windows_build', lambda: 22631)
    monkeypatch.setattr(winml, '_devices', {})
    assert not winml.supported() and winml.ensure() == [] and '24H2' in winml.last_error()
