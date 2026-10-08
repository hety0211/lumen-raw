"""Intel GPU on Linux through ONNX Runtime's OpenVINO provider (1.5.2).

The provider is inert unless ``onnxruntime-openvino`` is installed — the plain
wheel does not offer ``OpenVINOExecutionProvider``, so the plan is unchanged on
Windows and macOS.
"""
import onnxruntime as ort
import pytest

from lumen import compute

LINUX_GPU = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']


def plan(**kwargs):
    kwargs.setdefault('models', True)
    return compute.provider_plan(LINUX_GPU, None, True, **kwargs)


def test_openvino_attempt_is_the_only_gpu_path_on_intel_linux(monkeypatch):
    monkeypatch.delenv('LUMEN_COMPUTE', raising=False)
    monkeypatch.delenv('LUMEN_OPENVINO_FP16', raising=False)
    attempts = plan()
    assert [item[1] for item in attempts] == ['OpenVINOExecutionProvider']
    providers, provider, device = attempts[0]
    assert providers[0] == (compute.OPENVINO, {'device_type': 'GPU', 'precision': 'FP32'})
    assert providers[1] == 'CPUExecutionProvider'
    assert device == 'OpenVINO GPU'


def test_pointwise_graphs_always_ask_for_fp32(monkeypatch):
    """The parity tests compare the graphs against the NumPy reference."""
    monkeypatch.delenv('LUMEN_COMPUTE', raising=False)
    monkeypatch.setenv('LUMEN_OPENVINO_FP16', '1')
    providers, _, _ = plan(models=False)[0]
    assert providers[0][1]['precision'] == 'FP32'
    providers, _, _ = plan(models=True)[0]
    assert providers[0][1]['precision'] == 'FP16'      # only neural models opt in


def test_no_intel_gpu_is_left_alone(monkeypatch):
    monkeypatch.delenv('LUMEN_COMPUTE', raising=False)
    assert compute.provider_plan(['CPUExecutionProvider'], None, True, True) == []
    # CPU mode disables acceleration entirely.
    monkeypatch.setenv('LUMEN_COMPUTE', 'cpu')
    assert plan() == []
    # A crashed provider is skipped, exactly like DirectML / Core ML.
    assert plan(excluded={compute.OPENVINO}) == []
    # Requesting CPU keeps the previous behaviour even with the wheel installed.
    monkeypatch.setenv('LUMEN_COMPUTE', 'cpu')
    assert compute.provider_plan(LINUX_GPU, None, True, True) == []


def test_windows_and_macos_priority_is_unchanged(monkeypatch):
    """The Linux attempt sits last: the other platforms keep their provider."""
    monkeypatch.delenv('LUMEN_COMPUTE', raising=False)
    amd = (0, 'Radeon RX 9070 XT', 16 << 30, 0x1002, '32.0.1')
    attempts = compute.provider_plan(LINUX_GPU + ['DmlExecutionProvider'], amd, True, True)
    assert [item[1] for item in attempts] == ['DmlExecutionProvider', 'OpenVINOExecutionProvider']
    apple = (0, 'Apple M1 Pro', 0, 0x106B, 'Metal')
    assert [item[1] for item in compute.provider_plan(['CoreMLExecutionProvider'], apple, True, True)] \
        == ['CoreMLExecutionProvider']


@pytest.mark.skipif(compute.OPENVINO not in ort.get_available_providers(),
                    reason='onnxruntime-openvino is not installed here')
def test_openvino_session_accepts_the_options_we_pass_it():
    """With the wheel present the provider must accept device_type / precision."""
    from lumen import gpu_graphs
    session = ort.InferenceSession(gpu_graphs.model('tonal'),
                                   providers=[(compute.OPENVINO,
                                               {'device_type': 'GPU', 'precision': 'FP32'}),
                                              'CPUExecutionProvider'])
    assert session.get_providers()[0] == compute.OPENVINO
    assert 'CPUExecutionProvider' in session.get_providers()
