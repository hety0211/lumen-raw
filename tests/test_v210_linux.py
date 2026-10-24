"""2.1.0 Linux port: CUDA adapter parsing, provider planning and loader symlinks."""
import ctypes

from lumen import compute, cuda_env, timelapse


def test_cuda_adapters_parse_nvidia_smi(monkeypatch):
    class Result:
        stdout = '0, Example GPU A, 8192, 999.99\n1, Example GPU B, 16384, 999.99\n'

    monkeypatch.setattr(compute.os, 'name', 'posix')
    monkeypatch.setattr(compute.sys, 'platform', 'linux')
    monkeypatch.setattr(compute.shutil, 'which', lambda name: __file__)
    monkeypatch.setattr(compute.subprocess, 'run', lambda *args, **kwargs: Result())
    adapters = compute.cuda_adapters()
    assert adapters == [
        (0, 'Example GPU A', 8192 * 2**20, compute.NVIDIA, '999.99'),
        (1, 'Example GPU B', 16384 * 2**20, compute.NVIDIA, '999.99'),
    ]


def test_linux_cuda_runtime_ready_checks_cudnn(monkeypatch):
    monkeypatch.setattr(compute.sys, 'platform', 'linux')
    monkeypatch.setattr(ctypes, 'CDLL', lambda name: object())
    assert compute.cuda_runtime_ready()
    monkeypatch.setattr(ctypes, 'CDLL', lambda name: (_ for _ in ()).throw(OSError(name)))
    assert not compute.cuda_runtime_ready()


def test_provider_plan_uses_cuda_with_the_adapter_name(monkeypatch):
    monkeypatch.setattr(compute, 'requested_mode', lambda: 'cuda')
    monkeypatch.setattr(compute, 'cuda_runtime_ready', lambda: True)
    monkeypatch.delenv('LUMEN_CUDA_DEVICE', raising=False)
    monkeypatch.delenv('LUMEN_DML_DEVICE', raising=False)
    adapter = (0, 'Example GPU A', 8 * 2**30, compute.NVIDIA, '999.99')
    attempts = compute.provider_plan(['CUDAExecutionProvider', 'CPUExecutionProvider'], adapter)
    assert attempts
    providers, kind, label = attempts[0]
    assert providers == ['CUDAExecutionProvider', 'CPUExecutionProvider']
    assert kind == 'CUDAExecutionProvider'
    assert label == adapter[1]


def test_cuda_loader_links_unversioned_names(tmp_path, monkeypatch):
    site = tmp_path / 'site-packages'
    library = site / 'nvidia' / 'cu13' / 'lib'
    library.mkdir(parents=True)
    (library / 'libcudart.so.13').write_bytes(b'')
    (library / 'libcudnn.so.9').write_bytes(b'')
    monkeypatch.setattr(cuda_env.sys, 'platform', 'linux')
    monkeypatch.setattr(cuda_env.sysconfig, 'get_paths', lambda: {'purelib': str(site)})
    monkeypatch.setattr(cuda_env.sys, 'path', [str(site)])
    monkeypatch.setattr(cuda_env, 'cache_folder', lambda: tmp_path / 'cache')
    path = cuda_env.prepare()
    assert str(library) in path
    for name, target in (('libcudart.so', 'libcudart.so.13'), ('libcudnn.so', 'libcudnn.so.9')):
        link = tmp_path / 'cache' / name
        assert link.is_symlink()
        assert link.resolve() == (library / target).resolve()


def test_platform_hardware_skips_windows_only_encoders(monkeypatch):
    names = ('h264_amf', 'h264_nvenc', 'h264_qsv', 'h264_mf', 'h264_videotoolbox')
    monkeypatch.setattr(timelapse.os, 'name', 'posix')
    monkeypatch.setattr(timelapse.sys, 'platform', 'linux')
    assert timelapse.platform_hardware(names) == ['h264_nvenc', 'h264_qsv']
    monkeypatch.setattr(timelapse.sys, 'platform', 'darwin')
    assert timelapse.platform_hardware(names) == ['h264_videotoolbox']
