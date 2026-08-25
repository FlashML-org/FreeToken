"""Device-kind (CUDA vs ROCm vs CPU) detection and graceful-degradation gating.

These run without a working torch install: we load ``utils/arch.py`` by file path and
inject a fake ``torch``/``torch.version`` into ``sys.modules`` so the build-detection
(``torch.version.hip`` vs ``torch.version.cuda``) is unit-testable anywhere.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_ARCH_PATH = (
    Path(__file__).resolve().parents[2] / "python" / "freetoken" / "utils" / "arch.py"
)


@pytest.fixture(scope="module")
def arch():
    spec = importlib.util.spec_from_file_location("arch_mod", _ARCH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_torch(arch, monkeypatch, hip, cuda):
    tv = types.ModuleType("torch.version")
    tv.hip = hip
    tv.cuda = cuda
    t = types.ModuleType("torch")
    t.version = tv
    t.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", t)
    monkeypatch.setitem(sys.modules, "torch.version", tv)


def test_device_kind_rocm(arch, monkeypatch):
    _set_torch(arch, monkeypatch, hip="6.2.4100000", cuda=None)
    assert arch.device_kind() == "rocm"
    assert arch.is_rocm() is True
    assert arch.is_cuda() is False


def test_device_kind_cuda(arch, monkeypatch):
    _set_torch(arch, monkeypatch, hip=None, cuda="13.0")
    assert arch.device_kind() == "cuda"
    assert arch.is_rocm() is False
    assert arch.is_cuda() is True


def test_device_kind_cpu(arch, monkeypatch):
    _set_torch(arch, monkeypatch, hip=None, cuda=None)
    assert arch.device_kind() == "cpu"


def _torch_importable() -> bool:
    """True only when torch actually *imports* (a findable-but-broken torch -- e.g. a
    CUDA build missing its runtime libs -- must count as absent so the torch-free path
    is exercised on such boxes)."""
    try:
        import torch  # noqa: PLC0415

        return True
    except Exception:
        return False


def test_arch_gates_degrade_without_torch(arch, monkeypatch):
    # Only meaningful where torch is genuinely absent (the CUDA-locked dev box). On a
    # box with a working torch, Python re-imports the real torch after the delitem, so
    # the assertion target changes -- skip there.
    if _torch_importable():
        pytest.skip("torch is importable; no-torch degradation tested on a torch-free box")
    # With torch absent, every is_sm* gate is False and device_kind() is "cpu".
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "torch.version", raising=False)
    assert arch.is_sm90_supported() is False
    assert arch.is_sm100_supported() is False
    assert arch.is_sm90_family() is False
    assert arch.is_sm100_family() is False
    assert arch.device_kind() == "cpu"
