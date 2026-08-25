"""ROCm-aware optional-package probing in ``kernel/backend.py``.

Loads ``backend.py`` with a fake ``freetoken.utils.arch`` injected so the ROCm gating
(treat NVIDIA-only native packages as unavailable) is unit-testable without a working
torch or those packages installed.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_BACKEND_PATH = (
    Path(__file__).resolve().parents[2] / "python" / "freetoken" / "kernel" / "backend.py"
)


def _load_backend(rocm: bool):
    # Fake freetoken.utils.arch (backend.py imports is_rocm from it at module load).
    arch_mod = types.ModuleType("freetoken.utils.arch")
    arch_mod.is_rocm = lambda: rocm
    arch_mod.device_kind = lambda: "rocm" if rocm else "cpu"
    freetoken = types.ModuleType("freetoken")
    freetoken.__path__ = []
    utils = types.ModuleType("freetoken.utils")
    utils.__path__ = []
    utils.arch = arch_mod
    sys.modules["freetoken"] = freetoken
    sys.modules["freetoken.utils"] = utils
    sys.modules["freetoken.utils.arch"] = arch_mod

    spec = importlib.util.spec_from_file_location("freetoken.kernel.backend", _BACKEND_PATH)
    kernel = types.ModuleType("freetoken.kernel")
    kernel.__path__ = [str(_BACKEND_PATH.parent)]
    sys.modules["freetoken.kernel"] = kernel
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _clean_sys_modules():
    yield
    for name in ("freetoken", "freetoken.utils", "freetoken.utils.arch", "freetoken.kernel"):
        sys.modules.pop(name, None)


def test_rocm_native_packages_unavailable():
    backend = _load_backend(rocm=True)
    assert backend.is_flashinfer_installed() is False
    assert backend.is_sgl_kernel_installed() is False
    assert backend.is_triton_kernels_installed() is False
    assert backend.is_native_cuda_available() is False


def test_cuda_native_packages_follow_importability(monkeypatch):
    backend = _load_backend(rocm=False)
    # Without the packages installed, probes are False; is_native_cuda_available() needs a
    # real CUDA-capable torch (returns False here since torch.cuda is unavailable).
    assert backend.is_flashinfer_installed() is False
    assert backend.is_sgl_kernel_installed() is False
