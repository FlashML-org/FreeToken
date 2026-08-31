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
    # Snapshot-restore, NOT pop: when the real freetoken package was already imported
    # earlier in the session, popping the 4 fake names leaves the real submodules
    # (freetoken.attention, freetoken.kernel.pinned, ...) cached with a dangling parent,
    # so every later `getattr(freetoken, "attention")` / `import freetoken.kernel.pinned`
    # in unrelated tests raises AttributeError/ImportError. Restoring the exact pre-test
    # snapshot (and only creating entries we removed) keeps the session clean.
    saved = {
        key: mod
        for key, mod in sys.modules.items()
        if key == "freetoken" or key.startswith("freetoken.") or key == "torch"
    }
    yield
    for key in [k for k in sys.modules if k == "freetoken" or k.startswith("freetoken.")]:
        if key in saved and saved[key] is not None:
            sys.modules[key] = saved[key]
        else:
            sys.modules.pop(key, None)


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
