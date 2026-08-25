"""ROCm-aware NVFP4 backend selection (torch-free).

Loads ``nvfp4_backends.py`` with a fake ``freetoken.utils.arch`` so the AMD branch of
``select_nvfp4_backend`` (reject NVIDIA-only marlin/flashinfer, force triton) is
unit-testable without torch/vLLM/flashinfer.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_BACKEND_PATH = (
    Path(__file__).resolve().parents[2]
    / "python" / "freetoken" / "moe" / "nvfp4_backends.py"
)


def _torch_importable() -> bool:
    """True only when torch actually *imports* (a findable-but-broken torch -- e.g. a
    CUDA build missing its runtime libs -- must count as absent so the torch-free path
    is exercised on such boxes)."""
    try:
        import torch  # noqa: PLC0415

        return True
    except Exception:
        return False


def _load_nvfp4(rocm: bool):
    arch = types.ModuleType("freetoken.utils.arch")
    arch.is_rocm = lambda: rocm
    freetoken = types.ModuleType("freetoken")
    freetoken.__path__ = []
    utils = types.ModuleType("freetoken.utils")
    utils.__path__ = []
    utils.arch = arch
    utils.init_logger = lambda name: types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    sys.modules["freetoken"] = freetoken
    sys.modules["freetoken.utils"] = utils
    sys.modules["freetoken.utils.arch"] = arch

    # nvfp4_backends.py does `import torch` at module scope. On a torch-free box stub it
    # so the module imports; on a box with real torch use the real one (never mutate
    # sys.modules["torch"], which would corrupt the session's torch/triton state).
    if not _torch_importable():
        torch_stub = types.ModuleType("torch")
        torch_stub.no_grad = lambda: (lambda f: f)
        torch_stub.Tensor = object
        sys.modules["torch"] = torch_stub

    spec = importlib.util.spec_from_file_location("nvfp4_backends_mod", _BACKEND_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(autouse=True)
def _clean():
    yield
    for name in ("freetoken", "freetoken.utils", "freetoken.utils.arch"):
        sys.modules.pop(name, None)


def test_rocm_auto_resolves_triton():
    m = _load_nvfp4(rocm=True)
    dev = types.SimpleNamespace(type="cuda")
    assert m.select_nvfp4_backend(dev, 768, "auto") == "triton"
    assert m.select_nvfp4_backend(dev, 768, "triton") == "triton"


def test_rocm_rejects_nvidia_backends():
    m = _load_nvfp4(rocm=True)
    dev = types.SimpleNamespace(type="cuda")
    with pytest.raises(RuntimeError, match="NVIDIA-only"):
        m.select_nvfp4_backend(dev, 768, "marlin")
    with pytest.raises(RuntimeError, match="NVIDIA-only"):
        m.select_nvfp4_backend(dev, 768, "flashinfer")


def test_cuda_path_not_short_circuited_by_rocm_guard():
    # On CUDA the ROCm early-return must not be taken. We only exercise the guard
    # boundary (is_rocm()==False) without calling torch.cuda (unavailable here); the
    # full CUDA auto logic runs on the target box.
    m = _load_nvfp4(rocm=False)
    assert m.select_nvfp4_backend(
        types.SimpleNamespace(type="cpu"), 768, "auto"
    ) == "triton"
