"""ROCm cache/runtime version pairing and gfx-arch gating (torch-free).

Loads the relevant modules by file path with mocked torch so the logic is unit-testable
without a working torch/GPU install.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_UTILS_PATH = _ROOT / "python" / "freetoken" / "kernel" / "utils.py"
_ARCH_PATH = _ROOT / "python" / "freetoken" / "utils" / "arch.py"


@pytest.fixture(scope="module")
def ku():
    spec = importlib.util.spec_from_file_location("kernel_utils_mod", _UTILS_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def arch():
    spec = importlib.util.spec_from_file_location("arch_mod", _ARCH_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_cache_version_pairs_rocm(ku):
    # Same release + same g-sha + both rocm -> OK.
    assert ku._kernel_cache_version_ok(
        "0.1.1+rocm.g3f01615", "0.1.1+rocm.g3f01615"
    )
    # No g-stamp on either side -> release-only compare -> OK.
    assert ku._kernel_cache_version_ok("0.1.1+rocm", "0.1.1+rocm")


def test_cache_version_rejects_backend_tag_mismatch(ku):
    # CUDA runtime must never pair with a ROCm cache (SASS family differs).
    assert not ku._kernel_cache_version_ok(
        "0.1.1+rocm.g3f01615", "0.1.1+cu130.g3f01615"
    )
    assert not ku._kernel_cache_version_ok(
        "0.1.1+cu130.g3f01615", "0.1.1+rocm.g3f01615"
    )


def test_cache_version_rejects_different_build(ku):
    assert not ku._kernel_cache_version_ok(
        "0.1.1+rocm.g3f01615", "0.1.1+rocm.gdeadbee"
    )
    assert not ku._kernel_cache_version_ok("0.1.2+rocm", "0.1.1+rocm")


def _torch_importable() -> bool:
    """True only when torch actually *imports* (a findable-but-broken torch -- e.g. a
    CUDA build missing its runtime libs -- must count as absent so the torch-free path
    is exercised on such boxes)."""
    try:
        import torch  # noqa: PLC0415

        return True
    except Exception:
        return False


def test_gfx_arch_ge_degrades_without_torch(arch):
    if _torch_importable():
        pytest.skip("torch is importable; no-torch degradation tested on a torch-free box")
    assert arch.is_gfx_arch_ge(1100) is False


def test_gfx_arch_ge_parses_gfx_string(arch, monkeypatch):
    if _torch_importable():
        # Real torch present: is_gfx_arch_ge must reflect the actual device (gfx1100 on
        # an RX 7900 XTX). No fake torch injection to avoid cross-test state pollution.
        import torch

        assert arch.is_gfx_arch_ge(1100) is True  # RX 7000 target
        return
    arch._get_gfx_arch.cache_clear()
    # Fake a ROCm torch whose device name carries the gfx arch string.
    tv = types.ModuleType("torch.version")
    tv.hip = "6.2.4100000"
    tv.cuda = None
    t = types.ModuleType("torch")
    t.version = tv
    _props = types.SimpleNamespace(gcnArchName="gfx1100")
    t.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        get_device_name=lambda dev: "AMD Radeon RX 7900 XTX",
        get_device_properties=lambda dev: _props,
    )
    monkeypatch.setitem(sys.modules, "torch", t)
    monkeypatch.setitem(sys.modules, "torch.version", tv)
    assert arch.is_gfx_arch_ge(1100) is True
    assert arch.is_gfx_arch_ge(1103) is False
    # CUDA builds must return False.
    arch._get_gfx_arch.cache_clear()
    tv.hip = None
    tv.cuda = "13.0"
    assert arch.is_gfx_arch_ge(1100) is False
