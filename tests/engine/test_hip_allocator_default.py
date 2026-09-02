"""expandable_segments must stay OFF on HIP (ROCm ~1016 physical-handle cap). CPU-only."""

from __future__ import annotations

import pytest
import torch

from freetoken.engine import engine as engine_mod
from freetoken.runtime import gpu


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    gpu.vendor.cache_clear()
    yield
    gpu.vendor.cache_clear()


def _record_allocator_calls(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda.memory, "_set_allocator_settings", lambda s: calls.append(s))
    return calls


def test_hip_default_is_off(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GPU_VENDOR", "amd")
    gpu.vendor.cache_clear()
    assert engine_mod.expandable_segments_default() is False
    calls = _record_allocator_calls(monkeypatch)
    engine_mod._ensure_expandable_segments()
    assert calls == []


def test_nvidia_default_is_on(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GPU_VENDOR", "nvidia")
    gpu.vendor.cache_clear()
    assert engine_mod.expandable_segments_default() is True
    calls = _record_allocator_calls(monkeypatch)
    engine_mod._ensure_expandable_segments()
    assert calls == ["expandable_segments:True"]


def test_user_alloc_conf_is_respected_everywhere(monkeypatch):
    """An explicit PYTORCH_ALLOC_CONF wins on both vendors -- the HIP opt-in path."""
    calls = _record_allocator_calls(monkeypatch)
    for vendor in ("amd", "nvidia"):
        monkeypatch.setenv("FREETOKEN_GPU_VENDOR", vendor)
        gpu.vendor.cache_clear()
        monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        engine_mod._ensure_expandable_segments()
    assert calls == []


def test_handle_cap_documents_the_measured_ceiling():
    # 20 MiB large-pool handles: the cap is why a 32 GiB R9700 topped out near 20 GiB.
    assert engine_mod.HIP_EXPANDABLE_SEGMENTS_HANDLE_CAP * (20 << 20) < 21 << 30
