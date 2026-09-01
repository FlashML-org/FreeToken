"""HIP forces e4m3 emulation unless FREETOKEN_HIP_E4M3_NATIVE=1. No GPU required."""

from __future__ import annotations

import pytest

from freetoken.kernel.triton import e4m3_compat
from freetoken.runtime import gpu


@pytest.fixture(autouse=True)
def _reset():
    e4m3_compat._native = None
    gpu.vendor.cache_clear()
    yield
    e4m3_compat._native = None
    gpu.vendor.cache_clear()


def test_hip_forces_e4m3_emulation(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GPU_VENDOR", "amd")
    monkeypatch.delenv("FREETOKEN_HIP_E4M3_NATIVE", raising=False)
    gpu.vendor.cache_clear()
    assert e4m3_compat._hip_forces_e4m3_emu() is True
    assert e4m3_compat.e4m3_native() is False


def test_hip_e4m3_native_opt_in(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GPU_VENDOR", "amd")
    monkeypatch.setenv("FREETOKEN_HIP_E4M3_NATIVE", "1")
    gpu.vendor.cache_clear()
    assert e4m3_compat._hip_forces_e4m3_emu() is False


def test_nvidia_does_not_force_e4m3_emu(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GPU_VENDOR", "nvidia")
    monkeypatch.delenv("FREETOKEN_HIP_E4M3_NATIVE", raising=False)
    gpu.vendor.cache_clear()
    assert e4m3_compat._hip_forces_e4m3_emu() is False
