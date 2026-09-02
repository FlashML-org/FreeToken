"""HIP forces e4m3 emulation unless FREETOKEN_HIP_E4M3_NATIVE=1. No GPU required.

``HIP_EMU`` is resolved once at import (the kernel-side constexpr reads it), so a
test that changes the vendor must patch the constant too -- exactly what a real
process must not do after import, which the drift guard enforces."""

from __future__ import annotations

import ast
import inspect

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


def _as_hip(monkeypatch, *, native_opt_in: bool):
    monkeypatch.setenv("FREETOKEN_GPU_VENDOR", "amd")
    if native_opt_in:
        monkeypatch.setenv("FREETOKEN_HIP_E4M3_NATIVE", "1")
    else:
        monkeypatch.delenv("FREETOKEN_HIP_E4M3_NATIVE", raising=False)
    gpu.vendor.cache_clear()
    monkeypatch.setattr(e4m3_compat, "HIP_EMU", e4m3_compat._hip_forces_e4m3_emu())


def test_hip_forces_e4m3_emulation(monkeypatch):
    _as_hip(monkeypatch, native_opt_in=False)
    assert e4m3_compat.HIP_EMU is True
    assert e4m3_compat.e4m3_native() is False


def test_hip_e4m3_native_opt_in(monkeypatch):
    _as_hip(monkeypatch, native_opt_in=True)
    assert e4m3_compat._hip_forces_e4m3_emu() is False
    assert e4m3_compat.HIP_EMU is False


def test_nvidia_does_not_force_e4m3_emu(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GPU_VENDOR", "nvidia")
    monkeypatch.delenv("FREETOKEN_HIP_E4M3_NATIVE", raising=False)
    gpu.vendor.cache_clear()
    assert e4m3_compat._hip_forces_e4m3_emu() is False


def test_hip_decision_drift_after_import_is_loud(monkeypatch):
    """Vendor/env flipped after import without re-resolving HIP_EMU: the host and
    the compiled kernels would disagree on fp8-vs-uint8, so refuse."""
    _as_hip(monkeypatch, native_opt_in=False)
    monkeypatch.setenv("FREETOKEN_HIP_E4M3_NATIVE", "1")  # now disagrees with HIP_EMU=True
    with pytest.raises(RuntimeError, match="changed after import"):
        e4m3_compat.e4m3_native()


def test_constexpr_body_calls_no_plain_python():
    """ROCm Triton's compile-time evaluator rejects calls into ordinary Python
    inside a constexpr_function ("Unsupported function referenced"). The kernel
    gate may only reference module constants and ``target_info``."""
    cx = e4m3_compat.e4m3_native_cx
    fn = getattr(cx, "fn", None) or getattr(cx, "__wrapped__", None) or cx
    tree = ast.parse(inspect.getsource(fn))
    calls = [n.func for n in ast.walk(tree) if isinstance(n, ast.Call)]
    for call in calls:
        assert isinstance(call, ast.Attribute) and isinstance(call.value, ast.Name) and call.value.id == "target_info", (
            f"e4m3_native_cx calls {ast.dump(call)}: only target_info.* is allowed at compile time"
        )
