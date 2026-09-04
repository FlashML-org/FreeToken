"""Graph worker environment resolution tests."""

from __future__ import annotations

import freetoken.utils.graph_gate as gate


def test_cpu_graph_env_is_noop(monkeypatch):
    gate.graph_capture_env.cache_clear()
    monkeypatch.setenv("FREETOKEN_ROCM_BLAS", "hipblaslt")
    monkeypatch.setattr(gate, "_is_rocm", lambda: False)
    monkeypatch.setattr(gate, "run_graph_gate", lambda: (_ for _ in ()).throw(AssertionError()))
    assert gate.graph_capture_env() == {}
    gate.graph_capture_env.cache_clear()


def test_auto_graph_env_is_single_gate_result(monkeypatch):
    gate.graph_capture_env.cache_clear()
    monkeypatch.setenv("FREETOKEN_ROCM_BLAS", "auto")
    monkeypatch.setattr(gate, "_is_rocm", lambda: True)
    expected = {"ok": True, "env": {"TORCH_BLAS_PREFER_HIPBLASLT": "0"}}
    monkeypatch.setattr(gate, "run_graph_gate", lambda: expected)
    assert gate.graph_capture_env() == expected["env"]
    gate.graph_capture_env.cache_clear()
