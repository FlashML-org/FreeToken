from __future__ import annotations

import pytest
import importlib.util
from pathlib import Path


def _graph_gate():
    path = Path(__file__).parents[2] / "python/freetoken/utils/graph_gate.py"
    spec = importlib.util.spec_from_file_location("freetoken_test_graph_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rocm_blas_request_normalizes_and_rejects_unknown(monkeypatch):
    graph_gate = _graph_gate()

    assert graph_gate._rocm_blas_request(" HIPBLASLT ") == "hipblaslt"
    with pytest.raises(ValueError, match="expected auto, hipblas, hipblaslt, rocblas"):
        graph_gate._rocm_blas_request("cublas")


def test_explicit_rocm_blas_env_is_not_selected_for_cuda(monkeypatch):
    graph_gate = _graph_gate()

    monkeypatch.setattr(graph_gate, "_is_rocm", lambda: False)
    monkeypatch.setattr(
        graph_gate, "run_graph_gate", lambda: pytest.fail("CUDA ran ROCm graph gate")
    )
    monkeypatch.setenv("FREETOKEN_ROCM_BLAS", "hipblaslt")

    assert graph_gate.graph_capture_env() == {}
    graph_gate.graph_capture_env.cache_clear()


def test_graph_status_is_unknown_without_device(monkeypatch):
    graph_gate = _graph_gate()

    monkeypatch.setattr(
        graph_gate,
        "run_graph_gate",
        lambda: {"ok": False, "device_kind": "cpu", "detail": "no device"},
    )
    graph_gate.graph_capture_status.cache_clear()

    assert graph_gate.graph_capture_status() == "unknown"
    graph_gate.graph_capture_status.cache_clear()


def test_graph_status_fails_closed_when_disabled(monkeypatch):
    graph_gate = _graph_gate()

    monkeypatch.setenv("FREETOKEN_ROCM_GRAPH_CAPTURE", "0")
    monkeypatch.setattr(
        graph_gate,
        "run_graph_gate",
        lambda: pytest.fail("disabled graph capture ran the probe"),
    )
    graph_gate.graph_capture_status.cache_clear()

    assert graph_gate.graph_capture_status() == "fail"
    graph_gate.graph_capture_status.cache_clear()


def test_cached_gate_requires_runtime_identity(monkeypatch, tmp_path):
    graph_gate = _graph_gate()

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(graph_gate, "_device_kind", lambda: "rocm")
    monkeypatch.setattr(graph_gate, "_device_name", lambda: "AMD Radeon")
    monkeypatch.setattr(graph_gate, "_cache_identity", lambda: {"torch": "new"})
    path = graph_gate._cache_path()
    Path(path).write_text(
        '{"device_kind":"rocm","device":"AMD Radeon",'
        '"cache_identity":{"torch":"old"},"variant":"default"}'
    )

    assert graph_gate._load_cached() is None
