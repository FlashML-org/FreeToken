"""Pure policy tests for ROCm BLAS selection and graph-gate precedence."""

from __future__ import annotations

import pytest

import freetoken.utils.graph_gate as gate


def test_graph_runner_detects_nested_resident_format_without_module_assumptions():
    from freetoken.engine.graph import _has_weight_format

    class Node:
        pass

    root, child = Node(), Node()
    root.child = child
    root.loop = root
    child.weight_format = "gguf"
    assert _has_weight_format(root, "gguf")
    assert not _has_weight_format(root, "fp8_block")


def test_graph_runner_runtime_telemetry_is_read_only_metadata():
    from freetoken.engine.graph import GraphRunner

    runner = object.__new__(GraphRunner)
    runner.graph_telemetry = {
        "expert_storage": "resident_gguf",
        "expert_fetches": 0,
        "expert_remaps": 0,
    }
    runner.resident_gguf = True
    runner.graph_map = {1: object(), 4: object()}
    runner.sampler_graph_map = {1: object()}
    assert runner.runtime_telemetry() == {
        "expert_storage": "resident_gguf",
        "expert_fetches": 0,
        "expert_remaps": 0,
        "resident_gguf": True,
        "graph_batches": [1, 4],
        "sampler_graph_batches": [1],
    }


@pytest.mark.parametrize("value", ["auto", "hipblas", "hipblaslt", "rocblas"])
def test_rocm_blas_request_accepts_supported_values(value):
    assert gate._rocm_blas_request(value) == value


def test_rocm_blas_request_rejects_unknown_value():
    with pytest.raises(ValueError, match="expected auto, hipblas, hipblaslt, rocblas"):
        gate._rocm_blas_request("cublas")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("hipblas", {"TORCH_BLAS_PREFER_HIPBLASLT": "0"}),
     ("rocblas", {"TORCH_BLAS_PREFER_HIPBLASLT": "0"}),
     ("hipblaslt", {"TORCH_BLAS_PREFER_HIPBLASLT": "1"}),
     ("auto", {})],
)
def test_blas_env_aliases(value, expected):
    assert gate._blas_env(value) == expected


def test_explicit_policy_overrides_graph_gate(monkeypatch):
    monkeypatch.setenv("FREETOKEN_ROCM_BLAS", "hipblaslt")
    monkeypatch.setattr(gate, "_is_rocm", lambda: True)
    monkeypatch.setattr(gate, "run_graph_gate", lambda: pytest.fail("graph gate must not run"))
    assert gate.resolve_rocm_blas_env() == {"TORCH_BLAS_PREFER_HIPBLASLT": "1"}


def test_auto_policy_uses_passing_graph_gate(monkeypatch):
    monkeypatch.setenv("FREETOKEN_ROCM_BLAS", "auto")
    monkeypatch.setattr(gate, "_is_rocm", lambda: True)
    result = {"ok": True, "env": {"TORCH_BLAS_PREFER_HIPBLASLT": "0"}}
    monkeypatch.setattr(gate, "run_graph_gate", lambda: result)
    assert gate.resolve_rocm_blas_env() == result["env"]


def test_explicit_policy_fails_when_effective_api_unavailable(monkeypatch):
    monkeypatch.setenv("FREETOKEN_ROCM_BLAS", "hipblas")
    monkeypatch.setattr(gate, "_is_rocm", lambda: True)
    monkeypatch.setattr(gate, "torch", None, raising=False)
    class Backends:
        cuda = object()
    class Torch:
        backends = Backends()
    monkeypatch.setitem(__import__("sys").modules, "torch", Torch())
    with pytest.raises(RuntimeError, match="preferred_blas_library unavailable"):
        gate.resolve_rocm_blas_env()


def test_report_normalizes_rocm_alias(monkeypatch):
    monkeypatch.setenv("FREETOKEN_ROCM_BLAS", "rocblas")
    monkeypatch.setattr(gate, "_is_rocm", lambda: True)
    monkeypatch.setattr(
        gate,
        "_effective_blas",
        lambda: ("hipblas", "reported"),
    )
    report = gate.rocm_blas_report(gate={})
    assert report["requested"] == "rocblas"
    assert report["effective"] == "hipblas"
    assert report["verification"] == "verified"


def test_graph_capture_env_invalid_policy_is_not_swallowed(monkeypatch):
    gate.graph_capture_env.cache_clear()
    monkeypatch.setenv("FREETOKEN_ROCM_BLAS", "bad")
    with pytest.raises(ValueError):
        gate.graph_capture_env()
    gate.graph_capture_env.cache_clear()
