"""Pure provenance and rejection gates for ROCm decode benchmark adapters."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from bench_decode_moe import acceptance_status, execution_metadata, model_fingerprint  # noqa: E402
from bench_decode_ollama import ollama_blob_identity  # noqa: E402
from bench_llama_cpp_hip import _parse_cli_timing  # noqa: E402


def _args(**overrides):
    values = {
        "decode": 512,
        "problem": 0,
        "attention_backend": "triton",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_file_model_identity_is_full_sha256(tmp_path):
    model = tmp_path / "model.gguf"
    payload = b"stable model bytes\n"
    model.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    identity = model_fingerprint(str(model))

    assert identity["sha256"] == expected
    assert identity["identity"] == "full-file-sha256"
    assert identity["size_bytes"] == len(payload)


def test_directory_identity_ignores_mtime_but_tracks_content(tmp_path):
    model = tmp_path / "checkpoint"
    model.mkdir()
    shard = model / "weights.bin"
    shard.write_bytes(b"one")
    first = model_fingerprint(str(model))
    shard.touch()
    assert model_fingerprint(str(model))["sha256"] == first["sha256"]
    shard.write_bytes(b"two")
    assert model_fingerprint(str(model))["sha256"] != first["sha256"]


@pytest.mark.parametrize(
    ("completion", "events", "text", "graph_state", "accepted"),
    [
        (512, 4, "answer", "replay", True),
        (511, 4, "answer", "replay", False),
        (512, 1, "answer", "replay", False),
        (512, 4, "", "replay", False),
        (512, 4, "answer", "unknown", False),
    ],
)
def test_acceptance_status_rejects_non_comparable_rows(
    completion, events, text, graph_state, accepted
):
    result = acceptance_status(
        args=_args(),
        result={"stamps": [0.0] * events, "text": text},
        graph={"state": graph_state},
        usage={"completion_tokens": completion},
        model={"sha256": "model-sha"},
    )

    assert result["accepted"] is accepted
    assert result["status"] == ("accepted" if accepted else "rejected")


def test_ollama_manifest_digest_is_not_gguf_identity(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"same")

    unproven = ollama_blob_identity("sha256:manifest")
    verified = ollama_blob_identity(
        "sha256:manifest",
        ollama_gguf=str(model),
        reference_gguf=str(model),
    )

    assert unproven["same_blob"] is False
    assert unproven["status"] == "unproven"
    assert verified["same_blob"] is True
    assert verified["status"] == "verified"
    assert verified["manifest_digest_kind"] == "ollama-model-manifest"


def test_ollama_gguf_mismatch_is_explicit(tmp_path):
    ollama_model = tmp_path / "ollama.gguf"
    reference_model = tmp_path / "reference.gguf"
    ollama_model.write_bytes(b"ollama")
    reference_model.write_bytes(b"freetoken")

    identity = ollama_blob_identity(
        "sha256:manifest",
        ollama_gguf=str(ollama_model),
        reference_gguf=str(reference_model),
    )

    assert identity["same_blob"] is False
    assert identity["status"] == "mismatch"


def test_llama_cli_timing_parser_requires_eval_count():
    output = "eval time = 6400.0 ms / 512 runs (12.5 ms per token, 80.0 tokens per second)"
    assert _parse_cli_timing(output) == (512, 80.0)
    assert _parse_cli_timing("no timings") is None


def test_execution_metadata_reads_observed_cache_geometry():
    observed = {
        "effective_moe_backend": "fused",
        "expert_storage": "resident_gguf",
        "resident_gguf": True,
        "expert_fetches": 0,
        "expert_remaps": 0,
    }
    value = execution_metadata(
        args=_args(),
        backend="fused",
        graph={"state": "replay", "gate": "pass"},
        cache_status={"geometry": {"execution": observed}},
    )
    assert value["effective_moe_backend"] == "fused"
    assert value["expert_storage"] == "resident_gguf"
    assert value["graph_state"] == "replay"
