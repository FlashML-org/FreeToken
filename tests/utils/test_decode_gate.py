"""Pure final decode gate tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from check_decode_gate import evaluate_gate  # noqa: E402


MODEL_SHA = "a" * 64
PROMPT_SHA = "b" * 64
FIXTURE_SHA = "c" * 64
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
PROBE = {
    "schema": "qwen-moe-base-probe-v2",
    "model": {"sha256": MODEL_SHA},
    "prompt_sha256": PROMPT_SHA,
    "mtp": "off",
    "speculative": False,
    "eager": {"finite_logits": True, "decode_rows": 8},
    "graph": {"finite_logits": True, "decode_rows": 8},
    "comparison": {"token_ids_equal": True},
}


def _execution() -> dict:
    return {
        "effective_moe_backend": "fused",
        "expert_storage": "resident_gguf",
        "resident_gguf": True,
        "expert_fetches": 0,
        "expert_remaps": 0,
        "attention_backend": "triton",
        "graph_state": "replay",
        "graph_gate": "pass",
        "decode_batch_size": 1,
        "mtp": "off",
        "speculative": False,
    }


def _free(value: float, *, status: str = "accepted", repeat: int | None = None) -> dict:
    return {
        "status": status,
        "metadata": {"model_sha256": MODEL_SHA, "prompt_sha256": PROMPT_SHA, "sampling": SAMPLING, "mtp": "off"},
        "sampling": SAMPLING,
        "context": 9216,
        "batch": 512,
        "ubatch": 512,
        "kv_type": "q8_0",
        "fixture_sha256": FIXTURE_SHA,
        "completion_tokens": 512,
        "decode_requested": 512,
        "decode_tok_s": value,
        "repeat": repeat,
        "execution": _execution(),
        "acceptance": {"accepted": status == "accepted"},
    }


def _ollama(value: float, *, repeat: int | None = None) -> dict:
    return {
        "schema": "ollama-base-decode-v1",
        "status": "accepted",
        "reference_identity": {
            "status": "verified",
            "same_blob": True,
            "reference_gguf": {"sha256": MODEL_SHA},
        },
        "prompt_sha256": PROMPT_SHA,
        "context": 9216,
        "batch": 512,
        "ubatch": 512,
        "kv_type": "q8_0",
        "fixture_sha256": FIXTURE_SHA,
        "mtp": "off",
        "speculative": False,
        "acceptance": {"accepted": True, "checks": {"mtp_off": True}},
        "options": SAMPLING,
        "client_arrival_tok_s": value,
        "repeat": repeat,
    }


def test_gate_passes_only_with_verified_reference_and_confidence():
    free = [_free(82.0, repeat=i) for i in range(10)]
    reference = [_ollama(81.0, repeat=i) for i in range(10)]
    result = evaluate_gate(free, reference, PROBE, min_runs=10)
    assert result["gate"] is True
    assert result["reference_identity"]["kind"] == "ollama-client-arrival"
    assert result["runs"]["p02_5_bootstrap_tok_s"] == 82.0


def test_directional_threshold_never_promotes_unproven_reference():
    result = evaluate_gate([_free(90.0) for _ in range(10)], [], min_runs=10)
    assert result["gate"] is False
    assert result["threshold_source"] == "directional-ollama-unproven"
    assert any("matched reference unavailable" in reason for reason in result["reasons"])


@pytest.mark.parametrize("value", [69.9, 75.0])
def test_gate_rejects_low_run_or_confidence(value):
    result = evaluate_gate(
        [_free(value, repeat=i) for i in range(10)],
        [_ollama(60.0, repeat=i) for i in range(10)],
        min_runs=10,
    )
    assert result["gate"] is False
    assert any("below 70" in reason or "p02.5" in reason for reason in result["reasons"])


def test_rejected_rows_cannot_be_hidden_from_gate():
    result = evaluate_gate(
        [_free(90.0, repeat=0), _free(90.0, status="rejected", repeat=1)],
        [_ollama(1.0, repeat=0)],
        min_runs=1,
    )
    assert result["gate"] is False
    assert result["rejected"] == 1


def test_empty_input_is_machine_readable_rejection():
    result = evaluate_gate([], [], min_runs=1)
    assert result["gate"] is False
    assert result["runs"]["median_tok_s"] is None


def test_malformed_accepted_row_is_rejected_without_gate_crash():
    row = _free(90.0)
    row["model_fingerprint"] = None
    row["metadata"] = []
    row["acceptance"] = None
    result = evaluate_gate([row], [], min_runs=1)
    assert result["gate"] is False
    assert "FreeToken rows do not carry one full model SHA-256" in result["reasons"]
    assert "accepted rows contain exact-completion or acceptance mismatch" in result["reasons"]


def test_non_sha_model_identity_cannot_pass_as_full_identity():
    row = _free(90.0)
    row["metadata"]["model_sha256"] = "head-tail-sha256:abc"
    result = evaluate_gate([row for _ in range(10)], [], min_runs=10)
    assert result["gate"] is False
    assert "FreeToken rows do not carry one full model SHA-256" in result["reasons"]


def test_missing_execution_evidence_cannot_promote():
    rows = [_free(90.0, repeat=i) for i in range(10)]
    for row in rows:
        del row["execution"]
    result = evaluate_gate(rows, [], min_runs=10)
    assert result["gate"] is False
    assert any("missing execution-mode evidence" in reason for reason in result["reasons"])


def test_missing_finite_logit_probe_cannot_promote_verified_reference():
    result = evaluate_gate(
        [_free(90.0, repeat=i) for i in range(10)],
        [_ollama(81.0, repeat=i) for i in range(10)],
        min_runs=10,
    )
    assert result["gate"] is False
    assert "finite-logit/parity probe unavailable" in result["reasons"]


def test_probe_identity_and_parity_are_required():
    probe = dict(PROBE)
    probe["prompt_sha256"] = "c" * 64
    probe["comparison"] = {"token_ids_equal": False}
    result = evaluate_gate(
        [_free(90.0, repeat=i) for i in range(10)],
        [_ollama(81.0, repeat=i) for i in range(10)],
        probe,
        min_runs=10,
    )
    assert result["gate"] is False
    assert "finite-logit probe prompt SHA-256 does not match FreeToken rows" in result["reasons"]
    assert "eager/graph greedy token parity failed or is unavailable" in result["reasons"]


@pytest.mark.parametrize("field", ["context", "batch", "ubatch", "kv_type", "fixture_sha256"])
def test_gate_rejects_comparator_mismatch(field):
    rows = [_free(90.0, repeat=i) for i in range(10)]
    wrong = {
        "context": 8192,
        "batch": 1,
        "ubatch": 1,
        "kv_type": "bf16",
        "fixture_sha256": "d" * 64,
    }
    rows[0][field] = wrong[field]
    result = evaluate_gate(rows, [], PROBE, min_runs=10)
    assert result["gate"] is False
    needle = "fixture" if field == "fixture_sha256" else field
    assert any(needle in reason for reason in result["reasons"])


def test_gate_rejects_missing_fixture_identity():
    rows = [_free(90.0, repeat=i) for i in range(10)]
    del rows[0]["fixture_sha256"]
    result = evaluate_gate(rows, [], PROBE, min_runs=10)
    assert result["gate"] is False
    assert any("fixture SHA-256" in reason for reason in result["reasons"])
