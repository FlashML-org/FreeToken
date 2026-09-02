"""Inc 0 harness tests: replay manifest, gate lanes, and identity invalidation.

Covers the pure benchmark-harness logic (no GPU): the replay-manifest schema
contract, route hashing, warmup-aware step summaries, disjoint lane
classification in the promotion gate, and candidate-fallback invalidation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import bench_decode_replay as replay  # noqa: E402
import check_decode_gate as gate  # noqa: E402


# ---------------------------------------------------------------------------
# Replay manifest schema
# ---------------------------------------------------------------------------


def valid_manifest(**overrides) -> dict:
    manifest = {
        "schema": "freetoken-replay-manifest-v1",
        "lane": "teacher_forced_replay",
        "prompt_ids": [11, 12, 13],
        "continuation_ids": [70, 80, 90, 100],
        "warmup_tokens": 1,
        "measured_tokens": 4,
        "model_sha256": "0" * 64,
        "fixture_sha256": "2" * 64,
        "tokenizer_sha256": "3" * 64,
        "route_top_k": 8,
        "golden": {"ids_sha256": "a" * 64},
    }
    return {**manifest, **overrides}


def test_manifest_accepts_valid_contract():
    assert replay.validate_manifest(valid_manifest()) == []


def test_manifest_rejects_schema_lane_and_identity_drift():
    problems = replay.validate_manifest({**valid_manifest(), "schema": "other-schema"})
    assert any("schema" in problem for problem in problems)

    problems = replay.validate_manifest(
        {**valid_manifest(), "lane": "sampled_absolute"}
    )
    assert any("lane" in problem for problem in problems)

    problems = replay.validate_manifest({**valid_manifest(), "measured_tokens": 5})
    assert any("measured_tokens" in problem for problem in problems)

    problems = replay.validate_manifest({**valid_manifest(), "model_sha256": "short"})
    assert any("model_sha256" in problem for problem in problems)

    problems = replay.validate_manifest({**valid_manifest(), "route_top_k": None})
    assert any("route_top_k" in problem for problem in problems)

    problems = replay.validate_manifest({**valid_manifest(), "golden": {"ids_sha256": None}})
    assert any("golden" in problem for problem in problems)

    problems = replay.validate_manifest({**valid_manifest(), "continuation_ids": [1]})
    assert any("continuation_ids" in problem for problem in problems)

    problems = replay.validate_manifest({**valid_manifest(), "prompt_ids": [1, -1]})
    assert any("prompt_ids" in problem for problem in problems)

    problems = replay.validate_manifest({**valid_manifest(), "model_sha256": "z" * 64})
    assert any("model_sha256" in problem for problem in problems)

    problems = replay.validate_manifest(
        {**valid_manifest(), "prompt_text": "prompt", "prompt_text_sha256": "0" * 64}
    )
    assert any("prompt_text_sha256" in problem for problem in problems)


def test_load_manifest_rejects_invalid_file(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({**valid_manifest(), "schema": "wrong"}))
    with pytest.raises(ValueError, match="rejected"):
        replay.load_manifest(str(path))

    good = valid_manifest()
    path.write_text(json.dumps(good))
    assert replay.load_manifest(str(path)) == good


def test_route_hash_is_stable_and_disjoint():
    assert replay.route_hash([1, 2, 3]) == replay.route_hash([1, 2, 3])
    assert replay.route_hash([1]) != replay.route_hash([2])
    assert replay.route_hash(list(range(8))) != replay.route_hash(list(range(7, -1, -1)))


def test_route_digest_preserves_token_and_layer_order():
    one = replay._route_digest(
        [
            {"layer": 2, "start": 0, "hashes": ["b", "c"]},
            {"layer": 1, "start": 0, "hashes": ["a", "d"]},
        ]
    )
    two = replay._route_digest(
        [
            {"layer": 1, "start": 0, "hashes": ["a", "d"]},
            {"layer": 2, "start": 0, "hashes": ["b", "c"]},
        ]
    )
    assert one == two
    assert one["0"] != one["1"]


def test_prompt_text_uses_manifest_and_checks_hash():
    prompt = "hello"
    manifest = {"prompt_text": prompt, "prompt_text_sha256": replay._sha256_bytes(prompt.encode())}
    assert replay.prompt_text_of(manifest) == prompt
    with pytest.raises(ValueError, match="prompt_text"):
        replay.prompt_text_of({"prompt_text": prompt, "prompt_text_sha256": "0" * 64})


def test_tokens_list_accepts_llama_tokenize_response():
    assert replay.tokens_list({"tokens": [1, 2, 3]}) == [1, 2, 3]
    assert replay.tokens_list({"tokens": "bad"}) is None


def test_summarize_steps_excludes_warmup_keeps_raw():
    steps = [1.0, 2.0, 3.0, 100.0]
    stats = replay.summarize_steps(steps, warmup_steps=1)
    assert stats["steps"] == 3
    assert stats["warmup_steps"] == 1
    assert stats["raw_steps_ms"] == steps
    assert stats["ms_per_token_median"] == 3.0
    assert stats["ms_per_token_min"] == 2.0
    empty = replay.summarize_steps([], 0)
    assert empty["steps"] == 0 and empty["ms_per_token_median"] is None


# ---------------------------------------------------------------------------
# Gate lane classification
# ---------------------------------------------------------------------------


def _sampled_row(**overrides) -> dict:
    row = {
        "schema": "freetoken-base-decode-v2",
        "status": "accepted",
        "lane": "sampled_absolute",
        "sampling": {"temperature": 1.0},
        "decode_tok_s": 60.0,
        "metadata": {"lane": "sampled_absolute", "mtp": "off"},
    }
    return {**row, **overrides}


def _greedy_row() -> dict:
    return {
        "schema": "freetoken-base-decode-v2",
        "status": "accepted",
        "lane": "greedy_correctness",
        "sampling": {"temperature": 0.0},
        "metadata": {"lane": "greedy_correctness", "mtp": "off"},
    }


def _replay_row() -> dict:
    return {
        "schema": "freetoken-replay-v1",
        "status": "accepted",
        "lane": "teacher_forced_replay",
    }


def test_lane_classification_legacy_and_explicit():
    assert gate._row_lane(_sampled_row()) == gate.LANE_SAMPLED
    assert gate._row_lane(_greedy_row()) == gate.LANE_GREEDY
    assert gate._row_lane(_replay_row()) == gate.LANE_REPLAY
    legacy = {
        "schema": "freetoken-base-decode-v2",
        "sampling": {"temperature": 1.0},
        "metadata": {"mtp": "off"},
    }
    assert gate._row_lane(legacy) == gate.LANE_SAMPLED
    legacy["sampling"] = {"temperature": 0.0}
    assert gate._row_lane(legacy) == gate.LANE_GREEDY
    assert gate._row_lane({"schema": "unrelated-v9"}) is None


def test_lane_mixing_is_rejected():
    assert gate._lane_reasons([_sampled_row(), _sampled_row()]) == []
    mixed = gate._lane_reasons([_sampled_row(), _replay_row()])
    assert any("mix disjoint lanes" in reason for reason in mixed)
    unknown = gate._lane_reasons([{"schema": "unrelated-v9", "status": "accepted"}])
    assert any("unknown/mismatched lane" in reason for reason in unknown)


def test_fallback_evidence_rejects_gate_row():
    row = _sampled_row()
    row["metadata"] = {
        **row["metadata"],
        "kernel_observed": {"fallback_markers": 2, "source": "server_log"},
    }
    result = gate.evaluate_gate(
        [row], reference_rows=[], probe=None, min_runs=1, directional_threshold=75.0
    )
    assert any("candidate-eligible fallback" in reason for reason in result["reasons"])


def test_zero_fallback_keeps_row_eligible():
    row = _sampled_row()
    row["metadata"] = {
        **row["metadata"],
        "kernel_observed": {"fallback_markers": 0, "source": "server_log"},
    }
    result = gate.evaluate_gate(
        [row], reference_rows=[], probe=None, min_runs=1, directional_threshold=75.0
    )
    assert not any("fallback" in reason for reason in result["reasons"])


def _complete_replay(rate: float, *, repeat: int, route: dict | None = None) -> dict:
    return {
        "schema": "freetoken-replay-v1",
        "lane": "teacher_forced_replay",
        "status": "accepted",
        "runtime": "freetoken",
        "repeat": repeat,
        "forced": True,
        "ids_match": True,
        "route_hash_status": "matched",
        "route_digest": route or {"0": "route"},
        "model_sha256": "a" * 64,
        "fixture_sha256": "b" * 64,
        "tokenizer_sha256": "c" * 64,
        "manifest_ids_sha256": "d" * 64,
        "context": 9216,
        "batch": 512,
        "ubatch": 512,
        "kv_type": "q8_0",
        "mtp": "off",
        "speculative": False,
        "decode_batch_size": 1,
        "decode_tok_s": rate,
    }


def test_gate_b_is_reported_separately_for_matched_replay():
    candidate = [_complete_replay(82.0, repeat=0)]
    reference = [_complete_replay(81.0, repeat=0)]
    result = gate.evaluate_gate(candidate, reference, min_runs=1)
    assert result["gate_b"]["gate"] is True
    assert result["gates"] == {"gate_a": False, "gate_b": True}
    assert result["gate"] is False  # Gate A has no sampled lane.


def test_gate_b_rejects_route_mismatch():
    candidate = [_complete_replay(82.0, repeat=0)]
    reference = [_complete_replay(81.0, repeat=0, route={"0": "other"})]
    result = gate.evaluate_gate(candidate, reference, min_runs=1)
    assert result["gate_b"]["gate"] is False
    assert any("route hash mismatch" in reason for reason in result["gate_b"]["reasons"])
