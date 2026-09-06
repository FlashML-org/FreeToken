import math
import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from bench_decode_replay import build_replay_record  # noqa: E402
from bench_rocm_matrix import build_manifest  # noqa: E402
from check_decode_gate import evaluate_gate  # noqa: E402


def _rows(tmp_path, values, *, lane="teacher_forced_replay", fallback=0):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"stable")
    runtime = {
        "commit": "a" * 40, "dirty_diff": "b" * 64, "dirty": False,
        "gpu": "gfx1100", "driver": "rocm", "torch": "torch", "rocm": "7",
        "hip": "7", "triton": "3", "jit_sha": "jit", "env_digest": "c" * 64,
    }
    rows = [build_manifest(
        model=model, prompt="fixed", token_count=16, backend="rocm", quant="Q4_K",
        graph_mode="eager", route="legacy", cache_hits=1, fetches=0,
        fallbacks=fallback, finite_logits=True, completion_count=16, lane=lane,
        timings_tok_s=[value], runtime=runtime,
    ) for value in values]
    if lane == "teacher_forced_replay":
        rows = [build_replay_record(
            row, prompt_ids=[1, 2, 3], continuation_ids=[4, 5, 6], route_digest="d" * 64,
            oracle_id="torch-reference-v1",
        ) for row in rows]
    return rows


def test_gate_requires_reproducible_gain(tmp_path):
    result = evaluate_gate(_rows(tmp_path, [110, 112, 111]), _rows(tmp_path, [100, 101, 99]))
    assert result["gate"] is True
    assert result["gain"] >= 0.05


def test_gate_rejects_mixed_lanes(tmp_path):
    result = evaluate_gate(
        _rows(tmp_path, [110, 112, 111]),
        _rows(tmp_path, [100, 101, 99], lane="sampled_absolute"),
    )
    assert result["gate"] is False
    assert any("identity" in reason for reason in result["reasons"])


def test_gate_rejects_fallback_evidence(tmp_path):
    with pytest.raises(ValueError, match="fallbacks"):
        _rows(tmp_path, [110], fallback=1)


def test_gate_rejects_nonfinite_timing(tmp_path):
    candidate = _rows(tmp_path, [110])
    candidate[0]["timing"]["median_tok_s"] = math.nan

    result = evaluate_gate(candidate, _rows(tmp_path, [100]))

    assert result["gate"] is False
    assert any("median_tok_s" in reason for reason in result["reasons"])
    assert result["gain"] is None


@pytest.mark.parametrize("lane", ["sampled_absolute", "greedy_correctness"])
def test_gate_rejects_non_replay_speed_lanes(tmp_path, lane):
    result = evaluate_gate(
        _rows(tmp_path, [110, 112, 111], lane=lane),
        _rows(tmp_path, [100, 101, 99], lane=lane),
    )
    assert result["gate"] is False
    assert any("teacher_forced_replay" in reason for reason in result["reasons"])


def test_gate_rejects_route_digest_mismatch(tmp_path):
    candidate = _rows(tmp_path, [110, 112, 111])
    baseline = _rows(tmp_path, [100, 101, 99])
    baseline[0]["replay"]["route_digest"] = "f" * 64

    result = evaluate_gate(candidate, baseline)

    assert result["gate"] is False
    assert any("identity" in reason for reason in result["reasons"])
