import sys
from pathlib import Path

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from bench_decode_replay import (  # noqa: E402
    build_replay_record,
    ids_sha256,
    validate_replay_record,
)
from bench_rocm_matrix import build_manifest  # noqa: E402


def test_replay_record_pins_ids_and_matched_routes(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"stable")
    runtime = {
        "commit": "a" * 40, "dirty_diff": "b" * 64, "dirty": False,
        "gpu": "gfx1100", "driver": "rocm", "torch": "torch", "rocm": "7",
        "hip": "7", "triton": "3", "jit_sha": "jit", "env_digest": "c" * 64,
    }
    base = build_manifest(
        model=model, prompt="fixed", token_count=3, backend="rocm", quant="Q4_K",
        graph_mode="eager", route="legacy", cache_hits=0, fetches=0, fallbacks=0,
        finite_logits=True, completion_count=3, lane="teacher_forced_replay",
        timings_tok_s=[100], runtime=runtime,
    )
    record = build_replay_record(
        base, prompt_ids=[1, 2], continuation_ids=[3, 4], route_digest="e" * 64,
        oracle_id="torch-reference-v1",
    )
    assert validate_replay_record(record) == []
    assert record["replay"]["prompt_ids_sha256"] == ids_sha256([1, 2])


def test_replay_record_rejects_unmatched_routes(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"stable")
    runtime = {
        "commit": "a" * 40, "dirty_diff": "b" * 64, "dirty": False,
        "gpu": "gfx1100", "driver": "rocm", "torch": "torch", "rocm": "7",
        "hip": "7", "triton": "3", "jit_sha": "jit", "env_digest": "c" * 64,
    }
    base = build_manifest(
        model=model, prompt="fixed", token_count=2, backend="rocm", quant="Q4_K",
        graph_mode="eager", route="legacy", cache_hits=0, fetches=0, fallbacks=0,
        finite_logits=True, completion_count=2, lane="teacher_forced_replay",
        timings_tok_s=[100], runtime=runtime,
    )
    record = build_replay_record(
        base, prompt_ids=[1], continuation_ids=[2], route_digest="e" * 64,
        oracle_id="torch-reference-v1",
        route_hash_status="mismatch",
    )
    assert validate_replay_record(record)


def test_replay_record_rejects_malformed_identity_and_missing_oracle(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"stable")
    runtime = {
        "commit": "a" * 40, "dirty_diff": "b" * 64, "dirty": False,
        "gpu": "gfx1100", "driver": "rocm", "torch": "torch", "rocm": "7",
        "hip": "7", "triton": "3", "jit_sha": "jit", "env_digest": "c" * 64,
    }
    base = build_manifest(
        model=model, prompt="fixed", token_count=2, backend="rocm", quant="Q4_K",
        graph_mode="eager", route="legacy", cache_hits=0, fetches=0, fallbacks=0,
        finite_logits=True, completion_count=2, lane="teacher_forced_replay",
        timings_tok_s=[100], runtime=runtime,
    )
    record = build_replay_record(
        base, prompt_ids=[1], continuation_ids=[2], route_digest="e" * 64,
        oracle_id="torch-reference-v1",
    )
    record["replay"]["continuation_ids_sha256"] = "bad"
    record["replay"]["oracle_id"] = ""

    problems = validate_replay_record(record)

    assert any("continuation identity" in problem for problem in problems)
    assert any("oracle ID" in problem for problem in problems)
