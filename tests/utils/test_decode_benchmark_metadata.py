import math
import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from bench_rocm_matrix import build_manifest, sha256_path, summarize_timings, validate_manifest  # noqa: E402


def _runtime():
    return {
        "commit": "a" * 40,
        "dirty_diff": "b" * 64,
        "dirty": False,
        "gpu": "gfx1100",
        "driver": "rocm-7.2.1",
        "torch": "2.10.0",
        "rocm": "7.2.1",
        "hip": "7.2.1",
        "triton": "3.6.0",
        "jit_sha": "gguf-source-v1",
        "env_digest": "c" * 64,
    }


def _manifest(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    return build_manifest(
        model=model,
        prompt="fixed prompt",
        token_count=8,
        backend="rocm",
        quant="Q4_K",
        graph_mode="eager",
        route="legacy",
        cache_hits=2,
        fetches=0,
        fallbacks=0,
        finite_logits=True,
        completion_count=8,
        lane="teacher_forced_replay",
        timings_tok_s=[100, 110, 105],
        runtime=_runtime(),
    )


def test_file_identity_and_timing_are_content_based(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights").write_bytes(b"one")
    first = sha256_path(model)
    (model / "weights").touch()
    assert sha256_path(model) == first
    summary = summarize_timings([1, 2, 3], warmup=1)
    assert summary["repeats"] == 2
    assert summary["median_tok_s"] == 2.5


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_summarize_timings_rejects_nonfinite_values(value):
    with pytest.raises(ValueError, match="timing values"):
        summarize_timings([value])


def test_manifest_rejects_nonfinite_raw_timing(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["timing"]["raw_tok_s"][0] = math.nan

    assert any("raw_tok_s" in reason for reason in validate_manifest(manifest))


def test_manifest_contains_route_completion_and_runtime_identity(tmp_path):
    manifest = _manifest(tmp_path)
    assert validate_manifest(manifest) == []
    assert manifest["observed"]["completion_count"] == manifest["workload"]["token_count"]
    assert manifest["timing"]["lane"] == "teacher_forced_replay"


@pytest.mark.parametrize("field", ["route", "finite_logits", "completion_count"])
def test_manifest_rejects_missing_correctness_evidence(tmp_path, field):
    manifest = _manifest(tmp_path)
    if field == "route":
        manifest["observed"].pop(field)
    elif field == "finite_logits":
        manifest["observed"][field] = False
    else:
        manifest["observed"][field] = 7
    assert validate_manifest(manifest)
