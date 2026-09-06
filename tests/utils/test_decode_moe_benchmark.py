import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import bench_decode_moe as benchmark  # noqa: E402


def test_parse_args_exposes_resident_repeat_controls(tmp_path):
    args = benchmark.parse_args(
        ["--model", str(tmp_path / "model"), "--mode", "resident", "--repeats", "3", "--warmup", "1"]
    )

    assert (args.mode, args.repeats, args.warmup) == ("resident", 3, 1)


@pytest.mark.parametrize("option", ["--repeats", "--decode"])
def test_parse_args_rejects_non_measurable_windows(tmp_path, option):
    value = "0" if option == "--repeats" else "1"

    with pytest.raises(SystemExit):
        benchmark.parse_args(["--model", str(tmp_path / "model"), option, value])


def test_parse_args_rejects_incomplete_resident_protocol(tmp_path):
    with pytest.raises(SystemExit):
        benchmark.parse_args(
            ["--model", str(tmp_path / "model"), "--mode", "resident", "--repeats", "2"]
        )
    with pytest.raises(SystemExit):
        benchmark.parse_args(
            [
                "--model", str(tmp_path / "model"), "--mode", "resident", "--repeats", "3",
                "--warmup", "2",
            ]
        )


def test_build_benchmark_row_records_identity_and_incomplete_route(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    args = benchmark.parse_args(
        ["--model", str(model), "--mode", "resident", "--repeats", "3", "--decode", "3"]
    )
    monkeypatch.setattr(benchmark, "runtime_identity", lambda: {"gpu": "test"})
    result = {
        "t0": 10.0,
        "stamps": [11.0, 11.1, 11.2],
        "text": "answer",
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }

    row = benchmark.build_benchmark_row(
        args=args,
        backend="offload",
        problem="fixed prompt",
        sampling={"temperature": 0.0},
        sampling_src="test",
        result=result,
        stats={"vram_bytes": 1024, "instance_id": "server-1"},
        run_index=2,
        server_mode="resident",
        log_path="/tmp/server.log",
        startup_s=12.5,
        warmup_count=1,
    )

    assert row["schema"] == "freetoken-serving-benchmark-v1"
    assert row["status"] == "incomplete"
    assert row["identity"]["model_sha256"] == benchmark.sha256_path(model)
    assert row["identity"]["cache_policy"] == "reuse"
    assert row["identity"]["instance_id"] == "server-1"
    assert row["identity"]["route"] == "unreported"
    assert row["startup_s"] == pytest.approx(12.5)
    assert row["warmup_requests"] == 1
    assert row["decode_tok_s"] == pytest.approx(10.0)


def test_incomplete_row_has_no_promotable_timing(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    args = benchmark.parse_args(["--model", str(model), "--decode", "3"])
    monkeypatch.setattr(benchmark, "runtime_identity", lambda: {"gpu": "test"})

    row = benchmark.incomplete_row(args, "offload", "startup failed", 0)

    assert row["status"] == "incomplete"
    assert row["timing"] == {"status": "incomplete"}
    assert "median_tok_s" not in row["timing"]
