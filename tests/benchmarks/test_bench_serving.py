from __future__ import annotations

import pytest

from benchmarks.bench_serving import (
    BenchmarkError,
    StreamResult,
    aggregate,
    calibrate_prompt,
    generation_payload,
    parse_args,
    prompt_tolerance,
    sample_metrics,
    snapshot_server,
    stream_chat_completion,
)


def _stream_result(
    *, completion_tokens=4, cached_tokens=None, stamps=(10.0, 11.0, 12.0)
):
    usage = {"prompt_tokens": 100, "completion_tokens": completion_tokens}
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return StreamResult(
        started_at=9.0,
        first_token_at=stamps[0] if stamps else None,
        last_token_at=stamps[-1] if stamps else None,
        done_at=13.0,
        token_timestamps=list(stamps),
        text="abc",
        usage=usage,
    )


def test_parse_args_has_pr1_defaults():
    args = parse_args([])
    assert args.base_url == "http://127.0.0.1:1919"
    assert args.prefill_sizes == (512, 1024, 2048, 4096)
    assert args.decode_tokens == 256
    assert args.cache_prefix_tokens == 4096
    assert args.repetitions == 3


def test_parse_args_rejects_invalid_values():
    with pytest.raises(SystemExit):
        parse_args(["--prefill-sizes", "512,nope"])
    with pytest.raises(SystemExit):
        parse_args(["--decode-tokens", "0"])


def test_generation_payload_disables_thinking_and_requests_usage():
    payload = generation_payload("model", "prompt", 256)
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["ignore_eos"] is True


def test_prompt_tolerance_has_four_token_floor():
    assert prompt_tolerance(512) == 4
    assert prompt_tolerance(4096) == 21


def test_sse_parser_ignores_role_chunk_and_extracts_usage(monkeypatch):
    lines = [
        b": keepalive\n\n",
        b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"A"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"B"}}]}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2}}\n\n',
        b"data: [DONE]\n\n",
    ]

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return iter(lines)

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response())
    result = stream_chat_completion(
        "http://server", generation_payload("model", "prompt", 2), timeout=10
    )
    assert result.text == "AB"
    assert len(result.token_timestamps) == 2
    assert result.usage["prompt_tokens"] == 12
    assert result.done_at >= result.last_token_at


def test_sse_parser_supports_reasoning_content(monkeypatch):
    lines = [
        b'data: {"choices":[{"delta":{"reasoning_content":"thought"}}]}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n',
        b"data: [DONE]\n\n",
    ]

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return iter(lines)

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response())
    result = stream_chat_completion(
        "http://server", generation_payload("model", "prompt", 1), timeout=10
    )
    assert result.text == "thought"


def test_sse_parser_requires_usage(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return iter([b'data: {"choices":[]}\n\n', b"data: [DONE]\n\n"])

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response())
    with pytest.raises(BenchmarkError, match="usage chunk"):
        stream_chat_completion(
            "http://server", generation_payload("model", "prompt", 1), timeout=10
        )


def test_sample_metrics_uses_n_minus_one_decode_steps():
    metrics = sample_metrics(
        _stream_result(completion_tokens=4, cached_tokens=20), target_prompt_tokens=100
    )
    assert metrics["client_wall_ttft_ms"] == pytest.approx(1000.0)
    assert metrics["client_wall_total_ms"] == pytest.approx(4000.0)
    assert metrics["client_wall_decode_tps"] == pytest.approx(1.5)
    assert metrics["client_wall_effective_prefill_tps"] == pytest.approx(80.0)
    assert metrics["cache_hit_ratio"] == pytest.approx(0.2)


def test_sample_metrics_completion_one_does_not_divide_by_zero():
    metrics = sample_metrics(_stream_result(completion_tokens=1, stamps=(10.0,)))
    assert metrics["decode_steps"] == 0
    assert metrics["client_wall_decode_tps"] is None


def test_sample_metrics_missing_cache_report_is_explicit_none():
    metrics = sample_metrics(_stream_result(cached_tokens=None))
    assert metrics["cached_tokens"] is None
    assert metrics["cache_hit_ratio"] is None
    assert metrics["client_wall_effective_prefill_tps"] is None


def test_aggregate_reports_median_min_max_and_ignores_none():
    result = aggregate(
        [
            {"latency": 3, "decode": 10},
            {"latency": 1, "decode": None},
            {"latency": 2, "decode": 8},
        ],
        ("latency", "decode"),
    )
    assert result["sample_count"] == 3
    assert result["latency"] == {"median": 2.0, "min": 1.0, "max": 3.0}
    assert result["decode"] == {"median": 9.0, "min": 8.0, "max": 10.0}


def test_calibrate_prompt_binary_searches_and_keeps_nonce(monkeypatch):
    seen = []

    def fake_count(origin, model_id, prompt, timeout):
        seen.append(prompt)
        return len(prompt) // 4

    monkeypatch.setattr("benchmarks.bench_serving.count_prompt_tokens", fake_count)
    prompt, tokens = calibrate_prompt("http://server", "model", 20, "UNIQUE_NONCE", 10)
    assert "UNIQUE_NONCE" in prompt
    assert abs(tokens - 20) <= prompt_tolerance(20)
    assert seen


def test_snapshot_server_derives_cache_geometry(monkeypatch):
    docs = {
        "/health": {"status": "ok", "version": "1.2", "instance_id": "i"},
        "/v1/models": {"data": [{"id": "model", "max_model_len": 32768}]},
        "/v1/cache/status": {
            "geometry": {
                "num_pages": 128,
                "page_size": 16,
                "moe_cache_size": 40,
                "num_experts": 8,
                "num_moe_layers": 10,
                "num_mamba_slots": 3,
                "cache_budget_bytes": 999,
                "unit_bytes": {
                    "kv_per_token": 2,
                    "moe_per_expert": 5,
                    "mamba_per_slot": 7,
                },
            }
        },
        "/v1/stats": {
            "instance_id": "i",
            "gpus": [{"name": "GPU"}],
            "vram_bytes": 1000,
        },
    }
    monkeypatch.setattr(
        "benchmarks.bench_serving.get_json", lambda origin, path, timeout: docs[path]
    )
    result = snapshot_server("http://server", 10)
    assert result["cache"]["kv_tokens"] == 2048
    assert result["cache"]["kv_vram_bytes"] == 4096
    assert result["cache"]["moe_total_slots"] == 80
    assert result["cache"]["moe_residency"] == pytest.approx(0.5)
    assert result["cache"]["mamba_vram_bytes"] == 21
