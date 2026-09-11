"""Client-wall serving benchmark for an already-running FreeToken server.

The benchmark uses the public API: prompt sizes are calibrated with
``/v1/messages/count_tokens`` and streamed ``/v1/chat/completions`` events are timestamped
as they arrive. The server's cache geometry is recorded next to the measurements.

Run from the repository root::

    PYTHONPATH=python:. python benchmarks/bench_serving.py --json serving-bench.json

The server must already be running. Keeping startup and cache allocation outside the first
version makes the request measurements comparable and also permits remote endpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:1919"
DEFAULT_PREFILL_SIZES = (512, 1024, 2048, 4096)
DEFAULT_DECODE_TOKENS = 256
DEFAULT_CACHE_PREFIX_TOKENS = 4096
DEFAULT_CACHE_SUFFIX_TOKENS = 32
DEFAULT_REPETITIONS = 3
DEFAULT_WARMUP_RUNS = 1
DEFAULT_REQUEST_TIMEOUT = 1800.0
PROMPT_TOLERANCE_RATIO = 0.005
PROMPT_TOLERANCE_MIN = 4
CALIBRATION_MAX_CHARS = 4 * 1024 * 1024

FILLER_CORPUS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi "
    "rho sigma tau upsilon phi chi psi omega serving benchmark request latency prompt "
    "cache runtime measurement repeatable client stream context token throughput system "
)


class BenchmarkError(RuntimeError):
    """A user-facing benchmark failure."""


@dataclass
class StreamResult:
    started_at: float
    first_token_at: float | None
    last_token_at: float | None
    done_at: float
    token_timestamps: list[float]
    text: str
    usage: dict[str, Any]


def parse_prefill_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "prefill sizes must be comma-separated integers"
        ) from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("prefill sizes must contain positive integers")
    return sizes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="running FreeToken URL"
    )
    parser.add_argument(
        "--prefill-sizes",
        type=parse_prefill_sizes,
        default=DEFAULT_PREFILL_SIZES,
        help="target prompt sizes in tokens (default: 512,1024,2048,4096)",
    )
    parser.add_argument("--decode-tokens", type=int, default=DEFAULT_DECODE_TOKENS)
    parser.add_argument(
        "--cache-prefix-tokens", type=int, default=DEFAULT_CACHE_PREFIX_TOKENS
    )
    parser.add_argument(
        "--cache-suffix-tokens", type=int, default=DEFAULT_CACHE_SUFFIX_TOKENS
    )
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument(
        "--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT
    )
    parser.add_argument(
        "--json", dest="json_out", help="write the complete result to this file"
    )
    parser.add_argument(
        "--label", default=None, help="optional label stored in run metadata"
    )
    args = parser.parse_args(argv)
    if args.decode_tokens < 1:
        parser.error("--decode-tokens must be positive")
    if args.cache_prefix_tokens < 1:
        parser.error("--cache-prefix-tokens must be positive")
    if args.cache_suffix_tokens < 0:
        parser.error("--cache-suffix-tokens cannot be negative")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    return args


def _origin(base_url: str) -> str:
    return base_url.rstrip("/")


def _decode_error_body(error: urllib.error.HTTPError) -> str:
    try:
        return error.read(1000).decode("utf-8", errors="replace")
    except OSError:
        return "<unreadable response>"


def request_json(
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float,
) -> dict[str, Any]:
    data = (
        json.dumps(body, ensure_ascii=False).encode("utf-8")
        if body is not None
        else None
    )
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        raise BenchmarkError(
            f"{method} {url} failed with HTTP {exc.code}: {_decode_error_body(exc)}"
        ) from exc
    except (OSError, ValueError) as exc:
        raise BenchmarkError(f"{method} {url} failed: {exc}") from exc
    if not isinstance(result, dict):
        raise BenchmarkError(f"{method} {url} returned a non-object JSON response")
    return result


def get_json(origin: str, path: str, timeout: float) -> dict[str, Any]:
    return request_json("GET", f"{origin}{path}", timeout=timeout)


def post_json(
    origin: str, path: str, body: dict[str, Any], timeout: float
) -> dict[str, Any]:
    return request_json("POST", f"{origin}{path}", body=body, timeout=timeout)


def _usage_cached_tokens(usage: dict[str, Any]) -> int | None:
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict) or "cached_tokens" not in details:
        return None
    value = details["cached_tokens"]
    return int(value) if value is not None else None


def _delta_text(delta: dict[str, Any]) -> str:
    for value in (delta.get("reasoning_content"), delta.get("content")):
        if isinstance(value, str) and value:
            return value
    return ""


def stream_chat_completion(
    origin: str, payload: dict[str, Any], *, timeout: float
) -> StreamResult:
    request = urllib.request.Request(
        f"{origin}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    started_at = time.perf_counter()
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise BenchmarkError(
            f"POST {origin}/v1/chat/completions failed with HTTP {exc.code}: "
            f"{_decode_error_body(exc)}"
        ) from exc
    except OSError as exc:
        raise BenchmarkError(
            f"POST {origin}/v1/chat/completions failed: {exc}"
        ) from exc

    timestamps: list[float] = []
    pieces: list[str] = []
    usage: dict[str, Any] | None = None
    done_at: float | None = None
    with response:
        for raw_line in response:
            line = raw_line.strip()
            if not line or not line.startswith(b"data:"):
                continue
            data = line[len(b"data:") :].strip()
            if data == b"[DONE]":
                done_at = time.perf_counter()
                break
            try:
                chunk = json.loads(data)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BenchmarkError(
                    f"invalid SSE JSON from {origin}: {data[:200]!r}"
                ) from exc
            if not isinstance(chunk, dict):
                continue
            candidate_usage = chunk.get("usage")
            if isinstance(candidate_usage, dict):
                usage = candidate_usage
            for choice in chunk.get("choices", []):
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    continue
                text = _delta_text(delta)
                if text:
                    now = time.perf_counter()
                    timestamps.append(now)
                    pieces.append(text)
    if done_at is None:
        done_at = time.perf_counter()
    if usage is None:
        raise BenchmarkError(
            "stream ended without a usage chunk; use a FreeToken server with include_usage"
        )
    if "prompt_tokens" not in usage or "completion_tokens" not in usage:
        raise BenchmarkError(
            f"usage is missing prompt/completion token counts: {usage!r}"
        )
    return StreamResult(
        started_at=started_at,
        first_token_at=timestamps[0] if timestamps else None,
        last_token_at=timestamps[-1] if timestamps else None,
        done_at=done_at,
        token_timestamps=timestamps,
        text="".join(pieces),
        usage=usage,
    )


def generation_payload(model_id: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "thinking": {"type": "disabled"},
    }


def count_prompt_tokens(origin: str, model_id: str, prompt: str, timeout: float) -> int:
    result = post_json(
        origin,
        "/v1/messages/count_tokens",
        {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "disabled"},
        },
        timeout,
    )
    value = result.get("input_tokens")
    if not isinstance(value, int) or value < 0:
        raise BenchmarkError(f"count_tokens returned invalid input_tokens: {result!r}")
    return value


def _prompt_with_chars(nonce: str, chars: int) -> str:
    filler = (FILLER_CORPUS * (chars // len(FILLER_CORPUS) + 1))[:chars]
    return f"{nonce}\n{filler}"


def prompt_tolerance(target: int) -> int:
    return max(PROMPT_TOLERANCE_MIN, math.ceil(target * PROMPT_TOLERANCE_RATIO))


def calibrate_prompt(
    origin: str, model_id: str, target_tokens: int, nonce: str, timeout: float
) -> tuple[str, int]:
    """Find a deterministic prompt close to a token target without generation."""
    if target_tokens < 1:
        raise BenchmarkError("prompt target must be positive")

    def measure(chars: int) -> tuple[str, int]:
        prompt = _prompt_with_chars(nonce, chars)
        return prompt, count_prompt_tokens(origin, model_id, prompt, timeout)

    low_chars = 0
    low_prompt, low_tokens = measure(0)
    if low_tokens > target_tokens:
        return low_prompt, low_tokens
    high_chars = min(CALIBRATION_MAX_CHARS, max(32, target_tokens * 8))
    high_prompt, high_tokens = measure(high_chars)
    while high_tokens < target_tokens and high_chars < CALIBRATION_MAX_CHARS:
        low_chars, low_prompt, low_tokens = high_chars, high_prompt, high_tokens
        high_chars = min(CALIBRATION_MAX_CHARS, high_chars * 2)
        high_prompt, high_tokens = measure(high_chars)
    if high_tokens < target_tokens:
        raise BenchmarkError(
            f"could not calibrate {target_tokens} tokens within {CALIBRATION_MAX_CHARS} characters "
            f"(last count: {high_tokens})"
        )

    candidates = [(low_prompt, low_tokens), (high_prompt, high_tokens)]
    while low_chars + 1 < high_chars:
        mid_chars = (low_chars + high_chars) // 2
        mid_prompt, mid_tokens = measure(mid_chars)
        candidates.append((mid_prompt, mid_tokens))
        if mid_tokens < target_tokens:
            low_chars, low_prompt, low_tokens = mid_chars, mid_prompt, mid_tokens
        else:
            high_chars, high_prompt, high_tokens = mid_chars, mid_prompt, mid_tokens
    best_prompt, best_tokens = min(
        candidates, key=lambda item: abs(item[1] - target_tokens)
    )
    if abs(best_tokens - target_tokens) > prompt_tolerance(target_tokens):
        raise BenchmarkError(
            f"could not calibrate prompt to {target_tokens} ± {prompt_tolerance(target_tokens)} tokens "
            f"(closest count: {best_tokens})"
        )
    return best_prompt, best_tokens


def sample_metrics(
    result: StreamResult, target_prompt_tokens: int | None = None
) -> dict[str, Any]:
    usage = result.usage
    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    if result.first_token_at is None:
        raise BenchmarkError("stream contained no non-empty token delta")
    ttft_s = result.first_token_at - result.started_at
    total_s = result.done_at - result.started_at
    decode_span_s = (
        result.last_token_at - result.first_token_at
        if result.last_token_at is not None and len(result.token_timestamps) >= 2
        else None
    )
    decode_steps = max(0, completion_tokens - 1)
    cached_tokens = _usage_cached_tokens(usage)
    new_prompt_tokens = (
        prompt_tokens - cached_tokens if cached_tokens is not None else None
    )
    return {
        "target_prompt_tokens": target_prompt_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "cache_hit_ratio": cached_tokens / prompt_tokens
        if cached_tokens is not None and prompt_tokens
        else None,
        "client_wall_ttft_ms": ttft_s * 1000,
        "client_wall_total_ms": total_s * 1000,
        "client_wall_decode_tps": decode_steps / decode_span_s
        if decode_span_s and decode_steps
        else None,
        "client_wall_effective_prefill_tps": new_prompt_tokens / ttft_s
        if new_prompt_tokens is not None and ttft_s > 0
        else None,
        "decode_steps": decode_steps,
        "sse_token_events": len(result.token_timestamps),
    }


def aggregate(
    samples: Iterable[dict[str, Any]], fields: Iterable[str]
) -> dict[str, Any]:
    rows = list(samples)
    result: dict[str, Any] = {"sample_count": len(rows)}
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        result[field] = (
            {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
            if values
            else {"median": None, "min": None, "max": None}
        )
    return result


def _first_model(models: dict[str, Any]) -> dict[str, Any]:
    data = models.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise BenchmarkError(f"/v1/models returned no model cards: {models!r}")
    return data[0]


def snapshot_server(origin: str, timeout: float) -> dict[str, Any]:
    health = get_json(origin, "/health", timeout)
    if health.get("status") == "error":
        raise BenchmarkError(
            f"server reports an error: {health.get('message', health)}"
        )
    if (
        health.get("status") != "ok"
        or health.get("maintenance", "serving") != "serving"
    ):
        raise BenchmarkError(f"server is not ready for benchmarking: {health!r}")
    models = get_json(origin, "/v1/models", timeout)
    cache_status = get_json(origin, "/v1/cache/status", timeout)
    stats = get_json(origin, "/v1/stats", timeout)
    card = _first_model(models)
    stats_model = stats.get("model") if isinstance(stats.get("model"), dict) else {}
    context = (
        card.get("max_model_len")
        or card.get("context_length")
        or stats_model.get("ctx")
    )
    geometry = (
        cache_status.get("geometry")
        if isinstance(cache_status.get("geometry"), dict)
        else {}
    )
    unit_bytes = (
        geometry.get("unit_bytes")
        if isinstance(geometry.get("unit_bytes"), dict)
        else {}
    )
    page_size = int(geometry.get("page_size") or 0)
    num_pages = int(geometry.get("num_pages") or 0)
    kv_tokens = num_pages * page_size if page_size and num_pages else 0
    moe_slots = int(geometry.get("moe_cache_size") or 0)
    moe_total_slots = int(geometry.get("num_experts") or 0) * int(
        geometry.get("num_moe_layers") or 0
    )
    kv_per_token = int(
        unit_bytes.get("kv_per_token", unit_bytes.get("kv_bytes_per_token", 0)) or 0
    )
    moe_per_expert = int(
        unit_bytes.get("moe_per_expert", unit_bytes.get("moe_bytes_per_expert", 0)) or 0
    )
    mamba_per_slot = int(
        unit_bytes.get("mamba_per_slot", unit_bytes.get("mamba_bytes_per_slot", 0)) or 0
    )
    mamba_slots = int(geometry.get("num_mamba_slots") or 0)
    return {
        "server": {
            "version": health.get("version"),
            "instance_id": health.get("instance_id") or stats.get("instance_id"),
            "base_url": origin,
        },
        "model": {
            "id": card.get("id") or stats_model.get("id"),
            "context_capacity_tokens": int(context) if context is not None else None,
        },
        "hardware": {
            "gpus": stats.get("gpus", []),
            "vram_bytes": int(stats.get("vram_bytes") or 0),
        },
        "cache": {
            "kv_tokens": kv_tokens,
            "kv_vram_bytes": kv_tokens * kv_per_token,
            "moe_slots": moe_slots,
            "moe_total_slots": moe_total_slots,
            "moe_residency": moe_slots / moe_total_slots if moe_total_slots else None,
            "moe_vram_bytes": moe_slots * moe_per_expert,
            "mamba_slots": mamba_slots,
            "mamba_vram_bytes": mamba_slots * mamba_per_slot,
            "cache_budget_bytes": int(geometry.get("cache_budget_bytes") or 0),
            "unit_bytes": {
                "kv_per_token": kv_per_token,
                "moe_per_expert": moe_per_expert,
                "mamba_per_slot": mamba_per_slot,
            },
        },
    }


def _suffix(tokens: int) -> str:
    return (
        ""
        if tokens <= 0
        else "\n" + _prompt_with_chars("BENCHMARK_SUFFIX", max(1, tokens * 5))
    )


def run_request(
    origin: str, model_id: str, prompt: str, max_tokens: int, timeout: float
) -> dict[str, Any]:
    result = stream_chat_completion(
        origin, generation_payload(model_id, prompt, max_tokens), timeout=timeout
    )
    return sample_metrics(result)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    origin = _origin(args.base_url)
    snapshot = snapshot_server(origin, args.request_timeout)
    model_id = snapshot["model"]["id"]
    if not isinstance(model_id, str) or not model_id:
        raise BenchmarkError("could not determine the served model id from /v1/models")
    run_id = uuid.uuid4().hex[:12]
    print(
        f"FreeToken serving benchmark\nmodel: {model_id}\nserver: {origin}", flush=True
    )
    print(
        f"KV: {snapshot['cache']['kv_tokens']} tokens; "
        f"MoE: {snapshot['cache']['moe_slots']}/{snapshot['cache']['moe_total_slots']} slots",
        flush=True,
    )

    for warmup in range(args.warmup_runs):
        prompt, _ = calibrate_prompt(
            origin,
            model_id,
            256,
            f"BENCHMARK_WARMUP_{run_id}_{warmup}",
            args.request_timeout,
        )
        run_request(origin, model_id, prompt, 32, args.request_timeout)
    if args.warmup_runs:
        print(f"Warmup: {args.warmup_runs} run(s)", flush=True)

    prefill_targets: list[dict[str, Any]] = []
    prefill_fields = (
        "client_wall_ttft_ms",
        "client_wall_effective_prefill_tps",
        "client_wall_total_ms",
    )
    for target in args.prefill_sizes:
        samples: list[dict[str, Any]] = []
        for repetition in range(args.repetitions):
            prompt, _ = calibrate_prompt(
                origin,
                model_id,
                target,
                f"BENCHMARK_PREFILL_{run_id}_{target}_{repetition}",
                args.request_timeout,
            )
            sample = run_request(origin, model_id, prompt, 1, args.request_timeout)
            sample.update(target_prompt_tokens=target, repetition=repetition)
            samples.append(sample)
        prefill_targets.append(
            {
                "target_prompt_tokens": target,
                "samples": samples,
                "summary": aggregate(samples, prefill_fields),
            }
        )

    decode_samples: list[dict[str, Any]] = []
    decode_prompt_target = min(512, args.prefill_sizes[0])
    for repetition in range(args.repetitions):
        prompt, _ = calibrate_prompt(
            origin,
            model_id,
            decode_prompt_target,
            f"BENCHMARK_DECODE_{run_id}_{repetition}",
            args.request_timeout,
        )
        sample = run_request(
            origin, model_id, prompt, args.decode_tokens, args.request_timeout
        )
        sample["repetition"] = repetition
        decode_samples.append(sample)

    prefix_samples: list[dict[str, Any]] = []
    for repetition in range(args.repetitions):
        base, _ = calibrate_prompt(
            origin,
            model_id,
            args.cache_prefix_tokens,
            f"BENCHMARK_PREFIX_{run_id}_{repetition}",
            args.request_timeout,
        )
        row: dict[str, Any] = {"repetition": repetition}
        for name, prompt in {
            "fresh": base,
            "identical": base,
            "small_suffix": base + _suffix(args.cache_suffix_tokens),
        }.items():
            sample = run_request(origin, model_id, prompt, 1, args.request_timeout)
            sample["case"] = name
            row[name] = sample
        prefix_samples.append(row)

    result = {
        "schema_version": 1,
        "run": {
            "label": args.label or f"{model_id}-serving-{run_id}",
            "timestamp_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "run_id": run_id,
        },
        "server": snapshot["server"],
        "model": snapshot["model"],
        "hardware": snapshot["hardware"],
        "cache": snapshot["cache"],
        "configuration": {
            "thinking": "disabled",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "ignore_eos": True,
            "warmup_runs": args.warmup_runs,
            "repetitions": args.repetitions,
            "decode_tokens": args.decode_tokens,
            "cache_prefix_tokens": args.cache_prefix_tokens,
            "cache_suffix_tokens": args.cache_suffix_tokens,
        },
        "prefill": {"targets": prefill_targets},
        "decode": {
            "samples": decode_samples,
            "summary": aggregate(
                decode_samples,
                (
                    "client_wall_ttft_ms",
                    "client_wall_decode_tps",
                    "client_wall_total_ms",
                ),
            ),
        },
        "prefix_cache": {"samples": prefix_samples},
    }
    print_human_summary(result)
    return result


def _summary_median(result: dict[str, Any], field: str) -> Any:
    value = result.get("summary", {}).get(field, {})
    return value.get("median") if isinstance(value, dict) else None


def _format_number(value: Any, suffix: str = "") -> str:
    return f"{value:.2f}{suffix}" if isinstance(value, (int, float)) else "n/a"


def print_human_summary(result: dict[str, Any]) -> None:
    print("\nFresh prefill\n target   median TTFT       effective tok/s", flush=True)
    for target in result["prefill"]["targets"]:
        print(
            f" {target['target_prompt_tokens']:>6}   "
            f"{_format_number(_summary_median(target, 'client_wall_ttft_ms'), ' ms'):>16}   "
            f"{_format_number(_summary_median(target, 'client_wall_effective_prefill_tps')):>18}",
            flush=True,
        )
    decode = result["decode"]
    print(
        "\nDecode\n"
        f" client-wall median: {_format_number(_summary_median(decode, 'client_wall_decode_tps'), ' tok/s')}\n"
        f" TTFT median:        {_format_number(_summary_median(decode, 'client_wall_ttft_ms'), ' ms')}",
        flush=True,
    )
    print("\nPrefix cache", flush=True)
    for name in ("fresh", "identical", "small_suffix"):
        values = [
            row[name]["client_wall_ttft_ms"]
            for row in result["prefix_cache"]["samples"]
        ]
        print(
            f" {name:>12}: {_format_number(statistics.median(values), ' ms')}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_benchmark(args)
        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n"
            )
            print(f"\nJSON: {args.json_out}", flush=True)
    except BenchmarkError as exc:
        print(f"bench_serving: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
