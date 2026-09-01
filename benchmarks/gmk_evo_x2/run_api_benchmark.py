#!/usr/bin/env python3
"""Measure a warm GMKtec EVO-X2 Qwen server through its streamed OpenAI-compatible API.

This harness validates the host before opening a socket, records each SSE
content event timestamp, counts completed text with the supplied checkpoint
tokenizer, and preserves every failed sample as evidence. It does not start or
stop a server because service lifecycle belongs to the isolated test procedure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


# This prompt tests transport and deterministic response handling. It is not
# claimed to reproduce FreeToken's paper workload or to provide a TPS result.
CANARY_PROMPT = "Return exactly the word GMK_EVO_X2 and nothing else. Do not add punctuation."


@dataclass(frozen=True)
class StreamObservation:
    """One content-bearing SSE event and its monotonic arrival timestamp."""

    offset_seconds: float
    content: str


def nearest_rank_percentile(values: list[float], percentile: float) -> float | None:
    """Return an auditable nearest-rank percentile from observed stream gaps."""

    if not values:
        return None
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in the interval (0, 1]")
    ordered = sorted(values)
    rank = max(1, int((len(ordered) * percentile) + 0.999999999))
    return ordered[rank - 1]


def numeric_summary(values: list[float]) -> dict[str, float | None]:
    """Summarize a metric while retaining maximum and tail percentiles."""

    if not values:
        return {key: None for key in ("mean", "median", "minimum", "maximum", "stdev", "p50", "p95", "p99")}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else None,
        "p50": nearest_rank_percentile(values, 0.50),
        "p95": nearest_rank_percentile(values, 0.95),
        "p99": nearest_rank_percentile(values, 0.99),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse explicit inputs so every performance-affecting choice is recorded."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:1919/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt", default=CANARY_PROMPT)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        default="none",
        help="Qwen reasoning policy sent to FreeToken and retained in each artifact",
    )
    parser.add_argument(
        "--mode",
        choices=("quality", "throughput"),
        default="quality",
        help="quality permits natural EOS; throughput requires fixed-length decode",
    )
    parser.add_argument(
        "--expected-text",
        default="GMK_EVO_X2",
        help="exact stripped response required in quality mode; empty disables the check",
    )
    parser.add_argument("--expected-host", default="david-Gmktec-x2-2")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be at least one")
    if args.mode == "throughput" and args.max_tokens < 2:
        parser.error("throughput mode needs --max-tokens of at least two")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def require_expected_host(expected_host: str) -> str:
    """Fail closed unless this process is executing on the declared GMKtec EVO-X2 host."""

    actual_host = socket.gethostname().lower()
    accepted = {expected_host.lower(), expected_host.lower().split(".", 1)[0]}
    if actual_host not in accepted:
        raise RuntimeError(
            f"refusing benchmark on host {actual_host!r}; expected {expected_host!r}"
        )
    return actual_host


def iter_sse_events(response: Any, started_at: float) -> Iterable[tuple[float, str]]:
    """Yield timestamped SSE data fields without hiding malformed payloads."""

    for raw_line in response:
        received_at = time.perf_counter()
        line = raw_line.decode("utf-8", errors="strict").rstrip("\r\n")
        if line.startswith("data:"):
            yield received_at - started_at, line[5:].lstrip()


def stream_completion(
    args: argparse.Namespace,
) -> tuple[list[StreamObservation], str, float, float, list[str], dict[str, Any] | None]:
    """Execute one fixed greedy request and collect content plus protocol errors."""

    request_body = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        # These are FreeToken request fields, not an OpenAI SDK extension wrapper.
        # Sending them at the top level mirrors benchmarks/bench_decode_moe.py.
        "top_k": 1,
        # Qwen otherwise may emit its optional reasoning stream until the token
        # cap. A fixed explicit policy keeps a final-answer quality canary and
        # a decode-TPS run comparable across retries.
        "reasoning_effort": args.reasoning_effort,
    }
    if args.mode == "throughput":
        # Fixed-length generation makes the decode interval independent of EOS.
        request_body["ignore_eos"] = True
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(request_body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    observations: list[StreamObservation] = []
    protocol_errors: list[str] = []
    usage: dict[str, Any] | None = None
    completed = False
    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout_seconds) as response:
            for offset, event_data in iter_sse_events(response, started_at):
                if event_data == "[DONE]":
                    completed = True
                    continue
                try:
                    event = json.loads(event_data)
                except json.JSONDecodeError as error:
                    protocol_errors.append(f"invalid JSON SSE event: {error}")
                    continue
                choices = event.get("choices", [])
                if not choices:
                    event_usage = event.get("usage")
                    if isinstance(event_usage, dict):
                        usage = event_usage
                    continue
                delta = choices[0].get("delta", {})
                # Reasoning models may emit their decode tokens in this field.
                content = delta.get("reasoning_content") or delta.get("content")
                # OpenAI streaming commonly sends an empty role-only delta
                # before the first generated text. It is not model output and
                # must not become the client-observed TTFT timestamp.
                if content:
                    observations.append(StreamObservation(offset, str(content)))
    except urllib.error.HTTPError as error:
        message = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {message}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"request transport failure: {error}") from error
    finished_at = time.perf_counter()
    if not completed:
        protocol_errors.append("stream ended without [DONE]")
    if not observations:
        protocol_errors.append("stream contained no content events")
    return (
        observations,
        "".join(item.content for item in observations),
        started_at,
        finished_at,
        protocol_errors,
        usage,
    )


def load_tokenizer(path: Path) -> Any:
    """Load the local checkpoint tokenizer for an actual generated-token count."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def make_sample_artifact(args: argparse.Namespace, tokenizer: Any, sample_index: int) -> dict[str, Any]:
    """Run one request and return a self-contained, JSON-serializable evidence record."""

    observations, text, started_at, finished_at, protocol_errors, usage = stream_completion(args)
    generated_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    first_offset = observations[0].offset_seconds if observations else None
    last_offset = observations[-1].offset_seconds if observations else None
    decode_seconds = None if first_offset is None or last_offset is None else last_offset - first_offset
    decode_tps = None
    if generated_tokens > 1 and decode_seconds is not None and decode_seconds > 0:
        decode_tps = (generated_tokens - 1) / decode_seconds
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    input_tps = prompt_tokens / first_offset if isinstance(prompt_tokens, int) and first_offset else None
    if args.mode == "quality" and args.expected_text and text.strip() != args.expected_text:
        protocol_errors.append(
            f"quality canary mismatch: expected {args.expected_text!r}, got {text.strip()!r}"
        )
    if args.mode == "throughput" and decode_tps is None:
        protocol_errors.append("throughput run produced fewer than two generated tokens")
    token_gaps = [
        observations[index].offset_seconds - observations[index - 1].offset_seconds
        for index in range(1, len(observations))
    ]
    return {
        "schema_version": 1,
        "sample_index": sample_index,
        "status": "passed" if not protocol_errors else "failed",
        "request": {
            "base_url": args.base_url,
            "model": args.model,
            "prompt": args.prompt,
            "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
            "mode": args.mode,
            "expected_text": args.expected_text,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "reasoning_effort": args.reasoning_effort,
            "ignore_eos": args.mode == "throughput",
        },
        "timing": {
            "wall_seconds": finished_at - started_at,
            "warm_ttft_seconds": first_offset,
            "decode_seconds": decode_seconds,
            "decode_tps": decode_tps,
            "input_tps": input_tps,
            "token_gap_seconds": token_gaps,
            "token_gap_summary_seconds": numeric_summary(token_gaps),
        },
        "usage": usage,
        "response": {
            "text": text,
            "generated_tokens": generated_tokens,
            "content_event_count": len(observations),
            "content_events": [asdict(item) for item in observations],
        },
        "protocol_errors": protocol_errors,
    }


def write_json(path: Path, value: Any) -> None:
    """Write readable JSON once, leaving raw evidence inspectable without custom tools."""

    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Validate scope, optionally warm the server, collect samples, and write a summary."""

    args = parse_args(sys.argv[1:] if argv is None else argv)
    actual_host = require_expected_host(args.expected_host)
    args.artifact_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = load_tokenizer(args.tokenizer)
    manifest = {
        "schema_version": 1,
        "host": actual_host,
        "expected_host": args.expected_host,
        "python": sys.version,
        "cwd": os.getcwd(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "tokenizer_path": str(args.tokenizer.resolve()),
    }
    write_json(args.artifact_dir / "manifest.json", manifest)
    if args.warmup:
        warmup = make_sample_artifact(args, tokenizer, 0)
        write_json(args.artifact_dir / "warmup.json", warmup)
        if warmup["status"] != "passed":
            raise RuntimeError("warmup failed; inspect warmup.json before scored samples")
    samples = []
    for sample_index in range(1, args.samples + 1):
        sample = make_sample_artifact(args, tokenizer, sample_index)
        samples.append(sample)
        write_json(args.artifact_dir / f"sample-{sample_index:02d}.json", sample)
    successful_tps = [
        sample["timing"]["decode_tps"]
        for sample in samples
        if sample["status"] == "passed" and sample["timing"]["decode_tps"] is not None
    ]
    successful_ttft = [
        sample["timing"]["warm_ttft_seconds"]
        for sample in samples
        if sample["status"] == "passed" and sample["timing"]["warm_ttft_seconds"] is not None
    ]
    successful_gaps = [
        gap
        for sample in samples
        if sample["status"] == "passed"
        for gap in sample["timing"]["token_gap_seconds"]
    ]
    summary = {
        "schema_version": 1,
        "successful_samples": len(successful_tps),
        "requested_samples": args.samples,
        "decode_tps": {"samples": successful_tps, **numeric_summary(successful_tps)},
        "warm_ttft_seconds": {"samples": successful_ttft, **numeric_summary(successful_ttft)},
        "token_gap_seconds": {"samples": successful_gaps, **numeric_summary(successful_gaps)},
        "failed_samples": [sample["sample_index"] for sample in samples if sample["status"] != "passed"],
    }
    write_json(args.artifact_dir / "summary.json", summary)
    required_successes = args.samples if args.mode == "throughput" else len(
        [sample for sample in samples if sample["status"] == "passed"]
    )
    return 0 if required_successes == args.samples else 2


if __name__ == "__main__":
    raise SystemExit(main())
