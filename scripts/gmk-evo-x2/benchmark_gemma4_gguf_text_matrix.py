#!/usr/bin/env python3
"""Measure a fixed Gemma 4 GGUF text workload through the local OpenAI API.

The script is deliberately a client-side benchmark.  It does not start or
stop a server, change model settings, or alter llama-swap.  One warmup request
is discarded, then five fixed-length streamed requests are scored.  Every
sample retains its prompt and completion token usage, time to first visible
token, client-visible prefill rate, decode rate, token-gap distribution, raw
response, and protocol errors.  The resulting JSON is sufficient to audit a
claim without reconstructing timings from a console transcript.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

def percentile(values: list[float], fraction: float) -> float | None:
    """Return a nearest-rank percentile while preserving missing data."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, int(len(ordered) * fraction + 0.999999999))
    return ordered[rank - 1]


def summarize(values: list[float]) -> dict[str, float | None]:
    """Summarize a metric without inventing a value for an absent stream."""
    return {
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def stream_once(base_url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Send one SSE request and retain all visible-event timing boundaries."""
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    stamps: list[float] = []
    pieces: list[str] = []
    usage: dict[str, Any] = {}
    errors: list[str] = []
    completed = False
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310: loopback URL supplied by operator
        for raw in response:
            received = time.perf_counter()
            line = raw.decode("utf-8", errors="strict").rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            data = line[5:].lstrip()
            if data == "[DONE]":
                completed = True
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError as exc:
                errors.append(f"invalid SSE JSON: {exc}")
                continue
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                text = choice.get("text") or ""
                if text:
                    pieces.append(text)
                    stamps.append(received)
    visible = "".join(pieces)
    gaps_ms = [(later - earlier) * 1000 for earlier, later in zip(stamps, stamps[1:])]
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    ttft_s = stamps[0] - started if stamps else None
    decode_s = stamps[-1] - stamps[0] if len(stamps) > 1 else None
    return {
        "completed_sse": completed,
        "text": visible,
        "text_sha256": hashlib.sha256(visible.encode()).hexdigest(),
        "usage": usage,
        "events": len(stamps),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": ttft_s * 1000 if ttft_s is not None else None,
        "prefill_tok_s": prompt_tokens / ttft_s if isinstance(prompt_tokens, int) and ttft_s and ttft_s > 0 else None,
        "decode_tok_s": (completion_tokens - 1) / decode_s if isinstance(completion_tokens, int) and decode_s and decode_s > 0 else None,
        "token_gap_ms": summarize(gaps_ms),
        "wall_s": time.perf_counter() - started,
        "protocol_errors": errors,
    }


def main() -> int:
    """Run the warmup and scored samples, then write one immutable report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if args.samples < 1 or args.max_tokens < 2:
        parser.error("samples must be positive and max-tokens must be at least two")

    prompt = (
        "Write a concise technical explanation of how a graphics processor executes "
        "a quantized mixture-of-experts language model. Use complete sentences and "
        "continue until the requested token limit is reached."
    )
    body = {
        "model": args.model,
        # A raw completion prompt is intentional here.  The server performs
        # its own Gemma GGUF tokenization, and usage.prompt_tokens is the
        # authoritative count for the measured request.
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "add_special_tokens": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    warmup = stream_once(args.base_url, body, args.timeout)
    samples = [stream_once(args.base_url, body, args.timeout) for _ in range(args.samples)]
    report = {
        "schema_version": 1,
        "control": "Gemma4 GGUF fixed-length text matrix",
        "model": args.model,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "requested_samples": args.samples,
        "max_tokens": args.max_tokens,
        "warmup": warmup,
        "samples": samples,
        "summary": {
            "ttft_ms": summarize([s["ttft_ms"] for s in samples if s["ttft_ms"] is not None]),
            "prefill_tok_s": summarize([s["prefill_tok_s"] for s in samples if s["prefill_tok_s"] is not None]),
            "decode_tok_s": summarize([s["decode_tok_s"] for s in samples if s["decode_tok_s"] is not None]),
            "token_gap_p99_ms": percentile([s["token_gap_ms"]["p99"] for s in samples if s["token_gap_ms"]["p99"] is not None], 0.99),
        },
        "passed": all(s["completed_sse"] and not s["protocol_errors"] for s in samples),
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
