#!/usr/bin/env python3
"""Measure bounded Gemma 4 concurrent streaming requests through OpenAI SSE.

This is a read-only client benchmark.  It launches no server and changes no
runtime setting.  Each round submits a fixed prompt to a fixed number of
clients, records request-level TTFT, decode rate, token gaps, completion
status, and usage, then summarizes aggregate throughput and tail latency.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float | None:
    """Return an auditable nearest-rank percentile or None for no observations."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, int(len(ordered) * fraction + 0.999999999)) - 1]


def request_once(base_url: str, body: dict[str, Any], timeout: float, barrier: threading.Barrier) -> dict[str, Any]:
    """Synchronize one client with its peers and retain all SSE timing data."""
    barrier.wait()
    started = time.perf_counter()
    stamps: list[float] = []
    pieces: list[str] = []
    usage: dict[str, Any] = {}
    errors: list[str] = []
    completed = False
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(body, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310: operator-supplied loopback URL
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
                    errors.append(str(exc))
                    continue
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    text = choice.get("text") or ""
                    if text:
                        pieces.append(text)
                        stamps.append(received)
    except Exception as exc:  # retain failure evidence instead of hiding it
        errors.append(repr(exc))
    ttft = (stamps[0] - started) if stamps else None
    decode_window = (stamps[-1] - stamps[0]) if len(stamps) > 1 else None
    completion = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    gaps = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]
    return {
        "completed_sse": completed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion,
        "ttft_ms": ttft * 1000 if ttft is not None else None,
        "decode_tok_s": (completion - 1) / decode_window if isinstance(completion, int) and decode_window and decode_window > 0 else None,
        "token_gap_p99_ms": percentile(gaps, 0.99),
        "wall_s": time.perf_counter() - started,
        "text": "".join(pieces),
        "errors": errors,
    }


def main() -> int:
    """Run warmup, then synchronized rounds, and write immutable JSON evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if args.clients < 1 or args.rounds < 1 or args.max_tokens < 2:
        parser.error("clients and rounds must be positive and max-tokens at least two")
    prompt = (
        "Write a concise technical explanation of how a graphics processor executes "
        "a quantized mixture-of-experts language model. Use complete sentences and "
        "continue until the requested token limit is reached."
    )
    body = {"model": args.model, "prompt": prompt, "max_tokens": args.max_tokens,
            "ignore_eos": True, "temperature": 0.0, "top_p": 1.0, "top_k": -1,
            "stream": True, "stream_options": {"include_usage": True}}
    warmup = request_once(args.base_url, body, args.timeout, threading.Barrier(1))
    rounds: list[list[dict[str, Any]]] = []
    for _ in range(args.rounds):
        barrier = threading.Barrier(args.clients)
        with ThreadPoolExecutor(max_workers=args.clients) as executor:
            futures = [executor.submit(request_once, args.base_url, body, args.timeout, barrier) for _ in range(args.clients)]
            rounds.append([future.result() for future in futures])
    observations = [item for group in rounds for item in group]
    ttft = [x["ttft_ms"] for x in observations if x["ttft_ms"] is not None]
    decode = [x["decode_tok_s"] for x in observations if x["decode_tok_s"] is not None]
    gaps = [x["token_gap_p99_ms"] for x in observations if x["token_gap_p99_ms"] is not None]
    total_tokens = sum(x["completion_tokens"] or 0 for x in observations)
    total_wall = sum(x["wall_s"] for x in observations)
    report = {"schema_version": 1, "control": "Gemma4 fixed-length concurrent text matrix",
              "model": args.model, "prompt": prompt, "clients": args.clients, "rounds": args.rounds,
              "max_tokens": args.max_tokens, "warmup": warmup, "rounds_detail": rounds,
              "summary": {"completed": sum(bool(x["completed_sse"]) for x in observations),
                          "requests": len(observations), "ttft_ms": {"mean": statistics.mean(ttft) if ttft else None, "p95": percentile(ttft, .95), "p99": percentile(ttft, .99)},
                          "decode_tok_s": {"mean": statistics.mean(decode) if decode else None, "median": statistics.median(decode) if decode else None, "p95": percentile(decode, .95)},
                          "token_gap_p99_ms": {"mean": statistics.mean(gaps) if gaps else None, "p99": percentile(gaps, .99)},
                          "aggregate_decode_tok_s": total_tokens / total_wall if total_wall > 0 else None},
              "passed": len(observations) == args.clients * args.rounds and all(x["completed_sse"] and not x["errors"] for x in observations)}
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
