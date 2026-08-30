#!/usr/bin/env python3
"""Measure deterministic long-context retrieval on the isolated LAN-223 API.

This tool deliberately covers the context range exposed by the running Qwen
server.  It is a LAN-223 control, not a replication of the FreeToken paper's
much longer agent sessions.  It places an exact marker at the start of a
deterministic prompt, asks the model to retrieve only that marker, records
every visible SSE event and refuses to overwrite an existing artifact.
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


MARKER = "azure-17"
FILLER = (
    "This is deterministic filler for a context-retrieval control. "
    "Read it without changing the protected marker. "
)


def nearest_rank(values: list[float], percentile: float) -> float | None:
    """Return an observed percentile so short streams do not invent values."""

    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, int(len(ordered) * percentile + 0.999999999) - 1)]


def build_prompt(filler_repetitions: int, marker: str = MARKER) -> str:
    """Build a fixed retrieval prompt with the answer only at its beginning."""

    if filler_repetitions < 1:
        raise ValueError("filler repetitions must be positive")
    return (
        "Protected marker: " + marker + "\n"
        "Do not repeat or transform the marker while reading this material.\n\n"
        + FILLER * filler_repetitions
        + "\n\nReply with only the protected marker and no other text."
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse all material workload controls explicitly for a reproducible run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:1919/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--expected-host", default="lan-223")
    parser.add_argument("--filler-repetitions", type=int, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.filler_repetitions < 1:
        parser.error("--filler-repetitions must be positive")
    return args


def require_expected_host(expected_host: str) -> str:
    """Fail closed so this load never accidentally reaches a different host."""

    actual_host = socket.gethostname().lower()
    expected_short = expected_host.lower().split(".", 1)[0]
    if actual_host not in {expected_host.lower(), expected_short}:
        raise RuntimeError(f"refusing long-context control on {actual_host!r}; expected {expected_host!r}")
    return actual_host


def stream_sample(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    """Send one greedy streaming request and retain visible output timing."""

    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_tokens": args.max_tokens,
        "reasoning_effort": "none",
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    events: list[dict[str, Any]] = []
    raw_sse_events: list[dict[str, Any]] = []
    errors: list[str] = []
    usage: dict[str, Any] | None = None
    done = False
    try:
        with urllib.request.urlopen(request, timeout=args.timeout_seconds) as response:
            for raw_line in response:
                offset = time.perf_counter() - started
                line = raw_line.decode("utf-8", errors="strict").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue
                payload = line[5:].lstrip()
                if payload == "[DONE]":
                    done = True
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError as error:
                    errors.append(f"invalid JSON SSE event: {error}")
                    continue
                raw_sse_events.append({"offset_seconds": offset, "event": event})
                if isinstance(event.get("error"), dict):
                    errors.append(f"server error event: {event['error']}")
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    content = choice.get("delta", {}).get("content")
                    if content:
                        events.append({"offset_seconds": offset, "content": str(content)})
    except urllib.error.HTTPError as error:
        errors.append(f"HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}")
    except urllib.error.URLError as error:
        errors.append(f"transport failure: {error}")
    if not done:
        errors.append("stream ended without [DONE]")
    if not events:
        errors.append("stream contained no visible content events")
    gaps = [events[index]["offset_seconds"] - events[index - 1]["offset_seconds"] for index in range(1, len(events))]
    text = "".join(event["content"] for event in events)
    return {
        "text": text,
        "events": events,
        "raw_sse_events": raw_sse_events,
        "usage": usage,
        "errors": errors,
        "ttft_seconds": events[0]["offset_seconds"] if events else None,
        "token_gap_seconds": gaps,
        "quality_passed": text.strip() == MARKER and not errors,
    }


def main(argv: list[str] | None = None) -> int:
    """Run immutable samples, summarize tails, and exit nonzero on any failure."""

    args = parse_args(sys.argv[1:] if argv is None else argv)
    host = require_expected_host(args.expected_host)
    if args.artifact.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.artifact}")
    prompt = build_prompt(args.filler_repetitions)
    samples = [stream_sample(args, prompt) for _ in range(args.samples)]
    ttft = [sample["ttft_seconds"] for sample in samples if sample["ttft_seconds"] is not None]
    gaps = [gap for sample in samples for gap in sample["token_gap_seconds"]]
    prompt_token_counts = [sample["usage"].get("prompt_tokens") for sample in samples if sample["usage"]]
    artifact = {
        "schema_version": 1,
        "host": host,
        "classification": "LAN-223 long-context control, not paper replication",
        "request": {
            "base_url": args.base_url,
            "model": args.model,
            "filler_repetitions": args.filler_repetitions,
            "max_tokens": args.max_tokens,
            "samples": args.samples,
            "temperature": 0.0,
            "reasoning_effort": "none",
        },
        "prompt": {"marker": MARKER, "character_count": len(prompt), "text": prompt},
        "samples": samples,
        "summary": {
            "sample_count": len(samples),
            "passed_samples": sum(sample["quality_passed"] for sample in samples),
            "prompt_tokens_reported": prompt_token_counts,
            "ttft_seconds": {
                "mean": statistics.mean(ttft) if ttft else None,
                "p50": nearest_rank(ttft, 0.50),
                "p95": nearest_rank(ttft, 0.95),
                "p99": nearest_rank(ttft, 0.99),
                "max": max(ttft) if ttft else None,
            },
            "token_gap_seconds": {
                "p50": nearest_rank(gaps, 0.50),
                "p95": nearest_rank(gaps, 0.95),
                "p99": nearest_rank(gaps, 0.99),
                "max": max(gaps) if gaps else None,
            },
        },
        "status": "passed" if all(sample["quality_passed"] for sample in samples) else "failed",
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": artifact["status"], "summary": artifact["summary"]}, sort_keys=True))
    return 0 if artifact["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
