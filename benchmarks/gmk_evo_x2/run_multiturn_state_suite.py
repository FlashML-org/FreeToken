#!/usr/bin/env python3
"""Measure a deterministic GMKtec EVO-X2 multi-turn state-retention control.

This is a bounded intermediate workload between single prompts and the
FreeToken paper's tool-using agents. Each turn receives the full prior visible
conversation, records raw SSE timing, and must produce its exact expected
visible answer. It never starts or stops the server.
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


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse explicit workload and server inputs for one immutable artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:1919/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument(
        "--suite",
        default=Path(__file__).with_name("multiturn_state_suite.json"),
        type=Path,
    )
    parser.add_argument("--expected-host", default="david-Gmktec-x2-2")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args(argv)
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    return args


def require_expected_host(expected_host: str) -> str:
    """Fail closed so a local control cannot accidentally target another host."""

    actual_host = socket.gethostname().lower()
    if actual_host not in {expected_host.lower(), expected_host.lower().split(".", 1)[0]}:
        raise RuntimeError(f"refusing multi-turn suite on {actual_host!r}; expected {expected_host!r}")
    return actual_host


def nearest_rank(values: list[float], percentile: float) -> float | None:
    """Return an observed nearest-rank tail statistic for a short stream."""

    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, int(len(ordered) * percentile + 0.999999999) - 1)]


def stream_turn(args: argparse.Namespace, messages: list[dict[str, str]]) -> dict[str, Any]:
    """Send one deterministic chat turn and preserve only visible output events."""

    body = {
        "model": args.model,
        "messages": messages,
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
    gaps = [
        events[index]["offset_seconds"] - events[index - 1]["offset_seconds"]
        for index in range(1, len(events))
    ]
    return {
        "text": "".join(event["content"] for event in events),
        "events": events,
        "usage": usage,
        "errors": errors,
        "ttft_seconds": events[0]["offset_seconds"] if events else None,
        "token_gap_seconds": gaps,
    }


def main(argv: list[str] | None = None) -> int:
    """Execute all turns, retain the complete conversation, and score exact output."""

    args = parse_args(sys.argv[1:] if argv is None else argv)
    host = require_expected_host(args.expected_host)
    if args.artifact.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.artifact}")
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    turns = suite.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("suite must contain a non-empty turns list")
    messages: list[dict[str, str]] = []
    results: list[dict[str, Any]] = []
    for turn in turns:
        if not isinstance(turn, dict) or not isinstance(turn.get("user"), str) or not isinstance(turn.get("expected"), str):
            raise ValueError("every turn requires string user and expected values")
        messages.append({"role": "user", "content": turn["user"]})
        response = stream_turn(args, messages)
        passed = response["text"].strip() == turn["expected"] and not response["errors"]
        results.append({
            "id": turn.get("id"),
            "input_messages": list(messages),
            "expected": turn["expected"],
            "response": response,
            "status": "passed" if passed else "failed",
        })
        messages.append({"role": "assistant", "content": response["text"]})
    ttft = [item["response"]["ttft_seconds"] for item in results if item["response"]["ttft_seconds"] is not None]
    gaps = [gap for item in results for gap in item["response"]["token_gap_seconds"]]
    artifact = {
        "schema_version": 1,
        "host": host,
        "suite": str(args.suite.resolve()),
        "request": {"base_url": args.base_url, "model": args.model, "max_tokens": args.max_tokens},
        "results": results,
        "tail_metrics": {
            "turn_count": len(results),
            "max_ttft_seconds": max(ttft) if ttft else None,
            "mean_ttft_seconds": statistics.mean(ttft) if ttft else None,
            "p95_ttft_seconds": nearest_rank(ttft, 0.95),
            "p99_token_gap_seconds": nearest_rank(gaps, 0.99),
            "max_token_gap_seconds": max(gaps) if gaps else None,
        },
        "status": "passed" if all(item["status"] == "passed" for item in results) else "failed",
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": artifact["status"], "turns": len(results), "tail_metrics": artifact["tail_metrics"]}, sort_keys=True))
    return 0 if artifact["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

