#!/usr/bin/env python3
"""Run a small, versioned quality suite against the LAN-223 Qwen API.

The suite is intentionally separate from the paper's agent workloads. It
provides a repeatable precondition for local performance changes: every
candidate must preserve basic exact answers, structured JSON, and the visible
OpenAI response contract before its TPS is considered.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Read every external input explicitly for reproducible quality evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:1919/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument(
        "--suite",
        default=Path(__file__).with_name("quality_suite.json"),
        type=Path,
        help="versioned JSON fixture defining prompts and deterministic checks",
    )
    parser.add_argument("--expected-host", default="lan-223")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args(argv)
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def require_expected_host(expected_host: str) -> str:
    """Refuse any accidental quality traffic directed from another LAN host."""

    actual_host = socket.gethostname().lower()
    accepted = {expected_host.lower(), expected_host.lower().split(".", 1)[0]}
    if actual_host not in accepted:
        raise RuntimeError(
            f"refusing quality suite on host {actual_host!r}; expected {expected_host!r}"
        )
    return actual_host


def request_visible_text(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    """Stream one greedy chat response and preserve content-bearing SSE events."""

    request_body = {
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
        data=json.dumps(request_body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.perf_counter()
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    usage: dict[str, Any] | None = None
    completed = False
    try:
        with urllib.request.urlopen(request, timeout=args.timeout_seconds) as response:
            for raw_line in response:
                offset = time.perf_counter() - started
                line = raw_line.decode("utf-8", errors="strict").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue
                payload = line[5:].lstrip()
                if payload == "[DONE]":
                    completed = True
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError as error:
                    errors.append(f"invalid JSON SSE event: {error}")
                    continue
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    if content:
                        events.append({"offset_seconds": offset, "content": str(content)})
    except urllib.error.HTTPError as error:
        errors.append(f"HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}")
    except urllib.error.URLError as error:
        errors.append(f"transport failure: {error}")
    if not completed:
        errors.append("stream ended without [DONE]")
    if not events:
        errors.append("stream contained no visible content events")
    return {
        "text": "".join(event["content"] for event in events),
        "events": events,
        "usage": usage,
        "errors": errors,
    }


def evaluate_check(text: str, check: dict[str, Any]) -> tuple[bool, str | None]:
    """Evaluate one deterministic fixture rule without model-specific heuristics."""

    kind = check.get("kind")
    if kind == "exact":
        expected = check.get("value")
        passed = text.strip() == expected
        return passed, None if passed else f"expected exactly {expected!r}, got {text.strip()!r}"
    if kind == "json_fields":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            return False, f"visible output is not valid JSON: {error}"
        expected_fields = check.get("fields")
        if not isinstance(parsed, dict) or not isinstance(expected_fields, dict):
            return False, "fixture requires an object and an object field map"
        mismatches = {
            key: {"expected": value, "actual": parsed.get(key)}
            for key, value in expected_fields.items()
            if parsed.get(key) != value
        }
        return not mismatches, None if not mismatches else f"JSON field mismatch: {mismatches}"
    return False, f"unsupported check kind: {kind!r}"


def main(argv: list[str] | None = None) -> int:
    """Run every fixture, write one immutable artifact, and return its pass state."""

    args = parse_args(sys.argv[1:] if argv is None else argv)
    host = require_expected_host(args.expected_host)
    if args.artifact.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.artifact}")
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("suite must contain at least one case")
    results = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("prompt"), str):
            raise ValueError("every suite case requires a string prompt")
        response = request_visible_text(args, case["prompt"])
        check = case.get("check")
        if not isinstance(check, dict):
            raise ValueError(f"case {case.get('id')!r} requires a check object")
        passed, check_error = evaluate_check(response["text"], check)
        results.append({
            "id": case.get("id"),
            "prompt": case["prompt"],
            "check": check,
            "response": response,
            "check_error": check_error,
            "status": "passed" if passed and not response["errors"] else "failed",
        })
    artifact = {
        "schema_version": 1,
        "host": host,
        "request": {
            "base_url": args.base_url,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "reasoning_effort": "none",
        },
        "suite": str(args.suite.resolve()),
        "results": results,
        "status": "passed" if all(item["status"] == "passed" for item in results) else "failed",
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": artifact["status"], "cases": len(results)}, sort_keys=True))
    return 0 if artifact["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

