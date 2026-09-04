#!/usr/bin/env python3
"""Run a bounded deterministic Gemma 4 endurance and recovery control."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


def request_once(base_url: str, model: str, timeout: float) -> dict[str, Any]:
    """Request the exact arithmetic marker and retain response timing."""
    prompt = "What is 17 times 19? Reply with only the decimal number 323."
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16, "temperature": 0.0, "top_p": 1.0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    started = time.perf_counter(); first = None; text: list[str] = []; usage: dict[str, Any] = {}; done = False; errors: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:  # nosec B310: local endpoint supplied by operator
            for raw in response:
                now = time.perf_counter(); line = raw.decode().rstrip("\r\n")
                if not line.startswith("data:"): continue
                data = line[5:].lstrip()
                if data == "[DONE]": done = True; continue
                try: event = json.loads(data)
                except json.JSONDecodeError as exc: errors.append(str(exc)); continue
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content") or ""
                    if piece: text.append(piece); first = first or now
    except Exception as exc: errors.append(repr(exc))
    answer = "".join(text).strip()
    return {"passed": done and answer == "323" and not errors, "answer": answer,
            "usage": usage, "ttft_ms": (first - started) * 1000 if first else None,
            "wall_s": time.perf_counter() - started, "errors": errors}


def main() -> int:
    """Run sequential sessions at a fixed cadence and write immutable evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", required=True, type=Path); parser.add_argument("--sessions", type=int, default=30)
    parser.add_argument("--interval", type=float, default=1.0); parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if args.sessions < 1 or args.interval < 0: parser.error("sessions must be positive and interval nonnegative")
    records = []
    for index in range(1, args.sessions + 1):
        record = request_once(args.base_url, args.model, args.timeout); record["session"] = index; records.append(record)
        if not record["passed"]: break
        if index < args.sessions: time.sleep(args.interval)
    report = {"schema_version": 1, "control": "Gemma4 bounded deterministic endurance", "model": args.model,
              "requested_sessions": args.sessions, "completed_sessions": len(records), "interval_s": args.interval,
              "records": records, "passed": len(records) == args.sessions and all(r["passed"] for r in records)}
    args.artifact.parent.mkdir(parents=True, exist_ok=True); args.artifact.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
