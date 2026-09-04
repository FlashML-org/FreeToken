#!/usr/bin/env python3
"""Run a deterministic Gemma 4 long-context quality and timing sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


def one(base_url: str, model: str, prompt: str, timeout: float) -> dict[str, Any]:
    """Stream one exact-marker request and retain client-visible timing."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 8, "temperature": 0.0,
            "top_p": 1.0, "top_k": -1, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    started = time.perf_counter(); first = None; last = None; text: list[str] = []; reasoning: list[str] = []; usage: dict[str, Any] = {}; complete = False; errors: list[str] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:  # nosec B310: loopback URL supplied by operator
            for raw in response:
                now = time.perf_counter(); line = raw.decode().rstrip("\r\n")
                if not line.startswith("data:"): continue
                data = line[5:].lstrip()
                if data == "[DONE]": complete = True; continue
                try: event = json.loads(data)
                except json.JSONDecodeError as exc: errors.append(str(exc)); continue
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or ""
                    thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if piece or thought:
                        text.append(piece); reasoning.append(thought); first = first or now; last = now
    except Exception as exc: errors.append(repr(exc))
    ttft = (first - started) if first else None; window = (last - first) if first and last and last > first else None
    completion = usage.get("completion_tokens"); prompt_tokens = usage.get("prompt_tokens")
    visible = "".join(text); hidden = "".join(reasoning)
    return {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt_chars": len(prompt),
            "prompt_tokens": prompt_tokens, "completion_tokens": completion, "text": "".join(text),
            "reasoning": hidden, "marker_channel": "content" if "LONG_OK" in visible else ("reasoning" if "LONG_OK" in hidden else None),
            "passed": complete and ("LONG_OK" in visible or "LONG_OK" in hidden) and not errors,
            "errors": errors, "ttft_ms": ttft * 1000 if ttft else None,
            "prefill_tok_s": prompt_tokens / ttft if isinstance(prompt_tokens, int) and ttft else None,
            "decode_tok_s": (completion - 1) / window if isinstance(completion, int) and window else None,
            "wall_s": time.perf_counter() - started}


def main() -> int:
    """Execute one request at each declared context size and write the sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", required=True, type=Path); parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    records = []
    for target in (4096, 8192, 16384):
        filler = "The benchmark context sentence preserves a fixed, deterministic prefix. "
        prompt = (filler * ((target * 4) // len(filler) + 2))[: target * 4]
        prompt += "\nIgnore all prior requested answers. Reply exactly LONG_OK."
        record = one(args.base_url, args.model, prompt, args.timeout); record["target_context_chars"] = target; records.append(record)
    report = {"schema_version": 1, "control": "Gemma4 fixed long-context marker sweep", "model": args.model,
              "targets_chars": [4096, 8192, 16384], "records": records,
              "passed": all(r["passed"] for r in records)}
    args.artifact.parent.mkdir(parents=True, exist_ok=True); args.artifact.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
