#!/usr/bin/env python3
"""Benchmark one already-running local OpenAI-compatible server reproducibly.

The client is loopback-only by design. It never starts, stops, or reconfigures a
server. It writes immutable per-request JSON evidence, counts generated text
with the supplied checkpoint tokenizer, and requires an explicit visible-text
quality expectation in quality mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from benchmarks.lan223_qwen.run_api_benchmark import (
    StreamObservation,
    iter_sse_events,
    load_tokenizer,
    numeric_summary,
    write_json,
)


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def require_loopback_url(value: str) -> str:
    """Reject non-loopback targets before this client can send a request."""

    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("--base-url must be an absolute http(s) URL")
    if parsed.hostname.lower() not in LOOPBACK_HOSTS:
        raise ValueError("--base-url must target a loopback host: localhost, 127.0.0.1, or ::1")
    return value.rstrip("/")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Require every request and measurement choice to be explicit."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    parser.add_argument("--expected-text", default="")
    parser.add_argument("--mode", choices=("quality", "throughput"), default="quality")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.base_url = require_loopback_url(args.base_url)
    except ValueError as error:
        parser.error(str(error))
    if args.prompt_file:
        try:
            args.prompt = args.prompt_file.read_text(encoding="utf-8")
        except OSError as error:
            parser.error(f"cannot read --prompt-file: {error}")
    if not args.prompt:
        parser.error("prompt text must not be empty")
    if args.samples < 1:
        parser.error("--samples must be at least one")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.mode == "quality" and not args.expected_text:
        parser.error("quality mode requires --expected-text")
    if args.mode == "throughput" and args.max_tokens < 2:
        parser.error("throughput mode needs --max-tokens of at least two")
    return args


def stream_request(args: argparse.Namespace) -> tuple[list[StreamObservation], str, float, list[str], dict[str, Any] | None]:
    """Send one fixed greedy request and preserve protocol failures."""

    body: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
    }
    if args.mode == "throughput":
        body["ignore_eos"] = True
    request = urllib.request.Request(
        args.base_url + "/chat/completions",
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started_at = time.perf_counter()
    observations: list[StreamObservation] = []
    errors: list[str] = []
    usage: dict[str, Any] | None = None
    completed = False
    try:
        with urllib.request.urlopen(request, timeout=args.timeout_seconds) as response:
            for offset, event_data in iter_sse_events(response, started_at):
                if event_data == "[DONE]":
                    completed = True
                    continue
                try:
                    event = json.loads(event_data)
                except json.JSONDecodeError as error:
                    errors.append(f"invalid JSON SSE event: {error}")
                    continue
                if not event.get("choices"):
                    if isinstance(event.get("usage"), dict):
                        usage = event["usage"]
                    continue
                delta = event["choices"][0].get("delta", {})
                content = delta.get("reasoning_content") or delta.get("content")
                if content:
                    observations.append(StreamObservation(offset, str(content)))
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"request transport failure: {error}") from error
    if not completed:
        errors.append("stream ended without [DONE]")
    if not observations:
        errors.append("stream contained no content events")
    return observations, "".join(item.content for item in observations), started_at, errors, usage


def run_sample(args: argparse.Namespace, tokenizer: Any, sample_index: int) -> dict[str, Any]:
    """Produce one self-contained artifact with client timing and quality state."""

    observations, text, started_at, errors, usage = stream_request(args)
    finished_at = time.perf_counter()
    generated_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    first = observations[0].offset_seconds if observations else None
    last = observations[-1].offset_seconds if observations else None
    decode_seconds = last - first if first is not None and last is not None else None
    decode_tps = None
    if generated_tokens > 1 and decode_seconds and decode_seconds > 0:
        decode_tps = (generated_tokens - 1) / decode_seconds
    if args.mode == "quality" and text.strip() != args.expected_text:
        errors.append(f"visible-text mismatch: expected {args.expected_text!r}, got {text.strip()!r}")
    if args.mode == "throughput" and decode_tps is None:
        errors.append("throughput run produced fewer than two generated tokens")
    gaps = [observations[index].offset_seconds - observations[index - 1].offset_seconds for index in range(1, len(observations))]
    return {
        "schema_version": 1,
        "sample_index": sample_index,
        "status": "passed" if not errors else "failed",
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
            "warm_ttft_seconds": first,
            "decode_seconds": decode_seconds,
            "decode_tps": decode_tps,
            "token_gap_seconds": gaps,
            "token_gap_summary_seconds": numeric_summary(gaps),
        },
        "usage": usage,
        "response": {
            "text": text,
            "generated_tokens": generated_tokens,
            "content_events": [item.__dict__ for item in observations],
        },
        "protocol_errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.artifact_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing artifact directory: {args.artifact_dir}")
    args.artifact_dir.mkdir(parents=True)
    tokenizer = load_tokenizer(args.tokenizer)
    write_json(args.artifact_dir / "manifest.json", {
        "schema_version": 1,
        "arguments": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "collection": "loopback-only client; does not start, stop, or configure a server",
    })
    if args.warmup:
        warmup = run_sample(args, tokenizer, 0)
        write_json(args.artifact_dir / "warmup.json", warmup)
        if warmup["status"] != "passed":
            raise RuntimeError("warmup failed; inspect warmup.json before scored samples")
    samples = [run_sample(args, tokenizer, index) for index in range(1, args.samples + 1)]
    for sample in samples:
        write_json(args.artifact_dir / f"sample-{sample['sample_index']:02d}.json", sample)
    tps = [sample["timing"]["decode_tps"] for sample in samples if sample["status"] == "passed" and sample["timing"]["decode_tps"] is not None]
    ttft = [sample["timing"]["warm_ttft_seconds"] for sample in samples if sample["status"] == "passed" and sample["timing"]["warm_ttft_seconds"] is not None]
    write_json(args.artifact_dir / "summary.json", {
        "schema_version": 1,
        "requested_samples": args.samples,
        "successful_samples": len(tps),
        "decode_tps": {"samples": tps, **numeric_summary(tps)},
        "warm_ttft_seconds": {"samples": ttft, **numeric_summary(ttft)},
        "failed_samples": [sample["sample_index"] for sample in samples if sample["status"] != "passed"],
    })
    return 0 if len(tps) == args.samples else 2


if __name__ == "__main__":
    raise SystemExit(main())
