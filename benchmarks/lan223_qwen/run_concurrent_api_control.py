#!/usr/bin/env python3
"""Measure simultaneous LAN-223 streamed requests without changing server state.

The existing scheduler baseline measures one warm request at a time.  This
control releases a fixed number of requests together, preserves each raw
response and timing stream, and reports both individual latency and aggregate
throughput.  It is a local LAN-223 control, not a reproduction of an upstream
agent workload.  The program never starts, stops, or reconfigures a server.
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from benchmarks.lan223_qwen.run_api_benchmark import (
    load_tokenizer,
    nearest_rank_percentile,
    numeric_summary,
    stream_completion,
)


DEFAULT_PROMPT = (
    "The scheduler manages incoming inference requests by prioritizing, batching, "
    "and assigning them to available compute resources to optimize throughput and latency. "
) * 48


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse fixed workload, concurrency, and immutable artifact inputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:1919/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--expected-host", default="lan-223")
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.max_tokens < 2:
        parser.error("--max-tokens must be at least two for TPS")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def require_expected_host(expected_host: str) -> str:
    """Fail closed to keep concurrency traffic on the declared LAN-223 host."""

    actual_host = socket.gethostname().lower()
    expected_short = expected_host.lower().split(".", 1)[0]
    if actual_host not in {expected_host.lower(), expected_short}:
        raise RuntimeError(f"refusing concurrent control on {actual_host!r}; expected {expected_host!r}")
    return actual_host


def request_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the compatible greedy throughput request consumed by shared code."""

    return argparse.Namespace(
        base_url=args.base_url,
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        reasoning_effort="none",
        mode="throughput",
        expected_text="",
    )


def run_round(args: argparse.Namespace, tokenizer: Any, round_index: int) -> dict[str, Any]:
    """Release one synchronized request group and retain every request result."""

    barrier = threading.Barrier(args.concurrency)
    workload_args = request_args(args)
    suite_started = time.perf_counter()

    def one_request(request_index: int) -> dict[str, Any]:
        """Wait for the group, then record one independent streamed completion."""

        barrier.wait(timeout=30.0)
        observations, text, started, finished, errors, usage = stream_completion(workload_args)
        generated_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        ttft = observations[0].offset_seconds if observations else None
        last = observations[-1].offset_seconds if observations else None
        decode_seconds = last - ttft if ttft is not None and last is not None else None
        decode_tps = (
            (generated_tokens - 1) / decode_seconds
            if generated_tokens > 1 and decode_seconds is not None and decode_seconds > 0
            else None
        )
        gaps = [
            observations[index].offset_seconds - observations[index - 1].offset_seconds
            for index in range(1, len(observations))
        ]
        if not observations:
            errors.append("stream contained no content events")
        if decode_tps is None:
            errors.append("fewer than two generated tokens or no positive decode interval")
        return {
            "request_index": request_index,
            "started_offset_seconds": started - suite_started,
            "finished_offset_seconds": finished - suite_started,
            "wall_seconds": finished - started,
            "ttft_seconds": ttft,
            "decode_seconds": decode_seconds,
            "decode_tps": decode_tps,
            "generated_tokens": generated_tokens,
            "usage": usage,
            "response_text": text,
            "content_events": [
                {"offset_seconds": item.offset_seconds, "content": item.content}
                for item in observations
            ],
            "token_gap_seconds": gaps,
            "token_gap_summary_seconds": numeric_summary(gaps),
            "errors": errors,
            "status": "passed" if not errors else "failed",
        }

    with ThreadPoolExecutor(max_workers=args.concurrency, thread_name_prefix="lan223-load") as executor:
        requests = list(executor.map(one_request, range(1, args.concurrency + 1)))
    suite_finished = time.perf_counter()
    successful = [request for request in requests if request["status"] == "passed"]
    first_start = min((request["started_offset_seconds"] for request in requests), default=None)
    last_finish = max((request["finished_offset_seconds"] for request in requests), default=None)
    span = last_finish - first_start if first_start is not None and last_finish is not None else None
    aggregate_tokens = sum(request["generated_tokens"] for request in successful)
    return {
        "round_index": round_index,
        "started_epoch_seconds": suite_started,
        "wall_seconds": suite_finished - suite_started,
        "requests": requests,
        "summary": {
            "successful_requests": len(successful),
            "requested_requests": args.concurrency,
            "aggregate_generated_tokens": aggregate_tokens,
            "aggregate_tps": aggregate_tokens / span if span and span > 0 else None,
            "decode_tps": numeric_summary([request["decode_tps"] for request in successful if request["decode_tps"] is not None]),
            "ttft_seconds": numeric_summary([request["ttft_seconds"] for request in successful if request["ttft_seconds"] is not None]),
            "token_gap_seconds": numeric_summary([gap for request in successful for gap in request["token_gap_seconds"]]),
        },
        "status": "passed" if len(successful) == args.concurrency else "failed",
    }


def main(argv: list[str] | None = None) -> int:
    """Write the complete concurrent evidence package and propagate failures."""

    args = parse_args(sys.argv[1:] if argv is None else argv)
    host = require_expected_host(args.expected_host)
    if args.artifact.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.artifact}")
    tokenizer = load_tokenizer(args.tokenizer)
    rounds = [run_round(args, tokenizer, index) for index in range(1, args.rounds + 1)]
    aggregate_tps = [item["summary"]["aggregate_tps"] for item in rounds if item["summary"]["aggregate_tps"] is not None]
    all_ttft = [request["ttft_seconds"] for item in rounds for request in item["requests"] if request["ttft_seconds"] is not None]
    all_gaps = [gap for item in rounds for request in item["requests"] for gap in request["token_gap_seconds"]]
    artifact = {
        "schema_version": 1,
        "classification": "LAN-223 concurrent API control, not paper replication",
        "host": host,
        "request": {
            "base_url": args.base_url,
            "model": args.model,
            "concurrency": args.concurrency,
            "rounds": args.rounds,
            "max_tokens": args.max_tokens,
            "prompt": args.prompt,
            "reasoning_effort": "none",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "ignore_eos": True,
        },
        "rounds": rounds,
        "summary": {
            "successful_rounds": sum(item["status"] == "passed" for item in rounds),
            "requested_rounds": args.rounds,
            "aggregate_tps": numeric_summary(aggregate_tps),
            "ttft_seconds": numeric_summary(all_ttft),
            "token_gap_seconds": numeric_summary(all_gaps),
            "p99_ttft_seconds": nearest_rank_percentile(all_ttft, 0.99),
            "p99_token_gap_seconds": nearest_rank_percentile(all_gaps, 0.99),
        },
        "status": "passed" if all(item["status"] == "passed" for item in rounds) else "failed",
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": artifact["status"], "summary": artifact["summary"]}, sort_keys=True))
    return 0 if artifact["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
