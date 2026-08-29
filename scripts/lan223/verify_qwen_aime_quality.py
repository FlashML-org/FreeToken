#!/usr/bin/env python3
"""Verify LAN-223 Qwen output stability with the historical AIME-25 workload.

The benchmark uses the same question, greedy sampling, thinking-enabled template,
and forced 128-token decode that exposed the rejected HIP router candidate. It
targets an already-running loopback server and never starts, stops, or changes it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from benchmarks.bench_decode_moe import load_problem, resolve_sampling, stream_generate


REFERENCE_SHA1 = "0acef4eab6f4"


def parse_args() -> argparse.Namespace:
    """Read explicit server, checkpoint, and artifact inputs for one quality gate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:1919")
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--aime", default=None)
    parser.add_argument("--problem", type=int, default=0)
    parser.add_argument("--decode", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    """Warm the live server, score one deterministic stream, and persist raw evidence."""

    args = parse_args()
    problem, answer = load_problem(args.aime, args.problem)
    sampling, sampling_source = resolve_sampling(args.model, greedy=True)
    with urllib.request.urlopen(args.base_url.rstrip("/") + "/v1/models", timeout=10) as response:
        model_id = json.load(response)["data"][0]["id"]
    stream_args = SimpleNamespace(decode=args.decode)
    # Send one complete request first so the measured request observes a populated
    # expert cache instead of one-time load and scheduling work.
    stream_generate(args.base_url, model_id, problem, sampling, stream_args)
    # Capture the quality-gated request itself.  stream_generate records a
    # monotonic timestamp for every non-empty streamed text event.
    result = stream_generate(args.base_url, model_id, problem, sampling, stream_args)
    text = result["text"]
    output_sha1 = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    # The first event includes warm prompt processing.  The intervals after it
    # describe steady-state decode, which makes this directly comparable to the
    # historical AIME benchmark and avoids reporting prompt work as token rate.
    stamps = result["stamps"]
    completion_tokens = result["usage"]["completion_tokens"]
    decode_steps = max(completion_tokens - 1, 0)
    decode_seconds = stamps[-1] - stamps[0] if len(stamps) >= 2 else 0.0
    gaps_ms = sorted((later - earlier) * 1e3 for earlier, later in zip(stamps, stamps[1:]))
    metrics = {
        "decode_steps": decode_steps,
        "decode_seconds": decode_seconds,
        "decode_tok_s": decode_steps / decode_seconds if decode_seconds > 0 else 0.0,
        "ms_per_token": decode_seconds * 1e3 / decode_steps if decode_steps > 0 else 0.0,
        "event_ms_p50": gaps_ms[len(gaps_ms) // 2] if gaps_ms else 0.0,
        "event_ms_p99": gaps_ms[min(len(gaps_ms) - 1, int(len(gaps_ms) * 0.99))] if gaps_ms else 0.0,
        "ttft_ms": (stamps[0] - result["t0"]) * 1e3 if stamps else 0.0,
        "events": len(stamps),
    }
    artifact = {
        "schema_version": 1,
        "model_id": model_id,
        "problem": args.problem,
        "expected_answer": answer,
        "sampling": sampling,
        "sampling_source": sampling_source,
        "prompt_tokens": result["usage"]["prompt_tokens"],
        "completion_tokens": completion_tokens,
        "metrics": metrics,
        "expected_output_sha1": REFERENCE_SHA1,
        "output_sha1": output_sha1,
        "status": "passed" if output_sha1 == REFERENCE_SHA1 else "failed",
        "text": text,
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
