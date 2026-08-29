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
    stream_generate(args.base_url, model_id, problem, sampling, stream_args)
    result = stream_generate(args.base_url, model_id, problem, sampling, stream_args)
    text = result["text"]
    output_sha1 = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    artifact = {
        "schema_version": 1,
        "model_id": model_id,
        "problem": args.problem,
        "expected_answer": answer,
        "sampling": sampling,
        "sampling_source": sampling_source,
        "prompt_tokens": result["usage"]["prompt_tokens"],
        "completion_tokens": result["usage"]["completion_tokens"],
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
