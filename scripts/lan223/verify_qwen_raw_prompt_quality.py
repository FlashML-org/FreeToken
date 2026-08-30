#!/usr/bin/env python3
"""Capture a Qwen quality stream with one caller-rendered prompt.

This LAN-223 control intentionally avoids ``/v1/chat/completions``.  Different
servers can legitimately ship different Jinja renderers for the same GGUF, which
makes chat-token counts and output text incomparable even when their model
execution is correct.  The script renders the request once with an explicit
Hugging Face tokenizer, sends that exact string to ``/v1/completions``, and
preserves the prompt, prompt hash, server usage, timings, and emitted text.

Run it once against each isolated server.  Equal ``prompt_sha256`` values are a
hard precondition for comparing output text or decode timing between those runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

# Keep the AIME question loader shared with the existing chat quality gate so
# this raw-prompt control changes only prompt transport, not the math workload.
SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from benchmarks.bench_decode_moe import load_problem


def parse_args() -> argparse.Namespace:
    """Read every external input explicitly so the recorded artifact is reproducible."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Server origin without /v1.")
    parser.add_argument("--model", required=True, help="OpenAI model identifier sent to the server.")
    parser.add_argument("--tokenizer", required=True, help="Local HF tokenizer directory used once for rendering.")
    parser.add_argument("--artifact", required=True, type=Path, help="New JSON evidence path.")
    parser.add_argument("--aime", default=None, help="Optional local AIME JSONL source.")
    parser.add_argument("--problem", default=0, type=int, help="Zero-based AIME problem index.")
    parser.add_argument("--decode", default=128, type=int, help="Maximum generated tokens.")
    return parser.parse_args()


def render_prompt(tokenizer, problem: str) -> str:
    """Render one thinking-enabled user message into the sole server input string."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if not isinstance(prompt, str):
        raise TypeError("Qwen tokenizer returned a non-string chat prompt")
    return prompt


def stream_completion(base_url: str, model: str, prompt: str, decode: int) -> tuple[str, dict, list[float], float]:
    """Send one greedy raw completion and retain every client-visible text timestamp."""
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": decode,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    stamps: list[float] = []
    pieces: list[str] = []
    usage: dict = {}
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                text = choice.get("text")
                if text:
                    stamps.append(time.perf_counter())
                    pieces.append(text)
    return "".join(pieces), usage, stamps, started


def main() -> int:
    """Render, stream, calculate client timing, and write one self-contained artifact."""
    args = parse_args()
    problem, answer = load_problem(args.aime, args.problem)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    prompt = render_prompt(tokenizer, problem)
    text, usage, stamps, started = stream_completion(args.base_url, args.model, prompt, args.decode)
    decode_steps = max(len(stamps) - 1, 0)
    decode_seconds = stamps[-1] - stamps[0] if decode_steps else 0.0
    artifact = {
        "schema_version": 1,
        "control": "caller-rendered raw prompt via /v1/completions",
        "base_url": args.base_url,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "problem": args.problem,
        "expected_answer": answer,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_token_count_local": len(tokenizer.encode(prompt, add_special_tokens=False)),
        "usage": usage,
        "output_sha1": hashlib.sha1(text.encode("utf-8")).hexdigest()[:12],
        "text": text,
        "metrics": {
            "events": len(stamps),
            "decode_steps": decode_steps,
            "decode_seconds": decode_seconds,
            "decode_tok_s": decode_steps / decode_seconds if decode_seconds else 0.0,
            "ttft_ms": (stamps[0] - started) * 1e3 if stamps else 0.0,
        },
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
