#!/usr/bin/env python3
"""Capture a reproducible text-only Gemma4 GGUF quality and decode control."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from freetoken.utils.hf import load_tokenizer


def main() -> int:
    """Render one fixed arithmetic question, stream it once, and retain all evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--decode", type=int, default=256)
    args = parser.parse_args()

    question = "What is 17 times 19? Reply with only the decimal number."
    expected = "323"
    tokenizer = load_tokenizer(args.gguf)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
    )
    assert isinstance(prompt, str)
    body = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.decode,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "add_special_tokens": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    stamps: list[float] = []
    chunks: list[str] = []
    usage: dict = {}
    with urllib.request.urlopen(req, timeout=300) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                if text := choice.get("text"):
                    chunks.append(text)
                    stamps.append(time.perf_counter())
    text = "".join(chunks)
    steps = max(len(stamps) - 1, 0)
    duration = stamps[-1] - stamps[0] if steps else 0.0
    record = {
        "schema_version": 1,
        "control": "Gemma4 GGUF caller-rendered raw prompt",
        "question": question,
        "expected_answer": expected,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_token_count_local": len(tokenizer.encode(prompt, add_special_tokens=False)),
        "usage": usage,
        "text": text,
        "answer_present": expected in text,
        "metrics": {
            "events": len(stamps), "decode_steps": steps,
            "decode_tok_s": steps / duration if duration else 0.0,
            "ttft_ms": (stamps[0] - started) * 1000 if stamps else 0.0,
        },
    }
    args.artifact.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
