#!/usr/bin/env python3
"""Deterministic A/B prompt suite for the FreeToken TP1 vs TP2 correctness check.

Usage:  python ab_suite.py <base_url> <tag>
Sends 10 temperature-0 prompts and writes <tag>.json with the raw completions.
Compare two runs with:  python ab_suite.py --diff tp1.json tp2.json
"""
import json
import sys
import time
import urllib.request

PROMPTS = [
    "What is 17 * 23 + 5? Answer with just the number.",
    "Capital of France in one word.",
    "Write a Python function that returns the nth Fibonacci number iteratively.",
    "Explain tensor parallelism in two sentences.",
    "List the first 5 prime numbers, comma separated.",
    "Translate to German: The quick brown fox jumps over the lazy dog.",
    "Why is the sky blue? Three sentences max.",
    "Solve: if 3x + 7 = 22, what is x? Just the number.",
    "Write a haiku about GPUs.",
    "What does HTTP 418 mean? One sentence.",
]


def complete(base: str, prompt: str) -> dict:
    body = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 256,
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return {
        "prompt": prompt,
        "content": d["choices"][0]["message"]["content"],
        "reasoning": (d["choices"][0]["message"].get("reasoning_content") or "")[:200],
        "finish": d["choices"][0].get("finish_reason"),
        "usage": d.get("usage"),
        "wall_s": round(time.time() - t0, 2),
    }


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--diff":
        a = json.load(open(sys.argv[2]))
        b = json.load(open(sys.argv[3]))
        same = 0
        for ra, rb in zip(a, b):
            ok = ra["content"].strip() == rb["content"].strip()
            same += ok
            print(f"{'SAME ' if ok else 'DIFF '} | {ra['prompt'][:60]}")
            if not ok:
                print(f"  A: {ra['content'][:300]!r}")
                print(f"  B: {rb['content'][:300]!r}")
        print(f"\n{same}/{len(a)} identical")
        return
    base, tag = sys.argv[1], sys.argv[2]
    out = []
    for i, p in enumerate(PROMPTS):
        r = complete(base, p)
        print(f"[{i+1}/{len(PROMPTS)}] {p[:50]!r} -> {r['wall_s']}s {r['finish']}")
        out.append(r)
    json.dump(out, open(f"{tag}.json", "w"), indent=1)
    print(f"wrote {tag}.json")


if __name__ == "__main__":
    main()
