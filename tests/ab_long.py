#!/usr/bin/env python3
"""Long-form TP1 vs TP2 token-identity check: full reasoning + content, 512 tokens."""
import json
import sys
import urllib.request

PROMPTS = [
    "What is 17 * 23 + 5? Show your working, then answer with just the number.",
    "Write a Python function that returns the nth Fibonacci number iteratively. Include a docstring and one example.",
    "Explain in detail how tensor parallelism works in a transformer, including what is sharded and what collectives are used.",
    "Solve step by step: a train leaves A at 9:00 going 80 km/h. Another leaves B (240 km away) at 9:30 going 100 km/h toward A. When and where do they meet?",
    "Write a haiku about GPUs, then explain your syllable count.",
]


def complete(base: str, prompt: str) -> dict:
    body = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    return {
        "prompt": prompt,
        "full": (m.get("reasoning_content") or "") + "\n===ANSWER===\n" + (m.get("content") or ""),
        "finish": d["choices"][0].get("finish_reason"),
    }


def main():
    if sys.argv[1] == "--diff":
        a = json.load(open(sys.argv[2]))
        b = json.load(open(sys.argv[3]))
        same = 0
        for ra, rb in zip(a, b):
            ok = ra["full"] == rb["full"]
            same += ok
            print(f"{'SAME ' if ok else 'DIFF '} | {ra['prompt'][:60]}")
            if not ok:
                # find first divergence
                for i, (ca, cb) in enumerate(zip(ra["full"], rb["full"])):
                    if ca != cb:
                        print(f"  first divergence at char {i}:")
                        print(f"  A: ...{ra['full'][max(0,i-80):i+120]!r}")
                        print(f"  B: ...{rb['full'][max(0,i-80):i+120]!r}")
                        break
                else:
                    print(f"  length differs: {len(ra['full'])} vs {len(rb['full'])}")
        print(f"\n{same}/{len(a)} token-identical (full reasoning+answer)")
        return
    base, tag = sys.argv[1], sys.argv[2]
    out = []
    for i, p in enumerate(PROMPTS):
        r = complete(base, p)
        print(f"[{i+1}/{len(PROMPTS)}] {len(r['full'])} chars, {r['finish']}")
        out.append(r)
    json.dump(out, open(f"{tag}_long.json", "w"), indent=1)
    print(f"wrote {tag}_long.json")


if __name__ == "__main__":
    main()
