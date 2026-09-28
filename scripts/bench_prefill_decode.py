#!/usr/bin/env python
"""并发阶梯基准：decode 与 prefill 的单流/聚合吞吐。

Business Logic（为什么需要这个函数）:
    部署钉住配置后需要量化不同并发下的单流体验与总吞吐：decode 看生成 tok/s
    （单流体验 + 聚合吞吐），prefill 看提示词处理 tok/s（TTFT 折算）。提示词
    必须逐请求唯一，否则 radix 前缀缓存会把 prefill 变成缓存命中、测不出真实
    计算速度。

Code Logic（这个函数做什么）:
    每档并发 C 发一波 C 个流式请求。decode：短提示 + --decode-tokens 生成，
    单流 = (n-1)/(t_last - t_first)，聚合 = Σtokens / 波墙钟；prefill：唯一长
    代码提示 + max_tokens=4，单流 = prompt_tokens / TTFT，聚合 =
    Σprompt_tokens / max(TTFT)。usage 缺失时按字符数近似 token 数。
"""
from __future__ import annotations

import argparse
import json
import random
import string
import time
from concurrent.futures import ThreadPoolExecutor

import httpx


def _suffix(rng: random.Random, n: int = 24) -> str:
    """请求唯一的随机尾巴（避开 radix 前缀缓存）。"""
    return "".join(rng.choices(string.ascii_letters + string.digits, k=n))


def decode_messages(rng: random.Random) -> list[dict]:
    """短提示触发纯 decode 主导的请求。"""
    return [{"role": "user", "content": (
        "Write a Python function that merges two sorted lists into one sorted list, "
        "with a short docstring and a small demo.\n"
        f"# variant {_suffix(rng)}\n"
    )}]


def prefill_messages(rng: random.Random, target_chars: int) -> list[dict]:
    """拼接唯一长代码块到近似目标字符数，再接一个短问题（TTFT 主导）。

    每个代码块都织入本请求唯一的会话标识与逐块变化的常数：任何两个请求从第一个
    块起前缀就不同，radix 前缀缓存无从命中，测得的是真实 prefill 计算速度。
    """
    session = _suffix(rng, 16)
    salt = rng.randrange(10 ** 6)
    block = ("def f_{i}_{sess}(a, b):\n"
             "    # step: normalize inputs and compute the weighted dot product\n"
             "    total = 0.0\n"
             "    for j in range(len(a)):\n"
             "        total += a[j] * b[j] + (({i} * {salt}) % 89) * 0.5\n"
             "    return total / (1.0 + abs(total))\n\n")
    parts, i = [], 0
    while sum(map(len, parts)) < target_chars:
        parts.append(block.format(i=i, sess=session, salt=salt + i))
        i += 1
    parts.append(f"\n# session {session}\n"
                 "Summarize the code above in one short sentence.")
    return [{"role": "user", "content": "".join(parts)}]


def one_request(client: httpx.Client, url: str, model: str,
                messages: list[dict], max_tokens: int) -> dict:
    """单请求流式计时：TTFT、总时长与 usage（缺失时按字符近似 token）。"""
    n_prompt_chars = len(messages[0]["content"])
    t0 = time.perf_counter()
    ttft: float | None = None
    n_chunks = 0
    usage: dict | None = None
    with client.stream("POST", url, json={
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("choices"):
                n_chunks += 1
                if ttft is None:
                    ttft = time.perf_counter() - t0
    wall = time.perf_counter() - t0
    if usage:
        prompt_tokens = usage.get("prompt_tokens", n_prompt_chars // 3)
        completion_tokens = usage.get("completion_tokens", n_chunks)
    else:
        prompt_tokens, completion_tokens = n_prompt_chars // 3, n_chunks
    return {"ttft": ttft, "wall": wall,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


def run_wave(port: int, model: str, conc: int, msgs_fn, max_tokens: int,
             seed: int) -> list[dict]:
    """一波 C 个并发请求（线程池同时发出），返回逐请求计时。"""
    rng = random.Random(seed)
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    with httpx.Client(timeout=900) as client:
        with ThreadPoolExecutor(max_workers=conc) as pool:
            futures = [pool.submit(one_request, client, url, model,
                                   msgs_fn(rng, *args), max_tokens)
                       for args in wave_args]
            return [f.result() for f in futures]


wave_args: list[tuple] = []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=1936)
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-FP8")
    ap.add_argument("--concs", default="1,2,4,8")
    ap.add_argument("--decode-tokens", type=int, default=384)
    ap.add_argument("--prefill-chars", type=int, default=13000,
                    help="长提示目标字符数（约 3.5-3.8K token）")
    ap.add_argument("--reps", type=int, default=2, help="每档并发波数（取均值）")
    ap.add_argument("--out", default="/tmp/fp8_bench.json")
    args = ap.parse_args()

    report: dict = {"decode": {}, "prefill": {}}
    for conc in [int(c) for c in args.concs.split(",")]:
        for kind in ("decode", "prefill"):
            rows = []
            for rep in range(args.reps):
                global wave_args
                if kind == "decode":
                    wave_args = [()] * conc
                    msgs_fn = decode_messages
                    max_tokens = args.decode_tokens
                else:
                    wave_args = [(args.prefill_chars,)] * conc
                    msgs_fn = prefill_messages
                    max_tokens = 4
                rows += run_wave(args.port, args.model, conc, msgs_fn,
                                 max_tokens, seed=1000 * conc + rep)
            if kind == "decode":
                per = [r["completion_tokens"] / max(r["wall"] - r["ttft"], 1e-6)
                       for r in rows if r["ttft"]]
                agg = (sum(r["completion_tokens"] for r in rows)
                       / max(max(r["wall"] for r in rows), 1e-6))
            else:
                per = [r["prompt_tokens"] / max(r["ttft"], 1e-6) for r in rows if r["ttft"]]
                agg = (sum(r["prompt_tokens"] for r in rows)
                       / max(max(r["ttft"] for r in rows), 1e-6))
            per.sort()
            report[kind][str(conc)] = {
                "n_requests": len(rows),
                "single_median": round(per[len(per) // 2], 1) if per else None,
                "single_min": round(per[0], 1) if per else None,
                "single_max": round(per[-1], 1) if per else None,
                "aggregate": round(agg, 1),
                "median_ttft_s": round(sorted(r["ttft"] for r in rows)[len(rows) // 2], 3),
            }
            print(f"[{kind} C={conc}] {report[kind][str(conc)]}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
