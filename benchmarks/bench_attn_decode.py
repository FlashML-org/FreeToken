"""Micro-bench for the triton decode paged-attention kernel (Inc 5, rocm-perf-parity).

Times `freetoken.kernel.triton.attention.decode_paged_attention` at the served
Qwen3.6-35B-A3B decode shape (bs=1, 32 q-heads x 256 dim GQA over 2 kv-heads,
~8k paged KV) and byte-compares each swept (BLOCK_N, num_warps) build variant
against the incumbent config via --dump/--check.

Usage:
    PYTHONPATH=python .venv-rocm/bin/python benchmarks/bench_attn_decode.py \
        --kv 8192 --dump /tmp/attn-ref.pt           # incumbent config
    # after editing the call-site constexprs:
    PYTHONPATH=python .venv-rocm/bin/python benchmarks/bench_attn_decode.py \
        --kv 8192 --check /tmp/attn-ref.pt
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import sys
import time

import torch
from freetoken.kernel.triton.attention import (
    _MIN_BLOCK_KV,
    decode_paged_attention,
)

MAX_KV_SPLITS = 16


def build(kv_len: int, num_q_heads: int = 32, num_kv_heads: int = 2, head_dim: int = 256):
    dev = torch.device("cuda")
    gen = torch.Generator(device="cpu").manual_seed(11)
    q = torch.randn(1, num_q_heads, head_dim, generator=gen).to(dev).to(torch.bfloat16) * 0.3
    # paged KV of kv_len physical slots; identity page mapping
    k_cache = torch.randn(kv_len, num_kv_heads, head_dim, generator=gen).to(dev).to(torch.bfloat16) * 0.3
    v_cache = torch.randn(kv_len, num_kv_heads, head_dim, generator=gen).to(dev).to(torch.bfloat16) * 0.3
    indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=dev)
    indices = torch.arange(kv_len, dtype=torch.int32, device=dev)
    q_positions = torch.tensor([kv_len - 1], dtype=torch.int32, device=dev)
    num_kv_splits = torch.ones(1, dtype=torch.int32, device=dev)
    logits = torch.zeros(1, num_q_heads, MAX_KV_SPLITS, head_dim, dtype=torch.float32, device=dev)
    lse = torch.zeros(1, num_q_heads, MAX_KV_SPLITS, dtype=torch.float32, device=dev)
    return q, k_cache, v_cache, indptr, indices, q_positions, logits, lse, num_kv_splits


def run(q, k, v, indptr, indices, q_pos, logits, lse, nks, sm_scale):
    return decode_paged_attention(
        q=q,
        k_cache=k,
        v_cache=v,
        indptr=indptr,
        indices=indices,
        q_positions=q_pos,
        attn_logits=logits,
        attn_lse=lse,
        num_kv_splits=nks,
        max_kv_splits=MAX_KV_SPLITS,
        sm_scale=sm_scale,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kv", type=int, default=8192)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--dump", default=None)
    p.add_argument("--check", default=None)
    args = p.parse_args(argv)

    q, k, v, indptr, indices, q_pos, logits, lse, nks = build(args.kv)
    sm_scale = 256**-0.5

    out = run(q, k, v, indptr, indices, q_pos, logits, lse, nks, sm_scale)
    torch.cuda.synchronize()

    def bench() -> float:
        for _ in range(args.warmup):
            run(q, k, v, indptr, indices, q_pos, logits, lse, nks, sm_scale)
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            run(q, k, v, indptr, indices, q_pos, logits, lse, nks, sm_scale)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts) * 1e6

    us = bench()
    sha = hashlib.sha256(out.float().cpu().numpy().tobytes()).hexdigest()[:16]
    note = ""
    if args.dump:
        torch.save({"out": out.float().cpu()}, args.dump)
        note = f"  (saved reference to {args.dump})"
    if args.check:
        saved = torch.load(args.check, map_location="cpu", weights_only=True)
        diff = (saved["out"] - out.float().cpu()).abs().max().item()
        # Different (BLOCK_N, warps) change the fp32 reduction order -> not bit-equal;
        # the correctness gate is max-abs-diff in bf16-output terms (printed, not asserted).
        print(f"max_abs_diff vs saved reference: {diff:.3e}")
        note = f"  max_abs_diff={diff:.3e}"
    print(
        f"kv={args.kv} q_heads=32 kv_heads=2 dim=256 : {us:.1f} us / call  "
        f"out_sha={sha}{note}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


_ = _MIN_BLOCK_KV  # re-exported constant; keeps the import meaningful for type checkers