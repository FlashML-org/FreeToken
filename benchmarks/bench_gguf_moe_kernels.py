"""Micro-bench for the GGUF fused-MoE MMVQ kernel pair (Inc 4, .plans/rocm-perf-parity).

Times the two ``ggml_moe_a8_vec`` calls the decode hot loop makes per MoE layer at
the Qwen3.6-35B-A3B shapes (H=2048, I=512, top_k=8; Q4_K gate_up + Q8_0 down) on a
stacked-expert bank with ``slots`` resident experts, exactly like the serving path
(``moe/fused_gguf.py``).

Cross-variant numerics gate: run once at the incumbent config with ``--dump PATH``,
then at any other ``FREETOKEN_GGUF_MMV_Y`` build with ``--check PATH``; the bench
hard-fails if a single output byte differs.

Usage (each MMV_Y value triggers its own JIT build, ~1-2 min the first time):
    PYTHONPATH=python .venv-rocm/bin/python benchmarks/bench_gguf_moe_kernels.py \
        --iters 300 --dump /tmp/moe-ref.pt
    FREETOKEN_GGUF_MMV_Y=8 PYTHONPATH=python .venv-rocm/bin/python \
        benchmarks/bench_gguf_moe_kernels.py --iters 300 --check /tmp/moe-ref.pt
"""

from __future__ import annotations

import argparse
import hashlib
import os
import statistics
import sys
import time

import torch
from freetoken.layers.activation import silu_and_mul
from freetoken.kernel.gguf import ggml_moe_a8_vec

GGML_Q4_K = 12
GGML_Q8_0 = 8


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slots", type=int, default=5873, help="resident expert slot count")
    p.add_argument("--hidden", type=int, default=2048, help="model hidden size H")
    p.add_argument("--moe-ic", type=int, default=512, help="moe_intermediate_size")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--dump", default=None, help="save outputs to this path")
    p.add_argument("--check", default=None, help="byte-compare outputs against a saved dump")
    args = p.parse_args(argv)

    H, I, topk, slots = args.hidden, args.moe_ic, args.topk, args.slots
    n2 = 2 * I
    dev = torch.device("cuda")
    gen = torch.Generator(device="cpu").manual_seed(0)

    # gate_up bank: [slots, 2I, row_bytes(H, Q4_K)] (row bytes = H/256*144);
    # down bank: [slots, H, row_bytes(I, Q8_0)] (I/32*34) — matches moe/fused_gguf.py.
    bank_gu = torch.randint(
        0, 255, (slots, n2, H // 256 * 144), dtype=torch.uint8, generator=gen
    ).to(dev)
    bank_down = torch.randint(
        0, 255, (slots, H, I // 32 * 34), dtype=torch.uint8, generator=gen
    ).to(dev)
    x = torch.randn(1, H, generator=gen).to(dev).to(torch.bfloat16)
    ids = torch.randint(0, slots, (1, topk), generator=gen).to(dev).int()

    def run() -> torch.Tensor:
        gate_up = ggml_moe_a8_vec(x, bank_gu, ids, topk, GGML_Q4_K, n2, 1)
        inter = silu_and_mul(gate_up.reshape(1 * topk, n2))
        return ggml_moe_a8_vec(inter, bank_down, ids, 1, GGML_Q8_0, H, topk)

    # Warm + reference for the bytes-equality gate (run() is deterministic for a
    # fixed seed/config: no atomics, one CTA owns every output row).
    out = run()
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    us_pair = bench(run, args.iters)

    sha = hashlib.sha256(out.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]
    mmv_y = os.environ.get("FREETOKEN_GGUF_MMV_Y", "1")
    note = ""
    if args.dump:
        torch.save({"out": out.cpu()}, args.dump)
        note = f"  (saved reference to {args.dump})"
    if args.check:
        saved = torch.load(args.check, map_location="cpu", weights_only=True)
        if not torch.equal(saved["out"], out.cpu()):
            print(f"FAIL: MoE output bytes differ from the saved reference {args.check}", file=sys.stderr)
            return 1
        note = f"  bytes_equal_to_saved=True"

    print(
        f"MMV_Y={mmv_y} slots={slots} H={H} I={I} topk={topk} : "
        f"{us_pair:.1f} us / layer-pair  out_sha={sha}{note}"
    )
    return 0


def bench(fn, iters: int) -> float:
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6


if __name__ == "__main__":
    raise SystemExit(main())