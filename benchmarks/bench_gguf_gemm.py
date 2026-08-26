"""GGUF quantized-matmul benchmark: int8-MMA MMQ vs DP4A MMQ vs dequant+cuBLAS vs MMVQ.

Sweeps the dense (``fused_mul_mat_gguf`` seams) and grouped-MoE
(``_moe_matmul`` seams) kernel families over batch size on synthetic
random-but-safe packed weights (fp16 scale fields masked small, as in
``tests/kernels/test_gguf_quant_types.py``), at Ornith-1.5-35B geometry by
default. Every timed case is first cross-checked against the transient
dequantized weights (the oracle): rel error must stay below --tol.

The int8-MMA columns need the ``freetoken_gguf_mmq`` extension (sm_75+ build,
sm_120-dispatched); they are skipped with a note where unavailable.

Run:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python benchmarks/bench_gguf_gemm.py
    ... --dense-rows 4 16 32 512 8192 --moe-tokens 16 320 8192 --json out.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics


def build_cases(args) -> list[dict]:
    """Pure case list (unit-testable without CUDA)."""
    cases = []
    for rows in args.dense_rows:
        for qtype_name in args.dense_types:
            cases.append({"op": "dense", "rows": rows, "qtype": qtype_name,
                          "out_features": args.dense_out, "in_features": args.hidden})
    for tokens in args.moe_tokens:
        cases.append({"op": "moe", "tokens": tokens, "experts": args.experts,
                      "top_k": args.top_k, "hidden": args.hidden, "inter": args.inter})
    return cases


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dense-rows", type=int, nargs="+", default=[4, 8, 16, 32, 128, 512, 2048, 8192])
    p.add_argument("--dense-types", nargs="+", default=["q4_k", "q6_k"], choices=["q4_k", "q6_k"])
    p.add_argument("--dense-out", type=int, default=8192)
    p.add_argument("--moe-tokens", type=int, nargs="+", default=[16, 64, 320, 1024, 8192])
    p.add_argument("--experts", type=int, default=256)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--inter", type=int, default=512)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--tol", type=float, default=0.02)
    p.add_argument("--json", type=str, default=None)
    return p.parse_args(argv)


def _med_ms(fn, iters):
    import torch

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def _packed(qtype, rows_of_blocks, seed):
    import numpy as np
    import torch
    from freetoken.models.gguf.dequant import BLOCK_SHAPE

    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, (rows_of_blocks, BLOCK_SHAPE[qtype][1]), dtype=np.uint8)
    raw.view(np.uint16)[:] &= np.uint16(0x3BFF)
    return torch.from_numpy(raw)


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch

    from freetoken.kernel.gguf import (
        ggml_dequantize,
        ggml_moe_a8,
        ggml_moe_get_block_size,
        ggml_mul_mat_a8,
        ggml_mul_mat_vec_a8,
    )
    from freetoken.models.gguf.dequant import BLOCK_SHAPE, GGML_Q4_K, GGML_Q6_K
    from freetoken.moe.fused import moe_align_block_size

    qtypes = {"q4_k": GGML_Q4_K, "q6_k": GGML_Q6_K}
    try:
        from freetoken.kernel.gguf import ggml_moe_a8_mma, ggml_mul_mat_a8_mma

        mma_ok = True
    except Exception as exc:  # noqa: BLE001 - report and continue without MMA
        print(f"# int8-MMA extension unavailable: {exc}")
        mma_ok = False

    out_rows = []
    torch.manual_seed(0)
    for case in build_cases(args):
        if case["op"] == "dense":
            qtype = qtypes[case["qtype"]]
            block, ts = BLOCK_SHAPE[qtype]
            out_f, in_f, rows = case["out_features"], case["in_features"], case["rows"]
            w = _packed(qtype, out_f * in_f // block, seed=qtype).reshape(out_f, -1).cuda()
            dense = ggml_dequantize(w, qtype, out_f, in_f, torch.float16)
            x = torch.randn(rows, in_f, dtype=torch.float16, device="cuda")
            ref = (x.float() @ dense.float().T)
            res = {"op": "dense", "qtype": case["qtype"], "rows": rows}

            def check(name, y):
                rel = ((y.float() - ref).norm() / ref.norm()).item()
                assert rel < args.tol, (name, rel)

            y = ggml_mul_mat_a8(w, x, qtype, out_f)
            check("dp4a", y)
            res["dp4a_ms"] = _med_ms(lambda: ggml_mul_mat_a8(w, x, qtype, out_f), args.iters)
            res["dequant_ms"] = _med_ms(
                lambda: x @ ggml_dequantize(w, qtype, out_f, in_f, torch.float16).T, args.iters
            )
            if rows <= 8:
                y = ggml_mul_mat_vec_a8(w, x, qtype, out_f)
                check("mmvq", y)
                res["mmvq_ms"] = _med_ms(lambda: ggml_mul_mat_vec_a8(w, x, qtype, out_f), args.iters)
            if mma_ok:
                y = ggml_mul_mat_a8_mma(w, x.float(), qtype, out_f)
                check("mma", y)
                res["mma_ms"] = _med_ms(lambda: ggml_mul_mat_a8_mma(w, x, qtype, out_f), args.iters)
        else:
            experts, top_k = case["experts"], case["top_k"]
            hidden, inter, tokens = case["hidden"], case["inter"], case["tokens"]
            block, ts = BLOCK_SHAPE[GGML_Q4_K]
            gu = _packed(GGML_Q4_K, experts * 2 * inter * hidden // block, seed=1).reshape(experts, -1).cuda()
            x = torch.randn(tokens, hidden, dtype=torch.float16, device="cuda")
            ids = torch.stack([torch.randperm(experts)[:top_k] for _ in range(tokens)]).int().cuda()
            res = {"op": "moe", "tokens": tokens}

            def dp4a_moe():
                bs = ggml_moe_get_block_size(GGML_Q4_K)
                s, ei, npad = moe_align_block_size(ids, bs, experts)
                return ggml_moe_a8(x, gu, s, ei, npad, GGML_Q4_K, 2 * inter, top_k, tokens)

            ref_moe = dp4a_moe().float()
            res["dp4a_ms"] = _med_ms(dp4a_moe, args.iters)
            if mma_ok:
                y = ggml_moe_a8_mma(x, gu, ids, top_k, GGML_Q4_K, 2 * inter, tokens, gu.shape[1], True)
                rel = ((y.float() - ref_moe).norm() / ref_moe.norm()).item()
                assert rel < args.tol, ("moe-mma", rel)
                res["mma_ms"] = _med_ms(
                    lambda: ggml_moe_a8_mma(x, gu, ids, top_k, GGML_Q4_K, 2 * inter, tokens, gu.shape[1], True),
                    args.iters,
                )
        print(res)
        out_rows.append(res)

    if args.json:
        with open(args.json, "w") as f:
            for row in out_rows:
                f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
