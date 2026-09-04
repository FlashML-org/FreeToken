"""Micro-bench for the GGUF fused-MoE MMVQ kernel pair (Inc 1, .plans/qwen-moe-speed).

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
from freetoken.kernel.gguf import (
    ggml_moe_a8_vec,
    ggml_moe_a8_vec_workspace,
    ggml_moe_mmvq_id,
    ggml_moe_mmvdq_id,
    ggml_moe_gate_up_swiglu_id,
)
try:
    from benchmarks.lib.paired_stats import paired_summary
except ModuleNotFoundError:  # direct ``python benchmarks/bench_gguf_moe_kernels.py``
    from lib.paired_stats import paired_summary

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
    p.add_argument(
        "--trace",
        default=None,
        help="export a profiler trace and print quantize/MMVQ/activation/reduce stages",
    )
    p.add_argument("--impl", default=os.environ.get("FREETOKEN_GGUF_MOE_IMPL", "legacy"))
    p.add_argument("--down-type", choices=("q8_0", "q6_k"), default="q8_0")
    p.add_argument("--fuse-gate-up", action="store_true")
    p.add_argument("--dump", default=None, help="save outputs to this path")
    p.add_argument("--check", default=None, help="byte-compare outputs against a saved dump")
    p.add_argument(
        "--paired", action="store_true",
        help="interleave legacy and selected candidate, then emit paired statistics",
    )
    p.add_argument("--json", dest="json_out", default=None, help="write result artifact")
    args = p.parse_args(argv)
    if args.impl not in {"legacy", "auto", "gfx1100", "rdna3_mmid", "rdna3_mmvdq"}:
        p.error("--impl must be legacy, auto, gfx1100, rdna3_mmid, or rdna3_mmvdq")
    if args.impl == "rdna3_mmvdq" and args.down_type != "q6_k":
        p.error("rdna3_mmvdq benchmark requires --down-type q6_k")
    if args.fuse_gate_up and args.impl != "rdna3_mmid":
        p.error("--fuse-gate-up requires --impl rdna3_mmid")
    if args.paired and args.impl in {"legacy", "auto"}:
        p.error("--paired requires an explicit candidate --impl")
    if args.paired and args.fuse_gate_up:
        p.error("--paired does not support --fuse-gate-up; compare same unfused pair")
    os.environ["FREETOKEN_GGUF_MOE_IMPL"] = args.impl

    H, I, topk, slots = args.hidden, args.moe_ic, args.topk, args.slots
    n2 = 2 * I
    dev = torch.device("cuda")
    gen = torch.Generator(device="cpu").manual_seed(0)

    # gate_up bank: [slots, 2I, row_bytes(H, Q4_K)] (row bytes = H/256*144);
    # down bank: [slots, H, row_bytes(I, Q8_0)] (I/32*34) — matches moe/fused_gguf.py.
    gu_blocks = torch.zeros((slots, n2, H // 256, 144), dtype=torch.uint8, device=dev)
    gu_blocks[..., :4] = torch.tensor([128, 63, 128, 63], dtype=torch.uint8, device=dev)
    # Keep synthetic Q4_K metadata/quants bounded.  Arbitrary metadata bytes can
    # produce BF16 intermediates large enough to overflow Q8_1's half scale and
    # make a byte-comparison gate observe NaN payload differences.
    gu_blocks[..., 4:16] = 1
    gu_blocks[..., 16:] = 1
    bank_gu = gu_blocks.reshape(slots, n2, H // 256 * 144)
    if args.down_type == "q8_0":
        down_blocks = torch.zeros((slots, H, I // 32, 34), dtype=torch.uint8, device=dev)
        down_blocks[..., :2] = torch.tensor([128, 63], dtype=torch.uint8, device=dev)
        down_blocks[..., 2:] = torch.randint(
            0, 255, (slots, H, I // 32, 32), dtype=torch.uint8, generator=gen
        ).to(dev)
        down_type = GGML_Q8_0
    else:
        down_blocks = torch.randint(
            0, 255, (slots, H, I // 256, 210), dtype=torch.uint8, generator=gen
        ).to(dev)
        down_blocks[..., -2:] = torch.tensor([128, 63], dtype=torch.uint8, device=dev)
        down_type = 14
    down_row_bytes = I // 32 * 34 if args.down_type == "q8_0" else I // 256 * 210
    bank_down = down_blocks.reshape(slots, H, down_row_bytes)
    x = torch.randn(1, H, generator=gen).to(dev).to(torch.bfloat16)
    ids = torch.randint(0, slots, (1, topk), generator=gen).to(dev).int()
    weights = torch.rand(1, topk, generator=gen).to(dev)

    def q8_1_shape(tokens: int, cols: int) -> tuple[int, int]:
        padded = (cols + 512 - 1) // 512 * 512
        return tokens, padded // 32 * 9

    # Match serving path: both variants reuse fixed graph-address output and
    # Q8_1 scratch buffers instead of timing allocator churn.
    gate_output = torch.empty((topk, n2), dtype=x.dtype, device=dev)
    gate_quant_x = torch.empty(q8_1_shape(1, H), dtype=torch.int32, device=dev)
    down_output = torch.empty((topk, H), dtype=x.dtype, device=dev)
    down_quant_x = torch.empty(q8_1_shape(topk, I), dtype=torch.int32, device=dev)

    def run(impl: str | None = None) -> torch.Tensor:
        impl = impl or args.impl
        if args.fuse_gate_up and impl == args.impl:
            inter = ggml_moe_gate_up_swiglu_id(
                x, bank_gu, ids, topk, I, 1,
                int(bank_gu.stride(0)), int(bank_gu.stride(1)), "slot",
            )
        elif impl == "rdna3_mmvdq":
            gate_up = ggml_moe_mmvdq_id(
                x, bank_gu, ids, topk, GGML_Q4_K, n2, 1,
                int(bank_gu.stride(0)), int(bank_gu.stride(1)), "slot",
            )
        elif impl in {"rdna3_mmid", "gfx1100"}:
            gate_up = ggml_moe_mmvq_id(
                x, bank_gu, ids, topk, GGML_Q4_K, n2, 1,
                int(bank_gu.stride(0)), int(bank_gu.stride(1)), "slot",
                gate_output, gate_quant_x,
            )
        else:
            gate_up = ggml_moe_a8_vec_workspace(
                x, bank_gu, ids, topk, GGML_Q4_K, n2, 1,
                gate_output, gate_quant_x,
            )
        if not (args.fuse_gate_up and impl == args.impl):
            with torch.profiler.record_function("moe_activation"):
                inter = silu_and_mul(gate_up.reshape(1 * topk, n2))
        route_ids = ids.reshape(-1, 1) if impl in {"rdna3_mmid", "rdna3_mmvdq", "gfx1100"} else ids
        if impl == "rdna3_mmvdq":
            down = ggml_moe_mmvdq_id(
                inter, bank_down, route_ids, 1, down_type, H, topk,
                int(bank_down.stride(0)), int(bank_down.stride(1)), "slot",
            )
        elif impl in {"rdna3_mmid", "gfx1100"}:
            down = ggml_moe_mmvq_id(
                inter, bank_down, route_ids, 1, down_type, H, topk,
                int(bank_down.stride(0)), int(bank_down.stride(1)), "slot",
                down_output, down_quant_x,
            )
        else:
            down = ggml_moe_a8_vec_workspace(
                inter, bank_down, ids, 1, down_type, H, topk,
                down_output, down_quant_x,
            )
        with torch.profiler.record_function("moe_route_reduce"):
            return (down.reshape(1, topk, H) * weights.reshape(1, topk, 1)).sum(dim=1)

    # Warm + reference for the bytes-equality gate (run() is deterministic for a
    # fixed seed/config: no atomics, one CTA owns every output row).
    if args.paired:
        os.environ["FREETOKEN_GGUF_MOE_IMPL"] = "legacy"
        for _ in range(args.warmup):
            run("legacy")
        os.environ["FREETOKEN_GGUF_MOE_IMPL"] = args.impl
        for _ in range(args.warmup):
            run(args.impl)
        legacy_samples: list[float] = []
        candidate_samples: list[float] = []
        for _ in range(args.iters):
            os.environ["FREETOKEN_GGUF_MOE_IMPL"] = "legacy"
            legacy_samples.append(bench_once(lambda: run("legacy")))
            os.environ["FREETOKEN_GGUF_MOE_IMPL"] = args.impl
            candidate_samples.append(bench_once(lambda: run(args.impl)))
        paired = paired_summary(legacy_samples, candidate_samples)
        os.environ["FREETOKEN_GGUF_MOE_IMPL"] = args.impl
        out = run(args.impl)
    else:
        out = run()
        torch.cuda.synchronize()
        us_pair = bench(run, args.iters, args.warmup)
        paired = None
    torch.cuda.synchronize()
    stages = profile_stages(run, args.trace, args.warmup) if args.trace else {}

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

    result = {
        "schema": "freetoken-gguf-moe-microbench-v2",
        "impl": args.impl,
        "mmv_y": mmv_y,
        "slots": slots,
        "hidden": H,
        "moe_ic": I,
        "topk": topk,
        "down_type": args.down_type,
        "iters": args.iters,
        "warmup": args.warmup,
        "output_sha256": sha,
        "paired": paired,
    }
    if paired is None:
        result["median_us_layer_pair"] = us_pair
        print(
            f"impl={args.impl} MMV_Y={mmv_y} slots={slots} H={H} I={I} topk={topk} : "
            f"{us_pair:.1f} us / layer-pair  out_sha={sha}{note}"
        )
    else:
        print(
            f"legacy_vs_{args.impl} pairs={paired['pairs']} "
            f"legacy={paired['median_legacy_us']:.1f} us candidate={paired['median_candidate_us']:.1f} us "
            f"recovery={paired['median_recovery_us']:.1f} us "
            f"CI=[{paired['recovery_p02_5_us']:.1f},{paired['recovery_p97_5_us']:.1f}] us out_sha={sha}{note}"
        )
    if stages:
        print("  profiler stages: " + ", ".join(f"{k}={v:.2f} us" for k, v in stages.items()))
    if args.json_out:
        with open(args.json_out, "w") as handle:
            import json
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0


def bench_once(fn) -> float:
    """Time one already-warmed device call with a fresh event pair."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3


def bench(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        ts.append(start.elapsed_time(end))
    return statistics.median(ts) * 1e3


def profile_stages(fn, trace_path: str, warmup: int) -> dict[str, float]:
    """Profile actual device kernels; never infer stages from one fused-call wall timer."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if hasattr(torch.profiler.ProfilerActivity, "CUDA"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True, acc_events=True) as prof:
        for _ in range(max(1, warmup // 2)):
            fn()
        torch.cuda.synchronize()
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(trace_path)
    stage_events: dict[str, list[float]] = {
        "q8_quantize": [],
        "mmvq": [],
        "activation": [],
        "route_reduce": [],
    }
    def device_us(event) -> float:
        value = getattr(event, "device_time_total", None)
        if value is None:
            value = getattr(event, "self_device_time_total", 0.0)
        return float(value)

    for event in prof.key_averages():
        name = event.key.lower()
        if "moe_activation" in name:
            stage_events["activation"].append(event.self_cpu_time_total / max(event.count, 1))
        elif "moe_route_reduce" in name:
            stage_events["route_reduce"].append(event.self_cpu_time_total / max(event.count, 1))
        elif "quantize_q8_1" in name:
            stage_events["q8_quantize"].append(device_us(event) / max(event.count, 1))
        elif "moe_vec" in name:
            stage_events["mmvq"].append(device_us(event) / max(event.count, 1))
    # Profiler time units are microseconds. Multiple symbols can be emitted for one
    # stage, so report their sum and retain the trace for exact attribution.
    return {name: sum(values) for name, values in stage_events.items() if values}


if __name__ == "__main__":
    raise SystemExit(main())
