#!/usr/bin/env python3
"""Screen Qwen-shaped NVFP4 Marlin-style decode GEMV launch configurations.

The live Qwen3.6 MoE uses an eight-route decode with a gate/up projection of
``[1024, 2048]`` and a down projection of ``[2048, 512]``.  This helper calls
the production Triton kernel directly with deterministic, layout-correct NVFP4
banks, then reports HIP-event latency and a raw BF16 output SHA-1.  It is only
a bounded kernel screen. A matching hash is required before, but never replaces,
the full API quality gate after a server launch configuration changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics

import torch
import triton
import triton.language as tl


def parse_args() -> argparse.Namespace:
    """Accept only the small launch-config range relevant to the decode kernel."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", choices=("gate-up", "down"), required=True)
    parser.add_argument("--block-n", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--block-kw", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--warps", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def time_one(operation) -> float:
    """Return one synchronized native HIP event duration in milliseconds."""

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def main() -> int:
    """Construct the Qwen decode layout, execute the chosen kernel, and print JSON."""

    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("native HIP/CUDA device is required")
    from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view
    from freetoken.kernel.triton.nvfp4_fused_moe import _decode_nvfp4_marlin_kernel, _e2m1_lut

    torch.manual_seed(223_2048_512)
    device = torch.device("cuda")
    routes, slots = 8, 8
    if args.projection == "gate-up":
        n, k, a_rows, route_rows, routed_weight = 1024, 2048, 1, False, False
    else:
        n, k, a_rows, route_rows, routed_weight = 2048, 512, routes, True, True

    # NVFP4 stores eight four-bit codes in each int32 word and one e4m3 scale
    # for every sixteen K elements. This precisely matches the production bank
    # layout while remaining small enough to coexist with the live API process.
    activation = torch.randn(a_rows, k, device=device, dtype=torch.bfloat16) / 4
    packed = torch.randint(-(2**31), 2**31 - 1, (slots, n, k // 8), device=device, dtype=torch.int32)
    scale = (torch.rand(slots, n, k // 16, device=device) + 0.25).to(torch.float8_e4m3fn)
    global_scale = torch.full((slots, n), 0.125, device=device, dtype=torch.float16)
    output = torch.empty((1, routes, n), device=device, dtype=torch.bfloat16)
    topk_weights = torch.linspace(0.25, 1.0, routes, device=device, dtype=torch.float32).reshape(1, routes)
    topk_ids = torch.arange(routes, device=device, dtype=torch.int32).reshape(1, routes)
    packed_i32 = packed.contiguous()
    scale_kernel = e4m3_kernel_view(scale)

    def operation() -> torch.Tensor:
        """Issue the exact route-by-output-tile decode dispatch under test."""

        grid = (routes, triton.cdiv(n, args.block_n))
        _decode_nvfp4_marlin_kernel[grid](
            activation, packed_i32, scale_kernel, global_scale, output, topk_weights, topk_ids,
            _e2m1_lut(device.index), routes, n, k,
            activation.stride(0), activation.stride(1),
            packed_i32.stride(0), packed_i32.stride(1), packed_i32.stride(2),
            scale_kernel.stride(0), scale_kernel.stride(1), scale_kernel.stride(2),
            global_scale.stride(0), global_scale.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            topk_weights.stride(0), topk_weights.stride(1),
            topk_ids.stride(0), topk_ids.stride(1),
            BLOCK_SIZE_N=args.block_n, BLOCK_SIZE_KW=args.block_kw,
            TOP_K=routes, A_ROW_IS_ROUTE=route_rows,
            MUL_ROUTED_WEIGHT=routed_weight, compute_type=tl.bfloat16,
            num_warps=args.warps,
        )
        return output

    for _ in range(args.warmup):
        operation()
    torch.cuda.synchronize()
    samples_ms = [time_one(operation) for _ in range(args.iterations)]
    result = operation()
    torch.cuda.synchronize()
    digest = hashlib.sha1(result.view(torch.uint16).cpu().numpy().tobytes()).hexdigest()
    print(json.dumps({
        "schema_version": 1,
        "projection": args.projection,
        "shape": [n, k],
        "routes": routes,
        "block_n": args.block_n,
        "block_kw": args.block_kw,
        "warps": args.warps,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "result_sha1": digest,
        "latency_ms_median": statistics.median(samples_ms),
        "latency_ms_mean": statistics.mean(samples_ms),
        "latency_ms_p95": sorted(samples_ms)[int(0.95 * (len(samples_ms) - 1))],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
