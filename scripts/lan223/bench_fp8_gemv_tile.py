#!/usr/bin/env python3
"""Measure one isolated FP8 W8A16 GEMV tile on LAN-223's native HIP path.

This is deliberately a kernel screen, not a model-quality benchmark.  It uses
one of Qwen3.6's common ``[N, 2048]`` dense projection shapes, deterministic
synthetic tensors, a fixed number of warmup and timed launches, and reports a
SHA-1 of the BF16 result.  Invoke one process per tile because Triton reads the
tile environment setting when its module is imported.  A matching hash proves
this synthetic kernel result is identical, but a full model gate is still
required before any server configuration is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics

import torch


def parse_args() -> argparse.Namespace:
    """Parse the bounded, reproducible measurement parameters."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile", type=int, choices=(16, 32), required=True)
    parser.add_argument("--warps", type=int, choices=(1, 2), default=1)
    parser.add_argument("--rows", type=int, choices=(512, 2048, 4096, 8192), default=8192)
    parser.add_argument("--scale-activation", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def time_one(callable_operation) -> float:
    """Return one device-synchronized HIP elapsed time in milliseconds."""

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    callable_operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def main() -> int:
    """Allocate deterministic Qwen-shaped inputs, run the GEMV, and emit JSON."""

    args = parse_args()
    # Set before importing the module because the setting selects a Triton
    # specialization at import time.  Refuse a conflicting inherited setting.
    inherited = os.environ.get("FREETOKEN_FP8_GEMV_BLOCK_N")
    if inherited not in (None, str(args.tile)):
        raise SystemExit(
            f"requested tile {args.tile}, inherited FREETOKEN_FP8_GEMV_BLOCK_N={inherited}"
        )
    os.environ["FREETOKEN_FP8_GEMV_BLOCK_N"] = str(args.tile)
    # The wave-count selection is also import-time Triton specialization.
    inherited_warps = os.environ.get("FREETOKEN_FP8_GEMV_NUM_WARPS")
    if inherited_warps not in (None, str(args.warps)):
        raise SystemExit(
            f"requested warps {args.warps}, inherited FREETOKEN_FP8_GEMV_NUM_WARPS={inherited_warps}"
        )
    os.environ["FREETOKEN_FP8_GEMV_NUM_WARPS"] = str(args.warps)
    os.environ["FREETOKEN_FP8_GEMV_SCALE_ACTIVATION"] = "1" if args.scale_activation else "0"

    from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

    if not torch.cuda.is_available():
        raise SystemExit("native HIP/CUDA device is required")
    torch.manual_seed(223_8192_2048)
    device = torch.device("cuda")
    # These dimensions cover Qwen3.6's profiled dense projection widths. FP8
    # weights preserve the memory access width of the real decode kernel.
    activation = torch.randn(1, 2048, device=device, dtype=torch.bfloat16)
    weight = torch.randn(args.rows, 2048, device=device).clamp(-5, 5).to(torch.float8_e4m3fn)
    scale = torch.full((args.rows,), 1.0 / 32.0, device=device, dtype=torch.float32)

    def operation() -> torch.Tensor:
        """Execute the same M=1 dispatch that the model uses for W8A16 decode."""

        return fp8_pertensor_linear(activation, weight, scale)

    for _ in range(args.warmup):
        result = operation()
    torch.cuda.synchronize()
    samples_ms = [time_one(operation) for _ in range(args.iterations)]
    result = operation()
    torch.cuda.synchronize()
    # BF16 has no NumPy representation on some wheels, so hash its raw uint16
    # payload.  This is an exact, format-stable comparison across tile runs.
    result_hash = hashlib.sha1(result.view(torch.uint16).cpu().numpy().tobytes()).hexdigest()
    print(json.dumps({
        "schema_version": 1,
        "tile": args.tile,
        "warps": args.warps,
        "shape": [args.rows, 2048],
        "scale_activation": args.scale_activation,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "result_sha1": result_hash,
        "latency_ms_median": statistics.median(samples_ms),
        "latency_ms_mean": statistics.mean(samples_ms),
        "latency_ms_p95": sorted(samples_ms)[int(0.95 * (len(samples_ms) - 1))],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
