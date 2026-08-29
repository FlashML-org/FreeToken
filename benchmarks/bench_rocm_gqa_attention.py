"""Benchmark the Gemma 4 ROCm GQA decode tile without changing serving defaults.

Gemma 4's sliding attention is 16 query heads by 8 KV heads at head dimension
256.  ROCm serving pads this group-of-two GQA tile to 16 query-head lanes so
Triton can lower ``tl.dot`` to RDNA WMMA.  This tool calls the same attention
function twice on identical tensors: once with the default tile and once with
an explicitly requested HIP probe tile.  It checks numerical agreement before
reporting GPU-event latency, so a compilation success alone is never treated
as an optimization result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from freetoken.kernel.triton.attention import decode_paged_attention


def _parse_args() -> argparse.Namespace:
    """Parse reproducible ROCm GQA tile benchmark controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-block-h", type=int)
    parser.add_argument("--probe-block-n", type=int)
    parser.add_argument("--probe-num-warps", type=int)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def _event_us(call, repetitions: int) -> float:
    """Return post-warm-up accelerator event time for an already-built kernel."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(repetitions):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repetitions


def main() -> int:
    """Execute the exact Gemma sliding-GQA decode comparison on the HIP device."""
    args = _parse_args()
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("this benchmark requires a HIP PyTorch device")
    if args.sequence_length <= 0 or args.warmup <= 0 or args.repetitions <= 0:
        raise ValueError("sequence length, warmup, and repetitions must be positive")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    batch, query_heads, kv_heads, head_dim, splits = 1, 16, 8, 256, 8
    q = torch.randn(batch, query_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(args.sequence_length, kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    indptr = torch.tensor([0, args.sequence_length], dtype=torch.int32, device=device)
    indices = torch.arange(args.sequence_length, dtype=torch.int32, device=device)
    positions = torch.tensor([args.sequence_length - 1], dtype=torch.int64, device=device)
    mid_o = torch.empty(batch, query_heads, splits, head_dim, dtype=torch.float32, device=device)
    mid_lse = torch.empty(batch, query_heads, splits, dtype=torch.float32, device=device)
    num_splits = torch.full((batch,), splits, dtype=torch.int32, device=device)

    def call(probe_h: int | None, probe_n: int | None, probe_warps: int | None) -> torch.Tensor:
        return decode_paged_attention(
            q, k, v, indptr, indices, positions, mid_o, mid_lse, num_splits,
            splits, head_dim**-0.5, sliding_window=1024, rocm_block_h_probe=probe_h,
            rocm_block_n_probe=probe_n, rocm_num_warps_probe=probe_warps,
        )

    for _ in range(args.warmup):
        default = call(None, None, None)
    for _ in range(args.warmup):
        candidate = call(args.probe_block_h, args.probe_block_n, args.probe_num_warps)
    torch.cuda.synchronize()
    torch.testing.assert_close(candidate.float(), default.float(), atol=2e-2, rtol=2e-2)
    result = {
        "device": torch.cuda.get_device_name(device),
        "hip": torch.version.hip,
        "geometry": {"q_heads": query_heads, "kv_heads": kv_heads, "head_dim": head_dim},
        "sequence_length": args.sequence_length,
        "probe_block_h": args.probe_block_h,
        "probe_block_n": args.probe_block_n,
        "probe_num_warps": args.probe_num_warps,
        "default_us": _event_us(lambda: call(None, None, None), args.repetitions),
        "probe_us": _event_us(
            lambda: call(args.probe_block_h, args.probe_block_n, args.probe_num_warps),
            args.repetitions,
        ),
        "warmup": args.warmup,
        "repetitions": args.repetitions,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
