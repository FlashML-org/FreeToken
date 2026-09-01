#!/usr/bin/env python3
"""Screen fused HIP expert-copy grid width on Qwen3.6 NVFP4 geometry.

The production offload cache copies every missing expert through one fused,
six-bank ``fast_index_copy_multi`` launch. This script constructs that same
aligned mapped-host-memory layout, selects a fixed number of missing experts,
and compares the two AOT-compiled grid widths currently available in the
gfx1151 kernel cache. It verifies copied tensor equality before recording
device-event timing, so a timing row cannot represent a broken copy.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from freetoken.gpu_select import bind_assigned_gpu
from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
from freetoken.moe.benchbw import WORKLOADS, _build_gather_rig


def parse_args() -> argparse.Namespace:
    """Accept a bounded production-layout copy candidate and timing count."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks-per-bank", type=int, choices=(8, 64), required=True)
    parser.add_argument("--misses", type=int, choices=range(1, 9), required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    return parser.parse_args()


def launch_copy(cache, blocks_per_bank: int) -> None:
    """Launch the production fused gather with only its grid-width specialization changed."""

    fast_index_copy_multi_jit(
        cache._copy_dst_ptrs,
        cache._copy_src_ptrs[0],
        cache._copy_feat_bytes,
        cache.evict_slots,
        cache.src_indices,
        cache.num_indices,
        blocks_per_bank=blocks_per_bank,
    )


def assert_copied(cache, misses: int) -> None:
    """Prove every selected source row exactly reached its corresponding cache slot."""

    dst = cache.evict_slots[:misses].cpu()
    src = cache.src_indices[:misses].cpu()
    for source_layers, slot_cache in cache.banks:
        expected = source_layers[0][src].cpu()
        actual = slot_cache[dst].cpu()
        if not torch.equal(actual, expected):
            raise RuntimeError("fused copy result differs from the mapped-host source row")


def time_copy(cache, blocks_per_bank: int, iterations: int) -> list[float]:
    """Return HIP event durations for repeated fixed-state fused gathers."""

    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch_copy(cache, blocks_per_bank)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def main() -> int:
    """Build the Qwen layout, verify the candidate, and emit one reproducible JSON row."""

    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("native HIP/CUDA device is required")
    device = bind_assigned_gpu()
    cache, full_layer_bytes = _build_gather_rig("nvfp4", WORKLOADS["qwen3.6-moe"], device)
    cache.num_indices.fill_(args.misses)

    for _ in range(args.warmup):
        launch_copy(cache, args.blocks_per_bank)
    torch.cuda.synchronize(device)
    assert_copied(cache, args.misses)
    samples = time_copy(cache, args.blocks_per_bank, args.iterations)
    copied_bytes = full_layer_bytes * args.misses // cache.num_experts
    result = {
        "schema_version": 1,
        "blocks_per_bank": args.blocks_per_bank,
        "misses": args.misses,
        "banks": len(cache.banks),
        "copied_bytes": copied_bytes,
        "copy_verified": True,
        "iterations": args.iterations,
        "latency_ms_median": statistics.median(samples),
        "latency_ms_mean": statistics.mean(samples),
        "latency_ms_p95": sorted(samples)[int(0.95 * (len(samples) - 1))],
        "bandwidth_gbps": copied_bytes / (statistics.median(samples) * 1e6),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
