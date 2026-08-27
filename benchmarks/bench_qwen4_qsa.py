"""Microbenchmark Qwen4-Exp QSA selection and exact sparse attention."""

from __future__ import annotations

import argparse
import statistics

import torch

from freetoken.attention.qsa import select_qsa_logical_rows
from freetoken.kernel.triton.qsa import qsa_sparse_gqa


def _median_ms(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=262_144)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.context % 4:
        raise ValueError("context must divide by QSA's compression ratio of four")

    device = torch.device("cuda")
    query = torch.randn(1, 4, 128, dtype=torch.bfloat16, device=device)
    compressed = torch.randn(
        args.context // 4, 1, 128, dtype=torch.bfloat16, device=device
    )
    position = torch.tensor([args.context - 1], dtype=torch.int64, device=device)
    selection_ms = _median_ms(
        lambda: select_qsa_logical_rows(
            query,
            compressed,
            position,
            compress_ratio=4,
            token_budget=2048,
        ),
        5,
        args.repeats,
    )

    attention_query = torch.randn(1, 24, 256, dtype=torch.bfloat16, device=device)
    keys = torch.randn(2048, 2, 256, dtype=torch.bfloat16, device=device)
    values = torch.randn_like(keys)
    rows = torch.arange(2048, dtype=torch.int32, device=device).view(1, -1)
    counts = torch.tensor([2048], dtype=torch.int32, device=device)
    attention_ms = _median_ms(
        lambda: qsa_sparse_gqa(
            attention_query, keys, values, rows, counts, 256**-0.5
        ),
        5,
        args.repeats,
    )
    print(
        {
            "gpu": torch.cuda.get_device_name(device),
            "context": args.context,
            "selection_ms_per_layer": round(selection_ms, 3),
            "attention_ms_per_layer": round(attention_ms, 3),
            "qsa_ms_for_12_layers": round(12 * (selection_ms + attention_ms), 3),
        }
    )


if __name__ == "__main__":
    main()
