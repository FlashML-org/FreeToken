"""Diagnostic dense GGUF native matvec timing; emits route and output identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time

import torch

from freetoken.kernel.gguf import ggml_mul_mat_vec_a8
from freetoken.kernel.gguf import gguf_runtime_metadata
from freetoken.models.gguf.dequant import GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0, row_bytes


CASES = {
    "q4_k": (256, 256, GGML_Q4_K),
    "q5_k": (256, 256, GGML_Q5_K),
    "q6_k": (256, 256, GGML_Q6_K),
    "q8_0": (256, 256, GGML_Q8_0),
}


def _packed(rows: int, cols: int, quant_type: int) -> torch.Tensor:
    return torch.zeros((rows, row_bytes(cols, quant_type)), dtype=torch.uint8, device="cuda")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm device required")
    if args.batch < 1 or args.iters < 1:
        raise SystemExit("batch and iters must be positive")
    names = list(CASES) if args.case == "all" else [args.case]
    for name in names:
        rows, cols, quant_type = CASES[name]
        weight = _packed(rows, cols, quant_type)
        hidden = torch.randn((args.batch, cols), dtype=torch.bfloat16, device="cuda")
        for _ in range(args.warmup):
            output = ggml_mul_mat_vec_a8(weight, hidden, quant_type, rows)
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.iters):
            start = time.perf_counter()
            output = ggml_mul_mat_vec_a8(weight, hidden, quant_type, rows)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1e6)
        if not torch.isfinite(output).all():
            raise RuntimeError("non-finite GGUF output")
        record = {
            "case": name,
            "batch": args.batch,
            "quant_type": quant_type,
            "median_us": statistics.median(samples),
            "runtime": gguf_runtime_metadata(),
            "output_sha256": hashlib.sha256(output.float().cpu().numpy().tobytes()).hexdigest(),
        }
        print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
