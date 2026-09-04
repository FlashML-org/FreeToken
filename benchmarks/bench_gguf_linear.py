"""Exact-shape GGUF dense GEMV benchmark for Qwen3.6-MoE decode."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time

import torch

from freetoken.kernel.gguf import ggml_mul_mat_a8, ggml_mul_mat_vec_a8
from freetoken.models.gguf.dequant import GGML_Q6_K, GGML_Q8_0, row_bytes
from freetoken.utils.graph_gate import rocm_blas_report


CASES = {
    "attn_qkv": (8192, 2048, GGML_Q8_0),
    "attn_gate": (4096, 2048, GGML_Q8_0),
    "attn_output": (2048, 4096, GGML_Q8_0),
    "gdn_out": (2048, 4096, GGML_Q8_0),
    "shared_gate_up": (512, 2048, GGML_Q8_0),
    "shared_down": (2048, 512, GGML_Q8_0),
    "lm_head": (248320, 2048, GGML_Q6_K),
}


def _packed_weight(out_features: int, in_features: int, quant_type: int) -> torch.Tensor:
    if quant_type == GGML_Q8_0:
        blocks = torch.zeros(
            (out_features, in_features // 32, 34), dtype=torch.uint8, device="cuda"
        )
        # Valid positive fp16 scale, followed by bounded signed int8 payload.
        blocks[..., :2] = torch.tensor([128, 63], dtype=torch.uint8, device="cuda")
        blocks[..., 2:] = torch.randint(
            0, 255, blocks[..., 2:].shape, dtype=torch.uint8, device="cuda"
        )
        return blocks.reshape(out_features, row_bytes(in_features, quant_type))
    if quant_type == GGML_Q6_K:
        blocks = torch.zeros(
            (out_features, in_features // 256, 210), dtype=torch.uint8, device="cuda"
        )
        # Q6_K layout ends with fp16 d; scales and quants may be arbitrary bytes.
        blocks[..., 192:208] = torch.randint(
            1, 255, blocks[..., 192:208].shape, dtype=torch.uint8, device="cuda"
        )
        blocks[..., 208:210] = torch.tensor([128, 63], dtype=torch.uint8, device="cuda")
        return blocks.reshape(out_features, row_bytes(in_features, quant_type))
    raise ValueError(f"unsupported quant type {quant_type}")


def _time_call(fn, warmup: int, iters: int) -> tuple[float, torch.Tensor]:
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    # Host timing avoids event allocation in the measured loop; synchronize once
    # after each op so async failures cannot contaminate later samples.
    samples = []
    for i in range(iters):
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    if not torch.isfinite(out).all():
        raise RuntimeError("GGUF linear benchmark produced non-finite output")
    return statistics.median(samples), out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1, help="exact dense batch shape")
    parser.add_argument("--json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm device required")
    if args.batch < 1:
        raise SystemExit("--batch must be >= 1")
    torch.manual_seed(20260831)
    blas = rocm_blas_report(gate={})
    if blas["verification"] == "mismatch":
        raise SystemExit(f"requested BLAS policy not effective: {blas}")

    names = list(CASES) if args.case == "all" else [args.case]
    rows = []
    generator = torch.Generator(device="cpu").manual_seed(20260831)
    for name in names:
        out_features, in_features, quant_type = CASES[name]
        weight = _packed_weight(out_features, in_features, quant_type)
        x = torch.randn(
            args.batch, in_features, generator=generator, dtype=torch.bfloat16, device="cpu"
        ).to("cuda")
        calls = {
            "mmvq": lambda: ggml_mul_mat_vec_a8(weight, x, quant_type, out_features),
            "mmq": lambda: ggml_mul_mat_a8(weight, x, quant_type, out_features),
        }
        for impl, fn in calls.items():
            median_us, out = _time_call(fn, args.warmup, args.iters)
            rows.append(
                {
                    "case": name,
                    "impl": impl,
                    "out_features": out_features,
                    "in_features": in_features,
                    "quant_type": quant_type,
                    "batch": args.batch,
                    "blas": blas,
                    "median_us": round(median_us, 3),
                    "finite": bool(torch.isfinite(out).all().item()),
                    "output_sha256": hashlib.sha256(
                        out.detach().cpu().float().numpy().tobytes()
                    ).hexdigest(),
                }
            )
            print(json.dumps(rows[-1], sort_keys=True))
        del weight, x
        torch.cuda.empty_cache()
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
