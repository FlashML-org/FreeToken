"""Measure the dense native-GGUF Q4_0 vector kernels used by Gemma 4 on LAN-223.

The Gemma 4 26B A4B Q4_0 checkpoint has four recurring dense projection
geometries.  They are supplied as defaults here so a HIP optimization can be
measured before it is allowed into the full OpenAI-compatible server benchmark:

* 2816 by 4096 attention output projection;
* 8192 by 2816 full-attention QKV projection;
* 4224 by 2816 fused shared-MLP gate/up projection; and
* 10240 by 2816 sliding-window QKV projection.

Like ``bench_gguf_q4_moe_kernel.py``, this tool creates valid packed Q4_0
weights on the accelerator and measures only post-warm-up GPU event time.  It
does not claim an end-to-end serving rate and must be paired with the API
benchmark before a kernel candidate is accepted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from freetoken.kernel.gguf import ggml_mul_mat_vec_a8
from freetoken.models.gguf.dequant import GGML_Q4_0, row_bytes


# Output rows and input columns, recovered from the exact LAN-223 Gemma GGUF.
DEFAULT_SHAPES = ((2816, 4096), (8192, 2816), (4224, 2816), (10240, 2816))


def _parse_shape(value: str) -> tuple[int, int]:
    """Parse a ``ROWSxCOLS`` override and validate its Q4_0 block alignment."""
    try:
        rows_text, cols_text = value.lower().split("x", 1)
        rows, cols = int(rows_text), int(cols_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be ROWSxCOLS, for example 2816x4096") from error
    if rows <= 0 or cols <= 0 or cols % 32:
        raise argparse.ArgumentTypeError("rows must be positive and cols must be a positive multiple of 32")
    return rows, cols


def _parse_args() -> argparse.Namespace:
    """Parse reproducible dense-kernel benchmark controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        help="repeatable ROWSxCOLS override; defaults to all production shapes",
    )
    parser.add_argument("--vectors", type=int, default=1, help="input rows per kernel call")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument(
        "--output-fp32",
        action="store_true",
        help="isolated Q4_0 destination-type probe; never enables it for model serving",
    )
    parser.add_argument("--json", type=Path, help="write one JSON artifact")
    return parser.parse_args()


def _make_q4_weight(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    """Create finite, contiguous ``[rows, row_bytes(cols)]`` Q4_0 packed weights."""
    blocks = cols // 32
    weight = torch.randint(0, 256, (rows, blocks, 18), dtype=torch.uint8, device=device)
    # Q4_0 starts each 18-byte block with an FP16 scale.  Use 1/32 rather than
    # arbitrary random bytes so the measured real kernel cannot create NaNs.
    scale_bytes = torch.tensor([1.0 / 32.0], dtype=torch.float16, device=device).view(torch.uint8)
    weight[..., :2] = scale_bytes.reshape(1, 1, 2)
    return weight.reshape(rows, row_bytes(cols, GGML_Q4_0)).contiguous()


def _average_event_us(kernel, repetitions: int, device: torch.device) -> float:
    """Measure an already-warmed kernel with GPU events and return microseconds/call."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start.record()
    for _ in range(repetitions):
        kernel()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repetitions


def main() -> int:
    """Run the selected dense projection shapes and write a durable JSON result."""
    args = _parse_args()
    if args.vectors <= 0 or args.warmup <= 0 or args.repetitions <= 0:
        raise ValueError("--vectors, --warmup, and --repetitions must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA or HIP PyTorch device")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    shapes = args.shape or DEFAULT_SHAPES
    measurements = []

    for rows, cols in shapes:
        weight = _make_q4_weight(rows, cols, device)
        x = torch.randn(args.vectors, cols, dtype=torch.bfloat16, device=device)

        def call() -> torch.Tensor:
            return ggml_mul_mat_vec_a8(
                weight,
                x,
                int(GGML_Q4_0),
                rows,
                output_fp32=args.output_fp32,
            )

        for _ in range(args.warmup):
            result = call()
        torch.cuda.synchronize(device)
        if not torch.isfinite(result).all():
            raise RuntimeError(f"non-finite result for dense Q4_0 shape {rows}x{cols}")
        measurements.append(
            {
                "rows": rows,
                "cols": cols,
                "vectors": args.vectors,
                "output_shape": list(result.shape),
                "output_dtype": str(result.dtype),
                "average_us": _average_event_us(call, args.repetitions, device),
            }
        )

    output = {
        "device": torch.cuda.get_device_name(device),
        "hip": torch.version.hip,
        "torch": torch.__version__,
        "quant_type": "Q4_0",
        "output_fp32": args.output_fp32,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "measurements": measurements,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
