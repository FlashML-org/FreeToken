"""Measure FreeToken's native GGUF Q4_0 MoE vector kernels in isolation.

This benchmark deliberately uses the Gemma 4 26B A4B Q4_0 expert geometry
observed on GMKtec EVO-X2: 128 routed experts, top-k 8, hidden width 2816, and MoE
intermediate width 704.  It is not a replacement for the end-to-end OpenAI API
benchmark.  Instead, it supplies the kernel-level evidence needed before a HIP
port changes Q4_0 launch geometry, indexing, or register use.

The benchmark creates valid packed Q4_0 rows directly on the GPU.  Every block
has a finite FP16 scale and random packed nibbles, so the real production
``ggml_moe_a8_vec`` path, including activation quantization, runs without model
loading, host-cache copying, scheduler work, or HTTP overhead.  CUDA events are
used only after warm-up and synchronization; compilation and allocation are not
included in the reported microseconds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from freetoken.kernel.gguf import ggml_moe_a8_vec
from freetoken.models.gguf.dequant import GGML_Q4_0, row_bytes


# These defaults are the verified GMKtec EVO-X2 Gemma 4 26B A4B Q4_0 dimensions.
DEFAULT_EXPERTS = 128
DEFAULT_TOP_K = 8
DEFAULT_HIDDEN = 2816
DEFAULT_INTERMEDIATE = 704


def _parse_args() -> argparse.Namespace:
    """Parse only parameters that preserve a reproducible kernel experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=DEFAULT_EXPERTS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--intermediate", type=int, default=DEFAULT_INTERMEDIATE)
    parser.add_argument("--tokens", type=int, default=1, help="decoded token rows per call")
    parser.add_argument("--warmup", type=int, default=20, help="unmeasured calls per kernel")
    parser.add_argument("--repetitions", type=int, default=200, help="timed calls per kernel")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--json", type=Path, help="write one reproducible JSON result")
    return parser.parse_args()


def _require_valid_geometry(args: argparse.Namespace) -> None:
    """Reject shapes that cannot be represented by the Q4_0 block format."""
    for name in ("hidden", "intermediate"):
        value = getattr(args, name)
        if value <= 0 or value % 32:
            raise ValueError(f"--{name} must be a positive multiple of 32, got {value}")
    for name in ("experts", "top_k", "tokens", "warmup", "repetitions"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.top_k > args.experts:
        raise ValueError("--top-k cannot exceed --experts")
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA or HIP PyTorch device")


def _q4_scale_bytes(device: torch.device) -> torch.Tensor:
    """Return little-endian bytes for a finite FP16 Q4_0 scale of 1/32.

    Q4_0 stores two FP16 scale bytes before every 16-byte packed-nibble payload.
    A constant finite scale is sufficient for performance work and avoids random
    bit patterns that could otherwise create NaNs during the warm-up kernel.
    """
    scale = torch.tensor([1.0 / 32.0], dtype=torch.float16, device=device)
    return scale.view(torch.uint8).reshape(2)


def _make_q4_bank(experts: int, rows: int, columns: int, device: torch.device) -> torch.Tensor:
    """Create a contiguous GPU Q4_0 bank shaped exactly like an expert cache.

    The byte layout is ``[expert, output_row, columns//32, 18]`` before the
    final view.  Byte positions zero and one receive the valid scale, while the
    remaining sixteen bytes contain arbitrary Q4_0 nibbles.  The final shape
    mirrors the packed tensors passed by the Gemma GGUF offload cache.
    """
    packed_row_bytes = row_bytes(columns, GGML_Q4_0)
    blocks = columns // 32
    bank = torch.randint(
        0,
        256,
        (experts, rows, blocks, 18),
        dtype=torch.uint8,
        device=device,
    )
    bank[..., :2] = _q4_scale_bytes(device)
    return bank.reshape(experts, rows, packed_row_bytes).contiguous()


def _make_topk_ids(tokens: int, top_k: int, experts: int, device: torch.device) -> torch.Tensor:
    """Create deterministic valid expert selections without invoking router code."""
    ids = torch.arange(tokens * top_k, dtype=torch.int32, device=device)
    return (ids.remainder(experts)).reshape(tokens, top_k).contiguous()


def _event_time_us(callable_kernel, repetitions: int, device: torch.device) -> float:
    """Return average GPU elapsed time per invocation after explicit synchronization."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start.record()
    for _ in range(repetitions):
        callable_kernel()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repetitions


def main() -> int:
    """Build the two production-shaped calls, warm them, measure them, and emit JSON."""
    args = _parse_args()
    _require_valid_geometry(args)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    # Gate/up maps H to 2I and consumes one routing row for every selected expert.
    hidden = torch.randn(args.tokens, args.hidden, device=device, dtype=torch.bfloat16)
    gate_up = _make_q4_bank(args.experts, 2 * args.intermediate, args.hidden, device)
    route_ids = _make_topk_ids(args.tokens, args.top_k, args.experts, device)

    def gate_up_call() -> torch.Tensor:
        return ggml_moe_a8_vec(
            hidden, gate_up, route_ids, args.top_k, int(GGML_Q4_0), 2 * args.intermediate, args.tokens
        )

    # Down maps I to H.  Its input and routing layout match fused_q4_0.py exactly.
    inter = torch.randn(args.tokens * args.top_k, args.intermediate, device=device, dtype=torch.bfloat16)
    down = _make_q4_bank(args.experts, args.hidden, args.intermediate, device)

    def down_call() -> torch.Tensor:
        return ggml_moe_a8_vec(
            inter, down, route_ids, 1, int(GGML_Q4_0), args.hidden, args.tokens * args.top_k
        )

    # Materialize the extension and check that valid Q4_0 data produces finite outputs.
    for _ in range(args.warmup):
        gate_result = gate_up_call()
        down_result = down_call()
    torch.cuda.synchronize(device)
    if not torch.isfinite(gate_result).all() or not torch.isfinite(down_result).all():
        raise RuntimeError("synthetic Q4_0 data produced a non-finite kernel result")

    wall_start = perf_counter()
    gate_up_us = _event_time_us(gate_up_call, args.repetitions, device)
    down_us = _event_time_us(down_call, args.repetitions, device)
    torch.cuda.synchronize(device)

    result = {
        "device": torch.cuda.get_device_name(device),
        "hip": torch.version.hip,
        "torch": torch.__version__,
        "quant_type": "Q4_0",
        "experts": args.experts,
        "top_k": args.top_k,
        "hidden": args.hidden,
        "intermediate": args.intermediate,
        "tokens": args.tokens,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "gate_up_us": gate_up_us,
        "down_us": down_us,
        "pair_us": gate_up_us + down_us,
        "wall_seconds": perf_counter() - wall_start,
        "gate_up_shape": list(gate_result.shape),
        "down_shape": list(down_result.shape),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
