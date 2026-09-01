#!/usr/bin/env python3
"""Measure the exact Qwen3.6 Q4_K and Q5_K routed-MoE kernels on LAN-223.

This screening benchmark reads real packed rows from the qualified Qwen3.6
Q4_K_M GGUF instead of manufacturing bytes.  It copies eight actual experts
from one selected MoE layer to the accelerator, uses deterministic routes, and
calls FreeToken's production ``ggml_moe_a8_vec`` binding.  The gate/up call has
the model's Q4_K 512-to-2,048 shape; the down call has its Q5_K
2,048-to-512 shape.  GPU event time is useful for selecting a kernel candidate
but is never a substitute for the quality-gated OpenAI API measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from freetoken.kernel.gguf import ggml_moe_a8_vec
from freetoken.models.gguf.dequant import GGML_Q4_K, GGML_Q5_K
from freetoken.models.gguf.reader import GgufTensor, iter_gguf_tensors


# The qualified Qwen model has 256 experts and routes eight experts per token.
DEFAULT_EXPERT_COUNT = 256
DEFAULT_TOP_K = 8
DEFAULT_LAYER = 0


def _parse_args() -> argparse.Namespace:
    """Read explicit benchmark controls and refuse implicit model selection."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="qualified Q4_K_M GGUF")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER, help="MoE layer to sample")
    parser.add_argument("--warmup", type=int, default=30, help="unmeasured production-kernel calls")
    parser.add_argument("--repetitions", type=int, default=300, help="timed calls per projection")
    parser.add_argument("--json", type=Path, required=True, help="new JSON artifact path")
    return parser.parse_args()


def _require_inputs(args: argparse.Namespace) -> None:
    """Validate every input before mapping model data or reserving the GPU."""

    if not args.model.is_file():
        raise FileNotFoundError(f"GGUF model is missing: {args.model}")
    if args.layer < 0:
        raise ValueError("--layer must be non-negative")
    if args.warmup <= 0 or args.repetitions <= 0:
        raise ValueError("--warmup and --repetitions must be positive")
    if args.json.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {args.json}")
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA or HIP PyTorch device")


def _tensor_map(model: Path) -> dict[str, GgufTensor]:
    """Index GGUF tensor records once while retaining their zero-copy packed views."""

    return {tensor.name: tensor for tensor in iter_gguf_tensors(str(model))}


def _expert_bank(tensor: GgufTensor, device: torch.device) -> torch.Tensor:
    """Copy exactly eight real expert banks to GPU in FreeToken's packed layout.

    The qualified tensors expose torch shape ``[experts, rows, columns]`` and a
    packed CPU view ``[experts * rows, row_bytes]``.  Reshaping is metadata-only;
    selecting the first eight experts bounds device memory while retaining the
    quantization bytes used by the real model.
    """

    experts, rows, _columns = tensor.shape
    if experts != DEFAULT_EXPERT_COUNT:
        raise ValueError(f"expected {DEFAULT_EXPERT_COUNT} experts, got {experts} in {tensor.name}")
    packed = tensor.packed().reshape(experts, rows, -1)
    return packed[:DEFAULT_TOP_K].contiguous().to(device=device, non_blocking=False)


def _event_time_us(kernel, repetitions: int, device: torch.device) -> float:
    """Return synchronized average device time in microseconds for one call."""

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start.record()
    for _ in range(repetitions):
        kernel()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repetitions


def _finite(tensor: torch.Tensor, label: str) -> None:
    """Fail closed if an experimental kernel creates an invalid floating result."""

    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{label} produced non-finite output")


def main() -> int:
    """Load true packed experts, warm both projections, and write one evidence file."""

    args = _parse_args()
    _require_inputs(args)
    device = torch.device("cuda")
    tensors = _tensor_map(args.model)
    prefix = f"blk.{args.layer}."
    gate_name = prefix + "ffn_gate_exps.weight"
    up_name = prefix + "ffn_up_exps.weight"
    down_name = prefix + "ffn_down_exps.weight"
    missing = [name for name in (gate_name, up_name, down_name) if name not in tensors]
    if missing:
        raise KeyError(f"GGUF lacks required MoE tensors: {missing}")

    # Qwen stores gate and up separately, so screen each real Q4_K bank.  The
    # production fused path uses the same routed-vector binding for both banks.
    gate = _expert_bank(tensors[gate_name], device)
    up = _expert_bank(tensors[up_name], device)
    down = _expert_bank(tensors[down_name], device)
    if tensors[gate_name].ggml_type != GGML_Q4_K or tensors[up_name].ggml_type != GGML_Q4_K:
        raise ValueError("Qwen gate/up tensors must be Q4_K for this benchmark")
    if tensors[down_name].ggml_type != GGML_Q5_K:
        raise ValueError("Qwen down tensor must be Q5_K for this benchmark")

    # One decoded token selects each copied expert once, matching Qwen's top-k
    # cardinality while avoiding any router or scheduler work in this screen.
    route_ids = torch.arange(DEFAULT_TOP_K, dtype=torch.int32, device=device).reshape(1, -1)
    hidden = torch.randn(1, 512, dtype=torch.bfloat16, device=device)
    intermediate = torch.randn(DEFAULT_TOP_K, 2048, dtype=torch.bfloat16, device=device)

    def gate_call() -> torch.Tensor:
        return ggml_moe_a8_vec(hidden, gate, route_ids, DEFAULT_TOP_K, int(GGML_Q4_K), 2048, 1)

    def up_call() -> torch.Tensor:
        return ggml_moe_a8_vec(hidden, up, route_ids, DEFAULT_TOP_K, int(GGML_Q4_K), 2048, 1)

    def down_call() -> torch.Tensor:
        return ggml_moe_a8_vec(intermediate, down, route_ids, 1, int(GGML_Q5_K), 512, DEFAULT_TOP_K)

    for _ in range(args.warmup):
        gate_output = gate_call()
        up_output = up_call()
        down_output = down_call()
    torch.cuda.synchronize(device)
    _finite(gate_output, "Q4_K gate")
    _finite(up_output, "Q4_K up")
    _finite(down_output, "Q5_K down")

    result = {
        "schema_version": 1,
        "model": str(args.model.resolve()),
        "layer": args.layer,
        "device": torch.cuda.get_device_name(device),
        "hip": torch.version.hip,
        "torch": torch.__version__,
        "experts_copied": DEFAULT_TOP_K,
        "top_k": DEFAULT_TOP_K,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "gate_q4k_us": _event_time_us(gate_call, args.repetitions, device),
        "up_q4k_us": _event_time_us(up_call, args.repetitions, device),
        "down_q5k_us": _event_time_us(down_call, args.repetitions, device),
        "gate_shape": list(gate_output.shape),
        "up_shape": list(up_output.shape),
        "down_shape": list(down_output.shape),
    }
    result["three_projection_us"] = result["gate_q4k_us"] + result["up_q4k_us"] + result["down_q5k_us"]
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
