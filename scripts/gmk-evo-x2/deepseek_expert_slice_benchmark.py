#!/usr/bin/env python3
"""Benchmark an isolated real-shape DeepSeek expert slice on ROCm.

This harness never starts, stops, or contacts a model server.  It opens a
local safetensors checkpoint read-only, selects a bounded set of routed expert
tensors, copies them to the selected HIP device, copies them back, and writes
timing plus tensor-identity evidence.  The result is a transfer and packing
measurement only.  It is not a full-model serving benchmark.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_ids(value: str) -> list[int]:
    """Parse a comma-separated list of non-negative integer IDs."""

    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("IDs must be non-negative integers")
    return result


def main() -> int:
    """Run the guarded transfer measurement and emit JSON evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layers", type=parse_ids, default=[0])
    parser.add_argument("--experts", type=parse_ids, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    if not args.checkpoint.is_dir():
        parser.error("--checkpoint must be a local safetensors directory")

    # Imports are delayed so metadata and --help remain usable without a GPU
    # Python environment.  The actual benchmark requires PyTorch and the
    # safetensors package installed in the target ROCm environment.
    import torch
    from safetensors import safe_open

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA-compatible device is available; ROCm exposes HIP through torch.cuda")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("The isolated slice benchmark requires a HIP/CUDA device")

    names = []
    for layer in args.layers:
        for expert in args.experts:
            prefix = f"layers.{layer}.ffn.experts.{expert}"
            names.extend(f"{prefix}.{suffix}" for suffix in (
                "w1.weight", "w1.scale", "w2.weight", "w2.scale",
                "w3.weight", "w3.scale",
            ))

    # Locate every tensor through safetensors' index without loading the full
    # checkpoint.  Each shard is opened read-only and only selected tensors are
    # materialized, keeping this experiment bounded and reversible.
    index_path = args.checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    missing = [name for name in names if name not in weight_map]
    if missing:
        raise RuntimeError(f"Selected tensors are absent from the checkpoint: {missing[:3]}")

    tensors = []
    evidence = []
    opened: dict[str, object] = {}
    try:
        for name in names:
            shard = weight_map[name]
            if shard not in opened:
                opened[shard] = safe_open(str(args.checkpoint / shard), framework="pt", device="cpu")
            tensor = opened[shard].get_tensor(name)
            tensors.append(tensor)
            evidence.append({"name": name, "shard": shard, "dtype": str(tensor.dtype), "shape": list(tensor.shape), "bytes": tensor.numel() * tensor.element_size()})

        host_bytes = sum(item["bytes"] for item in evidence)
        source = torch.cat([tensor.reshape(-1).view(torch.uint8) for tensor in tensors])
        if source.numel() != host_bytes:
            raise RuntimeError("Tensor byte accounting mismatch")
        gpu = torch.empty_like(source, device=device)
        round_trips = []
        for _ in range(args.repeats):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            gpu.copy_(source, non_blocking=False)
            torch.cuda.synchronize(device)
            h2d_seconds = time.perf_counter() - start
            start = time.perf_counter()
            source.copy_(gpu, non_blocking=False)
            torch.cuda.synchronize(device)
            d2h_seconds = time.perf_counter() - start
            round_trips.append({
                "h2d_seconds": h2d_seconds,
                "d2h_seconds": d2h_seconds,
                "h2d_gib_per_second": host_bytes / h2d_seconds / 1024**3,
                "d2h_gib_per_second": host_bytes / d2h_seconds / 1024**3,
            })
    finally:
        for handle in opened.values():
            handle.__exit__(None, None, None)

    result = {
        "scope": "isolated real-shape routed expert transfer only",
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "layers": args.layers,
        "experts": args.experts,
        "repeats": args.repeats,
        "selected_tensor_count": len(evidence),
        "selected_bytes": host_bytes,
        "selected_mib": host_bytes / 1024**2,
        "tensors": evidence,
        "round_trips": round_trips,
        "protected_service_touched": False,
        "full_model_serving_claim": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
