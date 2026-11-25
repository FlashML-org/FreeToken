"""FREETOKEN_CHECK_SPLIT_EXPERTS=1: the ranks that split the routed experts compare each MoE layer's routes, and under
--moe-host-pool numa hybrid each layer's GPU/CPU split of them, and stop at the first layer where a rank differs.

Each rank computes its slice of the same routes and the MoE output sums the slices, so the ranks must route alike; a
node's shared CPU pool computes the CPU share of the routes for every rank on the node, so those ranks must also split
the routes alike. Both hold while the ranks' GPUs compute alike (the engine refuses split experts over different GPU
models), and nothing checks them at run time otherwise. The check syncs the host at every MoE layer, so it runs without
CUDA graphs.
"""

from __future__ import annotations

import hashlib
import os

import torch

ENABLED = os.environ.get("FREETOKEN_CHECK_SPLIT_EXPERTS") == "1"

_group = None
_split = False


def enable(group, *, hybrid_split: bool) -> None:
    """Compare over ``group`` (the TP CPU group); ``hybrid_split``: the ranks share CPU pools, so compare the split too."""
    global _group, _split
    _group, _split = group, hybrid_split


def digest(*tensors: torch.Tensor) -> int:
    h = hashlib.blake2b(digest_size=7)
    for t in tensors:
        h.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return int.from_bytes(h.digest(), "little")


def _agree(value: int) -> bool:
    both = torch.tensor([value, -value], dtype=torch.int64)
    torch.distributed.all_reduce(both, op=torch.distributed.ReduceOp.MIN, group=_group)
    return int(both[0]) == -int(both[1])


def check_routes(layer_id: int, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> None:
    if _group is not None and not torch.cuda.is_current_stream_capturing() and not _agree(digest(topk_ids, topk_weights)):
        raise RuntimeError(f"the ranks that split the routed experts routed MoE layer {layer_id} differently")


def check_split(layer_id: int, cpu_ids: torch.Tensor) -> None:
    if _group is not None and _split and not torch.cuda.is_current_stream_capturing() and not _agree(digest(cpu_ids)):
        raise RuntimeError(
            f"the ranks sharing a CPU expert pool split MoE layer {layer_id}'s routes between GPU and CPU differently"
        )


__all__ = ["ENABLED", "check_routes", "check_split", "digest", "enable"]
