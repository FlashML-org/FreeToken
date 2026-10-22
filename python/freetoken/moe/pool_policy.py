"""Slot pool policy for the offload MoE cache: which layers share a pool, and how many slots each pool gets.

Pure functions. The engine supplies the constraints and hands the result to ``OffloadMoeCache``:

    groups = group_layers(scope, layouts, slot_bytes)                    # --moe-cache-scope, once per load
    pools = make_pools(groups, num_experts, min_num_slots, slot_limit)   # the minimum follows the decode batch
    sizes = split_slots(total, pools, counts)                            # --moe-cache-split, counts from resolve_profile
    OffloadMoeCache(pool_of_layer=pool_of_layer(pools, num_layers), pool_sizes=sizes, ...)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

SCOPES = ("unified", "per-layer")


@dataclass(frozen=True)
class PoolSpec:
    layers: tuple[int, ...]
    min_num_slots: int
    max_num_slots: int


def group_layers(scope: str, layouts: list, slot_bytes: list[int]) -> list[list[int]]:
    """The layers of each pool, in placement order.

    ``unified``: one pool per layout key (see ``bank_layouts``); ``per-layer``: one pool per layer.
    Pools with wider slots come first: the prefill staging buffers sit at the start of every bank, so they then cover fewer slots.
    """
    if scope == "unified":
        groups: dict = {}
        for layer, layout in enumerate(layouts):
            groups.setdefault(layout, []).append(layer)
        pools = list(groups.values())
    elif scope == "per-layer":
        pools = [[layer] for layer in range(len(layouts))]
    else:
        raise ValueError(f"unknown --moe-cache-scope {scope!r}; want one of {SCOPES}")
    return sorted(pools, key=lambda layers: (-slot_bytes[layers[0]], layers[0]))


def make_pools(groups: list[list[int]], num_experts: int, min_num_slots: int, slot_limit: int | None) -> list[PoolSpec]:
    """Each group's slot range: at least ``min_num_slots``, at most one slot per expert of its layers and the kernel's ``slot_limit``."""
    pools = []
    for layers in groups:
        max_num_slots = len(layers) * num_experts
        if slot_limit is not None:
            max_num_slots = min(max_num_slots, slot_limit)
        pools.append(PoolSpec(tuple(layers), min(min_num_slots, max_num_slots), max_num_slots))
    return pools


def pool_of_layer(pools: list[PoolSpec], num_layers: int) -> list[int]:
    """Each layer's pool index, the form ``OffloadMoeCache`` takes."""
    owner = [-1] * num_layers
    for p, pool in enumerate(pools):
        for layer in pool.layers:
            owner[layer] = p
    assert min(owner) >= 0, owner
    return owner


def split_slots(total: int, pools: list[PoolSpec], counts: torch.Tensor | None = None) -> list[int]:
    """Slots per pool, each in [min_num_slots, max_num_slots], summing to ``total``.

    With ``counts`` (per-layer expert use from ``resolve_profile``) the slots follow expert use, else the pools' layer counts.
    """
    if len(pools) == 1:
        return [total]
    _check_budget(total, pools)
    return _split_equal(total, pools) if counts is None else _split_hot(total, pools, counts)


def _check_budget(total: int, pools: list[PoolSpec]) -> None:
    need = sum(p.min_num_slots for p in pools)
    room = sum(p.max_num_slots for p in pools)
    if total < need:
        raise ValueError(f"MoE cache of {total} slots is below what the pools need ({need} slots)")
    if total > room:
        raise ValueError(f"MoE cache of {total} slots exceeds what the pools can hold ({room} slots)")


def _split_equal(total: int, pools: list[PoolSpec]) -> list[int]:
    sizes = [p.min_num_slots for p in pools]
    weights = [len(p.layers) for p in pools]
    left = total - sum(sizes)
    while left > 0:
        open_ = [i for i, p in enumerate(pools) if sizes[i] < p.max_num_slots]
        # give the next slot to the pool with the fewest slots per layer
        i = min(open_, key=lambda j: (sizes[j] / weights[j], j))
        sizes[i] += 1
        left -= 1
    return sizes


def _split_hot(total: int, pools: list[PoolSpec], counts: torch.Tensor) -> list[int]:
    sizes = [p.min_num_slots for p in pools]
    # a pool's minimum covers its hottest experts; the rest, up to its maximum, competes across pools
    cand = []
    for p, pool in enumerate(pools):
        hot = counts[list(pool.layers)].flatten().sort(descending=True).values
        cand.extend((float(c), p) for c in hot[pool.min_num_slots : pool.max_num_slots])
    cand.sort(key=lambda c: -c[0])
    for _, p in cand[: total - sum(sizes)]:
        sizes[p] += 1
    return sizes


def resolve_profile(spec: str, model_path: str, num_layers: int, num_experts: int) -> torch.Tensor | None:
    """Expert use for ``--moe-cache-split``: None for ``equal``, else the profile at ``profile:<path>``."""
    if spec == "equal":
        return None
    path = Path(spec.removeprefix("profile:"))
    logger.info_rank0(f"MoE pool profile: {path}")
    return load_profile(path, num_layers, num_experts, model_path)


def load_profile(path: str | Path, num_layers: int, num_experts: int, model_path: str = "") -> torch.Tensor:
    """Per-layer expert use ``[num_layers, num_experts]`` from a profile, each row scaled to sum to 1."""
    data = json.loads(Path(path).read_text())
    model = str(data.get("model", ""))
    if model and model_path and not any(n.lower().startswith(model.lower()) for n in _model_names(model_path)):
        logger.warning_rank0(f"MoE pool profile {path} was taken on {model!r}, not on {model_path}")
    counts = torch.tensor(data["expert_counts"], dtype=torch.float64)
    if tuple(counts.shape) != (num_layers, num_experts):
        raise ValueError(
            f"profile {path} has {tuple(counts.shape)} expert counts; this model needs ({num_layers}, {num_experts})"
        )
    # every layer decodes the same tokens, so equal row sums keep the layers comparable
    return counts / counts.sum(dim=1, keepdim=True).clamp(min=1e-12)


def _model_names(model_path: str) -> list[str]:
    # every path component, so an HF cache snapshot (models--Org--Name/snapshots/<hash>) still finds its name
    names = []
    for part in reversed(Path(model_path).parts):
        name = part.split("--")[-1].removesuffix(".gguf")
        if name:
            names.append(name)
    return names
