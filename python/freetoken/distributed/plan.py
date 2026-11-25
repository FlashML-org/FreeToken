"""The contract between a model's parallel plan and the engine.

A family opts in with ``ModelSpec.parallel_plan``, naming a builder in its ``multi.py``. The engine builds the plan only
when --tp > 1, before any weight is allocated, and construction, loading and the pools read their partitions from it.
How one expert's tensors are cut belongs to its quant kernel (``layers/quantization/moe/shard.py``) and where each slice
lives to the system planner (``moe/placement.py``); a plan only says which ranks split the experts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

import torch

from .info import DistributedInfo


@dataclass(frozen=True)
class ParallelArgs:
    """The layout asked for on the command line: ``tp_size`` ranks, one per GPU, this process being ``rank``."""

    rank: int
    tp_size: int

    def __post_init__(self):
        if not 0 <= self.rank < self.tp_size:
            raise ValueError(f"rank {self.rank} is outside a tensor-parallel world of {self.tp_size}")


@dataclass(frozen=True)
class ExpertGroup:
    """Ranks that each compute one slice of every routed expert's intermediate dim; the MoE output sums over them.

    The MoE input follows an all-reduce over the attention group, so it is bit-identical on every rank: every rank
    routes the same tokens to the same experts without exchanging routes."""

    size: int
    intermediate: int
    # [start, stop) of the intermediate dim this rank computes; the engine sets it from the expert placement
    shard: tuple[int, int] | None = None


@dataclass(frozen=True)
class TensorPartition:
    """One axis of a tensor cut into ``size`` equal contiguous parts; rank ``rank`` keeps its part."""

    axis: int
    rank: int
    size: int

    def slice(self, tensor: torch.Tensor) -> torch.Tensor:
        n = tensor.shape[self.axis]
        if n % self.size:
            raise ValueError(f"cannot split axis {self.axis} of a {tuple(tensor.shape)} tensor {self.size} ways")
        step = n // self.size
        # a copy, so the whole tensor's storage is not kept alive by this rank's part
        return tensor.narrow(self.axis, self.rank * step, step).clone()


class ParallelPlan(Protocol):
    parallel_args: ParallelArgs
    # what attention, dense layers, vocab and KV heads shard over; the engine serves the whole TP group only
    dense: DistributedInfo
    # None: routed experts follow ``dense`` as plain TP, or the model has none
    experts: ExpertGroup | None

    def validate_checkpoint(self, model_path: str) -> None:
        """Reject a checkpoint the plan cannot read its partitions from, before any weight is allocated."""

    def shard(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """This rank's part of one dense weight, named as the model's state dict names it (routed experts load as banks)."""

    def place_experts(self, start: int, stop: int) -> None:
        """Give this rank ``[start, stop)`` of the expert group's intermediate dim; the engine calls it from the placement."""

    def bind(self, model: Any, cache: Any) -> None:
        """After loading: check that what was built matches the plan."""


@dataclass
class BaseParallelPlan:
    """What every plan shares: its fields in one order, whole dense tensors, and the expert group's checks. A family
    subclasses it and overrides ``shard`` and ``bind`` for the layers it splits."""

    parallel_args: ParallelArgs
    dense: DistributedInfo
    experts: ExpertGroup | None

    def validate_checkpoint(self, model_path: str) -> None:
        from freetoken.checkpoint.ftw import is_ftw_checkpoint

        # expert slices are read out of the original checkpoint; FTW banks store whole experts
        if self.experts is not None and is_ftw_checkpoint(model_path):
            raise ValueError(
                f"{model_path} is an FTW checkpoint, whose banks hold whole experts; tensor parallelism slices them from "
                "the original checkpoint, so serve that instead"
            )

    def shard(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def place_experts(self, start: int, stop: int) -> None:
        if self.experts is None:
            raise ValueError("this plan splits no experts")
        self.experts = replace(self.experts, shard=(start, stop))

    def bind(self, model: Any, cache: Any) -> None:
        if self.experts is None:
            return
        from freetoken.layers.moe import iter_moe_layers

        stray = [layer.layer_id for layer in iter_moe_layers(model) if layer.expert_group != self.experts]
        if stray:
            raise ValueError(f"MoE layers {stray} were not built for the parallel plan's expert group {self.experts}")


__all__ = [
    "BaseParallelPlan", "ExpertGroup", "ParallelArgs", "ParallelPlan", "TensorPartition",
]
