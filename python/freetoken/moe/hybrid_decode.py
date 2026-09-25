"""Platform-neutral orchestration for hybrid CPU/GPU MoE decode.

The cache decides which routed experts stay on the GPU, while the CPU executor
computes overflow routes.  This module owns only the ordering and route split:
CUDA/ROCm stream synchronization remains behind ``cpu_executor.decode_submit`` /
``decode_sync`` and model/quantization-specific GPU kernels remain behind the
``GpuExpertRunner`` callback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

import torch


def _overlap_enabled() -> bool:
    return os.getenv("FREETOKEN_HYBRID_OVERLAP", "1") != "0"


class HybridCache(Protocol):
    collect_stats: bool

    def ensure_experts_hybrid(
        self, layer_id: int, expert_ids: torch.Tensor
    ) -> None: ...

    def record_decode_stats_hybrid(self, layer_id: int) -> None: ...

    def copy_missing(self) -> None: ...


class CpuDecodeExecutor(Protocol):
    def decode_submit(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> Any: ...

    def decode_sync(self, pending: Any) -> torch.Tensor: ...


class GpuExpertRunner(Protocol):
    def __call__(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        slot_ids: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class HybridDecodeRequest:
    layer_id: int
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor


class HybridDecodeExecutor:
    """Split hybrid routes, overlap CPU/GPU work, and merge partial results.

    The operation order intentionally matches the original
    ``OffloadMoELayer._decode_hybrid`` implementation.  In particular, CPU
    submission stays before the PCIe fetch and GPU expert kernel so CUDA and
    ROCm retain their existing graph and overlap behavior.
    """

    def __init__(
        self,
        cache: HybridCache,
        cpu_executor: CpuDecodeExecutor,
        *,
        overlap: bool | None = None,
    ) -> None:
        self.cache = cache
        self.cpu_executor = cpu_executor
        self.overlap = _overlap_enabled() if overlap is None else bool(overlap)

    def decode(
        self,
        request: HybridDecodeRequest,
        gpu_expert_runner: GpuExpertRunner,
    ) -> torch.Tensor:
        cache = self.cache

        # Keep raw expert ids for CPU overflow routes.  The cache rewrites the
        # caller-owned tensor in place to GPU slot ids or -1 below.
        raw_ids = request.topk_ids.clone()
        cache.ensure_experts_hybrid(request.layer_id, request.topk_ids)
        if cache.collect_stats:
            cache.record_decode_stats_hybrid(request.layer_id)

        on_gpu = request.topk_ids >= 0
        cpu_ids = torch.where(
            on_gpu,
            raw_ids.new_full((), -1),
            raw_ids,
        ).contiguous()
        pending = self.cpu_executor.decode_submit(
            request.layer_id,
            request.hidden_states,
            request.topk_weights,
            cpu_ids,
        )

        if not self.overlap:
            cpu_result = self.cpu_executor.decode_sync(pending)

        cache.copy_missing()
        gpu_slot_ids = request.topk_ids.clamp_min(0)
        gpu_weights = torch.where(
            on_gpu,
            request.topk_weights,
            request.topk_weights.new_zeros(()),
        ).contiguous()
        gpu_result = gpu_expert_runner(
            request.hidden_states,
            gpu_weights,
            gpu_slot_ids,
        )

        if self.overlap:
            cpu_result = self.cpu_executor.decode_sync(pending)
        return gpu_result + cpu_result


__all__ = [
    "CpuDecodeExecutor",
    "GpuExpertRunner",
    "HybridCache",
    "HybridDecodeExecutor",
    "HybridDecodeRequest",
]
