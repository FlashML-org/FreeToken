"""Shared hybrid decode schedule; cache policy and CPU/GPU kernels stay separate.

CUDA and ROCm use the same routing and ordering here. Platform synchronization
and graph support belong to CpuMoeExecutor.decode_submit/decode_sync.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.offload_cache import OffloadMoeCache

# Read once at startup, as before the extraction. Zero serializes the CPU/GPU work.
_HYBRID_OVERLAP = os.getenv("FREETOKEN_HYBRID_OVERLAP", "1") != "0"


@dataclass(frozen=True)
class HybridDecodeRequest:
    layer_id: int
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor


class HybridDecodeExecutor:
    def __init__(
        self,
        cache: OffloadMoeCache,
        cpu_executor: CpuMoeExecutor,
        *,
        overlap: bool | None = None,
    ):
        self.cache = cache
        self.cpu_executor = cpu_executor
        self.overlap = _HYBRID_OVERLAP if overlap is None else overlap

    def decode(
        self,
        request: HybridDecodeRequest,
        gpu_expert_runner: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Rewrite routing in place, overlap CPU overflow with GPU experts, then sum.

        The callback receives zero weights and slot -1 for CPU-assigned routes;
        it must clamp those slots if its kernel cannot skip inactive routes.
        """
        cache, executor = self.cache, self.cpu_executor
        raw_ids = request.topk_ids.clone()
        cache.ensure_experts_hybrid(request.layer_id, request.topk_ids)
        on_gpu = request.topk_ids >= 0
        cpu_ids = torch.where(on_gpu, -1, raw_ids).contiguous()
        gpu_weights = torch.where(on_gpu, request.topk_weights, 0.0).contiguous()
        if cache.collect_stats:
            cache.record_decode_stats_hybrid(request.layer_id)

        # Even an all-GPU step submits and waits: capture must not freeze the split.
        pending = executor.decode_submit(
            request.layer_id, request.hidden_states, request.topk_weights, cpu_ids,
        )
        cpu_result = executor.decode_sync(pending) if not self.overlap else None
        cache.copy_missing()
        gpu_result = gpu_expert_runner(request.hidden_states, gpu_weights, request.topk_ids)
        if self.overlap:
            cpu_result = executor.decode_sync(pending)
        return gpu_result + cpu_result
