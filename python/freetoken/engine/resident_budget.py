"""Allocation-free fit planning for native GGUF resident MoE execution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal

_MIB = 1 << 20
GRAPH_RESERVE_BYTES = 768 * _MIB
MIN_LOAD_SCRATCH_BYTES = 512 * _MIB
MIN_SAFETY_BYTES = 1_500 * _MIB


@dataclass(frozen=True)
class PhaseMemory:
    """One synchronized allocator/driver high-water observation."""

    name: str
    start_free_bytes: int
    end_free_bytes: int
    allocator_peak_allocated_bytes: int
    allocator_peak_reserved_bytes: int
    minimum_driver_free_bytes: int
    total_driver_bytes: int

    @property
    def driver_used_high_water_bytes(self) -> int:
        return self.total_driver_bytes - self.minimum_driver_free_bytes

    @property
    def non_torch_bytes(self) -> int:
        # Driver usage already includes allocator-reserved memory. This is a diagnostic
        # remainder, never an additive term in required_bytes.
        return max(0, self.driver_used_high_water_bytes - self.allocator_peak_reserved_bytes)

    @property
    def required_bytes(self) -> int:
        return max(self.driver_used_high_water_bytes, self.allocator_peak_reserved_bytes)

    def as_dict(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "start_free_bytes": self.start_free_bytes,
            "end_free_bytes": self.end_free_bytes,
            "allocator_peak_allocated_bytes": self.allocator_peak_allocated_bytes,
            "allocator_peak_reserved_bytes": self.allocator_peak_reserved_bytes,
            "minimum_driver_free_bytes": self.minimum_driver_free_bytes,
            "total_driver_bytes": self.total_driver_bytes,
            "driver_used_high_water_bytes": self.driver_used_high_water_bytes,
            "non_torch_bytes": self.non_torch_bytes,
            "required_bytes": self.required_bytes,
        }


def phase_memory(
    name: str,
    *,
    start_free_bytes: int,
    end_free_bytes: int,
    allocator_peak_allocated_bytes: int,
    allocator_peak_reserved_bytes: int,
    minimum_driver_free_bytes: int,
    total_driver_bytes: int,
) -> PhaseMemory:
    """Build a phase observation; caller supplies synchronized device counters."""
    if minimum_driver_free_bytes < 0 or minimum_driver_free_bytes > total_driver_bytes:
        raise ValueError("minimum driver free must be within total device memory")
    if allocator_peak_allocated_bytes < 0 or allocator_peak_reserved_bytes < 0:
        raise ValueError("allocator peaks must be non-negative")
    return PhaseMemory(
        name=name,
        start_free_bytes=int(start_free_bytes),
        end_free_bytes=int(end_free_bytes),
        allocator_peak_allocated_bytes=int(allocator_peak_allocated_bytes),
        allocator_peak_reserved_bytes=int(allocator_peak_reserved_bytes),
        minimum_driver_free_bytes=int(minimum_driver_free_bytes),
        total_driver_bytes=int(total_driver_bytes),
    )


@dataclass(frozen=True)
class ResidentBudget:
    free_bytes: int
    total_vram_bytes: int
    packed_model_bytes: int
    kv_bytes: int
    gdn_state_bytes: int
    page_table_bytes: int
    graph_reserve_bytes: int
    peak_load_scratch_bytes: int
    safety_bytes: int
    phases: tuple[PhaseMemory, ...] = ()

    @property
    def required_bytes(self) -> int:
        static = sum(
            (
                self.packed_model_bytes,
                self.kv_bytes,
                self.gdn_state_bytes,
                self.page_table_bytes,
                self.graph_reserve_bytes,
                self.peak_load_scratch_bytes,
                self.safety_bytes,
            )
        )
        observed = max((phase.required_bytes for phase in self.phases), default=0)
        return max(static, observed)

    @property
    def fits(self) -> bool:
        return self.required_bytes <= self.free_bytes

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "free_bytes": self.free_bytes,
            "total_vram_bytes": self.total_vram_bytes,
            "packed_model_bytes": self.packed_model_bytes,
            "kv_bytes": self.kv_bytes,
            "gdn_state_bytes": self.gdn_state_bytes,
            "page_table_bytes": self.page_table_bytes,
            "graph_reserve_bytes": self.graph_reserve_bytes,
            "peak_load_scratch_bytes": self.peak_load_scratch_bytes,
            "safety_bytes": self.safety_bytes,
            "required_bytes": self.required_bytes,
            "fits": self.fits,
            "phases": [phase.as_dict() for phase in self.phases],
            "observed_required_bytes": max((phase.required_bytes for phase in self.phases), default=0),
        }


def required_phase_bytes(phases: Iterable[PhaseMemory], safety_bytes: int = 0) -> int:
    """Worst observed driver high-water plus explicit safety, without double counting."""
    return max((phase.required_bytes for phase in phases), default=0) + int(safety_bytes)


def _gguf_payload_bytes(model_path: str) -> tuple[int, int]:
    """Return (all packed tensor bytes, largest tensor bytes) from GGUF headers."""
    from freetoken.models.gguf.reader import _reader, gguf_shard_paths

    shards = gguf_shard_paths(model_path)
    total = 0
    largest = 0
    import gguf

    for shard in shards:
        for tensor in _reader(shard).tensors:
            shape = [int(dim) for dim in tensor.shape]
            block, type_size = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
            fastest = shape[0]
            if fastest % block:
                raise ValueError(
                    f"{tensor.name}: fastest dimension {fastest} is not a multiple of {block}"
                )
            size = math.prod(shape) // block * type_size
            total += size
            largest = max(largest, size)
    return total, largest


def _total_vram_bytes(free_bytes: int) -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
    except Exception:
        pass
    return free_bytes


def _page_table_width(max_seq_len: int, page_size: int) -> int:
    page_aligned = ((max_seq_len + page_size - 1) // page_size) * page_size
    return ((page_aligned + 31) // 32) * 32


def estimate_gguf_resident_budget(model_path, config, free_bytes: int) -> ResidentBudget:
    """Estimate complete native-resident startup footprint without CUDA allocation."""
    packed_model_bytes, largest_tensor_bytes = _gguf_payload_bytes(model_path)

    from freetoken.kvcache import resolve_pool_class
    from freetoken.kvcache.linear_state_pool import _linear_pool_num_slots, state_pool_bytes

    pool_cls = resolve_pool_class(config.model_config)
    per_page, fixed, page_tokens, _ = pool_cls.kv_cost(config)
    pages = max(1, (int(config.max_seq_len) + page_tokens - 1) // page_tokens)
    kv_bytes = pages * per_page + fixed
    # _adjust_config changes linear models' default radix cache to hybrid_radix after this
    # preflight. Price that final slot geometry without mutating the frozen config.
    linear = config.model_config.linear_attention_group()
    cache_type = getattr(config, "cache_type", "radix")
    if linear is not None and cache_type != "naive":
        max_req = int(config.max_running_req)
        cache_slots = max(4, int(getattr(config, "linear_state_cache_ratio", 2.0) * max_req))
        state_slots = 4 * max_req + cache_slots + 1
    else:
        state_slots = _linear_pool_num_slots(config)
    gdn_state = state_pool_bytes(config, num_slots=state_slots)
    page_table = (
        (int(config.max_running_req) + 1)
        * _page_table_width(int(config.max_seq_len), int(config.page_size))
        * 4
    )
    peak_scratch = max(MIN_LOAD_SCRATCH_BYTES, largest_tensor_bytes)
    total_vram = _total_vram_bytes(int(free_bytes))
    safety = max(MIN_SAFETY_BYTES, int(total_vram * 0.08))
    return ResidentBudget(
        free_bytes=int(free_bytes),
        total_vram_bytes=total_vram,
        packed_model_bytes=packed_model_bytes,
        kv_bytes=kv_bytes,
        gdn_state_bytes=int(gdn_state),
        page_table_bytes=page_table,
        graph_reserve_bytes=GRAPH_RESERVE_BYTES,
        peak_load_scratch_bytes=peak_scratch,
        safety_bytes=safety,
    )


def resolve_gguf_moe_backend(config, free_bytes: int) -> Literal["fused", "offload"]:
    """Resolve GGUF auto mode; callers handle explicit fused failure details."""
    if getattr(config.model_config, "moe_weight_format", None) != "gguf":
        return "offload"
    budget = estimate_gguf_resident_budget(config.model_path, config, int(free_bytes))
    return "fused" if budget.fits else "offload"


__all__ = [
    "PhaseMemory",
    "ResidentBudget",
    "estimate_gguf_resident_budget",
    "phase_memory",
    "required_phase_bytes",
    "resolve_gguf_moe_backend",
]
