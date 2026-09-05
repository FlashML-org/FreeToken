"""Pure GPU-memory budget policy shared by startup auto-sizing and runtime rebuild.

No torch/GPU side effects: every function here is integer/byte arithmetic over already-
measured quantities, so it is unit-testable without a device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from freetoken.utils import div_ceil

if TYPE_CHECKING:
    import torch


def expert_bytes_per_slot(sources: dict[str, "list[torch.Tensor]"]) -> int:
    """Bytes one expert slot occupies on GPU: summed row bytes over all banks.

    Each bank source is per-layer ``[num_experts, *row_shape]`` tensors and is
    already TP-sharded upstream, so the per-row byte count is the per-rank slot
    size.
    """
    # marlin/b12x gate_up/down alpha scales are fixed [L*E] residency (do not scale
    # with cache_size), so they are intentionally excluded from the per-slot growth term.
    # The GPU slot cache has one stride per bank, chosen from that bank's largest layer
    # row. See the matching slot-byte calculation in OffloadMoeCache.
    return sum(
        max(layer[0].numel() * layer.element_size() for layer in per_layer)
        for per_layer in sources.values()
    )


@dataclass(frozen=True)
class GeometryPoolPlan:
    layer_ids: tuple[int, ...]
    row_bytes: tuple[int, ...]
    slots: int


def plan_geometry_pool_slots(
    row_bytes_by_layer: list[tuple[int, ...]],
    *,
    legacy_cache_size: int,
    num_experts: int,
    top_k: int,
    max_decode_batch: int,
) -> tuple[GeometryPoolPlan, ...] | None:
    """Partition fixed max-stride bank arenas into exact-geometry decode pools.

    ``legacy_cache_size`` remains the external budget denomination. Each bank owns
    ``legacy_cache_size * max(layer_row_bytes)`` bytes; every planned class must fit
    all bank constraints independently.
    """
    if not row_bytes_by_layer:
        return ()
    num_banks = len(row_bytes_by_layer[0])
    if num_banks == 0 or any(len(row) != num_banks for row in row_bytes_by_layer):
        raise ValueError("every layer must describe the same non-empty bank set")
    if (
        legacy_cache_size <= 0
        or num_experts <= 0
        or top_k <= 0
        or max_decode_batch <= 0
    ):
        raise ValueError(
            "cache size, experts, top_k, and decode batch must be positive"
        )

    grouped: dict[tuple[int, ...], list[int]] = {}
    for layer_id, rows in enumerate(row_bytes_by_layer):
        if any(value <= 0 for value in rows):
            raise ValueError("geometry row bytes must be positive")
        grouped.setdefault(tuple(rows), []).append(layer_id)
    classes = [(rows, tuple(layer_ids)) for rows, layer_ids in grouped.items()]
    budgets = [
        legacy_cache_size * max(rows[bank] for rows in row_bytes_by_layer)
        for bank in range(num_banks)
    ]
    floor = min(num_experts, top_k * max_decode_batch)
    slots = [floor] * len(classes)

    def used(bank: int) -> int:
        return sum(slots[i] * classes[i][0][bank] for i in range(len(classes)))

    if any(used(bank) > budgets[bank] for bank in range(num_banks)):
        return None

    targets = [
        min(len(layer_ids) * num_experts, len(layer_ids) * top_k * max_decode_batch)
        for _, layer_ids in classes
    ]
    caps = [len(layer_ids) * num_experts for _, layer_ids in classes]

    def affordable(index: int) -> bool:
        rows = classes[index][0]
        return all(
            used(bank) + rows[bank] <= budgets[bank] for bank in range(num_banks)
        )

    def fill(limits: list[int]) -> None:
        while True:
            candidates = [
                i for i in range(len(classes)) if slots[i] < limits[i] and affordable(i)
            ]
            if not candidates:
                return
            index = min(candidates, key=lambda i: (slots[i] / limits[i], i))
            slots[index] += 1

    fill(targets)
    fill(caps)
    return tuple(
        GeometryPoolPlan(layer_ids=layer_ids, row_bytes=rows, slots=slots[index])
        for index, (rows, layer_ids) in enumerate(classes)
    )


def net_cache_budget_bytes(
    memory_ratio: float, baseline_free: int, weights_bytes: int, fixed_cache_size: int
) -> int:
    """Net GPU bytes available for the MoE + KV pools: ``memory_ratio`` of the pre-model
    baseline minus weights and fixed (non-paged) cache. The ``(1-memory_ratio)`` remainder
    is the CUDA-graph/activation headroom. Single source of truth for startup auto-sizing
    and the runtime-rebuild fit check."""
    return int(memory_ratio * baseline_free) - weights_bytes - fixed_cache_size


def required_bytes(
    moe_cache_size: int, num_pages: int, per_expert_bytes: int, cache_per_page: int
) -> int:
    """GPU bytes a ``(moe_cache_size, num_pages)`` geometry occupies (MoE slots + KV pages)."""
    return moe_cache_size * per_expert_bytes + num_pages * cache_per_page


def plan_cache_budget(
    budget_bytes: int,
    per_expert_bytes: int,
    cache_per_page: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_pages: int,
    max_slots: int,
) -> tuple[int, int, bool]:
    """Split ``budget_bytes`` MoE-first into (moe_cache_size, num_pages, prefill_overlap).

    ``budget_bytes`` is the net pool for MoE cache + KV cache (caller already subtracted
    weights + fixed_cache_size; the (1-memory_ratio) remainder is the graph headroom).
    Experts greedily fill the budget after reserving ``kv_reserve_pages`` for KV, clamped
    to ``[floor, min(total_experts, max_slots)]`` (floor is ``2*num_experts`` when prefill
    overlap is feasible else ``num_experts``); KV pages take whatever remains.
    """
    assert per_expert_bytes > 0, "per_expert_bytes must be positive"
    assert cache_per_page > 0, "cache_per_page must be positive (owned-KV models unsupported here)"

    hi = min(total_experts, max_slots)
    # Prefill overlap borrows two full expert-layer buffers, so it needs >= 2*num_experts
    # slots; disable it (and lower the floor) if the cap cannot fit that.
    overlap = prefill_overlap and hi >= 2 * num_experts
    lo = 2 * num_experts if overlap else num_experts
    assert hi >= lo, f"slot cap {hi} below the minimum {lo} slots"

    kv_reserve_bytes = kv_reserve_pages * cache_per_page
    # MoE-priority: reserve KV first, then experts greedily take the remaining budget.
    raw = (budget_bytes - kv_reserve_bytes) // per_expert_bytes
    moe_cache_size = max(lo, min(raw, hi))
    # A tiny budget may have forced moe_cache_size below 2*num_experts even with overlap on.
    overlap = overlap and moe_cache_size >= 2 * num_experts

    remaining = budget_bytes - moe_cache_size * per_expert_bytes
    num_pages = max(remaining // cache_per_page, kv_reserve_pages)
    # A tiny budget can floor num_pages at kv_reserve_pages even when ``remaining`` is below
    # the reserve (or negative), yielding a plan that exceeds budget_bytes. Reject here so
    # --moe-cache-auto fails in arithmetic instead of OOMing in a later CUDA allocation.
    total = moe_cache_size * per_expert_bytes + num_pages * cache_per_page
    assert total <= budget_bytes, (
        f"cache budget too small: minimum plan (moe={moe_cache_size} slots, "
        f"kv={num_pages} pages) needs {total} B > budget {budget_bytes} B "
        "(raise memory_ratio, lower kv_reserve_tokens, or free GPU memory)"
    )
    assert num_pages > 1, "not enough memory for KV cache after MoE allocation"
    return moe_cache_size, num_pages, overlap


def resolve_moe_cache_auto(
    *,
    baseline_free: int,
    weights_bytes: int,
    memory_ratio: float,
    cache_per_page: int,
    fixed_cache_size: int,
    per_expert_bytes: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_tokens: int,
    page_size: int,
    quant_format: str,
) -> tuple[int, int, bool]:
    """Resolve --moe-cache-auto into (moe_cache_size, num_pages, prefill_overlap).

    Applies memory_ratio to the persisted pre-model baseline exactly once, then defers
    the MoE-vs-KV split to plan_cache_budget. The (1-memory_ratio) remainder is the
    CUDA-graph/activation headroom (not subtracted here).
    """
    budget_bytes = net_cache_budget_bytes(memory_ratio, baseline_free, weights_bytes, fixed_cache_size)
    max_slots = 992 if quant_format == "nvfp4_marlin" else total_experts
    kv_reserve_pages = div_ceil(kv_reserve_tokens, page_size)
    return plan_cache_budget(
        budget_bytes=budget_bytes,
        per_expert_bytes=per_expert_bytes,
        cache_per_page=cache_per_page,
        num_experts=num_experts,
        total_experts=total_experts,
        prefill_overlap=prefill_overlap,
        kv_reserve_pages=kv_reserve_pages,
        max_slots=max_slots,
    )
