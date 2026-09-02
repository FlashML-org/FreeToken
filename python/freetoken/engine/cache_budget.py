"""Pure GPU-memory budget policy shared by startup auto-sizing and runtime rebuild.

No torch/GPU side effects: every function here is integer/byte arithmetic over already-
measured quantities, so it is unit-testable without a device.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from freetoken.utils import div_ceil

if TYPE_CHECKING:
    import torch

# HIP ``torch.empty`` of one bank tensor fails well below ``mem_get_info`` free
# (10.21 GiB refused with ~22 GiB free on gfx1201 / ROCm 7.14). The host expert
# pool is already registered shmem; the GPU slot cache only needs active rows.
# NVIDIA does not read these defaults (``max_contiguous_bytes`` stays None).
_HIP_MOE_MAX_BANK_GIB_DEFAULT = 4.0
_HIP_MOE_MAX_BANK_FRAC_DEFAULT = 0.45


def expert_bytes_per_slot(sources: dict[str, "list[torch.Tensor]"]) -> int:
    """Bytes one expert slot occupies on GPU: summed row bytes over all banks.

    Each bank source is per-layer ``[num_experts, *row_shape]`` tensors and is
    already TP-sharded upstream, so the per-row byte count is the per-rank slot
    size.
    """
    # marlin/b12x gate_up/down alpha scales are fixed [L*E] residency (do not scale
    # with cache_size), so they are intentionally excluded from the per-slot growth term.
    # tensor[0].numel() is the per-row element count (one expert slot); see the matching
    # slot-byte idiom in kvcache/linear_state_pool.py and kvcache/dsv4_paged_pool.py.
    return sum(t[0][0].numel() * t[0].element_size() for t in sources.values())


def max_bank_row_bytes(sources: dict[str, "list[torch.Tensor]"]) -> int:
    """Bytes of the largest single-bank expert row (one ``torch.empty`` unit)."""
    if not sources:
        return 0
    return max(t[0][0].numel() * t[0].element_size() for t in sources.values())


def planned_bank_cache_nbytes(
    cache_size: int, sources: dict[str, "list[torch.Tensor]"]
) -> dict[str, int]:
    """``cache_size × row_bytes`` per bank — the contiguous GPU allocs ``set_bank_sources`` issues."""
    return {
        name: cache_size * (layers[0][0].numel() * layers[0].element_size())
        for name, layers in sources.items()
    }


def hip_max_contiguous_bank_bytes(
    free_bytes: int,
    *,
    max_gib: float | None = None,
    frac: float | None = None,
) -> int:
    """Largest contiguous GPU bank tensor HIP is allowed to allocate.

    ``FREETOKEN_HIP_MOE_MAX_BANK_GIB`` (default 4) and
    ``FREETOKEN_HIP_MOE_MAX_BANK_FRAC`` (default 0.45 of current free) cap it.
    4 GiB is exactly 512 DSV4 ``gate_up_packed`` slots (2 × 256 experts) — the
    prefill-overlap floor — and is far below the 10.21 GiB alloc that failed
    with 22 GiB free.
    """
    if max_gib is None:
        raw = os.environ.get("FREETOKEN_HIP_MOE_MAX_BANK_GIB", "").strip()
        max_gib = float(raw) if raw else _HIP_MOE_MAX_BANK_GIB_DEFAULT
    if frac is None:
        raw = os.environ.get("FREETOKEN_HIP_MOE_MAX_BANK_FRAC", "").strip()
        frac = float(raw) if raw else _HIP_MOE_MAX_BANK_FRAC_DEFAULT
    if free_bytes <= 0 or max_gib <= 0 or frac <= 0:
        return 0
    return min(int(free_bytes * frac), int(max_gib * (1 << 30)))


def require_slot_cache_contiguous(
    cache_size: int,
    sources: dict[str, "list[torch.Tensor]"],
    *,
    max_contiguous_bytes: int | None,
) -> None:
    """Raise if any planned GPU bank tensor exceeds ``max_contiguous_bytes``.

    Host banks stay on registered shmem; this only guards the slot-cache
    ``torch.empty``. ``max_contiguous_bytes is None`` is a no-op (NVIDIA).
    """
    if max_contiguous_bytes is None:
        return
    planned = planned_bank_cache_nbytes(cache_size, sources)
    if not planned:
        return
    name, nbytes = max(planned.items(), key=lambda kv: kv[1])
    if nbytes <= max_contiguous_bytes:
        return
    row = max_bank_row_bytes(sources)
    fit = max_contiguous_bytes // row if row else 0
    raise RuntimeError(
        f"MoE GPU slot cache would allocate {nbytes / (1 << 30):.2f} GiB "
        f"contiguous for bank {name!r} (moe_cache_size={cache_size}) but the "
        f"HIP contiguous cap is {max_contiguous_bytes / (1 << 30):.2f} GiB. "
        f"Host expert banks are already registered; the GPU cache only holds "
        f"active expert rows. Pass --moe-cache-size {fit} (or --moe-cache-auto) "
        f"or raise FREETOKEN_HIP_MOE_MAX_BANK_GIB."
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
    max_bank_row_bytes: int = 0,
    max_contiguous_bytes: int | None = None,
) -> tuple[int, int, bool]:
    """Split ``budget_bytes`` MoE-first into (moe_cache_size, num_pages, prefill_overlap).

    ``budget_bytes`` is the net pool for MoE cache + KV cache (caller already subtracted
    weights + fixed_cache_size; the (1-memory_ratio) remainder is the graph headroom).
    Experts greedily fill the budget after reserving ``kv_reserve_pages`` for KV, clamped
    to ``[floor, min(total_experts, max_slots)]`` (floor is ``2*num_experts`` when prefill
    overlap is feasible else ``num_experts``); KV pages take whatever remains.

    ``max_contiguous_bytes`` (HIP) also caps ``moe_cache_size`` so the largest
    single bank tensor (``cache_size * max_bank_row_bytes``) fits. NVIDIA leaves
    it ``None`` and keeps the greedy total-byte plan.
    """
    assert per_expert_bytes > 0, "per_expert_bytes must be positive"
    assert cache_per_page > 0, "cache_per_page must be positive (owned-KV models unsupported here)"

    hi = min(total_experts, max_slots)
    if max_contiguous_bytes is not None and max_bank_row_bytes > 0:
        hi = min(hi, max_contiguous_bytes // max_bank_row_bytes)
    # Prefill overlap borrows two full expert-layer buffers, so it needs >= 2*num_experts
    # slots; disable it (and lower the floor) if the cap cannot fit that.
    overlap = prefill_overlap and hi >= 2 * num_experts
    lo = 2 * num_experts if overlap else num_experts
    if hi < lo:
        raise ValueError(
            f"GPU slot cache: largest bank row is {max_bank_row_bytes} B; "
            f"contiguous cap {max_contiguous_bytes} B allows only {hi} slots "
            f"but need at least {lo}. Lower --moe-cache-size, free VRAM, or raise "
            f"FREETOKEN_HIP_MOE_MAX_BANK_GIB."
        )

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
    max_bank_row_bytes: int = 0,
    max_contiguous_bytes: int | None = None,
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
        max_bank_row_bytes=max_bank_row_bytes,
        max_contiguous_bytes=max_contiguous_bytes,
    )
