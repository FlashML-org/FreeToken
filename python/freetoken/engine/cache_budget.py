"""Pure GPU-memory budget policy shared by startup auto-sizing and runtime rebuild.

No torch/GPU side effects: every function here is integer/byte arithmetic over already-
measured quantities, so it is unit-testable without a device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.utils import div_ceil

if TYPE_CHECKING:
    import torch


def layer_slot_bytes(sources: dict[str, "list[torch.Tensor]"]) -> list[int]:
    """Bytes one expert slot of each layer occupies on GPU: its row bytes summed over all banks.

    Each bank source is per-layer ``[num_experts, *row_shape]`` tensors and is
    already TP-sharded upstream, so the per-row byte count is the per-rank slot
    size.
    """
    # marlin/b12x gate_up/down alpha scales are fixed [L*E] residency (do not scale
    # with cache_size), so they are intentionally excluded from the per-slot growth term.
    # tensor[0].numel() is the per-row element count (one expert slot); see the matching
    # slot-byte idiom in kvcache/linear_state_pool.py and kvcache/dsv4/v4_pool.py.
    num_layers = len(next(iter(sources.values())))
    return [sum(t[l][0].numel() * t[l].element_size() for t in sources.values()) for l in range(num_layers)]


def expert_bytes_per_slot(sources: dict[str, "list[torch.Tensor]"]) -> int:
    """Bytes one expert slot occupies on GPU, rounded up to the mean over layers when their bank layouts differ."""
    per_layer = layer_slot_bytes(sources)
    return -(-sum(per_layer) // len(per_layer))


def net_cache_budget_bytes(
    memory_ratio: float, baseline_free: int, weights_bytes: int, fixed_cache_size: int
) -> int:
    """Net GPU bytes available for the MoE + KV pools: ``memory_ratio`` of the pre-model
    baseline minus weights and fixed (non-paged) cache. The ``(1-memory_ratio)`` remainder
    is the CUDA-graph/activation headroom. Single source of truth for startup auto-sizing
    and the runtime-rebuild fit check."""
    return int(memory_ratio * baseline_free) - weights_bytes - fixed_cache_size


def prefill_working_bytes(
    tokens: int, *, hidden: int, key_dim: int, value_dim: int, value_heads: int, head_dim: int, chunk: int, itemsize: int
) -> int:
    """GPU bytes the temporaries of a prefill of ``tokens`` tokens hold at their peak, on a model with GatedDeltaNet layers.

    The peak is in one of those layers, where the chunk kernel allocates its output (layers/linear_attn/gdn.py,
    kernel/fla/chunk.py): every term below is a tensor alive there, in elements per token. The count is from tensor
    shapes. Other architectures are not counted here; the caller plans without a reserve for them.
    """
    conv = 2 * key_dim + value_dim
    per_token = (
        4 * hidden  # the layer's input and residual, and the normed input and summed residual made from them
        + conv + value_dim  # in_proj: q, k, v before the conv, and the output gate z
        + conv  # the conv's output
        + conv  # q, k, v copied out of it head by head
        + 2 * key_dim  # q and k l2-normalised
        + value_heads * chunk  # A, the solved chunk-local matrix
        + value_heads * head_dim + value_dim  # w and u
        + 2 * value_dim  # v_new and the output o
    )
    states = div_ceil(tokens, chunk) * value_heads * head_dim * head_dim  # h: the recurrent state entering each kernel chunk
    return (tokens * per_token + states) * itemsize


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
    reserve_bytes: int = 0,
    kv_soft_pages: int = 0,
) -> tuple[int, int, bool]:
    """Split ``budget_bytes`` MoE-first into (moe_cache_size, num_pages, prefill_overlap).

    ``budget_bytes`` is the net pool for MoE cache + KV cache (caller already subtracted
    weights + fixed_cache_size; the (1-memory_ratio) remainder is the graph headroom).
    Experts greedily fill the budget after reserving ``kv_reserve_pages`` for KV, clamped
    to ``[floor, min(total_experts, max_slots)]`` (floor is ``2*num_experts`` when prefill
    overlap is feasible else ``num_experts``); KV pages take whatever remains.

    ``reserve_bytes`` are left out of both caches (the working memory of a prefill chunk).
    KV pages above ``kv_reserve_pages`` pay for them first and expert slots pay the rest;
    neither goes under its floor, so a budget that cannot afford the whole reserve keeps
    less of it rather than failing.

    ``kv_soft_pages`` is a KV floor that cannot fail a plan: KV gets that many pages ahead
    of the expert slots above their floor when the budget has them after the reserve, and
    what it has otherwise, never under ``kv_reserve_pages``.
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
    # Off the budget before the split: experts fill first and KV takes what is left, so the
    # bytes come out of KV's pages above the floor, then out of expert slots. The floors are
    # the slot floor and the KV reserve, or the two pages the plan needs at least.
    floors = lo * per_expert_bytes + max(kv_reserve_pages, 2) * cache_per_page
    budget_bytes -= max(0, min(reserve_bytes, budget_bytes - floors))
    # MoE-priority: reserve KV first, then experts greedily take the remaining budget. A soft
    # floor the budget cannot pay puts the slots at their floor (the clamp below), no lower.
    raw = (budget_bytes - max(kv_reserve_bytes, kv_soft_pages * cache_per_page)) // per_expert_bytes
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
    max_slots: int | None = None,
    reserve_bytes: int = 0,
    kv_soft_tokens: int = 0,
) -> tuple[int, int, bool]:
    """Resolve --moe-cache-auto into (moe_cache_size, num_pages, prefill_overlap).

    ``max_slots`` is the expert kernel's addressable slot limit; the plan never exceeds it.
    ``reserve_bytes`` is plan_cache_budget's: bytes neither cache may take. ``kv_soft_tokens``
    is its soft KV floor, in tokens.

    Applies memory_ratio to the persisted pre-model baseline exactly once, then defers
    the MoE-vs-KV split to plan_cache_budget. The (1-memory_ratio) remainder is the
    CUDA-graph/activation headroom (not subtracted here).
    """
    budget_bytes = net_cache_budget_bytes(memory_ratio, baseline_free, weights_bytes, fixed_cache_size)
    max_slots = total_experts if max_slots is None else min(max_slots, total_experts)
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
        reserve_bytes=reserve_bytes,
        kv_soft_pages=div_ceil(kv_soft_tokens, page_size),
    )
