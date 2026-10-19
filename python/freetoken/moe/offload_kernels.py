from __future__ import annotations

import os
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from flashlib.kernels.slot_cache import lru_ensure

# Hybrid backend: which of a step's missing experts to fetch (when capped below the miss
# count). "recency" (default) fetches the experts most-recently active before this step
# (LRU on the expert -> prioritizes recurring misses, lowering the steady miss rate);
# "lowest_id" fetches the smallest expert ids (the original, routing-blind heuristic).
_HYBRID_FETCH_BY_RECENCY = (
    os.getenv("FREETOKEN_HYBRID_FETCH", "recency").strip().lower() != "lowest_id"
)


def ensure_experts(cache, layer_id: int, expert_ids: torch.Tensor) -> None:
    """Make this layer's routed experts resident; rewrite ``expert_ids`` to slot ids.

    Delegates to flashlib's slot cache. ``id_base`` maps this layer's expert ids into the
    flat ``layer * num_experts + expert`` space the cache indexes by, and maps
    ``src_indices`` back, so ``copy_missing`` still resolves against this layer's own host
    tensor. ``out_indices`` aliases the input, preserving the in-place rewrite every
    downstream GEMM depends on.
    """
    id_of_slot, usage = cache.pool_state(layer_id)
    lru_ensure(
        expert_ids,
        cache.slot_for_id.view(-1),
        id_of_slot,
        usage,
        cache.step,
        expert_ids,
        cache.src_indices,
        cache.evict_slots,
        cache.num_indices,
        stats=cache.lru_stats[layer_id] if cache.collect_stats else None,
        id_base=layer_id * cache.num_experts,
    )


class HybridRoutes(NamedTuple):
    """One layer's routes split between the GPU and the CPU, each [bs, top_k]."""

    gpu_slots: torch.Tensor  # cache slot of each route; a CPU route is -1, or 0 with clamp
    gpu_w: torch.Tensor  # router weights, 0 for a CPU route
    cpu_ids: torch.Tensor  # expert ids, -1 for a GPU route


def ensure_experts_hybrid(
    cache,
    layer_id: int,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    max_fetch: int,
    fetch_fraction: float = 0.0,
    clamp: bool = False,
    fetch_cap: int | None = None,
) -> HybridRoutes:
    """Capped-fetch variant of ``ensure_experts`` (hybrid backend).

    Identical LRU bookkeeping, but only the first ``max_fetch`` of this step's missing
    experts are given a slot and scheduled for copy; the overflow misses stay
    non-resident and their ``expert_ids`` positions are rewritten to ``-1`` (compute on
    the CPU). ``fetch_fraction`` > 0 replaces the fixed cap with the bandwidth-matched
    split (fraction = pcie_bw / cpu_bw): fetch ~fraction of the step's misses, rounded to
    the integer that makes the PCIe fetch and the CPU overflow compute finish closest to
    together. ``num_indices`` = capped fetch count (copy_missing); ``num_missing_full`` =
    pre-cap miss count (stats).

    The same launch splits the routes: ``expert_ids`` becomes the GPU slots in place, and
    ``clamp`` gives CPU routes slot 0 instead of -1 for GEMM kernels that read every slot.
    ``fetch_cap`` bounds the fetch count after ``max_fetch`` / ``fetch_fraction`` (a fixed-size staging buffer)."""
    assert expert_ids.is_cuda, "the hybrid ensure kernel runs on the GPU"
    # Q16 fixed point: the kernel caps in integers.
    frac_q16 = min(1 << 16, max(0, round(fetch_fraction * (1 << 16))))
    weights = weights.contiguous()
    routes = HybridRoutes(expert_ids, torch.empty_like(weights), torch.empty_like(expert_ids))
    cap = cache.num_experts if fetch_cap is None else fetch_cap
    _ensure_experts_hybrid_gpu(cache, layer_id, expert_ids, weights, routes, clamp, max_fetch, frac_q16, cap)
    return routes


def prefill_hit_compact(cache, layer_id: int) -> None:
    """Compact this layer's cache-resident experts into gather indices, device-side.

    hit = slot_for_id[layer_id][e] >= the pool's staging threshold (the double buffer owns the
    slots below, so those bytes are volatile within a prefill chunk and classify
    as miss). Writes fixed-shape ``_prefill_hit_dst``/``_prefill_hit_src`` (buffer
    row / pool-local slot) and the count into ``_prefill_hit_num``; one launch on the
    current stream, no host sync. Safe against the concurrent buffer invalidation
    on the copy stream: that only rewrites entries already below the threshold."""
    num_experts = cache.num_experts
    _prefill_hit_compact_kernel[(1,)](
        cache.slot_for_id[layer_id],
        cache._prefill_hit_dst,
        cache._prefill_hit_src,
        cache._prefill_hit_num,
        cache._pool_stable_from[cache.pool_of_layer[layer_id]],
        num_experts,
        BLOCK=triton.next_power_of_2(num_experts),
    )


def materialize_layer(cache, layer_id: int) -> None:
    if len(cache.pool_sizes) == 1:
        _materialize_layer_gpu(cache, layer_id)
        return
    # staging buffer 0 may overlap several pools, so it only holds the layer for this forward and registers nothing
    E = cache.num_experts
    cache._invalidate_prefill_buffer(0, layer_id)
    rows = torch.arange(E, dtype=torch.int32, device=cache.device)
    cache.evict_slots[:E].copy_(rows)
    cache.src_indices[:E].copy_(rows)
    cache.num_indices.fill_(E)


def reset_cache(cache) -> None:
    _reset_cache_gpu(cache)






def _ensure_experts_hybrid_gpu(
    cache,
    layer_id: int,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    routes: HybridRoutes,
    clamp: bool,
    max_fetch: int,
    frac_q16: int,
    fetch_cap: int,
) -> None:
    id_of_slot, usage = cache.pool_state(layer_id)
    num_slots = id_of_slot.numel()
    block_e = triton.next_power_of_2(cache.num_experts)
    block_c = triton.next_power_of_2(num_slots)
    num_warps = 8 if block_c >= 2048 else 4
    _ensure_experts_hybrid_kernel[(1,)](
        expert_ids,
        cache.slot_for_id,
        id_of_slot,
        usage,
        cache.step,
        cache.active_mask,
        cache.evict_slots,
        cache.src_indices,
        cache.num_indices,
        cache.num_missing_full,
        cache.expert_recency,
        weights,
        routes.cpu_ids,
        routes.gpu_w,
        *(cache.hybrid_stats if cache.collect_stats else (expert_ids,) * 2),
        layer_id,
        expert_ids.numel(),
        int(max_fetch),
        int(frac_q16),
        int(fetch_cap),
        cache.num_experts,
        num_slots,
        BLOCK_E=block_e,
        BLOCK_C=block_c,
        BY_RECENCY=_HYBRID_FETCH_BY_RECENCY,
        CLAMP=clamp,
        STATS=cache.collect_stats,
        num_warps=num_warps,
    )


def _materialize_layer_gpu(cache, layer_id: int) -> None:
    block = triton.next_power_of_2(max(cache.num_experts, cache.cache_size))
    _materialize_layer_kernel[(1,)](
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.evict_slots,
        cache.src_indices,
        cache.num_indices,
        layer_id,
        cache.num_experts,
        cache.cache_size,
        BLOCK=block,
    )


def _reset_cache_gpu(cache) -> None:
    block = 256
    total_ids = cache.num_layers * cache.num_experts
    grid = (triton.cdiv(max(total_ids, cache.cache_size), block),)
    _reset_cache_kernel[grid](
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.active_mask,
        cache.num_indices,
        total_ids,
        cache.num_experts,
        cache.cache_size,
        BLOCK=block,
    )


@triton.jit
def _reset_cache_kernel(
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    active_mask_ptr,
    num_indices_ptr,
    total_ids: tl.constexpr,
    num_experts: tl.constexpr,
    cache_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(slot_for_id_ptr + off, -1, mask=off < total_ids)
    tl.store(id_of_slot_ptr + off, -1, mask=off < cache_size)
    tl.store(usage_ptr + off, 0, mask=off < cache_size)
    tl.store(active_mask_ptr + off, 0, mask=off < num_experts)
    if tl.program_id(0) == 0:
        tl.store(step_ptr, 0)
        tl.store(num_indices_ptr, 0)


@triton.jit
def _materialize_layer_kernel(
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    evict_slots_ptr,
    src_indices_ptr,
    num_indices_ptr,
    layer_id: tl.constexpr,
    num_experts: tl.constexpr,
    cache_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.arange(0, BLOCK)
    expert_mask = off < num_experts
    slot_mask = off < cache_size
    slot = off

    base = layer_id * num_experts
    old_id = tl.load(id_of_slot_ptr + slot, mask=slot_mask, other=-1)
    # Flat ids make "belongs to this layer" a range check instead of a field compare.
    same_layer = slot_mask & (old_id >= base) & (old_id < base + num_experts)
    tl.store(id_of_slot_ptr + slot, -1, mask=same_layer)
    tl.store(usage_ptr + slot, 0, mask=same_layer)

    old_valid = expert_mask & (old_id >= 0) & (~same_layer)
    tl.store(slot_for_id_ptr + old_id, -1, mask=old_valid)

    step = tl.load(step_ptr) + 1
    tl.store(step_ptr, step)
    tl.store(id_of_slot_ptr + slot, base + off, mask=expert_mask)
    tl.store(slot_for_id_ptr + base + off, slot, mask=expert_mask)
    tl.store(usage_ptr + slot, step, mask=expert_mask)
    tl.store(evict_slots_ptr + off, slot, mask=expert_mask)
    tl.store(src_indices_ptr + off, off, mask=expert_mask)  # layer-local row
    tl.store(num_indices_ptr, num_experts)




@triton.jit(do_not_specialize=["layer_id", "num_active", "max_fetch", "fetch_frac_q16", "fetch_cap", "cache_size", "id_of_slot_ptr", "usage_ptr"])
def _ensure_experts_hybrid_kernel(
    expert_ids_ptr,
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    active_mask_ptr,
    evict_slots_ptr,
    src_indices_ptr,
    num_indices_ptr,
    num_missing_full_ptr,
    expert_recency_ptr,
    weights_ptr,
    cpu_ids_ptr,
    gpu_w_ptr,
    stats_ptr,
    stats_layer_ptr,
    layer_id,
    num_active,
    max_fetch,
    fetch_frac_q16,
    fetch_cap,
    num_experts: tl.constexpr,
    cache_size,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BY_RECENCY: tl.constexpr,
    CLAMP: tl.constexpr,
    STATS: tl.constexpr,
):
    """Capped-fetch timestamp-LRU (hybrid backend).

    Same as ``_ensure_experts_lru_v2_kernel`` but only ``min(num_missing, max_fetch)``
    missing experts are evicted-into / scheduled for copy; the overflow misses stay
    non-resident, so Phase 3 rewrites their positions to -1 (the layer computes those on
    the CPU). ``fetch_frac_q16`` > 0 (Q16 fixed point) replaces the fixed cap with the
    bandwidth-matched split ``~frac * num_missing`` (see the Phase-1 comment), computed
    in-kernel because ``num_missing`` only exists device-side (CUDA graph). ``num_indices``
    = the capped fetch count (copy_missing), ``num_missing_full`` = the pre-cap miss count
    (stats).

    Which misses to fetch is the cap policy. ``BY_RECENCY`` (default) fetches the experts
    most-recently active before this step (LRU on the expert, via ``expert_recency``),
    breaking ties toward the lower expert id -- this prioritizes *recurring* misses for
    caching, lowering the steady miss rate. Otherwise the lowest expert ids are fetched
    (``missing_rank``), the original routing-blind heuristic."""
    step = tl.load(step_ptr) + 1
    tl.store(step_ptr, step)
    base = layer_id * num_experts

    # ---- Phase 1: active + missing over experts ----
    off_e = tl.arange(0, BLOCK_E)
    e_mask = off_e < num_experts
    is_active = tl.zeros((BLOCK_E,), dtype=tl.int1)
    for i in tl.range(num_active):
        e = tl.load(expert_ids_ptr + i)
        is_active = is_active | (off_e == e)
    tl.store(active_mask_ptr + off_e, is_active.to(tl.int32), mask=e_mask)
    slot = tl.load(slot_for_id_ptr + base + off_e, mask=e_mask, other=-1)
    is_missing = is_active & (slot == -1) & e_mask
    num_missing = tl.sum(is_missing.to(tl.int32))
    # Cap the fetches; the overflow misses are computed on the CPU (left non-resident).
    if fetch_frac_q16 > 0:
        # Bandwidth-matched split (fetch_frac = pcie_bw / cpu_bw): fetch time scales with
        # F * (1 - frac), CPU time with (M - F) * frac; they balance at F = frac * M. Pick
        # the integer neighbor that minimizes the slower (max) side of the overlap.
        lo = (num_missing * fetch_frac_q16) >> 16
        cost_lo = tl.maximum(lo * ((1 << 16) - fetch_frac_q16), (num_missing - lo) * fetch_frac_q16)
        cost_hi = tl.maximum(
            (lo + 1) * ((1 << 16) - fetch_frac_q16), (num_missing - lo - 1) * fetch_frac_q16
        )
        max_fetch = tl.where(cost_lo <= cost_hi, lo, lo + 1)
    is_hit = is_active & (slot >= 0)
    # a pool smaller than num_experts may hold fewer free slots than misses
    num_fetch = tl.minimum(tl.minimum(num_missing, tl.minimum(max_fetch, fetch_cap)), cache_size - tl.sum(is_hit.to(tl.int32)))
    tl.store(num_missing_full_ptr, num_missing.to(tl.int64))
    tl.store(num_indices_ptr, num_fetch.to(tl.int64))
    tl.store(usage_ptr + slot, step, mask=is_hit)

    # Fetch-selection priority: encode (recency desc, id asc) into one strictly-ordered
    # score so argmax has no ties (rec deltas are multiples of num_experts; the id term
    # spans only [0, num_experts), so it can only break exact-recency ties).
    if BY_RECENCY:
        rec = tl.load(expert_recency_ptr + base + off_e, mask=e_mask, other=-1).to(tl.int64)
        score = tl.where(
            is_missing, rec * num_experts + (num_experts - 1 - off_e), -1152921504606846976
        ).to(tl.int64)
    else:
        missing_rank = tl.cumsum(is_missing.to(tl.int32)) - 1

    # ---- Phase 2: evict victims by argmin(usage), only for the capped fetches ----
    if num_fetch > 0:
        off_c = tl.arange(0, BLOCK_C)
        c_mask = off_c < cache_size
        oid = tl.load(id_of_slot_ptr + off_c, mask=c_mask, other=-1)
        u = tl.load(usage_ptr + off_c, mask=c_mask, other=9223372036854775807).to(tl.int64)
        owner_active = c_mask & False
        for i in tl.range(num_active):
            ei = tl.load(expert_ids_ptr + i)
            owner_active = owner_active | (oid == base + ei)
        u = tl.where(owner_active | (~c_mask), 9223372036854775807, u)
        for i in tl.range(num_fetch):
            victim = tl.argmin(u, axis=0).to(tl.int32)
            old_id = tl.sum(tl.where(off_c == victim, oid, 0))
            if old_id >= 0:
                tl.store(slot_for_id_ptr + old_id, -1)
            if BY_RECENCY:
                e = tl.argmax(score, axis=0).to(tl.int32)
                score = tl.where(off_e == e, -1152921504606846976, score)
            else:
                e = tl.sum(tl.where((missing_rank == i) & is_missing, off_e, 0))
            tl.store(id_of_slot_ptr + victim, base + e)
            tl.store(slot_for_id_ptr + base + e, victim)
            tl.store(usage_ptr + victim, step)
            tl.store(evict_slots_ptr + i, victim)
            tl.store(src_indices_ptr + i, e)  # layer-local row
            u = tl.where(off_c == victim, 9223372036854775807, u)

    # ---- Phase 3: rewrite expert_ids -> slot id (hit/fetched) or -1 (overflow -> CPU) ----
    for i in tl.range(num_active):
        e = tl.load(expert_ids_ptr + i)
        s = tl.load(slot_for_id_ptr + base + e)
        on_gpu = s >= 0
        tl.store(cpu_ids_ptr + i, tl.where(on_gpu, -1, e))
        tl.store(gpu_w_ptr + i, tl.where(on_gpu, tl.load(weights_ptr + i), 0.0))
        if CLAMP:
            s = tl.maximum(s, 0)
        tl.store(expert_ids_ptr + i, s)

    # stats_ptr: (missing, fetched, active, calls); stats_layer_ptr: the same per layer, [num_layers, 4]
    if STATS:
        n_active = tl.sum((is_active & e_mask).to(tl.int64))
        tl.store(stats_ptr + 0, tl.load(stats_ptr + 0) + num_missing)
        tl.store(stats_ptr + 1, tl.load(stats_ptr + 1) + num_fetch)
        tl.store(stats_ptr + 2, tl.load(stats_ptr + 2) + n_active)
        tl.store(stats_ptr + 3, tl.load(stats_ptr + 3) + 1)
        row = stats_layer_ptr + layer_id * 4
        tl.store(row + 0, tl.load(row + 0) + num_missing)
        tl.store(row + 1, tl.load(row + 1) + num_fetch)
        tl.store(row + 2, tl.load(row + 2) + n_active)
        tl.store(row + 3, tl.load(row + 3) + 1)

    # Bump every active expert's recency to this step (LRU on the expert): an overflow miss
    # computed on the CPU now ranks high if it recurs, so it gets fetched next time.
    if BY_RECENCY:
        step_vec = tl.zeros((BLOCK_E,), dtype=tl.int64) + step
        tl.store(expert_recency_ptr + base + off_e, step_vec, mask=is_active & e_mask)


@triton.jit(do_not_specialize=["threshold"])
def _prefill_hit_compact_kernel(
    slot_ptr,     # [num_experts] int32: this layer's slot_for_id row
    dst_ptr,      # [num_experts] int32 out: buffer rows (expert ids), compacted
    src_ptr,      # [num_experts] int32 out: pool-local slots, compacted
    num_ptr,      # [1] int64 out: hit count
    threshold,    # first pool-local slot outside the staging buffers
    num_experts,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    lane = offs < num_experts
    slots = tl.load(slot_ptr + offs, mask=lane, other=-1)
    is_hit = lane & (slots >= threshold)
    pos = tl.cumsum(is_hit.to(tl.int32)) - 1
    tl.store(dst_ptr + pos, offs.to(tl.int32), mask=is_hit)
    tl.store(src_ptr + pos, slots, mask=is_hit)
    tl.store(num_ptr, tl.sum(is_hit.to(tl.int64)))
