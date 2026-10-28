"""Composite cost model + independent per-tier sizing for DSV4 paged KV.

Two pure (device-free) functions:

* ``dsv4_cache_per_page`` collapses the heterogeneous window / compressed /
  indexer / state-ring tiers into ONE bytes-per-P-token number, used only for
  the budget DIVISION (``num_pages = memory // cache_per_page``).
* ``dsv4_pool_sizes`` sizes each tier INDEPENDENTLY from the anchor
  ``full_token = num_pages * P``. The window tier is ``swa_ratio`` of full
  history (``swa_ratio < 1`` -> window pool much smaller than the compressed
  tier), NOT a fixed multiple of window pages.

Byte conventions:
  kv bytes    = head_dim       * 2  (bf16)
  index bytes = index_head_dim * 2  (bf16)
  state bytes = 2*(1+overlap)*head_dim * 4  (fp32);  overlap = (ratio == 4)
  ring_size   = 8 (ratio 4) / 128 (ratio 128)
"""

from __future__ import annotations

import os

from dataclasses import dataclass, field

_BF16_BYTES = 2
_FP32_BYTES = 4


def dsv4_reserved_window_pages(max_running_req: int, radix: bool) -> int:
    """Window pages the sliding pool must always keep for the concurrent working set: each
    running request's decode transients (2 per req + dummy) plus, in radix mode, PER concurrent
    request one locked live-tail page AND a retained (soft-pinned) prompt-end window -- the
    window is 2 pages here because the retention gap page-aligns to a whole extra page at
    P==window==128, so a distinct follow-up per running request can re-lock 2 pages each.
    Shared by the engine's window-floor and the manager's prefill_chunk_budget so both
    reserve the same set."""
    return 2 * (max_running_req + 1) + (3 * max_running_req if radix else 0) + 1


def ring_size_for_ratio(ratio: int) -> int:
    """Compress-state ring slots per window page (non-speculative)."""
    if ratio == 4:
        return 8
    if ratio == 128:
        return 128
    raise ValueError(f"no ring for ratio {ratio} (only 4 / 128)")


def _kv_bytes(args) -> int:
    return args.head_dim * _BF16_BYTES


def _index_bytes(args) -> int:
    return args.index_head_dim * _BF16_BYTES


def _state_bytes(args, ratio: int) -> int:
    overlap = 1 if ratio == 4 else 0
    return 2 * (1 + overlap) * args.head_dim * _FP32_BYTES


def _idx_state_bytes(args) -> int:
    # Indexer compressor (ratio-4 only): same ring geometry as the attention
    # compress-state ring but keyed on ``index_head_dim`` (overlap always True).
    return 2 * 2 * args.index_head_dim * _FP32_BYTES


def dsv4_cache_per_page(args, swa_ratio: float, P: int = 128) -> int:
    """Bytes per P-token across ALL tiers, summed over layers per ratio class.

    Window term scaled by ``swa_ratio``; compressed ``1/ratio``; indexer ``1/4``
    (ratio-4 layers). The compress-state / indexer-state rings are sized off the
    WINDOW pages (``state_slots = n_win_pages * ring_size`` in ``dsv4_pool_sizes``),
    so their per-page contribution is scaled by ``swa_ratio`` too -- matching the
    actual pool, not ``ring_size`` slots on every full page.
    """
    assert P % 1 == 0 and P > 0
    kv_b = _kv_bytes(args)
    idx_b = _index_bytes(args)

    total = 0
    for ratio in tuple(args.compress_ratios)[: args.n_layers]:
        # Window tier exists on EVERY layer (all-sliding), scaled by swa_ratio.
        total += round(swa_ratio * P) * kv_b
        if ratio == 0:
            continue
        # Compressed KV: P//ratio blocks per page.
        total += (P // ratio) * kv_b
        # Indexer KV: P//4 blocks per page (ratio-4 only).
        if ratio == 4:
            total += (P // 4) * idx_b
            # Indexer compress-state ring: its own pool (ring_size=8, fp32), sized off
            # the window pages -> swa_ratio-scaled per full page.
            total += round(swa_ratio * ring_size_for_ratio(4)) * _idx_state_bytes(args)
        # Compress-state ring: ring_size slots per WINDOW page (swa_ratio-scaled), fp32.
        total += round(swa_ratio * ring_size_for_ratio(ratio)) * _state_bytes(args, ratio)
    return int(total)


def dsv4_kv_unit_bytes(args, P: int = 128) -> int:
    """FULL-tier bytes per full-history token: compressed KV + indexer KV + the full->window
    mapping. These tiers scale with the full anchor (``cmp_blocks = full_token // ratio``), so
    the cost is independent of ``swa_ratio``. The window pool and its state rings are NOT here --
    see :func:`dsv4_window_unit_bytes`. This is the ``kv_bytes_per_token`` the cache-status slider
    divides the VRAM budget by."""
    kv_b = _kv_bytes(args)
    idx_b = _index_bytes(args)
    per_page = P * _INT64_BYTES  # full_to_window map: one int64 slot per full token
    for ratio in tuple(args.compress_ratios)[: args.n_layers]:
        if ratio == 0:
            continue
        per_page += (P // ratio) * kv_b  # compressed KV
        if ratio == 4:
            per_page += (P // 4) * idx_b  # indexer KV
    return -(-per_page // P)  # ceil to bytes/token (conservative slider max)


def dsv4_window_unit_bytes(args, P: int = 128) -> int:
    """WINDOW(swa)-tier bytes per window token: the sliding KV pool (every layer) plus the
    compress-state and indexer-state rings (both sized off the window pages,
    ``state_slots = n_win_pages * ring_size``). Independent of ``swa_ratio`` -- the ratio only sets
    how many window tokens exist, not the per-token cost. This is ``swa_bytes_per_token``."""
    kv_b = _kv_bytes(args)
    ratios = tuple(args.compress_ratios)[: args.n_layers]
    per_page = len(ratios) * P * kv_b  # window KV: P slots per page, every layer
    for ratio in ratios:
        if ratio == 0:
            continue
        per_page += ring_size_for_ratio(ratio) * _state_bytes(args, ratio)  # attn ring
        if ratio == 4:
            per_page += ring_size_for_ratio(4) * _idx_state_bytes(args)  # indexer ring
    return -(-per_page // P)  # ceil to bytes/window-token


@dataclass
class DSV4PoolSizes:
    """Per-tier slot counts derived from the budget anchor ``full_token``.

    ``cmp_blocks`` / ``idx_blocks`` / ``state_slots`` are per-layer (``None``
    where the tier does not exist on that layer).
    """

    P: int
    swa_ratio: float
    full_token: int  # num_pages * P  (the budget anchor)
    n_win_slots: int  # global window pool rows (bf16 kv)
    n_win_pages: int  # n_win_slots // P
    cmp_blocks: list[int | None] = field(default_factory=list)
    idx_blocks: list[int | None] = field(default_factory=list)
    state_slots: list[int | None] = field(default_factory=list)
    ring_sizes: list[int | None] = field(default_factory=list)
    # Indexer compress-state ring (ratio-4 layers only; ``None`` elsewhere). Its
    # own pool, distinct from ``state_slots`` (which is the attention ring).
    idx_state_slots: list[int | None] = field(default_factory=list)


def dsv4_pool_sizes(
    num_pages: int, args, swa_ratio: float, P: int = 128, n_win_pages: int | None = None
) -> DSV4PoolSizes:
    full_token = num_pages * P

    # Window tier: swa_ratio of full history rounded UP to a whole page count, OR the caller's
    # explicit page count (the working-set floor is applied exactly ONCE, by the caller, in pages;
    # never by inflating swa_ratio). Capped at the full history.
    if n_win_pages is None:
        raw = round(swa_ratio * full_token)
        n_win_pages = (raw + P - 1) // P
    n_win_pages = min(n_win_pages, num_pages)
    n_win_slots = n_win_pages * P

    cmp_blocks: list[int | None] = []
    idx_blocks: list[int | None] = []
    state_slots: list[int | None] = []
    ring_sizes: list[int | None] = []
    idx_state_slots: list[int | None] = []
    for ratio in tuple(args.compress_ratios)[: args.n_layers]:
        if ratio == 0:
            cmp_blocks.append(None)
            idx_blocks.append(None)
            state_slots.append(None)
            ring_sizes.append(None)
            idx_state_slots.append(None)
            continue
        rs = ring_size_for_ratio(ratio)
        cmp_blocks.append(full_token // ratio)
        idx_blocks.append(full_token // 4 if ratio == 4 else None)
        state_slots.append(n_win_pages * rs)
        ring_sizes.append(rs)
        # Indexer ring (ratio-4 only) is sized like the attention ratio-4 ring.
        idx_state_slots.append(n_win_pages * ring_size_for_ratio(4) if ratio == 4 else None)

    return DSV4PoolSizes(
        P=P,
        swa_ratio=swa_ratio,
        full_token=full_token,
        n_win_slots=n_win_slots,
        n_win_pages=n_win_pages,
        cmp_blocks=cmp_blocks,
        idx_blocks=idx_blocks,
        state_slots=state_slots,
        ring_sizes=ring_sizes,
        idx_state_slots=idx_state_slots,
    )


_INT64_BYTES = 8


def dsv4_pool_bytes(sizes: DSV4PoolSizes, args, n_scratch: int = 1) -> int:
    """Exact bytes a ``DSV4PagedKVCache`` built from ``sizes`` allocates (mirror of ``total_bytes``
    + the full_to_window mapping), including the scratch/sentinel rows (n_scratch per cmp/idx pool,
    +1 ring scratch row, +1 mapping sentinel row)."""
    kv_b = _kv_bytes(args)
    idx_b = _index_bytes(args)
    ratios = tuple(args.compress_ratios)[: args.n_layers]

    total = len(ratios) * sizes.n_win_slots * kv_b  # window pool, every layer
    total += (sizes.full_token + 1) * _INT64_BYTES  # full_to_window (+ sentinel row)
    for L, ratio in enumerate(ratios):
        if ratio == 0:
            continue
        total += (sizes.cmp_blocks[L] + n_scratch) * kv_b
        total += (sizes.state_slots[L] + 1) * _state_bytes(args, ratio)
        if ratio == 4:
            total += (sizes.idx_blocks[L] + n_scratch) * idx_b
            total += (sizes.idx_state_slots[L] + 1) * _idx_state_bytes(args)
    return int(total)


def dsv4_solve_num_pages(
    available_bytes: int,
    args,
    swa_ratio: float,
    floor_win_pages: int,
    P: int = 128,
    n_scratch: int = 1,
) -> DSV4PoolSizes:
    """Largest budget-respecting pool: max ``num_pages`` with exact
    ``dsv4_pool_bytes(sizes(num_pages, win=max(floor, ceil(r*num)))) <= available_bytes``.

    The window floor is honored in PAGES (never by inflating ``swa_ratio``) and the total is
    byte-checked: at small budgets the window pins at ``floor_win_pages`` and the full/cmp/idx
    anchor SHRINKS to fit, instead of every tier inflating past the budget. Raises ``ValueError``
    when even the minimal pool does not fit.
    """
    def _sizes(num: int) -> DSV4PoolSizes:
        win = max(floor_win_pages, (round(swa_ratio * num * P) + P - 1) // P)
        return dsv4_pool_sizes(num, args, swa_ratio, P=P, n_win_pages=win)

    lo = max(floor_win_pages, 2)  # full history must at least cover the window working set
    if dsv4_pool_bytes(_sizes(lo), args, n_scratch) > available_bytes:
        raise ValueError(
            f"DSV4 KV budget {available_bytes} bytes cannot fit the minimal pool "
            f"({lo} pages incl. the window working-set floor {floor_win_pages}); "
            "raise memory_ratio or lower max_running_req/max_seq_len"
        )
    hi = max(lo, available_bytes // max(1, dsv4_cache_per_page(args, 0.0, P)))
    while dsv4_pool_bytes(_sizes(hi), args, n_scratch) <= available_bytes:
        hi *= 2  # cheap upper bracket (cache_per_page(0.0) undercounts the window term)
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if dsv4_pool_bytes(_sizes(mid), args, n_scratch) <= available_bytes:
            lo = mid
        else:
            hi = mid
    return _sizes(lo)


_AUTO_KV_SLACK_BYTES = 2 << 30  # absorbs plan-vs-measured drift (observed ~265MiB) and leaves a usable pool


def dsv4_auto_cost_model(args, swa_ratio, floor_win_pages, P=128, n_scratch=1):
    """Affine (cache_per_page, fixed_cache_size, min_reserve_tokens) for the MoE-first auto planner:
    exact marginal per-page cost across all tiers (+ the full_to_window mapping), a fixed intercept
    anchored at the minimal viable pool, and a reserve floor covering the window working set plus a
    slack absorbing plan-vs-measured drift. Conservative at the shipped swa_ratio; an extreme
    swa_ratio can dip <1% under exact (harmless -- num_pages is re-solved byte-exactly from
    measured memory)."""
    per_page = dsv4_cache_per_page(args, swa_ratio, P) + P * _INT64_BYTES
    n0 = max(floor_win_pages, 2)
    win0 = max(floor_win_pages, (round(swa_ratio * n0 * P) + P - 1) // P)
    base = dsv4_pool_bytes(
        dsv4_pool_sizes(n0, args, swa_ratio, P=P, n_win_pages=win0), args, n_scratch
    )
    slack_pages = -(-_AUTO_KV_SLACK_BYTES // per_page)
    min_reserve_tokens = (n0 + slack_pages) * P
    return per_page, max(0, base - n0 * per_page), min_reserve_tokens


__all__ = [
    "DSV4PoolSizes",
    "dsv4_auto_cost_model",
    "dsv4_cache_per_page",
    "dsv4_kv_unit_bytes",
    "dsv4_pool_bytes",
    "dsv4_pool_sizes",
    "dsv4_reserved_window_pages",
    "dsv4_solve_num_pages",
    "dsv4_window_unit_bytes",
    "ring_size_for_ratio",
]


# ---- config-facing sizing (the engine/pool speak EngineConfig; the functions above are
# geometry-only). Window:full decoupling: the window tier is all-sliding (only the last 128
# positions are read), so it needs only ~swa_ratio of full history; cmp/idx tiers stay sized
# to the FULL history. The ratio lives on the config (a runtime rebuild can change it).


def _dsv4_swa_ratio(config) -> float:
    """The live DSV4 window/full ratio from the serving config (config.swa_full_tokens_ratio;
    a runtime rebuild can change it)."""
    return float(config.swa_full_tokens_ratio)


def _dsv4_window_floor_pages(config, P: int) -> int:
    """Minimum window pages = the live sliding working set the window pool must always hold:
    one prefill chunk's reach (capped at 8 pages == 1024 tok; chunked prefill bounds the rest)
    + each running request's 128-tail (<=2 pages mid-boundary, concurrency-scaled) + one
    radix-locked live tail page per concurrent cached prompt + the reserved dummy. This is the
    only HARD floor on DSV4 KV sizing -- the full-history (cmp/idx) capacity above it is purely
    memory-derived, and a request longer than it is gracefully gated by ``available_size``."""
    prefill_reach_pages = (config.max_seq_len + P - 1) // P
    # DSV4 config resolution rewrites cache_type to "swa_radix" (engine._adjust_dsv4_config),
    # so the radix term must key on "not naive" -- `== "radix"` was always False here.
    radix = config.cache_type != "naive"
    return min(prefill_reach_pages, 8) + dsv4_reserved_window_pages(config.max_running_req, radix)


def _dsv4_pool_sizes(config, num_pages: int, num_swa_pages: int | None = None):
    # P (the window-page size) is the sliding window (128), the radix block key -- independent of
    # the generic page_size. num_pages (P units, PHYSICAL incl the dummy) is the budget anchor;
    # full_token = num_pages * P. Window sizing precedence: an explicit num_swa_pages (validate's
    # target, usable pages) > config.swa_num_pages_override (a pinned window) > swa_ratio x full.
    P = config.model_config.dsv4_args.window_size
    swa_ratio = _dsv4_swa_ratio(config)
    # Test hook DSV4_FORCE_SMALL_POOL: shrink the budget anchor so a multi-prompt workload OVERFLOWS
    # the pool and exercises the bounded-cache (radix LRU) eviction path. BYPASSES the working-set
    # floor by design (the point is a too-small pool); window = ceil(ratio * num), never inflated.
    override = os.environ.get("DSV4_FORCE_SMALL_POOL")
    if override:
        num_pages = max(2, int(override))  # in 128-pages; keep >=2 so the dummy page + 1 fit
        return dsv4_pool_sizes(
            num_pages=num_pages, args=config.model_config.dsv4_args,
            swa_ratio=swa_ratio, P=P,
        )

    floor_pages = _dsv4_window_floor_pages(config, P)
    target = (
        num_swa_pages if num_swa_pages is not None
        else config.swa_num_pages_override
    )
    if target is not None:
        # Absolute window: `target` usable pages + 1 dummy, floored and capped at the full anchor.
        win = min(num_pages, max(floor_pages, int(target) + 1))
    else:
        # Default: ratio x full, rounded UP to whole window pages (floor applied ONCE, in pages).
        win = max(floor_pages, (round(swa_ratio * num_pages * P) + P - 1) // P)
    return dsv4_pool_sizes(
        num_pages=num_pages,
        args=config.model_config.dsv4_args,
        swa_ratio=swa_ratio,
        P=P,
        n_win_pages=win,
    )
