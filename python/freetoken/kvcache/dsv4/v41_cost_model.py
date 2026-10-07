"""Cost model + per-tier sizing for the DSV41 paged KV pool (DeepSeek-V4.1).

Same shape as ``v4_cost_model``: one bytes-per-P-token number for the budget division, and
independent per-tier sizing from the anchor ``full_token = num_pages * P``. The tiers:

* window pool, every layer          -- ``swa_ratio`` of the full history, packed fp8 rows
* main KV pool, per kv source       -- ``full_token // ratio`` packed fp4 rows
* index-key pool, per kv source     -- ``full_token // ratio`` packed fp4 rows
* compress-state ring, ratio>1 sources -- ``ring_size`` fp32 ``kv|score`` slots per WINDOW page

Sizing reads ``ModelConfig.dsv41_args`` by attribute, like ``v4_cost_model`` reads ``dsv4_args``:
the kvcache package never imports the model package. All byte widths come from the tier row formats
below, so there is exactly one place that knows what a row costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .v41_row_format import FP4_E4M3_B16, FP4_E8M0_B32, FP8_E8M0_B32
from .window_tier import reserved_window_pages

_INT64_BYTES = 8
_FP32_BYTES = 4

WIN_FMT = FP8_E8M0_B32
MAIN_FMT = FP4_E4M3_B16
IDX_FMT = FP4_E8M0_B32


def _win_row_bytes(args) -> int:
    return WIN_FMT.row_bytes(args.head_dim)


def _main_row_bytes(args) -> int:
    return MAIN_FMT.row_bytes(args.head_dim)


def _idx_row_bytes(args) -> int:
    return IDX_FMT.row_bytes(args.index_head_dim)


def _state_bytes(args) -> int:
    """One compress-state ring slot: fp32 ``kv | score`` of the latent width."""
    return 2 * args.head_dim * _FP32_BYTES


def private_window_layer_ids(args) -> tuple[int, ...]:
    """Layers whose window KV is REQUEST-PRIVATE: kept in a per-page-table-row ring of ``window``
    slots (slot = row * window + pos % window) instead of the shared, page-bound window pool, so
    it never lands in radix-shared pages. The decoder layers under Decoder SWA Bounded Replay:
    their KV is recomputed per request over the prompt's last window (positions before it are
    never computed), so a shared page could hold rows another request never wrote. Exact mode
    computes every position and keeps the decoder on the shared pool."""
    if args.swa_decoder_replay == "exact":
        return ()
    return tuple(range(args.decoder_start_layer, args.n_layers))


@dataclass
class DSV41PoolSizes:
    """Per-tier row counts derived from the budget anchor ``full_token``; the global tiers are
    keyed by kv-source layer id."""

    P: int
    swa_ratio: float
    full_token: int
    n_win_slots: int
    n_win_pages: int
    main_rows: dict[int, int] = field(default_factory=dict)
    idx_rows: dict[int, int] = field(default_factory=dict)
    state_slots: dict[int, int] = field(default_factory=dict)  # ratio>1 sources only


def dsv41_pool_sizes(
    num_pages: int, args, swa_ratio: float, P: int, n_win_pages: int | None = None
) -> DSV41PoolSizes:
    full_token = num_pages * P
    if n_win_pages is None:
        n_win_pages = (round(swa_ratio * full_token) + P - 1) // P
    n_win_pages = min(n_win_pages, num_pages)
    sizes = DSV41PoolSizes(P=P, swa_ratio=swa_ratio, full_token=full_token, n_win_slots=n_win_pages * P, n_win_pages=n_win_pages)
    for src in args.backbone_kv_sources:
        ratio = args.compress_ratios[src]
        sizes.main_rows[src] = full_token // ratio
        sizes.idx_rows[src] = full_token // ratio
        if ratio > 1:
            sizes.state_slots[src] = n_win_pages * ratio  # ring size == ratio
    return sizes


def dsv41_pool_bytes(sizes: DSV41PoolSizes, args, n_scratch: int = 1) -> int:
    """Exact bytes a ``DSV41PagedKVCache`` built from ``sizes`` allocates (mirror of ``total_bytes``)."""
    n_private = len(private_window_layer_ids(args))
    total = (args.n_layers - n_private) * sizes.n_win_slots * _win_row_bytes(args)
    total += n_private * n_scratch * sizes.P * _win_row_bytes(args)  # per-request rings
    total += (sizes.full_token + 1) * _INT64_BYTES  # full_to_window (+ sentinel row)
    for src in args.backbone_kv_sources:
        total += (sizes.main_rows[src] + n_scratch) * _main_row_bytes(args)
        total += (sizes.idx_rows[src] + n_scratch) * _idx_row_bytes(args)
        if src in sizes.state_slots:
            total += (sizes.state_slots[src] + 1) * _state_bytes(args)
    return int(total)


def dsv41_cache_per_page(args, swa_ratio: float, P: int) -> int:
    """Marginal bytes per P-token page across all tiers (window scaled by ``swa_ratio``)."""
    n_shared = args.n_layers - len(private_window_layer_ids(args))
    total = n_shared * round(swa_ratio * P) * _win_row_bytes(args)
    for src in args.backbone_kv_sources:
        ratio = args.compress_ratios[src]
        total += (P // ratio) * (_main_row_bytes(args) + _idx_row_bytes(args))
        if ratio > 1:
            total += round(swa_ratio * ratio) * _state_bytes(args)
    return int(total)


def dsv41_kv_unit_bytes(args, P: int) -> int:
    """FULL-tier bytes per full-history token (main + index pools + the mapping); the slider's
    ``kv_bytes_per_token``. 890 B/token for DeepSeek-V4.1-Flash before the mapping."""
    per_page = P * _INT64_BYTES
    for src in args.backbone_kv_sources:
        per_page += (P // args.compress_ratios[src]) * (_main_row_bytes(args) + _idx_row_bytes(args))
    return -(-per_page // P)


def dsv41_window_unit_bytes(args, P: int) -> int:
    """WINDOW-tier bytes per window token (sliding KV on every shared-window layer + the state rings)."""
    per_page = (args.n_layers - len(private_window_layer_ids(args))) * P * _win_row_bytes(args)
    for src in args.backbone_kv_sources:
        ratio = args.compress_ratios[src]
        if ratio > 1:
            per_page += ratio * _state_bytes(args)
    return -(-per_page // P)


def dsv41_solve_num_pages(
    available_bytes: int, args, swa_ratio: float, floor_win_pages: int, P: int, n_scratch: int = 1
) -> DSV41PoolSizes:
    """Largest budget-respecting pool; the window floor is honored in pages and the total is
    byte-checked. Raises ``ValueError`` when even the minimal pool does not fit."""

    def _sizes(num: int) -> DSV41PoolSizes:
        win = max(floor_win_pages, (round(swa_ratio * num * P) + P - 1) // P)
        return dsv41_pool_sizes(num, args, swa_ratio, P, n_win_pages=win)

    lo = max(floor_win_pages, 2)
    if dsv41_pool_bytes(_sizes(lo), args, n_scratch) > available_bytes:
        raise ValueError(
            f"DSV41 KV budget {available_bytes} bytes cannot fit the minimal pool ({lo} pages incl. the "
            f"window working-set floor {floor_win_pages}); raise memory_ratio or lower max_running_req/max_seq_len"
        )
    hi = max(lo, available_bytes // max(1, dsv41_cache_per_page(args, 0.0, P)))
    while dsv41_pool_bytes(_sizes(hi), args, n_scratch) <= available_bytes:
        hi *= 2
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if dsv41_pool_bytes(_sizes(mid), args, n_scratch) <= available_bytes:
            lo = mid
        else:
            hi = mid
    return _sizes(lo)


def dsv41_auto_cost_model(args, swa_ratio: float, floor_win_pages: int, P: int, n_scratch: int = 1):
    """Affine ``(cache_per_page, fixed_cache_size, min_reserve_tokens)`` for the MoE-first auto planner."""
    per_page = dsv41_cache_per_page(args, swa_ratio, P) + P * _INT64_BYTES
    n0 = max(floor_win_pages, 2)
    win0 = max(floor_win_pages, (round(swa_ratio * n0 * P) + P - 1) // P)
    base = dsv41_pool_bytes(dsv41_pool_sizes(n0, args, swa_ratio, P, n_win_pages=win0), args, n_scratch)
    # The engine combines this structural floor with the configured KV reserve.
    return per_page, max(0, base - n0 * per_page), n0 * P


# ---- config-facing sizing (EngineConfig in, sizes out) ----


def _dsv41_swa_ratio(config) -> float:
    return float(config.swa_full_tokens_ratio)


def _dsv41_window_floor_pages(config, args) -> int:
    """Minimum window pages: one prefill chunk's reach (capped at 8 pages) + the running requests'
    working set (see ``reserved_window_pages``)."""
    P = args.window_size
    prefill_reach_pages = (config.max_seq_len + P - 1) // P
    radix = config.cache_type != "naive"
    return min(prefill_reach_pages, 8) + reserved_window_pages(config.max_running_req, radix)


def _dsv41_pool_sizes(config, num_pages: int, num_swa_pages: int | None = None) -> DSV41PoolSizes:
    """Sizes for ``num_pages`` PHYSICAL pages (dummy included). Window precedence: an explicit
    ``num_swa_pages`` > ``config.swa_num_pages_override`` > ``swa_ratio`` x full, floored ONCE in pages."""
    args = config.model_config.dsv41_args
    P = args.window_size
    swa_ratio = _dsv41_swa_ratio(config)
    floor_pages = _dsv41_window_floor_pages(config, args)
    target = num_swa_pages if num_swa_pages is not None else config.swa_num_pages_override
    if target is not None:
        win = min(num_pages, max(floor_pages, int(target) + 1))
    else:
        win = max(floor_pages, (round(swa_ratio * num_pages * P) + P - 1) // P)
    return dsv41_pool_sizes(num_pages, args, swa_ratio, P, n_win_pages=win)


__all__ = [
    "DSV41PoolSizes",
    "dsv41_auto_cost_model",
    "dsv41_cache_per_page",
    "dsv41_kv_unit_bytes",
    "dsv41_pool_bytes",
    "dsv41_pool_sizes",
    "dsv41_solve_num_pages",
    "dsv41_window_unit_bytes",
    "private_window_layer_ids",
]
