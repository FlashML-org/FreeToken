"""Cost model + per-tier sizing for the DSV41 paged KV pool (DeepSeek-V4.1).

Same shape as ``dsv4_cost_model``: one bytes-per-P-token number for the budget division, and
independent per-tier sizing from the anchor ``full_token = num_pages * P``. The tiers:

* window pool, every layer          -- ``swa_ratio`` of the full history, packed fp8 rows
* main KV pool, per kv source       -- ``full_token // ratio`` packed fp4 rows
* index-key pool, per kv source     -- ``full_token // ratio`` packed fp4 rows
* compress-state ring, ratio>1 sources -- ``ring_size`` fp32 ``kv|score`` slots per WINDOW page

All byte widths come from ``DSV41Geometry`` (the row formats), so there is exactly one place
that knows what a row costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .dsv41_geometry import DSV41Geometry
from .window_tier import reserved_window_pages

_INT64_BYTES = 8


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
    num_pages: int, geom: DSV41Geometry, swa_ratio: float, P: int, n_win_pages: int | None = None
) -> DSV41PoolSizes:
    full_token = num_pages * P
    if n_win_pages is None:
        n_win_pages = (round(swa_ratio * full_token) + P - 1) // P
    n_win_pages = min(n_win_pages, num_pages)
    sizes = DSV41PoolSizes(P=P, swa_ratio=swa_ratio, full_token=full_token, n_win_slots=n_win_pages * P, n_win_pages=n_win_pages)
    for src in geom.kv_source_layer_ids:
        ratio = geom.ratio_of(src)
        sizes.main_rows[src] = full_token // ratio
        sizes.idx_rows[src] = full_token // ratio
        if ratio > 1:
            sizes.state_slots[src] = n_win_pages * geom.ring_size(src)
    return sizes


def dsv41_pool_bytes(sizes: DSV41PoolSizes, geom: DSV41Geometry, n_scratch: int = 1) -> int:
    """Exact bytes a ``DSV41PagedKVCache`` built from ``sizes`` allocates (mirror of ``total_bytes``)."""
    n_shared = len(geom.shared_window_layer_ids)
    total = n_shared * sizes.n_win_slots * geom.win_row_bytes
    total += len(geom.private_window_layer_ids) * n_scratch * sizes.P * geom.win_row_bytes  # per-request rings
    total += (sizes.full_token + 1) * _INT64_BYTES  # full_to_window (+ sentinel row)
    for src in geom.kv_source_layer_ids:
        total += (sizes.main_rows[src] + n_scratch) * geom.main_row_bytes
        total += (sizes.idx_rows[src] + n_scratch) * geom.idx_row_bytes
        if src in sizes.state_slots:
            total += (sizes.state_slots[src] + 1) * geom.state_bytes
    return int(total)


def dsv41_cache_per_page(geom: DSV41Geometry, swa_ratio: float, P: int) -> int:
    """Marginal bytes per P-token page across all tiers (window scaled by ``swa_ratio``)."""
    total = len(geom.shared_window_layer_ids) * round(swa_ratio * P) * geom.win_row_bytes
    for src in geom.kv_source_layer_ids:
        ratio = geom.ratio_of(src)
        total += (P // ratio) * (geom.main_row_bytes + geom.idx_row_bytes)
        if ratio > 1:
            total += round(swa_ratio * geom.ring_size(src)) * geom.state_bytes
    return int(total)


def dsv41_kv_unit_bytes(geom: DSV41Geometry, P: int) -> int:
    """FULL-tier bytes per full-history token (main + index pools + the mapping); the slider's
    ``kv_bytes_per_token``. 890 B/token for DeepSeek-V4.1-Flash before the mapping."""
    per_page = P * _INT64_BYTES
    for src in geom.kv_source_layer_ids:
        per_page += (P // geom.ratio_of(src)) * (geom.main_row_bytes + geom.idx_row_bytes)
    return -(-per_page // P)


def dsv41_window_unit_bytes(geom: DSV41Geometry, P: int) -> int:
    """WINDOW-tier bytes per window token (sliding KV on every shared-window layer + the state rings)."""
    per_page = len(geom.shared_window_layer_ids) * P * geom.win_row_bytes
    for src in geom.ring_sources:
        per_page += geom.ring_size(src) * geom.state_bytes
    return -(-per_page // P)


def dsv41_solve_num_pages(
    available_bytes: int, geom: DSV41Geometry, swa_ratio: float, floor_win_pages: int, P: int, n_scratch: int = 1
) -> DSV41PoolSizes:
    """Largest budget-respecting pool; the window floor is honored in pages and the total is
    byte-checked. Raises ``ValueError`` when even the minimal pool does not fit."""

    def _sizes(num: int) -> DSV41PoolSizes:
        win = max(floor_win_pages, (round(swa_ratio * num * P) + P - 1) // P)
        return dsv41_pool_sizes(num, geom, swa_ratio, P, n_win_pages=win)

    lo = max(floor_win_pages, 2)
    if dsv41_pool_bytes(_sizes(lo), geom, n_scratch) > available_bytes:
        raise ValueError(
            f"DSV41 KV budget {available_bytes} bytes cannot fit the minimal pool ({lo} pages incl. the "
            f"window working-set floor {floor_win_pages}); raise memory_ratio or lower max_running_req/max_seq_len"
        )
    hi = max(lo, available_bytes // max(1, dsv41_cache_per_page(geom, 0.0, P)))
    while dsv41_pool_bytes(_sizes(hi), geom, n_scratch) <= available_bytes:
        hi *= 2
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if dsv41_pool_bytes(_sizes(mid), geom, n_scratch) <= available_bytes:
            lo = mid
        else:
            hi = mid
    return _sizes(lo)


def dsv41_auto_cost_model(geom: DSV41Geometry, swa_ratio: float, floor_win_pages: int, P: int, n_scratch: int = 1):
    """Affine ``(cache_per_page, fixed_cache_size, min_reserve_tokens)`` for the MoE-first auto planner."""
    per_page = dsv41_cache_per_page(geom, swa_ratio, P) + P * _INT64_BYTES
    n0 = max(floor_win_pages, 2)
    win0 = max(floor_win_pages, (round(swa_ratio * n0 * P) + P - 1) // P)
    base = dsv41_pool_bytes(dsv41_pool_sizes(n0, geom, swa_ratio, P, n_win_pages=win0), geom, n_scratch)
    # The engine combines this structural floor with the configured KV reserve.
    return per_page, max(0, base - n0 * per_page), n0 * P


# ---- config-facing sizing (EngineConfig in, sizes out) ----


def dsv41_geometry(config) -> DSV41Geometry:
    """The DSV41 geometry a serving config carries (on its attention group)."""
    for group in config.model_config.attention_groups:
        geom = getattr(group, "geometry", None)
        if isinstance(geom, DSV41Geometry):
            return geom
    raise ValueError("model config has no DSV41 attention group")


def _dsv41_swa_ratio(config) -> float:
    return float(config.swa_full_tokens_ratio)


def _dsv41_window_floor_pages(config, geom: DSV41Geometry) -> int:
    """Minimum window pages: one prefill chunk's reach (capped at 8 pages) + the running requests'
    working set (see ``reserved_window_pages``)."""
    P = geom.window
    prefill_reach_pages = (config.max_seq_len + P - 1) // P
    radix = config.cache_type != "naive"
    return min(prefill_reach_pages, 8) + reserved_window_pages(config.max_running_req, radix)


def _dsv41_pool_sizes(config, num_pages: int, num_swa_pages: int | None = None) -> DSV41PoolSizes:
    """Sizes for ``num_pages`` PHYSICAL pages (dummy included). Window precedence: an explicit
    ``num_swa_pages`` > ``config.swa_num_pages_override`` > ``swa_ratio`` x full, floored ONCE in pages."""
    geom = dsv41_geometry(config)
    P = geom.window
    swa_ratio = _dsv41_swa_ratio(config)
    floor_pages = _dsv41_window_floor_pages(config, geom)
    target = num_swa_pages if num_swa_pages is not None else config.swa_num_pages_override
    if target is not None:
        win = min(num_pages, max(floor_pages, int(target) + 1))
    else:
        win = max(floor_pages, (round(swa_ratio * num_pages * P) + P - 1) // P)
    return dsv41_pool_sizes(num_pages, geom, swa_ratio, P, n_win_pages=win)


__all__ = [
    "DSV41PoolSizes",
    "dsv41_auto_cost_model",
    "dsv41_cache_per_page",
    "dsv41_geometry",
    "dsv41_kv_unit_bytes",
    "dsv41_pool_bytes",
    "dsv41_pool_sizes",
    "dsv41_solve_num_pages",
    "dsv41_window_unit_bytes",
]
