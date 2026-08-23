from __future__ import annotations

from typing import Any, Dict


def _supports_swa_ratio(config) -> bool:
    """Whether ``swa_full_tokens_ratio`` sizes a separate window pool for this model -- DSV4
    (always) or a radix-SWA model (Gemma). Gates the ratio in telemetry and rebuild."""
    mc = config.model_config
    if mc.dsv4_args is not None:
        return True
    return mc.has_swa_attention and config.cache_type == "swa_radix"


def compute_cache_unit_bytes(engine: "Engine") -> Dict[str, int]:
    """Per-unit VRAM byte cost of each runtime cache, measured from the real allocated pool
    tensors so the desktop cache panel can show a true VRAM delta per slider step:

      kv_bytes_per_token   -- total paged-KV bytes (across every attention layer) for one token
      moe_bytes_per_expert -- bytes of one MoE expert-cache (offload bank) slot
      mamba_bytes_per_slot -- bytes of one SSM/GDN recurrent-state slot (all linear layers)

    Each is 0 when the model has no such cache (dense / non-hybrid / non-offload MoE) or the pool
    can't be measured. Best-effort and total: any failure degrades that unit to 0 and never raises
    -- this runs on the readiness path and must not block startup.
    """

    def _kv_swa() -> tuple[int, int]:
        # KV (full pool) + SWA (window pool) per-token bytes come from the pool itself: the
        # generic pools measure their live buffers, DSV4 reads its per-tier cost model. This
        # layer never branches on the model. 0/0 for a fake engine or a model without the pool.
        pool = engine.kv_cache
        if pool is None:
            return 0, 0
        kv, swa = pool.unit_bytes()
        return int(kv), int(swa)

    def _moe() -> int:
        banks = getattr(engine.moe_offload_cache, "bank_caches", None)
        if not banks:
            return 0
        # Each bank cache is (cache_size, *row_shape); one slot's bytes = row bytes summed over
        # the format's banks (== cache_budget.expert_bytes_per_slot on the source rows).
        return int(sum(t[0].numel() * t.element_size() for t in banks.values()))

    def _mamba() -> int:
        pool = engine.linear_state_pool
        if pool is None:
            return 0
        return int(pool.bytes_per_slot())

    out = {
        "kv_bytes_per_token": 0,
        "moe_bytes_per_expert": 0,
        "mamba_bytes_per_slot": 0,
        "swa_bytes_per_token": 0,
    }
    try:
        out["kv_bytes_per_token"], out["swa_bytes_per_token"] = _kv_swa()
    except Exception:  # noqa: BLE001 -- best-effort; a bad read must never block readiness
        pass
    for key, fn in (("moe_bytes_per_expert", _moe), ("mamba_bytes_per_slot", _mamba)):
        try:
            out[key] = fn()
        except Exception:  # noqa: BLE001 -- best-effort; a bad read must never block readiness
            out[key] = 0
    return out


def _pool_budget_free_vram_bytes(engine: "Engine") -> int:
    """The pool-budget baseline captured in ``Engine.__init__``: free VRAM after weights
    loaded, before any runtime cache pool was allocated (``_post_weights_free``). This is the
    honest "spend it all on one pool" budget for the desktop slider upper bounds — stable for
    the process lifetime, unaffected by allocator caching / CUDA graphs / later rebuilds.
    0 when unavailable (fake engines in tests, capture failed). Never raises."""
    try:
        return max(0, int(engine._post_weights_free or 0))
    except Exception:  # noqa: BLE001
        return 0


def compute_cache_floors(engine: "Engine") -> Dict[str, int]:
    """Per-pool minimum (floor) unit counts the runtime cache-rebuild path enforces, in the
    units the desktop sliders use (KV tokens, MoE experts, usable mamba slots). Every floor is
    derived live from the model config / engine validation logic -- never a baked-in constant:

      kv_tokens   -- rebuild_runtime_cache rejects num_pages <= 0, so the floor is one page's
                     worth of tokens (page_size). Owned-KV models (dsv4) have no rebuildable
                     generic pool -> 0.
      moe_experts -- _require_offload_cache_size needs the offload cache to hold >= one MoE
                     layer's experts (num_experts). 0 when the model has no offload cache.
      mamba_slots -- rebuild_runtime_cache rejects num_mamba_slots below
                     _linear_pool_min_slots(config) - 1 (the physical floor minus the reserved
                     padding sink -> usable-slot floor). 0 when the model has no GDN state pool.

    Best-effort/total: any failure degrades that floor to 0 and never raises (readiness path)."""
    config = engine.config
    floors = {"kv_tokens": 0, "moe_experts": 0, "mamba_slots": 0, "swa_tokens": 0}
    if config is None:
        return floors

    def _kv() -> int:
        # The full-KV pool floor in tokens, from the pool family: generic = one page; DSV4 =
        # the window working-set floor that validate_rebuild() enforces.
        pool = getattr(engine, "kv_cache", None)
        if pool is None:
            return int(config.page_size)
        return int(type(pool).min_kv_tokens(config))

    def _moe() -> int:
        if engine.moe_offload_cache is None:
            return 0
        return int(config.model_config.num_experts)

    def _mamba() -> int:
        from .linear_state_pool import _linear_pool_min_slots

        if engine.linear_state_pool is None:
            return 0
        return int(_linear_pool_min_slots(config) - 1)

    def _swa() -> int:
        # Window-pool floor in tokens (matches the pool's own page unit x count):
        #   DSV4      -- the working-set floor pages _dsv4_pool_sizes clamps n_win_pages to, x P.
        #   radix-SWA -- the concurrent working set (+1 slot-0 sentinel) _swa_paged_num_tokens
        #                floors to.
        # 0 for models without a window pool.
        from .dsv4_cost_model import _dsv4_window_floor_pages
        from .hybrid_swa_pool import _swa_pool_floor

        mc = config.model_config
        if mc.dsv4_args is not None:
            P = mc.dsv4_args.window_size
            return int(_dsv4_window_floor_pages(config, P) * P)
        if not (mc.has_swa_attention and config.cache_type == "swa_radix"):
            return 0
        return int(_swa_pool_floor(config) + 1)

    for key, fn in (("kv_tokens", _kv), ("moe_experts", _moe), ("mamba_slots", _mamba),
                    ("swa_tokens", _swa)):
        try:
            floors[key] = fn()
        except Exception:  # noqa: BLE001 -- best-effort; a bad read must never block readiness
            floors[key] = 0
    return floors


def compute_cache_pools(engine: "Engine") -> Dict[str, Any]:
    """The ACTUAL pool sizes allocated at load, in API units. The frontend otherwise only
    learns them from per-generation UserReply snapshots — i.e. not until the first chat —
    so this seeds /v1/cache/status geometry with truth from the moment the server is ready.
    mamba is the usable slot count (num_slots minus the reserved padding sink), matching the
    scheduler's reported totals. 0 for pools the model lacks; never raises."""
    pools = {
        "num_pages": 0, "page_size": 0, "moe_cache_size": 0, "num_mamba_slots": 0,
        "swa_page_size": 0, "num_swa_pages": 0,
        "requested_prefill_tokens": 0, "pool_prefill_cap_tokens": 0,
        "effective_prefill_tokens": 0, "swa_capacity_source": "none",
        "prefill_limiting_reason": "none",
    }
    try:
        config = engine.config
        pools["num_pages"] = int(engine.num_pages or 0)
        if config is not None:
            pools["page_size"] = int(config.page_size or 0)
            requested = int(getattr(config, "max_extend_tokens", 0) or 0)
            pool_cap = int(
                getattr(getattr(engine, "kv_cache", None), "prefill_chunk_budget", 0) or 0
            )
            mc = config.model_config
            pools.update(prefill_geometry(
                requested,
                pool_cap,
                getattr(config, "swa_capacity_source", "none"),
                has_window_pool=bool(
                    getattr(mc, "dsv4_args", None) is not None
                    or (
                        getattr(mc, "has_swa_attention", False)
                        and getattr(config, "cache_type", None) == "swa_radix"
                    )
                ),
            ))
            # Window pool: its own page unit (swa_page_size -- DSV4 windows are P-token pages,
            # radix-SWA is token-granular page_size 1) and the concrete current size in that unit
            # (num_swa_pages, usable count). Same source as the scheduler's _current_cache_geometry.
            # Both 0 for models without a window pool. Lets a client denominate the swa control.
            mc = config.model_config
            if mc.dsv4_args is not None:
                pools["swa_page_size"] = int(mc.dsv4_args.window_size or 0)
                sizes = getattr(engine.kv_cache, "sizes", None)  # usable = physical minus dummy
                if sizes is not None:
                    pools["num_swa_pages"] = max(0, int(sizes.n_win_pages) - 1)
            elif mc.has_swa_attention and config.cache_type == "swa_radix":
                pools["swa_page_size"] = 1  # usable = pool tokens minus the slot-0 sentinel
                pools["num_swa_pages"] = max(0, int(getattr(engine.kv_cache, "swa_num_tokens", 0) or 0) - 1)
        moe = engine.moe_offload_cache
        if moe is not None:
            pools["moe_cache_size"] = int(moe.cache_size or 0)
        lsp = engine.linear_state_pool
        if lsp is not None:
            pools["num_mamba_slots"] = max(0, int(lsp.num_slots or 0) - 1)
    except Exception:  # noqa: BLE001 -- best-effort; readiness must not depend on this
        pass
    return pools


def prefill_geometry(
    requested_tokens: int,
    pool_cap_tokens: int,
    swa_capacity_source: str,
    *,
    has_window_pool: bool,
) -> Dict[str, Any]:
    """Canonical requested/pool/effective prefill readout for startup and live rebuilds."""
    requested = max(0, int(requested_tokens or 0))
    pool_cap = max(0, int(pool_cap_tokens or 0))
    effective = min(requested, pool_cap) if pool_cap else requested
    if requested <= 0:
        reason = "none"
    elif pool_cap and pool_cap < requested:
        reason = "swa_pool"
    elif pool_cap and pool_cap == requested:
        reason = "requested_and_swa_pool"
    else:
        reason = "requested_limit"
    return {
        "requested_prefill_tokens": requested,
        "pool_prefill_cap_tokens": pool_cap,
        "effective_prefill_tokens": effective,
        "swa_capacity_source": str(swa_capacity_source) if has_window_pool else "none",
        "prefill_limiting_reason": reason,
    }


def compute_cache_status_meta(engine: "Engine") -> Dict[str, Any]:
    """Full readiness ("meta", …) payload for the desktop cache panel: the per-unit VRAM costs
    (compute_cache_unit_bytes) plus the post-weights pre-pool free-VRAM baseline (the sliders'
    ideal-max budget), the per-pool floors, and the actual pool sizes allocated at load.
    Best-effort/total -- each piece independently degrades to 0/{} and this never raises
    (it runs on readiness)."""
    meta: Dict[str, Any] = dict(compute_cache_unit_bytes(engine))
    meta["free_vram_bytes"] = _pool_budget_free_vram_bytes(engine)
    meta["floors"] = compute_cache_floors(engine)
    pools = meta["pools"] = compute_cache_pools(engine)
    # Effective allocated window/full ratio, for DSV4 and radix-SWA; 0.0 otherwise. An absolute
    # derived/explicit window supersedes the configured fallback ratio, so report live geometry.
    cfg = engine.config
    has_swa_ratio = cfg is not None and _supports_swa_ratio(cfg)
    ratio = 0.0
    if has_swa_ratio:
        try:
            is_dsv4 = getattr(cfg.model_config, "dsv4_args", None) is not None
            full = int(pools["num_pages"]) if is_dsv4 else int(pools["num_pages"]) * int(pools["page_size"])
            window = int(pools["num_swa_pages"])
            ratio = min(1.0, window / full) if full > 0 else float(cfg.swa_full_tokens_ratio)
        except Exception:  # noqa: BLE001 -- best-effort readiness metadata
            ratio = float(cfg.swa_full_tokens_ratio)
    meta["swa_full_tokens_ratio"] = ratio
    # Exact total cache VRAM budget (all pools) the engine honors: memory_ratio of the
    # post-weights baseline minus weights — the same figure the rebuild fit-check uses, with
    # fixed=0 so it's the whole-cache ceiling (KV + MoE + Mamba + SWA), not the KV+MoE remainder.
    # Lets the desktop show the real ceiling instead of reverse-deriving it from the per-pool
    # limits. 0 when the baseline wasn't captured (fake engines in tests).
    from freetoken.engine.cache_budget import net_cache_budget_bytes as _net_budget

    try:
        _baseline = int(engine._baseline_free or 0)
        _weights = int(engine._weights_bytes or 0)
        _mr = float(cfg.memory_ratio) if cfg is not None else 1.0
        meta["cache_budget_bytes"] = max(0, _net_budget(_mr, _baseline, _weights, 0)) if _baseline > 0 else 0
    except Exception:  # noqa: BLE001 -- best-effort; readiness must not depend on this
        meta["cache_budget_bytes"] = 0
    return meta
