"""`/v1/stats` must report quantities the engine actually honors, not frontend-side defaults.

The frontend process never runs ``_adjust_config`` (that happens inside the scheduler process),
so ``config.page_size`` there can still be the CLI default while the engine pages at 64
(qsa_sparse). Reporting the default made ``total_pages`` read as a token count and hid the
enforced context ceiling from clients entirely.
"""
from __future__ import annotations

from types import SimpleNamespace

from freetoken.server.stats import build_stats


def _state(*, cache_pools=None, enforced=None, page_size=1, model_max=262144):
    pools = dict(cache_pools or {})
    state = SimpleNamespace(
        config=SimpleNamespace(
            served_model_name="unit-model",
            model_path="/models/unit-model",
            page_size=page_size,
            max_seq_len=model_max,
            model_config=SimpleNamespace(
                has_linear_attention=True, has_swa_attention=False, is_moe=True,
                dsv4_args=None,
            ),
        ),
        ready_at=None,
        instance_id="unit",
        gpus=[],
    )
    state.cache_pools = pools
    if enforced is not None:
        state.max_seq_len = enforced
    state.stats = SimpleNamespace(
        kv_used_pages=10, kv_total_pages=4096, mamba_used_slots=0, mamba_total_slots=0,
        swa_used_tokens=0, swa_total_tokens=0, vram_bytes=1 << 30, active=0, completed=0,
        prompt_tokens_total=0, completion_tokens_total=0, cached_tokens_total=0,
        moe_stats=None, decode_tps=lambda *_: 0.0, prefill_tps=lambda *_: 0.0,
    )
    return state


def test_stats_uses_the_engines_page_size_when_the_frontend_config_lags():
    """page_size comes from the engine's readiness meta, so pages x page_size is a token count.

    Regression: with the CLI default (1) the doc reported 4096 "tokens" for a pool the engine
    had actually sized at 4096 pages x 64 = 262144 tokens.
    """
    state = _state(cache_pools={"num_pages": 4096, "page_size": 64}, page_size=1)
    kv = build_stats(state, p95_ms=0, ttft_mean_ms=0)["kv"]

    assert kv["page_size"] == 64
    assert kv["total_pages"] * kv["page_size"] == 262144


def test_stats_falls_back_to_the_frontend_page_size_without_meta():
    """No readiness meta yet (older engine / mid-load): keep the old behaviour, never 0."""
    state = _state(cache_pools=None, page_size=1)
    assert build_stats(state, p95_ms=0, ttft_mean_ms=0)["kv"]["page_size"] == 1


def test_stats_publishes_the_enforced_context_ceiling_next_to_the_model_ceiling():
    """`limits.max_seq_len` is what the scheduler admits; the model's own ceiling is kept."""
    state = _state(enforced=245760, model_max=262144)
    limits = build_stats(state, p95_ms=0, ttft_mean_ms=0)["limits"]

    assert limits["max_seq_len"] == 245760
    assert limits["model_max_seq_len"] == 262144


def test_stats_limits_falls_back_to_the_model_ceiling_mid_load():
    """Before the meta lands the enforced value is unknown; report the ceiling, not 0."""
    state = _state(enforced=None, model_max=262144)
    limits = build_stats(state, p95_ms=0, ttft_mean_ms=0)["limits"]

    assert limits["max_seq_len"] == 262144


def test_stats_model_ctx_is_the_enforced_ceiling_not_the_checkpoint_ceiling():
    """``launch._stats_context_length`` reads ``model.ctx`` as its client-window fallback.

    Reporting the raw checkpoint ceiling there re-introduced exactly the bug ``limits``
    exists to prevent: a client sizes its window from the model card and then sends
    prompts the scheduler rejects. The raw ceiling stays in
    ``limits.model_max_seq_len``.
    """
    state = _state(enforced=245760, model_max=262144)
    doc = build_stats(state, p95_ms=0, ttft_mean_ms=0)

    assert doc["model"]["ctx"] == 245760
    assert doc["limits"]["max_seq_len"] == 245760
    assert doc["limits"]["model_max_seq_len"] == 262144


def test_stats_model_ctx_keeps_the_ceiling_while_the_meta_is_in_flight():
    """No readiness meta yet: ctx falls back to the ceiling rather than 0."""
    state = _state(enforced=None, model_max=262144)
    assert build_stats(state, p95_ms=0, ttft_mean_ms=0)["model"]["ctx"] == 262144
