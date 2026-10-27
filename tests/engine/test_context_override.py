"""Context.override: a draft forward sees its own KV pool and backend and none of the target's GDN pool or MoE offload cache, and everything comes back on exit."""

import pytest

from freetoken.core import Batch, Context


def _ctx() -> Context:
    ctx = Context(page_size=1)
    ctx.kv_cache = "target_kv"
    ctx.attn_backend = "target_backend"
    ctx.linear_state_pool = "gdn_pool"
    ctx.moe_offload_cache = "expert_cache"
    return ctx


def _state(ctx: Context) -> tuple:
    return ctx.kv_cache, ctx.attn_backend, ctx.linear_state_pool, ctx.moe_offload_cache


def test_override_hides_target_state_and_restores_on_exit_and_error():
    ctx = _ctx()
    before = _state(ctx)
    with ctx.override(kv_cache="draft_kv", attn_backend="draft_backend", hide_target_state=True):
        assert _state(ctx) == ("draft_kv", "draft_backend", None, None)
        with ctx.override(kv_cache="inner_kv"):
            assert _state(ctx) == ("inner_kv", "draft_backend", None, None)
        assert _state(ctx) == ("draft_kv", "draft_backend", None, None)
    assert _state(ctx) == before

    with pytest.raises(RuntimeError):
        with ctx.override(kv_cache="draft_kv", hide_target_state=True):
            raise RuntimeError
    assert _state(ctx) == before


def test_forward_batch_scopes_the_override_with_the_batch():
    ctx = _ctx()
    batch = Batch(reqs=[], phase="decode")
    with ctx.forward_batch(batch, kv_cache="draft_kv", hide_target_state=True):
        assert ctx.batch is batch
        assert _state(ctx) == ("draft_kv", "target_backend", None, None)
        with pytest.raises(AssertionError):
            with ctx.forward_batch(batch):
                pass
    assert ctx._batch is None
    assert _state(ctx) == ("target_kv", "target_backend", "gdn_pool", "expert_cache")
