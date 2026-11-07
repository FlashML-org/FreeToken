"""The fp8 KV pool: code buffer + per-(token, head) scales, sized and rebuilt together.

The pool-side half of --kv-cache-dtype fp8. The interesting failures are the silent
ones: a scale buffer that misses the layer_ids remap, a rebuild that resizes the codes
but not the scales, or a ``unit_bytes`` that drifts from ``kv_cost`` (which is what the
VRAM budget, the cache sliders and the runtime rebuild all divide by).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.models.config import KVCacheGroupSpec

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from freetoken.kernel.triton.kv_quant import kv_codes_dtype

DEV = torch.device("cuda")
HEADS, DIM, LAYERS, PAGES, PAGE_SIZE = 4, 64, 3, 6, 8


def _init_tp() -> None:
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


@pytest.fixture(autouse=True)
def _tp():
    _init_tp()


def _pool(kv_quant="fp8", num_pages=PAGES, layer_ids=None, num_layers=LAYERS):
    from freetoken.kvcache.mha_pool import MHAKVCache

    _init_tp()
    return MHAKVCache(
        num_kv_heads=HEADS,
        num_layers=num_layers,
        head_dim=DIM,
        num_pages=num_pages,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        device=DEV,
        layer_ids=layer_ids,
        kv_quant=kv_quant,
    )


def test_kv_store_dtype_selection():
    from freetoken.kvcache.mha_pool import _kv_store_dtype

    assert _kv_store_dtype(torch.bfloat16, "none") is torch.bfloat16
    assert _kv_store_dtype(torch.bfloat16, "fp8") is kv_codes_dtype()
    with pytest.raises(ValueError, match="kv_quant"):
        _kv_store_dtype(torch.bfloat16, "q6")


def test_fp8_pool_keeps_geometry_and_adds_scale_views():
    pool = _pool()
    assert pool.kv_quant == "fp8"
    # Same shape as the 16-bit pool -- only the element type changed.
    assert pool._kv_buffer.shape == (2, LAYERS, PAGES, PAGE_SIZE, HEADS, DIM)
    assert pool._kv_buffer.dtype == kv_codes_dtype()
    assert pool.store_dtype == kv_codes_dtype()
    # ``dtype`` stays the COMPUTE dtype (kvcache/base.py). Backends size their scratch
    # with it -- reporting codes here hands e4m3 to a 16-bit tl.dot, which does not fail
    # until CUDA-graph capture (exactly how QSA's indexer died in the field).
    assert pool.dtype == torch.bfloat16
    slots = PAGES * PAGE_SIZE
    for layer in range(LAYERS):
        assert pool.k_scale(layer).shape == (slots, HEADS)
        assert pool.v_scale(layer).shape == (slots, HEADS)
        assert pool.k_scale(layer).dtype == torch.float32
    # A 16-bit pool exposes no scales at all.
    assert _pool(kv_quant="none").k_scale(0) is None


def test_layer_ids_remap_applies_to_scales_too():
    layer_ids = (1, 3)
    pool = _pool(layer_ids=layer_ids, num_layers=4)
    assert pool._kv_buffer.shape[1] == 2
    rows = torch.full((1, HEADS * DIM), 2.0, device=DEV, dtype=torch.bfloat16)
    pool.store_kv(rows, rows * 3, torch.tensor([9], device=DEV, dtype=torch.int32), 3)
    torch.testing.assert_close(pool.k_scale(3)[9], torch.full((HEADS,), 2 / 448, device=DEV))
    torch.testing.assert_close(pool.v_scale(3)[9], torch.full((HEADS,), 6 / 448, device=DEV))
    assert not pool.k_scale(1).any()
    assert not pool.v_scale(1).any()
    with pytest.raises(KeyError):
        pool.k_scale(0)


def test_store_kv_scatters_codes_and_scales():
    from freetoken.kernel.triton.kv_quant import codes_to_f32

    tokens = 5
    torch.manual_seed(7)
    rows = torch.randn(tokens, HEADS * DIM, device=DEV, dtype=torch.bfloat16) * 2.0
    out_loc = torch.tensor(
        [3, 0, PAGES * PAGE_SIZE - 1, 40, 17], device=DEV, dtype=torch.int32
    )
    pool = _pool()
    pool.store_kv(rows, rows.clone(), out_loc, layer_id=0)
    torch.cuda.synchronize()

    codes = codes_to_f32(pool.k_cache(0).view(-1, HEADS, DIM))
    scale = pool.k_scale(0)
    f32 = rows.view(tokens, HEADS, DIM).to(torch.float32)
    deq = codes[out_loc.long()] * scale[out_loc.long()].unsqueeze(-1)
    amax = f32.abs().amax(dim=-1, keepdim=True)
    # The scales are per (token, head), so the bound is relative to each row's max.
    err = ((deq - f32).abs() / amax.clamp_min(1e-6)).max()
    assert float(err) < 0.08, float(err)
    # Rows nobody wrote must stay exactly zero -- the scatter touched only out_loc.
    untouched = torch.ones(PAGES * PAGE_SIZE, dtype=torch.bool, device=DEV)
    untouched[out_loc.long()] = False
    assert (codes[untouched] == 0).all()
    assert (scale[untouched] == 0).all()


def test_rebuild_resizes_codes_and_scales_together():
    pool = _pool()
    before = id(pool)
    pool.rebuild(11)
    assert id(pool) == before  # identity preserved (backends cache the object)
    assert pool._kv_buffer.shape == (2, LAYERS, 11, PAGE_SIZE, HEADS, DIM)
    slots = 11 * PAGE_SIZE
    assert pool.k_scale(0).shape == (slots, HEADS)
    assert pool.k_cache(0).shape[0] == 11
    # Per-token cost is page-count invariant, codes and scales alike.
    assert pool.unit_bytes() == _pool().unit_bytes()


def _sizing_config(kv_quant):
    from freetoken.attention import AttnType

    spec = KVCacheGroupSpec(
        name="full",
        layer_ids=tuple(range(LAYERS)),
        num_kv_heads=HEADS,
        head_dim=DIM,
        sliding_window=None,
        attn_type=AttnType.FULL,
    )
    mc = SimpleNamespace(
        has_swa_attention=False,
        has_linear_attention=False,
        num_layers=LAYERS,
        num_kv_heads=HEADS,
        head_dim=DIM,
        kv_cache_group_specs=lambda: (spec,),
    )
    return SimpleNamespace(
        model_config=mc,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        tp_info=SimpleNamespace(size=1),
        kv_quant=kv_quant,
    )


@pytest.mark.parametrize("kv_quant", ["none", "fp8"])
def test_unit_bytes_matches_the_cost_model_that_sized_the_pool(kv_quant):
    """The budget solve and the live pool must agree byte for byte."""
    from freetoken.kvcache.base import spec_kv_bytes_per_token
    from freetoken.kvcache.mha_pool import MHAKVCache

    config = _sizing_config(kv_quant)
    (spec,) = config.model_config.kv_cache_group_specs()
    per_token = spec_kv_bytes_per_token(spec, config)
    assert MHAKVCache.kv_cost(config)[0] == per_token * PAGE_SIZE

    pool = _pool(kv_quant=kv_quant)
    assert pool.unit_bytes() == (per_token, 0)


def test_fp8_lands_just_above_half_the_bytes():
    """Half the code bytes, plus a scale sidecar of 4 B per (token, slab, layer, head)."""
    plain, quantized = _pool("none").unit_bytes()[0], _pool("fp8").unit_bytes()[0]
    scales = 2 * LAYERS * HEADS * 4
    assert quantized == plain // 2 + scales, (plain, quantized, scales)


def test_fp8_runtime_budget_includes_workspace_and_dummy():
    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.kvcache.base import CacheRebuildRejected

    config = _sizing_config("fp8")
    config.num_page_override = None
    config.memory_ratio = 1.0
    persistent = MHAKVCache.kv_cost(config)[0]
    workspace = PAGE_SIZE * 2 * HEADS * DIM * 2
    per_page, fixed, _, _ = MHAKVCache.runtime_cost(config)
    assert per_page == persistent + workspace
    assert fixed == per_page
    budget = (PAGES + 1) * per_page
    assert MHAKVCache.solve_num_pages(config, budget) == PAGES
    config.num_page_override = PAGES + 1
    with pytest.raises(ValueError, match="budget"):
        MHAKVCache.solve_num_pages(config, budget)
    pool = _pool()
    before = pool._kv_buffer.data_ptr()
    with pytest.raises(CacheRebuildRejected):
        pool.validate_rebuild(config, num_pages=PAGES + 1, target_moe=0, per_expert_bytes=1,
                              baseline_free=budget, weights_bytes=0, current_num_pages=PAGES)
    assert pool._kv_buffer.data_ptr() == before
    assert MHAKVCache.runtime_cost(_sizing_config("none")) == MHAKVCache.kv_cost(_sizing_config("none"))
