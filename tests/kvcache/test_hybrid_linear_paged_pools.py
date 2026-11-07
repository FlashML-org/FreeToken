"""Pool-factory generalization: hybrid linear x ANY paged family.

The old factory hard-required "linear + one GQA group" (Qwen3.5 GDN shape);
glm5_next is linear (KDA) x DSA. Checks the factory dispatch, the MLA/DSA
layer-id remap (34 KDA layers cost no latent slabs), the kpool pool selection
(gate slab), and the KV cost model's kpool double-count of the index slabs.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kvcache import create_kvcache_pool, resolve_pool_class
from freetoken.kvcache.dsa_pool import DSAKVCache, KpoolDSAKVCache
from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)


@pytest.fixture(autouse=True)
def _single_rank_tp():
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _glm5_like_config(index_kpool=4, index_head_dim=128):
    n_layers = 12
    dsa_ids = tuple(range(3, n_layers, 4))  # 3, 7, 11
    kda_ids = tuple(i for i in range(n_layers) if i not in dsa_ids)
    rotary = RotaryConfig(head_dim=256, rotary_dim=0, max_position=4096, base=1e4, scaling=None)
    groups = (
        LinearGatedDeltaGroupConfig(
            name="linear", layer_ids=kda_ids,
            num_key_heads=4, num_value_heads=4, key_head_dim=128, value_head_dim=128,
            conv_kernel_dim=4, output_gate="sigmoid", variant="kda",
        ),
        FullAttentionGroupConfig(
            name="full", layer_ids=dsa_ids, num_kv_heads=1, head_dim=512,
            rotary_config=rotary, mla=True,
            index_head_dim=index_head_dim, num_index_layers=len(dsa_ids),
            index_ratio=index_kpool,
        ),
    )
    return ModelConfig(
        num_layers=n_layers, num_qo_heads=4, num_kv_heads=1, head_dim=512,
        hidden_size=256, vocab_size=1000, intermediate_size=512,
        rms_norm_eps=1e-5, rotary_config=rotary, hidden_act="silu",
        tie_word_embeddings=False, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=64, norm_topk_prob=True, model_type="glm5_next",
        architectures=["Glm5NextForCausalLM"], moe_enabled=True,
        attention_groups=groups,
    )


def test_factory_builds_kpool_pool_with_layer_remap():
    cfg = _glm5_like_config()
    assert resolve_pool_class(cfg) is KpoolDSAKVCache

    pool = create_kvcache_pool(
        model_config=cfg, num_pages=4, page_size=64,
        dtype=torch.bfloat16, device=torch.device("cpu"), num_req_slots=5,
    )
    assert isinstance(pool, KpoolDSAKVCache)
    # Latent slabs back ONLY the 3 DSA layers (34-of-45 economy at real scale).
    assert pool._kv_buffer.shape[1] == 3
    # Global layer-id addressing: DSA layers resolve, KDA layers have no slab.
    for lid in (3, 7, 11):
        assert pool.latent_rows(lid).shape == (256, 512)
    with pytest.raises(KeyError):
        pool.latent_rows(0)  # a KDA layer
    # Shadow index slab: tokens/ratio rows + one scratch row per request slot.
    assert pool.index_k_cache(0).shape == (256 // 4 + 5, 128)
    assert pool.cmp_scratch_base == 256 // 4
    # kpool tail rings exist at [num_req_slots, ratio, head_dim] per indexer layer.
    assert pool.tail_k(0).shape == pool.tail_gate(0).shape
    assert pool.tail_k(0).shape == (5, 4, 128)


def test_factory_kpool1_builds_plain_dsa_pool():
    cfg = _glm5_like_config(index_kpool=1)
    assert resolve_pool_class(cfg) is DSAKVCache
    pool = create_kvcache_pool(
        model_config=cfg, num_pages=16, page_size=1,
        dtype=torch.bfloat16, device=torch.device("cpu"),
    )
    assert type(pool) is DSAKVCache


@pytest.mark.parametrize("kv_quant", ["none", "fp8"])
def test_full_group_quantization_preserves_hybrid_layer_mapping_and_cost(kv_quant):
    from dataclasses import replace
    from types import SimpleNamespace

    from freetoken.kvcache.mha_pool import MHAKVCache

    cfg = _glm5_like_config()
    linear, full = cfg.attention_groups
    cfg = replace(
        cfg, num_qo_heads=16, num_kv_heads=2, head_dim=256,
        attention_groups=(
            replace(linear, variant="gdn", output_gate="silu"),
            FullAttentionGroupConfig(name="full", layer_ids=full.layer_ids,
                                     num_kv_heads=2, head_dim=256, rotary_config=cfg.rotary_config),
        ),
    )
    pool = create_kvcache_pool(cfg, 7, 8, torch.bfloat16, torch.device("cpu"), kv_quant=kv_quant)
    config = SimpleNamespace(model_config=cfg, tp_info=SimpleNamespace(size=1),
                             dtype=torch.bfloat16, page_size=8, kv_quant=kv_quant)
    # Only the three full layers own KV; the other nine have no code or scale slab.
    bytes_per_token = 3 * 2 * 2 * (256 + 4 if kv_quant == "fp8" else 256 * 2)
    assert MHAKVCache.kv_cost(config)[0] == bytes_per_token * 8
    for pages in (7, 11, 3):
        if pages != 7:
            pool.rebuild(pages)
        assert pool.unit_bytes() == (bytes_per_token, 0)
        assert pool._kv_buffer.numel() * pool._kv_buffer.element_size() == (
            3 * 2 * 2 * 256 * pages * 8 * (1 if kv_quant == "fp8" else 2))
        for lid in full.layer_ids:
            pool.k_cache(lid).fill_(lid)
            pool.v_cache(lid).fill_(lid + 1)
            if kv_quant == "fp8":
                pool.k_scale(lid).fill_(lid / 10)
                pool.v_scale(lid).fill_((lid + 1) / 10)
        for lid in full.layer_ids:
            assert torch.all(pool.k_cache(lid) == lid)
            assert torch.all(pool.v_cache(lid) == lid + 1)
            if kv_quant == "fp8":
                assert pool.k_scale(lid).shape == (pages * 8, 2)
                torch.testing.assert_close(pool.k_scale(lid), torch.full((pages * 8, 2), lid / 10))
                torch.testing.assert_close(pool.v_scale(lid), torch.full((pages * 8, 2), (lid + 1) / 10))
        for lid in linear.layer_ids:
            with pytest.raises(KeyError):
                pool.k_cache(lid)


def test_cost_model_kpool_shadow_slab_quarter_cost():
    """The shadow slab stores one row per index_ratio tokens: the kpool spec's
    index bytes are 1/ratio of the plain DSA slab (rings/scratch are per-request
    and not part of the per-token price)."""
    from types import SimpleNamespace

    from freetoken.kvcache.base import spec_kv_bytes_per_token

    tp = SimpleNamespace(size=1)
    econf = SimpleNamespace(tp_info=tp, dtype=torch.bfloat16)
    (spec,) = [
        s for s in _glm5_like_config().kv_cache_group_specs() if s.num_layers > 0
    ]
    (spec1,) = [
        s
        for s in _glm5_like_config(index_kpool=1).kv_cache_group_specs()
        if s.num_layers > 0
    ]
    index_full = spec.index_head_dim * spec.num_index_layers * 2
    assert (
        spec_kv_bytes_per_token(spec1, econf) - spec_kv_bytes_per_token(spec, econf)
        == index_full - index_full // 4
    )
