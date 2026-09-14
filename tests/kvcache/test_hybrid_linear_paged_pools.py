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