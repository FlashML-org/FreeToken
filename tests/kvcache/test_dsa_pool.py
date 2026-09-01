"""DSA/MLA pool classes + the rebuild-OOB regression the pool restructure exists for.

The bug class being pinned: the index-key storage used to live as a backend-private
tensor sized once at backend init; ``rebuild_runtime_cache`` growing the page count
made the allocator hand out rows past the index slab's end (device assert on the
write side, silent OOB reads in the scoring kernel). With storage in ``DSAKVCache``,
``rebuild`` resizes the latent and index slabs atomically and the backend re-derives
views per forward, so rows valid for one slab are valid for both.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

LATENT, IDX_DIM = 80, 32  # latent 64+16: kernel spans need pow-2 dims (512+64 in the real model)


def _pool(num_pages: int):
    from freetoken.kvcache.dsa_pool import DSAKVCache

    return DSAKVCache(
        latent_dim=LATENT, num_layers=2, num_pages=num_pages, page_size=1,
        dtype=torch.bfloat16, device=torch.device("cuda"),
        index_head_dim=IDX_DIM, num_index_layers=1,
    )


def test_rebuild_resizes_both_slabs_atomically():
    pool = _pool(33)
    assert pool.latent_rows(0).shape == (33, LATENT)
    assert pool.index_k_cache(0).shape == (33, IDX_DIM)
    pool.rebuild(129)
    assert pool.latent_rows(0).shape == (129, LATENT)
    assert pool.index_k_cache(0).shape == (129, IDX_DIM)

    # Rows past the ORIGINAL allocation must be writable/readable on BOTH slabs --
    # the exact class the old backend-private index tensor failed.
    out_loc = torch.arange(100, 120, device="cuda")
    c_kv = torch.randn(20, LATENT - 16, device="cuda", dtype=torch.bfloat16)
    k_rope = torch.randn(20, 16, device="cuda", dtype=torch.bfloat16)
    k_idx = torch.randn(20, IDX_DIM, device="cuda", dtype=torch.bfloat16)
    pool.store_kv(c_kv, k_rope, out_loc, layer_id=1)
    pool.store_index_k(k_idx, out_loc, slot=0)
    torch.cuda.synchronize()  # device-side asserts surface here
    got = pool.latent_rows(1)[100:120]
    assert torch.equal(got[:, : LATENT - 16], c_kv) and torch.equal(got[:, LATENT - 16 :], k_rope)
    assert torch.equal(pool.index_k_cache(0)[100:120], k_idx)
    # V aliases K (single latent slab)
    assert pool.v_cache(1).data_ptr() == pool.k_cache(1).data_ptr()


def test_sparse_decode_reads_grown_pool_through_backend_kernels():
    """Score + select + attend against rows past the original pool size, after a
    rebuild -- end-to-end through the same kernels the backend uses."""
    from freetoken.kernel.triton.glm_dsa_sparse import glm_dsa_decode_logits, glm_dsa_sparse_attn

    pool = _pool(33)
    pool.rebuild(257)
    live = 180  # > original 33: every one of these rows would have been OOB before
    rows_live = torch.randperm(256, device="cuda")[:live].to(torch.int32)
    pool.store_index_k(torch.randn(live, IDX_DIM, device="cuda", dtype=torch.bfloat16),
                       rows_live.long(), slot=0)
    pool.store_kv(torch.randn(live, LATENT - 16, device="cuda", dtype=torch.bfloat16),
                  torch.randn(live, 16, device="cuda", dtype=torch.bfloat16),
                  rows_live.long(), layer_id=0)

    rows = torch.full((1, 257), -1, device="cuda", dtype=torch.int32)
    rows[0, :live] = rows_live
    kvlen = torch.tensor([live], device="cuda", dtype=torch.int32)
    q_idx = torch.randn(1, 16, IDX_DIM, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1, 16, device="cuda").abs()

    s = glm_dsa_decode_logits(q_idx, w, pool.index_k_cache(0), rows, kvlen)
    k_ref = pool.index_k_cache(0)[rows_live.long()].float()
    ref = (torch.einsum("hd,td->ht", q_idx[0].float(), k_ref).relu() * w[0][:, None]).sum(0)
    assert (s[0, :live] - ref).abs().max().item() < 0.05
    assert torch.isinf(s[0, live:]).all()

    topk = 64
    vals, cols = s.topk(topk, -1)
    sel = rows.gather(1, cols)
    sel = torch.where(vals == float("-inf"), sel.new_full((), -1), sel)
    q = torch.randn(1, 1, 4, LATENT, device="cuda", dtype=torch.bfloat16)
    o = glm_dsa_sparse_attn(q, pool.latent_rows(0), sel.view(1, 1, topk), 0.1,
                            counts=torch.tensor([[topk]], device="cuda", dtype=torch.int32),
                            d_v=LATENT - 16)
    picked = pool.latent_rows(0)[sel.view(-1).long()].float()
    s2 = (q[0, 0].float() @ picked.T) * 0.1
    ref_o = s2.softmax(-1) @ picked[:, : LATENT - 16]
    assert (o[0, 0].float() - ref_o).abs().max().item() < 2e-2


def test_mla_pool_selected_by_group_spec():
    """The factory keys MLA/DSA pools off the attention-group spec, never the model
    payload -- and zeroed index dims (the dense ablation) fall back to MLAKVCache."""
    from freetoken.kvcache import create_kvcache_pool
    from freetoken.kvcache.dsa_pool import DSAKVCache, MLAKVCache
    from freetoken.models.config import FullAttentionGroupConfig, ModelConfig, RotaryConfig

    def cfg(index_dim, n_idx):
        rc = RotaryConfig(head_dim=LATENT, rotary_dim=16, max_position=512, base=1e4, scaling=None)
        return ModelConfig(
            num_layers=2, num_qo_heads=4, num_kv_heads=1, head_dim=LATENT,
            hidden_size=64, vocab_size=64, num_experts=0,
            intermediate_size=128, rms_norm_eps=1e-6, hidden_act="silu",
            tie_word_embeddings=False, num_experts_per_tok=0,
            moe_intermediate_size=0, norm_topk_prob=False,
            model_type="test_mla", architectures=("TestMLA",),
            rotary_config=rc,
            attention_groups=(
                FullAttentionGroupConfig(
                    name="full", layer_ids=(0, 1), num_kv_heads=1, head_dim=LATENT,
                    rotary_config=rc, mla=True,
                    index_head_dim=index_dim, num_index_layers=n_idx,
                ),
            ),
        )

    dsa = create_kvcache_pool(model_config=cfg(IDX_DIM, 1), num_pages=8, page_size=1,
                              device=torch.device("cuda"), dtype=torch.bfloat16)
    assert isinstance(dsa, DSAKVCache)
    mla = create_kvcache_pool(model_config=cfg(0, 0), num_pages=8, page_size=1,
                              device=torch.device("cuda"), dtype=torch.bfloat16)
    assert isinstance(mla, MLAKVCache) and not isinstance(mla, DSAKVCache)


def test_mla_pool_backs_only_declared_global_layer_ids():
    """Hybrid linear/MLA models must not allocate latent KV for linear layers."""
    from freetoken.kvcache.dsa_pool import MLAKVCache

    pool = MLAKVCache(
        latent_dim=LATENT,
        num_layers=8,
        num_pages=4,
        page_size=1,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        layer_ids=(3, 7),
    )
    assert pool._kv_buffer.shape == (1, 2, 4, 1, 1, LATENT)
    assert pool.latent_rows(3).shape == (4, LATENT)
    assert pool.latent_rows(7).shape == (4, LATENT)
    with pytest.raises(KeyError, match="no paged KV storage"):
        pool.latent_rows(0)


def test_rebuild_shrink_and_engine_wiring():
    """Shrink-direction rebuild (both slabs) + the engine-facing rebuild_from_config
    (usable pages in, +1 dummy page added by the pool)."""
    pool = _pool(129)
    pool.rebuild(17)
    assert pool.latent_rows(0).shape[0] == 17
    assert pool.index_k_cache(0).shape[0] == 17

    pool.rebuild_from_config(config=None, num_pages=63)
    assert pool.latent_rows(0).shape[0] == 64  # 63 + 1 dummy page
    assert pool.index_k_cache(0).shape[0] == 64
