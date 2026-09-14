"""NVFP4 KV against independent torch rounding and dense attention references."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kvcache.mha_pool import MHAKVCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _pool(dim=128, heads=2, slots=96, layer_ids=None):
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    return MHAKVCache(heads, 4, dim, slots // 4, 4, torch.bfloat16,
                      torch.device("cuda"), layer_ids=layer_ids, kv_quant="nvfp4")


def _reference(x):
    shape = x.shape
    x = x.float().reshape(*shape[:-1], -1, 16)
    row = x.abs().flatten(-2).amax(-1).clamp_min(1e-10) / 2688.0
    block = (x.abs().amax(-1) / (6 * row[..., None])).clamp_max(448).to(torch.float8_e4m3fn)
    denom = block.float() * row[..., None]
    normalized = torch.where(denom[..., None] > 0, x / denom[..., None], 0)
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    distance = (normalized.abs()[..., None] - grid).abs()
    nearest = distance == distance.amin(-1, keepdim=True)
    codes = torch.arange(8, device=x.device).expand_as(distance)
    # Prefer even codes on ties, independently of the kernel's threshold encoding.
    rank = torch.where(nearest, codes % 2 * 8 + codes, 32)
    code = rank.argmin(-1) | ((normalized < 0).long() * 8)
    code = code.reshape(shape)
    packed = (code[..., ::2] | (code[..., 1::2] << 4)).to(torch.uint8)
    return packed, block.view(torch.uint8), row


def _decode(pool, which, layer=1):
    codes = getattr(pool, f"{which}_cache")(layer).flatten(0, 1)
    block = getattr(pool, f"{which}_block_scale")(layer).view(torch.float8_e4m3fn).float()
    row = getattr(pool, f"{which}_scale")(layer)
    code = torch.stack((codes & 15, codes >> 4), -1).flatten(-2).long()
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                         0, -.5, -1, -1.5, -2, -3, -4, -6], device=codes.device)
    return grid[code] * block.repeat_interleave(16, -1) * row[..., None]


def _decode_latent(pool, layer=0):
    codes = pool.latent_rows(layer)
    code = torch.stack((codes & 15, codes >> 4), -1).flatten(-2).long()
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                         0, -.5, -1, -1.5, -2, -3, -4, -6], device=codes.device)
    block = pool.latent_block_scale(layer).view(torch.float8_e4m3fn).float()
    return grid[code] * block.repeat_interleave(16, -1) * pool.latent_scale(layer)[:, None]


@pytest.mark.parametrize("rope", [0, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_latent_scatter_matches_reference_and_preserves_prefix(rope, dtype):
    from freetoken.kvcache.dsa_pool import MLAKVCache

    torch.manual_seed(73)
    dim = 512 + rope
    pool = MLAKVCache(dim, 4, 2, 64, dtype, torch.device("cuda"),
                      layer_ids=(1, 3), kv_quant="nvfp4")
    x = torch.randn(7, dim + 32, device="cuda", dtype=dtype)[:, :dim]
    x[0].zero_()
    x[1, :16] *= 100
    x[2] *= 1e-5
    loc = torch.tensor([64, 3, 127, 14, 6, 90, 31], device="cuda")
    pool.store_kv(x[:4, :512], x[:4, 512:], loc[:4], 3)
    before = [v.clone() for v in (pool.latent_rows(3), pool.latent_scale(3),
                                  pool.latent_block_scale(3))]
    pool.store_kv(x[4:, :512], x[4:, 512:], loc[4:], 3)
    pool.store_kv(x[:0, :512], x[:0, 512:], loc[:0], 3)
    for got, saved in zip((pool.latent_rows(3), pool.latent_scale(3),
                           pool.latent_block_scale(3)), before):
        torch.testing.assert_close(got[loc[:4]], saved[loc[:4]], rtol=0, atol=0)
    packed, block, row = _reference(x)
    torch.testing.assert_close(pool.latent_rows(3)[loc], packed)
    torch.testing.assert_close(pool.latent_block_scale(3)[loc], block)
    torch.testing.assert_close(pool.latent_scale(3)[loc], row)
    assert torch.count_nonzero(pool.latent_rows(1)) == 0
    assert torch.isfinite(_decode_latent(pool, 3)).all()
    assert pool.k_cache(3).data_ptr() == pool.v_cache(3).data_ptr()
    # Reused physical slots replace codes and both scales together.
    x[4:].mul_(0.125)
    pool.store_kv(x[4:, :512], x[4:, 512:], loc[4:], 3)
    packed, block, row = _reference(x[4:])
    torch.testing.assert_close(pool.latent_rows(3)[loc[4:]], packed)
    torch.testing.assert_close(pool.latent_block_scale(3)[loc[4:]], block)
    torch.testing.assert_close(pool.latent_scale(3)[loc[4:]], row)


@pytest.mark.parametrize("rope", [0, 64])
@pytest.mark.parametrize("splits", [0, 4])
@pytest.mark.parametrize("queries,broadcast", [(1, False), (5, False), (5, True)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sparse_mla_nvfp4_matches_restored_reference(rope, splits, queries, broadcast, dtype):
    from freetoken.kernel.triton.glm_dsa_sparse import glm_dsa_sparse_attn
    from freetoken.kvcache.dsa_pool import MLAKVCache

    torch.manual_seed(74)
    dim, n = 512 + rope, 67
    pool = MLAKVCache(dim, 1, 2, 64, torch.bfloat16, torch.device("cuda"), kv_quant="nvfp4")
    x = torch.randn(n, dim, device="cuda", dtype=torch.bfloat16)
    x *= torch.linspace(.2, 2, n, device="cuda")[:, None]
    if rope:
        x[:, 512:] *= 3
    loc = torch.randperm(128, device="cuda")[:n]
    pool.store_kv(x[:, :512], x[:, 512:], loc, 0)
    q = torch.randn(2, queries, 19, dim, device="cuda", dtype=dtype)
    sel = loc.repeat(2, 1 if broadcast else queries, 1).to(torch.int32)
    sel[1] = sel[1].flip(-1)
    sel[..., 5] = -1
    cnt = torch.full((2, queries), n, device="cuda", dtype=torch.int32)
    cnt[0, 0] = 0
    cnt[1, 0] = 13
    out = glm_dsa_sparse_attn(
        q, pool.latent_rows(0), sel, .04, counts=cnt, d_v=512,
        pool_scale=pool.latent_scale(0), pool_block_scale=pool.latent_block_scale(0),
        kv_quant="nvfp4", force_splits=splits,
    )
    decoded = _decode_latent(pool)
    ref = torch.zeros_like(out, dtype=torch.float32)
    for b in range(2):
        for m in range(queries):
            rows = sel[b, 0 if broadcast else m, :int(cnt[b, m])]
            rows = rows[rows >= 0].long()
            if rows.numel():
                kv = decoded[rows]
                ref[b, m] = (q[b, m].float() @ kv.T * .04).softmax(-1) @ kv[:, :512]
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("dim", [16, 48, 64, 128, 256, 512])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_scatter_matches_independent_reference(dim, dtype):
    torch.manual_seed(41)
    pool = _pool(dim)
    # K is a projection slice; V is independently contiguous.
    k = torch.randn(5, 4 * dim, device="cuda", dtype=dtype)[:, :2 * dim]
    v = torch.randn(5, 2 * dim, device="cuda", dtype=dtype)
    k[0].zero_()
    k[1, :16] *= 100
    v[2] *= 1e-5
    loc = torch.tensor([7, 1, 95, 31, 12], device="cuda", dtype=torch.int64)
    pool.store_kv(k, v, loc, 1)
    for which, source in (("k", k), ("v", v)):
        packed, block, row = _reference(source.reshape(5, 2, dim))
        torch.testing.assert_close(getattr(pool, f"{which}_cache")(1).flatten(0, 1)[loc], packed)
        torch.testing.assert_close(getattr(pool, f"{which}_block_scale")(1)[loc], block)
        torch.testing.assert_close(getattr(pool, f"{which}_scale")(1)[loc], row)
        assert torch.isfinite(_decode(pool, which)).all()
        assert torch.count_nonzero(_decode(pool, which)[0]) == 0


def test_e2m1_grid_and_round_to_even_boundaries():
    positive = torch.tensor([0, .25, .5, .75, 1, 1.25, 1.5, 1.75,
                             2, 2.5, 3, 3.5, 4, 5, 6], device="cuda")
    values = torch.cat((positive, -positive))
    values = torch.cat((values, values.nextafter(torch.full_like(values, float("inf"))),
                        values.nextafter(torch.full_like(values, -float("inf")))))
    pool = _pool(dim=32, slots=192)
    rows = torch.zeros(values.numel(), 2, 32, device="cuda")
    rows[:, :, 0] = values[:, None]
    rows[:, :, 15] = 6
    rows[:, :, 31] = 2688  # Forces row_scale=1 and the first block_scale=1.
    loc = torch.arange(values.numel(), device="cuda", dtype=torch.int32)
    pool.store_kv(rows.flatten(1), rows.flatten(1), loc, 1)
    packed, block, row = _reference(rows)
    torch.testing.assert_close(pool.k_cache(1).flatten(0, 1)[loc], packed)
    torch.testing.assert_close(pool.k_block_scale(1)[loc], block)
    torch.testing.assert_close(pool.k_scale(1)[loc], row)


@pytest.mark.parametrize("dim", [64, 128, 256, 512])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["paged", "decode", "extend", "split"])
def test_attention_reads_packed_cache(dim, dtype, mode):
    from freetoken.kernel.triton.attention import (
        decode_paged_attention, extend_paged_attention, paged_attention,
    )

    torch.manual_seed(42)
    pool = _pool(dim)
    n, prefix = 67, 35
    k, v = [torch.randn(n, 2 * dim, device="cuda", dtype=dtype) for _ in range(2)]
    # Distinct token/head and block scales expose accidental FP8 rescaling of NVFP4.
    row_gain = torch.linspace(0.25, 2.0, n * 2, device="cuda").view(n, 2, 1)
    block_gain = torch.linspace(0.5, 1.5, dim // 16, device="cuda").repeat_interleave(16)
    k.view(n, 2, dim).mul_(row_gain * block_gain)
    v.view(n, 2, dim).mul_(row_gain.flip(0) * block_gain.flip(0))
    loc = torch.randperm(95, device="cuda")[:n] + 1
    pool.store_kv(k, v, loc, 1)
    tokens = 1 if mode in ("paged", "decode") else n - prefix
    q = torch.randn(tokens, 6, dim, device="cuda", dtype=dtype)
    indptr = torch.tensor([0, n], device="cuda", dtype=torch.int32)
    pos = torch.arange(n - tokens, n, device="cuda")
    args = dict(q=q, k_cache=pool.k_cache(1).flatten(0, 1),
                v_cache=pool.v_cache(1).flatten(0, 1), k_scale=pool.k_scale(1),
                v_scale=pool.v_scale(1), k_block_scale=pool.k_block_scale(1),
                v_block_scale=pool.v_block_scale(1), kv_quant="nvfp4", sm_scale=dim ** -.5)
    kd, vd = _decode(pool, "k")[loc], _decode(pool, "v")[loc]
    if mode == "split":
        kd[prefix:] = k[prefix:].view(-1, 2, dim).float()
        vd[prefix:] = v[prefix:].view(-1, 2, dim).float()
    # Tensor-core paths round restored K/V to the query dtype before dot products.
    if mode != "paged":
        kd, vd = kd.to(q.dtype).float(), vd.to(q.dtype).float()
    kd, vd = [x.repeat_interleave(3, 1).transpose(0, 1) for x in (kd, vd)]
    score = torch.einsum("thd,hnd->thn", q.float(), kd) * dim ** -.5
    score.masked_fill_(torch.arange(n, device="cuda")[None, None, :] > pos[:, None, None], -float("inf"))
    ref = torch.einsum("thn,hnd->thd", score.softmax(-1), vd).to(q.dtype)
    if mode == "paged":
        actual = paged_attention(**args, indptr=indptr, indices=loc,
                                 q_to_req=torch.zeros(tokens, device="cuda", dtype=torch.int32), q_positions=pos)
    elif mode == "decode":
        actual = decode_paged_attention(**args, indptr=indptr, indices=loc, q_positions=pos,
            attn_logits=torch.empty(1, 6, 8, dim, device="cuda"),
            attn_lse=torch.empty(1, 6, 8, device="cuda"),
            num_kv_splits=torch.tensor([8], device="cuda", dtype=torch.int32), max_kv_splits=8)
    else:
        extra = {} if mode == "extend" else dict(k_extend=k[prefix:].view(-1, 2, dim), v_extend=v[prefix:].view(-1, 2, dim))
        actual = extend_paged_attention(**args, **extra,
            qo_indptr=torch.tensor([0, tokens], device="cuda", dtype=torch.int32),
            kv_indptr=indptr, kv_indices=loc,
            prefix_lens=torch.tensor([prefix], device="cuda", dtype=torch.int32), max_q_len=tokens)
    torch.testing.assert_close(actual, ref, atol=0.008, rtol=0.025)


def test_pool_budget_rebuild_and_layer_mapping():
    from freetoken.kvcache.base import spec_kv_bytes_per_token
    from freetoken.models.config import KVCacheGroupSpec

    pool = _pool(layer_ids=(1, 3))
    spec = KVCacheGroupSpec(name="full", layer_ids=(1, 3), num_kv_heads=2, head_dim=128, sliding_window=None)
    cfg = SimpleNamespace(kv_quant="nvfp4", dtype=torch.bfloat16, tp_info=SimpleNamespace(size=1))
    assert spec_kv_bytes_per_token(spec, cfg) == pool.unit_bytes()[0] == 2 * 2 * 2 * 76
    with pytest.raises(KeyError):
        pool.k_block_scale(0)
    for pages in (32, 8):
        pool.rebuild(pages)
        assert pool.k_cache(3).shape == (pages, 4, 2, 64)
        assert pool.k_block_scale(3).shape == (pages * 4, 2, 8)
        assert pool.unit_bytes()[0] == 608
        assert torch.count_nonzero(_decode(pool, "k", 3)) == 0


def test_store_cuda_graph_replay_changes_slots():
    pool = _pool()
    k = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
    loc = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    pool.store_kv(k, k, loc, 1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pool.store_kv(k, k, loc, 1)
    k.mul_(2)
    loc.copy_(torch.tensor([4, 9], device="cuda", dtype=torch.int32))
    graph.replay()
    expected, _, _ = _reference(k.view(2, 2, 128))
    torch.testing.assert_close(pool.k_cache(1).flatten(0, 1)[loc], expected)


def test_hybrid_swa_pool_packs_both_groups_and_rebuilds(monkeypatch):
    from freetoken.distributed.info import DistributedInfo
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
    from freetoken.models.config import KVCacheGroupSpec

    monkeypatch.setattr(
        "freetoken.kvcache.hybrid_swa_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )
    groups = (
        KVCacheGroupSpec("full", (1,), 2, 64, None),
        KVCacheGroupSpec("swa", (0,), 2, 128, 32),
    )
    pool = HybridSWAKVCache(
        groups=groups,
        num_layers=2,
        num_full_pages=12,
        page_size=1,
        num_swa_tokens=18,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        kv_quant="nvfp4",
    )
    assert pool.k_cache(1).shape == (12, 1, 2, 32)
    assert pool.k_cache(0).shape == (18, 1, 2, 64)
    assert pool.k_block_scale(1).shape == (12, 2, 4)
    assert pool.k_block_scale(0).shape == (18, 2, 8)
    assert pool.unit_bytes() == (160, 304)

    full_loc = torch.tensor([4, 9], device="cuda", dtype=torch.int32)
    swa_loc = torch.tensor([2, 7], device="cuda", dtype=torch.int32)
    pool.alloc_swa(swa_loc)
    for layer, dim, loc, cache_loc in ((1, 64, full_loc, full_loc), (0, 128, swa_loc, None)):
        rows = torch.randn(2, 2 * dim, device="cuda", dtype=torch.bfloat16)
        pool.store_kv(rows, rows, loc, layer)
        if cache_loc is None:
            cache_loc = pool.translate_loc_from_full_to_swa(loc)
        expected, block, row = _reference(rows.view(2, 2, dim))
        codes = pool.k_cache(layer).flatten(0, 1)[cache_loc]
        torch.testing.assert_close(codes, expected)
        torch.testing.assert_close(pool.k_block_scale(layer)[cache_loc], block)
        torch.testing.assert_close(pool.k_scale(layer)[cache_loc], row)

    pool.rebuild(num_full_pages=6, num_swa_tokens=10)
    assert pool.k_cache(1).shape == (6, 1, 2, 32)
    assert pool.k_cache(0).shape == (10, 1, 2, 64)
    assert pool.unit_bytes() == (160, 304)
    assert pool.swa_available_size() == 9


def test_backend_decode_graph_replays_new_kv_and_page_tables(monkeypatch):
    from freetoken.attention.triton import TritonAttentionBackend, TritonMetadata
    from freetoken.kernel.triton.attention import paged_attention

    pool = _pool()
    monkeypatch.setattr("freetoken.attention.triton.get_global_ctx",
                        lambda: SimpleNamespace(kv_cache=pool))
    backend = TritonAttentionBackend(SimpleNamespace(num_qo_heads=6, head_dim=128))
    q = torch.randn(2, 6, 128, device="cuda", dtype=torch.bfloat16)
    k, v = [torch.randn(2, 256, device="cuda", dtype=q.dtype) for _ in range(2)]
    loc = torch.tensor([7, 11], device="cuda", dtype=torch.int32)
    indptr = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    positions = torch.zeros(2, device="cuda", dtype=torch.int32)
    indices = loc.clone()
    req = torch.arange(2, device="cuda", dtype=torch.int32)
    metadata = TritonMetadata(cu_seqlens_q_gpu=indptr, indptr=indptr, indices=indices,
        q_to_req=req, q_positions=positions, is_decode=True, prefix_lens=positions,
        max_q_len=1)
    batch = SimpleNamespace(out_loc=loc, attn_metadata=metadata)
    backend.forward(q, k, v, 1, batch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = backend.forward(q, k, v, 1, batch)
    for new_slots in ([31, 44], [7, 11]):
        k.normal_()
        v.normal_()
        q.normal_()
        loc.copy_(torch.tensor(new_slots, device="cuda", dtype=torch.int32))
        indices.copy_(loc)
        graph.replay()
        ref = paged_attention(q, _decode(pool, "k").to(q.dtype),
            _decode(pool, "v").to(q.dtype), indptr, indices, req, positions, 128 ** -.5)
        torch.testing.assert_close(actual, ref, atol=0.008, rtol=0.025)
