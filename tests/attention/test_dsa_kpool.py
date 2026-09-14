"""Glm5NextDSABackend (kpool indexer) vs an eager reference.

Exercises the full path -- raw K/gate stores, pool-completion compression,
pool-granular scoring, top-k + expansion + tail, gathered sparse MLA -- on a
single request at toy dims (Hi=16 index heads, Di=64, kpool=4, topk=32):

* kv_len <= index_topk: the identity/dense path must equal full softmax MLA.
* kv_len  > index_topk: prefill queries must match a subset-softmax reference
  built from an independently computed pooled-score top-k (+ tail).
* decode: pooled entries appear exactly at pool-completion steps and match the
  softmax(gate+APE) reference; decode outputs match the same subset reference.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

H, LATENT = 2, 64  # MLA heads, kv_lora_rank (== latent width: NoPE has no kpe half)
HI, DI = 16, 64  # index heads (kernel needs >= 16), index head dim (pow2)
KPOOL, TOPK = 4, 32
SM_SCALE = 0.125
DEV = "cuda"


def _args(num_layers=1):
    from freetoken.models.glm5_next.args import Glm5NextArgs

    return Glm5NextArgs(
        hidden_size=32, num_heads=H,
        q_lora_rank=16, kv_lora_rank=LATENT, qk_nope_head_dim=LATENT,
        qk_rope_head_dim=0, v_head_dim=LATENT, mla_nope=True, norm_eps=1e-5,
        max_position=4096,
        index_n_heads=HI, index_head_dim=DI, index_topk=TOPK,
        indexer_types=("full",) * num_layers, indexer_rope_interleave=True,
        index_kpool=KPOOL, index_kpool_compress=True,
        index_kpool_always_select_tail=True,
        linear_num_heads=0, linear_head_dim=0, linear_conv_kernel_dim=4,
        linear_lower_bound=-5.0,
        layer_types=("deepseek_sparse_attention",) * num_layers,
        mlp_layer_types=("dense",) * num_layers,
        mhc=False, mhc_num_residual_streams=1, hc_eps=1e-6,
        mhc_sinkhorn_iterations=0, mhc_tau=0.05, mhc_post_mult_value=2.0,
        mhc_no_norm_weight=False, swiglu_limit=None, rope_theta=10000.0,
    )


@pytest.fixture()
def harness(monkeypatch):
    from freetoken.attention.dsa_indexer_kpool import Glm5NextDSABackend
    from freetoken.kvcache.dsa_pool import KpoolDSAKVCache

    pool = KpoolDSAKVCache(
        latent_dim=LATENT, num_layers=1, num_pages=8, page_size=64,
        dtype=torch.bfloat16, device=torch.device(DEV),
        index_head_dim=DI, num_index_layers=1,
        index_ratio=KPOOL, num_req_slots=4,
    )
    page_table = torch.full((4, 512), -1, dtype=torch.int32, device=DEV)
    page_table[0, :512] = torch.arange(512, dtype=torch.int32, device=DEV)
    ctx = SimpleNamespace(kv_cache=pool, page_table=page_table)
    monkeypatch.setattr("freetoken.attention.dsa.get_global_ctx", lambda: ctx)

    config = SimpleNamespace(
        glm5_args=_args(), glm_dsa_args=None, num_qo_heads=H,
        attn_sm_scale=SM_SCALE, num_layers=1,
    )
    backend = Glm5NextDSABackend(config)
    torch.manual_seed(0)
    # The APE is a MODEL parameter, passed per call via DSAIndexerInputs.
    ape = torch.randn(KPOOL, DI, device=DEV, dtype=torch.float32) * 0.3
    return backend, pool, ape


def _req(device_len, cached_len=0):
    return SimpleNamespace(
        table_idx=0, device_len=device_len, extend_len=device_len - cached_len,
        cached_len=cached_len, linear_slot_idx=None,
    )


def _prefill_batch(t0, t1):
    return SimpleNamespace(
        phase="prefill", padded_reqs=[_req(t1, t0)], reqs=[_req(t1, t0)],
        positions=torch.arange(t0, t1, device=DEV),
        out_loc=torch.arange(t0, t1, device=DEV),
        active_table_idx=None,
    )


def _decode_batch(pos):
    return SimpleNamespace(
        phase="decode", padded_reqs=[_req(pos + 1, pos)], reqs=[_req(pos + 1, pos)],
        positions=torch.tensor([pos], device=DEV),
        out_loc=torch.tensor([pos], device=DEV),
        active_table_idx=torch.tensor([0], device=DEV),
    )


def _rand_seq(total, seed=1):
    torch.manual_seed(seed)
    mk = lambda *s: torch.randn(*s, device=DEV, dtype=torch.bfloat16)
    return dict(
        q_nope=mk(total, H, LATENT), c_kv=mk(total, LATENT),
        qi=mk(total, HI, DI), ki=mk(total, DI),
        wi=(torch.randn(total, HI, device=DEV).float() * 0.5),
        gate=mk(total, DI),
    )


def _run(backend, batch, d, sl, ape):
    from freetoken.attention.dsa import DSAIndexerInputs

    t = batch.positions.shape[0]
    backend.prepare_metadata(batch)
    return backend.mla_forward(
        d["q_nope"][sl], d["q_nope"].new_empty(t, H, 0),
        d["c_kv"][sl], d["c_kv"].new_empty(t, 0),
        0, batch,
        indexer_inputs=DSAIndexerInputs(
            q=d["qi"][sl], k=d["ki"][sl], w=d["wi"][sl],
            gate=d["gate"][sl], ape=ape,
        ),
    )


# ---- eager references -----------------------------------------------------------------


def _ref_pooled(d, ape, n_pools):
    """[n_pools, DI] softmax(gate+ape)-weighted pooled keys (fp32 -> bf16)."""
    k = d["ki"][: n_pools * KPOOL].view(n_pools, KPOOL, DI).float()
    g = d["gate"][: n_pools * KPOOL].view(n_pools, KPOOL, DI).float()
    w = torch.softmax(g + ape, dim=1)
    return (w * k).sum(1).to(torch.bfloat16)


def _ref_scores(d, ape, q_idx_t, w_t, n_pools):
    """Pool scores for one query: sum_h w_h * relu(q_h . k_pool) * DI**-0.5."""
    kp = _ref_pooled(d, ape, n_pools).float()
    s = torch.relu(q_idx_t.float() @ kp.T)  # [HI, n_pools]
    return ((w_t * DI**-0.5).unsqueeze(1) * s).sum(0)


def _ref_attend(d, q_t, positions):
    """Full softmax MLA over latent rows at ``positions`` for one query [H, LATENT]."""
    lat = d["c_kv"][positions].float()  # [n, LATENT]
    logits = q_t.float() @ lat.T * SM_SCALE  # [H, n]
    p = torch.softmax(logits, dim=-1)
    return (p @ lat).to(torch.bfloat16)


def _ref_selected_positions(d, ape, q_idx_t, w_t, pos):
    """Reference kpool selection for a query at ``pos``: top-k complete pools
    expanded to tokens, plus the tail [n_pools*KPOOL, pos]."""
    n_pools = (pos + 1) // KPOOL
    sel_pools = min(TOPK // KPOOL, n_pools)
    picked = torch.topk(
        _ref_scores(d, ape, q_idx_t, w_t, n_pools), sel_pools
    ).indices.tolist()
    positions = [p * KPOOL + o for p in picked for o in range(KPOOL)]
    positions += list(range(n_pools * KPOOL, pos + 1))
    return sorted(set(positions))


def test_dense_path_matches_full_softmax(harness):
    backend, pool, ape = harness
    total = 20  # < TOPK -> identity/dense path
    d = _rand_seq(total)
    out = _run(backend, _prefill_batch(0, total), d, slice(0, total), ape)
    for t in range(total):
        ref = _ref_attend(d, d["q_nope"][t], list(range(t + 1)))
        err = (out[t].float() - ref.float()).abs().max().item()
        assert err < 2e-2, f"dense query {t}: err {err}"


def test_sparse_prefill_matches_reference(harness):
    backend, pool, ape = harness
    total = 60  # > TOPK -> kpool scoring
    d = _rand_seq(total, seed=2)
    out = _run(backend, _prefill_batch(0, total), d, slice(0, total), ape)

    # Pooled entries in the slab match the compression reference.
    n_pools = total // KPOOL
    # Shadow slab: pool p lives at token_slot // KPOOL == p (identity page table).
    slab = pool.index_k_cache(0)[torch.arange(n_pools, device=DEV)]
    ref_pool = _ref_pooled(d, ape, n_pools)
    assert (slab.float() - ref_pool.float()).abs().max().item() < 2e-2

    for t in (35, 47, 59):  # queries past TOPK (sparse regime)
        sel = _ref_selected_positions(d, ape, d["qi"][t], d["wi"][t], t)
        ref = _ref_attend(d, d["q_nope"][t], sel)
        err = (out[t].float() - ref.float()).abs().max().item()
        assert err < 3e-2, f"sparse query {t}: err {err}"


def test_decode_completion_and_selection(harness):
    backend, pool, ape = harness
    total, extra = 60, 6  # decode positions 60..65; completion at 63
    d = _rand_seq(total + extra, seed=3)
    _run(backend, _prefill_batch(0, total), d, slice(0, total), ape)

    for pos in range(total, total + extra):
        out = _run(backend, _decode_batch(pos), d, slice(pos, pos + 1), ape)
        sel = _ref_selected_positions(d, ape, d["qi"][pos], d["wi"][pos], pos)
        ref = _ref_attend(d, d["q_nope"][pos], sel)
        err = (out[0].float() - ref.float()).abs().max().item()
        assert err < 3e-2, f"decode pos {pos}: err {err}"

        if pos % KPOOL == KPOOL - 1:  # a pool completed this step
            n_pools = (pos + 1) // KPOOL
            row = pos // KPOOL  # the pool's shadow row
            got = pool.index_k_cache(0)[row]
            want = _ref_pooled(d, ape, n_pools)[-1]
            assert (got.float() - want.float()).abs().max().item() < 2e-2


def test_sparse_batch_with_sub_pool_request(harness):
    """A sparse prefill batch may carry a request shorter than one pool: it has nothing
    to score and must come out as the dense (tail-only) attention over its own rows."""
    from freetoken.attention import dsa as dsa_mod

    backend, pool, ape = harness
    dsa_mod.get_global_ctx().page_table[1, :64] = torch.arange(
        448, 512, dtype=torch.int32, device=DEV
    )
    ta = 60  # > TOPK -> the batch takes the sparse path
    for tb in (1, 2, 3):
        d = _rand_seq(ta + tb, seed=6)
        req_b = _req(tb)
        req_b.table_idx = 1
        reqs = [_req(ta), req_b]
        batch = SimpleNamespace(
            phase="prefill", padded_reqs=reqs, reqs=reqs,
            positions=torch.cat([torch.arange(ta), torch.arange(tb)]).to(DEV),
            out_loc=torch.cat([torch.arange(ta), torch.arange(448, 448 + tb)]).to(DEV),
            active_table_idx=None,
        )
        out = _run(backend, batch, d, slice(0, ta + tb), ape)
        for j in range(tb):
            t = ta + j
            ref = _ref_attend(d, d["q_nope"][t], list(range(ta, t + 1)))
            err = (out[t].float() - ref.float()).abs().max().item()
            assert err < 3e-2, f"sub-pool request len {tb}, query {j}: err {err}"
        t = ta - 1
        sel = _ref_selected_positions(d, ape, d["qi"][t], d["wi"][t], t)
        ref = _ref_attend(d, d["q_nope"][t], sel)
        assert (out[t].float() - ref.float()).abs().max().item() < 3e-2


def test_chunked_prefill_mid_pool_start(harness):
    """A chunk may start MID-POOL (main's soft prefill_chunk_align keeps an
    unaligned end when the budget cannot fill a page): the straddling pool's
    older members come from the tail ring and the pooled slab + outputs must
    match the single-shot run."""
    backend, pool, ape = harness
    total, split = 60, 30  # split % KPOOL == 2 -> pool 7 straddles the chunks
    d = _rand_seq(total, seed=4)
    out1 = _run(backend, _prefill_batch(0, split), d, slice(0, split), ape)
    out2 = _run(backend, _prefill_batch(split, total), d, slice(split, total), ape)

    n_pools = total // KPOOL
    slab = pool.index_k_cache(0)[torch.arange(n_pools, device=DEV)]
    ref_pool = _ref_pooled(d, ape, n_pools)
    assert (slab.float() - ref_pool.float()).abs().max().item() < 2e-2

    for t in (35, 47, 59):
        sel = _ref_selected_positions(d, ape, d["qi"][t], d["wi"][t], t)
        ref = _ref_attend(d, d["q_nope"][t], sel)
        err = (out2[t - split].float() - ref.float()).abs().max().item()
        assert err < 3e-2, f"mid-pool chunked query {t}: err {err}"


def test_interleaved_decode_requests_do_not_pollute_rings(harness):
    """Two requests decoding in alternation: tail rings and shadow rows are keyed
    by table_idx, so neither request's pools may absorb the other's raw K/gate."""
    from freetoken.attention import dsa as dsa_mod

    backend, pool, ape = harness
    dsa_mod.get_global_ctx().page_table[1, :128] = torch.arange(
        256, 384, dtype=torch.int32, device=DEV
    )
    total, extra = 60, 6
    streams = {  # table_idx -> (data, physical row base)
        0: (_rand_seq(total + extra, seed=7), 0),
        1: (_rand_seq(total + extra, seed=8), 256),
    }

    def _batch_for(table, phase, t0, t1):
        base = streams[table][1]
        req = _req(t1, t0)
        req.table_idx = table
        return SimpleNamespace(
            phase=phase, padded_reqs=[req], reqs=[req],
            positions=torch.arange(t0, t1, device=DEV),
            out_loc=torch.arange(base + t0, base + t1, device=DEV),
            active_table_idx=(
                torch.tensor([table], device=DEV) if phase == "decode" else None
            ),
        )

    for table in (0, 1):
        d = streams[table][0]
        _run(backend, _batch_for(table, "prefill", 0, total), d, slice(0, total), ape)

    for pos in range(total, total + extra):
        for table in (0, 1):  # alternate every step
            d, base = streams[table]
            out = _run(
                backend, _batch_for(table, "decode", pos, pos + 1), d,
                slice(pos, pos + 1), ape,
            )
            sel = _ref_selected_positions(d, ape, d["qi"][pos], d["wi"][pos], pos)
            ref = _ref_attend(d, d["q_nope"][pos], sel)
            err = (out[0].float() - ref.float()).abs().max().item()
            assert err < 3e-2, f"table {table} decode pos {pos}: err {err}"

            if pos % KPOOL == KPOOL - 1:
                n_pools = (pos + 1) // KPOOL
                row = (base + pos) // KPOOL
                got = pool.index_k_cache(0)[row]
                want = _ref_pooled(d, ape, n_pools)[-1]
                err = (got.float() - want.float()).abs().max().item()
                assert err < 2e-2, f"table {table} pool at pos {pos}: err {err}"


def test_padding_and_empty_batch_leave_shadow_rows_clean():
    """The compression kernel writes every row somewhere, but masked-off rows
    (padding request == -1, or a pool that cannot close yet) must land only on
    their designated scratch rows -- the shadow region and the rings stay clean.
    An empty batch is a no-op."""
    from freetoken.kernel.triton.kpool_compress import kpool_compress_store

    shadow_n, n_req = 8, 2
    slab = torch.full((shadow_n + n_req, DI), 7.0, dtype=torch.bfloat16, device=DEV)
    ring_k = torch.full((n_req * KPOOL, DI), 3.0, dtype=torch.bfloat16, device=DEV)
    ring_g = ring_k.clone()
    ape = torch.randn(KPOOL, DI, dtype=torch.float32, device=DEV)
    k = torch.randn(2, DI, dtype=torch.bfloat16, device=DEV)
    gate = torch.randn(2, DI, dtype=torch.bfloat16, device=DEV)

    kpool_compress_store(
        k, gate, ring_k, ring_g, ape,
        ring_slots=torch.tensor([0, 1], dtype=torch.int32, device=DEV),
        token_to_req=torch.tensor([0, -1], dtype=torch.int32, device=DEV),
        cu_seqlens=torch.tensor([0, 1, 1], dtype=torch.int32, device=DEV),
        positions=torch.tensor([0, 0], device=DEV),  # pos 0: no pool can close
        slab=slab, cmp_rows=torch.tensor([8, 9], dtype=torch.int32, device=DEV),
        ratio=KPOOL,
    )
    assert torch.equal(slab[:shadow_n], torch.full_like(slab[:shadow_n], 7.0))
    assert torch.equal(ring_k, torch.full_like(ring_k, 3.0))
    assert torch.equal(ring_g, ring_k)

    before = slab.clone()
    kpool_compress_store(
        k[:0], gate[:0], ring_k, ring_g, ape,
        ring_slots=torch.tensor([0], dtype=torch.int32, device=DEV),
        token_to_req=torch.empty(0, dtype=torch.int32, device=DEV),
        cu_seqlens=torch.tensor([0, 0], dtype=torch.int32, device=DEV),
        positions=torch.empty(0, dtype=torch.int64, device=DEV),
        slab=slab, cmp_rows=torch.empty(0, dtype=torch.int32, device=DEV),
        ratio=KPOOL,
    )
    assert torch.equal(slab, before)
