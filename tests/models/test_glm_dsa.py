"""GLM-5.2 DSA numerical equivalence vs the HF reference (`modeling_glm_moe_dsa`).

Validates the three moving parts against the checkpoint-paired reference:
  * indexer scoring + causal top-k selection (rope treatment, relu/scale/weights math)
  * the Triton gathered-KV sparse MLA kernel vs a subset-softmax reference
  * dense == DSA exact equivalence for kv_len <= index_topk
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

H_IDX, D_IDX, ROPE_DIM, Q_LORA, HIDDEN = 32, 128, 64, 256, 512
THETA = 8_000_000.0


def _hf_indexer(seq: int, topk: int):
    """Run the HF reference indexer on random tensors; returns everything needed to
    replicate it (weights, inputs, expected topk indices)."""
    # HARD import: transformers is a core dependency and this test is the rope-
    # convention guard -- a silent skip would let the convention rot (the exact
    # env-gated failure mode that let the <= 5.12 half-split bug pass review).
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as tr

    class Cfg:  # minimal duck-typed config for GlmMoeDsaIndexer
        hidden_size = HIDDEN
        index_n_heads = H_IDX
        index_head_dim = D_IDX
        qk_rope_head_dim = ROPE_DIM
        index_topk = topk
        q_lora_rank = Q_LORA

    torch.manual_seed(7)
    idx = tr.GlmMoeDsaIndexer(Cfg(), layer_idx=0).cuda().to(torch.bfloat16)
    x = torch.randn(1, seq, HIDDEN, device="cuda", dtype=torch.bfloat16)
    q_resid = torch.randn(1, seq, Q_LORA, device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(seq, device="cuda")
    # cos/sin exactly as the HF rotary: freqs over ROPE_DIM, cat(freqs, freqs)
    inv = 1.0 / (THETA ** (torch.arange(0, ROPE_DIM, 2, device="cuda", dtype=torch.float) / ROPE_DIM))
    freqs = torch.outer(pos.float(), inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[None].to(torch.bfloat16), emb.sin()[None].to(torch.bfloat16)
    ref_topk = idx(x, q_resid, (cos, sin), None, pos[None])  # [1, S, topk]
    return idx, x, q_resid, pos, ref_topk


@pytest.mark.parametrize("seq,topk", [(300, 64), (128, 256)])
def test_indexer_matches_hf_reference(seq, topk):
    from freetoken.layers.rotary import get_rope

    idx_ref, x, q_resid, pos, ref_topk = _hf_indexer(seq, topk)

    # ours: same weights, config-driven INTERLEAVED partial rope (GLM-5.2 sets
    # indexer_rope_interleave=true; transformers >= 5.13 applies interleave in the
    # reference -- <= 5.12 wrongly half-split it, so this test requires >= 5.13)
    import transformers
    v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    # HARD FAIL, not skip: on < 5.13 the reference itself applies the wrong (half-
    # split) convention for GLM, so a skip here would silently disarm the guard.
    assert v >= (5, 13), (
        f"indexer parity requires transformers >= 5.13 (interleaved indexer rope); "
        f"this env has {transformers.__version__} -- upgrade the test env, do not skip."
    )
    with torch.device("cuda"):
        rope = get_rope(head_dim=D_IDX, rotary_dim=ROPE_DIM, max_position=4096, base=THETA, is_neox=False)
    q = (q_resid[0] @ idx_ref.wq_b.weight.T).view(seq, H_IDX * D_IDX)
    k = torch.nn.functional.layer_norm(
        x[0] @ idx_ref.wk.weight.T, (D_IDX,), idx_ref.k_norm.weight, idx_ref.k_norm.bias, 1e-6
    ).view(seq, D_IDX)
    q, k = rope.forward(pos, q.contiguous(), k.contiguous())
    q = q.view(seq, H_IDX, D_IDX)
    w = (x[0] @ idx_ref.weights_proj.weight.T).float() * (H_IDX**-0.5)

    s = torch.einsum("shd,td->sht", q, k)  # [S, H, T] bf16 (fp32 accumulate)
    s = torch.relu(s.float()) * (D_IDX**-0.5)
    scores = torch.einsum("sh,sht->st", w, s)
    cols = torch.arange(seq, device="cuda")
    scores = scores.masked_fill(cols[None, :] > pos[:, None], float("-inf"))
    ours_topk = scores.topk(min(topk, seq), dim=-1).indices

    # Selection SETS must agree (order within ties/topk may differ). Compare per query
    # over the causally-live prefix; allow a tiny boundary disagreement from bf16
    # rounding at the top-k cutoff (HF scores in fp32 matmul, we use bf16 tensor cores
    # with fp32 accumulate -- same as the reference's own fp8-kernel caveat).
    mismatch = 0
    for t in range(seq):
        live = t + 1
        k_eff = min(topk, seq)
        a = set(ref_topk[0, t][: min(k_eff, live)].tolist())
        b = set(ours_topk[t][: min(k_eff, live)].tolist())
        if live <= k_eff:
            assert a == b == set(range(live)), f"short-context selection must be all live rows @ {t}"
        else:
            mismatch += len(a ^ b)
    total = sum(min(topk, t + 1) for t in range(seq))
    assert mismatch / total < 0.02, f"topk set disagreement {mismatch}/{total}"

    # Negative control: the half-split (neox) convention -- what transformers <= 5.12
    # shipped for GLM and what this repo used before the fix -- must disagree BADLY
    # (measured at ~53%); if it ever agrees, the reference itself changed.
    with torch.device("cuda"):
        rope_bad = get_rope(head_dim=D_IDX, rotary_dim=ROPE_DIM, max_position=4095, base=THETA, is_neox=True)
    q2 = (q_resid[0] @ idx_ref.wq_b.weight.T).view(seq, H_IDX * D_IDX)
    k2 = torch.nn.functional.layer_norm(
        x[0] @ idx_ref.wk.weight.T, (D_IDX,), idx_ref.k_norm.weight, idx_ref.k_norm.bias, 1e-6
    ).view(seq, D_IDX)
    q2, k2 = rope_bad.forward(pos, q2.contiguous(), k2.contiguous())
    q2 = q2.view(seq, H_IDX, D_IDX)
    s2 = torch.relu(torch.einsum("shd,td->sht", q2, k2).float()) * (D_IDX**-0.5)
    scores2 = torch.einsum("sh,sht->st", w, s2)
    scores2 = scores2.masked_fill(cols[None, :] > pos[:, None], float("-inf"))
    bad_topk = scores2.topk(min(topk, seq), dim=-1).indices
    bad_mismatch = sum(
        len(set(ref_topk[0, t][: min(min(topk, seq), t + 1)].tolist())
            ^ set(bad_topk[t][: min(min(topk, seq), t + 1)].tolist()))
        for t in range(seq) if t + 1 > min(topk, seq)
    )
    live_total = sum(min(topk, t + 1) for t in range(seq) if t + 1 > min(topk, seq))
    if seq > topk:  # the control needs queries PAST the top-k boundary: only (300, 64)
        assert live_total > 0
        assert bad_mismatch / live_total > 0.10, (
            f"half-split unexpectedly agrees ({bad_mismatch}/{live_total}) -- reference convention changed?"
        )


def test_sparse_kernel_equals_dense_when_all_selected():
    """kv_len <= topk: sparse kernel over all live rows == dense softmax reference."""
    from freetoken.kernel.triton.glm_dsa_sparse import glm_dsa_sparse_attn

    torch.manual_seed(0)
    h, dv, dr, n = 40, 512, 64, 977
    q = torch.randn(1, 1, h, dv + dr, device="cuda", dtype=torch.bfloat16)
    pool = torch.randn(n + 13, dv + dr, device="cuda", dtype=torch.bfloat16) * 0.5
    rows = torch.randperm(n + 13, device="cuda")[:n].to(torch.int32)
    scale = (dv + dr) ** -0.5

    idx = torch.full((1, 1, 2048), -1, device="cuda", dtype=torch.int32)
    idx[0, 0, :n] = rows
    cnt = torch.tensor([[n]], device="cuda", dtype=torch.int32)
    o = glm_dsa_sparse_attn(q, pool, idx, scale, counts=cnt)

    k = pool[rows.long()].float()
    s = (q[0, 0].float() @ k.T) * scale
    ref = s.softmax(-1) @ k[:, :dv]
    assert (o[0, 0].float() - ref).abs().max().item() < 2e-2


def test_identity_selection_equals_topk_selection_at_short_kv():
    """The dense regimes serve through IDENTITY selection (one shared row list,
    stride-0 broadcast, causal counts). At kv <= index_topk the DSA select picks
    every live row, so the two paths must agree -- the invariant the all-Triton
    backend's dense path leans on."""
    from freetoken.attention.dsa_indexer import DSAIndexerMixin
    from freetoken.kernel.triton.glm_dsa_sparse import glm_dsa_sparse_attn

    torch.manual_seed(1)
    h, dv, dr, kv, topk = 8, 64, 16, 250, 512  # kv < topk
    pool = torch.randn(1024, dv + dr, device="cuda", dtype=torch.bfloat16) * 0.5
    rows = torch.randperm(1024, device="cuda")[:kv].to(torch.int32)
    m = 6
    positions = torch.arange(kv - m, kv, device="cuda")
    q = torch.randn(1, m, h, dv + dr, device="cuda", dtype=torch.bfloat16)
    scale = (dv + dr) ** -0.5

    # identity: shared row list broadcast over queries, causal counts
    o_ident = glm_dsa_sparse_attn(
        q, pool, rows.view(1, 1, kv), scale,
        counts=(positions + 1).to(torch.int32).view(1, m), d_v=dv,
    )

    # DSA select at kv < topk: causal top-k == all live rows (order may differ)
    mix = DSAIndexerMixin()
    scores = torch.randn(m, kv, device="cuda")  # arbitrary: selection covers all live
    picks = mix.indexer_select_prefill(
        scores.unsqueeze(0), start_pos=kv - m, seqlen=m, ratio=1,
        topk=min(topk, kv), offset=0,
    )[0]
    sel = mix.dsa_map_rows(picks, rows.view(1, kv).expand(m, kv))
    o_sel = glm_dsa_sparse_attn(
        q, pool, sel.view(1, m, -1), scale,
        counts=(positions + 1).clamp(max=min(topk, kv)).to(torch.int32).view(1, m), d_v=dv,
    )
    assert (o_ident.float() - o_sel.float()).abs().max().item() < 2e-2


def test_decode_logits_edges():
    """Fused gather+score kernel edges: tile-boundary live lengths, garbage rows
    past live, -inf tail, live == 1."""
    from freetoken.kernel.triton.glm_dsa_sparse import glm_dsa_decode_logits

    torch.manual_seed(2)
    H, D, W = 16, 128, 512
    pool = torch.randn(2048, D, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(3, H, D, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(3, H, device="cuda").abs()
    rows = torch.randint(0, 2048, (3, W), device="cuda", dtype=torch.int32)
    rows[0, 100:] = 987654  # garbage past live: must never be dereferenced
    live = torch.tensor([100, 64, 1], device="cuda", dtype=torch.int32)  # 64 = BLOCK_T edge

    out = glm_dsa_decode_logits(q, w, pool, rows, live)
    for b in range(3):
        n = live[b].item()
        k = pool[rows[b, :n].long()].float()
        ref = (torch.einsum("hd,td->ht", q[b].float(), k).relu() * w[b][:, None].float()).sum(0)
        assert (out[b, :n] - ref).abs().max().item() < 0.05, b
        assert torch.isinf(out[b, n:]).all(), b


def test_splitk_matches_single_program():
    """Both kernel variants over the same inputs (the auto heuristic only ever
    exercises one on a given GPU); log-sum-exp merge is not bit-identical, so
    tolerance is bf16-rounding scale."""
    from freetoken.kernel.triton.glm_dsa_sparse import glm_dsa_sparse_attn

    torch.manual_seed(3)
    h, dv, dr, K = 40, 512, 64, 2048
    pool = torch.randn(8192, dv + dr, device="cuda", dtype=torch.bfloat16) * 0.5
    idx = torch.randperm(8192, device="cuda")[:K].to(torch.int32).view(1, 1, K)
    idx[0, 0, 1500:] = -1
    q = torch.randn(1, 1, h, dv + dr, device="cuda", dtype=torch.bfloat16)
    cnt = torch.tensor([[1500]], device="cuda", dtype=torch.int32)
    o_single = glm_dsa_sparse_attn(q, pool, idx, 0.07, counts=cnt, force_splits=0)
    o_split = glm_dsa_sparse_attn(q, pool, idx, 0.07, counts=cnt, force_splits=16)
    assert (o_single.float() - o_split.float()).abs().max().item() < 1e-2


def _make_backend(dsa: bool, latent=80, dv=64, idx_dim=32, idx_heads=16, topk=64, pages=400):
    """Minimal ctx + pool + DSAAttnBackend (no engine)."""
    from types import SimpleNamespace

    import freetoken.core as core
    from freetoken.attention.dsa import DSAAttnBackend
    from freetoken.core import Context, set_global_ctx
    from freetoken.kvcache.dsa_pool import DSAKVCache, MLAKVCache

    core._GLOBAL_CTX = None  # test-only: the helper builds a fresh ctx per scenario
    ctx = Context(page_size=1)
    ctx.page_table = torch.zeros(4, pages, dtype=torch.int32, device="cuda")
    if dsa:
        ctx.kv_cache = DSAKVCache(latent, 2, pages, 1, torch.bfloat16, torch.device("cuda"),
                                  index_head_dim=idx_dim, num_index_layers=1)
    else:
        ctx.kv_cache = MLAKVCache(latent, 2, pages, 1, torch.bfloat16, torch.device("cuda"))
    set_global_ctx(ctx)
    args = SimpleNamespace(
        kv_lora_rank=dv, qk_rope_head_dim=latent - dv, qk_head_dim=latent,
        index_topk=topk, index_head_dim=idx_dim,
        indexer_types=("full", "shared"),
    )
    cfg = SimpleNamespace(glm_dsa_args=args, num_qo_heads=8, attn_sm_scale=None, num_layers=2)
    return DSAAttnBackend(cfg), ctx


def _ref_attend(q_cat, pool_rows, live_rows, scale, dv):
    k = pool_rows[live_rows.long()].float()
    s = (q_cat.float() @ k.T) * scale
    return s.softmax(-1) @ k[:, :dv]


def test_backend_ragged_prefill_identity_and_selection():
    """Two-request ragged prefill through the BACKEND (page-table slicing,
    counts = positions + 1, per-request segmentation, leader/follower reuse):
    request A stays under index_topk (selection == identity == dense), request B
    reaches past it (real causal top-k)."""
    from types import SimpleNamespace

    torch.manual_seed(5)
    dv, dr, h, idx_h, idx_d, topk = 64, 16, 8, 16, 32, 64
    backend, ctx = _make_backend(dsa=True, topk=topk)
    pool = ctx.kv_cache
    scale = backend.sm_scale

    # request A: rows 0..39 (kv 40 <= topk); request B: rows 40..139 (kv 100 > topk)
    ctx.page_table[0, :40] = torch.arange(40, device="cuda")
    ctx.page_table[1, :100] = torch.arange(40, 140, device="cuda")
    reqs = [SimpleNamespace(extend_len=8, device_len=40, table_idx=0),
            SimpleNamespace(extend_len=12, device_len=100, table_idx=1)]
    positions = torch.cat([torch.arange(32, 40), torch.arange(88, 100)]).cuda()
    out_loc = torch.cat([torch.arange(32, 40), torch.arange(128, 140)]).cuda()

    # pre-extend history (latent + index keys) for both requests, both layers
    hist_loc = torch.cat([torch.arange(0, 32), torch.arange(40, 128)]).cuda()
    n_hist = hist_loc.numel()
    hist_ckv = torch.randn(n_hist, dv, device="cuda", dtype=torch.bfloat16)
    hist_kpe = torch.randn(n_hist, dr, device="cuda", dtype=torch.bfloat16)
    for lid in (0, 1):  # identical content on both layers (leader/follower comparison)
        pool.store_kv(hist_ckv, hist_kpe, hist_loc, lid)
    pool.store_index_k(torch.randn(n_hist, idx_d, device="cuda", dtype=torch.bfloat16),
                       hist_loc, 0)

    batch = SimpleNamespace(reqs=reqs, positions=positions, out_loc=out_loc,
                            active_table_idx=None, attn_metadata=None)
    backend.prepare_metadata(batch)

    t = 20
    q_nope = torch.randn(t, h, dv, device="cuda", dtype=torch.bfloat16)
    q_pe = torch.randn(t, h, dr, device="cuda", dtype=torch.bfloat16)
    c_kv = torch.randn(t, dv, device="cuda", dtype=torch.bfloat16)
    k_rope = torch.randn(t, dr, device="cuda", dtype=torch.bfloat16)
    from freetoken.attention.dsa import DSAIndexerInputs

    qkw = DSAIndexerInputs(
        q=torch.randn(t, idx_h, idx_d, device="cuda", dtype=torch.bfloat16),
        k=torch.randn(t, idx_d, device="cuda", dtype=torch.bfloat16),
        w=torch.randn(t, idx_h, device="cuda").abs(),
    )

    o0 = backend.mla_forward(q_nope, q_pe, c_kv, k_rope, 0, batch, indexer_inputs=qkw)

    # request A (kv <= topk): selection covers all live -> equals dense reference
    q_cat = torch.cat([q_nope, q_pe], -1)
    slab = pool.latent_rows(0)
    for j in range(8):  # A's queries, positions 32..39
        live = ctx.page_table[0, : 33 + j]
        ref = _ref_attend(q_cat[j], slab, live, scale, dv)
        assert (o0[j].float() - ref).abs().max().item() < 3e-2, f"A q{j}"

    # request B (kv > topk): causal top-k reference from the same scoring math
    q_idx, k_idx, w = qkw.q, qkw.k, qkw.w
    for j in (0, 11):  # first and last of B's queries
        row = 8 + j
        pos = 88 + j
        rows_b = ctx.page_table[1, : pos + 1]
        keys = pool.index_k_cache(0)[rows_b.long()].float()
        s = (torch.einsum("hd,td->ht", q_idx[row].float(), keys).relu()
             * (idx_d**-0.5) * (w[row][:, None].float())).sum(0)
        sel_rows = rows_b[s.topk(min(topk, pos + 1)).indices]
        ref = _ref_attend(q_cat[row], slab, sel_rows, scale, dv)
        assert (o0[row].float() - ref).abs().max().item() < 3e-2, f"B q{j}"

    # leader/follower: layer 1 (shared, no indexer) reuses layer 0's selection; with
    # identical latent content its output must match layer 0's
    o1 = backend.mla_forward(q_nope, q_pe, c_kv, k_rope, 1, batch, indexer_inputs=None)
    assert (o0.float() - o1.float()).abs().max().item() < 3e-2

    # identity wiring (dense ablation): same batch through an MLAKVCache backend
    backend_d, ctx_d = _make_backend(dsa=False)
    ctx_d.page_table.copy_(ctx.page_table)
    for lid in (0, 1):
        ctx_d.kv_cache._kv_buffer.copy_(pool._kv_buffer)
    batch_d = SimpleNamespace(reqs=reqs, positions=positions, out_loc=out_loc,
                              active_table_idx=None, attn_metadata=None)
    backend_d.prepare_metadata(batch_d)
    od = backend_d.mla_forward(q_nope, q_pe, c_kv, k_rope, 0, batch_d, indexer_inputs=None)
    slab_d = ctx_d.kv_cache.latent_rows(0)
    for j in range(8):
        live = ctx_d.page_table[0, : 33 + j]
        ref = _ref_attend(q_cat[j], slab_d, live, scale, dv)
        assert (od[j].float() - ref).abs().max().item() < 3e-2, f"dense A q{j}"
    for j in (0, 11):
        live = ctx_d.page_table[1, : 89 + j]
        ref = _ref_attend(q_cat[8 + j], slab_d, live, scale, dv)
        assert (od[8 + j].float() - ref).abs().max().item() < 3e-2, f"dense B q{j}"
