"""DSV41 attention backend: addressing and selection contracts, CPU-only (kernels live in tests/kernels).

* prefill metadata: encoder segments, the decoder pass under exact vs bounded replay;
* window candidates with a floor (bounded replay truncates the window at the prompt's last window);
* decode snapshot staging and the layer-invariant ring context;
* the selection helpers' reshape, candidate gather and position-to-row mapping.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Batch, Context, Req, SamplingParams, get_global_ctx, set_global_ctx
from freetoken.kvcache.dsv4.v41_cost_model import dsv41_pool_sizes
from freetoken.kvcache.dsv4.v41_pool import DSV41PagedKVCache
from freetoken.models.deepseek_v41.args import DeepseekV41Args

P, MRR, DEVICE = 128, 4, torch.device("cpu")
RATIOS = (0, 2, 2, 1, 1)
SOURCES = (1, 3)


def _ctx(pool):
    try:
        ctx = get_global_ctx()
    except AssertionError:
        ctx = Context(page_size=P)
        set_global_ctx(ctx)
    ctx.kv_cache = pool
    return ctx


def _stack(swa_decoder_replay="exact", num_pages=32, max_seq_len=8192, index_sources=()):
    from freetoken.attention.dsv41_sparse import DSV41SparseAttnBackend

    args = DeepseekV41Args(n_layers=len(RATIOS), head_dim=512, index_head_dim=128, window_size=P, compress_ratios=RATIOS,
                           kv_source_layers=SOURCES, index_source_layers=index_sources, swa_decoder_replay=swa_decoder_replay)
    pool = DSV41PagedKVCache(dsv41_pool_sizes(num_pages + 1, args, 1.0, P), args, DEVICE, n_scratch=MRR + 1)
    pool._init_paged_state(MRR, True)
    pt = torch.zeros(MRR + 1, max_seq_len, dtype=torch.int32)
    pt[MRR].fill_(num_pages * P)
    pt[2, :300] = torch.arange(300, dtype=torch.int32)
    for page in range(3):
        pool.bind_window_pages(page * P, page * P)
    pool.full_loc_map = pt
    _ctx(pool)
    backend = DSV41SparseAttnBackend(SimpleNamespace(dsv41_args=args))
    return backend, pool, pt


def _req(table_idx, cached_len, n_new, output_len=16, uid=0, prompt_len=0):
    return Req(input_ids=torch.zeros(cached_len + n_new, dtype=torch.int32), table_idx=table_idx, cached_len=cached_len,
               output_len=output_len, uid=uid, sampling_params=SamplingParams(), cache_handle=None, prompt_len=prompt_len)


def _prefill_batch(reqs):
    batch = Batch(reqs=reqs, phase="prefill")
    batch.padded_reqs = reqs
    return batch


def test_prefill_metadata_exact_mode_runs_the_decoder_on_every_token():
    backend, pool, _ = _stack("exact")
    assert pool.prefix_replay_tokens == 0
    batch = _prefill_batch([_req(0, 0, 300), _req(2, 256, 44, uid=1)])
    backend.prepare_metadata(batch)
    md = batch.attn_metadata
    assert [(s.offset, s.n, s.table_idx, s.start_pos, s.window_floor) for s in md.segments] == [(0, 300, 0, 0, 0), (300, 44, 2, 256, 0)]
    assert md.decoder_segments is md.segments and md.decoder_rows is None
    assert md.last_indices.tolist() == [299, 343]


def test_prefill_metadata_bounded_mode_runs_the_decoder_on_the_last_window():
    """The encoder runs every new token; the decoder runs each request's part of its prompt's last
    window ``[L - P, L)``, floored at ``L - P``. A chunk ending before that window runs no decoder row
    and its ``last_indices`` entry points at the placeholder row the model appends."""
    backend, pool, _ = _stack("bounded")
    assert pool.prefix_replay_tokens == P and pool.sliding_window_size == P
    cold = _req(0, 0, 300)  # a whole prompt: L - P = 172
    early = _req(1, 256, 128, uid=1, prompt_len=600)  # chunk [256, 384) of a 600-token prompt: before 472
    straddle = _req(2, 384, 128, uid=2, prompt_len=600)  # chunk [384, 512): its rows [472, 512) are in the window
    batch = _prefill_batch([cold, early, straddle])
    backend.prepare_metadata(batch)
    md = batch.attn_metadata
    assert [(s.offset, s.n, s.start_pos, s.window_floor) for s in md.segments] == [(0, 300, 0, 0), (300, 128, 256, 0), (428, 128, 384, 0)]
    assert [(d.offset, d.n, d.table_idx, d.start_pos, d.window_floor) for d in md.decoder_segments] == [(0, 128, 0, 172, 172), (128, 40, 2, 472, 472)]
    assert md.decoder_rows.tolist() == list(range(172, 300)) + list(range(428 + 88, 556))
    assert md.decoder_pad and md.last_indices.tolist() == [127, 168, 167]  # 168: the placeholder after 168 decoder rows


def test_prefill_metadata_bounded_mode_without_any_decoder_row():
    backend, _, _ = _stack("bounded")
    batch = _prefill_batch([_req(0, 0, 256, prompt_len=1024)])
    backend.prepare_metadata(batch)
    md = batch.attn_metadata
    assert md.decoder_segments == [] and md.decoder_rows.numel() == 0
    assert md.decoder_pad and md.last_indices.tolist() == [0]


def test_prefill_metadata_short_prompt_runs_the_decoder_everywhere():
    """A prompt shorter than the window floors the decoder's window at 0: it sees what exact mode sees."""
    backend, _, _ = _stack("bounded")
    batch = _prefill_batch([_req(0, 0, 50)])
    backend.prepare_metadata(batch)
    md = batch.attn_metadata
    assert [(d.offset, d.n, d.start_pos, d.window_floor) for d in md.decoder_segments] == [(0, 50, 0, 0)]
    assert not md.decoder_pad and md.last_indices.tolist() == [49]


def test_prefill_metadata_encoder_replay_extends_the_encoder_segment_only():
    """An encoder bounded replay (``enc_replay_lo`` below the hit): the ENCODER segment starts at the
    replay start, floored there, with the hit as its publish floor; the decoder pass and
    last_indices are exactly the no-replay ones (the admission cap keeps the prompt's last window
    past the hit, so the decoder never touches the replayed rows)."""
    backend, _, _ = _stack("bounded")
    req = _req(2, 384, 216, prompt_len=600)  # hit 384, chunk reaches the prompt end 600
    req.enc_replay_lo = 384 - P
    batch = _prefill_batch([req])
    backend.prepare_metadata(batch)
    md = batch.attn_metadata
    assert [(s.offset, s.n, s.table_idx, s.start_pos, s.window_floor, s.publish_floor) for s in md.segments] == [
        (0, 344, 2, 256, 256, 384)]
    assert [(d.offset, d.n, d.table_idx, d.start_pos, d.window_floor) for d in md.decoder_segments] == [
        (0, 128, 2, 472, 472)]
    assert md.decoder_rows.tolist() == list(range(216, 344))
    assert not md.decoder_pad and md.last_indices.tolist() == [127]


def test_prefill_metadata_encoder_replay_under_exact_decoder_shares_the_segment():
    """Exact decoder + bounded encoder replay: the decoder runs every chunk token, sharing the
    replay segment; the window_floor masks reads below the replay start, which is what makes the
    shared segment safe (those slots are dead)."""
    backend, _, _ = _stack("exact")
    req = _req(2, 384, 216, prompt_len=600)
    req.enc_replay_lo = 384 - P
    batch = _prefill_batch([req])
    backend.prepare_metadata(batch)
    md = batch.attn_metadata
    assert [(s.start_pos, s.window_floor, s.publish_floor) for s in md.segments] == [(256, 256, 384)]
    assert md.decoder_segments is md.segments and md.decoder_rows is None
    assert md.last_indices.tolist() == [343]


def test_publish_prefill_masks_groups_below_the_publish_floor():
    """A replay segment re-sends the hit's last window; its groups below the hit (the publish floor)
    must not be re-published: those shared global rows are read as-is, and the floor is group-aligned
    (page-aligned floor, page % ratio == 0), so no group straddles it."""
    from freetoken.attention.dsv41_sparse import PrefillSegment
    from freetoken.models.deepseek_v41.attention import DSV41Attention

    backend, pool, pt = _stack("bounded", index_sources=SOURCES)
    args = backend.config.dsv41_args
    get_global_ctx().attn_backend = backend  # the model layer resolves its backend per access
    pt[2, :512] = torch.arange(512, dtype=torch.int32)  # the hit's pages
    layer = DSV41Attention(args, args.roles[1])  # the ratio-2 Full source
    layer._freqs = torch.ones(1024, args.rope_head_dim // 2, dtype=torch.complex64)  # CPU stand-in table
    layer.compressor.norm.forward = lambda t: t  # the fused RMSNorm is CUDA-only; values don't matter here
    layer.indexer.k_norm.forward = lambda t: t
    stored = []
    backend.compressed_rows_of = lambda ti, starts, ratio: starts  # row == group start (identity)
    backend.store_main = lambda latent, src, rows: stored.append(("main", rows.tolist()))
    backend.store_index = lambda k, src, rows: stored.append(("idx", rows.tolist()))
    layer.publish_prefill(torch.randn(344, args.dim), [PrefillSegment(0, 344, 2, 256, window_floor=256, publish_floor=384)])
    assert stored == [
        ("idx", list(range(384, 600, 2))),  # groups [384, 600): published, keys before main
        ("main", list(range(384, 600, 2))),
    ]
    # without a floor every group of the segment publishes (the pre-replay behavior)
    stored.clear()
    layer.publish_prefill(torch.randn(88, args.dim), [PrefillSegment(0, 88, 2, 256)])
    assert stored == [("idx", list(range(256, 344, 2))), ("main", list(range(256, 344, 2)))]


def test_private_window_layers_address_per_request_rings():
    """Under bounded replay the decoder layers keep their window KV in per-request rings (slot =
    table row * P + pos % P), never in the radix-shared window pages; the encoder layers stay on
    the shared pool."""
    from freetoken.attention.dsv41_sparse import PrefillSegment

    backend, pool, _ = _stack("bounded")
    assert pool.private_window_layer_ids == (3, 4) and not any(pool.is_private_window(l) for l in (0, 1, 2))
    assert pool.window_pool[3].shape[0] == (MRR + 1) * P and pool.window_pool[0].shape[0] == pool.sizes.n_win_slots
    # prefill: a decoder segment's slots are its ring row; an encoder segment's are page-bound
    seg = PrefillSegment(0, 44, 2, 256, window_floor=256)
    assert backend.layer_window_slots_of(3, 2, 256, 300).tolist() == [2 * P + p % P for p in range(256, 300)]
    assert backend.layer_window_slots_of(0, 2, 256, 300).tolist() == list(range(256, 300))
    g = backend.window_topk_prefill(seg, layer_id=3)[0]
    assert g.shape == (44, P) and g[0, 0].item() == 2 * P + 0 and (g[0, 1:] == -1).all()  # position 256 sees itself only (floor)
    assert set(g[43].tolist()) - {-1} == {2 * P + p % P for p in range(256, 300)}
    # decode: the private ring context keys on the batch row's page-table row
    batch = _decode_batch([2], [300])
    backend.prepare_metadata(batch)
    md = batch.attn_metadata
    slots, topk = md.private_window_ctx(torch.tensor([300]), torch.arange(1))
    assert slots.tolist() == [2 * P + 300 % P] and topk.shape == (1, 1, P)
    assert set(topk[0, 0].tolist()) == {2 * P + j for j in range(P)}  # every ring index holds a position <= 300


def test_window_candidates_honor_the_floor_and_the_retained_prefix():
    from freetoken.attention.dsv41_sparse import PrefillSegment

    backend, pool, _ = _stack()
    # a cold segment at positions [0, 10): query p sees [0, p]
    g = backend.window_topk_prefill(PrefillSegment(0, 10, 2, 0))
    assert g.shape == (1, 10, P)
    assert g[0, 3, :4].tolist() == [0, 1, 2, 3] and (g[0, 3, 4:] == -1).all()
    # an extend at [256, 300) sees back into the retained prefix, but not below the floor
    g = backend.window_topk_prefill(PrefillSegment(0, 44, 2, 256, window_floor=200))
    row = g[0, 0]  # query at 256 -> [200, 256]
    assert row[0].item() == 200 and row[56].item() == 256 and (row[57:] == -1).all()
    # without a floor the query at 256 sees [129, 256]
    g = backend.window_topk_prefill(PrefillSegment(0, 44, 2, 256))
    assert g[0, 0, 0].item() == 129 and g[0, 0, 127].item() == 256


def _decode_batch(rows, positions):
    reqs = [_req(int(t), 0, 1, uid=i) for i, t in enumerate(rows)]
    batch = Batch(reqs=reqs, phase="decode")
    batch.padded_reqs = reqs
    batch.active_table_idx = torch.tensor(rows, dtype=torch.int64)
    batch.positions = torch.tensor(positions, dtype=torch.int64)
    return batch


def test_decode_snapshot_and_ring_context():
    backend, pool, pt = _stack()
    batch = _decode_batch([2, MRR], [259, 0])
    backend.prepare_metadata(batch)
    md = batch.attn_metadata
    assert md.full_snap is None
    snap = md.full_snapshot()
    assert torch.equal(snap[0, :300], torch.arange(300))
    pt[2, :10] = 7  # a later mutation must not reach the snapshot
    assert torch.equal(snap[0, :300], torch.arange(300))
    pos, rows = batch.positions, torch.arange(2)
    ws, prev, topk = md.window_ctx(pos, rows)
    dummy_ws = pool.sizes.n_win_slots - P  # the dummy row's full loc binds to the dummy window page
    assert ws.tolist() == [259, dummy_ws] and prev.tolist() == [258, dummy_ws]
    assert topk.shape == (2, 1, P) and (topk[0] >= 0).all()
    assert set(topk[0, 0].tolist()) == set(range(259 - 127, 260))
    assert topk[1, 0, 0].item() == dummy_ws and (topk[1, 0, 1:] == -1).all()
    # the indexer derives compressed rows from the live locs: prefill off the page table row,
    # decode off the snapshot (position t -> full_loc(t * ratio) // ratio)
    assert torch.equal(backend.locs_prefill(2, 20), pt[2:3, :20])
    assert snap[0, :16].tolist() == list(range(16))  # taken before the mutation above
    # capture staging reuses one buffer
    backend.init_capture_graph(512, [1, 2])
    backend.prepare_for_replay(batch)
    assert batch.attn_metadata.full_snap.data_ptr() == backend.capture.full_snap.data_ptr()
    assert batch.attn_metadata.stage_width == 512


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the selection kernels need CUDA")
def test_topk_helpers():
    from freetoken.attention.dsv41_sparse import DSV41SparseAttnBackend as B

    scores = torch.tensor([[[0.1, 5.0, 3.0, 7.0, 9.0]]], device="cuda")  # columns past live are never read
    live = torch.tensor([[3]], device="cuda", dtype=torch.int32)
    assert B.select_topk(scores, live, 4).tolist() == [[[0, 1, 2, -1]]]
    assert B.select_topk(scores, live, 2).tolist() == [[[1, 2]]]
    # a candidate list is a sorted valid prefix; empty slots score -inf
    cand = torch.tensor([[[2, 7, 9, -1]]], device="cuda", dtype=torch.int32)
    cscores = torch.tensor([[[1.0, 4.0, -torch.inf, -torch.inf]]], device="cuda")
    assert B.select_topk_in_candidates(cscores, cand, 3).tolist() == [[[2, 7, -1]]]
    # rows come from the live locs: compressed position p -> locs[p * ratio] // ratio
    locs = torch.arange(20, 40, device="cuda", dtype=torch.int32).view(1, 20)
    locs[0, 14:16] = -1  # position 7 (ratio 2) has slid out
    rows = B.positions_to_rows(torch.tensor([[[2, 7, 9, -1]]], device="cuda", dtype=torch.int32), locs, 2)
    assert rows.tolist() == [[[12, -1, 19, -1]]]
