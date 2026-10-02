"""DSV41 attention backend: addressing and selection contracts, CPU-only (kernels live in tests/kernels).

* prefill metadata: encoder segments, the decoder pass under exact vs bounded replay;
* window candidates with a floor (bounded replay truncates the window at the replay start);
* decode snapshot staging and the layer-invariant ring context;
* the pure selection helpers against the reference ``select_candidate_blocks`` / top-k semantics.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from freetoken.core import Batch, Context, Req, SamplingParams, get_global_ctx, set_global_ctx
from freetoken.kvcache.dsv41_cost_model import dsv41_pool_sizes
from freetoken.kvcache.dsv41_geometry import DSV41Geometry
from freetoken.kvcache.dsv41_paged_pool import DSV41PagedKVCache

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


def _stack(swa_decoder_replay="exact", num_pages=32, max_seq_len=8192):
    from freetoken.attention.dsv41_sparse import DSV41SparseAttnBackend

    private = tuple(range(3, len(RATIOS))) if swa_decoder_replay != "exact" else ()
    geom = DSV41Geometry(n_layers=len(RATIOS), head_dim=512, index_head_dim=128, window=P, compress_ratios=RATIOS, kv_source_layer_ids=SOURCES,
                        resume_windows=1 if swa_decoder_replay == "exact" else 2, private_window_layer_ids=private)
    pool = DSV41PagedKVCache(dsv41_pool_sizes(num_pages + 1, geom, 1.0, P), geom, DEVICE, n_scratch=MRR + 1)
    pool._init_paged_state(MRR, True)
    pt = torch.zeros(MRR + 1, max_seq_len, dtype=torch.int32)
    pt[MRR].fill_(num_pages * P)
    pt[2, :300] = torch.arange(300, dtype=torch.int32)
    for page in range(3):
        pool.bind_window_pages(page * P, page * P)
    pool.full_loc_map = pt
    _ctx(pool)
    group = SimpleNamespace(geometry=geom)
    args = SimpleNamespace(swa_decoder_replay=swa_decoder_replay, decoder_start_layer=3)
    backend = DSV41SparseAttnBackend(SimpleNamespace(attention_groups=(group,), dsv41_args=args))
    return backend, pool, pt


def _req(table_idx, cached_len, n_new, output_len=16, uid=0):
    return Req(input_ids=torch.zeros(cached_len + n_new, dtype=torch.int32), table_idx=table_idx, cached_len=cached_len,
               output_len=output_len, uid=uid, sampling_params=SamplingParams(), cache_handle=None)


def _prefill_batch(reqs):
    batch = Batch(reqs=reqs, phase="prefill")
    batch.padded_reqs = reqs
    return batch


def test_prefill_metadata_exact_mode_runs_the_decoder_on_every_token():
    backend, _, _ = _stack("exact")
    batch = _prefill_batch([_req(0, 0, 300), _req(2, 256, 44, uid=1)])
    backend.prepare_metadata(batch)
    md = batch.attn_metadata
    assert [(s.offset, s.n, s.table_idx, s.start_pos, s.window_floor) for s in md.segments] == [(0, 300, 0, 0, 0), (300, 44, 2, 256, 0)]
    assert md.decoder_segments is md.segments and md.decoder_rows is None
    assert md.last_indices.tolist() == [299, 343]


def test_prefill_metadata_bounded_mode_replays_the_last_window():
    backend, pool, _ = _stack("bounded")
    r0, r1 = _req(0, 0, 300), _req(2, 256, 44, uid=1)
    r1.input_ids = torch.arange(300, dtype=torch.int32)  # distinguishable ids for the stream check
    batch = _prefill_batch([r0, r1])
    batch.input_ids = torch.cat([r0.input_ids, r1.input_ids[256:]])
    batch.positions = torch.cat([torch.arange(300), torch.arange(256, 300)])
    backend.prepare_metadata(batch)
    md = batch.attn_metadata
    enc, dec = md.segments, md.decoder_segments
    # request 0: the cold prompt is its own encoder stream; the decoder replays its last 128 tokens
    assert (enc[0].offset, enc[0].n, enc[0].start_pos, enc[0].write_from) == (0, 300, 0, 0)
    assert (dec[0].offset, dec[0].n, dec[0].start_pos, dec[0].window_floor) == (0, 128, 172, 172)
    # request 1 brought 44 new tokens after a 256-token hit: the encoder pass re-runs from the previous
    # window page (128) so the decoder can replay the prompt's whole last window [172, 300); the
    # recomputed history [128, 256) is read-only -- writes start at the hit
    assert (enc[1].offset, enc[1].n, enc[1].start_pos, enc[1].write_from, enc[1].write_offset) == (300, 172, 128, 256, 428) and md.extended
    assert (dec[1].offset, dec[1].n, dec[1].start_pos, dec[1].window_floor) == (128, 128, 172, 172)
    assert md.decoder_rows.tolist() == list(range(172, 300)) + list(range(300 + 44, 300 + 172))
    assert md.last_indices.tolist() == [127, 255]  # into the decoder stream
    with get_global_ctx().forward_batch(batch):
        ids, pos = backend.encoder_stream(batch)
    assert ids.shape == (472,) and torch.equal(ids[300:], torch.arange(128, 300, dtype=torch.int32))
    assert torch.equal(pos[300:], torch.arange(128, 300))


def test_private_window_layers_address_per_request_rings():
    """Under bounded replay the decoder layers keep their window KV in per-request rings (slot =
    table row * P + pos % P), never in the radix-shared window pages; the encoder layers stay on
    the shared pool."""
    from freetoken.attention.dsv41_sparse import PrefillSegment

    backend, pool, _ = _stack("bounded")
    geom = backend.geom
    assert geom.private_window_layer_ids == (3, 4) and geom.shared_window_layer_ids == (0, 1, 2)
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


def test_bounded_extension_reads_only_the_resume_history():
    """A bounded-mode extension recomputes from the previous window page, so the window keys it reads,
    ``[start - P + 1, cached_len)``, stay inside the two windows the cache keeps live behind a
    page-aligned hit (KVCacheGroupSpec.swa_resume_history). ``replay_history`` bounds that read for
    admission, which turns a hit reading further back into a miss; the planner does not re-check."""
    from freetoken.attention.dsv41_sparse import replay_history

    backend, _, _ = _stack("bounded")
    for cached, new in ((256, 44), (256, 1), (256, P - 1), (384, 10)):
        batch = _prefill_batch([_req(2, cached, new, uid=1)])
        backend.prepare_metadata(batch)
        start = batch.attn_metadata.segments[0].start_pos
        assert start == cached - P and batch.attn_metadata.extended
        assert cached - (start - P + 1) <= replay_history(cached, P) == 2 * P - 1 <= backend.geom.resume_history
    # off a window page the one-new-token recompute starts a page further back
    assert replay_history(0, P) == 0 and replay_history(3 * P + 1, P) == 2 * P and replay_history(3 * P + 2, P) > backend.geom.resume_history


def test_replay_start_rules():
    from freetoken.attention.dsv41_sparse import replay_start

    assert replay_start(_req(0, 0, 50), P, True) == 0  # a cold prompt shorter than a window
    assert replay_start(_req(0, 256, 300), P, True) == 256  # a whole window of new tokens
    assert replay_start(_req(0, 256, 44), P, True) == 128  # short suffix -> previous page boundary
    assert replay_start(_req(0, 384, 10), P, True) == 256
    assert replay_start(_req(0, 256, 44), P, False) == 256  # exact mode never extends


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


def _reference_candidate_mask(logits, compress_lens, topk_blocks, block_size):
    """The reference ``select_candidate_blocks`` (boolean mask over positions)."""
    b, s, width = logits.shape
    pad = -width % block_size
    blk = F.pad(logits, (0, pad), value=float("-inf")).view(b, s, -1, block_size).amax(dim=-1)
    nb = blk.shape[-1]
    last = torch.div(compress_lens - 1, block_size, rounding_mode="floor")
    blk = blk.masked_fill(torch.arange(nb, device=logits.device) == last, float("inf"))
    top = blk.topk(min(topk_blocks, nb), dim=-1)
    keep = torch.zeros_like(blk, dtype=torch.bool).scatter(-1, top.indices, ~torch.isneginf(top.values))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the selection kernels need CUDA")
def test_candidate_blocks_match_the_reference_mask():
    from freetoken.attention.dsv41_sparse import DSV41SparseAttnBackend

    torch.manual_seed(0)
    b, s, t, blk, kb = 2, 5, 61, 8, 3
    logits = torch.randn(b, s, t, device="cuda")
    live = torch.tensor([[61, 40, 17, 1, 0], [8, 9, 33, 61, 61]], device="cuda", dtype=torch.int32)
    logits = logits.masked_fill(torch.arange(t, device="cuda") >= live.unsqueeze(-1), -torch.inf)
    cand = DSV41SparseAttnBackend.select_candidate_blocks(logits, live, kb, blk)
    assert cand.shape == (b, s, kb * blk)
    want = _reference_candidate_mask(logits, live.unsqueeze(-1), kb, blk)
    for i in range(b):
        for j in range(s):
            got = [int(p) for p in cand[i, j] if p >= 0]
            ref = [int(p) for p in want[i, j].nonzero().flatten() if p < live[i, j]]
            assert got == ref, (i, j, got, ref)  # ascending valid prefix
            assert (cand[i, j, len(got):] == -1).all()
    # empty history: no candidates at all
    assert (cand[0, 4] == -1).all()


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
