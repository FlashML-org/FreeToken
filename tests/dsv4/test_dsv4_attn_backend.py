"""DSV4 attention backend: the decode addressing contract it took over from the pool.

The backend snapshots each decode batch's page-table rows into a graph-stable buffer, sizes
that buffer (and therefore the compressed-candidate staging grids) to the engine's live
ceiling, and resolves the 128-ring context every layer shares. These are CPU-only shape and
staging checks -- the kernels themselves are covered in tests/kernels/.
"""

from __future__ import annotations

import torch

from freetoken.core import Batch, Context, Req, SamplingParams, get_global_ctx, set_global_ctx
from freetoken.kvcache.dsv4.v4_pool import DSV4PagedKVCache
from freetoken.kvcache.dsv4.v4_cost_model import dsv4_pool_sizes
from freetoken.models.deepseek_v4.args import DeepseekV4Args

P, MRR, DEVICE = 128, 4, torch.device("cpu")
RATIOS = (0, 4, 128, 4)


def _ctx(pool):
    try:
        ctx = get_global_ctx()
    except AssertionError:
        ctx = Context(page_size=P)
        set_global_ctx(ctx)
    ctx.kv_cache = pool
    return ctx


def _stack(num_pages=32, max_seq_len=8192):
    args = DeepseekV4Args(
        max_batch_size=MRR + 1, dim=256, n_layers=len(RATIOS), n_heads=4, q_lora_rank=64,
        o_lora_rank=64, o_groups=2, moe_inter_dim=64, n_routed_experts=4,
        n_activated_experts=2, vocab_size=64, index_n_heads=2, index_topk=8,
        compress_ratios=RATIOS, max_seq_len=max_seq_len,
        head_dim=512, index_head_dim=128, window_size=P,
    )
    sizes = dsv4_pool_sizes(num_pages=num_pages + 1, args=args, swa_ratio=1.0, P=P)
    pool = DSV4PagedKVCache(sizes=sizes, args=args, device=DEVICE, P=P, n_scratch=MRR + 1)
    pool._init_paged_state(MRR, True)
    pt = torch.zeros(MRR + 1, max_seq_len, dtype=torch.int32)
    pt[MRR].fill_(num_pages * P)          # engine's dummy-row convention
    pt[2, :300] = torch.arange(300, dtype=torch.int32)
    for page in range(3):                 # bind row 2's window pages (positions 0..383)
        pool.bind_window_pages(page * P, page * P)
    pool.full_loc_map = pt
    _ctx(pool)

    from types import SimpleNamespace

    from freetoken.attention.dsv4_sparse import DSV4SparseAttnBackend

    # the backend reads only dsv4_args off the model config
    return DSV4SparseAttnBackend(SimpleNamespace(dsv4_args=args)), pool, pt


def _decode_batch(rows, positions):
    reqs = [
        Req(input_ids=torch.zeros(1, dtype=torch.int32), table_idx=int(t), cached_len=0,
            output_len=1, uid=i, sampling_params=SamplingParams(), cache_handle=None)
        for i, t in enumerate(rows)
    ]
    batch = Batch(reqs=reqs, phase="decode")
    batch.padded_reqs = reqs
    batch.active_table_idx = torch.tensor(rows, dtype=torch.int64)
    batch.positions = torch.tensor(positions, dtype=torch.int64)
    return batch


def test_eager_decode_snapshots_the_live_rows():
    """Even without a graph the metadata carries a COPY, so the next batch's allocate_paged
    cannot move the rows this forward reads. It is materialized lazily: a graph-eligible batch
    has its metadata replaced before any read, so taking it in prepare_metadata would be a
    full-width gather thrown away on every replay."""
    backend, pool, pt = _stack()
    batch = _decode_batch([2, MRR], [259, 0])
    backend.prepare_metadata(batch)
    assert batch.attn_metadata.full_snap is None          # deferred, not taken yet
    with get_global_ctx().forward_batch(batch):
        snap = backend.snapshot()
        assert backend.snapshot() is snap                 # materialized once, then cached
    assert snap is not None and snap.dtype == torch.int64
    assert torch.equal(snap[0, :300], pt[2, :300].to(torch.int64))
    assert int(snap[1, 0]) == 32 * P               # dummy row -> the reserved tail page
    pt[2, :300] = -7                                # a later allocate mutating the live table
    assert torch.equal(snap[0, :300], torch.arange(300, dtype=torch.int64))


def test_prefill_metadata_has_no_snapshot():
    backend, _, _ = _stack()
    reqs = [Req(input_ids=torch.zeros(5, dtype=torch.int32), table_idx=1, cached_len=0,
                output_len=1, uid=0, sampling_params=SamplingParams(), cache_handle=None)]
    batch = Batch(reqs=reqs, phase="prefill")
    batch.padded_reqs = reqs
    backend.prepare_metadata(batch)
    assert batch.attn_metadata.full_snap is None
    assert int(batch.attn_metadata.get_last_indices(1)[0]) == 4


def test_prefill_metadata_carries_segments():
    """Prefill addressing -- (offset, extend_len, table_idx, start_pos) per request, tiling the
    flat token stream -- rides the metadata, so the model's forward never reads Req fields.
    Cold (cached_len == 0) and radix-hit (cached_len > 0) segments mix freely."""
    backend, _, _ = _stack()
    reqs = [
        Req(input_ids=torch.zeros(300, dtype=torch.int32), table_idx=2, cached_len=256,
            output_len=1, uid=0, sampling_params=SamplingParams(), cache_handle=None),
        Req(input_ids=torch.zeros(5, dtype=torch.int32), table_idx=1, cached_len=0,
            output_len=1, uid=1, sampling_params=SamplingParams(), cache_handle=None),
    ]
    batch = Batch(reqs=reqs, phase="prefill")
    batch.padded_reqs = reqs
    backend.prepare_metadata(batch)
    md = batch.attn_metadata
    assert md.segments == [(0, 44, 2, 256), (44, 5, 1, 0)]
    assert int(md.get_last_indices(2)[0]) == 43 and int(md.get_last_indices(2)[1]) == 48


def test_staging_width_tracks_the_engine_ceiling():
    """init_capture_graph's max_seq_len is the live ceiling; the staged grids are sized off the
    snapshot, so the two cannot drift."""
    backend, _, _ = _stack()
    backend.init_capture_graph(max_seq_len=1024, bs_list=[1, 2])
    batch = _decode_batch([2, MRR], [259, 0])
    backend.prepare_for_replay(batch)
    assert batch.attn_metadata.stage_width == 1024
    with get_global_ctx().forward_batch(batch):
        assert backend.snapshot().shape[1] == 1024


def test_replay_reuses_the_captured_buffer():
    """The captured kernels read by address: every replay must overwrite the SAME tensor."""
    backend, _, pt = _stack()
    backend.init_capture_graph(max_seq_len=1024, bs_list=[2])
    first = _decode_batch([2, MRR], [259, 0])
    backend.prepare_for_replay(first)
    buf = batch_snap = first.attn_metadata.full_snap
    pt[3, :300] = torch.arange(1000, 1300, dtype=torch.int32)
    second = _decode_batch([3, MRR], [299, 0])
    backend.prepare_for_replay(second)
    assert second.attn_metadata.full_snap.data_ptr() == buf.data_ptr()
    assert torch.equal(batch_snap[0, :300], torch.arange(1000, 1300, dtype=torch.int64))


def test_capture_stages_the_dummy_row():
    """Capture runs on dummy rows only; they must resolve to the reserved page so the captured
    gather reads a real slot instead of the -1 fill."""
    backend, pool, _ = _stack()
    backend.init_capture_graph(max_seq_len=1024, bs_list=[2])
    batch = _decode_batch([MRR, MRR], [0, 0])
    backend.prepare_for_capture(batch)
    snap = batch.attn_metadata.full_snap
    assert int(snap[0, 0]) == 32 * P
    assert (pool.translate_full_to_window(snap[:, :4]) >= 0).all()


def test_window_ctx_is_layer_invariant_and_row_isolated():
    backend, pool, _ = _stack()
    batch = _decode_batch([2, MRR], [259, 0])
    backend.prepare_metadata(batch)
    with get_global_ctx().forward_batch(batch):
        pos = batch.positions
        rows = torch.arange(2)
        md = batch.attn_metadata
        ws, prev, ring = md.window_ctx(pos, rows)
        assert ws.shape == (2,) and prev.shape == (2,) and ring.shape == (2, 1, P)
        # row 1 sits at position 0: exactly one live ring column, the rest masked
        assert int((ring[1, 0] >= 0).sum()) == 1
        # row 0 is past a full window: every ring column is live
        assert int((ring[0, 0] >= 0).sum()) == P
        # NEVER cached: a second call recomputes (fresh tensors, equal values). Under graph
        # capture the warm and captured forwards share one metadata object -- a cached tensor
        # would leave these gathers OUT of the captured graph and freeze every replay at the
        # capture-time ring slots.
        again = md.window_ctx(pos, rows)
        assert again[2] is not ring and torch.equal(again[2], ring)
    # a FRESH metadata (next step's prepare) also recomputes to equal values
    batch2 = _decode_batch([2, MRR], [259, 0])
    backend.prepare_metadata(batch2)
    with get_global_ctx().forward_batch(batch2):
        again2 = batch2.attn_metadata.window_ctx(batch2.positions, torch.arange(2))
        assert torch.equal(again2[2], ring)


def test_recapture_reallocates_a_clean_buffer_at_the_new_ceiling():
    """Replaces the old pool-side reset: a rebuild tears the capture down and re-arms it, so a
    SHRUNKEN ceiling can never leave a stale positive loc in a column the new staging grids
    still gather (the wider era's locs index past the shrunken pool -> kernel-side OOB)."""
    backend, _, pt = _stack()
    backend.init_capture_graph(max_seq_len=2048, bs_list=[2])
    wide = _decode_batch([2, MRR], [259, 0])
    backend.prepare_for_replay(wide)
    assert int(wide.attn_metadata.full_snap[0, 200]) == 200      # wide-era column written

    backend.reset_capture()                                       # engine's rebuild teardown
    assert backend.capture is None and backend.capture_bs == []
    backend.init_capture_graph(max_seq_len=512, bs_list=[2])      # re-arm at a narrower ceiling
    # A brand-new -1 buffer, so no column can carry a wide-era loc into the narrower geometry
    assert backend.capture.full_snap.shape == (2, 512)
    assert (backend.capture.full_snap == -1).all()
    narrow = _decode_batch([MRR, MRR], [0, 0])
    backend.prepare_for_replay(narrow)
    assert narrow.attn_metadata.full_snap.shape[1] == 512
    assert int(narrow.attn_metadata.full_snap[0, 0]) == 32 * P
