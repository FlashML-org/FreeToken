"""DSV4 paged KV pool (CPU, no model). Sections:
  * pool construction / tier presence + state-ring layout;
  * state_loc derivation (scratch routing, page-disjoint ring blocks, boundary straddles);
  * full-loc -> physical-tier translation (bind/unbind, -1 gather-only sentinel, cmp_rows
    arithmetic) and the decode full-loc snapshot mirrors (scatter masking, topk masking).
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.dsv4.v4_cost_model import dsv4_pool_sizes
from freetoken.kvcache.dsv4.v4_pool import CompressStateRing, DSV4PagedKVCache
from freetoken.models.deepseek_v4.args import DeepseekV4Args

DEVICE = torch.device("cpu")
P = 128
RATIOS = (0, 0, 4, 128, 4, 128, 4, 0)


def _args(**over):
    base = dict(
        n_layers=8,
        compress_ratios=RATIOS,
        max_seq_len=1024,
        head_dim=512,
        index_head_dim=128,
        window_size=128,
    )
    base.update(over)
    return DeepseekV4Args(**base)


def _pool(num_pages=8, swa_ratio=0.5, n_scratch=1):
    args = _args()
    sizes = dsv4_pool_sizes(num_pages=num_pages, args=args, swa_ratio=swa_ratio, P=P)
    pool = DSV4PagedKVCache(
        sizes=sizes, args=args, device=DEVICE, dtype=torch.bfloat16, P=P, n_scratch=n_scratch
    )
    return pool, sizes, args


# --------------------------------------------------------------------------- #
# pool construction / tier presence
# --------------------------------------------------------------------------- #
def test_pool_tiers_present_per_ratio():
    pool, sizes, _ = _pool()
    for L, ratio in enumerate(RATIOS):
        assert pool.window_pool[L].shape == (sizes.n_win_slots, 512)
        assert pool.window_pool[L].dtype == torch.bfloat16
        if ratio == 0:
            assert pool.cmp_pool[L] is None
            assert pool.idx_pool[L] is None
            assert pool.state_ring[L] is None
        else:
            assert pool.cmp_pool[L] is not None
            assert pool.cmp_pool[L].shape[1] == 512
            assert pool.state_ring[L] is not None
            if ratio == 4:
                assert pool.idx_pool[L] is not None
                assert pool.idx_pool[L].shape[1] == 128
            else:
                assert pool.idx_pool[L] is None


def test_state_ring_layout_and_dtype():
    pool, sizes, _ = _pool()
    for L, ratio in enumerate(RATIOS):
        if ratio == 0:
            continue
        ring = pool.state_ring[L]
        overlap = ratio == 4
        last_dim = 2 * (1 + int(overlap)) * 512
        assert ring.buffer.dtype == torch.float32
        assert ring.buffer.shape == (sizes.state_slots[L] + 1, last_dim)
        # scratch row -1 initialized: kv == 0, score == -inf.
        assert torch.all(ring.buffer[-1, : ring.item_size] == 0)
        assert torch.all(torch.isneginf(ring.buffer[-1, ring.item_size :]))


def test_indexer_state_ring_separate_object_and_shape():
    # Ratio-4 layers own a SEPARATE indexer compress-state ring (index_head_dim),
    # distinct from the attention ring (head_dim) -- so the two compressors never
    # collide on the same per-page state slots.
    pool, sizes, _ = _pool()
    for L, ratio in enumerate(RATIOS):
        if ratio == 4:
            attn_ring = pool.state_ring[L]
            idx_ring = pool.indexer_state_ring[L]
            assert idx_ring is not None
            # distinct objects AND distinct backing buffers (no aliasing).
            assert idx_ring is not attn_ring
            assert idx_ring.buffer.data_ptr() != attn_ring.buffer.data_ptr()
            # indexer ring: overlap, head_dim=index_head_dim=128 -> item 2*128, last 4*128.
            assert idx_ring.item_size == 2 * 128
            assert idx_ring.ring_size == 8
            assert idx_ring.buffer.dtype == torch.float32
            assert idx_ring.buffer.shape == (sizes.idx_state_slots[L] + 1, 4 * 128)
            # attention ring is the wider head_dim=512 one (proves no collision).
            assert attn_ring.item_size == 2 * 512
            # scratch row initialized.
            assert torch.all(idx_ring.buffer[-1, : idx_ring.item_size] == 0)
            assert torch.all(torch.isneginf(idx_ring.buffer[-1, idx_ring.item_size :]))
        else:
            assert pool.indexer_state_ring[L] is None


# --------------------------------------------------------------------------- #
# state_loc derivation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ring_size", [8, 128])
def test_state_loc_negative_routes_to_scratch(ring_size):
    ws = torch.tensor([-1, -5, 0, 130], dtype=torch.int64)
    out = DSV4PagedKVCache.state_loc(ws, ring_size, P)
    assert out[0].item() == -1
    assert out[1].item() == -1
    # non-negative entries are real slots.
    assert out[2].item() == 0
    assert out[3].item() == (130 // P) * ring_size + (130 % ring_size)


@pytest.mark.parametrize("ring_size", [8, 128])
def test_state_loc_non_contiguous_pages_disjoint(ring_size):
    # Pages allocated out of order: G = 5, 2, 9. Each must map to a disjoint
    # [G*ring, G*ring + ring) block.
    for g in (5, 2, 9):
        ws = torch.arange(g * P, g * P + ring_size, dtype=torch.int64)
        out = DSV4PagedKVCache.state_loc(ws, ring_size, P)
        expected = torch.arange(g * ring_size, g * ring_size + ring_size, dtype=torch.int64)
        assert torch.equal(out, expected)
    # disjointness across the three pages
    blocks = []
    for g in (5, 2, 9):
        blocks.append(set(range(g * ring_size, g * ring_size + ring_size)))
    assert blocks[0].isdisjoint(blocks[1])
    assert blocks[0].isdisjoint(blocks[2])
    assert blocks[1].isdisjoint(blocks[2])


@pytest.mark.parametrize("ring_size", [8, 128])
def test_state_loc_distinct_reqs_distinct_pages_pairwise_distinct(ring_size):
    # Two reqs on distinct pages produce pairwise-distinct state_locs.
    ws_a = torch.arange(3 * P, 3 * P + ring_size, dtype=torch.int64)
    ws_b = torch.arange(7 * P, 7 * P + ring_size, dtype=torch.int64)
    sa = DSV4PagedKVCache.state_loc(ws_a, ring_size, P)
    sb = DSV4PagedKVCache.state_loc(ws_b, ring_size, P)
    assert set(sa.tolist()).isdisjoint(set(sb.tolist()))


# --------------------------------------------------------------------------- #
# per-token window slot -> state_loc correspondence (the C2 / C6 invariant)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ring_size", [8, 128])
def test_window_slot_carries_pos_mod_ring(ring_size):
    # window_slot = page_base + pos%P (page_base a multiple of P). So
    # ws // P == page_index and ws % ring == pos % ring.
    page_index = 4
    page_base = page_index * P
    pos = torch.arange(page_base, page_base + P, dtype=torch.int64)
    ws = page_base + (pos % P)
    assert torch.all(torch.div(ws, P, rounding_mode="floor") == page_index)
    assert torch.all((ws % ring_size) == (pos % ring_size))
    loc = DSV4PagedKVCache.state_loc(ws, ring_size, P)
    assert torch.all(loc == page_index * ring_size + (pos % ring_size))


@pytest.mark.parametrize("ring_size", [8, 128])
def test_carry_straddles_page_boundary(ring_size):
    # Positions 126..131 straddle the page-1/page-2 boundary; if pages are
    # contiguous (base = page*P) the state_locs are contiguous across the
    # boundary, but the page component flips at pos 128.
    pos = torch.arange(126, 132, dtype=torch.int64)
    # contiguous physical pages: ws = pos (page_base == page_index*P)
    ws = pos.clone()
    loc = DSV4PagedKVCache.state_loc(ws, ring_size, P)
    for p, l in zip(pos.tolist(), loc.tolist()):
        page = p // P
        assert l == page * ring_size + (p % ring_size)
    # the boundary: pos 127 is page 0, pos 128 is page 1.
    assert (loc[1] // ring_size).item() == 0  # pos 127
    assert (loc[2] // ring_size).item() == 1  # pos 128


def test_carry_straddle_noncontiguous_physical_pages():
    # Real allocation: tail page may be physically far from the prior page.
    # Map pos 0..255 across two NON-adjacent physical pages (base 9*P and 3*P).
    ring_size = 8
    ws = torch.empty(256, dtype=torch.int64)
    ws[:128] = 9 * P + torch.arange(128) % P
    ws[128:] = 3 * P + torch.arange(128) % P
    loc = DSV4PagedKVCache.state_loc(ws, ring_size, P)
    # page-0 tokens -> block [9*ring, 9*ring+ring); page-1 tokens -> [3*ring, ...)
    assert torch.all(torch.div(loc[:128], ring_size, rounding_mode="floor") == 9)
    assert torch.all(torch.div(loc[128:], ring_size, rounding_mode="floor") == 3)
    # within-block offset still tracks pos%ring
    assert torch.all((loc[:128] % ring_size) == (torch.arange(128) % ring_size))


# --------------------------------------------------------------------------- #
# set_state re-clears the scratch row each call
# --------------------------------------------------------------------------- #
def test_set_state_reclears_scratch_each_call():
    pool, sizes, _ = _pool()
    L = 2  # ratio 4
    ring = pool.state_ring[L]
    last_dim = ring.buffer.shape[1]
    # Corrupt the scratch row, then a set_state must restore it.
    ring.buffer[-1].fill_(1.0)
    state_loc = torch.tensor([0, 5], dtype=torch.int64)
    val = torch.randn(2, last_dim, dtype=torch.float32)
    pool.set_state(L, state_loc, val)
    # written rows hold the value
    assert torch.equal(pool.get_state(L, state_loc), val)
    # scratch row restored
    assert torch.all(ring.buffer[-1, : ring.item_size] == 0)
    assert torch.all(torch.isneginf(ring.buffer[-1, ring.item_size :]))
    # writing TO the scratch row (-1) is also re-cleared afterwards
    scratch_val = torch.randn(1, last_dim, dtype=torch.float32)
    pool.set_state(L, torch.tensor([-1]), scratch_val)
    assert torch.all(ring.buffer[-1, : ring.item_size] == 0)
    assert torch.all(torch.isneginf(ring.buffer[-1, ring.item_size :]))


def test_get_state_negative_returns_scratch():
    pool, _, _ = _pool()
    L = 3  # ratio 128
    ring = pool.state_ring[L]
    got = pool.get_state(L, torch.tensor([-1]))
    assert torch.all(got[0, : ring.item_size] == 0)
    assert torch.all(torch.isneginf(got[0, ring.item_size :]))


# --------------------------------------------------------------------------- #
# specialized writes
# --------------------------------------------------------------------------- #
def test_store_window_scatters():
    pool, sizes, _ = _pool()
    L = 0
    slots = torch.tensor([0, 3, 7], dtype=torch.int64)
    k = torch.randn(3, 512, dtype=torch.bfloat16)
    pool.store_window(k, L, slots)
    assert torch.equal(pool.window_pool[L][slots], k)


def test_store_compressed_and_indexer():
    pool, sizes, _ = _pool()
    L = 2  # ratio 4 -> has both cmp + idx
    cmp_slots = torch.tensor([1, 4], dtype=torch.int64)
    cmp = torch.randn(2, 512, dtype=torch.bfloat16)
    pool.store_compressed(cmp, L, cmp_slots)
    assert torch.equal(pool.cmp_pool[L][cmp_slots], cmp)

    idx_slots = torch.tensor([0, 2], dtype=torch.int64)
    idxk = torch.randn(2, 128, dtype=torch.bfloat16)
    pool.store_indexer(idxk, L, idx_slots)
    assert torch.equal(pool.idx_pool[L][idx_slots], idxk)


def test_store_indexer_rejects_ratio128_layer():
    pool, _, _ = _pool()
    L = 3  # ratio 128 -> no indexer pool
    with pytest.raises(AssertionError):
        pool.store_indexer(torch.randn(1, 128, dtype=torch.bfloat16), L, torch.tensor([0]))


# --------------------------------------------------------------------------- #
# ABC compat
# --------------------------------------------------------------------------- #
def test_abc_kv_aliases_window_and_total_bytes():
    pool, _, _ = _pool()
    assert pool.k_cache(0) is pool.window_pool[0]
    assert pool.v_cache(0) is pool.window_pool[0]
    assert pool.num_layers == 8
    assert pool.device == DEVICE
    assert pool.dtype == torch.bfloat16
    assert pool.total_bytes() > 0
    # store_kv shim writes the window pool
    k = torch.randn(2, 512, dtype=torch.bfloat16)
    out_loc = torch.tensor([1, 2], dtype=torch.int64)
    pool.store_kv(k, k, out_loc, 0)
    assert torch.equal(pool.window_pool[0][out_loc], k)


def test_compress_state_ring_overlap_widths():
    # overlap True (ratio4) -> item_size = 2*head_dim; False (ratio128) -> head_dim.
    r4 = CompressStateRing(4, 8, overlap=True, head_dim=512, device=DEVICE)
    r128 = CompressStateRing(4, 128, overlap=False, head_dim=512, device=DEVICE)
    assert r4.item_size == 2 * 512
    assert r128.item_size == 512
    assert r4.buffer.shape[1] == 4 * 512
    assert r128.buffer.shape[1] == 2 * 512


# --------------------------------------------------------------------------- #
# full-loc -> physical-tier translation (sglang deepseek_v4_memory_pool semantics).
# The logical full-token index space is the single bookkeeping currency: window slots
# come from ONE mapping (`full_to_window`, == sglang full_to_swa_index_mapping), the
# compressed/indexer rows are pure arithmetic (`full_loc // ratio`), the state ring
# derives from the window slot. Sentinel rules (C2-verified): -1 is GATHER-only-safe
# (the mapping keeps a permanent -1 last row, so a fancy-index wrap reads -1); no
# scatter/index_copy_ path may ever receive a negative slot (OOB raise, not a wrap).
# --------------------------------------------------------------------------- #
def test_mapping_shape_and_initially_unmapped():
    pool, sizes, _ = _pool(num_pages=16)
    assert pool.full_to_window.shape == (sizes.full_token + 1,)
    # Everything (incl. the permanent last sentinel row) starts unmapped == -1.
    assert (pool.full_to_window == -1).all()


def test_bind_translate_roundtrip_and_noncontiguous_pages():
    pool, sizes, _ = _pool(num_pages=16)
    # full page 0 -> window page at slot 256; full page 3 -> window page at slot 0
    pool.bind_window_pages(full_page_base=0, window_page_base=256)
    pool.bind_window_pages(full_page_base=3 * P, window_page_base=0)
    pos = torch.tensor([0, 1, 127], dtype=torch.int64)
    assert pool.translate_full_to_window(pos).tolist() == [256, 257, 256 + 127]
    pos2 = torch.tensor([3 * P, 3 * P + 5], dtype=torch.int64)
    assert pool.translate_full_to_window(pos2).tolist() == [0, 5]
    # In-between page 1 stays unmapped.
    assert pool.translate_full_to_window(torch.tensor([P + 7])).item() == -1


def test_translate_minus_one_is_gather_safe():
    pool, _, _ = _pool(num_pages=16)
    pool.bind_window_pages(full_page_base=0, window_page_base=0)
    # -1 full locs wrap to the permanent -1 sentinel LAST row -- never a live slot.
    out = pool.translate_full_to_window(torch.tensor([-1, 5, -1], dtype=torch.int64))
    assert out.tolist() == [-1, 5, -1]


def test_unbind_clears_mapping_and_is_idempotent():
    pool, _, _ = _pool(num_pages=16)
    pool.bind_window_pages(full_page_base=0, window_page_base=0)
    locs = torch.arange(P, dtype=torch.int64)
    pool.unbind_window_pages(locs)
    assert (pool.translate_full_to_window(locs) == -1).all()
    pool.unbind_window_pages(locs)  # freeing an already-unmapped range is a no-op
    assert (pool.translate_full_to_window(locs) == -1).all()
    # Negative entries in the free list are dropped, not scattered (OOB-unsafe).
    pool.unbind_window_pages(torch.tensor([-1, 0], dtype=torch.int64))


def test_cmp_rows_arithmetic_and_ratio_divides_P():
    pool, _, _ = _pool(num_pages=16)
    full = torch.tensor([0, 3, 4, 127, 128, 4 * P - 1], dtype=torch.int64)
    assert pool.cmp_rows(full, 4).tolist() == [0, 0, 1, 31, 32, P - 1]
    assert pool.cmp_rows(full, 128).tolist() == [0, 0, 0, 0, 1, 3]
    # -1 propagates negative (floor division), never a valid row.
    assert pool.cmp_rows(torch.tensor([-1]), 4).item() < 0


def test_cmp_rows_stay_below_scratch_base():
    pool, sizes, _ = _pool(num_pages=16)
    # The arithmetic range [0, full_token//ratio) must never collide with the
    # scratch rows appended at cmp_scratch_base (== sizes.cmp_blocks[L]).
    for L, ratio in enumerate(RATIOS):
        if ratio == 0:
            continue
        top = pool.cmp_rows(torch.tensor([sizes.full_token - 1]), ratio).item()
        assert top < pool.cmp_scratch_base[L], (
            f"layer {L}: arithmetic row {top} reaches scratch base {pool.cmp_scratch_base[L]}"
        )


def test_state_loc_across_page_boundary_from_translated_slots():
    pool, _, _ = _pool(num_pages=16)
    pool.bind_window_pages(full_page_base=0, window_page_base=2 * P)  # window page 2
    pool.bind_window_pages(full_page_base=P, window_page_base=5 * P)  # window page 5
    ws = pool.translate_full_to_window(torch.tensor([127, 128], dtype=torch.int64))
    ring = 8  # ratio-4
    locs = DSV4PagedKVCache.state_loc(ws, ring, P)
    # Distinct window pages -> disjoint ring blocks (2*8 + 127%8, 5*8 + 0).
    assert locs.tolist() == [2 * ring + 127 % ring, 5 * ring + 0]
    # Slid-out (-1) window slot routes to the -1 state scratch.
    assert DSV4PagedKVCache.state_loc(torch.tensor([-1]), ring, P).item() == -1


def test_cmp_rows_arithmetic_matches_expectation_across_page_boundary():
    # The row of the block covering [b*ratio, (b+1)*ratio) is full_loc(b*ratio) // ratio and is
    # CONSTANT across every position in the block, even where the block sits at a page boundary --
    # because a full page (128) is divisible by the ratio, so no block straddles two pages.
    pool, _, _ = _pool(num_pages=16)
    for ratio in (4, 128):
        for base in (0, 3 * P, 7 * P):  # arbitrary (non-contiguous) page bases
            for pos_in_block in range(ratio):
                # block starting at the LAST block of the page + an interior offset
                b_start = base + (P - ratio)
                full = torch.tensor([b_start + pos_in_block], dtype=torch.int64)
                assert pool.cmp_rows(full, ratio).item() == (b_start) // ratio


def test_decode_cmp_scatter_masking_mirror():
    # Mirror the in-graph decode cmp scatter (model.Compressor.decode_step): the completed block's
    # store row is full_snap(pos) // ratio; a row whose block did NOT complete this step (should
    # False) scatters to its OWN scratch row (scratch_base + row). Never a negative store index.
    # n_scratch = one row per possible decode row (the manager sizes it max_running_req + 1); 4
    # here covers the 3 rows below with collision-free per-row scratch.
    pool, _, _ = _pool(num_pages=16, n_scratch=4)
    ratio, L = 4, 2  # ratio-4 layer
    scratch_base = pool.cmp_scratch_base[L]
    n_scratch_rows = pool.cmp_pool[L].shape[0] - scratch_base
    # 3 decode rows: row 0 completes a block (pos+1 % ratio == 0), row 1 does not, row 2 = dummy.
    pool.full_loc_map = torch.full((3, 512), -1, dtype=torch.int64, device=DEVICE)
    pool.full_loc_map[0, :8] = torch.arange(3 * P, 3 * P + 8)  # window/full page 3
    pool.full_loc_map[1, :8] = torch.arange(7 * P, 7 * P + 8)  # window/full page 7
    pool.full_loc_map[2, :] = 0                                # dummy
    snap = pool.full_loc_map.clone()  # what the attention backend stages per decode batch
    rows = torch.arange(3)
    pos = torch.tensor([3, 4, 0])  # row0: (3+1)%4==0 completes; row1: no; row2: dummy pos 0
    should = (pos + 1) % ratio == 0
    cmp_row = pool.cmp_rows(snap[rows, pos], ratio)
    scratch = rows + scratch_base
    cmp_dst = torch.where(should, cmp_row, scratch)
    assert (cmp_dst >= 0).all()  # NEVER a negative store index (index_copy_ would OOB)
    assert cmp_dst[0].item() == (3 * P + 3) // ratio  # completed -> arithmetic row (< scratch)
    assert cmp_dst[0].item() < scratch_base
    assert cmp_dst[1].item() == scratch_base + 1  # incomplete -> own scratch row
    assert cmp_dst[2].item() == (0 // ratio if should[2] else scratch_base + 2)  # dummy safe
    assert n_scratch_rows >= 3  # one scratch row per decode row (collision-free)


def test_decode_topk_blocks_beyond_valid_masked_to_minus_one():
    # Mirror the in-graph decode cmp READ (model.Attention.decode_step): block b -> row
    # full_snap(b*ratio) // ratio, but blocks >= valid=(pos+1)//ratio are masked to -1 (a
    # gather-only sentinel the kernel ignores) and never reach a store.
    pool, sizes, _ = _pool(num_pages=16)
    ratio = 4
    pool.full_loc_map = torch.full((1, 512), -1, dtype=torch.int64, device=DEVICE)
    pool.full_loc_map[0, :16] = torch.arange(5 * P, 5 * P + 16)
    snap = pool.full_loc_map.clone()  # what the attention backend stages per decode batch
    pos = torch.tensor([7])          # valid blocks = (7+1)//4 = 2
    n_stage = 4
    blk = torch.arange(n_stage)      # candidate block indices 0..3
    valid = (pos + 1) // ratio       # 2
    col_valid = blk[None, :] < valid[:, None]
    full_at = snap[torch.zeros(1, dtype=torch.int64)[:, None], (blk * ratio)[None, :]]
    rows = pool.cmp_rows(full_at, ratio)
    picked = torch.where(col_valid, rows, torch.full_like(rows, -1))
    assert (picked[0, :2] >= 0).all()          # valid blocks -> real rows
    assert (picked[0, 2:] == -1).all()         # blocks beyond valid -> -1 sentinel


# --------------------------------------------------------------------------- #
# generic swa_pool duck-type (the CacheManager plug-in surface): page-atomic alloc_swa /
# free_swa over token-face indices, tail dummy binding, conservation, idempotency.
# --------------------------------------------------------------------------- #
def _paged_pool(num_pages=8, swa_ratio=0.5):
    pool, sizes, args = _pool(num_pages=num_pages, swa_ratio=swa_ratio)
    pool._init_paged_state(max_running_req=2, radix=True)
    return pool, sizes


def _expand(bases):
    off = torch.arange(P, dtype=torch.int64)
    return (torch.tensor(bases, dtype=torch.int64)[:, None] + off).flatten()


def test_swa_iface_alloc_binds_pages_and_preserves_in_page_offsets():
    pool, sizes = _paged_pool()
    cap = sizes.n_win_slots - P                       # tail window page reserved for the dummy
    assert pool.swa_available_size() == cap
    assert pool.swa_num_tokens - 1 == cap             # generic sentinel-slot cap convention

    pool.alloc_swa(_expand([0, 2 * P]))               # two full pages
    assert pool.swa_available_size() == cap - 2 * P
    for fbase in (0, 2 * P):
        ws = pool.translate_loc_from_full_to_swa(torch.arange(fbase, fbase + P))
        assert (ws >= 0).all()
        assert torch.equal(ws - ws[0], torch.arange(P, dtype=torch.int64))  # offsets preserved
        assert int(ws[0]) % P == 0                                          # page-aligned base


def test_swa_iface_free_returns_pages_and_is_idempotent():
    pool, sizes = _paged_pool()
    cap = sizes.n_win_slots - P
    pool.alloc_swa(_expand([0, P, 3 * P]))
    pool.free_swa(_expand([P]))
    assert pool.swa_available_size() == cap - 2 * P
    assert (pool.translate_loc_from_full_to_swa(torch.arange(P, 2 * P)) == -1).all()
    pool.free_swa(_expand([P]))                       # double free: unbound -> no-op
    assert pool.swa_available_size() == cap - 2 * P
    pool.free_swa(_expand([0, 3 * P]))
    assert pool.swa_available_size() == cap           # full conservation


def test_swa_iface_rejects_partial_pages_and_survives_int32():
    pool, _ = _paged_pool()
    pool.alloc_swa(_expand([0]).to(torch.int32))      # int32 page_table values accepted
    with pytest.raises(AssertionError):
        pool.free_swa(torch.arange(0, P // 2))        # half a page -> hard error
    pool.free_swa(_expand([0]).to(torch.int32))


def test_swa_iface_exhaustion_raises_and_dummy_stays_bound():
    pool, sizes = _paged_pool()
    n_alloc_pages = (sizes.n_win_slots - P) // P
    pool.alloc_swa(_expand([i * P for i in range(n_alloc_pages)]))
    assert pool.swa_available_size() == 0
    with pytest.raises(RuntimeError):
        pool.alloc_swa(_expand([n_alloc_pages * P]))
    # The tail dummy region is bound outside the free-list: dummy full page -> dummy window page.
    dummy_ws = pool.translate_loc_from_full_to_swa(
        torch.arange(sizes.full_token - P, sizes.full_token)
    )
    assert int(dummy_ws[0]) == sizes.n_win_slots - P and (dummy_ws >= 0).all()
