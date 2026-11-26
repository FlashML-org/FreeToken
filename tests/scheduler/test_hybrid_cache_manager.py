"""P2b integration: CacheManager hybrid path (match_req -> cache_req donate -> prefix hit).
CPU, real LinearStatePool + page_table, hand-built Reqs. Exercises the two-currency wiring
without the full scheduler/engine."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager


def _pool(num_slots=16):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)


def _pend(ids):
    # int32 to match production Req.input_ids dtype (fast_compare_key needs consistent dtype)
    t = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(input_ids=t, input_len=len(ids))


def test_hybrid_cache_manager_donate_then_hit():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)
    assert cm.is_hybrid

    # cold match on an empty tree
    mr = cm.match_req(_pend([1, 2, 3, 4, 5]))
    assert mr.cuda_handle.cached_len == 0 and mr.mamba_value is None

    # admit req A: allocate live + ping-pong, stage KV pages, mark a ×N snapshot at boundary 4
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    page_table[0, :4] = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
    reqA = Req(input_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32), table_idx=0,
               cached_len=4, output_len=1, uid=0, sampling_params=SamplingParams(),
               cache_handle=mr.cuda_handle)
    reqA.linear_slot_idx, reqA.mamba_ping_pong = live, pp
    reqA.mamba_next_track_idx = 1            # flipped from 0 in build_fla_metadata; frozen = pp[0]
    reqA.mamba_last_track_seqlen = 4
    cm.lock(mr.cuda_handle)

    free_before = pool.num_free_slots
    cm.cache_req(reqA, finished=False)       # donate pp[0] at boundary 4; replace it in the pair
    # pp[0] donated to the tree; a fresh replacement was alloc'd -> net free-slot count unchanged
    assert pool.num_free_slots == free_before - 1  # one replacement alloc'd (donated slot now tree-owned)
    assert reqA.mamba_ping_pong[0] != pp[0]        # slot 0 replaced; pp[0] now lives in the tree

    # req B shares the [1,2,3,4] prefix -> HIT: returns the donated snapshot + reused KV
    mrB = cm.match_req(_pend([1, 2, 3, 4, 9]))
    assert mrB.cuda_handle.cached_len == 4
    assert mrB.mamba_value == pp[0]
    assert mrB.cuda_handle.get_matched_indices().tolist() == [100, 101, 102, 103]


def test_hybrid_finish_donates_live_slot():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)

    mr = cm.match_req(_pend([7, 8, 9, 10]))
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    page_table[1, :3] = torch.tensor([200, 201, 202], dtype=torch.int32)
    req = Req(input_ids=torch.tensor([7, 8, 9, 10], dtype=torch.int32), table_idx=1,
              cached_len=3, output_len=1, uid=1, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    cm.lock(mr.cuda_handle)

    cm.cache_req(req, finished=True)         # donate the live slot directly (final state)
    # ping-pong pair freed; live slot kept (now owned by the tree)
    mr2 = cm.match_req(_pend([7, 8, 9, 10]))
    assert mr2.cuda_handle.cached_len == 3 and mr2.mamba_value == live


def test_free_req_slots_idempotent():
    """C2: a finish/abort double-free of the same request must NOT push its GDN slots twice."""
    pool = _pool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    req = Req(input_ids=torch.tensor([1, 2, 3], dtype=torch.int32), table_idx=0, cached_len=2,
              output_len=1, uid=0, sampling_params=SamplingParams(), cache_handle=None)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    base = pool.num_free_slots
    cm._free_req_slots(req)
    assert pool.num_free_slots == base + 3        # live + 2 ping-pong returned once
    cm._free_req_slots(req)                        # second free (abort/finish race)
    assert pool.num_free_slots == base + 3         # idempotent: nothing pushed twice


def test_rebuild_reclaims_donated_gdn_slots():
    """C5: a runtime cache rebuild must return the discarded tree's GDN snapshot slots (idle)."""
    pool = _pool(num_slots=16)
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    mr = cm.match_req(_pend([7, 8, 9, 10]))
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    pt[1, :3] = torch.tensor([200, 201, 202], dtype=torch.int32)
    req = Req(input_ids=torch.tensor([7, 8, 9, 10], dtype=torch.int32), table_idx=1, cached_len=3,
              output_len=1, uid=1, sampling_params=SamplingParams(), cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    cm.lock(mr.cuda_handle)
    cm.cache_req(req, finished=True)              # donates `live` to the tree, frees ping-pong
    assert pool.num_free_slots < pool.num_slots - 1   # a slot is now tree-owned
    cm.rebuild(64, pt)                            # idle rebuild discards the tree
    assert pool.num_free_slots == pool.num_slots - 1  # all GDN slots reclaimed (no leak)


def test_prefill_chunk_ends_on_a_page_boundary():
    """A hybrid chunk must end page-aligned: the snapshot commit skips any other boundary."""
    from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool()
    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "hybrid_radix", linear_state_pool=pool)
    assert cm.prefill_chunk_align == 64
    tm = TableManager(max_running_reqs=4, page_table=pt)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))

    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    req = adder.try_add_one(pending)
    assert isinstance(req, ChunkedReq) and req.extend_len == 64

    # a budget below one page keeps the unaligned chunk rather than stalling the request
    adder = PrefillAdder(token_budget=40, reserved_size=0, cache_manager=cm, table_manager=tm)
    assert adder.try_add_one(pending).extend_len == 40


def test_naive_cache_does_not_align_prefill_chunks():
    """The alignment hook is hybrid-only; every other cache keeps the raw budget chunk."""
    from freetoken.scheduler.prefill import PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "radix")
    assert cm.prefill_chunk_align == 1
    tm = TableManager(max_running_reqs=4, page_table=pt)
    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))
    assert adder.try_add_one(pending).extend_len == 100


def test_pool_sizing_covers_4mr_floor():
    """C6: pool must reserve the 4-slot-per-request non-evictable floor even at a tiny ratio."""
    from types import SimpleNamespace
    from freetoken.kvcache.linear_state_pool import _linear_pool_num_slots
    for mr in (1, 8, 64):
        c = SimpleNamespace(max_running_req=mr, cache_type="hybrid_radix",
                            linear_state_cache_ratio=0.1)
        assert _linear_pool_num_slots(c) >= 4 * mr + 1, (mr, _linear_pool_num_slots(c))


def _window_state_cache(monkeypatch, max_running_req: int, window: int = 8):
    """swa_radix with one window state (Inkling 2's convolution inputs, 3 tokens) at its pool floor, and a
    ``prefill(uid, ids, chunk)`` that runs a prompt through and commits it, returning (req, hit, rows per chunk)."""
    from freetoken import core
    from freetoken.core import Context
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache, _swa_paged_num_tokens, state_num_rows
    from freetoken.models.config import KVCacheGroupSpec, WindowStateSpec
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import PrefillManager
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", Context(page_size=1))
    groups = (KVCacheGroupSpec(name="full", layer_ids=(1,), num_kv_heads=1, head_dim=8, sliding_window=None),
              KVCacheGroupSpec(name="swa", layer_ids=(0,), num_kv_heads=1, head_dim=8, sliding_window=window))
    state = WindowStateSpec(name="conv", shape=(8,), layer_ids=(0, 1), window=3)
    mc = SimpleNamespace(kv_cache_group_specs=lambda: groups, window_states=(state,),
                         swa_attention_group=lambda: SimpleNamespace(sliding_window=window))
    cfg = SimpleNamespace(max_running_req=max_running_req, cache_type="swa_radix", page_size=1, swa_full_tokens_ratio=0.01,
                          swa_num_pages_override=None, decode_rows_per_req=1, model_config=mc)
    num_swa = _swa_paged_num_tokens(cfg, 256)
    kv = HybridSWAKVCache(groups=groups, num_layers=2, num_full_pages=256, page_size=1, dtype=torch.bfloat16,
                          device=torch.device("cpu"), num_swa_tokens=num_swa, window_states=(state,),
                          num_state_rows=state_num_rows(cfg, num_swa))
    pt = torch.zeros(max_running_req + 1, 128, dtype=torch.int32)
    cm = CacheManager(256, 1, pt, "swa_radix", swa_pool=kv, sliding_window_size=window)
    tm = TableManager(max_running_reqs=max_running_req, page_table=pt)
    pm = PrefillManager(cm, tm, DecodeManager(page_size=1))

    def prefill(uid, ids, chunk):
        pm.pending_list = [PendingReq(uid=uid, input_ids=torch.tensor(ids, dtype=torch.int32),
                                      sampling_params=SamplingParams(max_tokens=32))]
        hit, spans = None, []
        while True:
            batch = pm.schedule_next_batch(chunk)
            assert batch is not None, "admission waits for window rows"
            (req,) = batch.reqs
            hit = req.cache_handle.cached_len if hit is None else hit
            cm.free_swa_out_of_window_extend(batch.reqs)
            cm.allocate_paged(batch.reqs)
            spans.append(cm._state_spans(req, 2))
            req.complete_one()
            if req.device_len - 1 == len(ids):
                cm.cache_req(req, finished=False)
                return req, hit, spans

    return cm, tm, kv, pt, prefill


def test_window_state_radix_at_its_floor_resumes_where_a_request_ended(monkeypatch):
    """swa_radix with a window state (Inkling 2's convolution inputs) at one running request and
    the pools at their floor: a prompt prefilled in chunks keeps only each chunk's tail rows, and
    the next turn resumes where the request ended, reading rows the decode kept; every row goes
    back once the cache is flushed."""
    cm, tm, _, _, prefill = _window_state_cache(monkeypatch, max_running_req=1)
    rows = lambda req: cm._state_spans(req, 2)

    a, _, spans = prefill(0, list(range(1, 41)), chunk=16)
    # each chunk gets rows for its last 3 tokens only; the last chunk, shorter than the 16-token retain gap, all of them
    assert spans == [((13, 16),), ((13, 16), (29, 32)), ((29, 40),)]
    assert rows(a) == ()  # the commit handed the tail to the tree and freed the rest
    for _ in range(20):
        a.append_host(torch.tensor([99], dtype=torch.int32))
        a.decode_batch_idx += 1
        cm.maybe_free_swa_out_of_window([a])
        cm.allocate_paged([a])
        a.complete_one()
    assert sum(hi - lo for lo, hi in rows(a)) < 3 + 16 + 2  # decode frees behind the window
    cm.cache_req(a, finished=True)
    tm.free(a.table_idx)

    b, hit, _ = prefill(1, a.input_ids.tolist() + [7, 8, 9], chunk=64)
    assert hit == a.cached_len
    cm.cache_req(b, finished=True)
    tm.free(b.table_idx)
    cm.flush()
    cm.check_integrity()


def test_every_request_of_a_decode_batch_keeps_its_window_kv(monkeypatch):
    """The decode-time free of out-of-window KV uses the KV's window for every request of the batch, not the
    window state's shorter one: each request still holds the window KV of its last ``window`` tokens."""
    import freetoken.scheduler.cache as cache_mod

    window = 8
    monkeypatch.setattr(cache_mod, "_SWA_EVICTION_INTERVAL", 1)
    cm, _, kv, pt, prefill = _window_state_cache(monkeypatch, max_running_req=2, window=window)
    reqs = [prefill(0, list(range(1, 41)), chunk=64)[0], prefill(1, list(range(50, 90)), chunk=64)[0]]
    for _ in range(24):
        for req in reqs:
            req.append_host(torch.tensor([99], dtype=torch.int32))
            req.decode_batch_idx += 1
        cm.maybe_free_swa_out_of_window(reqs)
        cm.allocate_paged(reqs)
        for req in reqs:
            req.complete_one()
    for req in reqs:
        last = pt[req.table_idx, req.device_len - 1 - window : req.device_len - 1].long()
        assert bool((kv.full_to_swa_index_mapping[last] > 0).all()), req.uid


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
