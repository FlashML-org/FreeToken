"""Regression test for the chunked-prefill page-accounting bug.

An earlier gemma4 perf change added ``cache_manager.cache_req(req, finished=False)`` to the
``ChunkedReq`` branch of ``Scheduler._process_last_data`` -- caching every *intermediate*
chunk. Under overlap scheduling the next chunk is created (snapshotting the prior chunk's
``cache_handle``) BEFORE that cache_req runs, so each continuation carries a stale handle
whose ``cached_len`` is behind reality and ``cache_req`` re-frees the prior chunk's pages.
Any multi-chunk prefill (codex's large prompts) then crashes the scheduler at the next idle
with ``CacheManager integrity check failed``.

The fix reverts to mini-sglang's behavior: do NOT cache intermediate chunks; the whole
prompt is inserted once when the final (non-chunked) chunk is processed.
"""

from __future__ import annotations

import torch

CHUNK = 8
WIDTH = 64
MAX_RUNNING = 4
UID = 7


def _setup_context() -> None:
    from freetoken.core import Context, get_global_ctx, set_global_ctx

    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=1))


def _build_managers(num_pages):
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import PrefillManager
    from freetoken.scheduler.table import TableManager

    _setup_context()
    pt = torch.zeros((MAX_RUNNING + 1, WIDTH), dtype=torch.int32, device="cpu")
    cm = CacheManager(num_pages=num_pages, page_size=1, page_table=pt, type="radix")
    tm = TableManager(max_running_reqs=MAX_RUNNING, page_table=pt)
    dm = DecodeManager(page_size=1)
    pm = PrefillManager(cm, tm, dm)
    return cm, tm, dm, pm


def _drive_chunked_prefill(cm, tm, pm, n_chunks):
    """Drive one chunked-prefill request through ``n_chunks``, faithfully replicating overlap
    ordering: each iteration schedules+forwards the NEXT chunk (reading the prior chunk's
    cache_handle) BEFORE the PREVIOUS chunk is cached. Intermediate chunks are NOT cached
    (like the real scheduler's ``continue``); the whole prompt is cached once when the final
    non-chunked chunk is processed."""
    from freetoken.core import SamplingParams
    from freetoken.scheduler.prefill import ChunkedReq
    from freetoken.scheduler.utils import PendingReq

    prompt_len = n_chunks * CHUNK
    pm.pending_list = [PendingReq(uid=UID, input_ids=torch.arange(prompt_len, dtype=torch.int32),
                                  sampling_params=SamplingParams(max_tokens=4))]
    last_batch = None
    final_req = None
    while pm.runnable or last_batch is not None:
        batch = pm.schedule_next_batch(CHUNK)          # step 2: schedule next chunk
        if batch is not None:
            cm.allocate_paged(batch.reqs)              # _prepare_batch allocates pages
            for r in batch.reqs:
                r.complete_one()                       # forward advances cached_len
        if last_batch is not None:                     # step 3: process the PREVIOUS batch
            for r in last_batch.reqs:
                if not isinstance(r, ChunkedReq):
                    cm.cache_req(r, finished=False)    # final chunk cached once
                    final_req = r
        last_batch = batch
    cm.cache_req(final_req, finished=True)             # request finishes -> release
    tm.free(final_req.table_idx)


def test_multichunk_overlap_no_double_free():
    """The fix: not caching intermediate chunks keeps page accounting consistent across a
    multi-chunk prefill, and still caches the whole prompt (reusable) exactly once."""
    cm, tm, _dm, pm = _build_managers(num_pages=64)
    _drive_chunked_prefill(cm, tm, pm, n_chunks=4)

    cm.check_integrity()  # free_slots + cache == num_pages; no chunk freed twice
    si = cm.prefix_cache.size_info
    assert si.protected_size == 0  # request released -> no leaked locks / ref_count drift
    assert si.evictable_size == 4 * CHUNK  # whole prompt retained in the prefix cache
    assert len(cm.free_slots) == cm.num_pages - 4 * CHUNK


def test_radix_hit_admission_reports_nonzero_cached_tokens():
    """A second request sharing a cached prefix admits with the prefix-cache hit recorded."""
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    cm, tm, _dm, pm = _build_managers(num_pages=64)
    _drive_chunked_prefill(cm, tm, pm, n_chunks=4)  # whole 32-token prompt now in the radix cache

    cached_len = 4 * CHUNK
    prompt_len = cached_len + CHUNK
    pm.pending_list = [
        PendingReq(uid=UID + 1, input_ids=torch.arange(prompt_len, dtype=torch.int32),
                   sampling_params=SamplingParams(max_tokens=4))
    ]
    batch = pm.schedule_next_batch(prompt_len)
    assert batch is not None
    assert batch.prompt_admissions == [(UID + 1, prompt_len, cached_len)]
    assert batch.log_cached_tokens == cached_len


def test_chunked_prompt_admission_reports_complete_length_once():
    """The first prepared chunk carries the full prompt usage; continuations carry none."""
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    cm, _tm, _dm, pm = _build_managers(num_pages=64)
    prompt_len = 3 * CHUNK
    pm.pending_list = [
        PendingReq(
            uid=UID,
            input_ids=torch.arange(prompt_len, dtype=torch.int32),
            sampling_params=SamplingParams(max_tokens=4),
        )
    ]

    admissions = []
    while pm.runnable:
        batch = pm.schedule_next_batch(CHUNK)
        assert batch is not None
        admissions.append(list(batch.prompt_admissions))
        cm.allocate_paged(batch.reqs)
        for req in batch.reqs:
            req.complete_one()

    assert admissions[0] == [(UID, prompt_len, 0)]
    assert all(items == [] for items in admissions[1:])


def test_batched_prefill_carries_each_new_prompt_admission():
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    _cm, _tm, _dm, pm = _build_managers(num_pages=64)
    pm.pending_list = [
        PendingReq(1, torch.arange(3, dtype=torch.int32), SamplingParams(max_tokens=2)),
        PendingReq(2, torch.arange(5, dtype=torch.int32), SamplingParams(max_tokens=2)),
    ]
    batch = pm.schedule_next_batch(16)
    assert batch is not None
    assert batch.prompt_admissions == [(1, 3, 0), (2, 5, 0)]


def test_chunk_ends_before_an_image_it_would_cut_on_bidirectional_models():
    from freetoken.core import SamplingParams
    from freetoken.message import MMItem
    from freetoken.scheduler.utils import PendingReq

    def chunk_lens(keep_images_whole):
        cm, tm, dm, pm = _build_managers(num_pages=64)
        pm.keep_images_whole = keep_images_whole
        image = MMItem(modality="image", hash=1, pad_value=1, offsets=[[10, 20]], feature=torch.zeros(1))
        pm.pending_list = [PendingReq(uid=UID, input_ids=torch.arange(30, dtype=torch.int32),
                                      sampling_params=SamplingParams(max_tokens=4), mm_items=[image])]
        lens = []
        while pm.runnable:
            batch = pm.schedule_next_batch(15)
            cm.allocate_paged(batch.reqs)
            for r in batch.reqs:
                lens.append(r.extend_len)
                r.complete_one()
        return lens

    assert chunk_lens(True) == [10, 15, 5]  # a 15-token chunk would cut the image at 10..20: stop at 10, the image goes whole into the next chunk
    assert chunk_lens(False) == [15, 15]  # causal models lose nothing to a cut image and keep the plain chunking


def test_a_shared_pass_defers_an_image_it_cannot_hold_whole():
    """Another request took most of the pass: the image waits for a pass of its own instead of being cut at the leftover."""
    from freetoken.core import SamplingParams
    from freetoken.message import MMItem
    from freetoken.scheduler.utils import PendingReq

    cm, tm, dm, pm = _build_managers(num_pages=128)
    pm.keep_images_whole = True
    text = PendingReq(uid=UID, input_ids=torch.arange(35, dtype=torch.int32), sampling_params=SamplingParams(max_tokens=4))
    image = MMItem(modality="image", hash=1, pad_value=1, offsets=[[0, 20]], feature=torch.zeros(1))
    with_image = PendingReq(uid=UID + 1, input_ids=torch.arange(100, 130, dtype=torch.int32),
                            sampling_params=SamplingParams(max_tokens=4), mm_items=[image])
    pm.pending_list = [text, with_image]
    first = pm.schedule_next_batch(40)
    assert [(r.uid, r.extend_len) for r in first.reqs] == [(UID, 35)]  # the 5 tokens left cannot hold the 20-token image
    cm.allocate_paged(first.reqs)
    for r in first.reqs:
        r.complete_one()
    second = pm.schedule_next_batch(40)
    assert [(r.uid, r.extend_len) for r in second.reqs] == [(UID + 1, 30)]


def test_the_sliding_window_pool_cap_cannot_recut_an_image():
    """The image boundary is decided after the pool cap: a pool of 5000 tokens ends the chunk before the image at 4900, not inside it."""
    from freetoken.core import SamplingParams
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
    from freetoken.message import MMItem
    from freetoken.models.config import KVCacheGroupSpec
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.mm import cut_image_spans
    from freetoken.scheduler.prefill import PrefillManager
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    _setup_context()
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    groups = (
        KVCacheGroupSpec(name="full", layer_ids=(1,), num_kv_heads=1, head_dim=8, sliding_window=None),
        KVCacheGroupSpec(name="swa", layer_ids=(0,), num_kv_heads=1, head_dim=8, sliding_window=1024),
    )
    pool = HybridSWAKVCache(groups=groups, num_layers=2, num_full_pages=8192, page_size=1, dtype=torch.bfloat16,
                            device=torch.device("cpu"), num_swa_tokens=5000)
    pt = torch.zeros((MAX_RUNNING + 1, 8192), dtype=torch.int32)
    cm = CacheManager(num_pages=8192, page_size=1, page_table=pt, type="swa_radix", swa_pool=pool, sliding_window_size=1024)
    pm = PrefillManager(cm, TableManager(max_running_reqs=MAX_RUNNING, page_table=pt), DecodeManager(1), keep_images_whole=True)
    image = MMItem(modality="image", hash=1, pad_value=1, offsets=[[4900, 5156]], feature=torch.zeros(1))
    pm.pending_list = [PendingReq(uid=UID, input_ids=torch.arange(6000, dtype=torch.int32),
                                  sampling_params=SamplingParams(max_tokens=1), mm_items=[image])]
    ends = []
    while pm.runnable:
        batch = pm.schedule_next_batch(8192)
        assert cut_image_spans(batch.reqs) == []
        ends.append(batch.reqs[0].device_len)
        cm.free_swa_out_of_window_extend(batch.reqs)
        cm.allocate_paged(batch.reqs)
        for r in batch.reqs:
            r.complete_one()
    assert ends[0] == 4900 and ends[-1] == 6000
