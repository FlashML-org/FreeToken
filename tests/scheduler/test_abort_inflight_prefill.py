"""Abort vs overlap-scheduling races.

Under overlap scheduling a batch launches one iteration before _process_last_data drains
it, and an abort message is processed in between. Freeing the request's resources inside
the abort handler while its forward is in flight used to corrupt state: the hybrid
prefill-commit dereferenced the None'd GDN ping-pong slots (TypeError killed the
scheduler); plain radix would silently re-read the freed page-table row.

The scheduler now uses the SGLang-style single-owner design: the abort handler frees
immediately ONLY when the request has no forward in flight (not in ``self._last_data``'s
batch); otherwise it just sets ``req.aborted`` and _process_last_data frees the request
when the batch drains, after copy_done.synchronize(). A ``table_idx != -1`` sentinel on
the prefix-commit remains as defense-in-depth.

Tests drive the real (unbound) Scheduler methods against CPU-built hybrid managers.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.message import AbortBackendMsg
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.table import TableManager
from freetoken.scheduler.utils import PendingReq

UID = 2


def _pool(num_slots=16):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)


def _setup():
    """Hybrid managers + a stub Scheduler `self` for the real unbound methods."""
    pool = _pool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    tm = TableManager(max_running_reqs=4, page_table=pt)
    dm = DecodeManager(page_size=1)
    pm = PrefillManager(cm, tm, dm)
    sent = []
    stub = SimpleNamespace(
        cache_manager=cm,
        table_manager=tm,
        decode_manager=dm,
        prefill_manager=pm,
        finished_reqs=set(),
        eos_token_ids=set(),
        toolcall_anchor_id=None,
        config=SimpleNamespace(page_size=1),
        status_reporter=SimpleNamespace(report_batch=lambda *_, **__: None),
        send_result=sent.extend,
        _kv_usage_pages=cm.page_usage,
        _mamba_slot_usage=lambda: None,
        _swa_token_usage=lambda: None,
        _gpu_mem_bytes=lambda: 0,
        _match_stop_str=lambda _req: None,
        _pending_abort_acks=set(),
        _last_data=None,
    )
    stub._free_req_resources = lambda req: Scheduler._free_req_resources(stub, req)
    return pool, cm, tm, dm, pm, sent, stub


def _launch_req(pool, cm, tm, prompt, *, cls=Req, track_seqlen=None):
    """A launched (forward in flight) hybrid req: handle locked, pages allocated,
    GDN slots held, cached_len advanced -- the state _process_last_data will drain."""
    mr = cm.match_req(SimpleNamespace(input_ids=prompt, input_len=len(prompt),
                                      mm_embeds=None))
    req = cls(input_ids=prompt, table_idx=tm.allocate(), cached_len=0, output_len=4,
              uid=UID, sampling_params=SamplingParams(max_tokens=4),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    req.mamba_next_track_idx = 1
    cm.lock(mr.cuda_handle)
    cm.allocate_paged([req])
    req.complete_one()
    req.mamba_last_track_seqlen = track_seqlen
    return req


def _as_last_data(batch):
    return (
        SimpleNamespace(batch=batch),
        (None, torch.tensor([42], dtype=torch.int32),
         SimpleNamespace(synchronize=lambda: None)),
    )


def test_abort_inflight_final_chunk_marks_then_drains():
    """Abort while the final prefill chunk (plain Req, already in running_reqs) is in
    flight: the handler only marks; the same iteration's drain frees exactly once."""
    pool, cm, tm, dm, _pm, sent, stub = _setup()
    req = _launch_req(pool, cm, tm, torch.arange(1, 13, dtype=torch.int32),
                      track_seqlen=8)
    batch = Batch(reqs=[req], phase="prefill")
    dm.filter_reqs(batch.reqs)                  # _forward: joins running_reqs at launch
    stub._last_data = _as_last_data(batch)      # overlap_loop exposes the un-drained batch

    Scheduler._process_one_msg(stub, AbortBackendMsg(uid=UID))
    assert req.aborted and req.table_idx != -1  # marked, NOT freed under the forward
    assert req.mamba_ping_pong is not None
    assert req not in dm.running_reqs
    assert UID in stub._pending_abort_acks
    free_after_mark = pool.num_free_slots

    Scheduler._process_last_data(stub, stub._last_data)
    assert req.table_idx == -1                  # freed at the drain point
    assert pool.num_free_slots > free_after_mark
    assert req in stub.finished_reqs
    assert sent == []                           # no DetokenizeMsg: abort ack stays terminal
    cm.check_integrity()


def test_abort_inflight_intermediate_chunk_marks_then_drains():
    """Abort mid-chunked-prefill: prefill_manager.abort_req pops the pending continuation
    (no next chunk launches) and returns the in-flight ChunkedReq, which is marked and
    freed when its batch drains."""
    pool, cm, tm, _dm, pm, sent, stub = _setup()
    prompt = torch.arange(1, 13, dtype=torch.int32)
    chunk = _launch_req(pool, cm, tm, prompt[:8], cls=ChunkedReq)
    pending = PendingReq(uid=UID, input_ids=prompt,
                         sampling_params=SamplingParams(max_tokens=4))
    pending.chunked_req = chunk
    pm.pending_list = [pending]
    batch = Batch(reqs=[chunk], phase="prefill")
    stub._last_data = _as_last_data(batch)

    Scheduler._process_one_msg(stub, AbortBackendMsg(uid=UID))
    assert pm.pending_list == []                # continuation gone: no next chunk
    assert chunk.aborted and chunk.table_idx != -1

    Scheduler._process_last_data(stub, stub._last_data)
    assert chunk.table_idx == -1
    assert sent == []                           # chunks never reply
    cm.check_integrity()


def test_abort_starved_decode_req_frees_immediately():
    """A request with no forward in flight (e.g. a decode req starved behind a long
    chunked prefill) is freed by the abort handler right away -- deferring would leak
    until its next batch, which strict prefill-priority puts arbitrarily far away."""
    pool, cm, tm, dm, _pm, _sent, stub = _setup()
    req = _launch_req(pool, cm, tm, torch.arange(1, 13, dtype=torch.int32))
    dm.filter_reqs([req])
    # the un-drained batch belongs to some other request's prefill
    stub._last_data = (SimpleNamespace(batch=SimpleNamespace(reqs=[])), None)
    base_free = pool.num_free_slots

    Scheduler._process_one_msg(stub, AbortBackendMsg(uid=UID))
    assert not req.aborted
    assert req.table_idx == -1                  # freed immediately, no drain needed
    assert pool.num_free_slots > base_free
    assert req not in dm.running_reqs
    cm.check_integrity()


def test_prefix_commit_sentinel_guard():
    """Defense-in-depth: even if some future path frees a req early (bypassing the
    aborted mark), the finished=False prefix-commit must skip a freed req instead of
    dereferencing its None'd GDN slots (the original crash, cache.py _cache_req_hybrid)."""
    pool, cm, tm, dm, _pm, sent, stub = _setup()
    req = _launch_req(pool, cm, tm, torch.arange(1, 13, dtype=torch.int32),
                      track_seqlen=8)
    batch = Batch(reqs=[req], phase="prefill")
    dm.filter_reqs(batch.reqs)

    aborted = dm.abort_req(UID)
    assert aborted is req
    Scheduler._free_req_resources(stub, aborted)   # freed WITHOUT the aborted mark
    assert req.table_idx == -1 and req.mamba_ping_pong is None
    free_after_abort = pool.num_free_slots

    Scheduler._process_last_data(stub, _as_last_data(batch))  # pre-guard: TypeError

    assert pool.num_free_slots == free_after_abort  # nothing double-freed
    cm.check_integrity()
    assert [m.uid for m in sent] == [UID]  # un-marked path still publishes the token


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")


def test_post_terminal_overlap_step_is_dropped():
    """Overlap scheduling launches one more decode step for a request that already
    terminated (filter_reqs keeps it while output budget remains). The extra drain
    must not append its token, emit a second DetokenizeMsg, or free twice."""
    from freetoken.message import DetokenizeMsg

    pool, cm, tm, dm, _pm, sent, stub = _setup()
    stub.eos_token_ids = {42}  # the drained token (42) finishes the request by EOS
    req = _launch_req(pool, cm, tm, torch.arange(1, 13, dtype=torch.int32),
                      track_seqlen=8)
    dm.filter_reqs([req])

    Scheduler._process_last_data(stub, _as_last_data(Batch(reqs=[req], phase="prefill")))
    assert req in stub.finished_reqs and req.table_idx == -1
    terminal = [m for m in sent if isinstance(m, DetokenizeMsg)]
    assert len(terminal) == 1 and terminal[0].finished

    # The overlap extra step: the same req sits in the next batch's drain.
    output_len_before = req.output_len
    Scheduler._process_last_data(stub, _as_last_data(Batch(reqs=[req], phase="decode")))
    assert [m for m in sent if isinstance(m, DetokenizeMsg)] == terminal  # no 2nd msg
    assert req.output_len == output_len_before                           # no append
    cm.check_integrity()
