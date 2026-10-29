"""The scheduler side of a speculative verify step: what the drain commits, what it gives back, and in which order.

The real unbound Scheduler methods run against CPU-built hybrid managers; the engine is a recorder, so the target state commit is checked by its arguments and its order against the finish frees.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import PrefillManager
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.table import TableManager

K = 3
PROMPT = 12


def _setup():
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    pool = LinearStatePool(group=group, num_slots=16, dtype=torch.bfloat16, device=torch.device("cpu"), tp_size=1)
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(128, 1, page_table, "hybrid_radix", linear_state_pool=pool)
    tm = TableManager(max_running_reqs=4, page_table=page_table)
    dm = DecodeManager(page_size=1, token_headroom=K)
    pm = PrefillManager(cm, tm, dm, token_headroom=K)

    events = []
    engine = SimpleNamespace(
        commit_verify=lambda reqs, counts, out: events.append(("commit", [r.uid for r in reqs], list(counts))),
    )

    stub = SimpleNamespace(
        cache_manager=cm, table_manager=tm, decode_manager=dm, prefill_manager=pm, spec=None, engine=engine,
        config=SimpleNamespace(decode_rows_per_req=K + 1), device=torch.device("cpu"),
        token_pool=torch.zeros(4, 64, dtype=torch.int32), eos_token_ids=set(), toolcall_anchor_id=None,
        _match_stop_str=lambda _req: None,
    )

    def free(req):
        events.append(("free", req.uid))
        Scheduler._free_req_resources(stub, req)

    stub._free_req_resources = free
    stub._commit_tokens = lambda req, tokens: Scheduler._commit_tokens(stub, req, tokens)
    return pool, cm, tm, dm, stub, events


def _decoding_req(pool, cm, tm, uid: int, max_tokens: int) -> Req:
    """A request whose prefill has drained: KV for the prompt, its first output token pending at PROMPT."""
    prompt = torch.arange(1, PROMPT + 1, dtype=torch.int32) + 100 * uid
    match = cm.match_req(SimpleNamespace(input_ids=prompt, input_len=PROMPT))
    req = Req(input_ids=prompt, table_idx=tm.allocate(), cached_len=0, output_len=max_tokens, uid=uid,
              sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True), cache_handle=match.cuda_handle)
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    req.mamba_next_track_idx = 1
    cm.lock(match.cuda_handle)
    cm.allocate_paged([req])
    req.complete_one()
    req.append_host(torch.tensor([7], dtype=torch.int32))
    return req


def test_drain_commits_the_accepted_tokens_and_returns_the_rejected_slots():
    pool, cm, tm, dm, stub, events = _setup()
    # one accepted draft; the output limit cut inside the step; every draft accepted
    reqs = [_decoding_req(pool, cm, tm, 1, 16), _decoding_req(pool, cm, tm, 2, 3), _decoding_req(pool, cm, tm, 3, 16)]
    dm.filter_reqs(reqs)
    # the verify launch: k provisional slots past each pending token
    for req in reqs:
        req.widen_pending(K)
    cm.allocate_paged(reqs)
    free_before = cm.available_size
    out = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]])
    output = SimpleNamespace(next_tokens_gpu=out, next_tokens_cpu=out, emitted_cpu=torch.tensor([1, 3, 3]))
    batch = Batch(reqs=reqs, phase="verify", rows_per_req=K + 1)
    reply, finished = [], set()
    tables = [r.table_idx for r in reqs]

    committed = Scheduler._drain_verify(stub, batch, output, reply, finished)

    assert committed == 2 + 2 + 4
    assert [m.next_tokens for m in reply] == [[11, 12], [21, 22], [31, 32, 33, 34]]
    assert [m.finished for m in reply] == [False, True, False]
    # the target state is committed once, before the finished request gives its GDN slot back
    assert events == [("commit", [1, 2, 3], [2, 2, 4]), ("free", 2)]
    assert finished == {reqs[1]}
    a, c = reqs[0], reqs[2]
    assert (a.cached_len, a.device_len) == (PROMPT + 2, PROMPT + 3)
    assert (c.cached_len, c.device_len) == (PROMPT + 4, PROMPT + 5)
    # the last committed token is pending at cached_len for the next step
    assert int(stub.token_pool[tables[0], PROMPT + 2]) == 12
    assert int(stub.token_pool[tables[2], PROMPT + 4]) == 34
    # slots from cached_len on go back (allocate_paged allocates from cached_len again next step); b gave everything back
    assert cm.available_size - free_before == (K + 1 - 2) + (K + 1 - 4) + (PROMPT + K + 1)
    assert dm.running_reqs == {a, c}
    # no slot was freed twice: with every request gone the pool adds up
    for req in (a, c):
        stub._free_req_resources(req)
    cm.check_integrity()
