"""SWA pool allocation/store/translation integration test.

Exercises the scheduler-equivalent allocation (allocate full token slots, then
``HybridSWAKVCache.alloc_swa`` records the full->swa mapping), writes K/V through
``store_kv`` and verifies live-window slot contents and eviction sentinels. This
does not call MiMo attention or an attention backend, so it is deliberately not
a cached-decode parity test; that production-path regression lives separately.

Numeric guard for the store_kv + alloc_swa + translate + window slice stack: if
alloc_swa's global-paged slots or translate were wrong, the gathered window K/V
would not reproduce the reference window and the assert would fire (the old
harness silently used an arange() page table that bypassed alloc_swa).

Runs on GPU when present (store_kv drives the Triton/CUDA store kernel); skips
on CPU with a clear reason.
"""
from __future__ import annotations

import pytest
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SKIP = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="store_kv drives the Triton/CUDA store kernel; SWA decode numeric test needs CUDA",
)

WINDOW = 32
N_PREFILL = 40
N_DECODE = 6
FULL_HEADS, SWA_HEADS, HEAD_DIM = 2, 4, 96
NUM_FULL_PAGES = 1024
NUM_SWA_TOKENS = 512


def _patch_tp(monkeypatch) -> None:
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.hybrid_swa_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def _specs():
    from freetoken.models.config import KVCacheGroupSpec

    # layer 0 -> swa (window 32), layer 1 -> full; mirrors the tiny MiMo layer split.
    return (
        KVCacheGroupSpec(name="full", layer_ids=(1,), num_kv_heads=FULL_HEADS, head_dim=HEAD_DIM, sliding_window=None),
        KVCacheGroupSpec(name="swa", layer_ids=(0,), num_kv_heads=SWA_HEADS, head_dim=HEAD_DIM, sliding_window=WINDOW),
    )


def _build_pool():
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache

    return HybridSWAKVCache(
        groups=_specs(),
        num_layers=2,
        num_full_pages=NUM_FULL_PAGES,
        page_size=1,
        dtype=torch.bfloat16,
        device=torch.device(DEVICE),
        num_swa_tokens=NUM_SWA_TOKENS,
    )


class _SchedEquivAlloc:
    """Mirrors CacheManager.allocate_paged for page_size==1: pop full token slots
    off the free list, then HybridSWAKVCache.alloc_swa those full slots, and hand
    back the full slots (== the page_table row store_kv reads)."""

    def __init__(self, pool):
        self.pool = pool
        self.free = list(range(pool.full_num_tokens))  # full slot ids (page_size==1)
        self.full_of_token: list[int] = []             # token index -> its full slot id

    def alloc(self, n: int) -> torch.Tensor:
        assert n <= len(self.free), "full pool exhausted"
        full = torch.tensor(self.free[:n], dtype=torch.int32, device=torch.device(DEVICE))
        self.free = self.free[n:]
        self.pool.alloc_swa(full)
        self.full_of_token.extend(int(x) for x in full.tolist())
        return full


def _swa_window_gather(pool, layer_id, full_slots):
    """Gather the SWA pool KV for the given FULL slots exactly as the decode gather
    does: translate full->swa then index the swa cache. Returned flattened to
    [W, heads*dim] to line up with store_kv's 2D K/V contract."""
    swa = pool.translate_loc_from_full_to_swa(full_slots)
    k = pool.k_cache(layer_id)[swa.to(torch.long)]  # [W, 1, heads, dim]
    v = pool.v_cache(layer_id)[swa.to(torch.long)]
    return k.squeeze(1).reshape(-1, SWA_HEADS * HEAD_DIM), v.squeeze(1).reshape(-1, SWA_HEADS * HEAD_DIM)


def _sdpa_reference(q, k, v, sm_scale):
    """GQA-scaled single-query attention over already-windowed, flattened K/V
    ([W, heads*dim]); q is [num_q_heads, dim]."""
    k = k.reshape(k.shape[0], SWA_HEADS, HEAD_DIM)
    v = v.reshape(v.shape[0], SWA_HEADS, HEAD_DIM)
    group = q.shape[0] // k.shape[1]
    kf = k.float().repeat_interleave(group, dim=1)   # [W, num_q_heads, dim]
    vf = v.float().repeat_interleave(group, dim=1)
    qf = q.float()                                   # [num_q_heads, dim]
    scores = (kf * qf.unsqueeze(0)).sum(-1) * sm_scale  # [W, num_q_heads]
    probs = torch.softmax(scores, dim=0)             # softmax over the window keys
    return (probs.unsqueeze(-1) * vf).sum(0)         # [num_q_heads, dim]


@SKIP
def test_swa_pool_window_slots_match_stored_kv(monkeypatch):
    _patch_tp(monkeypatch)
    torch.manual_seed(0)
    dev = torch.device(DEVICE)
    pool = _build_pool()
    alloc = _SchedEquivAlloc(pool)
    swa_layer = 0
    sm_scale = HEAD_DIM ** -0.5

    # deterministic per-token K/V (SWA layer) we store (2D [token, heads*dim] rows,
    # store_kv's contract) and later re-read.
    total = N_PREFILL + N_DECODE
    k_store = torch.randn(total, SWA_HEADS * HEAD_DIM, device=dev, dtype=torch.bfloat16)
    v_store = torch.randn(total, SWA_HEADS * HEAD_DIM, device=dev, dtype=torch.bfloat16)
    q_dec = torch.randn(total, SWA_HEADS * 2, HEAD_DIM, device=dev, dtype=torch.bfloat16)

    # --- prefill N=40: alloc_swa the full slots, store every token via store_kv ---
    pre_full = alloc.alloc(N_PREFILL)
    pool.store_kv(k_store[:N_PREFILL], v_store[:N_PREFILL], pre_full, swa_layer)

    # in-window tokens (first 32) must translate to live (>0) swa slots; store then
    # translate-gather == what we stored.
    win = torch.tensor(alloc.full_of_token[:WINDOW], dtype=torch.int32, device=dev)
    assert torch.all(pool.translate_loc_from_full_to_swa(win) > 0)
    gk, gv = _swa_window_gather(pool, swa_layer, win)
    torch.testing.assert_close(gk.float(), k_store[:WINDOW].float(), atol=0, rtol=0)
    torch.testing.assert_close(gv.float(), v_store[:WINDOW].float(), atol=0, rtol=0)

    # --- decode steps: alloc_swa a slot, store, free out-of-window, check window ---
    for step in range(N_DECODE):
        tok = N_PREFILL + step                 # sequence position of the new token
        pos_full = alloc.alloc(1)
        pool.store_kv(k_store[tok : tok + 1], v_store[tok : tok + 1], pos_full, swa_layer)

        # proactive out-of-window free (== maybe_free_swa_out_of_window): keep the
        # trailing [tok-WINDOW+1 .. tok] window live, free the strictly-older token.
        out_of_window = tok - WINDOW
        if out_of_window >= 0:
            pool.free_swa(torch.tensor([alloc.full_of_token[out_of_window]], dtype=torch.int32, device=dev))

        lo = max(0, tok - WINDOW + 1)          # window slice [i-31, i] (inclusive)
        win_full = torch.tensor(alloc.full_of_token[lo : tok + 1], dtype=torch.int32, device=dev)
        swa_idx = pool.translate_loc_from_full_to_swa(win_full)
        assert torch.all(swa_idx > 0), "in-window full slot must map to a live (>0) swa slot"
        assert swa_idx.numel() == min(tok + 1, WINDOW), "window must hold exactly min(i+1, W) live slots"

        gk, gv = _swa_window_gather(pool, swa_layer, win_full)
        torch.testing.assert_close(gk.float(), k_store[lo : tok + 1].float(), atol=0, rtol=0)
        torch.testing.assert_close(gv.float(), v_store[lo : tok + 1].float(), atol=0, rtol=0)

        # This only sanity-checks the helper on values already proven equal above;
        # it does not execute or validate a production attention backend.
        got = _sdpa_reference(q_dec[tok], gk, gv, sm_scale)
        ref = _sdpa_reference(q_dec[tok], k_store[lo : tok + 1], v_store[lo : tok + 1], sm_scale)
        torch.testing.assert_close(got.float(), ref.float(), atol=1e-4, rtol=1e-4)

    # out-of-window token's swa slot was freed -> maps to the 0 sentinel; only the
    # LIVE window may gather (silent-slot-0 bug guard mirror).
    last = N_PREFILL + N_DECODE - 1
    assert int(pool.translate_loc_from_full_to_swa(
        torch.tensor([alloc.full_of_token[last - WINDOW]], dtype=torch.int32, device=dev))[0]) == 0


@SKIP
def test_alloc_swa_is_scheduler_equivalent_slot_for_slot(monkeypatch):
    """alloc_swa assigns distinct live swa slots and translate reads them back 1:1,
    matching the scheduler invariant that every allocated full token owns a swa slot."""
    _patch_tp(monkeypatch)
    pool = _build_pool()
    before = pool.swa_available_size()
    full = torch.arange(N_PREFILL, dtype=torch.int32, device=torch.device(DEVICE))
    pool.alloc_swa(full)
    swa = pool.translate_loc_from_full_to_swa(full)
    assert pool.swa_available_size() == before - N_PREFILL
    assert torch.all(swa >= 1) and len(set(swa.tolist())) == N_PREFILL
