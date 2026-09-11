"""Ordered MoE route trace: capture + offline LRU replay (PLAN_TP_EP.md 6A).

Why this exists
---------------
``--moe-collect-decode-freq`` only yields a per-(layer, expert) HISTOGRAM. A
histogram cannot reproduce LRU behaviour, adjacent-step overlap, or a miss
forecast, because it has thrown away the ORDER of expert activations. The EP2
capacity decision (does doubling unique slots actually cut miss, and by how much
on the slow rank) needs the ordered sequence.

This module records the RAW global expert ids in call order -- captured at the
same point ``collect_decode_freq`` snapshots them, i.e. BEFORE ``lru_ensure``
rewrites ``expert_ids`` into slot ids in place -- and replays them offline
through an LRU that mirrors ``flashlib.kernels.slot_cache.lru_ensure`` semantics
(batch hit-protection, ``(usage, slot)`` victim order, ascending-id miss/victim
pairing, dedup). See ``test_route_trace.py`` for the semantics lock.

Capture is host-side (a ``.cpu()`` per call), so it is NOT CUDA-graph safe: the
engine refuses ``--moe-trace-route`` unless ``--cuda-graph-max-bs 0``. That is the
intended use -- a correctness/sampling run on an experiment instance, never the
production perf path (plan 6A/8).

On-disk format (append-only binary, crash-safe; partial data is always usable):
    meta : ``<path>.meta.json``  (num_experts, num_layers, cache_size, top_k, ...)
    body : ``<path>``            repeated records, each
           ``struct '<bii' (phase, layer_id, n)`` then ``n`` little-endian int32 ids
``phase``: 0 = decode, 1 = prefill.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass

_RECORD_HDR = struct.Struct("<bii")  # phase(int8), layer_id(int32), n(int32)


@dataclass
class RouteTraceMeta:
    num_experts: int
    num_layers: int
    cache_size: int
    top_k: int
    model: str = ""
    records: int = 0
    overflow: bool = False
    decode_target: str = ""


class RouteTraceRecorder:
    """Append raw routed expert ids per ``ensure_experts`` call, in order.

    ``record`` is called from ``OffloadMoeCache.ensure_experts`` /
    ``ensure_experts_hybrid`` at the raw-ids point. Writes straight to disk so a
    killed experiment still leaves a usable trace. ``max_records`` (0 = unlimited)
    bounds the file; on overflow it stops recording and flags the meta.
    """

    def __init__(
        self,
        path: str,
        *,
        num_experts: int,
        num_layers: int,
        cache_size: int,
        top_k: int,
        model: str = "",
        decode_target: str = "",
        max_records: int = 0,
    ) -> None:
        self.path = path
        self.meta_path = path + ".meta.json"
        self.max_records = max_records
        self._meta = RouteTraceMeta(
            num_experts=num_experts, num_layers=num_layers, cache_size=cache_size,
            top_k=top_k, model=model, decode_target=decode_target,
        )
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        # truncate any previous run so the body and meta stay consistent
        self._f = open(path, "wb")
        self._buf = bytearray()
        self._n = 0
        self._overflow = False
        self._flush_every = 8192  # records per buffer flush (bounds memory + I/O size)
        self._write_meta()

    def _write_meta(self) -> None:
        self._meta.records = self._n
        self._meta.overflow = self._overflow
        with open(self.meta_path, "w") as f:
            json.dump(self._meta.__dict__, f, indent=1)

    def _flush(self) -> None:
        if self._buf:
            self._f.write(self._buf)
            self._buf.clear()
        self._f.flush()

    def record(self, layer_id: int, expert_ids, phase: int = 0) -> None:
        """``expert_ids``: the RAW global ids tensor for this call (pre-rewrite)."""
        if self._overflow:
            return
        if self.max_records and self._n >= self.max_records:
            self._overflow = True
            self._flush()
            self._write_meta()
            return
        # int32 little-endian, host sync (NOT graph-safe; engine gates on graphs off)
        import torch

        ids = expert_ids.reshape(-1).to(dtype=torch.int32).cpu().numpy()
        self._buf += _RECORD_HDR.pack(phase, int(layer_id), int(ids.size))
        self._buf += ids.tobytes()
        self._n += 1
        if self._n % self._flush_every == 0:  # periodic flush so a crash keeps most data
            self._flush()
            self._write_meta()

    def close(self) -> None:
        try:
            self._flush()
            self._f.close()
        finally:
            self._write_meta()

    def __del__(self):  # best-effort; append-binary means data is already on disk
        try:
            self.close()
        except Exception:
            pass


def read_trace(path: str):
    """Yield ``(phase, layer_id, ids_tuple)`` per record; return ``(meta, records)``.

    ``records`` is a list so the caller can replay repeatedly (multiple capacities).
    """
    with open(path + ".meta.json") as f:
        meta = RouteTraceMeta(**json.load(f))
    records = []
    with open(path, "rb") as f:
        while True:
            hdr = f.read(_RECORD_HDR.size)
            if len(hdr) < _RECORD_HDR.size:
                break
            phase, layer_id, n = _RECORD_HDR.unpack(hdr)
            raw = f.read(4 * n)
            if len(raw) < 4 * n:
                break  # truncated tail from a crash -- keep what parsed
            ids = tuple(struct.unpack(f"<{n}i", raw)) if n else ()
            records.append((phase, layer_id, ids))
    return meta, records


# ----------------------------------------------------------------- LRU mirror
class LRU:
    """Unified-pool LRU mirroring ``flashlib.lru_ensure`` (see test_route_trace).

    One ``ensure`` == one kernel call == one LRU step:
      * dedup ids; every HIT bumps ``usage[slot] = step`` first (batch protection:
        a slot touched this call is never this call's victim).
      * victims = ascending ``(usage, slot)``; the i-th ascending miss id takes the
        i-th coldest victim.
    """

    def __init__(self, cache_size: int, num_experts: int):
        import heapq

        self.C = cache_size
        self.E = num_experts
        self.slot_of: dict[int, int] = {}
        self.owner: list[int | None] = [None] * cache_size
        self.usage: list[int] = [0] * cache_size
        self.heap: list[tuple[int, int]] = []
        self.free = list(range(cache_size))
        self.step = 0
        self.miss = 0
        self.active = 0
        self._h = heapq

    def _evict_one(self) -> int:
        if self.free:
            return self.free.pop()
        while True:
            u, s = self._h.heappop(self.heap)
            if self.owner[s] is not None and self.usage[s] == u:
                del self.slot_of[self.owner[s]]
                self.owner[s] = None
                return s

    def ensure(self, layer: int, ids) -> None:
        self.step += 1
        base = layer * self.E
        uniq = sorted(set(ids))
        self.active += len(uniq)
        misses = []
        for e in uniq:
            fid = base + e
            s = self.slot_of.get(fid)
            if s is None:
                misses.append(e)
            else:
                self.usage[s] = self.step
                self._h.heappush(self.heap, (self.step, s))
        self.miss += len(misses)
        for e in misses:
            s = self._evict_one()
            fid = base + e
            self.slot_of[fid] = s
            self.owner[s] = fid
            self.usage[s] = self.step
            self._h.heappush(self.heap, (self.step, s))


def replay(records, cache_size: int, num_experts: int, *, phase: int = 0,
           ep_owner: tuple[int, int] | None = None):
    """Replay ``records`` through one LRU pool.

    ``ep_owner=(lo, hi)``: keep only ids in ``[lo, hi)`` (remapped to ``id-lo``) --
    models one EP rank's local cache. ``None`` replays all ids (today's TP1).
    Returns ``(miss, active, miss_rate)``.
    """
    lru = LRU(cache_size, num_experts if ep_owner is None else (ep_owner[1] - ep_owner[0]))
    for ph, layer, ids in records:
        if ph != phase:
            continue
        if ep_owner is not None:
            lo, hi = ep_owner
            ids = tuple(e - lo for e in ids if lo <= e < hi)
        lru.ensure(layer, ids)  # empty local set still advances the LRU clock
    rate = lru.miss / lru.active if lru.active else 0.0
    return lru.miss, lru.active, rate


def replay_ep2(records, cache_sizes: tuple[int, int], num_experts: int, *, phase: int = 0):
    """Replay both EP ranks; report per-rank miss and the per-step SLOW side.

    The decode step waits on max(rank0, rank1) per layer, so the slow-side miss
    (not the average) is what sets the step time. Returns a dict.
    """
    half = num_experts // 2
    ranks = [LRU(cache_sizes[r], half) for r in range(2)]
    slow_miss = slow_active = 0
    for ph, layer, ids in records:
        if ph != phase:
            continue
        m0 = [r.miss for r in ranks]
        a0 = [r.active for r in ranks]
        for r in range(2):
            lo = r * half
            loc = tuple(e - lo for e in ids if lo <= e < lo + half)
            ranks[r].ensure(layer, loc)
        dm = [ranks[i].miss - m0[i] for i in range(2)]
        da = [ranks[i].active - a0[i] for i in range(2)]
        slow_miss += max(dm)
        slow_active += max(da)
    return {
        "rank_miss": [r.miss for r in ranks],
        "rank_active": [r.active for r in ranks],
        "rank_rate": [r.miss / r.active if r.active else 0.0 for r in ranks],
        "slow_miss": slow_miss,
        "slow_active": slow_active,
        "slow_rate": slow_miss / slow_active if slow_active else 0.0,
    }
