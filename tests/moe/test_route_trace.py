"""Tests for moe/route_trace.py: capture round-trip, LRU-mirror semantics, EP2 slow-side.

Pure CPU, no torch/CUDA needed for the replay/LRU parts (the recorder's ``record``
takes a tensor, so it uses a tiny stub). Mirrors PLAN_TP_EP.md 6A's requirement that
the replay match flashlib lru_ensure semantics before it is trusted for the EP2 call.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Load route_trace.py directly by path: it is stdlib-only, so importing it without
# the freetoken.moe package __init__ (which pulls torch/transformers) keeps these
# tests runnable on a bare interpreter.
_MOD = Path("/home/zhanglei/code/freetoken/FreeToken/python/freetoken/moe/route_trace.py")
_spec = importlib.util.spec_from_file_location("route_trace", _MOD)
_rt = importlib.util.module_from_spec(_spec)
sys.modules["route_trace"] = _rt  # @dataclass resolves cls.__module__ via sys.modules
_spec.loader.exec_module(_rt)
LRU = _rt.LRU
RouteTraceRecorder = _rt.RouteTraceRecorder
read_trace = _rt.read_trace
replay = _rt.replay
replay_ep2 = _rt.replay_ep2


class _FakeTensor:
    """Minimal stand-in for the raw expert-ids tensor ``record`` receives."""

    def __init__(self, ids):
        self._ids = list(ids)

    def reshape(self, _):
        return self

    def tolist(self):
        return self._ids


def test_roundtrip(tmp_path):
    import struct

    path = str(tmp_path / "tr.bin")
    rec = RouteTraceRecorder(
        path, num_experts=512, num_layers=48, cache_size=2840, top_k=10,
        model="flash", decode_target="gpu",
    )
    # emulate record() without torch: write the same binary layout by hand
    def put(phase, layer, ids):
        rec._buf += struct.Struct("<bii").pack(phase, layer, len(ids))
        rec._buf += struct.pack(f"<{len(ids)}i", *ids)
        rec._n += 1

    put(0, 0, [5, 300, 7])
    put(0, 1, [10, 11])
    put(1, 0, [1, 2, 3])  # prefill record
    rec.close()

    meta, records = read_trace(path)
    assert meta.num_experts == 512 and meta.cache_size == 2840
    assert meta.records == 3
    assert records[0] == (0, 0, (5, 300, 7))
    assert records[1] == (0, 1, (10, 11))
    assert records[2] == (1, 0, (1, 2, 3))


def test_lru_batch_protection():
    # C=2: [1,2] then [0,2] -> expert 2 is a protected hit, only 0 misses (v1 bug = 2)
    c = LRU(2, 512)
    c.ensure(0, [1, 2])
    before = c.miss
    c.ensure(0, [0, 2])
    assert c.miss - before == 1, f"expected 1 miss, got {c.miss - before}"


def test_lru_stack_property():
    # same access sequence, larger cache never increases miss (standard LRU property,
    # the theoretical basis for the capacity-expansion main line)
    seq = [(0, [i % 40 for i in range(s, s + 10)]) for s in range(0, 300, 3)]
    counts = []
    for C in (8, 16, 32, 64):
        c = LRU(C, 512)
        for layer, ids in seq:
            c.ensure(layer, ids)
        counts.append(c.miss)
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)), counts


def test_lru_dedup():
    c = LRU(8, 512)
    c.ensure(0, [3, 3, 3, 7])
    assert c.miss == 2 and c.active == 2


def test_replay_tp1_vs_ep2(tmp_path):
    import struct

    path = str(tmp_path / "r.bin")
    rec = RouteTraceRecorder(
        path, num_experts=512, num_layers=2, cache_size=100, top_k=8, model="t",
        decode_target="gpu",
    )

    def put(layer, ids):
        rec._buf += struct.Struct("<bii").pack(0, layer, len(ids))
        rec._buf += struct.pack(f"<{len(ids)}i", *ids)
        rec._n += 1

    # Genuine cache pressure: each step touches 8 distinct experts drawn uniformly
    # from a 120-wide window that straddles BOTH halves (0-59 in rank0's [0,256),
    # 256-315 in rank1's). Working set (120) >> pool (40), so TP1 thrashes; under
    # EP2 each rank only caches its own 60, and a 40-slot pool covers 40/60 of it
    # vs 40/120 for TP1 -> strictly fewer misses on the slow rank.
    import random
    rng = random.Random(0)
    window = list(range(60)) + list(range(256, 316))
    for _ in range(400):
        put(0, rng.sample(window, 8))
        put(1, rng.sample(window, 8))
    rec.close()

    meta, records = read_trace(path)
    _, _, tp1 = replay(records, 40, 512)
    ep = replay_ep2(records, (40, 40), 512)
    assert ep["slow_rate"] < tp1, f"EP2 slow {ep['slow_rate']} should beat TP1 {tp1}"
    # symmetric window -> roughly balanced load per rank (random draw, not exact)
    a0, a1 = ep["rank_active"]
    assert abs(a0 - a1) <= 0.1 * max(a0, a1), (a0, a1)
    # slow-side miss >= each rank's miss (it is the per-step max), so it is the limiter
    assert ep["slow_miss"] >= max(ep["rank_miss"])
    assert tp1 > 0.3, f"expected real cache pressure, TP1 miss={tp1}"


def test_overflow_flag(tmp_path):
    import struct

    path = str(tmp_path / "o.bin")
    rec = RouteTraceRecorder(
        path, num_experts=512, num_layers=1, cache_size=10, top_k=2, model="t",
        decode_target="gpu", max_records=3,
    )

    def put(ids):
        if rec._overflow:
            return
        if rec._n >= rec.max_records:
            rec._overflow = True
            return
        rec._buf += struct.Struct("<bii").pack(0, 0, len(ids))
        rec._buf += struct.pack(f"<{len(ids)}i", *ids)
        rec._n += 1

    for _ in range(10):
        put([1, 2])
    rec.close()
    meta, records = read_trace(path)
    assert meta.overflow is True
    assert len(records) == 3
