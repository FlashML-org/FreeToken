"""``ft bench bw --batch``: the routing draw, the CLI parser, and the dedup A/B harness.

The headline bench runs at bs=1, where cross-token expert dedup is inert by construction,
so the batch sweep is what makes the change measurable. These cover the parts that decide
whether the two arms are comparable at all -- identical routing, restored environment --
plus one end-to-end sweep behind a CUDA guard.
"""

from __future__ import annotations

import argparse
import os

import pytest
import torch

from freetoken.moe.benchbw import (
    DTYPE_WORKLOADS,
    _batch_list,
    _batch_routing,
    _forced_dedup,
)


def test_routing_shape_and_dtype():
    steps = _batch_routing(bs=4, top_k=8, E=128, steps=3)
    assert len(steps) == 3
    for t in steps:
        assert t.shape == (4, 8)
        assert t.dtype == torch.int32
        assert int(t.min()) >= 0 and int(t.max()) < 128


def test_routes_are_distinct_within_a_token():
    # A router never sends one token to the same expert twice; if the draw allowed it,
    # "unique experts" would undercount and the reported reuse factor would be inflated.
    for t in _batch_routing(bs=8, top_k=6, E=64, steps=4):
        for row in t:
            assert len(set(row.tolist())) == 6


def test_routing_is_replayable():
    # Both arms of the A/B must see identical work, otherwise the comparison is between
    # two different random routings rather than between two kernels.
    a = _batch_routing(bs=4, top_k=8, E=128, steps=3)
    b = _batch_routing(bs=4, top_k=8, E=128, steps=3)
    for x, y in zip(a, b):
        assert torch.equal(x, y)
    assert not torch.equal(a[0], _batch_routing(bs=4, top_k=8, E=128, steps=3, seed=99)[0])


def test_routing_falls_back_when_experts_are_scarce():
    # Fewer experts than routes -> distinct-per-token is impossible; draw with replacement
    # rather than raising, so a tiny synthetic bank still benches.
    for t in _batch_routing(bs=2, top_k=8, E=4, steps=2):
        assert t.shape == (2, 8)
        assert int(t.max()) < 4


@pytest.mark.parametrize("s,want", [
    ("1", (1,)),
    ("1,8,32", (1, 8, 32)),
    (" 4 , 4 , 16 ", (4, 16)),   # de-duplicated, order preserved
])
def test_batch_list_accepts(s, want):
    assert _batch_list(s) == want


@pytest.mark.parametrize("s", ["", "0", "-4", "8,0", "abc", "1.5"])
def test_batch_list_rejects(s):
    with pytest.raises(argparse.ArgumentTypeError):
        _batch_list(s)


def test_forced_dedup_sets_and_restores():
    key = "FREETOKEN_CPU_MOE_DEDUP"
    prev = os.environ.get(key)
    try:
        os.environ.pop(key, None)
        with _forced_dedup(False):
            assert os.environ[key] == "0"
        assert key not in os.environ          # unset stays unset

        os.environ[key] = "sentinel"
        with _forced_dedup(True):
            assert os.environ[key] == "1"
        assert os.environ[key] == "sentinel"  # a pre-existing value is put back
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def test_forced_dedup_restores_on_exception():
    key = "FREETOKEN_CPU_MOE_DEDUP"
    prev = os.environ.get(key)
    try:
        os.environ.pop(key, None)
        with pytest.raises(ValueError):
            with _forced_dedup(True):
                raise ValueError("boom")
        assert key not in os.environ
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (banks are pinned)")
def test_batch_sweep_end_to_end():
    from freetoken.moe.benchbw import measure_cpu_moe_batch

    rows = measure_cpu_moe_batch("bf16", DTYPE_WORKLOADS["bf16"], (1, 4), iters=2)
    assert [r["batch"] for r in rows] == [1, 4]
    for r in rows:
        assert r["routes"] == r["batch"] * DTYPE_WORKLOADS["bf16"].top_k
        assert 0 < r["unique"] <= r["routes"]
        assert r["reuse"] == pytest.approx(r["routes"] / r["unique"], rel=1e-2)
        assert set(r["ms"]) == {"off", "on"} and all(v > 0 for v in r["ms"].values())
        assert all(v > 0 for v in r["eff_gbs"].values())
    # bs=1 cannot collide with itself, so there is nothing to dedup.
    assert rows[0]["unique"] == rows[0]["routes"]
    assert rows[0]["reuse"] == 1.0
