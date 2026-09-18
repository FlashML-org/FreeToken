"""TP core-partitioning tests for the CPU MoE executor's worker pool.

Regression test for the TP>1 pinning collision: every rank process auto-pinned
the whole machine, so the rank pools time-sliced on the same cores (~4-14x
decode loss measured on 2-GPU boxes). Under TP each rank must get a disjoint
slice of the physical cores. Pure Python -- no CUDA / compiled extension needed.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def fake_cores(monkeypatch):
    """16 physical cores, no SMT."""
    import freetoken.moe.cpu_executor as ce

    monkeypatch.setattr(ce, "physical_core_cpus", lambda: list(range(16)))
    return ce


def _disjoint_partition(ce, world_size, requested=0):
    slices = [
        ce.resolve_threads_and_affinity(requested, rank=r, world_size=world_size)[1]
        for r in range(world_size)
    ]
    for i, a in enumerate(slices):
        for b in slices[i + 1 :]:
            assert not set(a) & set(b), f"rank {i} and {b.index(b[0]) if b else '?'} overlap: {a} vs {b}"
    return slices


def test_tp2_auto_slices_are_disjoint_partition(fake_cores):
    n0, c0 = fake_cores.resolve_threads_and_affinity(0, rank=0, world_size=2)
    n1, c1 = fake_cores.resolve_threads_and_affinity(0, rank=1, world_size=2)
    assert (n0, n1) == (8, 8)
    assert not set(c0) & set(c1)
    assert sorted(c0 + c1) == list(range(16))


def test_tp4_auto_slices_are_disjoint_partition(fake_cores):
    slices = _disjoint_partition(fake_cores, world_size=4)
    assert sorted(sum(slices, [])) == list(range(16))


def test_explicit_count_stays_in_rank_slice(fake_cores):
    n, cores = fake_cores.resolve_threads_and_affinity(8, rank=1, world_size=2)
    assert n == 8
    assert set(cores) <= set(range(1, 16, 2))


def test_single_rank_unchanged(fake_cores):
    n, cores = fake_cores.resolve_threads_and_affinity(0)
    assert (n, cores) == (16, list(range(16)))
    # explicit count behaves as before (spread across all cores)
    n, cores = fake_cores.resolve_threads_and_affinity(4)
    assert n == 4 and len(set(cores)) == 4


def test_rank_wraps_around(fake_cores):
    # rank >= world_size must not crash or escape the partition scheme
    _, c = fake_cores.resolve_threads_and_affinity(0, rank=2, world_size=2)
    assert c == list(range(0, 16, 2))
