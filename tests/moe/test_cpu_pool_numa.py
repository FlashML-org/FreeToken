"""CPU MoE worker pool placement: confine to one NUMA node, and leave single-node alone.

The expert GEMV reads every expert byte from host DRAM exactly once, so a pool spanning
sockets runs half its workers against remote memory and ping-pongs the spin-barrier and
work counters over the interconnect. `cpu_moe_ext.cpp` says as much ("NUMA: a single node
is assumed") but nothing enforced it: `resolve_threads_and_affinity(0)` handed back every
physical core on the machine. Measured on 2x Xeon Gold 6526Y, bf16 expert GEMV: 67 GB/s
spanning both sockets (26% of the STREAM ceiling) vs 124 GB/s confined to one (91%).

The risk in that change is the *other* direction -- FreeToken's usual target is a
single-socket desktop, where the resolver must keep returning exactly what it always did.
`test_single_node_*` are the regression guards for that.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from freetoken.moe import cpu_executor as ce
from freetoken.utils import numa

# 2 nodes x 4 physical cores x SMT2, laid out like a real dual-socket box: node0 owns
# 0-3 (siblings 8-11), node1 owns 4-7 (siblings 12-15).
TWO_NODE = {c: (0 if c % 8 < 4 else 1) for c in range(16)}
SIBLINGS = {c: f"{c % 8},{c % 8 + 8}" for c in range(16)}
ALL_CPUS = list(range(16))
NODE0_CORES = [0, 1, 2, 3]
NODE1_CORES = [4, 5, 6, 7]


@pytest.fixture
def two_nodes():
    with (
        patch.object(numa, "_allowed_cpus", return_value=ALL_CPUS),
        patch.object(numa, "cpu_numa_node", side_effect=TWO_NODE.get),
        patch.object(numa, "thread_siblings", side_effect=SIBLINGS.get),
    ):
        yield


@pytest.fixture
def one_node():
    with (
        patch.object(numa, "_allowed_cpus", return_value=ALL_CPUS),
        patch.object(numa, "cpu_numa_node", return_value=0),
        patch.object(numa, "thread_siblings", side_effect=SIBLINGS.get),
    ):
        yield


# --------------------------------------------------------------- regression guards

def test_single_node_is_not_confined(one_node, monkeypatch):
    """One node means every core is already local: nothing to choose, nothing to change."""
    monkeypatch.delenv("FREETOKEN_CPU_MOE_NUMA", raising=False)
    assert numa.moe_pool_numa_node(None) is None


def test_single_node_pool_is_unchanged(one_node):
    """The pre-change resolver returned one thread per physical core, machine-wide."""
    assert ce.resolve_threads_and_affinity(0, None) == (8, [0, 1, 2, 3, 4, 5, 6, 7])
    assert ce.physical_core_cpus(None) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_no_node_argument_matches_old_behaviour(two_nodes):
    """Callers that pass no node keep spanning the machine, dual-socket included."""
    assert ce.resolve_threads_and_affinity(0) == (8, [0, 1, 2, 3, 4, 5, 6, 7])
    assert ce.resolve_threads_and_affinity(6)[1] == [0, 1, 2, 3, 4, 5]


def test_unreadable_topology_is_not_confined(monkeypatch):
    """No sysfs node info -> None, so odd platforms fall back to today's behaviour."""
    monkeypatch.delenv("FREETOKEN_CPU_MOE_NUMA", raising=False)
    with (
        patch.object(numa, "_allowed_cpus", return_value=ALL_CPUS),
        patch.object(numa, "cpu_numa_node", return_value=None),
    ):
        assert numa.moe_pool_numa_node(None) is None


# ------------------------------------------------------------------- confinement

def test_pool_follows_the_gpu_node(two_nodes, monkeypatch):
    """The GPU's node keeps CPU compute *and* the offload gather's DMA local."""
    monkeypatch.delenv("FREETOKEN_CPU_MOE_NUMA", raising=False)
    with patch.object(numa, "gpu_numa_node", return_value=1):
        assert numa.moe_pool_numa_node(0) == 1


def test_unknown_gpu_node_falls_back_to_a_real_node(two_nodes, monkeypatch):
    monkeypatch.delenv("FREETOKEN_CPU_MOE_NUMA", raising=False)
    with patch.object(numa, "gpu_numa_node", return_value=None):
        assert numa.moe_pool_numa_node(None) == 0


def test_confined_default_keeps_the_thread_count_and_only_moves_it(two_nodes):
    """Confining to half the machine must not halve the pool.

    The node's 4 cores plus their 4 SMT siblings = 8 threads, exactly what the
    unconfined pool used to run machine-wide. Physical cores come first so the
    bandwidth-bound formats still get one thread per core before any core doubles up.
    mxfp4 regressed 23% when this returned physical cores only.
    """
    n, cores = ce.resolve_threads_and_affinity(0, 0)
    assert (n, cores) == (8, [0, 1, 2, 3, 8, 9, 10, 11])
    assert {TWO_NODE[c] for c in cores} == {0}

    n1, cores1 = ce.resolve_threads_and_affinity(0, 1)
    assert (n1, cores1) == (8, [4, 5, 6, 7, 12, 13, 14, 15])
    assert {TWO_NODE[c] for c in cores1} == {1}


def test_confined_default_matches_the_old_machine_wide_count(two_nodes):
    """Same number of threads as before, different placement -- the whole point."""
    old_count, _ = ce.resolve_threads_and_affinity(0, None)
    new_count, _ = ce.resolve_threads_and_affinity(0, 0)
    assert new_count == old_count


def test_confined_physical_cores_helper_still_reports_cores_only(two_nodes):
    """physical_core_cpus stays "one per physical core" -- the torch clamp reads it."""
    assert ce.physical_core_cpus(0) == NODE0_CORES
    assert ce.physical_core_cpus(1) == NODE1_CORES


def test_explicit_count_fills_local_smt_before_crossing(two_nodes):
    """4 cores + 4 siblings on node0 are all used before any node1 core is touched."""
    _, cores = ce.resolve_threads_and_affinity(8, 0)
    assert cores == [0, 1, 2, 3, 8, 9, 10, 11]
    assert {TWO_NODE[c] for c in cores} == {0}


def test_oversubscription_spills_to_the_other_node_last(two_nodes):
    _, cores = ce.resolve_threads_and_affinity(10, 0)
    assert cores[:8] == [0, 1, 2, 3, 8, 9, 10, 11]
    assert {TWO_NODE[c] for c in cores[8:]} == {1}


# ----------------------------------------------------------------- escape hatches

@pytest.mark.parametrize("value", ["off", "none", "OFF", " off "])
def test_env_off_disables_confinement(two_nodes, monkeypatch, value):
    monkeypatch.setenv("FREETOKEN_CPU_MOE_NUMA", value)
    assert numa.moe_pool_numa_node(0) is None


@pytest.mark.parametrize("node", [0, 1])
def test_env_pins_an_explicit_node(two_nodes, monkeypatch, node):
    """An integer is always a node id -- "0" pins node 0, it does not mean "off"."""
    monkeypatch.setenv("FREETOKEN_CPU_MOE_NUMA", str(node))
    with patch.object(numa, "gpu_numa_node", return_value=1 - node):
        assert numa.moe_pool_numa_node(0) == node


def test_env_garbage_does_not_confine(two_nodes, monkeypatch):
    """A typo must not silently pin the pool somewhere arbitrary."""
    monkeypatch.setenv("FREETOKEN_CPU_MOE_NUMA", "nodezero")
    assert numa.moe_pool_numa_node(0) is None
    monkeypatch.setenv("FREETOKEN_CPU_MOE_NUMA", "7")
    assert numa.moe_pool_numa_node(0) is None


# ------------------------------------------------------- bank memory placement

def test_prefer_node_is_a_noop_without_a_node():
    """No NUMA choice to make -> no syscall, so single-socket boxes are untouched."""
    assert numa.prefer_node(0x1000, 4096, None) is False


def test_prefer_node_places_a_real_mapping():
    """mbind must actually take on an untouched anonymous mapping.

    Worker confinement alone made throughput bimodal (56-123 GB/s run to run) because
    the banks landed wherever the loader threads ran. This is the other half.
    """
    import ctypes
    import mmap as _mmap

    nodes = sorted(numa.numa_nodes())
    if len(nodes) < 2 or numa._mbind() is None:
        pytest.skip("needs a multi-node Linux box with mbind")
    buf = _mmap.mmap(-1, 4 << 20)
    try:
        addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        assert numa.prefer_node(addr, len(buf), nodes[-1]) is True
    finally:
        buf.close()
