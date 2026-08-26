"""Spreading tensor-parallel ranks over NUMA nodes.

A rank's expert banks are anonymous memory: their pages land on whichever node first
TOUCHES them. Get the placement wrong and nothing fails -- the banks simply sit on the
far side of the interconnect from the cores that read them, and decode is slower for
reasons no log mentions.

So the placement is checked against topologies this host does not have: one node, four,
eight, uneven rank counts, and a cpuset that hides part of a node.
"""

from __future__ import annotations

import pytest

from freetoken.utils import numa


@pytest.fixture
def fake_topology(monkeypatch):
    """Install a synthetic NUMA layout: a list of CPU lists, one per node."""

    def install(nodes: list[list[int]], allowed: set[int] | None = None):
        mask = allowed if allowed is not None else {c for n in nodes for c in n}
        monkeypatch.setattr(numa, "allowed_cpus", lambda: set(mask))
        monkeypatch.setattr(
            numa, "numa_nodes",
            lambda a=None: [sorted(set(n) & set(mask)) for n in nodes if set(n) & set(mask)],
        )
        # Synthetic topology means synthetic GPUs too. Without this the host's REAL
        # GPU affinity leaks in and silently overrides the layout under test -- which
        # is how these tests first broke when GPU-aware placement landed.
        monkeypatch.setattr(numa, "device_numa_node", lambda i: None)

    return install


class TestNoPlacementNeeded:
    def test_single_node_is_a_no_op(self, fake_topology):
        fake_topology([list(range(16))])
        assert numa.rank_placement(0, 4) is None, "one node: nothing to spread"

    def test_single_rank_is_a_no_op(self, fake_topology):
        fake_topology([[0, 1], [2, 3]])
        assert numa.rank_placement(0, 1) is None


class TestOneRankPerNode:
    def test_four_ranks_over_four_nodes_get_one_each(self, fake_topology):
        fake_topology([[0, 1], [2, 3], [4, 5], [6, 7]])
        seen = [numa.rank_placement(r, 4) for r in range(4)]
        assert [p[0] for p in seen] == [0, 1, 2, 3]
        assert all(p[2] == 1 for p in seen), "each rank owns its node alone"

    def test_two_ranks_over_four_nodes_spread_out(self, fake_topology):
        # Half a machine should still use distinct memory controllers, not crowd node 0.
        fake_topology([[0], [1], [2], [3]])
        assert numa.rank_placement(0, 2)[0] == 0
        assert numa.rank_placement(1, 2)[0] == 2

    def test_a_rank_never_straddles_two_nodes(self, fake_topology):
        fake_topology([[0, 1], [2, 3], [4, 5], [6, 7]])
        for r in range(3):
            _node, cpus, _s, _i = numa.rank_placement(r, 3)
            assert cpus in ([0, 1], [2, 3], [4, 5], [6, 7]), (
                "splitting one rank over nodes recreates the remote access this avoids"
            )


class TestMoreRanksThanNodes:
    def test_four_ranks_over_two_nodes_pair_up(self, fake_topology):
        fake_topology([list(range(20)), list(range(20, 40))])
        placements = [numa.rank_placement(r, 4) for r in range(4)]
        assert [p[0] for p in placements] == [0, 0, 1, 1], "contiguous blocks"
        assert [p[2] for p in placements] == [2, 2, 2, 2], "two ranks per node"
        assert [p[3] for p in placements] == [0, 1, 0, 1], "index within the node"

    def test_an_uneven_split_reports_the_exact_count_per_node(self, fake_topology):
        # 3 ranks over 2 nodes: one node holds two, the other one. A rank that assumed
        # the average would either oversubscribe its cores or leave half of them idle.
        fake_topology([[0, 1, 2, 3], [4, 5, 6, 7]])
        p0, p1, p2 = (numa.rank_placement(r, 3) for r in range(3))
        assert (p0[0], p0[2], p0[3]) == (0, 2, 0)
        assert (p1[0], p1[2], p1[3]) == (0, 2, 1)
        assert (p2[0], p2[2], p2[3]) == (1, 1, 0), "the lone rank has its node to itself"

    def test_eight_ranks_over_two_nodes(self, fake_topology):
        fake_topology([list(range(20)), list(range(20, 40))])
        assert [numa.rank_placement(r, 8)[0] for r in range(8)] == [0, 0, 0, 0, 1, 1, 1, 1]

    def test_every_rank_is_placed_exactly_once(self, fake_topology):
        fake_topology([[0, 1], [2, 3], [4, 5]])
        for world in (2, 3, 4, 5, 6, 7):
            per_node: dict[int, list[int]] = {}
            for r in range(world):
                node, _cpus, siblings, index = numa.rank_placement(r, world)
                per_node.setdefault(node, []).append(index)
                assert 0 <= index < siblings
            for node, indices in per_node.items():
                assert sorted(indices) == list(range(len(indices))), (
                    f"world={world} node={node}: indices must be 0..n-1 with no gaps"
                )


class TestRestrictedCpusets:
    def test_a_node_with_no_usable_cpu_is_dropped(self, fake_topology):
        # A memory-only node, or one whose cores a container excludes, is not somewhere
        # a rank can run -- placing one there would leave it with an empty mask.
        fake_topology([[0, 1], [2, 3]], allowed={0, 1})
        assert numa.rank_placement(0, 2) is None, "only one usable node remains"

    def test_placement_uses_only_permitted_cpus(self, fake_topology):
        fake_topology([[0, 1, 2, 3], [4, 5, 6, 7]], allowed={0, 1, 4, 5})
        for r in range(2):
            _n, cpus, _s, _i = numa.rank_placement(r, 2)
            assert set(cpus) <= {0, 1, 4, 5}


class TestParsing:
    @pytest.mark.parametrize(
        "spec,expected",
        [
            ("0-3", {0, 1, 2, 3}),
            ("0,2,4", {0, 2, 4}),
            ("0-1,8-9", {0, 1, 8, 9}),
            ("5", {5}),
            ("", set()),
        ],
    )
    def test_cpulist_forms(self, spec, expected):
        assert numa._parse_cpulist(spec) == expected


def test_a_rank_outside_the_world_is_refused(fake_topology):
    fake_topology([[0], [1]])
    with pytest.raises(ValueError, match="outside world size"):
        numa.rank_placement(5, 4)


class TestPlacementSurvivesBinding:
    """Binding erases the evidence the placement was derived from.

    Once a rank's affinity is one node, the topology LOOKS single-node, so
    rank_placement correctly answers "nothing to spread" -- and every later consumer
    (the torch thread split, the CPU MoE core split) falls back to dividing by the whole
    world size. Each rank then gets a fraction of a fraction and most of the socket
    sits idle, with nothing in the logs but two lines that disagree:

        rank 0 -> NUMA node 0 (40 cpus, 2 rank(s) on this node)
        torch intra-op threads: 10 per rank (shared with 3 sibling rank(s))
    """

    def test_the_answer_is_remembered_across_the_bind(self, fake_topology):
        numa.reset_placement()
        fake_topology([list(range(20)), list(range(20, 40))])
        before = numa.resolve_placement(0, 4)
        assert before[2] == 2, "two ranks share node 0"

        # Simulate the bind: affinity is now one node, so the topology looks single.
        fake_topology([list(range(20)), list(range(20, 40))], allowed=set(range(20)))
        assert numa.rank_placement(0, 4) is None, "a fresh derivation loses it"
        assert numa.placement() == before, "the remembered answer must survive"
        numa.reset_placement()

    def test_resolve_is_idempotent(self, fake_topology):
        numa.reset_placement()
        fake_topology([[0, 1], [2, 3]])
        first = numa.resolve_placement(1, 4)
        second = numa.resolve_placement(1, 4)
        assert first is second
        numa.reset_placement()

    def test_an_unplaced_process_reports_none(self):
        numa.reset_placement()
        assert numa.placement() is None


class TestGpuAffinityDrivesPlacement:
    """A rank should follow its GPU, not its index.

    Each GPU hangs off a PCIe root complex owned by one socket, and for an offload MoE
    the per-step expert fetch is host-to-device traffic. Placing a rank by index happens
    to be right when CUDA enumerates devices in socket order -- and is silently wrong
    otherwise, sending every fetch across the interconnect with nothing in the logs.
    """

    def test_placement_follows_the_gpu_not_the_rank_index(self, fake_topology, monkeypatch):
        numa.reset_placement()
        fake_topology([list(range(20)), list(range(20, 40))])
        # A machine that enumerates GPUs across sockets: 0,2 -> node 0; 1,3 -> node 1.
        # Index-derived placement would put ranks 0,1 on node 0 and 2,3 on node 1, so
        # ranks 1 and 2 would each be on the wrong socket for their card.
        monkeypatch.setattr(numa, "device_numa_node", lambda i: {0: 0, 1: 1, 2: 0, 3: 1}[i])
        assert numa.rank_placement(0, 4)[0] == 0
        assert numa.rank_placement(1, 4)[0] == 1
        assert numa.rank_placement(2, 4)[0] == 0
        assert numa.rank_placement(3, 4)[0] == 1

    def test_ranks_on_a_node_are_indexed_without_gaps(self, fake_topology, monkeypatch):
        fake_topology([list(range(20)), list(range(20, 40))])
        monkeypatch.setattr(numa, "device_numa_node", lambda i: {0: 0, 1: 1, 2: 0, 3: 1}[i])
        # Ranks 0 and 2 share node 0; they must take index 0 and 1 there, so the CPU
        # MoE pool hands them different cores.
        assert numa.rank_placement(0, 4)[2:] == (2, 0)
        assert numa.rank_placement(2, 4)[2:] == (2, 1)

    def test_it_falls_back_to_the_index_when_affinity_is_unknown(self, fake_topology, monkeypatch):
        # No sysfs, a virtualized GPU, or a driver that does not report it.
        fake_topology([list(range(20)), list(range(20, 40))])
        monkeypatch.setattr(numa, "device_numa_node", lambda i: None)
        assert [numa.rank_placement(r, 4)[0] for r in range(4)] == [0, 0, 1, 1]
