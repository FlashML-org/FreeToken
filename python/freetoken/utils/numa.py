"""NUMA topology, and how tensor-parallel ranks should be spread across it.

Why an engine cares. A rank's host-side working set -- for an offload MoE, tens of GiB
of expert banks -- is anonymous memory, so its pages land on whichever node FIRST
TOUCHES them. If the loader's threads are unbound they scatter across every socket, and
the CPU expert pool, pinned to one socket's cores, then reads much of its data across
the interconnect. On a two-socket Xeon that is the difference between one memory
controller and two: each socket has its own channels, so binding ranks to nodes turns
one socket's bandwidth into the machine's.

Everything here is read from sysfs and intersected with the process's affinity mask, so
it is correct on a single-socket laptop (no-op), a 2-socket server, a 4- or 8-node
system, and inside a container whose cpuset covers part of a node.
"""

from __future__ import annotations

import glob
import os


def _parse_cpulist(spec: str) -> set[int]:
    """``"0-3,8,12-13"`` -> ``{0,1,2,3,8,12,13}``."""
    cpus: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def allowed_cpus() -> set[int]:
    """CPUs this process may run on. Falls back to every CPU where unsupported."""
    try:
        return set(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return set(range(os.cpu_count() or 1))


def numa_nodes(allowed: set[int] | None = None) -> list[list[int]]:
    """CPUs per NUMA node, restricted to what this process may use.

    Nodes with no usable CPU are dropped -- a memory-only node (CXL, or a node whose
    cores a cpuset excludes) is not somewhere a rank can run, and treating it as one
    would strand a rank with an empty affinity mask.
    """
    if allowed is None:
        allowed = allowed_cpus()
    nodes: list[list[int]] = []
    for path in sorted(
        glob.glob("/sys/devices/system/node/node[0-9]*/cpulist"),
        key=lambda p: int(p.rsplit("node", 1)[1].split("/", 1)[0]),
    ):
        try:
            with open(path) as f:
                cpus = _parse_cpulist(f.read())
        except OSError:
            return []
        usable = sorted(cpus & allowed)
        if usable:
            nodes.append(usable)
    return nodes


def device_numa_node(device_index: int) -> int | None:
    """The NUMA node a CUDA device hangs off, or None if it cannot be determined.

    A GPU sits behind a PCIe root complex owned by ONE socket. Host-to-device traffic
    from the other socket crosses the interconnect -- which, for an offload MoE, is the
    per-step expert fetch. So a rank's CPU should live on its GPU's node, and that node
    is a property of the hardware, not of the rank's index.
    """
    try:
        import torch

        props = torch.cuda.get_device_properties(device_index)
        bdf = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    except Exception:
        return None
    try:
        with open(f"/sys/bus/pci/devices/{bdf}/numa_node") as f:
            node = int(f.read().strip())
    except (OSError, ValueError):
        return None
    return node if node >= 0 else None


def rank_placement(rank: int, world_size: int) -> tuple[int, list[int], int, int] | None:
    """Where rank ``rank`` of ``world_size`` should run.

    Returns ``(node_index, cpus, siblings, index_among_siblings)``, or None when there
    is nothing to decide -- one node, one rank, or no readable topology.

    Ranks are dealt to nodes in contiguous blocks, so consecutive ranks share a node and
    a node's ranks are adjacent. ``siblings`` is the EXACT number of ranks placed on this
    rank's node, not an average: with 3 ranks over 2 nodes one node holds two and the
    other one, and a rank that assumed the average would either oversubscribe its cores
    or leave half of them idle.

    With more nodes than ranks each rank still gets a whole node, and the surplus nodes
    go unused -- spreading one rank's threads over two nodes would recreate the remote
    access this exists to avoid.
    """
    nodes = numa_nodes()
    if len(nodes) < 2 or world_size < 2:
        return None
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} outside world size {world_size}")

    n_nodes = len(nodes)

    # Prefer the node this rank's GPU actually hangs off. Deriving it from the rank
    # index instead only works when CUDA enumerates devices in the same order as the
    # sockets own them -- true on some machines, and silently wrong on others, where
    # every rank would then reach its own card across the interconnect.
    gpu_node = device_numa_node(rank)
    if gpu_node is not None and gpu_node < n_nodes:
        mates = [r for r in range(world_size) if device_numa_node(r) == gpu_node]
        if mates:
            return gpu_node, nodes[gpu_node], len(mates), mates.index(rank)
    if world_size <= n_nodes:
        # Spread out: every rank owns a node, evenly spaced so a partly used machine
        # still uses distinct memory controllers.
        node = (rank * n_nodes) // world_size
        return node, nodes[node], 1, 0

    # More ranks than nodes: contiguous blocks, remainder spread over the first nodes.
    base, extra = divmod(world_size, n_nodes)
    node, first = 0, 0
    for i in range(n_nodes):
        count = base + (1 if i < extra else 0)
        if rank < first + count:
            node = i
            return node, nodes[i], count, rank - first
        first += count
    raise AssertionError("unreachable: ranks did not cover the node blocks")


_PLACEMENT: tuple[int, list[int], int, int] | None = None


def resolve_placement(rank: int, world_size: int) -> tuple[int, list[int], int, int] | None:
    """``rank_placement``, computed once and remembered for the rest of the process.

    Binding a rank ERASES the evidence the placement was derived from: afterwards the
    affinity mask is one node, so ``numa_nodes`` sees a single node and
    ``rank_placement`` correctly answers "nothing to spread". Every later consumer --
    the torch thread split, the CPU MoE core split -- would then fall back to dividing
    by the whole world size and hand each rank a fraction of a fraction.

    So the answer is computed BEFORE the bind and reused. Call this once, early.
    """
    global _PLACEMENT
    if _PLACEMENT is None:
        _PLACEMENT = rank_placement(rank, world_size)
    return _PLACEMENT


def placement() -> tuple[int, list[int], int, int] | None:
    """The remembered placement, or None if this process was never placed."""
    return _PLACEMENT


def reset_placement() -> None:
    """Forget the remembered placement (tests)."""
    global _PLACEMENT
    _PLACEMENT = None


__all__ = [
    "allowed_cpus",
    "numa_nodes",
    "placement",
    "rank_placement",
    "reset_placement",
    "resolve_placement",
]
