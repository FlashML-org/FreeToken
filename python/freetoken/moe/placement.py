"""Where each rank's routed-expert shard lives: its GPU slice, the host bank it fetches from, the CPU pool.

Under expert tensor parallelism every rank computes one slice of each routed expert's
intermediate dim on its GPU. The host side is pooled one of two ways:

* ``rank``: each rank keeps a host bank of exactly its GPU slice, bound to its GPU's NUMA node,
  and runs its own CPU pool over it on that node's cores.
* ``numa``: the ranks whose GPUs share a NUMA node share one host bank of the union of their
  slices, bound to that node and loaded once. One CPU pool on that node computes the bank's
  whole slice for CPU-assigned routes; each GPU fetches its narrow slice out of the shared bank.

Correctness only needs the GPU slices to cover the intermediate dim once and the CPU pools to
cover it once: GPU-assigned routes are summed over the GPU slices, CPU-assigned ones over the
pools, and one all-reduce adds every partial. Locality needs each GPU slice inside the bank it
fetches from, which the planner gets by numbering the slices node by node.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from freetoken.layers.quantization.moe.shard import Interval

PoolMode = Literal["rank", "numa"]


@dataclass(frozen=True)
class HostBankSpec:
    """One host expert bank: the slice of every expert it stores and the ranks that fetch from it.

    The first rank loads it; ``node`` is the NUMA node its pages are bound to (None: unbound)."""

    node: int | None
    interval: Interval
    ranks: tuple[int, ...]

    @property
    def owner(self) -> int:
        return self.ranks[0]


@dataclass(frozen=True)
class CpuPoolSpec:
    """One CPU expert pool: it runs in ``owner``'s process over that rank's bank, on ``cores``,
    and computes ``interval`` of every CPU-assigned route; the owner adds the partial once."""

    node: int | None
    interval: Interval
    owner: int
    cores: tuple[int, ...]


@dataclass(frozen=True)
class ExpertPlacement:
    intermediate: int
    mode: PoolMode
    gpu: tuple[Interval, ...]
    banks: tuple[HostBankSpec, ...]
    pools: tuple[CpuPoolSpec, ...] = ()

    @property
    def world_size(self) -> int:
        return len(self.gpu)

    def bank_of(self, rank: int) -> HostBankSpec:
        return next(bank for bank in self.banks if rank in bank.ranks)

    def pool_owned_by(self, rank: int) -> CpuPoolSpec | None:
        return next((pool for pool in self.pools if pool.owner == rank), None)

    def validate(self, alignment: int = 1) -> None:
        """Raise unless the slices sum every route exactly once and each GPU slice is fetchable from its bank."""
        _check_cover("GPU slices", self.gpu, self.intermediate, alignment)
        holders = sorted(r for bank in self.banks for r in bank.ranks)
        if holders != list(range(self.world_size)):
            raise ValueError(f"every rank needs exactly one host bank, the banks list ranks {holders}")
        for bank in self.banks:
            for rank in bank.ranks:
                if not bank.interval.contains(self.gpu[rank]):
                    raise ValueError(f"rank {rank}'s GPU slice {_fmt(self.gpu[rank])} is outside its bank {_fmt(bank.interval)}")
        if not self.pools:
            return
        _check_cover("CPU pool slices", tuple(p.interval for p in self.pools), self.intermediate, alignment)
        for pool in self.pools:
            bank = self.bank_of(pool.owner)
            if bank.owner != pool.owner or bank.interval != pool.interval:
                raise ValueError(f"the CPU pool of rank {pool.owner} must compute exactly the bank that rank loads")
        cores = [c for pool in self.pools for c in pool.cores]
        if len(cores) != len(set(cores)) or any(not pool.cores for pool in self.pools):
            raise ValueError("CPU pools need disjoint, nonempty core sets")

    def describe(self) -> str:
        banks = "; ".join(f"node {b.node}: {_fmt(b.interval)} <- ranks {list(b.ranks)}" for b in self.banks)
        pools = "; ".join(f"rank {p.owner}: {_fmt(p.interval)} on {len(p.cores)} cores" for p in self.pools) or "none"
        return f"expert placement ({self.mode}): GPU slices {[_fmt(g) for g in self.gpu]}; host banks {banks}; CPU pools {pools}"


def plan_expert_placement(
    intermediate: int,
    rank_nodes: Sequence[int | None],
    *,
    mode: PoolMode,
    alignment: int = 1,
    node_cores: Mapping[int | None, Sequence[int]] | None = None,
    threads: int = 0,
    reserve: int = 1,
) -> ExpertPlacement:
    """Place ``len(rank_nodes)`` equal GPU slices, their host banks and (with ``node_cores``) CPU pools.

    ``rank_nodes[r]`` is rank r's GPU NUMA node, None when unknown. ``node_cores`` maps each node to
    its usable physical cores; ``reserve`` of them stay free per rank on the node for its framework
    threads, and ``threads`` > 0 caps each pool's cores. Without ``node_cores`` nothing decodes on the CPU.
    A GPU of unknown locality on a host with several NUMA nodes has no node to bind its bank or pool
    to, so then the whole host is one pool: unbound banks, CPU pools cut from all its cores.
    """
    if mode not in ("rank", "numa"):
        raise ValueError(f"unknown expert pool mode {mode!r}; expected 'rank' or 'numa'")
    if node_cores is not None and None in rank_nodes and set(node_cores) != {None}:
        rank_nodes = [None] * len(rank_nodes)
        node_cores = {None: sorted(c for cores in node_cores.values() for c in cores)}
    world = len(rank_nodes)
    nodes = list(dict.fromkeys(rank_nodes))
    members = {node: [r for r in range(world) if rank_nodes[r] == node] for node in nodes}
    order = [r for node in nodes for r in members[node]]
    slices = Interval.split(intermediate, world, align=alignment)
    gpu: list[Interval] = [Interval(0, 0)] * world
    for position, rank in enumerate(order):
        gpu[rank] = slices[position]

    if mode == "rank":
        banks = tuple(HostBankSpec(rank_nodes[r], gpu[r], (r,)) for r in range(world))
    else:
        banks = tuple(
            HostBankSpec(node, Interval(gpu[members[node][0]].start, gpu[members[node][-1]].stop), tuple(members[node]))
            for node in nodes
        )
    pools: tuple[CpuPoolSpec, ...] = ()
    if node_cores is not None:
        pools = tuple(
            CpuPoolSpec(bank.node, bank.interval, bank.owner, cores)
            for bank, cores in zip(banks, _pool_cores(banks, members, node_cores, threads, reserve))
        )
    placement = ExpertPlacement(intermediate, mode, tuple(gpu), banks, pools)
    placement.validate(alignment)
    return placement


def _pool_cores(banks, members, node_cores, threads: int, reserve: int) -> list[tuple[int, ...]]:
    """Each bank's pool cores: its node's cores past the per-rank reserve, split evenly among the node's pools."""
    by_node: dict = {}
    for i, bank in enumerate(banks):
        by_node.setdefault(bank.node, []).append(i)
    out: list[tuple[int, ...]] = [()] * len(banks)
    for node, ids in by_node.items():
        cores = list(node_cores.get(node, ()))
        free = cores[: max(len(cores) - reserve * len(members[node]), 0)]
        share = len(free) // len(ids)
        if share < 1:
            raise ValueError(
                f"NUMA node {node} has {len(cores)} usable cores for {len(ids)} CPU pools and {len(members[node])} ranks; "
                "too few to run the CPU experts there"
            )
        for k, i in enumerate(ids):
            mine = free[k * share : (k + 1) * share]
            out[i] = tuple(mine[:threads] if threads > 0 else mine)
    return out


def _check_cover(what: str, intervals: Sequence[Interval], total: int, alignment: int) -> None:
    edge = 0
    for iv in sorted(intervals):
        if iv.start != edge or iv.width == 0:
            raise ValueError(f"{what} {[_fmt(i) for i in intervals]} do not cover [0, {total}) exactly once")
        if any(e % alignment for e in (iv.start, iv.stop) if e not in (0, total)):
            raise ValueError(f"{what} {[_fmt(i) for i in intervals]} split a {alignment}-wide quantization group")
        edge = iv.stop
    if edge != total:
        raise ValueError(f"{what} {[_fmt(i) for i in intervals]} do not cover [0, {total}) exactly once")


def _fmt(iv: Interval) -> str:
    return f"[{iv.start}, {iv.stop})"


# --- host topology --------------------------------------------------------------------------------


def gpu_numa_node(device_index: int) -> int | None:
    """The NUMA node of a CUDA device's PCI slot; None when the host has one node or does not say."""
    import torch

    props = torch.cuda.get_device_properties(device_index)
    path = f"/sys/bus/pci/devices/{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0/numa_node"
    try:
        with open(path) as f:
            node = int(f.read().strip())
    except (OSError, ValueError):
        node = -1
    nodes = numa_nodes()
    if node < 0 or len(nodes) <= 1:
        return nodes[0] if len(nodes) == 1 else None
    return node


def numa_nodes() -> list[int]:
    try:
        return sorted(int(n[4:]) for n in os.listdir("/sys/devices/system/node") if n.startswith("node") and n[4:].isdigit())
    except OSError:
        return []


def node_memory_bytes(node: int) -> int | None:
    """NUMA ``node``'s total RAM; None when the host does not say."""
    try:
        with open(f"/sys/devices/system/node/node{node}/meminfo") as f:
            for line in f:  # "Node 0 MemTotal:       131504972 kB"
                if "MemTotal:" in line:
                    return int(line.split()[-2]) * 1024
    except (OSError, ValueError):
        pass
    return None


def node_physical_cores() -> dict[int | None, list[int]]:
    """One logical CPU per physical core in this process's affinity, by NUMA node (key None without NUMA info)."""
    from freetoken.moe.cpu_executor import physical_core_cpus

    cores = physical_core_cpus()
    nodes = numa_nodes()
    if len(nodes) <= 1:
        return {nodes[0] if nodes else None: cores}
    out: dict[int | None, list[int]] = {}
    for node in nodes:
        try:
            with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
                members = _parse_cpulist(f.read())
        except OSError:
            continue
        out[node] = [c for c in cores if c in members]
    return out


def _parse_cpulist(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


__all__ = [
    "CpuPoolSpec", "ExpertPlacement", "HostBankSpec", "PoolMode",
    "gpu_numa_node", "node_memory_bytes", "node_physical_cores", "numa_nodes", "plan_expert_placement",
]
