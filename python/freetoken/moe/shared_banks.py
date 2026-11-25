"""Host expert banks of placed expert shards, each loaded once and shared by the ranks that use it.

Every bank of an ``ExpertPlacement`` is a ``memfd`` region whose pages are bound to its NUMA node.
All ranks map all banks while loading: each rank reads one disjoint range of MoE layers from the
checkpoint and packs every bank's slice of those layers, so the checkpoint is read once whatever
the placement. Afterwards a rank keeps serving from one bank: under ``numa`` placement the ranks
of a node share their node's bank (one copy per node, which the node's CPU pool also reads);
under ``rank`` placement each rank has its own.
"""

from __future__ import annotations

import ctypes
import math
import mmap
import os
import platform
from typing import Callable

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_PAGE = 4096
_MPOL_BIND = 2
_SYS_MBIND = {"x86_64": 237, "aarch64": 235}

# mappings and fds live as long as the process: the offload cache and the CPU pool read the banks forever
_LIVE: list = []


def bind_to_node(addr: int, nbytes: int, node: int | None) -> bool:
    """Bind ``[addr, addr + nbytes)``'s future pages to NUMA ``node`` (MPOL_BIND); False if the kernel refused.

    Must run before the pages are first touched. On a shared mapping the policy belongs to the
    shared object, so whichever process faults a page in, it lands on ``node``."""
    if node is None:
        return True
    nr = _SYS_MBIND.get(platform.machine())
    if nr is None:
        return False
    bits = 8 * ctypes.sizeof(ctypes.c_ulong)
    mask = (ctypes.c_ulong * (node // bits + 1))()
    mask[node // bits] = 1 << (node % bits)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    ok = libc.syscall(nr, ctypes.c_void_p(addr), ctypes.c_ulong(nbytes), _MPOL_BIND, mask, ctypes.c_ulong(len(mask) * bits + 1), 0) == 0
    if not ok:
        err = ctypes.get_errno()
        logger.warning(f"mbind of {nbytes / 2**30:.1f} GiB of expert banks to NUMA node {node} failed: {os.strerror(err)}; leaving them unbound")
    return ok


def _round_page(n: int) -> int:
    return -(-n // _PAGE) * _PAGE


class SharedBank:
    """One bank carved out of a :class:`SharedBankArena`; quacks like ``host_banks.HostBank`` for the bank builder."""

    __slots__ = ("tensor", "addr", "nbytes", "offset", "_span", "_pinned")

    def __init__(self, arena: "SharedBankArena", offset: int, shape: tuple[int, ...], dtype: torch.dtype):
        self.nbytes = math.prod(shape) * dtype.itemsize
        self.offset = offset
        self._span = _round_page(self.nbytes)
        self.addr = arena.addr + offset
        self.tensor = torch.frombuffer(arena.buffer, dtype=dtype, count=self.nbytes // dtype.itemsize, offset=offset).view(*shape)
        self._pinned = False

    @property
    def residency(self):
        from freetoken.moe.host_banks import HostResidency

        return HostResidency.PINNED if self._pinned else HostResidency.PAGEABLE

    def pin(self) -> None:
        if self._pinned:
            return
        from freetoken.kernel.pinned import host_register

        if self._span:
            host_register(self.addr, self._span)
        self._pinned = True

    def lock(self) -> None:
        raise ValueError("shared NUMA expert banks are always pinned; they do not combine with OS-locked layers")


class SharedBankArena:
    """A ``memfd`` region bound to one NUMA node, handed out as page-aligned banks."""

    def __init__(self, nbytes: int, node: int | None):
        self.size = _round_page(max(nbytes, 1))
        self.node = node
        self.fd = os.memfd_create("freetoken-expert-banks", 0)
        os.ftruncate(self.fd, self.size)
        self.buffer = mmap.mmap(self.fd, self.size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self.buffer))
        self.bound = bind_to_node(self.addr, self.size, node)
        self._next = 0
        self.banks: list[SharedBank] = []
        _LIVE.extend((self.buffer, self.fd))

    def allocate(self, shape: tuple[int, ...], dtype: torch.dtype) -> SharedBank:
        bank = SharedBank(self, self._next, tuple(shape), dtype)
        if self._next + bank._span > self.size:
            raise ValueError(f"shared expert banks overflow their {self.size} byte arena")
        self._next += bank._span
        self.banks.append(bank)
        return bank

    def handle(self) -> dict:
        """What a peer on the same host needs to map this arena."""
        return {"path": f"/proc/{os.getpid()}/fd/{self.fd}", "size": self.size, "node": self.node}


def attach_arena(handle: dict) -> tuple[mmap.mmap, int]:
    """Map a peer's arena; returns the mapping and its address. The pages are the owner's, not a copy."""
    fd = os.open(handle["path"], os.O_RDWR)
    try:
        buffer = mmap.mmap(fd, handle["size"], flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)
    _LIVE.append(buffer)
    return buffer, ctypes.addressof(ctypes.c_char.from_buffer(buffer))


def arena_bytes(layout: dict, num_experts: int, num_layers: int) -> int:
    """Bytes an arena needs for ``num_layers`` layers of banks in ``layout`` (page-aligned per bank)."""
    return num_layers * sum(
        _round_page(num_experts * math.prod(spec.shape) * spec.dtype.itemsize) for spec in layout.values() if not spec.resident
    )


def _gather(group, value) -> list:
    import torch.distributed as dist

    out: list = [None] * dist.get_world_size(group)
    dist.all_gather_object(out, value, group=group)
    return out


def _raise_any(group, failure: BaseException | None) -> None:
    """Every rank reaches this; all of them raise if one failed, so none waits out the timeout."""
    errors = [e for e in _gather(group, None if failure is None else f"{type(failure).__name__}: {failure}") if e]
    if failure is not None:
        raise failure
    if errors:
        raise RuntimeError(f"another rank's expert bank load failed: {errors[0]}")


def load_placed_expert_banks(placement, rank: int, method, read_pieces: Callable | None, *, num_experts: int, num_layers: int,
                             device: torch.device, timeout: float = 3 * 3600):
    """This rank's host banks under ``placement``; a collective over the default process group.

    Every bank is a ``memfd`` arena bound to its NUMA node, created by its owner and mapped by all
    ranks. Each rank reads one disjoint range of MoE layers, ``read_pieces(layers)`` (expert pieces
    of just those layers; None fabricates dummy rows), and packs every bank's slice of them, so the
    checkpoint is read once in total for any placement and the reads spread over all ranks. Each
    rank then pins the bank it serves from: its own (``rank``), or its node's (``numa``). The ranks
    wait for each other on a gloo group of their own with ``timeout`` seconds, since the load takes
    minutes, far past the engine's collective timeout.
    """
    from datetime import timedelta

    import torch.distributed as dist

    from freetoken.kernel.pinned import host_register
    from freetoken.moe.expert_banks import ExpertBanks, _dummy_fill
    from freetoken.moe.legacy_format import legacy_format_for

    group = dist.new_group(backend="gloo", timeout=timedelta(seconds=timeout))
    slicing, kernel = method.expert_slicing(), method.kernel
    # the banks are packed for one kernel and the collectives below follow its layout; every rank raises on a mismatch
    kernels = _gather(group, kernel.name)
    if len(set(kernels)) != 1:
        raise ValueError(f"the ranks would pack the shared expert banks for different kernels: {kernels} (by rank)")
    E, L, world = num_experts, num_layers, placement.world_size
    configs = [slicing.config_at(bank.interval.width) for bank in placement.banks]
    layouts = [{role: spec for role, spec in kernel.layout(cfg).items() if not spec.resident} for cfg in configs]

    created, failure = {}, None
    try:
        for i, bank in enumerate(placement.banks):
            if bank.owner == rank:
                arena = SharedBankArena(arena_bytes(layouts[i], E, L), bank.node)
                offsets = {role: [arena.allocate((E, *spec.shape), spec.dtype).offset for _ in range(L)] for role, spec in layouts[i].items()}
                created[i] = (arena, offsets)
    except BaseException as exc:
        failure = exc
    handles = {i: (arena.handle(), offsets) for i, (arena, offsets) in created.items()}
    published = {i: h for part in _gather(group, handles) for i, h in part.items()}
    _raise_any(group, failure)

    mappings, targets = [], []
    for i, bank in enumerate(placement.banks):
        if i in created:
            buffer, addr = created[i][0].buffer, created[i][0].addr
        else:
            buffer, addr = attach_arena(published[i][0])
        mappings.append((buffer, addr, published[i][0]["size"]))
        offsets = published[i][1]
        targets.append({
            role: [
                torch.frombuffer(buffer, dtype=spec.dtype, count=E * math.prod(spec.shape), offset=o).view(E, *spec.shape)
                for o in offsets[role]
            ]
            for role, spec in layouts[i].items()
        })

    mine = list(range(rank * L // world, (rank + 1) * L // world))
    resident = {role: spec for role, spec in kernel.layout(configs[0]).items() if spec.resident}
    alphas = {role: torch.zeros(L * E, dtype=torch.float32) for role in resident}
    # every rank packs at once on one host: each takes its share of the cores, or the intra-op pools oversubscribe them (5x slower)
    threads = torch.get_num_threads()
    torch.set_num_threads(max(1, threads // world))
    try:
        if read_pieces is None:
            for banks in targets:
                for role, per_layer in banks.items():
                    for layer in mine:
                        _dummy_fill(role, per_layer[layer])
            for a in alphas.values():
                a.view(L, E)[mine] = 1.0
        else:
            written = torch.zeros(L, E, dtype=torch.int32)
            for layer, e0, e1, piece in read_pieces(mine):
                if layer not in mine or not 0 <= e0 < e1 <= E or written[layer, e0:e1].any():
                    raise ValueError(f"expert piece out of this rank's layers or repeated: layer {layer}, experts {e0}:{e1}")
                written[layer, e0:e1] = 1
                for bank, cfg, banks in zip(placement.banks, configs, targets):
                    out = {role: per_layer[layer][e0:e1] for role, per_layer in banks.items()}
                    got = kernel.pack(slicing.slice_pieces(piece, bank.interval), cfg, out)
                for role, values in got.items():
                    alphas[role][layer * E + e0 : layer * E + e1] = values.float()
            missing = (written[mine] == 0).nonzero().tolist()
            if missing:
                raise ValueError(f"expert banks were not filled: {len(missing)} (layer, expert) rows missing (first {missing[:4]})")
    except BaseException as exc:
        failure = exc
    finally:
        torch.set_num_threads(threads)
    _raise_any(group, failure)
    for a in alphas.values():  # each row came from one rank, zeros elsewhere: the sum is exact
        dist.all_reduce(a, group=group)

    bank = placement.bank_of(rank)
    i = placement.banks.index(bank)
    buffer, addr, size = mappings[i]
    host_register(addr, size)  # pin after fill; every rank pins the one bank it fetches from
    dist.barrier(group=group)
    logger.info(
        f"expert banks: rank {rank} read MoE layers {mine[0] if mine else '-'}..{mine[-1] if mine else '-'} of {L} once into "
        f"{len(placement.banks)} bank(s); serving slice [{bank.interval.start}, {bank.interval.stop}) from NUMA node {bank.node}"
    )
    dtypes = {role: spec.dtype for role, spec in resident.items()}
    return ExpertBanks(
        legacy_format_for(method.kind, kernel.name), targets[i],
        gate_up_alpha=alphas["gate_up_alpha"].to(device, dtypes["gate_up_alpha"]) if "gate_up_alpha" in alphas else None,
        down_alpha=alphas["down_alpha"].to(device, dtypes["down_alpha"]) if "down_alpha" in alphas else None,
        kind=method.kind, kernel=kernel.name, layout=kernel.layout(configs[i]),
        stored=bank.interval if bank.interval.width != slicing.intermediate else None,
    )


__all__ = [
    "SharedBank", "SharedBankArena", "arena_bytes", "attach_arena", "bind_to_node", "load_placed_expert_banks",
]
