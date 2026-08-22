"""NUMA topology and memory placement for the CPU MoE path.

The expert GEMV reads every routed expert's bytes from host DRAM exactly once per
token, so on a multi-socket box the whole path lives or dies on locality. Two halves
have to agree, and neither works alone:

* **Where the worker threads run** -- see ``moe.cpu_executor``.
* **Where the expert banks live** -- this module. Confining the workers while the
  banks land wherever the loader threads happened to run turns a stable mediocre
  result into a coin flip: measured 56-123 GB/s run to run on a 2x Xeon Gold 6526Y,
  against a steady 69 GB/s before confinement.

Placement uses ``mbind`` with ``MPOL_PREFERRED`` -- a hint, not a reservation. The
banks are 130+ GiB and a strict ``MPOL_BIND`` would OOM rather than spill to the
other node once the target fills, which is a far worse failure than a remote read.

Everything degrades to a no-op off Linux, on a single-node machine, or where libnuma
policy is unavailable, so single-socket desktops are untouched.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import functools
import os

from .logger import init_logger

logger = init_logger(__name__)

MPOL_PREFERRED = 1
# glibc exports no mbind() wrapper (it lives in libnuma), so go through syscall() and
# keep libnuma off the dependency list. Numbers are stable per ABI.
_SYS_MBIND = {"x86_64": 237, "aarch64": 235}
_SYS_SET_MEMPOLICY = {"x86_64": 238, "aarch64": 237}
MPOL_DEFAULT = 0


def _allowed_cpus() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return list(range(os.cpu_count() or 1))


def cpu_numa_node(cpu: int) -> int | None:
    """NUMA node of a logical CPU, from the ``cpuN/nodeX`` sysfs symlink (None if absent)."""
    try:
        for entry in os.listdir(f"/sys/devices/system/cpu/cpu{cpu}"):
            if entry.startswith("node") and entry[4:].isdigit():
                return int(entry[4:])
    except OSError:
        pass
    return None


def thread_siblings(cpu: int) -> str | None:
    """The ``cpuN/topology/thread_siblings_list`` string, or None where sysfs is silent."""
    try:
        with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as f:
            return f.read().strip()
    except OSError:
        return None


def numa_nodes() -> set[int]:
    """NUMA nodes this process may run on. Empty/singleton means "nothing to choose"."""
    return {n for cpu in _allowed_cpus() if (n := cpu_numa_node(cpu)) is not None}


def gpu_numa_node(device) -> int | None:
    """NUMA node the CUDA device's PCI function hangs off, or None if unknown.

    The BDF is rebuilt from torch's device properties and looked up in sysfs; a driver
    that reports -1 (no affinity) yields None.
    """
    import torch

    if not torch.cuda.is_available():
        return None
    try:
        index = device if isinstance(device, int) else (
            None if device is None else torch.device(device).index)
        if index is None:  # device=None, or a bare "cuda" with no ordinal
            index = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(index)
        bdf = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
        with open(f"/sys/bus/pci/devices/{bdf}/numa_node") as f:
            node = int(f.read().strip())
    except (AttributeError, OSError, ValueError, RuntimeError):
        return None
    return node if node >= 0 else None


def moe_pool_numa_node(device=None) -> int | None:
    """Which NUMA node the CPU MoE path should keep its threads and banks on.

    The GPU's node wins: it keeps the expert GEMV *and* the offload gather's DMA local.
    Returns None -- meaning "do not confine", today's behaviour -- on a single-node
    machine, when the topology cannot be read, or under ``FREETOKEN_CPU_MOE_NUMA=off``.
    """
    # "auto" | "off" | <node id>. No boolean spellings on purpose: "0" and "1" are node
    # ids, so accepting them as on/off would make the common case ambiguous.
    setting = os.getenv("FREETOKEN_CPU_MOE_NUMA", "auto").strip().lower()
    if setting in ("off", "none"):
        return None
    nodes = numa_nodes()
    if setting not in ("auto", ""):
        try:
            forced = int(setting)
        except ValueError:
            logger.warning(
                f"ignoring FREETOKEN_CPU_MOE_NUMA={setting!r}: expected 'auto', 'off' "
                "or a NUMA node id"
            )
            return None
        if forced not in nodes:
            logger.warning(f"FREETOKEN_CPU_MOE_NUMA={forced} has no runnable CPU; not confining")
            return None
        return forced
    if len(nodes) < 2:
        return None  # single node (or unknown): every core is already local
    gpu_node = gpu_numa_node(device)
    return gpu_node if gpu_node in nodes else min(nodes)


@functools.lru_cache(maxsize=1)
def _mbind():
    """A callable ``mbind(addr, len, mode, mask, maxnode, flags)``, or None if unavailable."""
    import platform

    nr = _SYS_MBIND.get(platform.machine())
    if nr is None or not hasattr(os, "sched_getaffinity"):  # unknown ABI, or not Linux
        return None
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    except OSError:
        return None
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    syscall.argtypes = [ctypes.c_long, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int,
                        ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong, ctypes.c_uint]

    def call(addr, length, mode, mask, maxnode, flags):
        return syscall(nr, addr, length, mode, mask, maxnode, flags)

    return call


def prefer_node(addr: int, length: int, node: int | None) -> bool:
    """Ask the kernel to place ``[addr, addr+length)`` on ``node`` when it faults.

    A hint (``MPOL_PREFERRED``), so an over-full node spills to a neighbour instead of
    OOM-ing. Must be called *before* the range is touched -- already-resident pages are
    not moved (and pinned ones could not be). Returns whether the policy was applied.
    """
    if node is None or length <= 0:
        return False
    fn = _mbind()
    if fn is None:
        return False
    mask = ctypes.c_ulong(1 << node)
    # maxnode counts bits in the mask, not nodes set.
    rc = fn(ctypes.c_void_p(addr), ctypes.c_ulong(length), MPOL_PREFERRED,
            ctypes.byref(mask), ctypes.c_ulong(ctypes.sizeof(mask) * 8), 0)
    if rc != 0:
        logger.debug(
            f"mbind(MPOL_PREFERRED, node {node}) failed: {os.strerror(ctypes.get_errno())}"
        )
        return False
    return True


@functools.lru_cache(maxsize=1)
def _set_mempolicy():
    """A callable ``set_mempolicy(mode, mask, maxnode)``, or None if unavailable."""
    import platform

    nr = _SYS_SET_MEMPOLICY.get(platform.machine())
    if nr is None or not hasattr(os, "sched_getaffinity"):
        return None
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    except OSError:
        return None
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    syscall.argtypes = [ctypes.c_long, ctypes.c_int, ctypes.POINTER(ctypes.c_ulong),
                        ctypes.c_ulong]

    def call(mode, mask, maxnode):
        return syscall(nr, mode, mask, maxnode)

    return call


@contextlib.contextmanager
def allocating_on_node(node: int | None):
    """Place pages this thread faults in while inside the block on ``node``.

    ``prefer_node`` cannot help an allocator that faults its pages up front --
    ``cudaHostAlloc`` pins as it allocates, and pinned pages can never be migrated.
    Setting the *thread's* policy first is the only way to steer those. Restores
    ``MPOL_DEFAULT`` on exit; a no-op when ``node`` is None or the syscall is missing.
    """
    fn = None if node is None else _set_mempolicy()
    if fn is not None:
        mask = ctypes.c_ulong(1 << node)
        if fn(MPOL_PREFERRED, ctypes.byref(mask), ctypes.sizeof(mask) * 8) != 0:
            logger.debug(f"set_mempolicy(node {node}): {os.strerror(ctypes.get_errno())}")
            fn = None
    try:
        yield
    finally:
        if fn is not None:
            fn(MPOL_DEFAULT, None, 0)
