from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    rank: int
    size: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        return self.rank == 0


_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> None:
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO


def reset_tp_info() -> None:
    """Clear process-global TP info. Tests only."""
    global _TP_INFO
    _TP_INFO = None


# The gloo CPU subgroup created alongside the GPU (NCCL/RCCL) world. Kept
# process-global so code far from the engine -- e.g. the shared expert-bank
# handoff in freetoken.moe.host_banks, which broadcasts rank 0's memfd paths --
# can run a CPU-only collective without routing a group handle through every
# call site. Such a collective must NOT run on the GPU group: on HIP, torch
# guesses the device id from the global rank (ProcessGroupNCCL "Guessing device
# ID based on global rank ... can cause a hang if rank to GPU mapping is
# heterogeneous"), and handing over a few strings needs no GPU at all.
_TP_CPU_GROUP: object | None = None


def set_tp_cpu_group(group: object | None) -> None:
    global _TP_CPU_GROUP
    _TP_CPU_GROUP = group


def try_get_tp_cpu_group() -> object | None:
    """The gloo CPU group, or None before ``init_tp_process_group``."""
    return _TP_CPU_GROUP


def reset_tp_cpu_group() -> None:
    """Clear the process-global CPU group. Tests and teardown."""
    global _TP_CPU_GROUP
    _TP_CPU_GROUP = None


__all__ = [
    "DistributedInfo",
    "set_tp_info",
    "get_tp_info",
    "try_get_tp_info",
    "reset_tp_info",
    "set_tp_cpu_group",
    "try_get_tp_cpu_group",
    "reset_tp_cpu_group",
]
