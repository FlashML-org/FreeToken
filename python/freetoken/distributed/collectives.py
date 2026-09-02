"""Tensor-parallel process-group setup: NVIDIA PyNCCL vs AMD RCCL.

PyNCCL (``pynccl.cu``, ``-lnccl``, NCCL 2.27 ``ncclMemAlloc`` /
``ncclCommWindowRegister``) is NVIDIA-only. RCCL does not implement that
window API; hipifying it is the wrong path.

On HIP, TP>1 uses ``torch.distributed`` with backend ``rccl`` when the
wheel exposes it, otherwise the ROCm alias still named ``nccl`` (that
build links RCCL, not NVIDIA NCCL). Fail loudly if a HIP process tries
to load PyNCCL or if neither backend exists.

This module does not implement new collectives. Dense TP all-reduce /
all-gather stay on :class:`TorchDistributedImpl` (HIP) or
:class:`PyNCCLDistributedImpl` (NVIDIA default).
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import TYPE_CHECKING

from freetoken.runtime.gpu import is_hip, nvidia_only_error
from freetoken.utils import init_logger

if TYPE_CHECKING:
    import torch
    from freetoken.engine.config import EngineConfig

logger = init_logger(__name__)


def pynccl_supported() -> bool:
    """True only on NVIDIA CUDA. HIP never JITs ``pynccl.cu`` / ``-lnccl``."""
    return not is_hip()


def require_pynccl_allowed() -> None:
    """Raise if this process must not load NVIDIA NCCL / PyNCCL."""
    if is_hip():
        raise nvidia_only_error(
            "PyNCCL / NVIDIA NCCL 2.27 (pynccl.cu, -lnccl, ncclMemAlloc, "
            "ncclCommWindowRegister)",
            hint=(
                "On AMD use torch.distributed RCCL (backend 'rccl' or the ROCm "
                "'nccl' alias). HIP never loads PyNCCL; do not pass a CUDA NCCL "
                "plugin. See docs/amd.md."
            ),
        )


def gpu_dist_backend() -> str:
    """GPU collective backend name for ``torch.distributed.init_process_group``.

    HIP: ``rccl`` if the wheel exposes it, else the ROCm ``nccl`` alias (RCCL).
    NVIDIA: ``nccl``. Raises if the backend is missing — never silently
    fall through to a CUDA-only plugin on AMD.
    """
    import torch.distributed as dist

    if is_hip():
        if hasattr(dist, "is_rccl_available") and dist.is_rccl_available():
            return "rccl"
        if dist.is_nccl_available():
            return "nccl"
        raise RuntimeError(
            "HIP tensor parallel needs RCCL via torch.distributed "
            "(backend 'rccl' or the ROCm 'nccl' alias). This PyTorch build "
            "has neither. Install a ROCm PyTorch wheel (not cu130). "
            "NVIDIA NCCL / PyNCCL is not used on AMD."
        )
    if dist.is_nccl_available():
        return "nccl"
    raise RuntimeError("NCCL is not available in this PyTorch build")


def tp_comm_plan(tp_size: int, use_pynccl: bool, *, hip: bool | None = None) -> str:
    """Which process-group stack this rank will use. No torch.distributed init.

    Returns one of: ``gloo``, ``gloo+pynccl``, ``nccl+gloo``, ``rccl+gloo``.
    """
    if hip is None:
        hip = is_hip()
    if tp_size <= 1:
        return "gloo"
    if hip:
        return "rccl+gloo"
    if use_pynccl:
        return "gloo+pynccl"
    return "nccl+gloo"


def apply_hip_rccl_env() -> dict[str, str]:
    """Desktop dual-card defaults. RCCL still reads ``NCCL_*`` names.

    Only sets vars the user has not already exported. Dual R9700 boxes have
    no InfiniBand; leaving IB enabled can stall RCCL init.
    """
    written: dict[str, str] = {}
    if not is_hip():
        return written
    if not os.environ.get("NCCL_IB_DISABLE"):
        os.environ["NCCL_IB_DISABLE"] = "1"
        written["NCCL_IB_DISABLE"] = "1"
    return written


def hip_tp_missing(tp_size: int) -> list[str]:
    """Precise leftover list when HIP TP cannot start in this process.

    Empty means the *wiring* is present (RCCL backend + enough discrete
    cards after iGPU hide). Does not claim an e2e generate on R9700.
    """
    if tp_size <= 1 or not is_hip():
        return []
    missing: list[str] = []
    try:
        gpu_dist_backend()
    except RuntimeError as exc:
        missing.append(str(exc))
    from freetoken.runtime.gpu import discrete_tp_devices

    devs = discrete_tp_devices()
    if len(devs) < tp_size:
        names = ", ".join(
            f"{d.get('name', '?')}({d.get('arch') or '-'})" for d in devs
        ) or "none"
        missing.append(
            f"discrete GPUs: TP={tp_size} needs {tp_size} cards after hiding the "
            f"Granite Ridge iGPU; `ft gpu` sees {len(devs)} ({names}). "
            f"Ranks must be the two R9700s, not gfx1036."
        )
    return missing


def describe_hip_tp_plan(tp_size: int) -> str:
    plan = tp_comm_plan(tp_size, use_pynccl=True, hip=True)
    try:
        backend = gpu_dist_backend()
    except RuntimeError:
        backend = "(unavailable)"
    return (
        f"HIP TP={tp_size}: plan={plan} torch.distributed backend={backend} "
        f"(RCCL; PyNCCL/NCCL 2.27 disabled)"
    )


def init_tp_process_group(config: EngineConfig, *, dtype: "torch.dtype"):
    """Init the TP process group and return the CPU (gloo) group.

    NVIDIA default (``use_pynccl``): gloo world + PyNCCL plugin — unchanged.
    NVIDIA ``--disable-pynccl``: NCCL world + gloo subgroup — unchanged.
    HIP TP>1: never PyNCCL; RCCL (or ROCm nccl alias) world + gloo subgroup.
    """
    import torch.distributed as dist

    from freetoken.distributed.impl import enable_pynccl_distributed
    from freetoken.distributed.info import set_tp_cpu_group

    timeout = timedelta(seconds=config.distributed_timeout)
    rank = config.tp_info.rank
    world_size = config.tp_info.size
    init_method = config.distributed_addr

    if world_size == 1:
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            timeout=timeout,
            init_method=init_method,
        )
        group = dist.group.WORLD
        assert group is not None
        set_tp_cpu_group(group)
        return group

    if is_hip():
        if config.use_pynccl:
            logger.warning(
                "HIP: PyNCCL is NVIDIA NCCL 2.27-only; using torch.distributed RCCL"
            )
        apply_hip_rccl_env()
        missing = hip_tp_missing(world_size)
        if missing:
            raise RuntimeError(
                "HIP tensor parallel cannot start:\n"
                + "\n".join(f"  - {item}" for item in missing)
            )
        backend = gpu_dist_backend()
        logger.info(
            f"HIP TP: init_process_group(backend={backend!r}) + gloo CPU group "
            f"(NCCL_IB_DISABLE={os.environ.get('NCCL_IB_DISABLE', '')})"
        )
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timeout,
            init_method=init_method,
        )
        tp_cpu_group = dist.new_group(backend="gloo")
        assert tp_cpu_group is not None
        set_tp_cpu_group(tp_cpu_group)
        return tp_cpu_group

    if config.use_pynccl:
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            timeout=timeout,
            init_method=init_method,
        )
        tp_cpu_group = dist.group.WORLD
        assert tp_cpu_group is not None
        set_tp_cpu_group(tp_cpu_group)
        max_bytes = config.max_forward_len * config.model_config.hidden_size * dtype.itemsize
        enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        return tp_cpu_group

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timeout,
        init_method=init_method,
    )
    tp_cpu_group = dist.new_group(backend="gloo")
    assert tp_cpu_group is not None
    set_tp_cpu_group(tp_cpu_group)
    return tp_cpu_group
