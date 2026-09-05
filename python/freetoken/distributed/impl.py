from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from freetoken.distributed import DistributedInfo
    from freetoken.kernel import PyNCCLCommunicator


@dataclass
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class SingleRankWork:
    """Completed work handle for single-rank process-group operations."""

    def wait(self) -> bool:
        return True


@dataclass(frozen=True)
class SingleRankProcessGroup:
    """Minimal CPU process-group contract when there is only one rank.

    This deliberately implements only the operations FreeToken uses on its
    CPU group.  It must never be selected for tensor parallel sizes above one.
    """

    def barrier(self) -> SingleRankWork:
        return SingleRankWork()

    def broadcast(self, tensor: torch.Tensor, root: int = 0) -> SingleRankWork:
        if root != 0:
            raise ValueError(f"single-rank broadcast root must be 0, got {root}")
        _ = tensor
        return SingleRankWork()


@dataclass
class SingleRankDistributedImpl(DistributedImpl):
    """Identity layer collectives for tensor parallel size one."""

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return x


@dataclass
class TorchDistributedImpl(DistributedImpl):
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        from .info import get_tp_info

        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result


class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)


def torch_distributed_process_group_available() -> bool:
    """Whether this torch build can provide FreeToken's process-group path."""

    is_available = getattr(dist, "is_available", None)
    if not callable(is_available):
        return False
    try:
        if not is_available():
            return False
    except Exception:
        return False

    required_apis = (
        "init_process_group",
        "destroy_process_group",
        "get_world_size",
        "all_reduce",
    )
    return all(callable(getattr(dist, name, None)) for name in required_apis)


def enable_single_rank_distributed() -> None:
    """Select identity collectives for a process with tensor parallel size one."""

    DistributedCommunicator.plugins.append(SingleRankDistributedImpl())


def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """
    Enable PyNCCL-based distributed communication for tensor parallelism.
    """
    if tp_info.size == 1:
        return
    from freetoken.kernel import init_pynccl

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))


def destroy_distributed() -> None:
    """
    Destroy all the distributed communication plugins.
    """
    DistributedCommunicator.plugins = []
