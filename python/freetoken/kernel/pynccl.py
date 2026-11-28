from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, Any, Literal

from freetoken.env import ENV

from .utils import load_aot

if TYPE_CHECKING:
    from abc import abstractmethod

    import torch
    from tvm_ffi import Module

    class PyNCCLCommunicator:
        @abstractmethod
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...
        @abstractmethod
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...
        @abstractmethod
        def get_buffer(self) -> int: ...

else:
    PyNCCLCommunicator = Any


def _nccl_ldflags() -> list[str]:
    """Link the NCCL that torch loads. pip's nvidia-nccl ships only ``libnccl.so.2``, with no ``libnccl.so`` for
    ``-lnccl`` to find, so link that file by name from the package; a system NCCL keeps ``-lnccl``."""
    try:
        import nvidia.nccl
    except ImportError:
        return ["-lnccl"]
    lib = next((os.path.join(d, "lib") for d in nvidia.nccl.__path__ if os.path.exists(os.path.join(d, "lib", "libnccl.so.2"))), None)
    if lib is None:
        return ["-lnccl"]
    return [f"-L{lib}", "-l:libnccl.so.2", f"-Wl,-rpath,{lib}"]


@functools.cache
def _load_nccl_module() -> Module:
    # TODO(ROCm): NCCL -> RCCL migration for multi-GPU tensor parallelism on AMD.
    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=_nccl_ldflags())


@functools.cache
def _get_pynccl_wrapper_cls():
    import tvm_ffi

    @tvm_ffi.register_object("freetoken.NCCLWrapper")
    class PyNCCLImpl(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return PyNCCLImpl


def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
) -> PyNCCLCommunicator:
    import torch

    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    module = _load_nccl_module()
    cls = _get_pynccl_wrapper_cls()

    if tp_rank == 0:
        id_list = [module.create_nccl_uid()]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )
    else:
        id_list = [None]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )

    nccl_id = id_list[0]
    assert not nccl_id is None, f"Failed to get NCCL unique ID on {tp_rank = }"

    # bypass type checking for the FFI object
    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)  # type: ignore
