from __future__ import annotations

import functools
import os
import sys
from pathlib import Path
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


@functools.cache
def _load_nccl_module() -> Module:
    # NVIDIA's pip/conda NCCL wheel intentionally ships only the SONAME file
    # ``libnccl.so.2`` (no development symlink ``libnccl.so``), while a system NCCL
    # development package normally provides the latter. ``-lnccl`` therefore works on
    # the system layout but fails in zl_freetoken even when LD_LIBRARY_PATH is correct.
    # Pass the wheel's absolute path when present; retain ``-lnccl`` for system installs.
    candidates = []
    env_nccl = os.environ.get("NCCL_LIB")
    if env_nccl:
        candidates.append(Path(env_nccl) / "libnccl.so.2")
    candidates.extend(
        Path(entry) / "nvidia" / "nccl" / "lib" / "libnccl.so.2"
        for entry in sys.path
        if entry
    )
    nccl = next((path for path in candidates if path.is_file()), None)
    ldflags = [str(nccl)] if nccl is not None else ["-lnccl"]
    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=ldflags)


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
