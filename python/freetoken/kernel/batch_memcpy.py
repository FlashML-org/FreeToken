from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from .utils import load_jit

if TYPE_CHECKING:
    from tvm_ffi import Module


@lru_cache(maxsize=None)
def _jit_batch_memcpy_module() -> Module:
    return load_jit(
        "batch_memcpy",
        cuda_files=["batch_memcpy.cuh"],
        cuda_wrappers=[("batch_memcpy", "&BatchMemcpy::run")],
    )


def _probe(fn) -> None:
    """One real 16-byte H2D through the binding. A version-gated build (panic
    branch) or a driver without batch-memcpy support loads cleanly and only fails
    at call time; probing here turns every such mode into a load_batch_memcpy
    exception the caller's fallback path can catch."""
    src = torch.arange(16, dtype=torch.uint8).pin_memory()
    dst = torch.zeros(16, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    fn(
        torch.tensor([dst.data_ptr()]),
        torch.tensor([src.data_ptr()]),
        torch.tensor([16]),
        stream.cuda_stream,
    )
    stream.synchronize()
    if not torch.equal(dst.cpu(), src):
        raise RuntimeError("cudaMemcpyBatchAsync probe copied wrong bytes")


def load_batch_memcpy():
    """Build (once), probe, and return the batch-memcpy entry point, or raise.

    On ROCm the HIP per-copy grid kernel is used (no CUDA version gate). On CUDA the
    8-argument cudaMemcpyBatchAsync signature this binding uses is CUDA 13.0's
    (12.8/12.9 had an extra failIdx parameter); gate on the torch runtime version
    before paying for the JIT build, then verify with a real copy.
    """
    if torch.version.hip is not None:
        fn = _jit_batch_memcpy_module().batch_memcpy
        _probe(fn)
        return fn
    cuda = torch.version.cuda
    if cuda is None or tuple(int(x) for x in cuda.split(".")[:2]) < (13, 0):
        raise RuntimeError(f"cudaMemcpyBatchAsync binding requires CUDA >= 13.0 (torch built with {cuda})")
    fn = _jit_batch_memcpy_module().batch_memcpy
    _probe(fn)
    return fn


def batch_memcpy_jit(
    dst_ptrs: torch.Tensor,
    src_ptrs: torch.Tensor,
    sizes: torch.Tensor,
    stream: int,
) -> None:
    """Enqueue one cudaMemcpyBatchAsync of ``len(sizes)`` independent copies.

    ``dst_ptrs``/``src_ptrs``/``sizes`` are same-length CPU int64 tensors of raw
    addresses and byte counts; ``stream`` is a raw cudaStream_t handle
    (``torch.cuda.Stream.cuda_stream``), which must not be the legacy NULL stream.
    """
    load_batch_memcpy()(dst_ptrs, src_ptrs, sizes, stream)
