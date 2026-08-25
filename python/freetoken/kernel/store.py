from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch
from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@functools.cache
def _jit_store_module(
    element_size: int,
    *,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, *config)
    return load_jit(
        "store",
        *args,
        cuda_files=["store.cu"],
        cuda_wrappers=[("launch", f"StoreKernel<{args}>::run")],
    )


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    # CPU-only fallback: the store kernel is a CUDA-only JIT extension. On a
    # GPU-less box we scatter k/v into the paged cache rows with plain indexing,
    # which is numerically identical for the dense decode/append path.
    if not torch.cuda.is_available():
        # `k_cache`/`v_cache` arrive as [num_slots, num_heads, head_dim] (the
        # page_size==1 storage view). The incoming `k`/`v` are flat
        # [num_new_tokens, num_heads*head_dim] (qkv split keeps the head dim
        # flattened), so reshape into [num_new_tokens, num_heads, head_dim] and
        # write each token's block to its slot via plain advanced indexing on
        # dim-0. This keeps the head dimension correctly separated (the earlier
        # flatten-and-scatter variant collapsed every head onto head-0).
        nh, hd = k_cache.shape[1], k_cache.shape[2]
        k_cache[indices] = k.view(-1, nh, hd)
        v_cache[indices] = v.view(-1, nh, hd)
        return

    num_tokens = k_cache.shape[0]
    k_cache = k_cache.view(num_tokens, -1)
    v_cache = v_cache.view(num_tokens, -1)
    element_size = k_cache.shape[1] * k_cache.element_size()
    module = _jit_store_module(element_size)
    module.launch(k_cache, v_cache, indices, k, v)
