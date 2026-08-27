import os as _os


_installed_kernel_dir = _os.environ.get("FREETOKEN_INSTALLED_KERNEL_DIR")
if (
    _installed_kernel_dir
    and _os.path.isdir(_installed_kernel_dir)
    and _installed_kernel_dir not in __path__
):
    # A source checkout can reuse the matching native extensions from an
    # installed FreeToken runtime.  Spawned server workers import this package
    # again, so the path must be applied here instead of only in a launcher.
    __path__.append(_installed_kernel_dir)

from .index import indexing
from .fast_index_copy import fast_index_copy_jit, update_copy_flag_jit
from .moe_impl import (
    fused_moe_decode_kernel_triton,
    fused_moe_kernel_triton,
    gpt_oss_fused_routing,
    gpt_oss_swiglu_triton,
    get_fp4_lut,
    moe_align_block_size_triton,
    moe_sum_reduce_triton,
    mxfp4_fused_moe_kernel_t_triton,
    mxfp4_splitk_gemv_triton,
)
from .pinned import copy_to_pinned_tensor, create_pinned_tensor_like
from .pynccl import PyNCCLCommunicator, init_pynccl
from .radix import fast_compare_key
from .store import store_cache
from .tensor import test_tensor

__all__ = [
    "indexing",
    "fast_index_copy_jit",
    "update_copy_flag_jit",
    "fast_compare_key",
    "store_cache",
    "test_tensor",
    "init_pynccl",
    "PyNCCLCommunicator",
    "fused_moe_kernel_triton",
    "fused_moe_decode_kernel_triton",
    "gpt_oss_fused_routing",
    "mxfp4_fused_moe_kernel_t_triton",
    "mxfp4_splitk_gemv_triton",
    "get_fp4_lut",
    "gpt_oss_swiglu_triton",
    "moe_align_block_size_triton",
    "moe_sum_reduce_triton",
    "create_pinned_tensor_like",
    "copy_to_pinned_tensor",
]
