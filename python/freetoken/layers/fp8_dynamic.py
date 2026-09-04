"""Load-time per-tensor FP8 dense linear (W8A8 through cuBLASLt ``torch._scaled_mm``).

The weight is e4m3 ``[out_local, in_local]`` with one fp32 ``weight_scale`` (shape ``()``),
produced by the model's weight reader from the bf16 checkpoint tensor (after TP sharding).
The activation is quantized per call with a dynamic per-tensor scale: one fused Triton launch
(amax pass, then cast) at decode sizes, a torch reduction plus a cast kernel above that. No
host sync anywhere (the scales stay on the device), so the decode path is CUDA-graph safe;
the branch between the two paths is on the tensor *shape*, never on its values.

Measured on an RTX 6000 Ada (sm_89, torch 2.11.0+cu130) at the qwen4_exp TP=2 shapes, per
decode step per rank over 48 layers: bf16 cuBLAS 3.2-3.4 ms, this path 1.9 ms at M=1/8/16.
Requires sm_89+ (``_scaled_mm``'s floor; Ampere has no FP8 tensor cores).
"""

from __future__ import annotations

from typing import List

import torch
import triton
import triton.language as tl
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.utils import div_even

from .base import BaseOP

FP8 = torch.float8_e4m3fn
E4M3_MAX = 448.0
_FUSED_MAX_ELEMENTS = (
    65536  # one program handles the whole tensor below this (decode sizes)
)
_SCALE_FLOOR = 1e-12  # an all-zero activation (graph warmup buffers) must not give 1/0


@triton.jit
def _quant_fused_kernel(x_ptr, out_ptr, scale_ptr, n, BLOCK: tl.constexpr):
    """One program: max|x| over the tensor, then the cast. Decode-sized inputs only."""
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
        acc = tl.maximum(acc, tl.abs(v))
    amax = tl.maximum(tl.max(acc, axis=0), 1e-12)
    tl.store(scale_ptr, amax / 448.0)
    inv = 448.0 / amax
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < n
        v = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32) * inv
        v = tl.minimum(tl.maximum(v, -448.0), 448.0)
        tl.store(out_ptr + offs, v.to(tl.float8e4nv), mask=mask)


@triton.jit
def _quant_cast_kernel(x_ptr, out_ptr, scale_ptr, n, BLOCK: tl.constexpr):
    """Cast under a scale already on the device (prefill sizes; the amax is a torch reduction)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    inv = 1.0 / tl.load(scale_ptr)
    v = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32) * inv
    v = tl.minimum(tl.maximum(v, -448.0), 448.0)
    tl.store(out_ptr + offs, v.to(tl.float8e4nv), mask=mask)


def quant_per_tensor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(x_fp8, scale)`` with ``x ~= x_fp8 * scale``; ``x`` contiguous, ``scale`` fp32 ``()``."""
    n = x.numel()
    out = torch.empty_like(x, dtype=FP8)
    if n <= _FUSED_MAX_ELEMENTS:
        scale = torch.empty((), dtype=torch.float32, device=x.device)
        _quant_fused_kernel[(1,)](x, out, scale, n, BLOCK=4096, num_warps=8)
        return out, scale
    amax = torch.linalg.vector_norm(x, ord=float("inf")).float()
    scale = amax.clamp_min_(_SCALE_FLOOR).div_(E4M3_MAX)
    _quant_cast_kernel[(triton.cdiv(n, 4096),)](
        x, out, scale, n, BLOCK=4096, num_warps=4
    )
    return out, scale


def fp8_dynamic_linear(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    """``x @ (weight * weight_scale)^T`` in W8A8; ``weight`` [N, K] e4m3 row-major, whose ``.t()``
    is the column-major operand cuBLASLt wants (a stride change, never a copy)."""
    *lead, k = x.shape
    x2 = x.reshape(-1, k).contiguous()
    x8, scale = quant_per_tensor(x2)
    y = torch._scaled_mm(
        x8, weight.t(), scale_a=scale, scale_b=weight_scale, out_dtype=x.dtype
    )
    return y.reshape(*lead, weight.shape[0])


class Fp8DynamicLinear(BaseOP):
    """Per-tensor FP8 linear over the local shard; ``all_reduce`` adds the TP sum (row-parallel)."""

    def __init__(self, local_isize: int, local_osize: int, *, all_reduce: bool = False):
        assert local_isize % 16 == 0 and local_osize % 16 == 0, (
            local_isize,
            local_osize,
        )
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.weight = torch.empty(local_osize, local_isize, dtype=FP8)
        self.weight_scale = torch.empty((), dtype=torch.float32)
        self._comm = (
            DistributedCommunicator() if all_reduce and get_tp_info().size > 1 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = fp8_dynamic_linear(x, self.weight, self.weight_scale)
        if self._comm is not None:
            y = self._comm.all_reduce(y)
        return y


class Fp8DynamicColMerged(Fp8DynamicLinear):
    """Drop-in for ``LinearColParallelMerged``: one weight concatenating several projections
    along the output dim; the caller splits the output by the local sizes as before."""

    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        local_output_sizes: List[int] | None = None,
    ):
        tp = get_tp_info()
        if local_output_sizes is None:
            local_output_sizes = [div_even(size, tp.size) for size in output_sizes]
        self.output_sizes = list(output_sizes)
        self.local_output_sizes = list(local_output_sizes)
        super().__init__(input_size, sum(local_output_sizes))


class Fp8DynamicRowParallel(Fp8DynamicLinear):
    """Drop-in for ``LinearOProj`` / ``LinearRowParallel``: the input dim is sharded, the
    all-reduce runs after the local GEMM (each rank scales its own shard)."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__(
            div_even(input_size, get_tp_info().size), output_size, all_reduce=True
        )


__all__ = [
    "FP8",
    "E4M3_MAX",
    "Fp8DynamicColMerged",
    "Fp8DynamicLinear",
    "Fp8DynamicRowParallel",
    "fp8_dynamic_linear",
    "quant_per_tensor",
]
