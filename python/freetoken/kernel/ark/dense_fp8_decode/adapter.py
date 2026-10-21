"""M=1 FreeToken block_fp8_linear, including bias and leading dimensions."""

import torch
from freetoken.kernel.ark.common.trace import record
from freetoken.kernel.ark.common.policy import allowed, key
from freetoken.kernel.ark.common.contracts import gfx1201
from freetoken.kernel.ark.common.native import load

NW = 1


def block_fp8_linear(x, weight, weight_scale, bias=None):
    k = x.shape[-1]
    if not (
        weight.ndim == 2
        and allowed("dense_fp8_decode", key(x.numel() // k, weight.shape[0], k))
        and gfx1201(x)
        and x.dtype == torch.bfloat16
        and x.numel() == k
        and x.is_contiguous()
        and weight.dtype == torch.float8_e4m3fn
        and weight.ndim == 2
        and weight.is_contiguous()
        and weight.shape[1] == k
        and k % 128 == 0
        and weight_scale.dtype == torch.bfloat16
        and weight_scale.ndim == 2
        and weight_scale.is_contiguous()
        and weight_scale.shape[0] * 128 >= weight.shape[0]
        and weight_scale.shape[1] >= k // 128
        and weight.device == x.device
        and weight_scale.device == x.device
    ):
        record("dense_fp8_decode", key(x.numel() // k, weight.shape[0], k), "freetoken")
        from freetoken.kernel.triton.fp8_block_linear import block_fp8_linear as baseline

        return baseline(x, weight, weight_scale, bias)
    record("dense_fp8_decode", key(x.numel() // k, weight.shape[0], k), "ark")
    load()
    out = torch.ops.freetoken_ark.dense_decode(x.reshape(1, k), weight, weight_scale, NW).reshape(*x.shape[:-1], weight.shape[0])
    return out + bias.to(out.dtype) if bias is not None else out
