"""bf16-weight linear with fp32 math and fp32 output (DeepSeek-V4).

Several DSV4 decode ops need fp32 *math* on bf16 weights (the compressor wkv/wgate
gated pool, the MoE router) for numerical stability. Doing it as
``F.linear(x.float(), w.float())`` materializes an fp32 copy of the weight in HBM
every step (read bf16 + write fp32 + re-read fp32) and runs a heavier fp32 GEMM.
Decode is M==1, a GEMV that upcasts the bf16 weight in registers instead; M>1
(prefill) falls back to F.linear.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.kernel.triton.bf16_gemv import bf16_gemv


def bf16_linear_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``out = x @ weight.T`` in fp32, reading ``weight`` (bf16) straight from HBM.

    ``x``: ``[..., K]`` (leading dims collapse to M). ``weight``: ``[N, K]`` bf16.
    Returns ``[..., N]`` fp32. Bit-exact to ``F.linear(x.float(), weight.float())``
    up to fp32 accumulation order; for M>1 it *is* that call (prefill)."""
    M = x.numel() // x.shape[-1]
    if M != 1:
        return F.linear(x.float(), weight.float())
    return bf16_gemv(x, weight, torch.float32)


__all__ = ["bf16_linear_fp32"]
