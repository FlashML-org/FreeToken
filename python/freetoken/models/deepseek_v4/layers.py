"""Shared DSV4 primitives: weight-only linears, RMSNorm, and the static window /
compressed top-k index builders."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.kernel.triton.dsv4.fp8_linear import block_fp8_linear
from freetoken.kernel.triton.dsv4.norm import rms_norm

FP8 = torch.float8_e4m3fn


# Decode-sized fp32 GEMVs go through hipBLASLt with a 64x64 macro tile on gfx1201:
# ONE workgroup walks the whole K loop, so hc_pre's [M<=4, 16384] x [24, 16384]^T
# costs ~965 us per call (~1 ms x 87 calls = 82 ms of every decode step) where the
# 1.5 MiB read is worth ~15 us. Below this M the broadcast-multiply + row reduce is
# 14-30 us and numerically an fp32 sum of the same products; above it the GEMM's
# fixed tile cost is amortized and the temp (M x N x K fp32) would not be.
SMALL_M_LINEAR_MAX = 16


def linear_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``F.linear`` for fp32 ``x`` [.., K] and ``weight`` [N, K] that is fast at decode M."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if x2.shape[0] <= SMALL_M_LINEAR_MAX:
        out = (x2.unsqueeze(1) * weight.unsqueeze(0)).sum(-1)
        return out.view(*lead, weight.shape[0])
    return F.linear(x, weight)


class Linear(nn.Module):
    """BF16 / FP32 / block-scaled-FP8 linear (weight-only).

    ``kind="fp8"``: weight float8_e4m3fn + 128x128 e8m0 ``scale`` (dequant in the
    Triton GEMM). ``kind="fp32"``/``"bf16"``: plain ``F.linear``.
    """

    def __init__(self, in_features, out_features, bias=False, kind="fp8"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kind = kind
        if kind == "fp8":
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=FP8), requires_grad=False)
            self.scale = nn.Parameter(
                torch.empty(out_features // 128, in_features // 128, dtype=torch.float8_e8m0fnu),
                requires_grad=False,
            )
        else:
            dtype = torch.float32 if kind == "fp32" else torch.bfloat16
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype), requires_grad=False)
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "fp8":
            return block_fp8_linear(x, self.weight, self.scale, self.bias)
        return F.linear(x, self.weight.to(x.dtype), self.bias)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor):
        return rms_norm(x, self.weight, self.eps)


def get_window_topk_idxs(window_size, bsz, seqlen, start_pos):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat([torch.arange(start_pos + 1, window_size), torch.arange(0, start_pos + 1)], dim=0)
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


def get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)
