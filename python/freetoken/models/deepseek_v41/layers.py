"""Resident V4.1 projections preserve block-32 FP8 storage and activation quantization."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, kind="fp8"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kind = kind
        if kind not in {"fp8", "fp32", "bf16"}:
            raise ValueError(f"Unsupported linear storage kind: {kind}")
        if kind == "fp8":
            if in_features % 32 or out_features % 32:
                raise ValueError("V4.1 FP8 dimensions must be divisible by 32")
            dtype = torch.float8_e4m3fn
            self.scale = nn.Parameter(torch.empty(out_features // 32, in_features // 32,
                                                  dtype=torch.float8_e8m0fnu), requires_grad=False)
        else:
            dtype = torch.float32 if kind == "fp32" else torch.bfloat16
            self.register_parameter("scale", None)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype), requires_grad=False) if bias else None

    def forward(self, x):
        if self.kind == "fp8":
            from freetoken.kernel.triton.dsv41.quant import block_fp8_linear

            return block_fp8_linear(x, self.weight, self.scale, self.bias, block_size=32)
        return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-20):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        if x.is_cuda:
            from freetoken.kernel.triton.dsv4.norm import rms_norm

            return rms_norm(x, self.weight, self.eps)
        value = x.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
                * self.weight.float()).to(x.dtype)


class OutputHead(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.bfloat16), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            return F.linear(x.float(), self.weight.float())
        from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32

        # The reference returns FP32 logits; keep its BF16 checkpoint head resident.
        flat = x.reshape(-1, x.shape[-1])
        logits = torch.stack([bf16_linear_fp32(row, self.weight) for row in flat])
        return logits.reshape(*x.shape[:-1], self.weight.shape[0])


__all__ = ["Linear", "RMSNorm", "OutputHead"]
