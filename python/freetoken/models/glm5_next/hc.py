"""GLM-5.3 manifold-constrained Hyper-Connections.

The released ``glm5_next`` equations match FreeToken's DeepSeek-V4 mHC kernels.
This module supplies the GLM parameter layout and deliberately reuses those fused
Sinkhorn/collapse/expand kernels instead of maintaining a second implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.layers import BaseOP


class HyperConnection(BaseOP):
    """One attention or FFN mHC site over ``hc_mult`` residual streams."""

    def __init__(
        self,
        hidden_size: int,
        hc_mult: int = 4,
        norm_eps: float = 1e-5,
        sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
    ) -> None:
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.hc_eps = hc_eps
        mix_size = (2 + hc_mult) * hc_mult
        self.fn = torch.empty(mix_size, hc_mult * hidden_size, dtype=torch.float32)
        self.base = torch.empty(mix_size, dtype=torch.float32)
        self.scale = torch.empty(3, dtype=torch.float32)

    def to(self, device) -> HyperConnection:
        """Small compatibility helper for direct module tests; engine loading replaces
        BaseOP tensors through its state-dict path."""
        self.fn = self.fn.to(device)
        self.base = self.base.to(device)
        self.scale = self.scale.to(device)
        return self

    def mix(self, hidden_streams: torch.Tensor):
        """Return collapsed sublayer input plus the placement/mixing coefficients."""
        if hidden_streams.shape[-2:] != (self.hc_mult, self.hidden_size):
            raise ValueError(
                "expected trailing hidden-stream shape "
                f"({self.hc_mult}, {self.hidden_size}), got {tuple(hidden_streams.shape[-2:])}"
            )
        shape = hidden_streams.shape
        flat = hidden_streams.flatten(-2).float()
        inv_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, self.fn) * inv_rms
        pre_w, post_w, comb_w = mixes.split(
            [self.hc_mult, self.hc_mult, self.hc_mult * self.hc_mult], dim=-1
        )
        pre_b, post_b, comb_b = self.base.split(
            [self.hc_mult, self.hc_mult, self.hc_mult * self.hc_mult]
        )
        pre = torch.sigmoid(pre_w * self.scale[0] + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * self.scale[1] + post_b)
        comb = torch.softmax(
            comb_w.reshape(-1, self.hc_mult, self.hc_mult) * self.scale[2]
            + comb_b.view(self.hc_mult, self.hc_mult),
            dim=-1,
        )
        comb = comb + self.hc_eps
        comb = comb / (comb.sum(-2, keepdim=True) + self.hc_eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(-2, keepdim=True) + self.hc_eps)
        collapsed = (
            (pre.unsqueeze(-1) * hidden_streams.float())
            .sum(-2)
            .to(hidden_streams.dtype)
        )
        return (
            collapsed.reshape(*shape[:-2], self.hidden_size),
            post.reshape(*shape[:-2], self.hc_mult),
            comb.reshape(*shape[:-2], self.hc_mult, self.hc_mult),
        )

    def combine(
        self,
        residual: torch.Tensor,
        block_output: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Place a sublayer output back into and mix the residual streams."""
        shape = residual.shape
        tokens = residual.numel() // (self.hc_mult * self.hidden_size)
        block = block_output.reshape(tokens, self.hidden_size).float()
        streams = residual.reshape(tokens, self.hc_mult, self.hidden_size).float()
        post = post.reshape(tokens, self.hc_mult).float()
        comb = comb.reshape(tokens, self.hc_mult, self.hc_mult).float()
        mixed = post.unsqueeze(-1) * block.unsqueeze(-2)
        mixed = mixed + torch.matmul(comb.transpose(-1, -2), streams)
        mixed = mixed.to(residual.dtype)
        return mixed.reshape(shape)


def collapse_head(hidden_streams: torch.Tensor) -> torch.Tensor:
    """GLM-5.3's final stream collapse is an unweighted mean."""
    return hidden_streams.mean(dim=-2)


__all__ = ["HyperConnection", "collapse_head"]
