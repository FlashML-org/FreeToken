"""GLM-5.3 manifold-constrained Hyper-Connections.

The released ``glm5_next`` equations match FreeToken's DeepSeek-V4 mHC kernels.
This module supplies the GLM parameter layout and deliberately reuses those fused
Sinkhorn/collapse/expand kernels instead of maintaining a second implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from freetoken.kernel.triton.dsv4.hc import hc_post_combine, hc_pre_combine
from freetoken.kernel.triton.dsv4.sinkhorn import hc_split_sinkhorn


class HyperConnection(nn.Module):
    """One attention or FFN mHC site over ``hc_mult`` residual streams."""

    def __init__(
        self,
        hidden_size: int,
        hc_mult: int = 4,
        norm_eps: float = 1e-5,
        sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.hc_eps = hc_eps
        mix_size = (2 + hc_mult) * hc_mult
        self.fn = nn.Parameter(
            torch.empty(mix_size, hc_mult * hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.base = nn.Parameter(torch.empty(mix_size, dtype=torch.float32), requires_grad=False)
        self.scale = nn.Parameter(torch.empty(3, dtype=torch.float32), requires_grad=False)

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
        pre, post, comb = hc_split_sinkhorn(
            mixes.reshape(-1, mixes.shape[-1]),
            self.scale,
            self.base,
            self.hc_mult,
            self.sinkhorn_iters,
            self.hc_eps,
        )
        tokens = flat.numel() // (self.hc_mult * self.hidden_size)
        collapsed = hc_pre_combine(
            flat.reshape(tokens, self.hc_mult, self.hidden_size),
            pre,
            hidden_streams.dtype,
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
        mixed = hc_post_combine(
            block_output.reshape(tokens, self.hidden_size),
            residual.reshape(tokens, self.hc_mult, self.hidden_size),
            post.reshape(tokens, self.hc_mult),
            comb.reshape(tokens, self.hc_mult, self.hc_mult),
        )
        return mixed.reshape(shape)


def collapse_head(hidden_streams: torch.Tensor) -> torch.Tensor:
    """GLM-5.3's final stream collapse is an unweighted mean."""
    return hidden_streams.mean(dim=-2)


__all__ = ["HyperConnection", "collapse_head"]
