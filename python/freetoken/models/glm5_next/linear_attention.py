"""Kimi Delta Attention used by GLM-5.3-Flash's linear layers."""

from __future__ import annotations

import torch

from freetoken.core import get_global_ctx
from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.layers import BaseOP, LinearReplicated

from .kda import kda_decode


class _DepthwiseConv1d(BaseOP):
    def __init__(self, dim: int, kernel: int) -> None:
        self.weight = torch.empty(dim, 1, kernel)


class _SigmoidGatedRMSNorm(BaseOP):
    def __init__(self, dim: int, eps: float) -> None:
        self.weight = torch.empty(dim)
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.fla import rms_norm_gated

        return rms_norm_gated(
            x=x,
            weight=self.weight,
            bias=None,
            z=gate,
            eps=self.eps,
            is_rms_norm=True,
            norm_before_gate=True,
            activation="sigmoid",
        )


class Glm5NextLinearAttention(BaseOP):
    """FreeToken stateful KDA op, preserving the released checkpoint key layout."""

    def __init__(self, config, layer_id: int) -> None:
        args = config.glm5_args
        self.layer_id = layer_id
        self.num_heads = args.linear_num_heads
        self.head_dim = args.linear_head_dim
        self.qkv_dim = self.num_heads * self.head_dim
        self.conv_dim = 3 * self.qkv_dim
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.gate_lower_bound = args.linear_lower_bound

        self.q_proj = LinearReplicated(config.hidden_size, self.qkv_dim, has_bias=False)
        self.k_proj = LinearReplicated(config.hidden_size, self.qkv_dim, has_bias=False)
        self.v_proj = LinearReplicated(config.hidden_size, self.qkv_dim, has_bias=False)
        self.conv1d = _DepthwiseConv1d(self.conv_dim, self.conv_kernel_size)
        self.b_proj = LinearReplicated(
            config.hidden_size, self.num_heads, has_bias=False
        )
        self.f_a_proj = LinearReplicated(
            config.hidden_size, self.head_dim, has_bias=False
        )
        self.f_b_proj = LinearReplicated(self.head_dim, self.qkv_dim, has_bias=False)
        self.g_a_proj = LinearReplicated(
            config.hidden_size, self.head_dim, has_bias=False
        )
        self.g_b_proj = LinearReplicated(self.head_dim, self.qkv_dim, has_bias=False)
        self.A_log = torch.empty(self.num_heads, dtype=torch.float32)
        self.dt_bias = torch.empty(self.qkv_dim, dtype=torch.float32)
        self.o_norm = _SigmoidGatedRMSNorm(self.head_dim, config.rms_norm_eps)
        self.o_proj = LinearReplicated(self.qkv_dim, config.hidden_size, has_bias=False)

    def _conv_weight(self) -> torch.Tensor:
        return self.conv1d.weight.squeeze(1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla

        raw = torch.cat(
            [
                self.q_proj.forward(hidden_states),
                self.k_proj.forward(hidden_states),
                self.v_proj.forward(hidden_states),
            ],
            dim=-1,
        )
        li = pool.local_index(self.layer_id)
        if batch.is_decode:
            mixed = causal_conv1d_decode(
                raw, pool.conv_states[li], self._conv_weight(), fla.cache_indices
            )
        else:
            mixed = (
                causal_conv1d_varlen(
                    raw.transpose(0, 1).contiguous(),
                    self._conv_weight(),
                    pool.conv_states[li],
                    fla.cu_seqlens,
                    fla.cache_indices,
                    fla.has_initial_state,
                )
                .transpose(0, 1)
                .contiguous()
            )

        total = hidden_states.shape[0]
        q, k, v = torch.split(mixed, [self.qkv_dim] * 3, dim=-1)
        shape = (1, total, self.num_heads, self.head_dim)
        q, k, v = q.view(shape), k.view(shape), v.view(shape)
        gate_a = self.f_b_proj.forward(self.f_a_proj.forward(hidden_states))
        beta = self.b_proj.forward(hidden_states)
        if fla.fresh_state_indices is not None:
            pool.recurrent_states[li].index_fill_(0, fla.fresh_state_indices, 0.0)
        core = kda_decode(
            q,
            k,
            v,
            gate_a,
            beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            state_source=pool.recurrent_states[li],
            indices=fla.cache_indices,
            cu_seqlens=fla.cu_seqlens,
            gate_lower_bound=self.gate_lower_bound,
        )
        gate = self.g_b_proj.forward(self.g_a_proj.forward(hidden_states))
        out = self.o_norm.forward(
            core.reshape(-1, self.head_dim), gate.reshape(-1, self.head_dim)
        )
        return self.o_proj.forward(out.reshape(total, self.qkv_dim))


__all__ = ["Glm5NextLinearAttention"]
