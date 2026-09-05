from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from freetoken.attention import AttentionSpec
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearReplicated, RMSNorm
from freetoken.layers.rotary import get_rope
from freetoken.models.config import FullAttentionGroupConfig, SWAAttentionGroupConfig
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class LagunaAttention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.layer_id = layer_id
        group = config.attention_group_for_layer(layer_id)
        if not isinstance(group, (FullAttentionGroupConfig, SWAAttentionGroupConfig)):
            raise ValueError(f"LagunaAttention does not support {group.kind!r} layers")

        rotary_config = group.rotary_config
        self.head_dim = group.head_dim
        self.num_kv_heads = group.num_kv_heads
        self.num_qo_heads = config.qo_heads(layer_id)

        self.q_dim = self.num_qo_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim

        # Separate q/k/v projections (not LinearQKVMerged): laguna GGUFs quantize
        # attn_v at a different ggml type than q/k on some layers (XS Q4_K_M), and
        # packed rows of different types cannot be fused into one buffer.
        self.q_proj = LinearReplicated(config.hidden_size, self.q_dim, has_bias=False)
        self.k_proj = LinearReplicated(config.hidden_size, self.kv_dim, has_bias=False)
        self.v_proj = LinearReplicated(config.hidden_size, self.kv_dim, has_bias=False)
        self.gate_proj = LinearReplicated(config.hidden_size, self.num_qo_heads, has_bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = LinearReplicated(self.q_dim, config.hidden_size, has_bias=False)
        self.attn_spec = AttentionSpec(
            sliding_window=group.sliding_window if isinstance(group, SWAAttentionGroupConfig) else None
        )

        self.rotary = get_rope(
            head_dim=self.head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=(
                tuple(rotary_config.scaling.items())
                if rotary_config.scaling
                else None
            ),
        )

    def _apply_rope(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = positions.reshape(-1)
        if positions.device != q.device or positions.dtype != torch.long:
            positions = positions.to(device=q.device, dtype=torch.long)
        q_view = q.contiguous().view(q.shape[0], -1)
        k_view = k.contiguous().view(k.shape[0], -1)
        self.rotary.forward(positions, q_view, k_view)
        return q_view.view_as(q), k_view.view_as(k)

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        T = x.shape[0]

        gate = F.softplus(self.gate_proj.forward(x).float())
        q_lin = self.q_proj.forward(x)
        k_lin = self.k_proj.forward(x)
        v_lin = self.v_proj.forward(x)

        q = q_lin.view(T, self.num_qo_heads, self.head_dim)
        k = k_lin.view(T, self.num_kv_heads, self.head_dim)
        v = v_lin.view(T, self.num_kv_heads, self.head_dim)

        self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))

        q, k = self._apply_rope(ctx.batch.positions, q, k)

        k = k.reshape(T, self.kv_dim)
        v = v.reshape(T, self.kv_dim)

        o = ctx.attn_backend.forward(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            self.layer_id,
            ctx.batch,
            attn_spec=self.attn_spec,
        )
        o = o.view(T, self.num_qo_heads, self.head_dim)
        o = o * gate.unsqueeze(-1).to(o.dtype)
        return self.o_proj.forward(o.reshape(T, self.q_dim))


__all__ = ["LagunaAttention"]
