from __future__ import annotations

import torch

from freetoken.attention import AttentionSpec
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearColParallelMerged, LinearReplicated, RMSNorm
from freetoken.layers.rotary import get_rope
from freetoken.models.config import ModelConfig, SWAAttentionGroupConfig
from freetoken.utils import nvtx_annotate


class RicursiveAttention(BaseOP):
    """Gated GQA for one sliding-window or full-attention (NoPE) layer.

        q, k = split(qk_proj(x));  v = v_proj(lerp(value_residual, x, blend))
        q = q_norm(q);  k = k_norm(k)            # per-head RMSNorm
        q, k = rope(q, k)                        # sliding layers only
        out = o_proj(attn(q, k, v) * sigmoid(g_proj(x)))

    ``value_res_blend`` is the reference's ``sigmoid(value_res_lambda)``, a constant the loader folds.
    """

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self.layer_id = layer_id
        group = config.attention_group_for_layer(layer_id)
        self.num_qo_heads = config.num_qo_heads
        self.head_dim = config.head_dim
        self.q_dim = config.num_qo_heads * config.head_dim
        self.kv_dim = config.num_kv_heads * config.head_dim
        H = config.hidden_size
        self.qk_proj = LinearColParallelMerged(
            H, [self.q_dim, self.kv_dim], has_bias=False, quant_config=config.quant, prefix=f"{prefix}.qk_proj",
        )
        self.v_proj = LinearReplicated(H, self.kv_dim, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.v_proj")
        self.g_proj = LinearReplicated(H, self.q_dim, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.g_proj")
        self.o_proj = LinearReplicated(self.q_dim, H, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.o_proj")
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.value_res_blend = torch.empty(1)
        self.attn_spec = AttentionSpec(
            sliding_window=group.sliding_window if isinstance(group, SWAAttentionGroupConfig) else None,
        )
        rope = group.rotary_config
        self.rotary = get_rope(
            head_dim=self.head_dim, rotary_dim=rope.rotary_dim, max_position=rope.max_position, base=rope.base,
        ) if rope.rotary_dim else None

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor, value_residual: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        q, k = self.qk_proj.forward(x).split((self.q_dim, self.kv_dim), dim=-1)
        v = self.v_proj.forward(torch.lerp(value_residual, x, self.value_res_blend))
        # The reference's gate_full_precision needs no cast: bf16 sigmoid already computes in fp32 and rounds once.
        gate = torch.sigmoid(self.g_proj.forward(x))
        # Flatten heads for FlashInfer's 2D CUDA norm.
        q = self.q_norm.forward(q.reshape(-1, self.head_dim)).reshape(-1, self.q_dim)
        k = self.k_norm.forward(k.reshape(-1, self.head_dim)).reshape(-1, self.kv_dim)
        if self.rotary is not None:
            q, k = self.rotary.forward(ctx.batch.get_attn_positions(), q, k)
        out = ctx.attn_backend.forward(
            q.view(-1, self.num_qo_heads, self.head_dim), k, v, self.layer_id, ctx.batch, attn_spec=self.attn_spec,
        )
        return self.o_proj.forward(out.reshape(-1, self.q_dim) * gate)
