from __future__ import annotations

import torch

from freetoken.attention import AttentionSpec
from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.layers import BaseOP, LinearOProj, LinearQKVMerged, RMSNorm
from freetoken.layers.rotary import get_rope
from freetoken.models.config import ModelConfig, SWAAttentionGroupConfig
from freetoken.utils import div_even, nvtx_annotate


class FlintAttention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self.layer_id = layer_id
        group = config.attention_group_for_layer(layer_id)
        self.num_qo_heads = div_even(config.num_qo_heads, get_tp_info().size)
        self.num_kv_heads = div_even(config.num_kv_heads, get_tp_info().size, allow_replicate=True)
        self.head_dim = config.head_dim
        self.q_dim = self.num_qo_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_proj = LinearQKVMerged(
            config.hidden_size, self.head_dim, config.num_qo_heads, config.num_kv_heads,
            has_bias=False, quant_config=config.quant, prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = LinearOProj(
            config.num_qo_heads * self.head_dim, config.hidden_size,
            has_bias=False, quant_config=config.quant, prefix=f"{prefix}.o_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.softmax_offset = torch.empty(self.num_qo_heads, dtype=torch.float32)
        self.sliding_window = group.sliding_window if isinstance(group, SWAAttentionGroupConfig) else None
        rope = group.rotary_config
        self.rotary = get_rope(
            head_dim=self.head_dim, rotary_dim=rope.rotary_dim,
            max_position=rope.max_position, base=rope.base, rope_scaling=None,
        ) if rope.rotary_dim else None

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        qkv = self.qkv_proj.forward(x)
        q, k, v = qkv.split((self.q_dim, self.kv_dim, self.kv_dim), dim=-1)
        # Flatten heads for FlashInfer's 2D CUDA norm; normalizing the concatenated
        # heads would silently change both the attention and the MoE routes.
        q = self.q_norm.forward(q.reshape(-1, self.head_dim)).reshape(-1, self.q_dim)
        k = self.k_norm.forward(k.reshape(-1, self.head_dim)).reshape(-1, self.kv_dim)
        if self.rotary is not None:
            q, k = self.rotary.forward(ctx.batch.get_attn_positions(), q, k)
        out = ctx.attn_backend.forward(
            q.view(-1, self.num_qo_heads, self.head_dim), k.contiguous(), v.contiguous(),
            self.layer_id, ctx.batch,
            attn_spec=AttentionSpec(sliding_window=self.sliding_window, sinks=self.softmax_offset),
        )
        return self.o_proj.forward(out.reshape(-1, self.q_dim))
