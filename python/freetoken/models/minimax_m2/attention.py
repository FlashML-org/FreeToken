from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.layers import BaseOP, LinearOProj, LinearQKVMerged, RMSNorm
from freetoken.layers.rotary import get_rope
from freetoken.utils import div_even, nvtx_annotate

if TYPE_CHECKING:
    import torch

    from freetoken.models.config import ModelConfig


class MiniMaxM2Attention(BaseOP):
    """MiniMax-M2 attention.

    Two differences from the Qwen3 MoE attention:
    - qk-norm is applied "per layer", i.e. a single RMSNorm over the *whole* q
      projection (num_qo_heads * head_dim) and the *whole* k projection
      (num_kv_heads * head_dim), before splitting into heads. This is exact only
      when the projections are not sharded (TP=1, the single-device offload case).
    - RoPE is partial (rotary_dim < head_dim): only the first ``rotary_dim`` dims
      of each head are rotated.
    """

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        head_dim = config.head_dim
        self.layer_id = layer_id
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(config.num_qo_heads, tp_size)
        self.num_kv_heads = div_even(config.num_kv_heads, tp_size, allow_replicate=True)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        self.head_dim = head_dim
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=False,
            quant_config=config.quant,
            prefix=f"{prefix}.qkv_proj",
        )
        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.qo_attn_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.kv_attn_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=config.rotary_config.rotary_dim,
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,
            rope_scaling=(
                tuple(config.rotary_config.scaling.items())
                if config.rotary_config.scaling
                else None
            ),
        )
        self.o_proj = LinearOProj(
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
            quant_config=config.quant,
            prefix=f"{prefix}.o_proj",
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        qkv = self.qkv_proj.forward(x)
        del x
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        # split() returns strided views into the merged qkv buffer; the per-layer qk-norm
        # (a single RMSNorm over the full q/k vector) and rotary produce contiguous q/k,
        # but v must be made contiguous explicitly or the KV-cache store kernel (which
        # requires stride#0 == kv_attn_dim) rejects it.
        q = self.q_norm.forward(q.contiguous()) if self.q_norm is not None else q.contiguous()
        k = self.k_norm.forward(k.contiguous()) if self.k_norm is not None else k.contiguous()
        v = v.contiguous()
        q, k = self.rotary.forward(ctx.batch.positions, q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        return self.o_proj.forward(o.view(-1, self.qo_attn_dim))


__all__ = ["MiniMaxM2Attention"]
