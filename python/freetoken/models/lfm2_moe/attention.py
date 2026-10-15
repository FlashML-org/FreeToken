"""LFM2-MoE full attention: GQA with per-head q/k RMSNorm and full-width NeoX rope.

``q = q_layernorm(q_proj(x)); k = k_layernorm(k_proj(x)); v = v_proj(x); (q, k) = rope(q, k)``
then paged attention and ``out_proj``. The norms are PLAIN RMSNorm (HF's ``Lfm2MoeRMSNorm``
scales by the raw weight -- not Gemma's ``1 + weight``). Scaling stays the backend default
``1/sqrt(head_dim)``, matching HF's ``self.scaling = head_dim**-0.5``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearReplicated, RMSNorm
from freetoken.layers.rotary import get_rope
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Lfm2Attention(BaseOP):
    """GQA attention with per-head q/k RMSNorm (all projections replicated: TP=1 milestone)."""

    def __init__(self, config: "ModelConfig", layer_id: int, *, prefix: str = "") -> None:
        head_dim = config.head_dim
        self.layer_id = layer_id
        self.num_q = config.num_qo_heads
        self.num_kv = config.num_kv_heads
        self.head_dim = head_dim
        self.qo_attn_dim = self.num_q * head_dim
        self.kv_attn_dim = self.num_kv * head_dim

        # No fused qkv: the three checkpoint tensors are separate, and fusing them would only
        # save a kernel launch while forcing row-concatenation of the fp8 block scales.
        self.q_proj = LinearReplicated(
            config.hidden_size, self.qo_attn_dim, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.q_proj",
        )
        self.k_proj = LinearReplicated(
            config.hidden_size, self.kv_attn_dim, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.k_proj",
        )
        self.v_proj = LinearReplicated(
            config.hidden_size, self.kv_attn_dim, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.v_proj",
        )
        self.out_proj = LinearReplicated(
            self.qo_attn_dim, config.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.out_proj",
        )
        self.q_layernorm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = RMSNorm(head_dim, eps=config.rms_norm_eps)
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

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        q = self.q_proj.forward(x).view(-1, self.num_q, self.head_dim)
        k = self.k_proj.forward(x).view(-1, self.num_kv, self.head_dim)
        v = self.v_proj.forward(x)
        self.q_layernorm.forward_inplace(q)
        self.k_layernorm.forward_inplace(k)
        q, k = self.rotary.forward(
            ctx.batch.get_attn_positions(), q.view(-1, self.qo_attn_dim), k.view(-1, self.kv_attn_dim)
        )
        o = ctx.attn_backend.forward(q.view(-1, self.num_q, self.head_dim), k, v, self.layer_id, ctx.batch)
        return self.out_proj.forward(o.view(-1, self.qo_attn_dim))


__all__ = ["Lfm2Attention"]
