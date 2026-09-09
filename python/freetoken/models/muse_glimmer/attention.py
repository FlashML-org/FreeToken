from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.attention import AttentionSpec
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, GemmaRMSNorm, LinearColParallelMerged, LinearReplicated
from freetoken.layers.rotary import get_rope
from freetoken.models.config import SWAAttentionGroupConfig
from freetoken.utils import nvtx_annotate


if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class MuseGlimmerAttention(BaseOP):
    """Gated GQA for one sliding-window or full-attention (NoPE) layer.

        q, k, v, gate = split(qkvg_proj(x))
        q = qk_norm(q); k = qk_norm(k)          # weightless per-head RMSNorm
        q, k = rope(q, k)                        # sliding layers only; full layers are NoPE
        attn = paged_attention(q, k, v)          # sm_scale carries qk_scale_factor/sqrt(d)
        out = o_proj(attn * sigmoid(gate))

    The reference multiplies q by ``qk_scale_factor`` before rope; rope is a rotation, so
    the factor commutes and is folded into ``AttentionSpec.sm_scale`` instead (parse_config
    sets ``attn_sm_scale = qk_scale_factor / sqrt(head_dim)``). The gate is computed from
    the same normed layer input as q/k/v, so it rides the fused projection.
    """

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self.layer_id = layer_id
        group = config.attention_group_for_layer(layer_id)
        self.is_swa = isinstance(group, SWAAttentionGroupConfig)
        head_dim = group.head_dim
        self.head_dim = head_dim
        self.num_q = config.num_qo_heads
        self.num_kv = group.num_kv_heads
        self.qo_attn_dim = self.num_q * head_dim
        self.kv_attn_dim = self.num_kv * head_dim

        self._qkvg_split = [self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim, self.qo_attn_dim]
        self.qkvg_proj = LinearColParallelMerged(
            config.hidden_size, self._qkvg_split, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.qkvg_proj",
        )
        self.o_proj = LinearReplicated(
            self.qo_attn_dim, config.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.o_proj",
        )
        # Weightless: the checkpoint carries no q/k norm weights; the runtime ones vector
        # is intentionally not part of state_dict.
        self.qk_norm = GemmaRMSNorm(head_dim, eps=config.rms_norm_eps, with_scale=False)
        self.attn_spec = AttentionSpec(
            sliding_window=group.sliding_window if self.is_swa else None,
            sm_scale=config.attn_sm_scale,
        )
        # base == 0.0 marks the NoPE full-attention layers; RotaryEmbedding must never
        # see it (0 ** x in the inv_freq computation).
        rotary_config = group.rotary_config
        self.rotary = (
            get_rope(
                head_dim=head_dim,
                rotary_dim=rotary_config.rotary_dim,
                max_position=rotary_config.max_position,
                base=rotary_config.base,
            )
            if rotary_config.base > 0.0
            else None
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        T = x.shape[0]

        qkvg = self.qkvg_proj.forward(x)
        q, k, v, gate = torch.split(qkvg, self._qkvg_split, dim=-1)
        del qkvg
        q = self.qk_norm.forward(q.contiguous().view(T, self.num_q, self.head_dim))
        k = self.qk_norm.forward(k.contiguous().view(T, self.num_kv, self.head_dim))
        v = v.contiguous()  # the split view has the qkvg row stride; the KV store needs contiguous

        q = q.reshape(T, self.qo_attn_dim)
        k = k.reshape(T, self.kv_attn_dim)
        if self.rotary is not None:
            q, k = self.rotary.forward(ctx.batch.positions, q, k)

        o = ctx.attn_backend.forward(
            q.view(T, self.num_q, self.head_dim),
            k,
            v,
            self.layer_id,
            ctx.batch,
            attn_spec=self.attn_spec,
        )
        gated = o.reshape(T, self.qo_attn_dim) * torch.sigmoid(gate)
        return self.o_proj.forward(gated)


__all__ = ["MuseGlimmerAttention"]
