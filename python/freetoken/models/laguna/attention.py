from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from freetoken.attention import AttentionSpec
from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.layers import BaseOP, LinearReplicated
from freetoken.layers.norm import RMSNorm
from freetoken.layers.rotary import get_rope
from freetoken.models.config import FullAttentionGroupConfig, SWAAttentionGroupConfig
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class LagunaAttention(BaseOP):
    """Laguna attention with per-layer head count, QK RMSNorm, and per-head gating."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self.layer_id = layer_id
        group = config.attention_group_for_layer(layer_id)
        self.is_swa = isinstance(group, SWAAttentionGroupConfig)
        if not isinstance(group, (FullAttentionGroupConfig, SWAAttentionGroupConfig)):
            raise ValueError(f"LagunaAttention does not support {group.kind!r} layers")

        # Per-layer head count (SWA 72 vs full 48 for S 2.1). Falls back to global.
        if getattr(config, "num_attention_heads_per_layer", None) is not None:
            heads = config.num_attention_heads_per_layer[layer_id]  # type: ignore[index]
        else:
            heads = config.num_qo_heads
        self.num_qo_heads_global = int(heads)
        self.head_dim = group.head_dim
        self.num_kv_heads_global = group.num_kv_heads

        # TP partition. Every projection below is sized from the *_global head counts
        # (Laguna keeps q/k/v/g/o split, and the per-layer head count varies), so this
        # model is TP=1 only; assert rather than silently replicate into wrong shapes.
        tp_size = get_tp_info().size
        assert tp_size == 1, (
            "Laguna does not support tensor parallelism: its per-layer head counts "
            f"(48 full / 72 SWA) are not sharded by this module (tp_size={tp_size})"
        )

        self.q_dim = self.num_qo_heads_global * self.head_dim
        self.kv_dim = self.num_kv_heads_global * self.head_dim
        # Laguna stores split q/k/v/g/o as separate linears (see configuration_laguna.py
        # base_model_tp_plan). We keep them split to avoid a fusion layer.
        hidden = config.hidden_size
        self.q_proj = LinearReplicated(hidden, self.q_dim, has_bias=False)
        self.k_proj = LinearReplicated(hidden, self.kv_dim, has_bias=False)
        self.v_proj = LinearReplicated(hidden, self.kv_dim, has_bias=False)
        self.o_proj = LinearReplicated(self.num_qo_heads_global * self.head_dim, hidden, has_bias=False)

        # Per-head gating (config.json gating "per-head").
        args = config.laguna_args
        gating = args.gating if args is not None else "per-head"
        self.gating_enabled = bool(gating)
        self.gate_per_head = gating == "per-head"
        if self.gating_enabled:
            if self.gate_per_head:
                g_out = self.num_qo_heads_global
            else:
                # per-element
                g_out = self.num_qo_heads_global * self.head_dim
            self.g_proj = LinearReplicated(hidden, g_out, has_bias=False)
        else:
            self.g_proj = None  # type: ignore[assignment]

        # QK RMSNorm (LagunaRMSNorm in HF -> vanilla RMSNorm with scale).
        eps = config.rms_norm_eps
        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)

        # Rope per group (full=YARN, swa=default).
        rotary_config = group.rotary_config
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
        self.attn_spec = AttentionSpec(
            sliding_window=group.sliding_window if self.is_swa else None,
            sm_scale=config.attn_sm_scale,
        )

    @nvtx_annotate("LAGUNA_MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        T = x.shape[0]

        # Split projections.
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)

        # Per-head QK RMSNorm, applied BEFORE RoPE (modeling_laguna.py). Normalize in
        # place on a [-1, head_dim] view of the projection output -- the flat and
        # [T, heads, head_dim] views share storage, so RoPE below sees the normed values.
        self.q_norm.forward_inplace(q.view(-1, self.head_dim))
        self.k_norm.forward_inplace(k.view(-1, self.head_dim))

        # RoPE rotates the first ``rotary_dim`` dims of each head (partial rotary: 64 of
        # 128 on full layers, 128 on SWA). The kernel takes [T, heads*head_dim] and
        # derives the head count from head_size, so it handles both widths unchanged.
        pos = ctx.batch.positions.reshape(-1)
        if pos.device != q.device or pos.dtype != torch.long:
            pos = pos.to(device=q.device, dtype=torch.long)
        q, k = self.rotary.forward(pos, q, k)

        o = ctx.attn_backend.forward(
            q.view(T, self.num_qo_heads_global, self.head_dim).contiguous(),
            k.contiguous(),
            v.contiguous(),
            self.layer_id,
            ctx.batch,
            attn_spec=self.attn_spec,
        )
        o = o.reshape(T, self.num_qo_heads_global * self.head_dim)

        # Softplus output gating, applied BEFORE o_proj (modeling_laguna.py:452-461).
        if self.gating_enabled:
            gate = F.softplus(self.g_proj.forward(x).float()).to(o.dtype)
            if self.gate_per_head:
                # [T, heads] broadcast across head_dim
                o = (o.view(T, self.num_qo_heads_global, self.head_dim) * gate.unsqueeze(-1)).reshape(
                    T, self.num_qo_heads_global * self.head_dim
                )
            else:
                o = o * gate
        return self.o_proj.forward(o)


__all__ = ["LagunaAttention"]
