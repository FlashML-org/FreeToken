from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.attention import AttentionSpec
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, GemmaRMSNorm, LinearQKVMerged, LinearReplicated
from freetoken.layers.rotary import get_rope
from freetoken.models.config import FullAttentionGroupConfig, SWAAttentionGroupConfig
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Gemma4Attention(BaseOP):
    """Gemma 4 attention for one full-context or SWA layer.

    On E2B / E4B the KV-shared tail projects q only and attends over the k, v of the last
    earlier layer of its attention type. That layer hands its rows over in ``shared_kv``, a
    dict living for one forward, so a captured graph records the same tensors. Each shared
    layer still stores them in its own KV slab; reading the source's slab would save that memory.
    """

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self.layer_id = layer_id
        group = config.attention_group_for_layer(layer_id)
        self.is_swa = isinstance(group, SWAAttentionGroupConfig)
        if not isinstance(group, (FullAttentionGroupConfig, SWAAttentionGroupConfig)):
            raise ValueError(f"Gemma4Attention does not support {group.kind!r} layers")
        rotary_config = group.rotary_config
        self.head_dim = group.head_dim
        self.num_kv_heads = group.num_kv_heads
        self.num_qo_heads = config.num_qo_heads
        self.k_eq_v = isinstance(group, FullAttentionGroupConfig) and group.k_eq_v
        e_args = config.gemma4_e_args
        self.kv_shared = e_args is not None and e_args.is_kv_shared(layer_id)
        self.kv_source = e_args is not None and e_args.is_kv_source(layer_id, group.layer_ids)
        self._group_name = group.name

        self.q_dim = self.num_qo_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        if self.kv_shared:
            self.q_proj = LinearReplicated(
                config.hidden_size, self.q_dim, has_bias=False,
                quant_config=config.quant, prefix=f"{prefix}.q_proj",
            )
        else:
            self.qkv_proj = LinearQKVMerged(
                config.hidden_size,
                self.head_dim,
                self.num_qo_heads,
                self.num_kv_heads,
                has_bias=False,
                quant_config=config.quant,
                prefix=f"{prefix}.qkv_proj",
            )
            self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.v_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)
        self.o_proj = LinearReplicated(
            self.q_dim, config.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.o_proj",
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_spec = AttentionSpec(
            sliding_window=group.sliding_window if self.is_swa else None,
            sm_scale=config.attn_sm_scale,
            bidirectional_mm_blocks=self.is_swa and group.bidirectional_mm_blocks,
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
    def forward(self, x: torch.Tensor, shared_kv: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        ctx = get_global_ctx()
        positions = ctx.batch.positions
        T = x.shape[0]

        if self.kv_shared:
            q = self.q_proj.forward(x).view(T, self.num_qo_heads, self.head_dim)
            q = self.q_norm.forward(q)
            # the rope kernels rotate a key along with the query: give them one head of zeros
            q, _ = self._apply_rope(positions, q, q.new_zeros(T, 1, self.head_dim))
            k, v = shared_kv[self._group_name]
        else:
            qkv = self.qkv_proj.forward(x)
            q_lin, k_lin, v_lin = qkv.split(
                (self.q_dim, self.kv_dim, self.kv_dim),
                dim=-1,
            )
            del qkv
            q = q_lin.view(T, self.num_qo_heads, self.head_dim)
            k = k_lin.view(T, self.num_kv_heads, self.head_dim)
            v = v_lin.view(T, self.num_kv_heads, self.head_dim)

            q = self.q_norm.forward(q)
            k = self.k_norm.forward(k)
            v = self.v_norm.forward(v)

            q, k = self._apply_rope(positions, q, k)

            k = k.reshape(T, self.num_kv_heads * self.head_dim)
            v = v.reshape(T, self.num_kv_heads * self.head_dim)
            if self.kv_source:
                shared_kv[self._group_name] = (k, v)
        o = ctx.attn_backend.forward(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            self.layer_id,
            ctx.batch,
            attn_spec=self.attn_spec,
        )
        o = o.reshape(T, self.num_qo_heads * self.head_dim)
        return self.o_proj.forward(o)


__all__ = ["Gemma4Attention"]
