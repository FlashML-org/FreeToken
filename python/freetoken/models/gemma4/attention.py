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

    Supports the E-series (E2B/E4B) KV-layer-sharing scheme: the last
    ``num_kv_shared_layers`` layers carry no K/V projections and reuse the K/V
    states produced by the last non-shared layer of the *same* attention type
    (sliding vs full). Sharing is wired via a per-forward stash (``ctx`` buffers
    keyed by attention-group name): a "source" layer writes its post-rope K and
    post-norm V into the stash, and each shared layer reads them and feeds them
    to its own KV cache slot. The stash is a stable, model-owned buffer so it is
    captured/replayed correctly inside the decode CUDA graph.
    """

    def __init__(self, config: ModelConfig, layer_id: int):
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

        # --- E-series KV-layer sharing bookkeeping (no-op when num_kv_shared_layers == 0) ---
        self._group_name = group.name  # stash key: shared + source layers of a type must match
        self.is_kv_shared = config.is_kv_shared_layer(layer_id)
        self.is_kv_source = config.is_kv_source_layer(layer_id)
        # Filled in by Gemma4Model after build: the model-owned {group_name: (k_buf, v_buf)}
        # stash and the live token count. None on non-E-series models.
        self._kv_stash: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None

        self.q_dim = self.num_qo_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        if self.is_kv_shared:
            # Shared layers project Q only (K/V come from the stash of their type).
            self.q_proj = LinearReplicated(config.hidden_size, self.q_dim, has_bias=False)
        else:
            self.qkv_proj = LinearQKVMerged(
                config.hidden_size,
                self.head_dim,
                self.num_qo_heads,
                self.num_kv_heads,
                has_bias=False,
            )
            self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.v_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)
        self.o_proj = LinearReplicated(self.q_dim, config.hidden_size, has_bias=False)
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_spec = AttentionSpec(
            sliding_window=group.sliding_window if self.is_swa else None,
            sm_scale=config.attn_sm_scale,
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

    def _apply_rope_q(self, positions: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        positions = positions.reshape(-1)
        if positions.device != q.device or positions.dtype != torch.long:
            positions = positions.to(device=q.device, dtype=torch.long)
        q_view = q.contiguous().view(q.shape[0], -1)
        # rotary.forward rotates q and k together; pass q as both when we only need q.
        self.rotary.forward(positions, q_view, q_view)
        return q_view.view_as(q)

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
        positions = ctx.batch.positions
        T = x.shape[0]

        if self.is_kv_shared:
            q_lin = self.q_proj.forward(x)
            q = q_lin.view(T, self.num_qo_heads, self.head_dim)
            q = self.q_norm.forward(q)
            q = self._apply_rope_q(positions, q)
            k_buf, v_buf = self._kv_stash[self._group_name]
            k = k_buf[:T]
            v = v_buf[:T]
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
            # A "source" layer publishes its full K/V for the shared layers of its type.
            if self.is_kv_source and self._kv_stash is not None:
                k_buf, v_buf = self._kv_stash[self._group_name]
                k_buf[:T].copy_(k)
                v_buf[:T].copy_(v)

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
