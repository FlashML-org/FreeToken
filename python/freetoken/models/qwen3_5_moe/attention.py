from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.layers import BaseOP, GemmaRMSNorm, LinearOProj
from freetoken.layers.linear import _LinearTPImpl
from freetoken.layers.rotary import get_rope
from freetoken.utils import div_even, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen3_5Attention(BaseOP):
    """Gated full attention: per-head output gate, q/k RMSNorm, partial NeoX rope.

        query, gate = chunk(q_proj(x).view(.., num_q, head_dim*2), 2, -1)
        q = qnorm(query); k = knorm(k_proj(x)); v = v_proj(x)
        q, k = rope(q, k)                       # first rotary_dim dims
        attn = paged_attention(q, k, v)
        out = o_proj(attn * sigmoid(gate))

    TP: qkv is column-parallel (heads split across ranks, KV replicated when
    num_kv_heads < tp_size), o_proj is row-parallel (all-reduce). The forward works
    in TP-local head counts; the KV cache and attention backend are already TP-local.
    """

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        head_dim = config.head_dim
        self.layer_id = layer_id
        tp = get_tp_info().size
        full_num_q = config.num_qo_heads
        full_num_kv = config.num_kv_heads
        # Per-rank heads: q splits evenly, KV replicates when there are fewer than tp ranks.
        self.num_q = div_even(full_num_q, tp)
        self.num_kv = div_even(full_num_kv, tp, allow_replicate=True)
        self.head_dim = head_dim
        self.qo_attn_dim = self.num_q * head_dim
        self.kv_attn_dim = self.num_kv * head_dim

        # Fused q/k/v projection (one GEMM instead of three); the q half is 2x for the output
        # gate. GQA-aware column-parallel (like LinearQKVMerged, but with the gated 2x q): the q
        # heads split across ranks while the KV heads REPLICATE when num_kv_heads < tp_size
        # (a plain uniform divide would slice a KV head in half). local sizes match the loader's
        # shard_tensor and the attention backend's div_even(..., allow_replicate=True).
        local_q2 = self.num_q * head_dim * 2
        self._qkv_split = [local_q2, self.kv_attn_dim, self.kv_attn_dim]
        self.qkv_proj = _LinearTPImpl(
            full_isize=config.hidden_size,
            full_osize=full_num_q * head_dim * 2 + 2 * full_num_kv * head_dim,
            local_isize=config.hidden_size,
            local_osize=local_q2 + 2 * self.kv_attn_dim,
            has_bias=False,
            output_sizes=self._qkv_split,
            quant_config=config.quant,
            prefix=f"{prefix}.qkv_proj",
        )
        # Qwen3.5 uses Gemma-style (1+weight) RMSNorm; the weight loader bakes the +1
        # into the stored weight (GemmaRMSNorm scales by the raw weight).
        self.q_norm = GemmaRMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(head_dim, eps=config.rms_norm_eps)
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
            mrope_section=(
                tuple(config.rotary_config.mrope_section)
                if config.rotary_config.mrope_section is not None
                else None
            ),
            mrope_layout=config.rotary_config.mrope_layout,
        )
        # Row-parallel: the full gated-attention output is sharded across ranks on the
        # input dim; LinearOProj divides qo_attn_dim by tp and all-reduces the result.
        self.o_proj = LinearOProj(
            full_num_q * head_dim, config.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.o_proj",
        )

    def _project(self, x: torch.Tensor):
        """Returns (q, k, v, gate): q [N, num_q, head_dim] post qk-norm+rope,
        k [N, num_kv*head_dim] post norm+rope, v [N, num_kv*head_dim], gate [N, num_q*head_dim]."""
        positions = get_global_ctx().batch.get_attn_positions()
        qkv = self.qkv_proj.forward(x)
        qg, k, v = torch.split(qkv, self._qkv_split, dim=-1)
        qg = qg.view(-1, self.num_q, self.head_dim * 2)
        q = qg[..., : self.head_dim].contiguous()  # [N, num_q, head_dim]
        gate = qg[..., self.head_dim :].reshape(-1, self.qo_attn_dim)
        k = k.view(-1, self.num_kv, self.head_dim).contiguous()
        v = v.contiguous()  # split view has the qkv row stride; the KV store needs contiguous
        q = self.q_norm.forward(q).reshape(-1, self.qo_attn_dim)
        k = self.k_norm.forward(k).reshape(-1, self.kv_attn_dim)
        q, k = self.rotary.forward(positions, q, k)
        return q.view(-1, self.num_q, self.head_dim), k, v, gate

    def _combine(self, attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gated = attn_out.reshape(-1, self.qo_attn_dim) * torch.sigmoid(gate)
        return self.o_proj.forward(gated)

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        q, k, v, gate = self._project(x)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        return self._combine(o, gate)


__all__ = ["Qwen3_5Attention"]
