from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.layers import BaseOP, GemmaRMSNorm
from freetoken.layers.rotary import get_rope
from freetoken.utils import div_even, nvtx_annotate

from .quant_linear import make_col_merged, make_row_parallel

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen3_5Attention(BaseOP):
    """Gated full attention: per-head output gate, q/k RMSNorm, partial NeoX rope.

        query, gate = chunk(q_proj(x).view(.., num_q, head_dim*2), 2, -1)
        q = qnorm(query); k = knorm(k_proj(x)); v = v_proj(x)
        q, k = rope(q, k)                       # first rotary_dim dims
        attn = paged_attention(q, k, v)
        out = o_proj(attn * sigmoid(gate))
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        head_dim = config.head_dim
        self.layer_id = layer_id
        self.num_q = config.num_qo_heads
        self.num_kv = config.num_kv_heads
        self.head_dim = head_dim
        self.qo_attn_dim = self.num_q * head_dim
        self.kv_attn_dim = self.num_kv * head_dim

        tp = get_tp_info()
        # Fused q/k/v projection (one GEMM instead of three); q half is 2x for the
        # output gate. Split sizes: [num_q*head_dim*2, num_kv*head_dim, num_kv*head_dim].
        # _qkv_split is TP-local (for torch.split in forward); make_col_merged gets full sizes.
        # Block-fp8 (Fp8BlockColMerged) when the checkpoint is quantized, else bf16
        # LinearColParallelMerged. q/k/v out dims are all /128, so the merged fp8 weight +
        # weight_scale_inv concatenate cleanly along the output dim.
        full_split = [self.num_q * head_dim * 2, self.kv_attn_dim, self.kv_attn_dim]
        self._local_num_q = div_even(self.num_q, tp.size)
        self._local_num_kv = div_even(self.num_kv, tp.size, allow_replicate=True)
        self._local_qo_attn_dim = self._local_num_q * head_dim
        self._local_kv_attn_dim = self._local_num_kv * head_dim
        self._qkv_split = [self._local_num_q * head_dim * 2, self._local_kv_attn_dim, self._local_kv_attn_dim]
        local_qkv_sizes = [self._local_num_q * head_dim * 2, self._local_kv_attn_dim, self._local_kv_attn_dim]
        self.qkv_proj = make_col_merged(config, config.hidden_size, full_split, has_bias=False, local_output_sizes=local_qkv_sizes)
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
        )
        self.o_proj = make_row_parallel(config, self.qo_attn_dim, config.hidden_size, has_bias=False)

    def _project(self, x: torch.Tensor):
        """Returns (q, k, v, gate): q [N, num_q, head_dim] post qk-norm+rope,
        k [N, num_kv*head_dim] post norm+rope, v [N, num_kv*head_dim], gate [N, num_q*head_dim]."""
        positions = get_global_ctx().batch.positions
        qkv = self.qkv_proj.forward(x)
        qg, k, v = torch.split(qkv, self._qkv_split, dim=-1)
        qg = qg.view(-1, self._local_num_q, self.head_dim * 2)
        q = qg[..., : self.head_dim].contiguous()  # [N, num_q, head_dim]
        gate = qg[..., self.head_dim :].reshape(-1, self._local_qo_attn_dim)
        k = k.reshape(-1, self._local_num_kv, self.head_dim).contiguous()
        v = v.contiguous()  # split view has the qkv row stride; the KV store needs contiguous
        q = self.q_norm.forward(q).reshape(-1, self._local_qo_attn_dim)
        k = self.k_norm.forward(k).reshape(-1, self._local_kv_attn_dim)
        q, k = self.rotary.forward(positions, q, k)
        return q.view(-1, self._local_num_q, self.head_dim), k, v, gate

    def _combine(self, attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gated = attn_out.reshape(-1, self._local_qo_attn_dim) * torch.sigmoid(gate)
        return self.o_proj.forward(gated)

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        q, k, v, gate = self._project(x)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        return self._combine(o, gate)


__all__ = ["Qwen3_5Attention"]
