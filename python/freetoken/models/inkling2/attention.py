from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from freetoken.attention.triton import TritonMetadata
from freetoken.core import get_global_ctx
from freetoken.kernel.triton.inkling2 import EPI_L2_CLIP, EPI_RMS_NORM, decode_attention, extend_attention, q_prep, short_conv
from freetoken.layers import BaseOP, LinearColParallelMerged, LinearReplicated
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from .config import Inkling2Args
    from .state import ConvState

# query rows whose bias table is materialized at once: [rows, heads, rel_extent] bf16
_BIAS_ROWS = 2048


class Inkling2Attention(BaseOP):
    """Sliding-window or global GQA of one decoder layer, run once per visit with that visit's KV.

        q = l2clip(Wq xq * q_scale)            (* log_scale(pos) on global layers)
        k = l2clip(conv(Wk xk))                 static taps, or static + per-token taps on global layers
        v = rmsnorm(conv(Wv xv))                per head, no weight
        bias[t, h, d] = l2clip(Wr xq * rel_scale)[t, h] @ rel_proj[:, d]   for d < rel_extent
        o = Wo(sigmoid(Wg xq) * softmax([q.k * scale + bias, sink(xq)]) v)

    The sink is one extra logit per (row, head) with no value. ``q_side`` fuses the four
    projections of the query input, ``k_side`` / ``v_side`` the key / value projection with,
    on global layers, the projection of its per-token convolution taps.
    """

    def __init__(self, args: Inkling2Args, layer_id: int, *, prefix: str):
        self.layer_id = layer_id
        self.is_swa = args.is_swa(layer_id)
        self.args = args
        H = args.hidden_size
        self.num_heads = args.num_heads
        self.num_kv_heads = args.num_kv_heads
        self.head_dim = args.head_dim
        self.d_rel = args.d_rel
        self.rel_extent = args.sliding_window if self.is_swa else args.rel_extent
        self.sliding_window = args.sliding_window if self.is_swa else None
        self.taps = args.conv_kernel
        qo, kv = self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
        self._qo, self._kv = qo, kv
        self._rel_off = qo
        self._gate_off = qo + self.num_heads * self.d_rel
        self._sink_off = self._gate_off + qo
        self.q_side = LinearColParallelMerged(H, [qo, self.num_heads * self.d_rel, qo, self.num_heads], has_bias=False, prefix=f"{prefix}.q_side")
        dyn = [] if self.is_swa else [self.num_kv_heads * self.taps]
        self.k_side = LinearColParallelMerged(H, [kv, *dyn], has_bias=False, prefix=f"{prefix}.k_side")
        self.v_side = LinearColParallelMerged(H, [kv, *dyn], has_bias=False, prefix=f"{prefix}.v_side")
        self.o_proj = LinearReplicated(qo, H, has_bias=False, prefix=f"{prefix}.o_proj")
        # [channels, taps] fp32 indexed by backward lag (lag 0 = the current row)
        self.k_conv = torch.empty(kv, self.taps, dtype=torch.float32)
        self.v_conv = torch.empty(kv, self.taps, dtype=torch.float32)
        self.q_scale = torch.empty(self.num_heads, self.head_dim)
        self.rel_scale = torch.empty(self.num_heads, self.d_rel)
        self.rel_proj = torch.empty(self.d_rel, self.rel_extent)
        self._conv_scale = 1.0 / math.sqrt((self.taps if self.is_swa else 2 * self.taps) + 1)
        self._sm_scale = self.head_dim**-0.5
        self._log_scaling = None if self.is_swa else (args.log_scaling_n_floor, args.log_scaling_alpha)

    def _conv(self, x: torch.Tensor, weight: torch.Tensor, dyn: torch.Tensor | None, state: ConvState, visit: int, name: str, epilogue: int, epi_arg: float) -> torch.Tensor:
        batch = get_global_ctx().batch
        md = batch.attn_metadata
        return short_conv(
            x, weight, *state.rows(visit, name), md.indices, md.indptr, md.prefix_lens, self._conv_scale,
            decode=not batch.is_prefill, q_to_req=md.q_to_req, cu_q=md.cu_seqlens_q_gpu,
            dynamic=dyn, head_dim=self.head_dim, epilogue=epilogue, epi_arg=epi_arg,
        )

    @nvtx_annotate("Attention")
    def forward(self, xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor, visit: int, state: ConvState) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        md = batch.attn_metadata
        assert isinstance(md, TritonMetadata), "inkling2 attention reads the triton backend's metadata"
        T = xq.shape[0]
        Hq, Hkv, D, kv = self.num_heads, self.num_kv_heads, self.head_dim, self._kv

        qs = self.q_side.forward(xq)
        ks = self.k_side.forward(xk)
        vs = self.v_side.forward(xv)
        q, rel, sinks = q_prep(
            qs, self.q_scale, self.rel_scale, batch.positions, rel_off=self._rel_off, sink_off=self._sink_off,
            q_clip=self.args.qk_clip, rel_clip=self.args.rel_clip, log_scaling=self._log_scaling,
        )
        k_dyn = None if self.is_swa else ks[:, kv:].view(T, Hkv, self.taps)
        v_dyn = None if self.is_swa else vs[:, kv:].view(T, Hkv, self.taps)
        k = self._conv(ks[:, :kv], self.k_conv, k_dyn, state, visit, "k", EPI_L2_CLIP, self.args.qk_clip)
        v = self._conv(vs[:, :kv], self.v_conv, v_dyn, state, visit, "v", EPI_RMS_NORM, self.args.rms_norm_eps)

        kvcache = ctx.kv_cache
        kvcache.store_kv(k, v, batch.out_loc, visit)
        k_cache = kvcache.k_cache(visit).view(-1, Hkv, D)
        v_cache = kvcache.v_cache(visit).view(-1, Hkv, D)
        quant = dict(kv_format=kvcache.kv_format, k_scale=kvcache.k_scale(visit), v_scale=kvcache.v_scale(visit))
        indices = md.swa_indices if self.is_swa and md.swa_indices is not None else md.indices
        gate = qs[:, self._gate_off : self._gate_off + self._qo]
        out = torch.empty((T, Hq * D), dtype=xq.dtype, device=xq.device)
        rel = rel.view(T * Hq, self.d_rel)
        if batch.is_prefill:
            for lo in range(0, T, _BIAS_ROWS):
                hi = min(T, lo + _BIAS_ROWS)
                bias = torch.matmul(rel[lo * Hq : hi * Hq], self.rel_proj).view(hi - lo, Hq, self.rel_extent)
                extend_attention(
                    q, k_cache, v_cache, gate, bias, sinks, md.cu_seqlens_q_gpu, md.indptr, indices, md.prefix_lens,
                    md.max_q_len, self._sm_scale, self.sliding_window, self.rel_extent, out, row_lo=lo, row_hi=hi, **quant,
                    k_extend=k.view(T, Hkv, D), v_extend=v.view(T, Hkv, D),
                )
                del bias  # else the next table is built while this one is still live
        else:
            bias = torch.matmul(rel, self.rel_proj).view(T, Hq, self.rel_extent)
            decode_attention(
                q, k_cache, v_cache, gate, bias, sinks, md.indptr, indices, self._sm_scale, self.sliding_window, self.rel_extent, out,
                **quant,
            )
        return self.o_proj.forward(out)


__all__ = ["Inkling2Attention"]
