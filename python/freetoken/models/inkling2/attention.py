from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from freetoken.attention.triton import TritonMetadata
from freetoken.core import get_global_ctx
from freetoken.kernel.triton.inkling2 import decode_attention, extend_attention, short_conv
from freetoken.layers import BaseOP, LinearColParallelMerged, LinearReplicated
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from .config import Inkling2Args
    from .state import ConvState

# query rows whose bias table is materialized at once: [rows, heads, rel_extent] bf16
_BIAS_ROWS = 4096


def l2_clip(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Scale each last-dim vector down to ``threshold`` when its L2 norm is larger (fp32 in, fp32 out)."""
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * (threshold / (norm + 1e-6)).clamp(max=1.0)


def log_scale(positions: torch.Tensor, args: Inkling2Args) -> torch.Tensor:
    """``1 + alpha * log(max((pos + 1) / n_floor, 1))`` per row, fp32 ``[T, 1, 1]``."""
    ratio = (positions.float() + 1.0) / args.log_scaling_n_floor
    return (1.0 + args.log_scaling_alpha * torch.log(ratio.clamp(min=1.0)))[:, None, None]


def unweighted_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


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
        self._q_split = [qo, self.num_heads * self.d_rel, qo, self.num_heads]
        self.q_side = LinearColParallelMerged(H, self._q_split, has_bias=False, prefix=f"{prefix}.q_side")
        dyn = [] if self.is_swa else [self.num_kv_heads * self.taps]
        self._kv_split = [kv, *dyn]
        self.k_side = LinearColParallelMerged(H, self._kv_split, has_bias=False, prefix=f"{prefix}.k_side")
        self.v_side = LinearColParallelMerged(H, self._kv_split, has_bias=False, prefix=f"{prefix}.v_side")
        self.o_proj = LinearReplicated(qo, H, has_bias=False, prefix=f"{prefix}.o_proj")
        # [channels, taps] fp32 indexed by backward lag (lag 0 = the current row)
        self.k_conv = torch.empty(kv, self.taps, dtype=torch.float32)
        self.v_conv = torch.empty(kv, self.taps, dtype=torch.float32)
        self.q_scale = torch.empty(self.num_heads, self.head_dim)
        self.rel_scale = torch.empty(self.num_heads, self.d_rel)
        self.rel_proj = torch.empty(self.d_rel, self.rel_extent)
        self._conv_scale = 1.0 / math.sqrt((self.taps if self.is_swa else 2 * self.taps) + 1)
        self._sm_scale = self.head_dim**-0.5

    def _conv(self, x: torch.Tensor, weight: torch.Tensor, dyn: torch.Tensor | None, state: ConvState, visit: int, name: str, md: TritonMetadata) -> torch.Tensor:
        batch = get_global_ctx().batch
        return short_conv(
            x, weight, state.view(visit, name), state.batch_slots(batch.padded_size), self._conv_scale,
            decode=not batch.is_prefill, q_to_req=md.q_to_req, cu_q=md.cu_seqlens_q_gpu, prefix_lens=md.prefix_lens,
            dynamic=dyn, head_dim=self.head_dim,
        )

    @nvtx_annotate("Attention")
    def forward(self, xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor, visit: int, state: ConvState) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        md = batch.attn_metadata
        assert isinstance(md, TritonMetadata), "inkling2 attention reads the triton backend's metadata"
        T = xq.shape[0]
        Hq, Hkv, D = self.num_heads, self.num_kv_heads, self.head_dim

        q, rel, gate, sink = torch.split(self.q_side.forward(xq), self._q_split, dim=-1)
        k_out = self.k_side.forward(xk)
        v_out = self.v_side.forward(xv)
        if self.is_swa:
            k_raw, v_raw, k_dyn, v_dyn = k_out, v_out, None, None
        else:
            k_raw, k_dyn = torch.split(k_out, self._kv_split, dim=-1)
            v_raw, v_dyn = torch.split(v_out, self._kv_split, dim=-1)
            k_dyn = k_dyn.float().contiguous().view(T, Hkv, self.taps)
            v_dyn = v_dyn.float().contiguous().view(T, Hkv, self.taps)

        q = l2_clip(q.float().view(T, Hq, D) * self.q_scale.float(), self.args.qk_clip).to(q.dtype)
        rel = l2_clip(rel.float().view(T, Hq, self.d_rel) * self.rel_scale.float(), self.args.rel_clip).to(rel.dtype)
        if not self.is_swa:
            tau = log_scale(batch.positions, self.args)
            q = (q.float() * tau).to(q.dtype)
            rel = (rel.float() * tau).to(rel.dtype)

        k = self._conv(k_raw.contiguous(), self.k_conv, k_dyn, state, visit, "k", md)
        v = self._conv(v_raw.contiguous(), self.v_conv, v_dyn, state, visit, "v", md)
        k = l2_clip(k.float().view(T, Hkv, D), self.args.qk_clip).to(k.dtype)
        v = unweighted_rms_norm(v.view(T, Hkv, D), self.args.rms_norm_eps).to(v.dtype)

        kvcache = ctx.kv_cache
        assert not kvcache.kv_format.quantized, "inkling2 attention reads bf16 KV only"
        kvcache.store_kv(k.view(T, Hkv * D), v.view(T, Hkv * D), batch.out_loc, visit)
        k_cache = kvcache.k_cache(visit).view(-1, Hkv, D)
        v_cache = kvcache.v_cache(visit).view(-1, Hkv, D)
        indices = md.swa_indices if self.is_swa and md.swa_indices is not None else md.indices
        sinks = sink.float().contiguous()
        out = torch.empty((T, Hq, D), dtype=xq.dtype, device=xq.device)
        rel = rel.contiguous().view(T * Hq, self.d_rel)
        if batch.is_prefill:
            for lo in range(0, T, _BIAS_ROWS):
                hi = min(T, lo + _BIAS_ROWS)
                bias = torch.matmul(rel[lo * Hq : hi * Hq], self.rel_proj).view(hi - lo, Hq, self.rel_extent)
                extend_attention(
                    q, k_cache, v_cache, bias, sinks, md.cu_seqlens_q_gpu, md.indptr, indices, md.prefix_lens,
                    md.max_q_len, self._sm_scale, self.sliding_window, self.rel_extent, out, row_lo=lo, row_hi=hi,
                )
        else:
            bias = torch.matmul(rel, self.rel_proj).view(T, Hq, self.rel_extent)
            decode_attention(
                q, k_cache, v_cache, bias, sinks, md.indptr, indices, self._sm_scale, self.sliding_window, self.rel_extent, out,
            )
        out = (out.view(T, Hq * D).float() * torch.sigmoid(gate.float())).to(out.dtype)
        return self.o_proj.forward(out)


__all__ = ["Inkling2Attention", "l2_clip", "log_scale", "unweighted_rms_norm"]
