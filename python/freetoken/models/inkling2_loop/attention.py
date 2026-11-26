"""Inkling 2 attention: L2-clipped q / k, unweighted-RMS v, short convolutions on k and v (dynamic taps on the
global layers), a data-dependent relative position bias, one sink logit per head and a sigmoid output gate.

The global layers also scale q and the relative states by ``1 + alpha * ln(max((pos + 1) / n_floor, 1))``.
Every run of a layer reads and writes its own cache layer (``cache_layer``); the weights are shared.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearReplicated
from freetoken.utils import nvtx_annotate

from .config import CONV_STATES, Inkling2LoopArgs
from .conv import ConvMetadata, short_conv

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


def l2_clip(x: torch.Tensor, threshold: float, scale: torch.Tensor | None = None) -> torch.Tensor:
    """Each head vector (last dim) scaled down to ``threshold`` when longer; fp32 in, fp32 out."""
    if scale is not None:
        x = x * scale
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * (threshold / (norm + 1e-6)).clamp(max=1.0)


def log_scale(positions: torch.Tensor, n_floor: int, alpha: float) -> torch.Tensor:
    """``[T, 1, 1]`` fp32 ``1 + alpha * ln(max((pos + 1) / n_floor, 1))``."""
    tau = 1.0 + alpha * torch.log(((positions.float() + 1.0) / n_floor).clamp(min=1.0))
    return tau.view(-1, 1, 1)


def unweighted_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


class Inkling2Attention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        args: Inkling2LoopArgs = config.inkling2_args
        self._args = args
        self.layer_id = layer_id
        self.is_sliding = args.is_sliding(layer_id)
        self.num_heads = args.swa_num_heads if self.is_sliding else args.num_heads
        self.num_kv_heads = args.swa_num_kv_heads if self.is_sliding else args.num_kv_heads
        self.head_dim = args.swa_head_dim if self.is_sliding else args.head_dim
        H, q_dim, kv_dim = args.hidden_size, self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
        K = args.conv_kernel
        quant = config.quant
        self.q_proj = LinearReplicated(H, q_dim, has_bias=False, quant_config=quant, prefix=f"{prefix}.q_proj")
        self.k_proj = LinearReplicated(H, kv_dim, has_bias=False, quant_config=quant, prefix=f"{prefix}.k_proj")
        self.v_proj = LinearReplicated(H, kv_dim, has_bias=False, quant_config=quant, prefix=f"{prefix}.v_proj")
        self.rel_proj = LinearReplicated(H, self.num_heads * args.d_rel, has_bias=False, quant_config=quant, prefix=f"{prefix}.rel_proj")
        self.sink_proj = LinearReplicated(H, self.num_heads, has_bias=False, quant_config=quant, prefix=f"{prefix}.sink_proj")
        self.gate_proj = LinearReplicated(H, q_dim, has_bias=False, quant_config=quant, prefix=f"{prefix}.gate_proj")
        self.o_proj = LinearReplicated(q_dim, H, has_bias=False, quant_config=quant, prefix=f"{prefix}.o_proj")
        self.q_scale = torch.empty(self.num_heads, self.head_dim, dtype=torch.float32)
        self.rel_scale = torch.empty(self.num_heads, args.d_rel, dtype=torch.float32)
        self.rel_table = torch.empty(args.d_rel, args.rel_extent)
        # taps by lag: [0] weights the current token
        if self.is_sliding:
            self.k_conv = torch.empty(K, kv_dim, dtype=torch.float32)
            self.v_conv = torch.empty(K, kv_dim, dtype=torch.float32)
        else:
            self.k_conv = torch.empty(K, kv_dim)
            self.v_conv = torch.empty(K, kv_dim)
            self.k_conv_proj = LinearReplicated(H, self.num_kv_heads * K, has_bias=False, quant_config=quant, prefix=f"{prefix}.k_conv_proj")
            self.v_conv_proj = LinearReplicated(H, self.num_kv_heads * K, has_bias=False, quant_config=quant, prefix=f"{prefix}.v_conv_proj")
        self._scale = self.head_dim**-0.5

    def _kv(self, x: torch.Tensor, proj, conv: torch.Tensor, conv_proj, state: torch.Tensor, meta: ConvMetadata, stash) -> torch.Tensor:
        T = x.shape[0]
        y = proj.forward(x)
        dynamic = None if conv_proj is None else conv_proj.forward(x).float().view(T, self.num_kv_heads, -1)
        return short_conv(y, conv.float(), state, meta, dynamic=dynamic, head_dim=self.head_dim, verify_stash=stash)

    @nvtx_annotate("MHA")
    def forward(self, xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor, cache_layer: int, meta: ConvMetadata) -> torch.Tensor:
        """``xq`` feeds q, the relative states, the sinks and the output gate; ``xk`` / ``xv`` the keys / values."""
        ctx = get_global_ctx()
        batch = ctx.batch
        args = self._args
        pool = ctx.linear_state_pool
        T, dt = xq.shape[0], xq.dtype
        H, D = self.num_heads, self.head_dim
        positions = batch.positions

        def stash(name: str):
            return pool.verify_handler(name).stash(cache_layer) if meta.verify else None

        q = l2_clip(self.q_proj.forward(xq).view(T, H, D).float(), args.qk_clip, self.q_scale).to(dt)
        rel = l2_clip(self.rel_proj.forward(xq).view(T, H, args.d_rel).float(), args.rel_clip, self.rel_scale).to(dt)
        if not self.is_sliding:
            tau = log_scale(positions, args.log_n_floor, args.log_alpha)
            q = (q.float() * tau).to(dt)
            rel = (rel.float() * tau).to(dt)
        dyn = not self.is_sliding
        k = self._kv(xk, self.k_proj, self.k_conv, self.k_conv_proj if dyn else None,
                     pool.slot_state(CONV_STATES[0], cache_layer), meta, stash(CONV_STATES[0]))
        v = self._kv(xv, self.v_proj, self.v_conv, self.v_conv_proj if dyn else None,
                     pool.slot_state(CONV_STATES[1], cache_layer), meta, stash(CONV_STATES[1]))
        k = l2_clip(k.view(T, self.num_kv_heads, D).float(), args.qk_clip).to(dt)
        v = unweighted_rms_norm(v.view(T, self.num_kv_heads, D), args.rms_norm_eps).to(dt)
        sinks = self.sink_proj.forward(xq)

        o = self._attend(q, k, v, rel, sinks, cache_layer)
        gate = torch.sigmoid(self.gate_proj.forward(xq).float())
        o = (o.view(T, H * D).float() * gate).to(dt)
        return self.o_proj.forward(o)

    def _attend(self, q, k, v, rel, sinks, cache_layer: int) -> torch.Tensor:
        from freetoken.kernel.triton.inkling2_attention import inkling2_paged_attention

        ctx = get_global_ctx()
        batch = ctx.batch
        kv = ctx.kv_cache
        md = batch.attn_metadata
        kv.store_kv(k.reshape(k.shape[0], -1), v.reshape(v.shape[0], -1), batch.out_loc, cache_layer)
        k_cache = kv.k_cache(cache_layer)
        v_cache = kv.v_cache(cache_layer)
        k_cache = k_cache.view(-1, k_cache.shape[-2], k_cache.shape[-1])
        v_cache = v_cache.view(-1, v_cache.shape[-2], v_cache.shape[-1])
        indices = md.swa_indices if (self.is_sliding and md.swa_indices is not None) else md.indices
        return inkling2_paged_attention(
            q, k_cache, v_cache,
            qo_indptr=md.cu_seqlens_q_gpu, kv_indptr=md.indptr, kv_indices=indices,
            q_positions=md.q_positions, max_q_len=md.max_q_len,
            rel=rel, table=self.rel_table, sinks=sinks, sm_scale=self._scale,
            sliding_window=self._args.sliding_window if self.is_sliding else None,
        )


__all__ = ["Inkling2Attention", "l2_clip", "log_scale", "unweighted_rms_norm"]
