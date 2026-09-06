"""GLM-5.3-Flash KDA (Kimi Delta Attention) op.

Per-channel gated delta rule over H=64 heads of D=128, with SEPARATE q/k/v short
convolutions (vs GDN's one fused conv), a low-rank forget gate (``f_a`` ->
``f_b`` -> raw per-channel logits; the bounded safe gate ``lower_bound *
sigmoid(exp(A_log) * (g + dt_bias))`` is computed inside the kernels), a
per-head sigmoid beta (``b_proj``), and a sigmoid-gated output RMSNorm
(``o_norm`` gated by ``g_b(g_a(x))``).

State lives in ``ctx.linear_state_pool`` exactly like GDN: the conv state is the
merged q|k|v stream (width 3*H*D == the pool's ``2*K*dk + V*dv``) and the
recurrent state is one [D, D] matrix per head, stored in the KERNEL's [V, K]
layout (coincides with the pool's [K, V] declaration because D_k == D_v; same
convention as GDN, see qwen3_5_moe/gdn.py).

Kernels: ``fused_recurrent_kda`` decodes with in-kernel gate/beta/l2norm and
per-slot state read/write (slot 0 is its NULL sentinel == the pool's padding
slot); ``chunk_kda_with_fused_gate`` prefills from an explicitly gathered
initial state and returns the final state, which this op scatters back (the
kernel CLOBBERS its v buffer -- v here is an ephemeral conv output, so that is
free). Hybrid-radix track snapshots ride the per-chunk h (``return_h``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.layers import BaseOP, LinearColParallelMerged, LinearReplicated
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class _DepthwiseConv1d(BaseOP):
    """Merged q|k|v depthwise conv weight ``[3*H*D, 1, K]`` (key ``conv1d.weight``;
    the loader concatenates the checkpoint's q/k/v_conv1d along channels)."""

    def __init__(self, conv_dim: int, kernel: int):
        self.weight = torch.empty(conv_dim, 1, kernel)


class _GatedRMSNormSigmoid(BaseOP):
    """RMSNorm(x) * sigmoid(z), fused (KDA's o_norm; GDN's variant gates with silu)."""

    def __init__(self, dim: int, eps: float):
        self.weight = torch.empty(dim)
        self.eps = eps

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.fla import rms_norm_gated

        return rms_norm_gated(
            x=x, weight=self.weight, bias=None, z=z, eps=self.eps,
            is_rms_norm=True, norm_before_gate=True, activation="sigmoid",
        )


class Glm5NextKDA(BaseOP):
    """KDA op; state is held in ``ctx.linear_state_pool`` keyed by the request's
    linear slot (``FLAMetadata.cache_indices``). Parameter names follow the
    checkpoint modulo two load-time fusions (see weight.py): ``in_proj`` is
    q|k|v|b|f_a|g_a concatenated, ``conv1d`` is q|k|v conv concatenated."""

    def __init__(self, config: ModelConfig, layer_id: int):
        args = config.glm5_args
        self.layer_id = layer_id
        self.num_heads = args.linear_num_heads
        self.head_dim = args.linear_head_dim
        self.proj_size = self.num_heads * self.head_dim  # H * D
        self.conv_dim = 3 * self.proj_size  # merged q|k|v stream
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.lower_bound = args.linear_lower_bound
        self.scale = self.head_dim**-0.5

        p, h, d = self.proj_size, self.num_heads, self.head_dim
        # q|k|v dominate the resident read (3 * 8192 x 4096 = 201 MB/layer bf16 --
        # the single biggest dense-weight stream in the model); under the fp8
        # resident mode they split into their own W8A16 GEMM while the small,
        # precision-sensitive gate projections (b|f_a|g_a) stay bf16 (the GDN
        # qkvz/ba split precedent). BF16 mode keeps the single fused GEMM.
        self._fp8 = config.attn_quant == "fp8_pertensor"
        self._bfg_split = [h, d, d]
        if self._fp8:
            from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorColMerged

            self.in_proj_qkv = Fp8PerTensorColMerged(
                args.hidden_size, [p, p, p], has_bias=False
            )
            self.in_proj_bfg = LinearColParallelMerged(
                args.hidden_size, self._bfg_split, has_bias=False
            )
        else:
            self._in_proj_split = [p, p, p, h, d, d]
            self.in_proj = LinearColParallelMerged(
                args.hidden_size, self._in_proj_split, has_bias=False
            )
        # Low-rank gate up-projections (128 -> 8192): forget gate and output gate.
        self.f_b_proj = LinearReplicated(d, p, has_bias=False)
        self.g_b_proj = LinearReplicated(d, p, has_bias=False)
        self.conv1d = _DepthwiseConv1d(self.conv_dim, self.conv_kernel_size)
        # Gate params stay fp32 (exp/sigmoid precision; the kernels read fp32).
        # models/weight.py exempts *.A_log / *.dt_bias from the model-dtype downcast.
        self.A_log = torch.empty(h, dtype=torch.float32)
        self.dt_bias = torch.empty(p, dtype=torch.float32)
        self.o_norm = _GatedRMSNormSigmoid(d, eps=args.norm_eps)
        # o_proj follows the resident quant mode (rationale: the split comment
        # at _fp8 above); f_b/g_b stay bf16 alongside the gate slice.
        from .attention import _make_proj

        self.o_proj = _make_proj(config.attn_quant, p, args.hidden_size)

    def _conv_weight(self) -> torch.Tensor:
        return self.conv1d.weight.squeeze(1)  # [conv_dim, kernel]

    def _write_track_snapshot(self, pool, li, conv_in, h, fla) -> None:
        """Hybrid-radix: snapshot recurrent + conv state at the chunk-aligned track
        boundary into a donatable pool slot (same contract as GDN, see
        qwen3_5_moe/gdn.py). h rows are the kernel's per-chunk [V, K] states --
        a direct copy into the pool's [K, V] slots (D_k == D_v)."""
        rec = pool.recurrent_states[li]
        rec.index_copy_(0, fla.track_dst, h[0, fla.track_h_row].to(rec.dtype))
        cv = pool.conv_states[li]
        conv_win = conv_in[fla.track_conv_src].transpose(-1, -2).contiguous()
        cv.index_copy_(0, fla.track_dst, conv_win.to(cv.dtype))

    @nvtx_annotate("KDA")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        total = hidden_states.shape[0]
        dtype = hidden_states.dtype
        h, d, p = self.num_heads, self.head_dim, self.proj_size

        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla

        if self._fp8:
            conv_in = self.in_proj_qkv.forward(hidden_states)
            b, f_a, g_a = torch.split(
                self.in_proj_bfg.forward(hidden_states), self._bfg_split, dim=-1
            )
        else:
            proj = self.in_proj.forward(hidden_states)
            conv_in, b, f_a, g_a = torch.split(
                proj, [self.conv_dim, h, d, d], dim=-1
            )
        g1 = self.f_b_proj.forward(f_a)  # raw forget-gate logits [T, H*D]
        g2 = self.g_b_proj.forward(g_a)  # output-gate logits [T, H*D]
        li = pool.local_index(self.layer_id)

        if batch.is_decode:
            mixed = causal_conv1d_decode(
                conv_in, pool.conv_states[li], self._conv_weight(), fla.cache_indices
            )
            bsz = mixed.shape[0]
            q, k, v = (
                t.reshape(1, bsz, h, d).to(dtype)
                for t in torch.split(mixed, [p, p, p], dim=-1)
            )
            core_out, _ = _fused_recurrent(
                q, k, v,
                g=g1.view(1, bsz, h, d),
                beta=b.view(1, bsz, h),
                state_pool=pool.recurrent_states[li],
                indices=fla.cache_indices,
                cu_seqlens=fla.cu_seqlens,
                a_log=self.A_log,
                dt_bias=self.dt_bias,
                lower_bound=self.lower_bound,
                scale=self.scale,
            )
        else:
            x = conv_in.transpose(0, 1).contiguous()  # [conv_dim, total]
            mixed = causal_conv1d_varlen(
                x, self._conv_weight(), pool.conv_states[li],
                fla.cu_seqlens, fla.cache_indices, fla.has_initial_state,
            ).transpose(0, 1)
            q, k, v = (
                t.reshape(1, total, h, d).to(dtype)
                for t in torch.split(mixed, [p, p, p], dim=-1)
            )
            # Fresh sequences start from a zeroed slot; then gather every request's
            # initial state (the chunk kernel takes it dense, [N, H, D, D]).
            rec = pool.recurrent_states[li]
            if fla.fresh_state_indices is not None:
                rec.index_fill_(0, fla.fresh_state_indices, 0.0)
            slot_ids = fla.cache_indices.long()
            initial = rec.index_select(0, slot_ids)

            from freetoken.kernel.fla import chunk_kda_with_fused_gate

            track = fla.track_dst is not None
            result = chunk_kda_with_fused_gate(
                q=q, k=k, v=v,  # NOTE: v (ephemeral conv output) is clobbered
                raw_g=g1.view(1, total, h, d),
                beta=b.float().sigmoid().view(1, total, h),
                A_log=self.A_log,
                g_bias=self.dt_bias,
                scale=self.scale,
                initial_state=initial,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=fla.cu_seqlens,
                safe_gate=True,
                lower_bound=self.lower_bound,
                return_h=track,
            )
            if track:
                core_out, final_state, chunk_h = result
                self._write_track_snapshot(pool, li, conv_in, chunk_h, fla)
            else:
                core_out, final_state = result
            rec.index_copy_(0, slot_ids, final_state.to(rec.dtype))

        core_out = core_out.reshape(-1, d)
        out = self.o_norm.forward(core_out, g2.reshape(-1, d)).reshape(total, -1)
        return self.o_proj.forward(out.to(dtype))


def _fused_recurrent(
    q, k, v, g, beta, state_pool, indices, cu_seqlens,
    a_log, dt_bias, lower_bound, scale,
):
    """Decode via the vendored recurrent kernel: gate + beta-sigmoid + q/k l2norm
    in-kernel, state read/written in place at ``indices`` (int32, 1 token/req)."""
    from freetoken.kernel.fla import fused_recurrent_kda

    return fused_recurrent_kda(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        initial_state=state_pool,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=indices,
        sigmoid_beta=True,
        a_log=a_log,
        g_bias=dt_bias,
        compute_gate=True,
        lower_bound=lower_bound,
    )


__all__ = ["Glm5NextKDA"]
