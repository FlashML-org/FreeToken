"""Kimi-K3 text-tower mixers: NoPE MLA and Kimi Delta Attention (KDA)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from freetoken.core import get_global_ctx
from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.layers import BaseOP, LinearReplicated, RMSNorm
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class _DepthwiseConv1d(BaseOP):
    """Checkpoint-compatible FLA ShortConvolution weight holder."""

    def __init__(self, channels: int, kernel_size: int):
        # The public checkpoint deliberately stores convolution kernels in FP32,
        # even when the surrounding projections are BF16. Model construction runs
        # under a BF16 default-dtype context, so make this exception explicit.
        self.weight = torch.empty(channels, 1, kernel_size, dtype=torch.float32)


class _SigmoidRMSNormGated(BaseOP):
    """RMSNorm(x) * sigmoid(gate), matching FLA's FusedRMSNormGated."""

    def __init__(self, size: int, eps: float):
        # KDA's gated output norm is FP32 in the released checkpoint.
        self.weight = torch.empty(size, dtype=torch.float32)
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.fla import rms_norm_gated

        return rms_norm_gated(
            x=x,
            weight=self.weight,
            bias=None,
            z=gate,
            eps=self.eps,
            is_rms_norm=True,
            norm_before_gate=True,
            activation="sigmoid",
        )


class KimiMLAAttention(BaseOP):
    """NoPE MLA with low-rank Q/KV projections and an output gate.

    ``kv_b_proj`` is absorbed into Q and the output, so the paged cache stores only
    ``compressed_kv | k_rope``.  Kimi calls the latter field ``rope`` for checkpoint
    compatibility, but ``mla_use_nope`` means neither it nor Q's matching dimensions
    receive a positional rotation.
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        args = config.kimi_k3_args
        if args is None:
            raise ValueError("KimiMLAAttention requires ModelConfig.kimi_k3_args")
        self.layer_id = layer_id
        self.num_heads = args.num_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.q_head_dim = args.qk_head_dim
        self.v_head_dim = args.v_head_dim

        self.q_a_proj = LinearReplicated(
            args.hidden_size, args.q_lora_rank, has_bias=False
        )
        self.q_a_layernorm = RMSNorm(args.q_lora_rank, eps=1e-6)
        self.q_b_proj = LinearReplicated(
            args.q_lora_rank, args.num_heads * args.qk_head_dim, has_bias=False
        )
        self.kv_a_proj_with_mqa = LinearReplicated(
            args.hidden_size, args.kv_lora_rank + args.qk_rope_head_dim, has_bias=False
        )
        self.kv_a_layernorm = RMSNorm(args.kv_lora_rank, eps=1e-6)
        self.kv_b_proj = LinearReplicated(
            args.kv_lora_rank,
            args.num_heads * (args.qk_nope_head_dim + args.v_head_dim),
            has_bias=False,
        )
        self.g_proj = LinearReplicated(
            args.hidden_size, args.num_heads * args.v_head_dim, has_bias=False
        )
        self.o_proj = LinearReplicated(
            args.num_heads * args.v_head_dim, args.hidden_size, has_bias=False
        )
        self._w_uk: torch.Tensor | None = None
        self._w_uv: torch.Tensor | None = None

    def _kv_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._w_uk is None:
            if self.kv_b_proj.weight is None:
                raise RuntimeError(
                    "Kimi MLA kv_b projection was freed before repacking"
                )
            w = self.kv_b_proj.weight.view(
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
                self.kv_lora_rank,
            )
            self._w_uk = w[:, : self.qk_nope_head_dim].contiguous()
            self._w_uv = w[:, self.qk_nope_head_dim :].transpose(1, 2).contiguous()
        assert self._w_uv is not None
        return self._w_uk, self._w_uv

    def prepare_for_runtime(self) -> None:
        self._kv_b()
        self.kv_b_proj.weight = None

    @nvtx_annotate("MLA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        n = x.shape[0]
        w_uk, w_uv = self._kv_b()

        q = self.q_b_proj.forward(self.q_a_layernorm.forward(self.q_a_proj.forward(x)))
        q = q.view(n, self.num_heads, self.q_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_a_proj_with_mqa.forward(x)
        c_kv, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_a_layernorm.forward(c_kv)

        q_absorbed = torch.bmm(q_nope.transpose(0, 1).contiguous(), w_uk).transpose(
            0, 1
        )
        o_latent = ctx.attn_backend.mla_forward(
            q_absorbed.contiguous(),
            q_pe.contiguous(),
            c_kv.contiguous(),
            k_pe.contiguous(),
            self.layer_id,
            ctx.batch,
        )
        o = torch.bmm(o_latent.transpose(0, 1).contiguous(), w_uv).transpose(0, 1)
        o = o.reshape(n, self.num_heads * self.v_head_dim)
        o = o * torch.sigmoid(self.g_proj.forward(x))
        return self.o_proj.forward(o)


class KimiDeltaAttention(BaseOP):
    """KDA over FreeToken's per-request convolution and recurrent state pools."""

    def __init__(self, config: ModelConfig, layer_id: int):
        args = config.kimi_k3_args
        if args is None:
            raise ValueError("KimiDeltaAttention requires ModelConfig.kimi_k3_args")
        self.layer_id = layer_id
        self.hidden_size = args.hidden_size
        self.num_heads = args.kda_num_heads
        self.head_dim = args.kda_head_dim
        self.projection_size = self.num_heads * self.head_dim
        self.conv_size = args.kda_conv_kernel
        self.gate_lower_bound = args.kda_gate_lower_bound

        self.q_proj = LinearReplicated(
            args.hidden_size, self.projection_size, has_bias=False
        )
        self.k_proj = LinearReplicated(
            args.hidden_size, self.projection_size, has_bias=False
        )
        self.v_proj = LinearReplicated(
            args.hidden_size, self.projection_size, has_bias=False
        )
        self.q_conv1d = _DepthwiseConv1d(self.projection_size, self.conv_size)
        self.k_conv1d = _DepthwiseConv1d(self.projection_size, self.conv_size)
        self.v_conv1d = _DepthwiseConv1d(self.projection_size, self.conv_size)

        # The production checkpoint's bounded gate stores one decay per key
        # dimension.  The small development checkpoint uses the original
        # unbounded, per-head KDA parameterization.
        a_log_size = (
            self.head_dim if self.gate_lower_bound is not None else self.num_heads
        )
        self.A_log = torch.empty(a_log_size, dtype=torch.float32)
        self.f_a_proj = LinearReplicated(
            args.hidden_size, self.head_dim, has_bias=False
        )
        self.f_b_proj = LinearReplicated(
            self.head_dim, self.projection_size, has_bias=False
        )
        self.dt_bias = torch.empty(self.projection_size, dtype=torch.float32)
        self.b_proj = LinearReplicated(args.hidden_size, self.num_heads, has_bias=False)
        self.g_proj = LinearReplicated(
            args.hidden_size, self.projection_size, has_bias=False
        )
        self.o_norm = _SigmoidRMSNormGated(self.head_dim, config.rms_norm_eps)
        self.o_proj = LinearReplicated(
            self.projection_size, args.hidden_size, has_bias=False
        )

    def _conv_weight(self, dtype: torch.dtype) -> torch.Tensor:
        """Return KDA's checkpoint FP32 convolution weights in the activation dtype.

        The public checkpoint intentionally stores these weights in FP32, but the
        fused ``sgl_kernel`` causal-convolution op requires its input and weight
        dtypes to match.  Kimi-K3 activations are BF16 in production.
        """
        return torch.cat(
            [self.q_conv1d.weight, self.k_conv1d.weight, self.v_conv1d.weight], dim=0
        ).squeeze(1).to(dtype=dtype)

    def _safe_a(self, a: torch.Tensor) -> torch.Tensor:
        """Fold per-key A into ``a`` and apply Kimi's lower bound exactly.

        The vendored recurrent decode kernel accepts one ``A_log`` scalar per
        head, whereas the K3 checkpoint has one per key dimension.  Passing a
        zero head decay and folding ``exp(A_log)`` through inverse-softplus is
        algebraically equivalent and avoids silently misinterpreting weights.
        """
        if self.gate_lower_bound is None:
            raise RuntimeError("safe KDA gate requested without a lower bound")
        x = a.float() + self.dt_bias
        decay = self.A_log.exp().repeat(self.num_heads)
        # Official safe KDA gate:
        #   g = lower_bound * sigmoid(exp(A_log) * (a + dt_bias))
        # The vendored recurrent kernel implements
        #   g = -exp(A_head) * softplus(a' + dt_bias).
        # Feed it A_head=0 below and inverse-softplus the positive magnitude
        # here.  This preserves the checkpoint's per-key A_log and its bounded
        # [-5, 0) decay exactly instead of approximating it with a clamp.
        magnitude = -self.gate_lower_bound * torch.sigmoid(decay * x)
        # Stable inverse softplus; values above 20 are already indistinguishable from x.
        inverse = torch.where(
            magnitude > 20.0, magnitude, torch.log(torch.expm1(magnitude))
        )
        return inverse - self.dt_bias

    def _recurrent(self, q, k, v, a, beta, pool, fla) -> torch.Tensor:
        from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update

        if fla.track_dst is not None:
            raise NotImplementedError(
                "Kimi-K3 KDA does not yet support hybrid-radix mid-chunk state snapshots"
            )
        li = pool.local_index(self.layer_id)
        if fla.fresh_state_indices is not None:
            pool.recurrent_states[li].index_fill_(0, fla.fresh_state_indices, 0.0)
        safe_gate = self.gate_lower_bound is not None
        return fused_sigmoid_gating_delta_rule_update(
            A_log=(self.A_log.new_zeros(self.num_heads) if safe_gate else self.A_log),
            a=(self._safe_a(a) if safe_gate else a),
            dt_bias=self.dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            b=beta,
            initial_state_source=pool.recurrent_states[li],
            initial_state_indices=fla.cache_indices,
            scale=self.head_dim**-0.5,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=fla.cu_seqlens,
            is_kda=True,
        )

    @nvtx_annotate("KDA")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        if pool is None:
            raise RuntimeError("Kimi KDA requires a LinearStatePool")
        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla

        qkv_in = torch.cat(
            [
                self.q_proj.forward(hidden_states),
                self.k_proj.forward(hidden_states),
                self.v_proj.forward(hidden_states),
            ],
            dim=-1,
        )
        li = pool.local_index(self.layer_id)
        if batch.is_decode:
            mixed = causal_conv1d_decode(
                qkv_in,
                pool.conv_states[li],
                self._conv_weight(qkv_in.dtype),
                fla.cache_indices,
            )
        else:
            mixed = causal_conv1d_varlen(
                qkv_in.transpose(0, 1).contiguous(),
                self._conv_weight(qkv_in.dtype),
                pool.conv_states[li],
                fla.cu_seqlens,
                fla.cache_indices,
                fla.has_initial_state,
            ).transpose(0, 1)

        q, k, v = mixed.split([self.projection_size] * 3, dim=-1)
        n = hidden_states.shape[0]
        q = q.view(1, n, self.num_heads, self.head_dim)
        k = k.view(1, n, self.num_heads, self.head_dim)
        v = v.view(1, n, self.num_heads, self.head_dim)
        a = self.f_b_proj.forward(self.f_a_proj.forward(hidden_states))
        beta = self.b_proj.forward(hidden_states).float()
        out = self._recurrent(q, k, v, a, beta, pool, fla)
        gate = self.g_proj.forward(hidden_states).view(n, self.num_heads, self.head_dim)
        out = self.o_norm.forward(out.view(n, self.num_heads, self.head_dim), gate)
        return self.o_proj.forward(out.reshape(n, self.projection_size))


__all__ = ["KimiDeltaAttention", "KimiMLAAttention"]
