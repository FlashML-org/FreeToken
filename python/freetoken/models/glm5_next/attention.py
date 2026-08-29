"""GLM-5.3-Flash hybrid-attention mixers.

* full layers (deepseek_sparse_attention) -> glm_moe_dsa's MLA+DSA attention verbatim
  (NoPE: qk_rope_head_dim == 0, rope guards are in glm_moe_dsa/attention.py).
* linear layers (linear_attention) -> KDA (Kimi Delta Attention), verified against the HF
  reference ``Glm5NextTextLinearAttention`` symbol-by-symbol:

    - separate q/k/v projections (served as one merged GEMM; the loader concatenates
      [q|k|v] in HF's own ``torch.cat`` channel order, so the depthwise conv sees the
      exact reference channel layout);
    - depthwise causal conv (kernel 4, no bias, silu) over the qkv concat;
    - ForgetGate: ``g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))``
      -- per-CHANNEL ([heads, head_dim]) log-space decay, computed here in fp32
      EXACTLY as the HF reference (which also precomputes g outside its kernels);
    - ``beta = sigmoid(b_proj(x))`` per-head, fp32 (HF does the same outside);
    - delta rule via the upstream fla KDA kernels (``chunk_kda`` prefill,
      ``fused_recurrent_kda`` decode) with ``use_qk_l2norm_in_kernel=True`` -- the same
      kernels/flags the HF integration uses, fed the same precomputed ``g``/``beta``;
    - output gate ``z = g_b(g_a(x))`` through a sigmoid-gated RMSNorm (HF RMSNormGated
      with activation="sigmoid"), then ``o_proj``.

  State lives in ``ctx.linear_state_pool`` (slot per request). The upstream fla kernels
  take a contiguous ``initial_state`` instead of slot indices, so we gather the slots
  before the call and scatter the final state back after. CUDA-graph hygiene: the
  gather/scatter are ``index_select``/``index_copy_`` on a FIXED-ADDRESS index buffer
  (``fla.cache_indices``) with capture-static shapes -- replays read the refreshed index
  values, same contract the vendored slot-indexed kernels rely on.
"""

from __future__ import annotations

import os as _os

import torch
import torch.nn.functional as F

from freetoken.core import get_global_ctx
from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.layers import BaseOP, LinearColParallelMerged, LinearReplicated
from freetoken.models.config import ModelConfig
from freetoken.models.glm_moe_dsa.attention import GlmMoeDsaAttention
from freetoken.models.qwen3_5_moe.gdn import _DepthwiseConv1d

# Full (MLA + DSA) layers: ModelConfig.glm_dsa_args carries the derived GlmMoeDsaArgs.
FullAttention = GlmMoeDsaAttention


class _SigmoidGatedRMSNorm(BaseOP):
    """RMSNorm(x) * sigmoid(z), fused (HF Glm5NextTextRMSNormGated, activation='sigmoid')."""

    def __init__(self, dim: int, eps: float):
        self.weight = torch.empty(dim)
        self.eps = eps

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.fla import rms_norm_gated

        return rms_norm_gated(
            x=x, weight=self.weight, bias=None, z=z, eps=self.eps,
            is_rms_norm=True, norm_before_gate=True, activation="sigmoid",
        )


class KdaAttention(BaseOP):
    """KDA linear attention for GLM-5.3's 34 linear layers (structure == HF reference)."""

    def __init__(self, config: ModelConfig, layer_id: int):
        a = config.glm5_args
        kda = a.kda
        self.layer_id = layer_id
        self.hidden_size = a.hidden_size
        self.num_heads = kda.num_heads          # 64
        self.head_dim = kda.head_dim            # 128
        self.qkv_dim = self.num_heads * self.head_dim  # 8192
        self.conv_dim = 3 * self.qkv_dim        # 24576 == pool's 2*key_dim+value_dim
        self.conv_kernel_size = kda.short_conv_kernel_size
        self.gate_lower_bound = kda.gate_lower_bound  # -5.0

        # Merged [q|k|v] projection -- HF cat order, so conv channels match the reference.
        # KDA fp8 (per-row): only the two big GEMMs; q/k pass the kernel's l2norm, so
        # their quant scale error largely cancels.
        if a.kda_fp8:
            from freetoken.kernel.triton.fp8_pertensor_linear import (
                Fp8PerTensorColMerged, Fp8PerTensorLinear,
            )

            self.in_proj_qkv = Fp8PerTensorColMerged(
                self.hidden_size, [self.qkv_dim, self.qkv_dim, self.qkv_dim], has_bias=False
            )
        else:
            self.in_proj_qkv = LinearColParallelMerged(
                self.hidden_size, [self.qkv_dim, self.qkv_dim, self.qkv_dim], has_bias=False
            )
        self.conv1d = _DepthwiseConv1d(self.conv_dim, self.conv_kernel_size)

        # ForgetGate (low-rank) + input gate + output gate (low-rank) -- bf16 GEMMs,
        # gate math in fp32 (matches HF's .float() upcasts).
        self.f_a_proj = LinearReplicated(self.hidden_size, self.head_dim, has_bias=False)
        self.f_b_proj = LinearReplicated(self.head_dim, self.qkv_dim, has_bias=False)
        self.dt_bias = torch.empty(self.qkv_dim, dtype=torch.float32)
        self.A_log = torch.empty(self.num_heads, dtype=torch.float32)
        self.b_proj = LinearReplicated(self.hidden_size, self.num_heads, has_bias=False)
        self.g_a_proj = LinearReplicated(self.hidden_size, self.head_dim, has_bias=False)
        self.g_b_proj = LinearReplicated(self.head_dim, self.qkv_dim, has_bias=False)
        self.o_norm = _SigmoidGatedRMSNorm(self.head_dim, eps=a.norm_eps)
        if a.kda_fp8:
            from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorLinear

            self.o_proj: object = Fp8PerTensorLinear(
                self.qkv_dim, self.hidden_size, has_bias=False
            )
        else:
            self.o_proj = LinearReplicated(self.qkv_dim, self.hidden_size, has_bias=False)

    # ---- gate math: EXACT HF ForgetGate (fp32) --------------------------------------
    def _forget_gate(self, x: torch.Tensor) -> torch.Tensor:
        """[total, hidden] -> per-channel log-decay g [total, heads, head_dim] fp32."""
        raw = self.f_b_proj.forward(self.f_a_proj.forward(x))
        g = raw.float() + self.dt_bias.view(1, -1)
        g = g.view(-1, self.num_heads, self.head_dim)
        decay = torch.exp(self.A_log.float()).view(1, self.num_heads, 1)
        return self.gate_lower_bound * torch.sigmoid(decay * g)

    # ---- fused gate path (2026-08-28): merged stage-1 GEMV + batched stage-2 bmm ----
    # f_a/g_a/b share input x -> one [2*hd+H, hidden] GEMV; f_b/g_b -> one bmm.
    # 5 cublas calls + ~7 fp32 elementwise per layer collapse to 3 kernels.
    def _build_gate_merge(self) -> None:
        w1 = torch.cat(
            [self.f_a_proj.weight, self.g_a_proj.weight, self.b_proj.weight], 0
        ).contiguous()                                      # [2*hd + H, hidden]
        w2t = torch.stack(
            [self.f_b_proj.weight, self.g_b_proj.weight]
        ).transpose(1, 2).contiguous()                      # [2, hd, qkv]
        decay = torch.exp(self.A_log.float()).contiguous()  # [H], cached (was per-step)
        self._gate_merged = (w1, w2t, decay)

    def _gates_fused(self, x: torch.Tensor):
        """Same math as _forget_gate + b/g chains. Lazy build runs on the first
        (uncaptured) forward, so CUDA-graph capture sees only fixed-shape kernels."""
        from freetoken.kernel.triton.kda_gate import kda_gate

        merged = getattr(self, "_gate_merged", None)
        if merged is None:
            self._build_gate_merge()
            merged = self._gate_merged
        w1, w2t, decay = merged
        total = x.shape[0]
        hd = self.head_dim
        s1 = F.linear(x, w1)                                # [total, 2*hd + H]
        # [2, total, hd] view over s1 cols [0:2*hd] -- no copy (strided bmm input)
        a2 = s1.as_strided((2, total, hd), (hd, s1.stride(0), 1))
        r2 = torch.bmm(a2, w2t)                             # [2, total, qkv]
        g, beta = kda_gate(
            r2[0], s1[:, 2 * hd:], self.dt_bias, decay, self.gate_lower_bound)
        return g, beta, r2[1]

    # ---- conv helpers (same pool contract as the GDN path) --------------------------
    def _conv_weight(self) -> torch.Tensor:
        return self.conv1d.weight.squeeze(1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        import os as _os
        if _os.environ.get("FREETOKEN_GLM5_STUB_KDA", "0") == "1":  # ablation timing only
            return torch.zeros_like(hidden_states)
        from fla.ops.kda import chunk_kda, fused_recurrent_kda

        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        total = hidden_states.shape[0]
        dtype = hidden_states.dtype

        fla_md = batch.fla_metadata
        if fla_md is None:
            from freetoken.attention.linear import build_fla_metadata

            fla_md = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla_md

        conv_in = self.in_proj_qkv.forward(hidden_states)  # [total, 3*qkv] = [q|k|v]
        if _os.environ.get("FREETOKEN_KDA_FUSED_GATE", "1") == "1":
            g, beta, z = self._gates_fused(hidden_states)  # ~12 kernels -> 3
        else:
            g = self._forget_gate(hidden_states)               # [total, H, K] fp32
            beta = torch.sigmoid(self.b_proj.forward(hidden_states).float())  # [total, H]
            z = self.g_b_proj.forward(self.g_a_proj.forward(hidden_states))   # [total, qkv]

        li = pool.local_index(self.layer_id)
        rec = pool.recurrent_states[li]  # [slots, H, K, V] fp32
        idx = fla_md.cache_indices
        # index_select/index_copy_ need int64; the cast kernel re-reads the (refreshed)
        # int32 buffer on every CUDA-graph replay, so this stays graph-safe.
        idx_l = idx.long()

        if getattr(batch, "spec_stash", None) is not None:
            # Speculative VERIFY: the batch is k+1 single-token fake reqs of ONE request
            # at sequential positions. Run conv+delta as ONE varlen sequence (exact
            # sequential semantics) and archive inputs + pre-states so spec_commit can
            # re-derive the state at the accepted prefix (the fla kernels only expose
            # the final state).
            dev = conv_in.device
            # Slot id via the REFRESHED device buffer (graph-safe: a captured verify
            # graph re-reads it each replay; a host int would be baked at capture).
            if batch.linear_table_idx is not None:
                idx1 = batch.linear_table_idx[:1]
            else:
                idx1 = torch.tensor(
                    [batch.reqs[0].table_idx], dtype=torch.int32, device=dev)
            idx1_l = idx1.long()
            # Constants cached per verify width: torch.tensor(list) is an UNPINNED H2D
            # copy, illegal inside CUDA graph capture. The warm (uncaptured) forward
            # populates the cache; the captured run hits it with zero host traffic.
            consts = getattr(self, "_spec_consts", None)
            if consts is None:
                consts = self._spec_consts = {}
            if total not in consts:
                consts[total] = (
                    torch.tensor([0, total], dtype=torch.int32, device=dev),
                    torch.ones(1, dtype=torch.bool, device=dev),
                )
            cu, has_init = consts[total]
            conv_before = pool.conv_states[li].index_select(0, idx1_l).clone()
            rec_before = rec.index_select(0, idx1_l)  # a copy
            x_c = conv_in.transpose(0, 1).contiguous()
            mixed = causal_conv1d_varlen(
                x_c, self._conv_weight(), pool.conv_states[li], cu, idx1, has_init,
            ).transpose(0, 1)
            qf, kf, vf = torch.split(mixed, [self.qkv_dim] * 3, dim=-1)
            q = qf.reshape(1, total, self.num_heads, self.head_dim).to(dtype)
            k = kf.reshape(1, total, self.num_heads, self.head_dim).to(dtype)
            v = vf.reshape(1, total, self.num_heads, self.head_dim).to(dtype)
            gv = g.view(1, total, self.num_heads, self.head_dim)
            bv = beta.view(1, total, self.num_heads)
            # fused_recurrent (not chunk): the SAME sequential-scan kernel the decode
            # path uses, so verify outputs / spec_commit / plain decode stay bit-aligned.
            core, state = fused_recurrent_kda(
                q=q, k=k, v=v, g=gv, beta=bv,
                scale=self.head_dim ** -0.5,
                initial_state=rec_before.clone(), output_final_state=True,
                use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
            )
            rec.index_copy_(0, idx1_l, state.to(rec.dtype))
            batch.spec_stash[self.layer_id] = dict(
                q=q, k=k, v=v, g=gv, beta=bv,
                rec_before=rec_before, conv_before=conv_before, conv_in=conv_in,
            )
        elif batch.is_decode:
            mixed = causal_conv1d_decode(conv_in, pool.conv_states[li], self._conv_weight(), idx)
            B = mixed.shape[0]
            qf, kf, vf = torch.split(mixed, [self.qkv_dim] * 3, dim=-1)
            q = qf.reshape(1, B, self.num_heads, self.head_dim).to(dtype)
            k = kf.reshape(1, B, self.num_heads, self.head_dim).to(dtype)
            v = vf.reshape(1, B, self.num_heads, self.head_dim).to(dtype)
            state = rec.index_select(0, idx_l)  # gather (fixed-address idx buffer: graph-safe)
            core, state = fused_recurrent_kda(
                q=q, k=k, v=v,
                g=g.view(1, B, self.num_heads, self.head_dim),
                beta=beta.view(1, B, self.num_heads),
                scale=self.head_dim ** -0.5,
                initial_state=state, output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                # varlen: B one-token sequences (cu_seqlens = arange(B+1)); without this the
                # kernel would treat the batch as ONE sequence of B timesteps and return a
                # single merged state (caught by the bs=2 graph-capture regression).
                cu_seqlens=fla_md.cu_seqlens,
            )
            rec.index_copy_(0, idx_l, state.to(rec.dtype))  # scatter back
        else:
            if fla_md.fresh_state_indices is not None:
                rec.index_fill_(0, fla_md.fresh_state_indices, 0.0)
            x_c = conv_in.transpose(0, 1).contiguous()
            mixed = causal_conv1d_varlen(
                x_c, self._conv_weight(), pool.conv_states[li],
                fla_md.cu_seqlens, idx, fla_md.has_initial_state,
            ).transpose(0, 1)
            qf, kf, vf = torch.split(mixed, [self.qkv_dim] * 3, dim=-1)
            q = qf.reshape(1, total, self.num_heads, self.head_dim).to(dtype)
            k = kf.reshape(1, total, self.num_heads, self.head_dim).to(dtype)
            v = vf.reshape(1, total, self.num_heads, self.head_dim).to(dtype)
            state = rec.index_select(0, idx_l)
            core, state = chunk_kda(
                q=q, k=k, v=v,
                g=g.view(1, total, self.num_heads, self.head_dim),
                beta=beta.view(1, total, self.num_heads),
                scale=self.head_dim ** -0.5,
                initial_state=state, output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=fla_md.cu_seqlens,
            )
            rec.index_copy_(0, idx_l, state.to(rec.dtype))

        core = core.reshape(-1, self.head_dim)
        z = z.reshape(-1, self.head_dim)
        out = self.o_norm.forward(core, z).reshape(total, -1)
        return self.o_proj.forward(out.to(dtype))

    @torch.inference_mode()
    def spec_restore(self, st: dict, slot: int) -> None:
        """Roll this layer back to the archived pre-verify state (KL replay mode)."""
        pool = get_global_ctx().linear_state_pool
        li = pool.local_index(self.layer_id)
        pool.recurrent_states[li][slot] = st["rec_before"][0]
        pool.conv_states[li][slot] = st["conv_before"][0]

    @torch.inference_mode()
    def spec_commit(self, st: dict, keep: int, slot: int) -> None:
        """Re-commit recurrent+conv state at the accepted prefix (``keep`` of the
        archived verify tokens), from the archived pre-verify state. Post-conv
        q/k/v of accepted tokens are causal (independent of rejected ones), so a
        single fused_recurrent over the archive is exact."""
        from fla.ops.kda import fused_recurrent_kda

        ctx = get_global_ctx()
        pool = ctx.linear_state_pool
        li = pool.local_index(self.layer_id)
        cu = torch.tensor([0, keep], dtype=torch.int32, device=st["q"].device)
        _, state = fused_recurrent_kda(
            q=st["q"][:, :keep].contiguous(), k=st["k"][:, :keep].contiguous(),
            v=st["v"][:, :keep].contiguous(),
            g=st["g"][:, :keep].contiguous(), beta=st["beta"][:, :keep].contiguous(),
            scale=self.head_dim ** -0.5,
            initial_state=st["rec_before"].clone(), output_final_state=True,
            use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
        )
        rec = pool.recurrent_states[li]
        rec[slot] = state[0].to(rec.dtype)
        win = torch.cat([st["conv_before"][0], st["conv_in"][:keep].t()], dim=1)
        pool.conv_states[li][slot] = win[:, -(self.conv_kernel_size - 1):]


__all__ = ["FullAttention", "KdaAttention"]
