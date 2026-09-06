"""GLM-5.2 Multi-head Latent Attention with DSA (DeepSeek Sparse Attention).

MLA weight-absorption (as in vLLM/SGLang DeepSeek): instead of materializing per-head
K/V, absorb ``kv_b`` into the query (nope part -> kv_lora_rank) and onto the output
(latent -> v_head_dim). The paged pool stores only the latent ``ckv (kv_lora_rank) | kpe
(qk_rope_head_dim)`` per token, ~28x smaller than full K/V.

DSA (faithful to the HF reference ``modeling_glm_moe_dsa``): each query attends only its
top-``index_topk`` (2048) history tokens, scored by a lightweight indexer. GLM-5.2's
IndexShare scheme puts a full indexer on a subset of layers (``config.indexer_types``,
one "full" leader per group); "shared" layers reuse the leader's selection. The indexer
scores ``sum_h w_h(x) * relu(q_h . k) * head_dim**-0.5`` with q from the shared
``q_a`` latent (``wq_b``), k from the hidden state (``k_norm(wk(x))``), and — unlike
the main MLA rope, which is interleaved — NON-interleaved (NeoX half-split) rope on the
first ``qk_rope_head_dim`` dims of both (the HF reference is explicit about this).
Selection and the sparse attention itself live in the ``dsa`` backend; this module
owns the indexer projections and hands the backend the per-token (q, k, weights).

For ``kv_len <= index_topk`` the top-k covers every live token, so DSA equals dense
MLA exactly — the backend serves short contexts through the same Triton kernel's
identity-selection path (no scoring/top-k) and switches to real selection only where
it can differ.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearReplicated, RMSNorm
from freetoken.layers.rotary import get_rope
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


def _make_proj(quant: str, in_features: int, out_features: int) -> BaseOP:
    """A resident projection in the model's resolved quant mode: ``"fp8_pertensor"``
    (W8A16, per-row scale, quantized at load -- see weight.py) or bf16."""
    if quant == "fp8_pertensor":
        from freetoken.kernel.triton.fp8_pertensor_linear import Fp8PerTensorLinear

        return Fp8PerTensorLinear(in_features, out_features, has_bias=False)
    return LinearReplicated(in_features, out_features, has_bias=False)


class _IdxLayerNorm(BaseOP):
    """LayerNorm with bias (the indexer's ``k_norm``; RMSNorm everywhere else)."""

    def __init__(self, size: int, eps: float = 1e-6) -> None:
        self.eps = eps
        self.weight = torch.empty(size)
        self.bias = torch.empty(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class GlmDsaIndexer(BaseOP):
    """DSA lightning indexer ("full" layers only; "shared" layers carry no weights).

    Kept bf16 in every quant mode: the projections are small (~17 MB/layer) and the
    top-k boundary is precision-sensitive. The HF reference itself is the bf16
    equivalent of DeepSeek's fp8 scoring kernel, so bf16 scoring is faithful.
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        args = config.glm_dsa_args
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.softmax_scale = args.index_head_dim**-0.5
        self.wq_b = LinearReplicated(args.q_lora_rank, self.n_heads * self.head_dim, has_bias=False)
        self.wk = LinearReplicated(args.hidden_size, self.head_dim, has_bias=False)
        self.k_norm = _IdxLayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = LinearReplicated(args.hidden_size, self.n_heads, has_bias=False)
        # Partial rope on the first qk_rope_head_dim dims of the 128-dim index heads,
        # same frequency table as the main rope. The convention is CONFIG-DRIVEN
        # (sglang: is_neox_style = not indexer_rope_interleave): GLM-5.2 sets
        # indexer_rope_interleave=true -> interleaved, where DeepSeek-V3.2 defaults
        # half-split. transformers >= 5.13 is explicit about this difference; <= 5.12
        # applied half-split for GLM too, which selects the wrong top-k past 2048.
        self._rope = get_rope(
            head_dim=self.head_dim,
            rotary_dim=args.qk_rope_head_dim,
            max_position=args.max_position,
            base=args.rope_theta,
            is_neox=not args.indexer_rope_interleave,
        )

    def compute(
        self, x: torch.Tensor, q_resid: torch.Tensor, positions: torch.Tensor
    ) -> "DSAIndexerInputs":
        """Per-token indexer projections: q [T, H, D], k [T, D], weights [T, H] fp32."""
        from freetoken.attention.dsa import DSAIndexerInputs

        t = x.shape[0]
        q = self.wq_b.forward(q_resid).view(t, self.n_heads * self.head_dim)
        k = self.k_norm.forward(self.wk.forward(x)).view(t, self.head_dim)
        q, k = self._rope.forward(positions, q, k)
        w = self.weights_proj.forward(x).float() * (self.n_heads**-0.5)
        return DSAIndexerInputs(q=q.view(t, self.n_heads, self.head_dim), k=k, w=w)


class GlmMoeDsaAttention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        args = config.glm_dsa_args
        self.layer_id = layer_id
        # IndexShare: "full" layers own an indexer; "shared" layers reuse the most
        # recent full layer's top-k selection (resolved in the backend).
        idx_types = args.indexer_types
        self.indexer = (
            GlmDsaIndexer(config, layer_id)
            if idx_types and idx_types[layer_id] == "full"
            else None
        )
        self.num_heads = args.num_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        # Interleaved (GPT-J style, ``rope_interleave``) rope on the 64-dim rope
        # halves, through the shared cache-based path (flashinfer kernel when
        # installed, in-repo Triton drop-in otherwise). ``get_rope`` sizes the fp32
        # cos/sin table from the model's max_position (1M -> ~256 MB, same
        # documented cost as DSV4's freqs table) and shares the one instance
        # across all layers via its cache.
        self.rope = get_rope(
            head_dim=args.qk_rope_head_dim,
            rotary_dim=args.qk_rope_head_dim,
            max_position=args.max_position,
            base=args.rope_theta,
            is_neox=not args.rope_interleave,  # GLM-5.2: interleaved (config-driven)
        )

        quant = config.attn_quant
        self.q_a_proj = _make_proj(quant, args.hidden_size, args.q_lora_rank)
        self.q_a_layernorm = RMSNorm(args.q_lora_rank, eps=args.norm_eps)
        self.q_b_proj = _make_proj(quant, args.q_lora_rank, self.num_heads * self.qk_head_dim)
        self.kv_a_proj_with_mqa = _make_proj(
            quant, args.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=args.norm_eps)
        # kv_b stays bf16 in every mode: it is consumed as bmm operands by the MLA
        # absorption (below), not through a Linear forward.
        self.kv_b_proj = LinearReplicated(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            has_bias=False,
        )
        self.o_proj = _make_proj(quant, self.num_heads * self.v_head_dim, args.hidden_size)
        # Contiguous per-head kv_b split, cached on first forward (see _kv_b). Absorbing
        # kv_b into Q/O runs as a per-head bf16 bmm on these instead of re-slicing +
        # re-upcasting the bf16 weight per token (bf16 einsum with tensor-core fp32
        # accumulate is numerically identical to the fp32 upcast, at ~2x less HBM traffic).
        self._w_uk: torch.Tensor | None = None
        self._w_uv: torch.Tensor | None = None

    def _kv_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-head kv_b split, cached once in bmm-ready bf16 layout:
        W_uk [H, qk_nope, kv_lora] (for q_nope[H,T,nope] @ W_uk -> [H,T,lora]) and
        W_uv_t [H, kv_lora, v_head] (for o_latent[H,T,lora] @ W_uv_t -> [H,T,v]).

        Materialized for every layer by the model's ``prepare_for_runtime()`` (post-
        load, PRE-KV-sizing, so the ~28 MB/layer repack is budgeted by
        ``solve_num_pages`` instead of overcommitting it after the fact), which
        also FREES the checkpoint-layout original -- ``kv_b_proj.weight`` is dead
        after the split (forward only touches the repacked forms; it exists only as
        the strict-load target). Same pattern as gpt_oss's ``_ensure_decode_weights``.
        The lazy branch stays for direct/test use of the module without the hook.
        """
        if self._w_uk is None:
            w = self.kv_b_proj.weight.view(
                self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
            )
            self._w_uk = w[:, : self.qk_nope_head_dim, :].contiguous()
            self._w_uv = w[:, self.qk_nope_head_dim :, :].transpose(1, 2).contiguous()
        return self._w_uk, self._w_uv

    def prepare_for_runtime(self) -> None:
        self._kv_b()
        self.kv_b_proj.weight = None  # checkpoint layout freed; repacked forms serve

    @nvtx_annotate("MLA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        t = x.shape[0]
        w_uk, w_uv = self._kv_b()

        q_a_resid = self.q_a_layernorm.forward(self.q_a_proj.forward(x))
        q = self.q_b_proj.forward(q_a_resid)
        q = q.view(t, self.num_heads, self.qk_head_dim)
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv = self.kv_a_proj_with_mqa.forward(x)
        c_kv, k_rope = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_a_layernorm.forward(c_kv)

        # The shared inplace kernel wants [T, heads*head_size] with head_size == the
        # rotated width (64); q_rope is a non-contiguous split view, so reshape copies
        # it into the buffer the kernel then mutates.
        rope_dim = self.qk_rope_head_dim
        q_rope = q_rope.reshape(t, self.num_heads * rope_dim)
        k_rope = k_rope.reshape(t, rope_dim)
        q_rope, k_rope = self.rope.forward(ctx.batch.positions, q_rope, k_rope)
        q_rope = q_rope.view(t, self.num_heads, rope_dim)

        # Absorb kv_b's k-part into the query: q_nope[H,T,nope] @ W_uk[H,nope,lora].
        q_absorbed = torch.bmm(q_nope.transpose(0, 1).contiguous(), w_uk).transpose(0, 1)

        # DSA: full layers hand the backend this token's indexer projections (the
        # backend caches the keys, scores the history, and selects top-k); shared
        # layers pass None and reuse their group leader's selection.
        indexer_inputs = (
            self.indexer.compute(x, q_a_resid, ctx.batch.positions)
            if self.indexer is not None and getattr(ctx.attn_backend, "dsa_enabled", False)
            else None
        )

        # The pool scatters the two latent halves (c_kv | k_rope) directly -- no
        # concatenated latent copy on the hot path.
        o_latent = ctx.attn_backend.mla_forward(
            q_absorbed.contiguous(), q_rope.contiguous(), c_kv.contiguous(),
            k_rope.contiguous(), self.layer_id, ctx.batch, indexer_inputs=indexer_inputs,
        )  # [T, H, kv_lora_rank]

        # Absorb kv_b's v-part onto the output: o_latent[H,T,lora] @ W_uv_t[H,lora,v].
        o = torch.bmm(o_latent.transpose(0, 1).contiguous(), w_uv).transpose(0, 1)
        return self.o_proj.forward(o.reshape(t, self.num_heads * self.v_head_dim))


__all__ = ["GlmMoeDsaAttention"]
