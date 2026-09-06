"""GLM-5.3-Flash NoPE Multi-head Latent Attention with a kpool DSA indexer.

MLA weight-absorption as in glm_moe_dsa (kv_b absorbed into Q and onto the
output; the paged pool stores one latent row per token) with two GLM-5.3
differences:

* **NoPE** (``mla_use_nope``, ``qk_rope_head_dim == 0``): no rotary embedding
  anywhere in the main attention -- Q is all-nope [T, H, 256], the latent row is
  bare ckv (512, no kpe half). All rope plumbing degenerates to zero-width
  tensors, which the DSA backend's cat/scatter handle natively. Positional
  information enters ONLY through the indexer's pool-compression APE.
* **kpool indexer**: every DSA layer owns its indexer (no IndexShare). The
  indexer K cache stores one entry per ``index_kpool`` (4) tokens: a per-channel
  ``softmax(gate + ape)``-weighted sum of the raw keys. Scoring runs at pool
  granularity (select_k = index_topk / kpool) and the selected pools expand back
  to token rows, with the in-progress tail pool force-included
  (``index_kpool_always_select_tail``). This module owns the PROJECTIONS (wq_b /
  wk+k_norm / weights_proj / compress gate + APE); pooling, scoring, selection
  and the tail buffer live in the backend (attention/dsa_indexer_kpool.py).

Faithfulness note: the reference stack stores pooled entries as
Hadamard-rotated fp8 (a quantization device; the rotation cancels in the dot
product). FreeToken stores pooled entries in bf16 -- mathematically the same
score with strictly less quantization error -- matching the GLM-5.2 precedent
of bf16 indexer keys.
TODO: fp8 index slab (+ Hadamard rotation, upstream parity) to halve the slab bytes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearReplicated, RMSNorm
# Shared with GLM-5.2 (weight.py imports privately from the same package).
from freetoken.models.glm_moe_dsa.attention import _IdxLayerNorm, _make_proj
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Glm5NextIndexer(BaseOP):
    """kpool DSA indexer projections (every DSA layer owns one; no rope -- the
    checkpoint's NoPE geometry leaves ``qk_rope_head_dim == 0`` so position enters
    only via the compression APE).

    Kept bf16 in every quant mode: small (~17 MB/layer) and the top-k boundary is
    precision-sensitive (same reasoning as glm_moe_dsa's indexer).
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        args = config.glm5_args
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.kpool = args.index_kpool
        self.wq_b = LinearReplicated(
            args.q_lora_rank, self.n_heads * self.head_dim, has_bias=False
        )
        self.wk = LinearReplicated(args.hidden_size, self.head_dim, has_bias=False)
        self.k_norm = _IdxLayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = LinearReplicated(
            args.hidden_size, self.n_heads, has_bias=False
        )
        # Pool-compression parameters (checkpoint names, no ".weight" suffix on
        # the gate: it is stored as a bare [head_dim, hidden] tensor).
        self.index_kpool_compress_gate = torch.empty(
            self.head_dim, args.hidden_size
        )
        # Per-pool-slot position bias, fp32 (models/weight.py exempts it from the
        # model-dtype downcast alongside A_log/dt_bias).
        self.index_kpool_compress_ape = torch.empty(
            self.kpool, self.head_dim, dtype=torch.float32
        )

    def compute(self, x: torch.Tensor, q_resid: torch.Tensor) -> "DSAIndexerInputs":
        """Per-token indexer projections as a typed inputs object: q [T, Hi, Di],
        k [T, Di], weights [T, Hi] fp32, plus the kpool gate scores [T, Di] and
        the [kpool, Di] APE parameter (passed per call; the backend keeps no copy)."""
        from freetoken.attention.dsa import DSAIndexerInputs

        t = x.shape[0]
        q = self.wq_b.forward(q_resid).view(t, self.n_heads, self.head_dim)
        k = self.k_norm.forward(self.wk.forward(x))
        w = self.weights_proj.forward(x).float() * (self.n_heads**-0.5)
        gate = torch.nn.functional.linear(x, self.index_kpool_compress_gate)
        return DSAIndexerInputs(
            q=q, k=k, w=w, gate=gate, ape=self.index_kpool_compress_ape
        )


class Glm5NextAttention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        args = config.glm5_args
        self.layer_id = layer_id
        self.indexer = Glm5NextIndexer(config, layer_id)
        self.num_heads = args.num_heads
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim  # 0 (NoPE)
        self.qk_head_dim = args.qk_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        assert args.qk_rope_head_dim == 0 and args.mla_nope, (
            "glm5_next attention implements the NoPE geometry; a roped variant "
            "would need the glm_moe_dsa rope plumbing back"
        )

        quant = config.attn_quant
        self.q_a_proj = _make_proj(quant, args.hidden_size, args.q_lora_rank)
        self.q_a_layernorm = RMSNorm(args.q_lora_rank, eps=args.norm_eps)
        self.q_b_proj = _make_proj(
            quant, args.q_lora_rank, self.num_heads * self.qk_head_dim
        )
        # NoPE: kv_a projects to bare ckv (no +qk_rope_head_dim rows).
        self.kv_a_proj_with_mqa = _make_proj(
            quant, args.hidden_size, self.kv_lora_rank
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=args.norm_eps)
        # kv_b stays bf16 in every mode (bmm absorption operand, not a Linear).
        self.kv_b_proj = LinearReplicated(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            has_bias=False,
        )
        self.o_proj = _make_proj(
            quant, self.num_heads * self.v_head_dim, args.hidden_size
        )
        self._w_uk: torch.Tensor | None = None
        self._w_uv: torch.Tensor | None = None

    def _kv_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-head kv_b split in bmm-ready bf16 layout (same contract and
        prepare_for_runtime budgeting as glm_moe_dsa; see that module)."""
        if self._w_uk is None:
            w = self.kv_b_proj.weight.view(
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
                self.kv_lora_rank,
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
        # NoPE: the whole head is the nope part; no rope split, no rope kernel.
        q_nope = q.view(t, self.num_heads, self.qk_head_dim)

        c_kv = self.kv_a_layernorm.forward(self.kv_a_proj_with_mqa.forward(x))

        # Absorb kv_b's k-part into the query: q_nope[H,T,nope] @ W_uk[H,nope,lora].
        q_absorbed = torch.bmm(q_nope.transpose(0, 1).contiguous(), w_uk).transpose(0, 1)

        indexer_inputs = (
            self.indexer.compute(x, q_a_resid)
            if getattr(ctx.attn_backend, "dsa_enabled", False)
            else None
        )

        # Zero-width rope halves: cat/scatter no-ops in the backend.
        q_pe = q.new_empty(t, self.num_heads, 0)
        k_rope = q.new_empty(t, 0)
        o_latent = ctx.attn_backend.mla_forward(
            q_absorbed.contiguous(), q_pe, c_kv.contiguous(), k_rope,
            self.layer_id, ctx.batch, indexer_inputs=indexer_inputs,
        )  # [T, H, kv_lora_rank]

        # Absorb kv_b's v-part onto the output: o_latent[H,T,lora] @ W_uv_t[H,lora,v].
        o = torch.bmm(o_latent.transpose(0, 1).contiguous(), w_uv).transpose(0, 1)
        return self.o_proj.forward(o.reshape(t, self.num_heads * self.v_head_dim))


__all__ = ["Glm5NextAttention", "Glm5NextIndexer"]
