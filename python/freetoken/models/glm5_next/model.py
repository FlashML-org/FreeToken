"""GLM-5.3-Flash (glm5_next) model: DSV4-style manifold-constrained Hyper-Connections (mHC,
4 residual streams) wrapping a hybrid per-layer mixer -- KDA linear attention (linear layers)
or MLA+DSA (full layers) -- plus a GLM MoE / dense MLP feed-forward.

Reuse map:
  * mHC (hc_pre / hc_post / hc_head + sinkhorn) -> DSV4 kernels verbatim; hc_mult == 4.
  * full-layer mixer -> glm_moe_dsa GlmMoeDsaAttention (via FullAttention).
  * linear-layer mixer -> KdaAttention (attention.py).
  * MoE / dense MLP / FP8 lm_head -> glm_moe_dsa GlmMoeDsaSparseBlock / GlmDsaGatedMLP / GlmFp8LMHead.

Hidden state carries the 4 residual streams as ``[total_tokens, hc_mult, dim]`` (ragged-flat,
matching the glm_moe_dsa/KDA mixers which consume ``[total, dim]``). hc_pre collapses the
streams to ``[total, dim]`` for the mixer; hc_post expands back. FIRST PASS: the ragged-flat
shape handling is verified by a dummy-weight forward before real weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv4.hc import hc_post_combine, hc_pre_combine
from freetoken.kernel.triton.dsv4.hc_norm import hc_mix_gemv, hc_rms_cast
from freetoken.kernel.triton.dsv4.sinkhorn import hc_split_sinkhorn
from freetoken.layers import (
    BaseOP,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.config import ModelConfig
from freetoken.models.glm_moe_dsa.model import GlmFp8LMHead

from .mlp import Glm5ClampedMLP
from .moe import Glm5SparseBlock

from .attention import FullAttention, KdaAttention


class Glm5DecoderLayer(BaseOP):
    """mHC block: hc_pre -> norm -> mixer -> hc_post ; hc_pre -> norm -> ffn -> hc_post.
    ``mixer`` is KDA (linear layers) or MLA+DSA (full); ``ffn`` is MoE or dense MLP."""

    def __init__(self, config: ModelConfig, layer_id: int):
        a = config.glm5_args
        self.dim = config.hidden_size
        self.norm_eps = config.rms_norm_eps
        self.layer_id = layer_id

        if a.is_kda_layer(layer_id):
            self.self_attn: BaseOP = KdaAttention(config, layer_id)
        else:
            self.self_attn = FullAttention(config, layer_id)

        if a.is_dense_layer(layer_id):
            self.mlp: BaseOP = Glm5ClampedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                quant=config.dense_quant,
                limit=float(config.swiglu_limit or 10.0),
            )
        else:
            self.mlp = Glm5SparseBlock(config, layer_id)

        # Attribute names match the checkpoint layout (weight.py yields these paths).
        self.input_layernorm = RMSNorm(size=self.dim, eps=self.norm_eps)
        self.post_attention_layernorm = RMSNorm(size=self.dim, eps=self.norm_eps)

        # Hyper-connection mix (DSV4 layout): hc_mult == a.hc_mult (4).
        self.hc_mult = hc = a.hc_mult
        self.hc_sinkhorn_iters = a.hc_sinkhorn_iters
        self.hc_eps = a.hc_eps
        mix_hc = (2 + hc) * hc
        hc_dim = hc * self.dim
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False)
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False)
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32), requires_grad=False)
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32), requires_grad=False)
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32), requires_grad=False)
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32), requires_grad=False)

    def hc_pre(self, x, hc_fn, hc_scale, hc_base):
        # x: [total, hc_mult, dim] -> collapsed y: [total, dim] for the mixer.
        total = x.shape[0]
        dtype = x.dtype
        # tail-fusion (2026-08-28): cast+square+mean+rsqrt -> one kernel; gemv with
        # rsqrt epilogue -> one kernel; pre_combine upcasts bf16 in-register (the
        # eager chain was ~7 launches + two 64KB-per-token fp32 casts per call).
        xf, rs = hc_rms_cast(x.reshape(total, self.hc_mult * self.dim), self.norm_eps)
        mixes = hc_mix_gemv(xf, hc_fn, rs)
        pre, post, comb = hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )
        y = hc_pre_combine(x.reshape(total, self.hc_mult, self.dim), pre, dtype)
        return y.reshape(total, self.dim), post, comb

    def hc_post(self, x, residual, post, comb):
        # x: [total, dim] (mixer out); residual: [total, hc_mult, dim] -> [total, hc_mult, dim].
        total = residual.shape[0]
        y = hc_post_combine(
            x.reshape(total, self.dim), residual.reshape(total, self.hc_mult, self.dim), post, comb
        )
        return y.reshape(total, self.hc_mult, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import os as _os
        _stub = _os.environ.get("FREETOKEN_GLM5_STUB_MHC", "0")
        if _stub in ("1", "2", "3"):  # ablation timing only
            # "2": compute hc, discard.  "3": REAL hc_pre feeds the sublayer, simple add out.
            if _stub == "3":
                y, _, _ = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
            else:
                if _stub == "2":
                    self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
                y = self.input_layernorm.forward(x[:, 0])
            if _stub == "3":
                y = self.input_layernorm.forward(y)
            y = self.self_attn.forward(y)
            x = x + y.unsqueeze(1)
            if _stub == "3":
                y, _, _ = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
                y = self.post_attention_layernorm.forward(y)
            else:
                if _stub == "2":
                    self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
                y = self.post_attention_layernorm.forward(x[:, 0])
            y = self.mlp.forward(y)
            return x + y.unsqueeze(1)
        residual = x
        y, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        y = self.input_layernorm.forward(y)
        y = self.self_attn.forward(y)
        x = self.hc_post(y, residual, post, comb)

        residual = x
        y, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        y = self.post_attention_layernorm.forward(y)
        y = self.mlp.forward(y)
        x = self.hc_post(y, residual, post, comb)
        return x


class Glm5Model(BaseOP):
    def __init__(self, config: ModelConfig):
        a = config.glm5_args
        self.dim = config.hidden_size
        self.norm_eps = config.rms_norm_eps
        self.hc_mult = a.hc_mult
        self.hc_eps = a.hc_eps
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList([Glm5DecoderLayer(config, i) for i in range(config.num_layers)])
        self.norm = RMSNorm(size=self.dim, eps=self.norm_eps)

    def hc_head(self, x: torch.Tensor) -> torch.Tensor:
        # GLM-5.3 collapses the final residual streams with an UNWEIGHTED MEAN -- verified
        # against HF Glm5NextTextHyperHead ("Unlike DeepSeek-V4, this is an unweighted
        # mean"); there are no hc_head weights in the checkpoint.
        return x.mean(dim=1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens.forward(input_ids)  # [total, dim]
        h = h.unsqueeze(1).repeat(1, self.hc_mult, 1)  # [total, hc_mult, dim]
        for layer in self.layers.op_list:
            h = layer.forward(h)
        h = self.hc_head(h)  # [total, dim]
        # Stash for the MTP block: DeepSeek-style drafting consumes the collapsed hidden
        # BEFORE the final norm (a ref, not a copy; None-cost when MTP is off).
        self.last_pre_norm = h
        # RMSNorm.forward returns a single tensor (no residual fusion here -- mHC already
        # carried the residual through hc_post). Call it exactly once (graph-safe, no dup work).
        return self.norm.forward(h)


class Glm5NextForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = Glm5Model(config)
        if config.glm5_args.mtp_enabled:
            from .mtp import Glm5MtpBlock

            self.mtp = Glm5MtpBlock(config)
        if config.lm_head_quant == "fp8_pertensor" and not config.tie_word_embeddings:
            self.lm_head: BaseOP = GlmFp8LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        super().__init__()

    def prepare_for_runtime(self) -> None:
        for layer in self.model.layers.op_list:
            if hasattr(layer.self_attn, "prepare_for_runtime"):
                layer.self_attn.prepare_for_runtime()
        torch.cuda.empty_cache()

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        output = self.model.forward(batch.input_ids)
        logits = self.lm_head.forward(output)
        # ---- MTP acceptance probe (measurement only; eager decode, bs==1) ----
        # Compares last step's draft against this step's main-model argmax and drafts the
        # next token. Requires --cuda-graph-max-bs 0 (host-side counters do not replay).
        import os as _os

        if (
            _os.environ.get("FREETOKEN_GLM5_MTP_PROBE", "0") == "1"
            and getattr(self, "mtp", None) is not None
            and batch.is_decode
            and logits.shape[0] == 1
            and not torch.cuda.is_current_stream_capturing()
        ):
            nxt = logits.argmax(dim=-1)  # greedy t_{p+2} truth (temperature-0 runs)
            prev = getattr(self, "_probe_draft", None)
            if prev is not None:
                self._probe_total = getattr(self, "_probe_total", 0) + 1
                self._probe_agree = getattr(self, "_probe_agree", 0) + int(
                    (prev == nxt).item()
                )
                if self._probe_total % 50 == 0:
                    print(
                        f"[mtp-probe] accept={self._probe_agree}/{self._probe_total}"
                        f" = {self._probe_agree/self._probe_total:.3f}",
                        flush=True,
                    )
            emb = self.model.embed_tokens.weight[nxt]
            d_logits = self.mtp.draft_logits(
                self.model.last_pre_norm, emb, self.lm_head
            )
            self._probe_draft = d_logits.argmax(dim=-1)
        return logits


__all__ = ["Glm5NextForCausalLM", "Glm5Model", "Glm5DecoderLayer"]
