"""GLM-5.3-Flash (glm5_next) model: hybrid KDA/DSA decoder with mHC residual streams.

Layer layout comes from the checkpoint's ``layer_types`` (34 KDA linear-attention
layers, 11 NoPE-MLA/DSA layers at 3:1) and ``mlp_layer_types`` (3 dense + 42 MoE).
The residual stream is mHC-widened to ``hc_mult`` (4) parallel streams:

    layer 0:  residual = hc_expand(x);  (post, comb, x) = mhc_pre(residual, hc_attn_*)
    each sublayer boundary fuses the previous hc_post with the next hc_pre
    (mhc_fused_post_pre), and the sublayer input is RMS-normed AFTER the mix
    (the reference fuses the norm into its hc kernels; decomposed here, same math).
    last layer: x = mhc_post(...); x = hc_contract(x)  -> final norm -> lm_head.

The deferred (post, comb) pair threads through the layer loop exactly like
glm_moe_dsa's (x, residual) pair. lm_head quant mirrors glm_moe_dsa (optional
W8A16 fp8 at load; the ~1.2 GiB bf16 head is read every decode step).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
)
from freetoken.layers.mhc import hc_contract, hc_expand, mhc_fused_post_pre, mhc_post, mhc_pre
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Glm5NextAttention
from .kda import Glm5NextKDA
from .mlp import Glm5NextGatedMLP
from .moe import Glm5NextSparseBlock

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Glm5Fp8LMHead(ParallelLMHead):
    """W8A16 lm_head (fp8-e4m3 weight + per-row scale, quantized at load); the
    full-vocab decode GEMV reads the whole head every step -- fp8 halves it.
    Same contract as glm_moe_dsa's GlmFp8LMHead."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim, tie_word_embeddings=False)
        self.weight = torch.empty(num_embeddings, embedding_dim, dtype=torch.float8_e4m3fn)
        self.weight_scale = torch.empty(num_embeddings, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        return fp8_pertensor_linear(x, self.weight, self.weight_scale)


class Glm5NextDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        args = config.glm5_args
        self._layer_id = layer_id
        self._is_last = layer_id == config.num_layers - 1
        self.mhc = args.mhc
        self._n = args.mhc_num_residual_streams
        self._hc_eps = args.hc_eps
        self._rms_eps = args.norm_eps
        self._post_mult = args.mhc_post_mult_value
        self._sinkhorn = args.mhc_sinkhorn_iterations

        if args.is_kda_layer(layer_id):
            self.self_attn: BaseOP = Glm5NextKDA(config, layer_id)
        else:
            self.self_attn = Glm5NextAttention(config, layer_id)
        if layer_id >= config.first_k_dense_replace:
            self.mlp: BaseOP = Glm5NextSparseBlock(config, layer_id)
        else:
            self.mlp = Glm5NextGatedMLP(
                config.hidden_size, config.intermediate_size,
                quant=config.dense_quant, swiglu_limit=config.swiglu_limit,
            )
        self.input_layernorm = RMSNorm(size=config.hidden_size, eps=args.norm_eps)
        self.post_attention_layernorm = RMSNorm(size=config.hidden_size, eps=args.norm_eps)

        if self.mhc:
            n, hidden = self._n, config.hidden_size
            mix = 2 * n + n * n
            # fp32 mHC weights (models/weight.py exempts hc_* from the dtype downcast).
            self.hc_attn_fn = torch.empty(mix, n * hidden, dtype=torch.float32)
            self.hc_attn_base = torch.empty(mix, dtype=torch.float32)
            self.hc_attn_scale = torch.empty(3, dtype=torch.float32)
            self.hc_ffn_fn = torch.empty(mix, n * hidden, dtype=torch.float32)
            self.hc_ffn_base = torch.empty(mix, dtype=torch.float32)
            self.hc_ffn_scale = torch.empty(3, dtype=torch.float32)

    def _pre(self, residual, fn, scale, base):
        # Layer 0's standalone pre rides the fused kernel too (HAS_POST=False
        # path; x/post/comb are the no-post sentinels) -- same dispatch, same
        # numerics, and the kernel wins at every batch size (see layers/mhc.py).
        if residual.is_cuda:
            _, post, comb, x = mhc_fused_post_pre(
                residual.new_empty(residual.shape[0], residual.shape[-1]),
                residual, None, None, fn, scale, base,
                self._rms_eps, self._hc_eps, self._post_mult, self._sinkhorn,
            )
            return post, comb, x
        return mhc_pre(
            residual, fn, scale, base,
            self._rms_eps, self._hc_eps, self._post_mult, self._sinkhorn,
        )

    def _fused(self, x, residual, post, comb, fn, scale, base):
        return mhc_fused_post_pre(
            x, residual, post, comb, fn, scale, base,
            self._rms_eps, self._hc_eps, self._post_mult, self._sinkhorn,
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        post: torch.Tensor | None,
        comb: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if post is None:
            if residual is None:
                residual = hc_expand(x, self._n)
            post, comb, x = self._pre(
                residual, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
            )
        else:
            residual, post, comb, x = self._fused(
                x, residual, post, comb,
                self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
            )
        x = self.input_layernorm.forward(x)
        x = self.self_attn.forward(x)

        residual, post, comb, x = self._fused(
            x, residual, post, comb,
            self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
        )
        x = self.post_attention_layernorm.forward(x)
        x = self.mlp.forward(x)

        if self._is_last:
            x = mhc_post(x, residual, post, comb)
            return hc_contract(x), None, None, None
        return x, residual, post, comb


class Glm5NextModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Glm5NextDecoderLayer(config, i) for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual = post = comb = None
        for layer in self.layers.op_list:
            x, residual, post, comb = layer.forward(x, residual, post, comb)
        return self.norm.forward(x)


class Glm5NextForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self._config = config
        self.model = Glm5NextModel(config)
        if config.lm_head_quant == "fp8_pertensor" and not config.tie_word_embeddings:
            self.lm_head: BaseOP = Glm5Fp8LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )

    def prepare_for_runtime(self) -> None:
        """Post-load, pre-KV-sizing hook: materialize the DSA layers' bmm-ready
        kv_b splits and free the checkpoint-layout originals (glm_moe_dsa
        precedent)."""
        for layer in self.model.layers.op_list:
            if isinstance(layer.self_attn, Glm5NextAttention):
                layer.self_attn.prepare_for_runtime()
        torch.cuda.empty_cache()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["Glm5NextForCausalLM"]
