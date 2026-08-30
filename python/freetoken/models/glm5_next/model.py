from __future__ import annotations

import torch

from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Glm5NextAttention
from .hc import HyperConnection, collapse_head
from .linear_attention import Glm5NextLinearAttention
from .mlp import Glm5NextMLP
from .moe import Glm5NextSparseBlock


class Glm5NextDecoderLayer(BaseOP):
    def __init__(self, config, layer_id: int) -> None:
        args = config.glm5_args
        self._layer_id = layer_id
        self.self_attn = (
            Glm5NextLinearAttention(config, layer_id)
            if config.is_linear_layer(layer_id)
            else Glm5NextAttention(config, layer_id)
        )
        self.mlp = (
            Glm5NextMLP(
                config.hidden_size, config.intermediate_size, config.swiglu_limit
            )
            if layer_id < config.first_k_dense_replace
            else Glm5NextSparseBlock(config, layer_id)
        )
        hc_kw = {
            "hidden_size": config.hidden_size,
            "hc_mult": args.hc_mult,
            "norm_eps": config.rms_norm_eps,
            "sinkhorn_iters": args.hc_sinkhorn_iters,
            "hc_eps": args.hc_eps,
        }
        self.attn_hc = HyperConnection(**hc_kw)
        self.ffn_hc = HyperConnection(**hc_kw)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x, post, comb = self.attn_hc.mix(hidden)
        y = self.self_attn.forward(self.input_layernorm.forward(x))
        hidden = self.attn_hc.combine(hidden, y, post, comb)
        x, post, comb = self.ffn_hc.mix(hidden)
        y = self.mlp.forward(self.post_attention_layernorm.forward(x))
        return self.ffn_hc.combine(hidden, y, post, comb)


class Glm5NextModel(BaseOP):
    def __init__(self, config) -> None:
        self.hc_mult = config.glm5_args.hc_mult
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = OPList(
            [Glm5NextDecoderLayer(config, i) for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = (
            self.embed_tokens.forward(input_ids)
            .unsqueeze(-2)
            .expand(-1, self.hc_mult, -1)
            .contiguous()
        )
        for layer in self.layers.op_list:
            hidden = layer.forward(hidden)
        return self.norm.forward(collapse_head(hidden))


class Glm5NextForCausalLM(BaseLLMModel):
    def __init__(self, config) -> None:
        self.model = Glm5NextModel(config)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens
            if config.tie_word_embeddings
            else None,
        )
        super().__init__()

    def prepare_for_runtime(self) -> None:
        for layer in self.model.layers.op_list:
            if isinstance(layer.self_attn, Glm5NextAttention):
                layer.self_attn.prepare_for_runtime()
        torch.cuda.empty_cache()

    def forward(self) -> torch.Tensor:
        return self.lm_head.forward(
            self.model.forward(get_global_ctx().batch.input_ids)
        )


__all__ = ["Glm5NextForCausalLM"]
