from __future__ import annotations

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNorm, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import LagunaAttention
from .moe import LagunaMLP, LagunaSparseMoeBlock


class LagunaDecoderLayer(BaseOP):
    def __init__(self, config, layer_id: int):
        self._layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LagunaAttention(config, layer_id)
        self.mlp = (LagunaMLP(config.hidden_size, config.intermediate_size)
                    if layer_id < config.first_k_dense_replace
                    else LagunaSparseMoeBlock(config, layer_id))

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn.forward(self.input_layernorm.forward(x))
        x = x + self.mlp.forward(self.ffn_norm.forward(x))
        return x


class LagunaModel(BaseOP):
    def __init__(self, config):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList([LagunaDecoderLayer(config, i) for i in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        for layer in self.layers.op_list:
            x = layer.forward(x)
        return self.norm.forward(x)


class LagunaForCausalLM(BaseLLMModel):
    def __init__(self, config):
        self.model = LagunaModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size,
                                      tie_word_embeddings=config.tie_word_embeddings,
                                      tied_embedding=None)
        super().__init__()
        from .gguf import convert_laguna_to_gguf, is_gguf_model
        if is_gguf_model(config):
            convert_laguna_to_gguf(self, config)
        elif config.gguf_model_path:
            # GGUF-sourced but no tensor table (a converted FTW dir's metadata-only
            # source_metadata.gguf): the per-tensor quant types are unrecoverable, so
            # neither module conversion nor expert banks can be built. Refuse loudly
            # instead of constructing a dense model that fails weight loading.
            raise NotImplementedError(
                "laguna FTW conversion is not supported yet -- serve the .gguf file "
                "directly (per-tensor quant types live only in its tensor table)"
            )

    def forward(self) -> torch.Tensor:
        return self.lm_head.forward(self.model.forward(get_global_ctx().batch.input_ids))


__all__ = ["LagunaDecoderLayer", "LagunaModel", "LagunaForCausalLM"]
