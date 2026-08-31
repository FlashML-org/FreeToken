from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from freetoken.utils import nvtx_annotate

from freetoken.models.blocks import BaseLLMModel, GatedMLP as LlamaMLP

from .attention import LlamaAttention as LlamaAttn

from freetoken.models.llama.gguf import is_gguf_model, convert_llama_to_gguf, parse_gguf_config

from freetoken.layers.moe import make_moe_layer

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class LlamaDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = LlamaAttn(config, layer_id)       
        # build the router and the smaller experts
        if config.num_experts > 1:
            self.mlp = make_moe_layer(
                config, 
                layer_id=layer_id, 
                weight_format= getattr(config,"moe_weight_format","bf16")
                )           
            # Build the massive Shared Expert
            if getattr(config, "shared_expert_intermediate_size", 0) > 0:
                self.shared_expert=LlamaMLP(config)
            else:
                self.shared_expert = None               
        else:
            # standard dense LLaMA 1/2/3     
            self.mlp = LlamaMLP(config)
            
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        
        # --- NEW ROUTING LOGIC ---  
        if getattr(self, "shared_expert", None) is not None:
            routed_out = self.mlp.forward(x)
            shared_out = self.shared_expert.forward(x)
            # adding the shareed expert and MoE expert together
            x = routed_out + shared_out
        else:
            # standard dense routing
            x = self.mlp.forward(x)
        return x, residual


class LlamaModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [LlamaDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class LlamaForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()
        if is_gguf_model(config):
            convert_llama_to_gguf(self,config)


    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["LlamaForCausalLM"]
