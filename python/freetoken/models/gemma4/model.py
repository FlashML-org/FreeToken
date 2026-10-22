from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.utils import nvtx_annotate

from freetoken.models.blocks import BaseLLMModel, embed_input_ids

from .attention import Gemma4Attention
from .moe import Gemma4DenseMLP, Gemma4MLP
from .vision import Gemma4MultimodalEmbedder, Gemma4UnifiedVisionEmbedder, Gemma4VisionModel

if TYPE_CHECKING:
    from freetoken.message import MMItem
    from freetoken.models.config import ModelConfig


class Gemma4DecoderLayer(BaseOP):
    """Gemma 4 decoder block: attention sandwich + feed-forward sandwich, scaled by a
    per-layer ``layer_scalar``. The feed-forward is the dual (shared MLP || routed MoE)
    branch for MoE checkpoints, or a single dense MLP branch for dense checkpoints."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self._layer_id = layer_id
        self.self_attn = Gemma4Attention(config, layer_id, prefix=f"{prefix}.self_attn")
        self.feed_forward = (
            Gemma4MLP(config, layer_id, prefix=f"{prefix}.feed_forward")
            if config.is_moe
            else Gemma4DenseMLP(config, prefix=f"{prefix}.feed_forward")
        )

        eps = config.rms_norm_eps
        H = config.hidden_size
        self.input_layernorm = GemmaRMSNorm(H, eps=eps)
        self.post_attention_layernorm = GemmaRMSNorm(H, eps=eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(H, eps=eps)
        self.per_layer_input_dim = config.per_layer_input_dim
        if self.per_layer_input_dim:
            self.per_layer_input_gate = LinearReplicated(
                H,
                self.per_layer_input_dim,
                has_bias=False,
                quant_config=config.quant,
                prefix=f"{prefix}.per_layer_input_gate",
            )
            self.per_layer_projection = LinearReplicated(
                self.per_layer_input_dim,
                H,
                has_bias=False,
                quant_config=config.quant,
                prefix=f"{prefix}.per_layer_projection",
            )
            self.post_per_layer_input_norm = GemmaRMSNorm(H, eps=eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        *,
        per_layer_input: torch.Tensor | None = None,
        shared_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        # --- attention sandwich ---
        residual = x
        h = self.input_layernorm.forward(x)
        h = self.self_attn.forward(h, shared_kv_states)
        h = self.post_attention_layernorm.forward(h)
        pre_ff, x = self.pre_feedforward_layernorm.forward_add_residual(h, residual)
        if not self.per_layer_input_dim:
            return self.feed_forward.forward(pre_ff, x)
        if per_layer_input is None:
            raise ValueError(f"Gemma4 layer {self._layer_id} requires its PLE input")

        x = self.feed_forward.forward(pre_ff, x, apply_layer_scalar=False)
        residual = x
        gate = F.gelu(self.per_layer_input_gate.forward(x), approximate="tanh")
        x = self.per_layer_projection.forward(gate * per_layer_input)
        x = self.post_per_layer_input_norm.forward(x)
        return (residual + x) * self.feed_forward.layer_scalar


class Gemma4Model(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.hidden_size = config.hidden_size
        self.shared_kv_source_layers = config.shared_kv_source_layers
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            embed_scale=config.embedding_scale,
            quant_config=config.quant,
            prefix=f"{prefix}.embed_tokens",
        )
        if config.per_layer_input_dim:
            ple_vocab_size = config.per_layer_input_vocab_size or config.vocab_size
            if config.gguf_per_layer_embedding_type is None:
                self.embed_tokens_per_layer = VocabParallelEmbedding(
                    num_embeddings=ple_vocab_size,
                    embedding_dim=config.num_layers * config.per_layer_input_dim,
                    embed_scale=config.per_layer_input_dim**0.5,
                    quant_config=config.quant,
                    prefix=f"{prefix}.embed_tokens_per_layer",
                )
            else:
                from freetoken.layers.gguf import GGUFEmbedding

                self.embed_tokens_per_layer = GGUFEmbedding(
                    num_embeddings=ple_vocab_size,
                    embedding_dim=config.num_layers * config.per_layer_input_dim,
                    quant_type=config.gguf_per_layer_embedding_type,
                    embed_scale=config.per_layer_input_dim**0.5,
                )
            self.per_layer_model_projection = LinearReplicated(
                config.hidden_size,
                config.num_layers * config.per_layer_input_dim,
                has_bias=False,
                quant_config=config.quant,
                prefix=f"{prefix}.per_layer_model_projection",
            )
            self.per_layer_projection_norm = GemmaRMSNorm(
                config.per_layer_input_dim, eps=config.rms_norm_eps
            )
        self.layers = OPList(
            [
                Gemma4DecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = embed_input_ids(self.embed_tokens, input_ids, get_global_ctx().batch)
        per_layer_inputs = None
        if self.layers.op_list[0].per_layer_input_dim:
            ple_vocab_size = self.embed_tokens_per_layer.num_embeddings
            token_inputs = self.embed_tokens_per_layer.forward(
                input_ids.clamp(max=ple_vocab_size - 1)
            ).reshape(-1, len(self.layers.op_list), self.layers.op_list[0].per_layer_input_dim)
            context_inputs = self.per_layer_model_projection.forward(x)
            context_inputs = context_inputs * (self.hidden_size**-0.5)
            context_inputs = context_inputs.reshape_as(token_inputs)
            context_inputs = self.per_layer_projection_norm.forward(context_inputs)
            per_layer_inputs = (context_inputs + token_inputs) * (2.0**-0.5)

        shared_kv_states = {} if self.shared_kv_source_layers else None
        for layer_id, layer in enumerate(self.layers.op_list):
            per_layer_input = (
                per_layer_inputs[:, layer_id, :] if per_layer_inputs is not None else None
            )
            x = layer.forward(
                x,
                per_layer_input=per_layer_input,
                shared_kv_states=shared_kv_states,
            )
        return self.norm.forward(x)


class Gemma4ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Gemma4Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        self._final_logit_softcapping = config.final_logit_softcapping
        super().__init__()
        from .gguf import convert_gemma4_to_gguf, is_gguf_model

        if is_gguf_model(config):
            convert_gemma4_to_gguf(self, config)

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        if self._final_logit_softcapping is not None:
            cap = self._final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits


class Gemma4ForConditionalGeneration(Gemma4ForCausalLM):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.is_multimodal:
            self.vision_tower = Gemma4VisionModel(config.vision_config)
            self.embed_vision = Gemma4MultimodalEmbedder(config.vision_config)

    def place_encoder_weights(self, mode: str) -> None:
        self.vision_tower.place_weights(mode)

    def encode(self, item: MMItem) -> torch.Tensor:
        device = self.embed_vision.embedding_projection.weight.device
        feature = item.feature.to(device, non_blocking=True)[None]
        positions = item.position_ids.to(device, non_blocking=True).long()[None]
        return self.embed_vision.forward(self.vision_tower.forward(feature, positions))


class Gemma4UnifiedForConditionalGeneration(Gemma4ForCausalLM):
    """The gemma4_unified release: no vision tower, a linear embedder turns each 48x48 super-patch into one soft token."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.is_multimodal:
            self.vision_embedder = Gemma4UnifiedVisionEmbedder(config.vision_config)
            self.embed_vision = Gemma4MultimodalEmbedder(config.vision_config)

    def place_encoder_weights(self, mode: str) -> None:
        """Nothing to stream: the embedder is a few dense tensors, resident under either placement."""
        if mode not in ("gpu", "host"):
            raise ValueError(f"unknown vision weight placement {mode!r}")

    def encode(self, item: MMItem) -> torch.Tensor:
        device = self.embed_vision.embedding_projection.weight.device
        feature = item.feature.to(device, non_blocking=True)[None]
        positions = item.position_ids.to(device, non_blocking=True).long()[None]
        return self.embed_vision.forward(self.vision_embedder.forward(feature, positions))[0]


__all__ = ["Gemma4ForCausalLM", "Gemma4ForConditionalGeneration", "Gemma4UnifiedForConditionalGeneration"]
