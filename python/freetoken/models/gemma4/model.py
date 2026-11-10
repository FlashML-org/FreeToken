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

# every release's pad token; the per-layer table row a multimodal row reads
_PAD_TOKEN_ID = 0


class Gemma4DecoderLayer(BaseOP):
    """Gemma 4 decoder block: attention sandwich + feed-forward sandwich, scaled by a
    per-layer ``layer_scalar``. The feed-forward is the dual (shared MLP || routed MoE)
    branch for MoE checkpoints, or a single dense MLP branch for dense checkpoints.
    E2B / E4B add their per-layer input as one more residual step before the scalar."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self._layer_id = layer_id
        self.self_attn = Gemma4Attention(config, layer_id, prefix=f"{prefix}.self_attn")
        self.feed_forward = (
            Gemma4MLP(config, layer_id, prefix=f"{prefix}.feed_forward")
            if config.is_moe
            else Gemma4DenseMLP(config, layer_id, prefix=f"{prefix}.feed_forward")
        )

        eps = config.rms_norm_eps
        H = config.hidden_size
        self.input_layernorm = GemmaRMSNorm(H, eps=eps)
        self.post_attention_layernorm = GemmaRMSNorm(H, eps=eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(H, eps=eps)
        if config.gemma4_e_args is not None:
            P = config.gemma4_e_args.per_layer_hidden_size
            self.per_layer_input_gate = LinearReplicated(
                H, P, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.per_layer_input_gate"
            )
            self.per_layer_projection = LinearReplicated(
                P, H, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.per_layer_projection"
            )
            self.post_per_layer_input_norm = GemmaRMSNorm(H, eps=eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        per_layer_input: torch.Tensor | None,
        shared_kv: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        # --- attention sandwich ---
        residual = x
        h = self.input_layernorm.forward(x)
        h = self.self_attn.forward(h, shared_kv)
        h = self.post_attention_layernorm.forward(h)
        pre_ff, x = self.pre_feedforward_layernorm.forward_add_residual(h, residual)
        if per_layer_input is None:
            return self.feed_forward.forward(pre_ff, x)
        h = self.feed_forward.forward_unscaled(pre_ff, x)
        g = F.gelu(self.per_layer_input_gate.forward(h), approximate="tanh") * per_layer_input
        g = self.post_per_layer_input_norm.forward(self.per_layer_projection.forward(g))
        return (h + g) * self.feed_forward.layer_scalar


class Gemma4Model(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            embed_scale=config.embedding_scale,
            quant_config=config.quant,
            prefix=f"{prefix}.embed_tokens",
        )
        self.layers = OPList(
            [
                Gemma4DecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._has_per_layer_inputs = config.gemma4_e_args is not None
        if self._has_per_layer_inputs:
            P = config.gemma4_e_args.per_layer_hidden_size
            self._per_layer_shape = (config.num_layers, P)
            self.embed_tokens_per_layer = VocabParallelEmbedding(
                num_embeddings=config.gemma4_e_args.per_layer_vocab_size,
                embedding_dim=config.num_layers * P,
                embed_scale=P**0.5,
                quant_config=config.quant,
                prefix=f"{prefix}.embed_tokens_per_layer",
            )
            self.per_layer_model_projection = LinearReplicated(
                config.hidden_size,
                config.num_layers * P,
                has_bias=False,
                quant_config=config.quant,
                prefix=f"{prefix}.per_layer_model_projection",
            )
            self.per_layer_projection_norm = GemmaRMSNorm(P, eps=config.rms_norm_eps)
            self._per_layer_projection_scale = config.hidden_size**-0.5

    def _per_layer_inputs(self, input_ids: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``[T, num_layers, P]``: each row's token embedding plus a projection of its input embedding."""
        table = self.embed_tokens_per_layer
        # multimodal rows carry content pad ids above the vocab
        ids = input_ids if get_global_ctx().batch.mm_embeds is None else input_ids.masked_fill(input_ids >= table.num_embeddings, _PAD_TOKEN_ID)
        token = table.forward(ids).view(-1, *self._per_layer_shape)
        context = self.per_layer_model_projection.forward(x) * self._per_layer_projection_scale
        context = self.per_layer_projection_norm.forward(context.view(-1, *self._per_layer_shape))
        return (context + token) * 2.0**-0.5

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = embed_input_ids(self.embed_tokens, input_ids, get_global_ctx().batch)
        per_layer = self._per_layer_inputs(input_ids, x) if self._has_per_layer_inputs else None
        shared_kv: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for i, layer in enumerate(self.layers.op_list):
            x = layer.forward(x, None if per_layer is None else per_layer[:, i], shared_kv)
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
