from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.utils import init_logger, nvtx_annotate

from freetoken.models.blocks import BaseLLMModel

from .attention import Gemma4Attention
from .moe import Gemma4DenseMLP, Gemma4MLP
from .vision import Gemma4MultimodalEmbedder, Gemma4VisionModel

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


logger = init_logger(__name__)


def _log_vision_embedding_summary(label: str, tensor: torch.Tensor) -> None:
    """Write a bounded numerical fingerprint for an opt-in vision parity run.

    ``FREETOKEN_GEMMA4_VISION_DEBUG=1`` enables this diagnostic while investigating
    a reference mismatch.  It deliberately records only shape, finite-state,
    aggregate statistics, and the first sixteen scalar values: that is enough to
    compare FreeToken's vision output with llama.cpp's mtmd debug output without
    placing a full image embedding in a server log.  The environment guard keeps
    normal serving free from the device synchronization caused by ``cpu()``.
    """
    if os.environ.get("FREETOKEN_GEMMA4_VISION_DEBUG") != "1":
        return
    values = tensor.detach().float()
    sample = values.reshape(-1)[:16].cpu().tolist()
    logger.info_rank0(
        "Gemma4 vision debug %s: shape=%s finite=%s mean=%.8f std=%.8f "
        "min=%.8f max=%.8f sum=%.8f first16=%s",
        label,
        tuple(values.shape),
        bool(torch.isfinite(values).all().item()),
        float(values.mean().item()),
        float(values.std(unbiased=False).item()),
        float(values.min().item()),
        float(values.max().item()),
        float(values.sum().item()),
        ",".join(f"{value:.8f}" for value in sample),
    )


class Gemma4DecoderLayer(BaseOP):
    """Gemma 4 decoder block: attention sandwich + feed-forward sandwich, scaled by a
    per-layer ``layer_scalar``. The feed-forward is the dual (shared MLP || routed MoE)
    branch for MoE checkpoints, or a single dense MLP branch for dense checkpoints."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self.self_attn = Gemma4Attention(config, layer_id)
        self.feed_forward = (
            Gemma4MLP(config, layer_id) if config.is_moe else Gemma4DenseMLP(config)
        )

        eps = config.rms_norm_eps
        H = config.hidden_size
        self.input_layernorm = GemmaRMSNorm(H, eps=eps)
        self.post_attention_layernorm = GemmaRMSNorm(H, eps=eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(H, eps=eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- attention sandwich ---
        residual = x
        h = self.input_layernorm.forward(x)
        h = self.self_attn.forward(h)
        h = self.post_attention_layernorm.forward(h)
        pre_ff, x = self.pre_feedforward_layernorm.forward_add_residual(h, residual)
        return self.feed_forward.forward(pre_ff, x)


class Gemma4Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            embed_scale=config.embedding_scale,
        )
        self.layers = OPList(
            [Gemma4DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._image_token_id = config.image_token_id

    def _merge_multimodal(self, input_ids: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Scatter precomputed image soft-token embeddings at image-token positions.

        ``mm_embeds`` (set by the scheduler from each request's vision features) is a
        ``[num_image_tokens, hidden]`` tensor whose rows replace the placeholder
        embeddings produced for ``image_token_id``. Only runs during prefill batches
        that carry images; decode batches never do.
        """
        batch = get_global_ctx().batch
        mm_embeds = getattr(batch, "mm_embeds", None)
        if mm_embeds is None or self._image_token_id is None:
            return x
        mask = input_ids == self._image_token_id
        n_slots = int(mask.sum().item())
        assert n_slots == mm_embeds.shape[0], (
            f"image-token slots ({n_slots}) != vision features ({mm_embeds.shape[0]}); "
            "image tokens must not be split across prefill chunks"
        )
        return x.masked_scatter(mask.unsqueeze(-1), mm_embeds.to(x.dtype))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        x = self._merge_multimodal(input_ids, x)
        for layer in self.layers.op_list:
            x = layer.forward(x)
        return self.norm.forward(x)


class Gemma4ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Gemma4Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        self._final_logit_softcapping = config.final_logit_softcapping
        if config.is_multimodal:
            self.vision_tower = Gemma4VisionModel(config.vision_config)
            self.embed_vision = Gemma4MultimodalEmbedder(config.vision_config)
        super().__init__()

        # GGUF checkpoints carry native block-quantized weights: swap the dense
        # projections + embedding for GGUF-quant ops (experts stay on the offload cache).
        from .gguf import convert_gemma4_to_gguf, is_gguf_model

        if is_gguf_model(config):
            convert_gemma4_to_gguf(self, config)

    @torch.inference_mode()
    def encode_images(
        self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor
    ) -> torch.Tensor:
        """Run the vision tower + projector. Returns ``[num_valid_soft_tokens, text_hidden]``.

        ``pixel_values``: ``[num_images, num_patches, 3*patch**2]``;
        ``image_position_ids``: ``[num_images, num_patches, 2]`` with ``(-1, -1)`` padding.
        """
        features = self.vision_tower.forward(pixel_values, image_position_ids)
        _log_vision_embedding_summary("tower", features)
        projected = self.embed_vision.forward(features)
        _log_vision_embedding_summary("projected", projected)
        return projected

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        if self._final_logit_softcapping is not None:
            cap = self._final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits


__all__ = ["Gemma4ForCausalLM"]
