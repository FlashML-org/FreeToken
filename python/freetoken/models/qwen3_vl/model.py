from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx

from freetoken.models.qwen3.model import Qwen3ForCausalLM, Qwen3Model
from freetoken.models.blocks import embed_input_ids
from freetoken.models.qwen3_moe.model import Qwen3MoeForCausalLM, Qwen3Model as Qwen3MoeModel

from .vision import Qwen3VLVisionModel, QwenVLVisionMixin

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


def deepstack_add(
    x: torch.Tensor, rows: torch.Tensor, mm_embeds: torch.Tensor, level: int, hidden_size: int
) -> None:
    """Add DeepStack level `level` (the column block after the main embedding) onto the image rows of x, in place."""
    lo = hidden_size * (level + 1)
    x.index_add_(0, rows, mm_embeds[:, lo : lo + hidden_size].to(x.dtype))


class Qwen3VLTextModel(Qwen3Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._deepstack_levels = len(config.vision_config.deepstack_visual_indexes) if config.vision_config else 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch
        x = embed_input_ids(self.embed_tokens, input_ids, batch)
        residual: torch.Tensor | None = None
        for level, layer in enumerate(self.layers.op_list):
            x, residual = layer.forward(x, residual)
            if batch.mm_embeds is not None and level < self._deepstack_levels:
                deepstack_add(x, batch.mm_rows, batch.mm_embeds, level, x.shape[1])
        return self.norm.forward(x, residual)[0]


class Qwen3VLMoeTextModel(Qwen3MoeModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._deepstack_levels = len(config.vision_config.deepstack_visual_indexes) if config.vision_config else 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch
        x = embed_input_ids(self.embed_tokens, input_ids, batch)
        residual: torch.Tensor | None = None
        for level, layer in enumerate(self.layers.op_list):
            x, residual = layer.forward(x, residual)
            if batch.mm_embeds is not None and level < self._deepstack_levels:
                deepstack_add(x, batch.mm_rows, batch.mm_embeds, level, x.shape[1])
        return self.norm.forward(x, residual)[0]


class Qwen3VLForConditionalGeneration(QwenVLVisionMixin, Qwen3ForCausalLM):
    model_cls = Qwen3VLTextModel

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.is_multimodal:
            self.visual = Qwen3VLVisionModel(config.vision_config, quant_config=config.quant, prefix="visual")


class Qwen3VLMoeForConditionalGeneration(QwenVLVisionMixin, Qwen3MoeForCausalLM):
    model_cls = Qwen3VLMoeTextModel

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.is_multimodal:
            self.visual = Qwen3VLVisionModel(config.vision_config, quant_config=config.quant, prefix="visual")


__all__ = [
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration",
    "deepstack_add",
]
