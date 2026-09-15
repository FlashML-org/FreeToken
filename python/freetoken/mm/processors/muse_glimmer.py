"""Muse-Glimmer image processor: patch-grid ViT with a 2x2 pixel shuffle, a token-maximum budget and 1-D rope; the replacement adds the start/end wrapper the checkpoint processor renders."""

from __future__ import annotations

import struct
from typing import Any

import torch

from freetoken.message import MMItem
from freetoken.mm import mm_pad_value
from freetoken.mm.config import MultimodalConfig
from freetoken.mm.processor import MMProcessor, PromptReplacement, content_hash

# the tokenizer's <|image_start|> / <|image_end|>; the config only names the <|patch|> id
_IMAGE_START_ID = 200080
_IMAGE_END_ID = 200081


class MuseGlimmerMMProcessor(MMProcessor):
    def __init__(self, hf_config: Any, model_path: str, mm: MultimodalConfig) -> None:
        super().__init__(model_path, mm)
        vc = hf_config.vision_config
        self.image_token_id = hf_config.image_token_id
        # the template renders one <|patch|> per image
        self.placeholder = [self.image_token_id]
        self.merge = vc.merge_size
        self.patch_dim = vc.patch_temporal * 3 * vc.patch_size**2

    def get_mm_processor_kwargs(self, mm: MultimodalConfig) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"return_tensors": "pt"}
        # the image processor's only budget is a token maximum; a minimum has no knob
        if mm.image_max_tokens is not None:
            kwargs["max_image_tokens"] = mm.image_max_tokens
        return {**kwargs, **mm.processor_kwargs}

    def process(self, images: list[Any]) -> list[MMItem]:
        processor = self._image_processor()
        kwargs = self.get_mm_processor_kwargs(self.mm)
        items: list[MMItem] = []
        for pil in images:
            out = processor(images=pil, **kwargs)
            pixel_values = out["pixel_values"].to(torch.bfloat16)
            grid = out["image_grid_thw"][0].tolist()
            t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
            if t != 1:
                raise ValueError("video/temporal input is not supported")
            # hash the bf16 wire buffer: half the bytes, and the exact tensor the cache and radix key agree on
            item_hash = content_hash(pixel_values, struct.pack("<3i", t, h, w))
            items.append(
                MMItem(
                    modality="image",
                    hash=item_hash,
                    pad_value=mm_pad_value(item_hash),
                    offsets=[],
                    feature=pixel_values,
                    model_specific_data={"grid_thw": [t, h, w]},
                )
            )
        return items

    def prompt_replacement(self, item: MMItem) -> PromptReplacement:
        t, h, w = item.grid_thw
        pads = [self.image_token_id] * ((t * h * w) // (self.merge * self.merge))
        return PromptReplacement.select_token_id([_IMAGE_START_ID, *pads, _IMAGE_END_ID], self.image_token_id)

    def dummy_items(self, dtype: torch.dtype, device: torch.device) -> list[MMItem]:
        merge = self.merge
        return [MMItem(
            modality="image",
            hash=0,
            pad_value=0,
            offsets=[[0, 1]],
            feature=torch.zeros(merge * merge, self.patch_dim, dtype=dtype, device=device),
            model_specific_data={"grid_thw": [1, merge, merge]},
        )]


__all__ = ["MuseGlimmerMMProcessor"]
