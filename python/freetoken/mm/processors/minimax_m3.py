"""MiniMax-M3 image processor: patch-grid ViT with a 2x2 merger, a pixel-area budget and 1-D rope."""

from __future__ import annotations

import struct
from typing import Any

import torch

from freetoken.message import MMItem
from freetoken.mm import mm_pad_value
from freetoken.mm.config import MultimodalConfig
from freetoken.mm.processor import MMProcessor, PromptReplacement, content_hash

# the config names only the image token; the reference processor reads these two from the tokenizer's fixed added tokens
IMAGE_START_ID = 200029  # ]<]start of image[>[
IMAGE_END_ID = 200030  # ]<]end of image[>[


def _compression(vc: Any, key: str) -> int:
    """The native vision config carries the merge sizes flat; the checkpoint's own config nests them under img_token_compression_config."""
    value = getattr(vc, key, None)
    return int(vc.img_token_compression_config[key] if value is None else value)


class MiniMaxM3MMProcessor(MMProcessor):
    def __init__(self, hf_config: Any, model_path: str, mm: MultimodalConfig) -> None:
        super().__init__(model_path, mm)
        vc = hf_config.vision_config
        # the native config maps image_token_id onto the checkpoint's image_token_index
        self.image_token_id = getattr(hf_config, "image_token_id", getattr(hf_config, "image_token_index", None))
        # the template renders one image token per image; the wrappers come from the replacement
        self.placeholder = [self.image_token_id]
        self.merge = _compression(vc, "spatial_merge_size")
        self.pixels_per_token = (vc.patch_size * self.merge) ** 2
        self.patch_dim = vc.num_channels * _compression(vc, "temporal_patch_size") * vc.patch_size**2

    def get_mm_processor_kwargs(self, mm: MultimodalConfig) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"return_tensors": "pt"}
        if mm.image_min_tokens is not None or mm.image_max_tokens is not None:
            # the budget travels as pixel areas in size.shortest_edge / longest_edge
            size = dict(self._image_processor().size)
            if mm.image_min_tokens is not None:
                size["shortest_edge"] = mm.image_min_tokens * self.pixels_per_token
            if mm.image_max_tokens is not None:
                size["longest_edge"] = mm.image_max_tokens * self.pixels_per_token
            kwargs["size"] = size
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
        full = [IMAGE_START_ID] + [self.image_token_id] * ((t * h * w) // (self.merge * self.merge)) + [IMAGE_END_ID]
        return PromptReplacement.select_token_id(full, self.image_token_id)

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


__all__ = ["IMAGE_END_ID", "IMAGE_START_ID", "MiniMaxM3MMProcessor"]
