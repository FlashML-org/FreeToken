"""Qwen VL family image processor (Qwen2-VL through Qwen3.8): patch-grid ViT with a spatial merger, 3-axis rope positions."""

from __future__ import annotations

import struct
import threading
from typing import Any

import torch

from freetoken.message import MMItem
from freetoken.mm import mm_pad_value
from freetoken.mm.processor import MMProcessor, PromptReplacement, content_hash, image_positions


class QwenVLMMProcessor(MMProcessor):
    def __init__(self, hf_config: Any, model_path: str) -> None:
        self.model_path = model_path
        vc = hf_config.vision_config
        self.image_token_id = hf_config.image_token_id
        self.placeholder = [self.image_token_id]
        self.merge = vc.spatial_merge_size
        self.patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size**2
        self.is_mrope = "mrope_section" in hf_config.text_config.rope_parameters
        self._processor: Any = None
        self._lock = threading.Lock()

    def _image_processor(self) -> Any:
        with self._lock:
            if self._processor is None:
                from transformers import AutoImageProcessor

                self._processor = AutoImageProcessor.from_pretrained(self.model_path)
            return self._processor

    def process(self, images: list[Any], max_pixels: int | None) -> list[MMItem]:
        processor = self._image_processor()
        kwargs: dict[str, Any] = {"return_tensors": "pt"}
        if max_pixels is not None:
            # the pixel budget travels in size.longest_edge; a bare max_pixels kwarg is ignored
            size = dict(processor.size)
            size["longest_edge"] = max_pixels
            kwargs["size"] = size
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
        n_tokens = (t * h * w) // (self.merge * self.merge)
        return PromptReplacement([self.image_token_id] * n_tokens)

    def image_grid(self, item: MMItem) -> tuple[int, int]:
        _, h, w = item.grid_thw
        return h // self.merge, w // self.merge

    def positions(self, length: int, items: list[MMItem]) -> tuple[torch.Tensor, int] | None:
        if not self.is_mrope:
            return None
        blocks = []
        for item in items:
            ((start, end),) = item.offsets
            rows, cols = self.image_grid(item)
            blocks.append((start, end - start, rows, cols))
        return image_positions(length, blocks)

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


__all__ = ["QwenVLMMProcessor"]
