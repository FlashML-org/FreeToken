"""Inkling 2 image processor: no resize, the image is padded to whole 40-pixel tiles and every tile is one image token.

Mirrors transformers' Inkling2ImageProcessor (the checkpoint's processor_config.json): transparent pixels are
composited onto an 8-pixel checkerboard, the pad value is -1 on the 0-255 scale, and the chat template renders one
``<|unused_200054|>`` per image that expands to rows * cols copies.
"""

from __future__ import annotations

import io
import math
import struct
from typing import Any

import torch

from freetoken.message import MMItem
from freetoken.mm import mm_pad_value
from freetoken.mm.config import MultimodalConfig
from freetoken.mm.processor import MMProcessor, PromptReplacement, content_hash

# the config class default; the checkpoint's config.json leaves it out
IMAGE_TOKEN_ID = 200054
# processor_config.json: rescale by 1/255, then normalize with mean = std = 0.5
_RESCALE = 0.00392156862745098
_MEAN_STD = 0.5
# 0-255 value of the padding, before normalization
_PAD_FILL = -1.0
_CHECKER = 8
_CHECKER_LIGHT, _CHECKER_DARK = 204, 153


def _section(config: Any, *names: str) -> Any:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    raise AttributeError(f"vision_config has none of {names}")


def _to_rgb_uint8(image: Any) -> torch.Tensor:
    """[3, H, W] uint8; alpha is composited onto a checkerboard of 8-pixel squares, light at the origin."""
    import numpy as np

    rgba = image.has_transparency_data
    pixels = torch.from_numpy(np.array(image.convert("RGBA" if rgba else "RGB"), dtype=np.uint8)).permute(2, 0, 1)
    if not rgba:
        return pixels.contiguous()
    height, width = pixels.shape[-2:]
    rows = torch.arange(height)[:, None] // _CHECKER
    columns = torch.arange(width)[None, :] // _CHECKER
    background = torch.where((rows + columns) % 2 == 0, _CHECKER_LIGHT, _CHECKER_DARK)
    pixels = pixels.to(torch.int32)
    alpha = pixels[3:4]
    return ((pixels[:3] * alpha + background * (255 - alpha) + 127) // 255).to(torch.uint8)


def preprocess(image: Any, tile: int) -> tuple[torch.Tensor, int, int]:
    """fp32 [rows * cols, tile, tile, 3] tiles in raster order, and the tile grid rows, cols."""
    pixels = _to_rgb_uint8(image)
    channels, height, width = pixels.shape
    rows, cols = math.ceil(height / tile), math.ceil(width / tile)
    padded = torch.full((channels, rows * tile, cols * tile), _PAD_FILL, dtype=torch.float32)
    padded[:, :height, :width] = pixels
    # the reference folds the rescale into the mean and std: (x - 127.5) / 127.5 in fp32
    scale = torch.tensor([_MEAN_STD] * channels) * (1.0 / _RESCALE)
    padded = (padded - scale[:, None, None]) / scale[:, None, None]
    tiles = padded.reshape(channels, rows, tile, cols, tile).permute(1, 3, 2, 4, 0)
    return tiles.reshape(rows * cols, tile, tile, channels), rows, cols


class Inkling2MMProcessor(MMProcessor):
    def __init__(self, hf_config: Any, model_path: str, mm: MultimodalConfig) -> None:
        super().__init__(model_path, mm)
        vc = hf_config.vision_config
        # the checkpoint's raw config names them native_patch_size / adapter_merge_size
        self.tile = int(_section(vc, "patch_size", "native_patch_size")) * int(_section(vc, "spatial_merge_size", "adapter_merge_size"))
        self.image_token_id = getattr(hf_config, "image_token_id", None) or IMAGE_TOKEN_ID
        self.placeholder = [self.image_token_id]

    def decode_image(self, raw: bytes) -> Any:
        from PIL import Image

        return Image.open(io.BytesIO(raw))

    def process(self, images: list[Any]) -> list[MMItem]:
        items: list[MMItem] = []
        for image in images:
            tiles, rows, cols = preprocess(image, self.tile)
            # the tower casts the pixels to its bf16 weights first, so bf16 on the wire loses nothing
            feature = tiles.to(torch.bfloat16)
            item_hash = content_hash(feature, struct.pack("<3i", 1, rows, cols))
            items.append(
                MMItem(
                    modality="image",
                    hash=item_hash,
                    pad_value=mm_pad_value(item_hash),
                    offsets=[],
                    feature=feature,
                    model_specific_data={"grid_thw": [1, rows, cols]},
                )
            )
        return items

    def prompt_replacement(self, item: MMItem) -> PromptReplacement:
        _, rows, cols = item.grid_thw
        return PromptReplacement([self.image_token_id] * (rows * cols))

    def _image_processor(self) -> None:
        """None: the preprocessing is local (:func:`preprocess`); released transformers has no Inkling2ImageProcessor."""
        return None

    def dummy_items(self, dtype: torch.dtype, device: torch.device) -> list[MMItem]:
        return [MMItem(
            modality="image",
            hash=0,
            pad_value=0,
            offsets=[[0, 1]],
            feature=torch.zeros(1, self.tile, self.tile, 3, dtype=dtype, device=device),
            model_specific_data={"grid_thw": [1, 1, 1]},
        )]


__all__ = ["IMAGE_TOKEN_ID", "Inkling2MMProcessor", "preprocess"]
