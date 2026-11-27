"""Inkling 2 image processor: no resize -- the image is padded to whole 40x40-pixel cells and every cell
becomes one soft token; the template's single image placeholder expands to rows x cols of them.

The checkpoint's processor class (``Inkling2ImageProcessor``) needs a newer transformers than FreeToken
pins, so its preprocessing is reproduced here: blend transparent pixels onto a gray checkerboard of
8-pixel squares, pad bottom/right with -1 on the 0-255 scale, rescale by 1/255, normalize with
mean = std = 0.5, cut into cells in raster order.
"""

from __future__ import annotations

import math
import struct
from typing import Any

import numpy as np
import torch

from freetoken.message import MMItem
from freetoken.mm import mm_pad_value
from freetoken.mm.config import MultimodalConfig
from freetoken.mm.processor import MMProcessor, PromptReplacement, content_hash
from freetoken.models.inkling2.config import IMAGE_TOKEN_ID


def _flatten_alpha(pil: Any) -> Any:
    """RGB ``pil``, transparent pixels blended onto the 8-pixel 204 / 153 checkerboard."""
    if not pil.has_transparency_data:
        return pil.convert("RGB")
    from PIL import Image

    rgba = np.asarray(pil.convert("RGBA"), dtype=np.int32)
    height, width = rgba.shape[:2]
    board = np.where((np.arange(height)[:, None] // 8 + np.arange(width)[None, :] // 8) % 2 == 0, 204, 153)[..., None]
    alpha = rgba[..., 3:]
    rgb = (rgba[..., :3] * alpha + board * (255 - alpha) + 127) // 255
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def _get(cfg: Any, key: str, default: Any) -> Any:
    value = cfg.get(key) if isinstance(cfg, dict) else getattr(cfg, key, None)
    return default if value is None else value


class Inkling2MMProcessor(MMProcessor):
    def __init__(self, hf_config: Any, model_path: str, mm: MultimodalConfig) -> None:
        super().__init__(model_path, mm)
        vc = hf_config.vision_config
        self.image_token_id = getattr(hf_config, "image_token_id", None) or IMAGE_TOKEN_ID
        self.placeholder = [self.image_token_id]
        self.cell = _get(vc, "native_patch_size", 20) * _get(vc, "adapter_merge_size", 2)

    def _grid(self, width: int, height: int) -> tuple[int, int]:
        return math.ceil(height / self.cell), math.ceil(width / self.cell)

    def _fit(self, pil: Any) -> Any:
        """Downscale (aspect kept) only when --mm-image-max-tokens caps the cells below what the image needs."""
        budget = self.mm.image_max_tokens
        rows, cols = self._grid(*pil.size)
        if budget is None or rows * cols <= budget:
            return pil
        scale = math.sqrt(budget / (rows * cols))
        while True:
            size = (max(1, int(pil.width * scale)), max(1, int(pil.height * scale)))
            rows, cols = self._grid(*size)
            if rows * cols <= budget:
                return pil.resize(size, resample=3)  # bicubic
            scale *= 0.95

    def pixel_cells(self, pil: Any) -> tuple[torch.Tensor, int, int]:
        """``[rows * cols, cell, cell, 3]`` float32 cells of an RGB image and its grid."""
        rgb = np.asarray(pil.convert("RGB"), dtype=np.float32)
        height, width, _ = rgb.shape
        rows, cols = self._grid(width, height)
        padded = np.full((rows * self.cell, cols * self.cell, 3), -1.0, dtype=np.float32)
        padded[:height, :width] = rgb
        normed = (padded / 255.0 - 0.5) / 0.5
        cells = normed.reshape(rows, self.cell, cols, self.cell, 3).transpose(0, 2, 1, 3, 4)
        return torch.from_numpy(np.ascontiguousarray(cells.reshape(rows * cols, self.cell, self.cell, 3))), rows, cols

    def process(self, images: list[Any]) -> list[MMItem]:
        items: list[MMItem] = []
        for pil in images:
            cells, rows, cols = self.pixel_cells(self._fit(_flatten_alpha(pil)))
            # the tower casts the pixels to its own dtype first; hashing the bf16 wire buffer keys exactly that
            feature = cells.to(torch.bfloat16)
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

    def dummy_items(self, dtype: torch.dtype, device: torch.device) -> list[MMItem]:
        return [MMItem(
            modality="image",
            hash=0,
            pad_value=0,
            offsets=[[0, 1]],
            feature=torch.zeros(1, self.cell, self.cell, 3, dtype=dtype, device=device),
            model_specific_data={"grid_thw": [1, 1, 1]},
        )]


__all__ = ["Inkling2MMProcessor"]
