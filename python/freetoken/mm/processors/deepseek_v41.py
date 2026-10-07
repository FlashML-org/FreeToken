"""DeepSeek-V4.1 image resize/patch layout and learned image-span delimiters."""

from __future__ import annotations

import math
import struct

import numpy as np
import torch
from PIL import ImageOps

from freetoken.message import MMItem
from freetoken.mm import mm_pad_value
from freetoken.mm.processor import MMProcessor, PromptReplacement, content_hash


def image_grid(width, height, patch, ratio, min_pixels, max_tokens, max_wh_ratio=None):
    if max_wh_ratio is not None:
        width = min(width, height * max_wh_ratio)
    if width * height < min_pixels:
        scale = (min_pixels / (width * height)) ** 0.5
        width, height = int(width * scale), int(height * scale)
    w, h = math.ceil(width / patch) * patch, math.ceil(height / patch) * patch
    if math.ceil(h / (patch * ratio)) * (math.ceil(w / (patch * ratio)) + 1) + 2 > max_tokens:
        aspect, cell = height / width, patch * ratio
        wf = math.sqrt((max_tokens - 2) / aspect + 0.25) - 0.5
        hf = wf * aspect
        if wf < 1:
            h, w = (max_tokens - 2) // 2 * cell, cell
        elif hf < 1:
            h, w = cell, (max_tokens - 3) * cell
        else:
            scale = min(math.floor(wf) * cell / width, math.floor(hf) * cell / height)
            h, w = math.floor(height * scale / patch) * patch, math.floor(width * scale / patch) * patch
    return h, w


class DeepseekV41MMProcessor(MMProcessor):
    def __init__(self, hf_config, model_path, mm):
        super().__init__(model_path, mm)
        self.vc = hf_config.vision_config
        self.placeholder = [hf_config.image_token_id]
        self.min_pixels = self.vc.min_pixels
        self.max_tokens = mm.image_max_tokens if mm.image_max_tokens is not None else self.vc.max_image_tokens
        if mm.image_min_tokens is not None:
            # Delimiters and row breaks count against the total span budget.
            self.min_pixels = max(1, mm.image_min_tokens - 2) * (self.vc.patch_size * self.vc.downsample_ratio)**2
        allowed = {"min_pixels", "max_image_tokens", "max_wh_ratio"}
        unknown = set(mm.processor_kwargs) - allowed
        if unknown:
            raise ValueError(f"unsupported DeepSeek image processor options: {sorted(unknown)}")
        self.min_pixels = mm.processor_kwargs.get("min_pixels", self.min_pixels)
        self.max_tokens = mm.processor_kwargs.get("max_image_tokens", self.max_tokens)
        self.max_wh_ratio = mm.processor_kwargs.get("max_wh_ratio", self.vc.max_wh_ratio)
        if self.max_tokens < 4 or self.min_pixels < 0:
            raise ValueError("image token budget must be >= 4 and min_pixels must be nonnegative")
        if self.max_wh_ratio is not None and self.max_wh_ratio <= 0:
            raise ValueError("max_wh_ratio must be positive")

    def process(self, images):
        p, r = self.vc.patch_size, self.vc.downsample_ratio
        items = []
        for image in images:
            h, w = image_grid(image.width, image.height, p, r, self.min_pixels, self.max_tokens, self.max_wh_ratio)
            if self.max_wh_ratio is not None and image.width >= self.max_wh_ratio * image.height:
                image = image.resize((w, h))
            else:
                image = ImageOps.pad(image, (w, h), color=(127, 127, 127))
            x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
            x = ((x - 0.5) / 0.5).to(torch.bfloat16)
            nh, nw = h // p, w // p
            patches = x.reshape(3, nh, p, nw, p).permute(1, 3, 0, 2, 4).reshape(nh * nw, 3, p, p).contiguous()
            item_hash = content_hash(patches, struct.pack("<3i", 1, nh, nw))
            items.append(MMItem(modality="image", hash=item_hash, pad_value=mm_pad_value(item_hash), offsets=[], feature=patches,
                                model_specific_data={"grid_thw": [1, nh, nw]}))
        return items

    def prompt_replacement(self, item):
        _, h, w = item.grid_thw
        r = self.vc.downsample_ratio
        n = ((h + r - 1) // r) * ((w + r - 1) // r + 1) + 2
        # Every span row, including delimiters/newlines, has a learned visual embedding.
        return PromptReplacement([self.placeholder[0]] * n)

    def dummy_items(self, dtype, device):
        p = self.vc.patch_size
        return [MMItem(modality="image", hash=0, pad_value=0, offsets=[[0, 4]],
                       feature=torch.zeros(1, 3, p, p, dtype=dtype, device=device), model_specific_data={"grid_thw": [1, 1, 1]})]
