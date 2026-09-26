"""DeepSeek-V4 image preprocessing: one image -> ViT patches plus its grid metadata.

Port of the reference ``inference/image_processor.py:load_image``. The PIL work stays on the
CPU and must be bit-identical across runs: an image's radix key rides on these patches.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image, ImageOps

from .config import DSV4VisionConfig
from .vision import safe_resize


def load_image(image: Image.Image, vc: DSV4VisionConfig, max_n_token: int | None = None) -> tuple[torch.Tensor, int, int, int, int]:
    """One image -> ``(patches bf16 [n_vit_h*n_vit_w, 3, p, p], n_vit_h, n_vit_w, n_llm_h, n_llm_w)``.

    ``max_n_token`` lowers the checkpoint's per-image block cap (``--image-max-tokens``).
    """
    image = image.convert("RGB")
    p = vc.vision_patch_size
    width, height = image.size
    if vc.vision_max_wh_ratio is not None and width > height * vc.vision_max_wh_ratio:
        width = height * vc.vision_max_wh_ratio
    if 0 < width * height < vc.vision_min_pixels:
        ratio = (vc.vision_min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        height, width, best_height, best_width, p, vc.vision_downsample_ratio,
        vc.vision_max_n_token if max_n_token is None else min(vc.vision_max_n_token, max_n_token),
    )
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if vc.vision_max_wh_ratio is not None and image.width >= vc.vision_max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, p, p)
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


__all__ = ["load_image"]
