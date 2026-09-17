"""DeepSeek-V4 image processor: each ``<｜deepseek_image｜>`` expands into the image's block of tokens.

Where the reference streams ``vocab_size + type`` pseudo ids, the whole span carries one
content pad id and the block's embeddings -- sentinels included -- come from the encoder.
"""

from __future__ import annotations

import struct
from typing import Any

import torch

from freetoken.message import MMItem
from freetoken.mm import mm_pad_value
from freetoken.mm.config import MultimodalConfig
from freetoken.mm.processor import MMProcessor, PromptReplacement, content_hash
from freetoken.models.deepseek_v4.config import IMAGE_TOKEN_ID, DSV4VisionConfig, parse_vision_config
from freetoken.models.deepseek_v4.image_processor import load_image
from freetoken.models.deepseek_v4.vision import COMPRESS_PAD_TO, build_image_block


class DSV4MMProcessor(MMProcessor):
    def __init__(self, hf_config: Any, model_path: str, mm: MultimodalConfig) -> None:
        super().__init__(model_path, mm)
        vc = parse_vision_config(hf_config)
        if vc is None:
            raise ValueError(f"{model_path} carries no vision tower (vision_n_layers <= 0)")
        self.vc: DSV4VisionConfig = vc
        self.image_token_id = IMAGE_TOKEN_ID
        self.placeholder = [self.image_token_id]
        # --image-max-tokens lowers the checkpoint's per-image block cap
        self.max_n_token = min(vc.vision_max_n_token, mm.image_max_tokens) if mm.image_max_tokens else vc.vision_max_n_token

    def process(self, images: list[Any]) -> list[MMItem]:
        items: list[MMItem] = []
        for image in images:
            patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(image, self.vc, self.max_n_token)
            # layout, hash and token count are only known at the insertion offset
            items.append(
                MMItem(
                    modality="image",
                    hash=0,
                    pad_value=0,
                    offsets=[],
                    feature=patches,
                    model_specific_data={
                        "n_vit_h": n_vit_h,
                        "n_vit_w": n_vit_w,
                        "n_llm_h": n_llm_h,
                        "n_llm_w": n_llm_w,
                    },
                )
            )
        return items

    def prompt_replacement(self, item: MMItem, start: int = 0) -> PromptReplacement:
        n_llm_h, n_llm_w = item.n_llm_h, item.n_llm_w
        types, _perm = build_image_block(n_llm_h, n_llm_w, start)
        # the lead pads align the block, so the same image at another offset is another
        # sequence: the layout belongs in the hash or the two would share a radix key
        compress_pad = COMPRESS_PAD_TO - 1 - start % COMPRESS_PAD_TO
        item.hash = content_hash(item.feature, struct.pack("<3i", n_llm_h, n_llm_w, compress_pad))
        item.pad_value = mm_pad_value(item.hash)
        item.model_specific_data["start"] = start
        return PromptReplacement([self.image_token_id] * types.numel())

    def dummy_items(self, dtype: torch.dtype, device: torch.device) -> list[MMItem]:
        # The smallest block the tower can run: one 2x2 merge grid out of a 6x6 patch grid.
        r, p = self.vc.vision_downsample_ratio, self.vc.vision_patch_size
        n_vit, n_llm = 2 * r, 2
        types, _perm = build_image_block(n_llm, n_llm, 0)
        return [
            MMItem(
                modality="image",
                hash=0,
                pad_value=0,
                offsets=[[0, types.numel()]],
                feature=torch.zeros(n_vit * n_vit, 3, p, p, dtype=dtype, device=device),
                model_specific_data={
                    "n_vit_h": n_vit,
                    "n_vit_w": n_vit,
                    "n_llm_h": n_llm,
                    "n_llm_w": n_llm,
                    "start": 0,
                },
            )
        ]


__all__ = ["DSV4MMProcessor"]
