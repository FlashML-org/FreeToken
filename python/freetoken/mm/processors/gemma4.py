"""Gemma 4 image processor: aspect-ratio-preserving 16x16 patches, pooled 3x3 into soft tokens, 1-D rope."""

from __future__ import annotations

import struct
from typing import Any

import torch

from freetoken.message import MMItem
from freetoken.mm import mm_pad_value
from freetoken.mm.config import MultimodalConfig
from freetoken.mm.processor import MMProcessor, PromptReplacement, content_hash

# the only soft-token budgets the image processor accepts
_SOFT_TOKEN_BUDGETS = (70, 140, 280, 560, 1120)


def _soft_token_budget(max_tokens: int) -> int:
    """Largest accepted budget within max_tokens."""
    return max(b for b in _SOFT_TOKEN_BUDGETS if b <= max_tokens)


class Gemma4MMProcessor(MMProcessor):
    def __init__(self, hf_config: Any, model_path: str, mm: MultimodalConfig) -> None:
        super().__init__(model_path, mm)
        if mm.image_max_tokens is not None and mm.image_max_tokens < _SOFT_TOKEN_BUDGETS[0]:
            raise ValueError(f"--image-max-tokens {mm.image_max_tokens} is below the smallest soft-token budget this image processor accepts ({_SOFT_TOKEN_BUDGETS[0]})")
        vc = hf_config.vision_config
        self.image_token_id = hf_config.image_token_id
        self.boi_token_id = hf_config.boi_token_id
        self.eoi_token_id = hf_config.eoi_token_id
        self.placeholder = [self.image_token_id]
        self.pooling_kernel_size = vc.pooling_kernel_size
        self.patch_dim = 3 * vc.patch_size**2

    def get_mm_processor_kwargs(self, mm: MultimodalConfig) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"return_tensors": "pt"}
        if mm.image_max_tokens is not None:
            # the processor scales every image to its soft-token budget as far as the aspect ratio allows, so a lower bound has no effect
            kwargs["max_soft_tokens"] = _soft_token_budget(mm.image_max_tokens)
        return {**kwargs, **mm.processor_kwargs}

    def process(self, images: list[Any]) -> list[MMItem]:
        processor = self._image_processor()
        kwargs = self.get_mm_processor_kwargs(self.mm)
        items: list[MMItem] = []
        for pil in images:
            out = processor(images=pil, **kwargs)
            positions = out["image_position_ids"][0]
            valid = (positions != -1).all(dim=-1)
            positions = positions[valid].to(torch.int32)
            feature = out["pixel_values"][0][valid]
            n_soft = int(out["num_soft_tokens_per_image"][0])
            item_hash = content_hash(feature, positions.numpy().tobytes() + struct.pack("<i", n_soft))
            items.append(
                MMItem(
                    modality="image",
                    hash=item_hash,
                    pad_value=mm_pad_value(item_hash),
                    offsets=[],
                    feature=feature,
                    model_specific_data={"position_ids": positions, "num_soft_tokens": n_soft},
                )
            )
        return items

    def prompt_replacement(self, item: MMItem) -> PromptReplacement:
        full = [self.boi_token_id] + [self.image_token_id] * item.num_soft_tokens + [self.eoi_token_id]
        return PromptReplacement.select_token_id(full, self.image_token_id)

    def dummy_items(self, dtype: torch.dtype, device: torch.device) -> list[MMItem]:
        k = self.pooling_kernel_size
        ys, xs = torch.meshgrid(torch.arange(k), torch.arange(k), indexing="ij")
        positions = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).to(torch.int32)
        return [MMItem(
            modality="image",
            hash=0,
            pad_value=0,
            offsets=[[0, 1]],
            feature=torch.zeros(k * k, self.patch_dim, dtype=dtype, device=device),
            model_specific_data={"position_ids": positions.to(device), "num_soft_tokens": 1},
        )]


class Gemma4UnifiedMMProcessor(Gemma4MMProcessor):
    """The gemma4_unified release: one 48x48 super-patch per soft token and no pooling; the processor call and its budgets are the same."""

    def __init__(self, hf_config: Any, model_path: str, mm: MultimodalConfig) -> None:
        super().__init__(hf_config, model_path, mm)
        self.patch_dim = 3 * hf_config.vision_config.model_patch_size**2

    def dummy_items(self, dtype: torch.dtype, device: torch.device) -> list[MMItem]:
        return [MMItem(
            modality="image",
            hash=0,
            pad_value=0,
            offsets=[[0, 1]],
            feature=torch.zeros(1, self.patch_dim, dtype=dtype, device=device),
            model_specific_data={"position_ids": torch.zeros(1, 2, dtype=torch.int32, device=device), "num_soft_tokens": 1},
        )]


__all__ = ["Gemma4MMProcessor", "Gemma4UnifiedMMProcessor"]
