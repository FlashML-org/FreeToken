"""Native DeepSeek image spans on the shared multimodal wire and cache."""

import struct
from dataclasses import replace

import torch

from freetoken.message import MMItem
from freetoken.mm import mm_pad_value
from freetoken.mm.processor import MMProcessor, MMResult, PromptReplacement, content_hash

from .args import load_args
from .image_processor import image_token_types, process_image


class DeepseekV41MMProcessor(MMProcessor):
    def __init__(self, hf_config, model_path, mm):
        super().__init__(model_path, mm)
        self.args = load_args(hf_config)
        if mm.image_min_tokens is not None:
            raise ValueError("DeepSeek-V4.1 uses processor_kwargs.vision_min_pixels instead of image_min_tokens")
        allowed = {"vision_min_pixels", "vision_max_n_token", "vision_max_wh_ratio"}
        if set(mm.processor_kwargs) - allowed:
            raise ValueError(f"Unsupported DeepSeek-V4.1 image processor options: {sorted(set(mm.processor_kwargs) - allowed)}")
        overrides = dict(mm.processor_kwargs)
        if mm.image_max_tokens is not None:
            overrides.setdefault("vision_max_n_token", mm.image_max_tokens)
        self.args = replace(self.args, **overrides)
        self.placeholder = [self.args.image_token_id]

    @staticmethod
    def _item(patches, n_h, n_w, types, start=0):
        feature = patches.to(device="cpu", dtype=torch.bfloat16).contiguous()
        types = torch.as_tensor(types, dtype=torch.int64).tolist()
        extra = struct.pack("<2i", n_h, n_w) + bytes(types)
        digest = content_hash(feature, extra)
        return MMItem(
            modality="image", hash=digest, pad_value=mm_pad_value(digest),
            offsets=[[start, start + len(types)]], feature=feature,
            model_specific_data={"n_vit_h": n_h, "n_vit_w": n_w, "types": types},
        )

    def from_media(self, input_ids, media):
        """Convert native chat-encoder output without reprocessing an image."""
        ids = input_ids.clone()
        items = []
        for image in media:
            image = image if isinstance(image, dict) else vars(image)
            item = self._item(image["patches"], image["n_vit_h"], image["n_vit_w"],
                              image["types"], image["start"])
            lo, hi = item.offsets[0]
            if lo < 0 or hi > ids.numel() or not torch.all(ids[lo:hi] == self.args.image_token_id):
                raise ValueError("DeepSeek-V4.1 image span does not match its placeholder tokens")
            ids[lo:hi] = item.pad_value
            item.validate()
            items.append(item)
        return MMResult(ids, items, None, 0)

    def process(self, images):
        items = []
        for image in images:
            patches, n_h, n_w, llm_h, llm_w = process_image(image, self.args)
            items.append(self._item(patches, n_h, n_w, image_token_types(llm_h, llm_w)))
        return items

    def prompt_replacement(self, item):
        return PromptReplacement([self.args.image_token_id] * len(item.types))

    def dummy_items(self, dtype, device):
        ratio, patch = self.args.vision_downsample_ratio, self.args.vision_patch_size
        return [MMItem(
            modality="image", hash=0, pad_value=0, offsets=[[0, 4]],
            feature=torch.zeros(ratio * ratio, 3, patch, patch, dtype=dtype, device=device),
            model_specific_data={"n_vit_h": ratio, "n_vit_w": ratio,
                                 "types": image_token_types(1, 1).tolist()},
        )]
