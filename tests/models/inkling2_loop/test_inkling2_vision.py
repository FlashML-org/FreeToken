"""Inkling 2 image processor against transformers' Inkling2ImageProcessor and Inkling2Processor
(huggingface/transformers#51050)."""

from __future__ import annotations

import io
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("transformers.models.inkling2", reason="needs transformers with Inkling 2 (huggingface/transformers#51050)")
from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor  # noqa: E402

from freetoken.mm import restore_placeholder  # noqa: E402
from freetoken.mm.config import MultimodalConfig  # noqa: E402
from freetoken.mm.processors.inkling2 import IMAGE_TOKEN_ID, Inkling2MMProcessor, preprocess  # noqa: E402

CHECKPOINT = os.environ.get("FREETOKEN_INKLING2_MODEL", "")
needs_checkpoint = pytest.mark.skipif(
    not os.path.exists(os.path.join(CHECKPOINT, "config.json")), reason="FREETOKEN_INKLING2_MODEL not set"
)


def _image(mode: str, width: int, height: int, seed: int = 0):
    from PIL import Image

    rng = np.random.default_rng(seed)
    if mode == "P":
        image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)).quantize(16)
        image.info["transparency"] = 3
        return image
    channels = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4}[mode]
    pixels = rng.integers(0, 256, (height, width, channels), dtype=np.uint8)
    return Image.fromarray(pixels[..., 0] if channels == 1 else pixels, mode)


def _png(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# odd sizes pad both edges; alpha and palette transparency take the checkerboard composite
@pytest.mark.parametrize(
    "mode, width, height",
    [("RGB", 123, 77), ("RGB", 120, 80), ("RGB", 7, 5), ("L", 90, 50), ("RGBA", 83, 61), ("LA", 45, 45), ("P", 64, 48)],
)
def test_tiles_match_the_reference_processor_pixel_for_pixel(mode, width, height):
    image = _image(mode, width, height)
    want = Inkling2ImageProcessor()(images=image, return_tensors="pt")
    tiles, rows, cols = preprocess(image, 40)
    assert [1, rows, cols] == want["image_grid_thw"][0].tolist()
    assert torch.equal(tiles, want["pixel_values"])


def test_each_placeholder_expands_to_one_token_per_tile_of_its_own_image():
    """Two images through apply: the request bytes keep their alpha, and each span is the image's own tile count."""
    hf_config = SimpleNamespace(vision_config=SimpleNamespace(native_patch_size=20, adapter_merge_size=2))
    processor = Inkling2MMProcessor(hf_config, "/unused", MultimodalConfig())
    images = [_image("RGBA", 83, 61, seed=1), _image("RGB", 200, 41, seed=2)]
    ids = torch.tensor([5, IMAGE_TOKEN_ID, 6, 7, IMAGE_TOKEN_ID, 8])
    result = processor.apply(ids, [_png(im) for im in images])
    reference = Inkling2ImageProcessor()
    expected = [5]
    for item, image, tail in zip(result.mm_items, images, ([6, 7], [8])):
        want = reference(images=image, return_tensors="pt")
        n = int(want["image_grid_thw"][0].prod())
        assert torch.equal(item.feature, want["pixel_values"].to(torch.bfloat16))
        assert item.offsets == [[len(expected), len(expected) + n]]
        expected += [IMAGE_TOKEN_ID] * n + tail
    assert restore_placeholder(result.input_ids, IMAGE_TOKEN_ID).tolist() == expected
    assert result.mrope_positions is None


@needs_checkpoint
def test_prompt_matches_the_reference_processor_on_the_checkpoint():
    """The chat template's placeholder, expanded by us, gives the ids transformers' Inkling2Processor gives."""
    from transformers import AutoProcessor, AutoTokenizer

    from freetoken.utils import cached_load_hf_config

    image = _image("RGBA", 161, 95, seed=3)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]}]
    hf = AutoProcessor.from_pretrained(CHECKPOINT)
    text = hf.apply_chat_template(messages, tokenize=False)
    want = hf(text=text, images=[image], return_tensors="pt")["input_ids"][0]
    ids = torch.tensor(AutoTokenizer.from_pretrained(CHECKPOINT)(text, add_special_tokens=False)["input_ids"])
    ours = Inkling2MMProcessor(cached_load_hf_config(CHECKPOINT), CHECKPOINT, MultimodalConfig())
    got = ours.apply(ids, [_png(image)]).input_ids
    assert restore_placeholder(got, ours.image_token_id).tolist() == want.tolist()
