"""The image processor refuses an image past the token cap: the reference never resizes, and the tower attends over
every patch of an image in one pass, so an unbounded image would take the engine's memory."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _processor(**mm):
    from freetoken.mm.config import MultimodalConfig
    from freetoken.mm.processors.inkling2 import Inkling2MMProcessor

    hf_config = SimpleNamespace(vision_config=SimpleNamespace(native_patch_size=20, adapter_merge_size=2))
    return Inkling2MMProcessor(hf_config, "/unused", MultimodalConfig(**mm))


def _image(width, height):
    from PIL import Image

    return Image.new("RGB", (width, height), (10, 20, 30))


def test_images_past_the_token_limit_are_refused():
    processor = _processor(image_max_tokens=4)
    # 80x80 is 2x2 tiles of 40 pixels; one more pixel of width starts a third column
    (item,) = processor.process([_image(80, 80)])
    assert item.grid_thw == [1, 2, 2]
    with pytest.raises(ValueError, match="takes 6 image tokens .* above the limit of 4"):
        processor.process([_image(81, 80)])


def test_the_default_limit_holds_without_the_flag():
    from freetoken.mm.processors.inkling2 import DEFAULT_MAX_IMAGE_TOKENS

    processor = _processor()
    assert processor.max_tokens == DEFAULT_MAX_IMAGE_TOKENS
    side = 40 * 129  # 129 x 129 tiles = 16641 > 16384
    with pytest.raises(ValueError, match="above the limit of 16384"):
        processor.process([_image(side, side)])
