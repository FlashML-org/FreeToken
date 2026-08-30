"""CPU-only regression coverage for Gemma4 OpenAI image preprocessing."""

from __future__ import annotations

import base64
import io

import torch
from PIL import Image

from freetoken.tokenizer.gemma4_image import decode_openai_image_data_url, gemma4_image_inputs


def _png_data_url() -> str:
    image = Image.new("RGB", (16, 16), (255, 0, 0))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def test_data_url_becomes_gemma4_patch_and_position_tensors() -> None:
    """A tiny image scales to the valid pooled grid and retains channel-major RGB."""
    inputs = gemma4_image_inputs(decode_openai_image_data_url({"url": _png_data_url()}))
    assert inputs.pixel_values.shape == (2304, 768)
    assert inputs.image_position_ids.shape == (2304, 2)
    assert inputs.soft_token_count == 256
    assert torch.equal(inputs.image_position_ids[0], torch.tensor([0, 0]))
    assert torch.equal(inputs.image_position_ids[1], torch.tensor([1, 0]))
    assert torch.allclose(inputs.pixel_values[0, :256], torch.ones(256))
    assert torch.allclose(inputs.pixel_values[0, 256:], torch.zeros(512))


def test_remote_image_url_is_rejected_without_network_fetching() -> None:
    """The local inference endpoint must not become an arbitrary network client."""
    try:
        decode_openai_image_data_url("https://example.com/image.png")
    except ValueError as exc:
        assert "only data:image URLs" in str(exc)
    else:  # pragma: no cover - keeps the failure obvious if the security policy regresses
        raise AssertionError("remote image URL was unexpectedly accepted")
