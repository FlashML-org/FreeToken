"""CPU-only regression coverage for Gemma4 OpenAI image preprocessing."""

from __future__ import annotations

import base64
import io

import torch
from PIL import Image

from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg
from freetoken.tokenizer.gemma4_image import decode_openai_image_data_url, gemma4_image_inputs
from freetoken.tokenizer.tokenize import TokenizeManager


def _png_data_url() -> str:
    image = Image.new("RGB", (16, 16), (255, 0, 0))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def test_data_url_becomes_gemma4_patch_and_position_tensors() -> None:
    """A tiny image scales to a valid grid and preserves convolution channel planes."""
    inputs = gemma4_image_inputs(decode_openai_image_data_url({"url": _png_data_url()}))
    assert inputs.pixel_values.shape == (2520, 768)
    assert inputs.image_position_ids.shape == (2520, 2)
    assert inputs.soft_token_count == 256
    assert torch.equal(inputs.image_position_ids[0], torch.tensor([0, 0]))
    assert torch.equal(inputs.image_position_ids[1], torch.tensor([1, 0]))
    assert torch.equal(inputs.image_position_ids[2304], torch.tensor([-1, -1]))
    # v.patch_embd.weight has the standard [out, channel, patch_y, patch_x]
    # convolution layout. The first vector is therefore all red samples,
    # followed by green and blue, not one RGB triplet per source pixel.
    channel_size = 16 * 16
    first_patch = inputs.pixel_values[0]
    assert torch.equal(first_patch[:channel_size], torch.ones(channel_size))
    assert torch.equal(first_patch[channel_size : 2 * channel_size], torch.zeros(channel_size))
    assert torch.equal(first_patch[2 * channel_size :], torch.zeros(channel_size))


def test_remote_image_url_is_rejected_without_network_fetching() -> None:
    """The local inference endpoint must not become an arbitrary network client."""
    try:
        decode_openai_image_data_url("https://example.com/image.png")
    except ValueError as exc:
        assert "only data:image URLs" in str(exc)
    else:  # pragma: no cover - keeps the failure obvious if the security policy regresses
        raise AssertionError("remote image URL was unexpectedly accepted")


def test_tokenizer_expands_one_image_marker_and_stages_shaped_cpu_tensors() -> None:
    """The online tokenizer computes the placeholder count from the same processed image."""
    msg = TokenizeMsg(
        uid=1,
        text="unused",
        sampling_params=SamplingParams(),
        image_urls=[{"url": _png_data_url()}],
    )
    manager = TokenizeManager.__new__(TokenizeManager)
    prompt = manager._expand_gemma4_images(msg, "before<|freetoken-image|>after")
    assert prompt == "before<|image>" + "<|image|>" * 256 + "<image|>after"
    assert msg.mm_pixel_values is not None
    assert msg.mm_image_position_ids is not None
    assert msg.mm_pixel_values.shape == (1, 2520, 768)
    assert msg.mm_image_position_ids.shape == (1, 2520, 2)


def test_preflight_expansion_uses_the_same_single_marker_pass_as_worker() -> None:
    """A streamed request can validate image tensors without changing worker semantics."""

    msg = TokenizeMsg(
        uid=1,
        text=[{"role": "user", "content": "image request"}],
        sampling_params=SamplingParams(),
        image_urls=[{"url": _png_data_url()}],
    )
    manager = TokenizeManager.__new__(TokenizeManager)
    manager._sanitize_effort = lambda kwargs: kwargs
    manager._render = lambda _messages, _tools, _kwargs: "before<|freetoken-image|>after"

    rendered = manager.render_prompt(msg)
    prompt = manager._expand_gemma4_images(msg, rendered)

    assert prompt == "before<|image>" + "<|image|>" * 256 + "<image|>after"
    assert msg.mm_pixel_values is not None
    assert msg.mm_image_position_ids is not None
