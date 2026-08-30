"""Safe, deterministic Gemma4 image preprocessing for the online API.

The frontend accepts OpenAI's ``image_url`` data-URL representation and turns it
into CPU tensors that can safely cross the tokenizer-to-scheduler process
boundary.  Remote URL fetching is intentionally not implemented here: doing so
inside a LAN inference service would introduce an SSRF-capable network client.
The caller gets a precise error and can provide the same bytes as a data URL.
"""

from __future__ import annotations

import base64
import binascii
import io
import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError


_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_IMAGE_PIXELS = 16_000_000
_PATCH_SIZE = 16
_POOLING_KERNEL_SIZE = 3
_MAX_SOFT_TOKENS = 280


@dataclass(frozen=True)
class Gemma4ImageInputs:
    """One image in the exact tensor layout consumed by ``Gemma4VisionModel``."""

    pixel_values: torch.Tensor
    image_position_ids: torch.Tensor
    soft_token_count: int


def decode_openai_image_data_url(value: object) -> Image.Image:
    """Decode one OpenAI ``image_url`` value into a verified RGB Pillow image.

    OpenAI clients commonly send either the URL string directly or an object with
    a ``url`` member.  Only base64 ``data:image/*`` URLs are accepted.  The
    explicit byte and pixel limits avoid request-driven memory exhaustion before
    the image reaches the GPU-serving process.
    """
    url = value.get("url") if isinstance(value, dict) else value
    if not isinstance(url, str):
        raise ValueError("image_url must be a data:image URL string or an object with a url field")
    if not url.startswith("data:image/"):
        raise ValueError(
            "only data:image URLs are supported for local Gemma4 vision; "
            "download remote images client-side and send their bytes as a data URL"
        )
    header, separator, encoded = url.partition(",")
    if not separator or ";base64" not in header.lower():
        raise ValueError("image_url must use base64 data:image/...;base64,... encoding")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_url contains invalid base64 image data") from exc
    if not raw or len(raw) > _MAX_IMAGE_BYTES:
        raise ValueError(f"image_url must contain 1 to {_MAX_IMAGE_BYTES} bytes")
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            opened.verify()
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.width * opened.height > _MAX_IMAGE_PIXELS:
                raise ValueError(f"image_url exceeds {_MAX_IMAGE_PIXELS} decoded pixels")
            return opened.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("image_url does not contain a recognized image") from exc


def gemma4_image_inputs(image: Image.Image) -> Gemma4ImageInputs:
    """Resize, patchify, and position one RGB image using Gemma4's public contract.

    The resize equation matches Gemma4's processor: at most 280 soft tokens
    after 3-by-3 pooling, dimensions aligned to ``16 * 3`` pixels, and a
    lower bound of one pooled patch.  Every image is then padded to the fixed
    2,520-patch vision sequence used by the official processor. Patch pixels
    are channel-planar RGB values in [0, 1], exactly matching the projector's
    convolution-kernel layout before the model applies ``2 * (x - 0.5)``.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    source_width, source_height = image.size
    max_patches = _MAX_SOFT_TOKENS * _POOLING_KERNEL_SIZE**2
    source_patches = (source_height / _PATCH_SIZE) * (source_width / _PATCH_SIZE)
    scale = math.sqrt(max_patches / source_patches)
    unit = _PATCH_SIZE * _POOLING_KERNEL_SIZE
    target_height = max(unit, int(math.floor(source_height * scale / unit)) * unit)
    target_width = max(unit, int(math.floor(source_width * scale / unit)) * unit)
    resized = image.resize((target_width, target_height), Image.Resampling.BICUBIC)

    # HWC RGB -> [grid_y, grid_x, channels, patch_y, patch_x] -> flattened.
    # The sibling mmproj stores ``v.patch_embd.weight`` as a conventional
    # convolution kernel: [output, channel, patch_y, patch_x]. Its input vector
    # therefore contains one complete red patch, then green, then blue. An
    # RGB-interleaved vector is shape-compatible but produces wrong vision
    # features for otherwise simple, deterministic color controls.
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    grid_y, grid_x = target_height // _PATCH_SIZE, target_width // _PATCH_SIZE
    patches = (
        pixels.transpose(2, 0, 1)
        .reshape(3, grid_y, _PATCH_SIZE, grid_x, _PATCH_SIZE)
        .transpose(1, 3, 0, 2, 4)
        .reshape(grid_y * grid_x, 3 * _PATCH_SIZE**2)
    )
    # The vision implementation treats coordinate 0 as x and coordinate 1 as
    # y when assigning pooled spatial buckets, so emit that ordering directly.
    xs, ys = np.meshgrid(np.arange(grid_x), np.arange(grid_y), indexing="xy")
    positions = np.stack((xs.reshape(-1), ys.reshape(-1)), axis=-1).astype(np.int64)
    soft_token_count = (grid_y * grid_x) // _POOLING_KERNEL_SIZE**2
    assert soft_token_count <= _MAX_SOFT_TOKENS
    # The Gemma 4 vision tower derives its pooled output length from the input
    # tensor length, not from the count of valid patches. Pad each image to the
    # official max-patch budget so a 256-token image is processed in the same
    # 280-slot geometry as the reference implementation. The -1 coordinates
    # mark padding for both attention and the pooler's final validity mask.
    patches = np.pad(
        patches,
        ((0, max_patches - patches.shape[0]), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )
    positions = np.pad(
        positions,
        ((0, max_patches - positions.shape[0]), (0, 0)),
        mode="constant",
        constant_values=-1,
    )
    return Gemma4ImageInputs(
        pixel_values=torch.from_numpy(patches),
        image_position_ids=torch.from_numpy(positions),
        soft_token_count=soft_token_count,
    )


__all__ = ["Gemma4ImageInputs", "decode_openai_image_data_url", "gemma4_image_inputs"]
