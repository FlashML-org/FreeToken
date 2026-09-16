"""Image preprocessing (upstream MIT attribution: NOTICE).

An image becomes a `n_vit_h x n_vit_w` patch grid for the ViT and a `n_llm_h x n_llm_w` token grid
after the 3x3 aligner downsample, which the LLM sees as

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

Every one of those positions carries `image_token_id` in `input_ids`; only the token type tells them
apart. The IMAGE slots are filled with aligner rows in reading order.
"""

import base64
import io
import ipaddress
import math
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

import numpy as np
import torch
from PIL import Image, ImageOps

TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_REQUEST_IMAGES = 16
MAX_IMAGE_PIXELS = 64 * 1024 * 1024


@dataclass
class ImageInput:
    start: int
    patches: torch.Tensor
    n_vit_h: int
    n_vit_w: int
    types: torch.Tensor


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height: int, best_width: int, patch_size: int, downsample_ratio: int):
    """Token grid the aligner produces from a patch grid of this pixel size."""
    return math.ceil((best_height // patch_size) / downsample_ratio), math.ceil(
        (best_width // patch_size) / downsample_ratio
    )


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    """Largest aspect-preserving pixel size whose token grid still fits in max_n_token."""
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:  # very tall: collapse to a single column
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:  # very wide: collapse to a single row
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return math.floor(height * beta / patch_size) * patch_size, math.floor(width * beta / patch_size) * patch_size


def safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token):
    """Shrink the pixel size until the image costs at most max_n_token LLM tokens."""
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def _validate_image_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in ("https", "http") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("images require an HTTP(S) URL or a base64 data URL")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except OSError as exc:
        raise ValueError("could not resolve image URL") from exc
    if not addresses or any(not ipaddress.ip_address(entry[4][0]).is_global for entry in addresses):
        raise ValueError("image URLs must resolve to public addresses")


class _ImageRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_image_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _decode_base64(data: str) -> bytes:
    if len(data) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
        raise ValueError("image exceeds the 32 MiB upload limit")
    try:
        decoded = base64.b64decode(data, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid base64 image") from exc
    if len(decoded) > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds the 32 MiB upload limit")
    return decoded


def load_image_bytes(record) -> bytes:
    """Load bounded API image data without allowing local filesystem paths."""
    data = record.get("data")
    if isinstance(data, bytes):
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("image exceeds the 32 MiB upload limit")
        return data
    if isinstance(data, str):
        return _decode_base64(data)

    source = record.get("source")
    if isinstance(source, dict):
        if source.get("data") is not None:
            return _decode_base64(source["data"])
        if source.get("url"):
            return load_image_bytes({"url": source["url"]})

    url = record.get("url")
    if isinstance(url, str) and url:
        if url.startswith("data:"):
            header, _, payload = url.partition(",")
            if ";base64" not in header:
                raise ValueError(f"Unsupported data URL encoding: {header}")
            return _decode_base64(payload)
        if url.startswith(("http://", "https://")):
            _validate_image_url(url)
            with build_opener(_ImageRedirectHandler()).open(url, timeout=30) as response:
                data = response.read(MAX_IMAGE_BYTES + 1)
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError("image exceeds the 32 MiB download limit")
            return data
        raise ValueError("images require an HTTP(S) URL or a base64 data URL")

    raise ValueError(f"Cannot load image from record: {list(record.keys())}")


def plan_image_grid(width: int, height: int, args):
    """Resize plan for an image of the given original size; a pure function of its arguments."""
    p = args.vision_patch_size
    if width <= 0 or height <= 0 or p <= 0 or args.vision_downsample_ratio <= 0 or args.vision_max_n_token < 4:
        raise ValueError("invalid image dimensions or vision configuration")
    if args.vision_max_wh_ratio is not None and width > height * args.vision_max_wh_ratio:
        width = height * args.vision_max_wh_ratio
    if 0 < width * height < args.vision_min_pixels:
        ratio = (args.vision_min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return safe_resize(height, width, best_height, best_width, p, args.vision_downsample_ratio, args.vision_max_n_token)


def load_image(record, args):
    """Load and transform one image record into ViT patches."""
    with Image.open(io.BytesIO(load_image_bytes(record))) as source:
        if source.width * source.height > MAX_IMAGE_PIXELS:
            raise ValueError("image exceeds the 64 megapixel limit")
        image = source.convert("RGB")
    return process_image(image, args)


def process_image(image, args):
    """Apply the native resize and normalization to an already decoded image."""
    p = args.vision_patch_size
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("image exceeds the 64 megapixel limit")
    image = image.convert("RGB")
    n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(image.width, image.height, args)
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if args.vision_max_wh_ratio is not None and image.width >= args.vision_max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = (x - 0.5) / 0.5
    patches = x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, p, p)
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def image_token_types(n_llm_h: int, n_llm_w: int) -> torch.Tensor:
    """Default layout: the aligner grid in reading order, one IMAGE_NEW_LINE per row."""
    types = [IMAGE_START]
    types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    types.append(IMAGE_END)
    return torch.tensor(types, dtype=torch.int64)


def prepare_vl_inputs(prompt, images, tokenizer, args):
    """Tokenize `prompt`, expanding each image placeholder token into its image span.

    Returns (tokens, token_types, image_inputs). Image-span positions carry `args.image_token_id` in
    `tokens` and are distinguished only by `token_types` (TEXT elsewhere). `image_inputs` is None when
    the prompt has no images."""
    from .encoding import IMAGE_PLACEHOLDER

    if len(images) > MAX_REQUEST_IMAGES:
        raise ValueError(f"at most {MAX_REQUEST_IMAGES} images are supported per request")

    # The placeholder is spelled differently across tokenizer revisions, so the id comes from the
    # config; only cross-check it when this tokenizer does know the training-time spelling.
    image_token_id = args.image_token_id
    placeholder_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    if placeholder_id is not None and placeholder_id != tokenizer.unk_token_id:
        assert placeholder_id == image_token_id, (placeholder_id, image_token_id)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    num_placeholders = sum(token == image_token_id for token in prompt_tokens)
    if num_placeholders != len(images):
        raise ValueError(f"Found {num_placeholders} image tokens but got {len(images)} images")
    if num_placeholders and not args.vision_enabled:
        raise ValueError("The model config has no vision tower (vision_n_layers == 0) but the prompt contains images")

    tokens, token_types, image_inputs = [], [], []
    image_iter = iter(images)
    for tok in prompt_tokens:
        if tok != image_token_id:
            tokens.append(tok)
            token_types.append(TEXT)
            continue
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(next(image_iter), args)
        types = image_token_types(n_llm_h, n_llm_w)
        image_inputs.append(ImageInput(len(tokens), patches, n_vit_h, n_vit_w, types))
        tokens += [image_token_id] * types.numel()
        token_types += types.tolist()
    return tokens, token_types, image_inputs or None
