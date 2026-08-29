"""Server-side image preprocessing for GLM-5.3-Flash vision.

Vendored port of HF ``Glm5NextImageProcessorPil`` (transformers 5.16.1): the runtime
env pins transformers 5.15.1, which has no glm5_next classes, and swapping the whole
package under a serving stack for one preprocessing routine is not worth the risk.
Constants mirror the checkpoint's processor_config.json.

Pipeline per image (PIL in): smart_resize -> BICUBIC resize to the content box ->
zero-pad to the aligned canvas -> /255 -> CLIP normalize -> patchify (merge-block-major
patch order, per-patch feature layout [C][T][ph][pw] -- exactly what the vision tower's
Conv3d view expects). Output: pixel_values [P, 3*2*14*14] fp32 + grid (1, gh, gw)."""
from __future__ import annotations

import math
import os

import numpy as np

PATCH = 14
MERGE = 2
TEMPORAL = 2
MIN_TOKENS = 16
# Patch budget per image (HF default 8000 patches = 2000 LM tokens after the 2x2
# merge). Lower it to trade large-image fidelity for prompt length / encode time.
MAX_TOKENS = int(os.getenv("FREETOKEN_GLM5_MAX_IMAGE_TOKENS", "8000"))
IMAGE_TOKEN = "<|image|>"
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(3, 1, 1)
_MAX_BYTES = 30 * 1024 * 1024


def smart_resize(num_frames, height, width, temporal_factor=2, factor=28,
                 min_pixels=16, max_pixels=8000):
    """Aligned canvas within the spatiotemporal budget (verbatim HF port)."""
    pixels_per_token = temporal_factor * factor**2
    min_pixels *= pixels_per_token
    max_pixels *= pixels_per_token

    def align(value, f):
        return math.ceil(value / f) * f

    def fit_within_budget(aligned_frames):
        minimum_pixels = aligned_frames * factor**2
        if max_pixels < minimum_pixels:
            raise ValueError(f"max_pixels={max_pixels} too small for one aligned patch")
        low, high = 1, height
        best_height, best_width = factor, factor
        while low <= high:
            content_height = (low + high) // 2
            content_width = max(1, math.floor(width * content_height / height))
            candidate_height = align(content_height, factor)
            candidate_width = align(content_width, factor)
            if aligned_frames * candidate_height * candidate_width <= max_pixels:
                best_height, best_width = candidate_height, candidate_width
                low = content_height + 1
            else:
                high = content_height - 1
        return best_height, best_width

    aligned_frames = max(temporal_factor, round(num_frames / temporal_factor) * temporal_factor)
    aligned_height = align(height, factor)
    aligned_width = align(width, factor)
    aligned_pixel_budget = aligned_frames * aligned_height * aligned_width
    if aligned_pixel_budget < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        aligned_height = align(max(1, math.ceil(height * scale)), factor)
        aligned_width = align(max(1, math.ceil(width * scale)), factor)
        aligned_pixel_budget = aligned_frames * aligned_height * aligned_width
    if aligned_pixel_budget > max_pixels:
        aligned_height, aligned_width = fit_within_budget(aligned_frames)
    return aligned_height, aligned_width


def _patchify(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    """(C, H, W) fp32 -> ([gh*gw, C*T*P*P] block-major, gh, gw) (verbatim HF port)."""
    image = np.asarray(image, dtype=np.float32)
    channel, resized_height, resized_width = image.shape
    grid_h, grid_w = resized_height // PATCH, resized_width // PATCH
    patches = image.reshape(
        channel, grid_h // MERGE, MERGE, PATCH, grid_w // MERGE, MERGE, PATCH
    )
    patches = np.transpose(patches, (1, 4, 2, 5, 0, 3, 6))
    patches = np.broadcast_to(
        patches[:, :, :, :, :, None, :, :],
        (*patches.shape[:5], TEMPORAL, *patches.shape[5:]),
    )
    flat = patches.reshape(grid_h * grid_w, channel * TEMPORAL * PATCH * PATCH)
    return np.ascontiguousarray(flat), grid_h, grid_w


def preprocess_image(img) -> tuple[np.ndarray, tuple[int, int, int]]:
    from PIL import Image

    img = img.convert("RGB")
    width, height = img.size
    factor = PATCH * MERGE
    target_h, target_w = smart_resize(
        TEMPORAL, height, width, temporal_factor=TEMPORAL, factor=factor,
        min_pixels=MIN_TOKENS, max_pixels=MAX_TOKENS,
    )
    pixels_per_token = TEMPORAL * factor**2
    scale = min(target_h / height, target_w / width)
    if TEMPORAL * height * width >= pixels_per_token * MIN_TOKENS:
        scale = min(1.0, scale)  # never upscale an image already over the minimum
    content_h = max(1, min(target_h, math.floor(height * scale)))
    content_w = max(1, min(target_w, math.floor(width * scale)))
    if (content_h, content_w) != (height, width):
        img = img.resize((content_w, content_h), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)  # CHW
    arr = np.pad(arr, ((0, 0), (0, target_h - content_h), (0, target_w - content_w)))
    x = arr.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    flat, gh, gw = _patchify(x)
    return flat, (1, gh, gw)


def decode_image_part(image_url):
    """OpenAI image_url payload (str or {'url': ...}; data URI or http(s)) -> PIL image."""
    import base64
    import io
    import urllib.request

    from PIL import Image

    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    if not isinstance(url, str) or not url:
        raise ValueError("image_url must be a non-empty string or {'url': ...}")
    if url.startswith("data:"):
        _, _, b64 = url.partition(",")
        raw = base64.b64decode(b64)
    elif url.startswith(("http://", "https://")):
        req = urllib.request.Request(url, headers={"User-Agent": "freetoken-vision"})
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read(_MAX_BYTES + 1)
    else:
        raise ValueError("image_url must be a data: URI or http(s) URL")
    if len(raw) > _MAX_BYTES:
        raise ValueError("image too large (>30MB)")
    return Image.open(io.BytesIO(raw))





# ---------------------------------------------------------------------------------
# Video: sampled frames -> patch grid (t, gh, gw). Mirrors HF Glm5NextVideoProcessor
# (same smart_resize with a frame-count-aware budget, same content-box + zero-pad,
# same paired-frame temporal patchify); resize uses PIL bicubic like our image path
# (HF's video backend is torchvision -- visually identical, not bitwise).
# ---------------------------------------------------------------------------------
VIDEO_FPS = float(os.getenv("FREETOKEN_GLM5_VIDEO_FPS", "2"))
# Patch budget for the WHOLE clip (HF default 240000 -> 60000 LM tokens, far beyond
# what a PCIe-offload box should prefill). 8000 patches = 2000 LM tokens.
MAX_VIDEO_TOKENS = int(os.getenv("FREETOKEN_GLM5_MAX_VIDEO_TOKENS", "8000"))
MAX_VIDEO_FRAMES = int(os.getenv("FREETOKEN_GLM5_MAX_VIDEO_FRAMES", "64"))
_MAX_VIDEO_BYTES = 128 * 1024 * 1024


def sample_frame_indices(total_frames: int, native_fps: float, duration: float):
    """Frame indices at FREETOKEN_GLM5_VIDEO_FPS spacing, even count (HF sampler)."""
    max_seconds = int(duration) if duration else max(1, round(total_frames / max(native_fps, 1e-6)))
    extract_t = min(int(max(duration, 1e-6) * VIDEO_FPS) or 2, MAX_VIDEO_FRAMES)
    timestamps = [i / max(native_fps, 1e-6) for i in range(total_frames)]
    if total_frames < extract_t:
        idx = np.linspace(0, total_frames - 1, extract_t, dtype=int).tolist()
    else:
        idx = []
        current = 0.0
        inv = 1.0 / VIDEO_FPS
        for i in range(total_frames):
            if timestamps[i] >= current:
                current += inv
                idx.append(i)
                if current >= max_seconds:
                    break
        if len(idx) < extract_t:
            a, b = (idx[0], idx[-1]) if idx else (0, max(total_frames - 1, 0))
            idx = np.linspace(a, b, extract_t, dtype=int).tolist()
        elif len(idx) > extract_t:
            idx = np.linspace(0, total_frames - 1, extract_t, dtype=int).tolist()
    seen, uniq = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i); uniq.append(i)
    if len(uniq) & 1:
        uniq.append(uniq[-1])
    ts = [timestamps[i] for i in uniq]
    return uniq, ts


def preprocess_video(frames, timestamps):
    """frames: even-length list of PIL images (same size). Returns
    (pixel_values [t*gh*gw, D] fp32, grid (t, gh, gw), unit_timestamps [t])."""
    from PIL import Image

    T = len(frames)
    assert T >= 2 and T % 2 == 0, "sampled frame count must be even"
    frames = [f.convert("RGB") for f in frames]
    width, height = frames[0].size
    factor = PATCH * MERGE
    target_h, target_w = smart_resize(
        T, height, width, temporal_factor=TEMPORAL, factor=factor,
        min_pixels=MIN_TOKENS, max_pixels=MAX_VIDEO_TOKENS,
    )
    pixels_per_token = TEMPORAL * factor**2
    scale = min(target_h / height, target_w / width)
    if T * height * width >= pixels_per_token * MIN_TOKENS:
        scale = min(1.0, scale)
    content_h = max(1, min(target_h, math.floor(height * scale)))
    content_w = max(1, min(target_w, math.floor(width * scale)))
    chw = []
    for f in frames:
        if (content_h, content_w) != (height, width):
            f = f.resize((content_w, content_h), Image.BICUBIC)
        a = np.asarray(f, dtype=np.uint8).transpose(2, 0, 1)
        a = np.pad(a, ((0, 0), (0, target_h - content_h), (0, target_w - content_w)))
        chw.append(a)
    vid = np.stack(chw).astype(np.float32) / 255.0          # [T, C, H, W]
    vid = (vid - _MEAN[None]) / _STD[None]
    grid_t, gh, gw = T // TEMPORAL, target_h // PATCH, target_w // PATCH
    x = vid.reshape(grid_t, TEMPORAL, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    x = np.transpose(x, (0, 3, 6, 4, 7, 2, 1, 5, 8))        # unit, block-major, m, m, C, T, ph, pw
    flat = np.ascontiguousarray(
        x.reshape(grid_t * gh * gw, 3 * TEMPORAL * PATCH * PATCH)
    )
    unit_ts = [float(timestamps[min(k * TEMPORAL, T - 1)]) for k in range(grid_t)]
    return flat, (grid_t, gh, gw), unit_ts


def decode_video_part(video_url):
    """OpenAI-style video_url payload -> (frames list of PIL, unit timestamps).
    data: URI (base64 mp4/webm) or http(s) URL; decoded via PyAV."""
    import base64
    import io
    import tempfile
    import urllib.request

    import av
    from PIL import Image

    url = video_url.get("url") if isinstance(video_url, dict) else video_url
    if not isinstance(url, str) or not url:
        raise ValueError("video_url must be a non-empty string or {'url': ...}")
    if url.startswith("data:"):
        _, _, b64 = url.partition(",")
        raw = base64.b64decode(b64)
    elif url.startswith(("http://", "https://")):
        req = urllib.request.Request(url, headers={"User-Agent": "freetoken-vision"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read(_MAX_VIDEO_BYTES + 1)
    else:
        raise ValueError("video_url must be a data: URI or http(s) URL")
    if len(raw) > _MAX_VIDEO_BYTES:
        raise ValueError("video too large (>128MB)")

    with av.open(io.BytesIO(raw)) as container:
        stream = container.streams.video[0]
        native_fps = float(stream.average_rate or 24.0)
        decoded = [f for f in container.decode(stream)]
    total = len(decoded)
    if total == 0:
        raise ValueError("could not decode any video frames")
    duration = total / max(native_fps, 1e-6)
    idx, ts = sample_frame_indices(total, native_fps, duration)
    frames = [Image.fromarray(decoded[i].to_ndarray(format="rgb24")) for i in idx]
    return frames, ts

__all__ = ["preprocess_image", "decode_image_part", "preprocess_video", "decode_video_part", "smart_resize", "IMAGE_TOKEN", "MERGE"]
