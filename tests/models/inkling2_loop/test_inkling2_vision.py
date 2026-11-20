"""Inkling 2 image processor and vision tower against transformers' Inkling2ImageProcessor and Inkling2VisionModel
(huggingface/transformers#51050), on random weights and on the Inkling-2-Mini checkpoint."""

from __future__ import annotations

import io
import json
import os
import re
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
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


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


# ---------------------------------------------------------------------------- tower


@pytest.fixture
def _single_rank():
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _hf_names(state: dict) -> dict:
    """Our tower's tensors under the names transformers loads the checkpoint's model.visual.* into."""
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping

    renames = [
        (src, dst)
        for rule in get_checkpoint_conversion_mapping("inkling2")
        if type(rule).__name__ == "WeightRenaming"
        for src, dst in zip(rule.source_patterns, rule.target_patterns)
    ]
    out = {}
    for key, value in state.items():
        name = f"model.visual.{key}"
        for src, dst in renames:
            name = re.sub(src, dst, name)
        assert name.startswith("model.vision_tower."), name
        out[name[len("model.vision_tower."):]] = value
    return out


def _reference_tower(config, state: dict, dtype: torch.dtype):
    from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel as HFVision

    config._attn_implementation = "sdpa"
    tower = HFVision(config).to("cuda", dtype).eval()
    missing, unexpected = tower.load_state_dict({k: v.to("cuda", dtype) for k, v in _hf_names(state).items()}, strict=False)
    # the rope tables are buffers the reference builds itself
    assert not unexpected and all("rotary_emb" in k for k in missing), (missing, unexpected)
    return tower


def _build(vc, dtype):
    from freetoken.models.inkling2_loop.vision import Inkling2VisionModel

    torch.set_default_dtype(dtype)
    try:
        with torch.device("cuda"):
            return Inkling2VisionModel(vc)
    finally:
        torch.set_default_dtype(torch.float32)


@needs_cuda
@pytest.mark.parametrize("placement", ["gpu", "host"])
def test_tower_matches_the_reference_on_random_weights(_single_rank, placement):
    """A 4x4 position table resampled up and down, two images of different grids encoded one at a time against the
    reference's packed batch."""
    from transformers.models.inkling2.configuration_inkling2 import Inkling2VisionConfig

    from freetoken.models.inkling2_loop.vision import parse_vision_config

    config = Inkling2VisionConfig(
        hidden_size=64, num_hidden_layers=3, num_attention_heads=4, patch_size=4, spatial_merge_size=2,
        posemb_grid_size=4, projection_hidden_dim=96, text_hidden_size=48,
    )
    vc = parse_vision_config(SimpleNamespace(vision_config=config))
    tower = _build(vc, torch.float32)
    torch.manual_seed(0)
    for value in tower.state_dict().values():
        value.normal_(0, 0.1)
    reference = _reference_tower(config, tower.state_dict(), torch.float32)
    tower.place_weights(placement)

    grids = [(2, 3), (3, 1)]
    pixels = [torch.rand(r * c, 8, 8, 3) * 2 - 1 for r, c in grids]
    with torch.no_grad():
        want = reference(
            pixel_values=torch.cat(pixels).cuda(), image_grid_thw=torch.tensor([[1, r, c] for r, c in grids])
        ).pooler_output
    got = torch.cat([tower.forward(p, [1, r, c]) for p, (r, c) in zip(pixels, grids)])
    assert got.shape == (sum(r * c for r, c in grids), 48)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def _visual_tensors(path: str) -> dict:
    from safetensors import safe_open

    with open(os.path.join(path, "model.safetensors.index.json")) as f:
        shards = {v for k, v in json.load(f)["weight_map"].items() if k.startswith("model.visual.")}
    out = {}
    for shard in shards:
        with safe_open(os.path.join(path, shard), framework="pt") as f:
            for key in f.keys():
                if key.startswith("model.visual."):
                    out[key[len("model.visual."):]] = f.get_tensor(key)
    return out


@needs_checkpoint
@needs_cuda
def test_tower_matches_the_reference_on_the_checkpoint(_single_rank):
    """bf16 like the engine; the bar is the reference's own bf16 error against its fp32 run."""
    from transformers import AutoConfig

    from freetoken.models.inkling2_loop.vision import parse_vision_config
    from freetoken.utils import cached_load_hf_config

    stored = _visual_tensors(CHECKPOINT)
    tower = _build(parse_vision_config(cached_load_hf_config(CHECKPOINT)), torch.bfloat16)
    # every checkpoint tensor under model.visual loads by its own name, none left over or missing
    tower.load_state_dict({k: v.cuda() for k, v in stored.items()})

    config = AutoConfig.from_pretrained(CHECKPOINT).vision_config
    tiles, rows, cols = preprocess(_image("RGB", 333, 217, seed=4), 40)
    grid = torch.tensor([[1, rows, cols]])
    with torch.no_grad():
        want32 = _reference_tower(config, stored, torch.float32)(pixel_values=tiles.cuda(), image_grid_thw=grid).pooler_output
        want16 = _reference_tower(config, stored, torch.bfloat16)(pixel_values=tiles.cuda(), image_grid_thw=grid).pooler_output
    got = tower.forward(tiles.to(torch.bfloat16), [1, rows, cols])
    assert got.shape == (rows * cols, 2304) and got.dtype == torch.bfloat16

    def rel(x):
        return ((x.float() - want32).norm() / want32.norm()).item()

    assert rel(got) <= 1.25 * rel(want16), (rel(got), rel(want16))
