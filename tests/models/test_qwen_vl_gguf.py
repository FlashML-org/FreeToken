"""The Qwen VL tower from a llama.cpp qwen3vl_merger mmproj: the same vision config, image processor output and weights as the HF checkpoint it was converted from.

A BF16 mmproj stores the HF weights bit for bit once the split Conv3d patch embedding is stacked back, so any difference is a mapping bug.
"""
from __future__ import annotations

import glob
import os

import pytest
import torch

MMPROJ = os.environ.get("FREETOKEN_QWEN_MMPROJ", "")
HF_DIR = os.environ.get("FREETOKEN_QWEN36_MODEL", "")

pytestmark = pytest.mark.skipif(
    not (os.path.isfile(MMPROJ) and os.path.isdir(HF_DIR)),
    reason="FREETOKEN_QWEN_MMPROJ (BF16 mmproj) and FREETOKEN_QWEN36_MODEL (its HF checkpoint) not set",
)


@pytest.fixture(scope="module")
def gguf_side():
    from transformers import AutoConfig

    from freetoken.models.gguf.mmproj import MMPROJ_ENV, find_mmproj
    from freetoken.models.qwen3_vl.gguf import gguf_mm_config, iter_mmproj_weights

    os.environ[MMPROJ_ENV] = MMPROJ
    try:
        mm = find_mmproj(MMPROJ)
    finally:
        del os.environ[MMPROJ_ENV]
    hf = AutoConfig.from_pretrained(HF_DIR)
    text = hf.text_config
    # the text GGUF keys gguf_mm_config reads, as llama.cpp writes them for this checkpoint
    text_md = {
        "general.architecture": "qwen35",
        "qwen35.embedding_length": text.hidden_size,
        "qwen35.rope.dimension_sections": list(text.rope_parameters["mrope_section"]) + [0],
        "tokenizer.ggml.tokens": ["<unused>"] * hf.image_token_id + ["<|image_pad|>"],
    }
    fields, preprocessor = gguf_mm_config(text_md, mm, text.vocab_size)
    return hf, fields, preprocessor, dict(iter_mmproj_weights(mm))


def test_config_and_image_processor(gguf_side):
    import numpy as np
    from PIL import Image
    from transformers import AutoImageProcessor
    from transformers.models.auto.image_processing_auto import get_image_processor_class_from_name

    from freetoken.models.qwen3_vl.config import parse_vision_config
    from freetoken.utils.hf import RawConfigShim

    hf, fields, preprocessor, _ = gguf_side
    assert parse_vision_config(RawConfigShim(fields)) == parse_vision_config(hf)
    assert fields["image_token_id"] == hf.image_token_id
    ours = get_image_processor_class_from_name(preprocessor["image_processor_type"]).from_dict(preprocessor)
    ref = AutoImageProcessor.from_pretrained(HF_DIR)
    rng = np.random.default_rng(0)
    for h, w in [(32, 32), (480, 640), (37, 2900), (5000, 4000)]:
        image = Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
        a, b = ours(images=image, return_tensors="pt"), ref(images=image, return_tensors="pt")
        assert torch.equal(a["image_grid_thw"], b["image_grid_thw"])
        assert torch.equal(a["pixel_values"], b["pixel_values"])


def test_weights_bit_exact(gguf_side):
    import safetensors

    _, _, _, ours = gguf_side
    ref = {}
    for path in sorted(glob.glob(os.path.join(HF_DIR, "*.safetensors"))):
        with safetensors.safe_open(path, "pt") as f:
            for key in f.keys():
                if key.startswith("model.visual."):
                    ref[key.removeprefix("model.")] = f.get_tensor(key)
    assert set(ours) == set(ref)
    for key, want in ref.items():
        assert ours[key].shape == want.shape, key
        assert torch.equal(ours[key].to(want.dtype), want), key
