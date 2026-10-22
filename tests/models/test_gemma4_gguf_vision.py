"""The Gemma 4 tower from a llama.cpp gemma4v / gemma4uv mmproj: the same vision config, token ids, image processor output and weights as the HF checkpoint it was converted from.

A BF16 mmproj stores the HF weights bit for bit once the channel-major patch layout (and the unified position table) is turned back, so any difference is a mapping bug.
"""
from __future__ import annotations

import os

import pytest
import torch

_PAIRS = {
    "gemma4v": ("FREETOKEN_GEMMA4_MMPROJ", "FREETOKEN_GEMMA4_MODEL"),
    "gemma4uv": ("FREETOKEN_GEMMA4_UNIFIED_MMPROJ", "FREETOKEN_GEMMA4_UNIFIED_MODEL"),
}


@pytest.fixture(scope="module", params=sorted(_PAIRS))
def gguf_side(request):
    from transformers import AutoConfig

    from freetoken.models.gemma4.gguf import gguf_mm_config, iter_mmproj_weights
    from freetoken.models.gguf.mmproj import MMPROJ_ENV, find_mmproj

    mmproj_env, hf_env = _PAIRS[request.param]
    mmproj, hf_dir = os.environ.get(mmproj_env, ""), os.environ.get(hf_env, "")
    if not (os.path.isfile(mmproj) and os.path.isdir(hf_dir)):
        pytest.skip(f"{mmproj_env} (BF16 mmproj) and {hf_env} (its HF checkpoint) not set")
    os.environ[MMPROJ_ENV] = mmproj
    try:
        mm = find_mmproj(mmproj)
    finally:
        del os.environ[MMPROJ_ENV]
    assert mm.projector == request.param
    hf = AutoConfig.from_pretrained(hf_dir)
    ids = {"<|image|>": hf.image_token_id, "<|image>": hf.boi_token_id, "<image|>": hf.eoi_token_id}
    tokens = ["<unused>"] * (max(ids.values()) + 1)
    for token, i in ids.items():
        tokens[i] = token
    # the text GGUF keys gguf_mm_config reads, as llama.cpp writes them for this checkpoint
    text_md = {"gemma4.embedding_length": hf.text_config.hidden_size, "tokenizer.ggml.tokens": tokens}
    fields, preprocessor = gguf_mm_config(text_md, mm, hf.text_config.vocab_size)
    return hf_dir, hf, fields, preprocessor, dict(iter_mmproj_weights(mm))


def test_config_and_image_processor(gguf_side):
    import numpy as np
    from PIL import Image
    from transformers import AutoImageProcessor
    from transformers.models.auto.image_processing_auto import get_image_processor_class_from_name

    from freetoken.models.gemma4.config import _parse_vision_config
    from freetoken.utils.hf import RawConfigShim

    hf_dir, hf, fields, preprocessor, _ = gguf_side
    width = hf.text_config.hidden_size
    assert _parse_vision_config(RawConfigShim(fields), width) == _parse_vision_config(hf, width)
    assert (fields["image_token_id"], fields["boi_token_id"], fields["eoi_token_id"]) == (hf.image_token_id, hf.boi_token_id, hf.eoi_token_id)
    ours = get_image_processor_class_from_name(preprocessor["image_processor_type"]).from_dict(preprocessor)
    ref = AutoImageProcessor.from_pretrained(hf_dir)
    rng = np.random.default_rng(0)
    for h, w in [(32, 32), (480, 640), (37, 2900), (2048, 1536)]:
        image = Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
        for kwargs in ({}, {"max_soft_tokens": 70}, {"max_soft_tokens": 1120}):
            a, b = ours(images=image, return_tensors="pt", **kwargs), ref(images=image, return_tensors="pt", **kwargs)
            assert a.keys() == b.keys()
            for key in a:
                assert torch.equal(a[key], b[key]), (key, h, w, kwargs)


def test_weights_bit_exact(gguf_side):
    from freetoken.models.gemma4.weight import iter_vision_weights

    hf_dir, _, _, _, ours = gguf_side
    ref = dict(iter_vision_weights(hf_dir, torch.device("cpu")))
    assert set(ours) == set(ref)
    for key, want in ref.items():
        assert ours[key].shape == want.shape, key
        assert torch.equal(ours[key].to(want.dtype), want), key
