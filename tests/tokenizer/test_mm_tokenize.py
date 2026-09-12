"""Image tokenization against real Qwen VL checkpoints (skipped when absent)."""

from __future__ import annotations

import io
import os

import pytest
import torch

from freetoken.mm import MM_PAD_SHIFT_VALUE, mm_pad_value

MODELS = [
    p
    for p in (os.environ.get("FREETOKEN_QWEN36_MODEL", ""), os.environ.get("FREETOKEN_QWEN3VL_MODEL", ""))
    if os.path.exists(os.path.join(p, "config.json"))
]

pytestmark = pytest.mark.skipif(
    not MODELS, reason="neither FREETOKEN_QWEN36_MODEL nor FREETOKEN_QWEN3VL_MODEL points at a checkpoint"
)


def _png(w, h):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 30, 200)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module", params=MODELS, ids=os.path.basename)
def manager(request):
    from freetoken.tokenizer.tokenize import TokenizeManager
    from freetoken.utils.hf import load_tokenizer

    return TokenizeManager(load_tokenizer(request.param))


def _msg(images, n_parts=1):
    from freetoken.core import SamplingParams
    from freetoken.message import TokenizeMsg

    content = [{"type": "text", "text": "描述"}] + [{"type": "image"}] * n_parts
    return TokenizeMsg(
        uid=1, text=[{"role": "user", "content": content}],
        sampling_params=SamplingParams(), images=images,
    )


def test_expansion_mrope_and_radix_ids(manager):
    r = manager.tokenize([_msg([_png(512, 512)])])[0]
    item = r.mm_items[0]
    L = r.input_ids.numel()
    assert item.num_tokens == 256 and item.grid_thw == [1, 32, 32]
    assert r.mrope_positions.shape == (3, L)
    assert int(r.mrope_positions.max()) + 1 - L == r.mrope_delta
    # the image span carries one content pad id above the vocab; text is untouched
    (start, end), = item.offsets
    span = r.input_ids[start:end]
    assert span.unique().numel() == 1
    assert span[0].item() == item.pad_value == mm_pad_value(item.hash) >= MM_PAD_SHIFT_VALUE
    assert bool((r.input_ids[:start] < MM_PAD_SHIFT_VALUE).all())


def test_same_image_same_hash_and_pad(manager):
    a = manager.tokenize([_msg([_png(512, 512)])])[0]
    b = manager.tokenize([_msg([_png(512, 512)])])[0]
    assert a.mm_items[0].hash == b.mm_items[0].hash
    assert torch.equal(a.input_ids, b.input_ids)
    c = manager.tokenize([_msg([_png(256, 256)])])[0]
    assert c.mm_items[0].hash != a.mm_items[0].hash


def test_max_pixels_clamps(manager):
    msg = _msg([_png(3840, 2160)])
    msg.mm_max_pixels = 2 * 1024 * 1024
    r = manager.tokenize([msg])[0]
    assert r.mm_items[0].num_tokens <= 2048


def test_count_prompt_tokens_counts_the_expanded_image(manager):
    import asyncio
    import base64
    from types import SimpleNamespace

    from freetoken.server.generation import count_prompt_tokens

    png = _png(512, 512)
    messages = [{"role": "user", "content": [
        {"type": "image", "freetoken_ref": {"kind": "b64", "data": base64.b64encode(png).decode()}},
        {"type": "text", "text": "描述"},
    ]}]
    state = SimpleNamespace(
        frontend_tokenizer=lambda: manager,
        config=SimpleNamespace(served_modalities=frozenset({"image"}), mm=SimpleNamespace(text_model_only=False, disabled_encoders=frozenset(), max_pixels=None)),
    )
    counted = asyncio.run(count_prompt_tokens(messages, None, {}, state))
    expanded = manager.tokenize([_msg([png])])[0].input_ids.numel()
    assert counted == expanded and counted > 256
