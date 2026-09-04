"""mRoPE for qwen4_exp image prompts: FreeToken's port against the HF reference (CPU).

The HF checks need a transformers with ``qwen4_exp``; the pure-FreeToken checks always run.
"""

from __future__ import annotations

import base64
from types import MethodType, SimpleNamespace

import pytest
import torch
from freetoken.models.qwen4_exp.mrope import mrope_cos_sin, rope_index
from freetoken.server.generation import render_messages
from freetoken.tokenizer.tokenize import _expand_image_tokens

IMAGE = 248056
MERGE = 2
SECTION = (11, 11, 10)


def _prompt(
    grids: list[tuple[int, int, int]], text_len: int = 5, tail: int = 7
) -> torch.Tensor:
    ids = list(range(1, text_len + 1))
    for t, h, w in grids:
        ids += [IMAGE] * (t * (h // MERGE) * (w // MERGE)) + [50, 51]
    ids += list(range(100, 100 + tail))
    return torch.tensor(ids, dtype=torch.int32)


def test_rope_index_text_only():
    ids = _prompt([])
    pos, delta = rope_index(ids, None, IMAGE, MERGE)
    assert delta == 0
    assert torch.equal(pos, torch.arange(len(ids)).expand(3, -1))


def test_rope_index_image_layout():
    grid = torch.tensor([[1, 4, 6]])
    ids = _prompt([(1, 4, 6)], text_len=3, tail=2)
    pos, delta = rope_index(ids, grid, IMAGE, MERGE)
    # text 0..2, image at start 3: T=3, H in 3..4, W in 3..5, then text resumes at 3 + max(2, 3)
    img = pos[:, 3:9]
    assert img[0].tolist() == [3] * 6
    assert img[1].tolist() == [3, 3, 3, 4, 4, 4]
    assert img[2].tolist() == [3, 4, 5, 3, 4, 5]
    assert pos[:, 9].tolist() == [6, 6, 6]
    assert delta == int(pos.max()) + 1 - len(ids)
    assert delta <= 0


def test_mrope_cos_sin_text_rows_equal_cache():
    inv_freq = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float) / 64))
    pos = torch.arange(0, 37).expand(3, -1)
    table = mrope_cos_sin(pos, inv_freq, SECTION)
    freqs = torch.einsum("i,j -> ij", torch.arange(37, dtype=torch.float), inv_freq)
    assert torch.equal(table, torch.cat((freqs.cos(), freqs.sin()), dim=-1))


def test_expand_image_tokens():
    ids = torch.tensor([1, IMAGE, 2, IMAGE, 3], dtype=torch.int32)
    out = _expand_image_tokens(ids, IMAGE, [3, 2])
    assert out.tolist() == [1, IMAGE, IMAGE, IMAGE, 2, IMAGE, IMAGE, 3]
    with pytest.raises(ValueError):
        _expand_image_tokens(ids, IMAGE, [3])


def test_render_messages_images():
    png = b"\x89PNG fake"
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "hi"},
            ],
        },
    ]
    images: list[bytes] = []
    out = render_messages(messages, images)
    assert out[0]["content"] == "sys"  # text-only lists still flatten
    assert out[1]["content"] == [{"type": "image"}, {"type": "text", "text": "hi"}]
    assert images == [png]
    with pytest.raises(ValueError):  # text-only adapters keep refusing image parts
        render_messages(messages)
    with pytest.raises(ValueError):  # remote URLs are never fetched
        render_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
                    ],
                }
            ],
            [],
        )


# ----------------------------------------------------------------------------- HF reference
hf = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")


def _hf_rope_index(ids: torch.Tensor, grid: torch.Tensor | None):
    cfg = SimpleNamespace(
        image_token_id=IMAGE,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        vision_config=SimpleNamespace(spatial_merge_size=MERGE),
    )
    stub = SimpleNamespace(config=cfg)
    stub.get_vision_position_ids = MethodType(
        hf.Qwen4ExpModel.get_vision_position_ids, stub
    )
    ids = ids.view(1, -1).long()
    pos, delta = hf.Qwen4ExpModel.get_rope_index(
        stub, ids, mm_token_type_ids=(ids == IMAGE).int(), image_grid_thw=grid
    )
    return pos[:, 0], int(delta.reshape(-1)[0])


@pytest.mark.parametrize(
    "grids",
    [[(1, 4, 6)], [(1, 8, 8), (1, 2, 12)], [(1, 6, 2)]],
)
def test_rope_index_matches_hf(grids):
    ids = _prompt(grids)
    grid = torch.tensor(grids)
    pos, delta = rope_index(ids, grid, IMAGE, MERGE)
    ref_pos, ref_delta = _hf_rope_index(ids, grid)
    assert torch.equal(pos, ref_pos), (pos, ref_pos)
    assert delta == ref_delta


def test_mrope_cos_sin_matches_hf():
    text_cfg = SimpleNamespace(
        head_dim=256,
        hidden_size=2560,
        num_attention_heads=24,
        max_position_embeddings=262144,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1e7,
            "partial_rotary_factor": 0.25,
            "mrope_section": list(SECTION),
            "mrope_interleaved": True,
        },
    )
    rotary = hf.Qwen4ExpTextRotaryEmbedding(text_cfg)
    pos = torch.stack(
        [torch.randint(0, 5000, (3, 29)) for _ in range(1)], dim=1
    )  # [3, 1, 29]
    x = torch.zeros(1, 29, 256)
    cos, sin = rotary(x, pos)
    table = mrope_cos_sin(pos[:, 0], rotary.inv_freq.float(), SECTION)
    torch.testing.assert_close(table[:, :32], cos[0, :, :32], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(table[:, 32:], sin[0, :, :32], atol=1e-5, rtol=1e-5)
