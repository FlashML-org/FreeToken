"""MMProcessor: placeholder expansion on synthetic families, the Qwen VL processor and its positions, the registry (no checkpoint)."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
import torch

from freetoken.message import MMItem
from freetoken.mm import MM_PAD_SHIFT_VALUE
from freetoken.mm.processor import MMProcessor, PromptReplacement, image_positions
from freetoken.mm.processors.qwen_vl import QwenVLMMProcessor

PLACEHOLDER = 7
BOI, SOFT, EOI = 90, 91, 92
ARCH = "Qwen3_5ForConditionalGeneration"


def _png():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    return buf.getvalue()


class _Family(MMProcessor):
    """Fixed-size images wrapped in begin/end tokens; only the soft tokens take embeddings."""

    placeholder = [PLACEHOLDER]

    def __init__(self, n_soft):
        self.n_soft = n_soft

    def process(self, images, max_pixels):
        return [
            MMItem(modality="image", hash=100 + i, pad_value=MM_PAD_SHIFT_VALUE + 100 + i, offsets=[], feature=torch.zeros(1))
            for i, _ in enumerate(images)
        ]

    def prompt_replacement(self, item):
        return PromptReplacement.select_token_id([BOI] + [SOFT] * self.n_soft + [EOI], SOFT)

    def dummy_items(self, dtype, device):
        raise NotImplementedError


class _Tiled(_Family):
    """Rows of soft tokens separated by a break token, like a tiled image."""

    def prompt_replacement(self, item):
        full = ([SOFT] * 2 + [BOI]) * 3
        return PromptReplacement.select_token_id(full, SOFT)


def test_replacement_keeps_wrapper_tokens_and_pads_only_embedding_slots():
    ids = torch.tensor([1, 2, PLACEHOLDER, 3, PLACEHOLDER, 4], dtype=torch.int32)
    r = _Family(n_soft=3).apply(ids, [_png(), _png()], None)
    a, b = r.mm_items
    assert r.input_ids.tolist() == [1, 2, BOI, a.pad_value, a.pad_value, a.pad_value, EOI, 3, BOI, b.pad_value, b.pad_value, b.pad_value, EOI, 4]
    assert a.offsets == [[3, 6]] and b.offsets == [[9, 12]]
    assert a.num_tokens == 3 and r.mrope_positions is None and r.mrope_delta == 0


def test_replacement_with_several_embedding_runs_yields_several_spans():
    ids = torch.tensor([PLACEHOLDER, 5], dtype=torch.int32)
    r = _Tiled(n_soft=0).apply(ids, [_png()], None)
    (item,) = r.mm_items
    p = item.pad_value
    assert r.input_ids.tolist() == [p, p, BOI, p, p, BOI, p, p, BOI, 5]
    assert item.offsets == [[0, 2], [3, 5], [6, 8]] and item.num_tokens == 6


def test_placeholder_count_is_checked_before_any_image_is_decoded():
    ids = torch.tensor([PLACEHOLDER, PLACEHOLDER], dtype=torch.int32)
    with pytest.raises(ValueError, match="2 image placeholders but the request carries 1"):
        _Family(n_soft=1).apply(ids, [b"not an image"], None)


def test_image_positions_freeze_t_spread_hw_and_advance_by_the_longer_side():
    # [text x2][img 2x3 = 6][text x1][img 3x1 = 3][text x2] -> 14 tokens
    pos, delta = image_positions(14, [(2, 6, 2, 3), (9, 3, 3, 1)])
    assert pos[:, :2].tolist() == [[0, 1]] * 3
    assert pos[0, 2:8].tolist() == [2] * 6
    assert pos[1, 2:8].tolist() == [2, 2, 2, 3, 3, 3]
    assert pos[2, 2:8].tolist() == [2, 3, 4, 2, 3, 4]
    assert pos[:, 8].tolist() == [5, 5, 5]  # 2 + max(2, 3)
    assert pos[0, 9:12].tolist() == [6] * 3
    assert pos[1, 9:12].tolist() == [6, 7, 8]
    assert pos[2, 9:12].tolist() == [6, 6, 6]
    assert pos[:, 12:].tolist() == [[9, 10]] * 3  # 6 + max(3, 1)
    assert delta == 10 + 1 - 14


def _hf_config(vision=True, mrope=True, arch=ARCH):
    rope = {"rope_theta": 1e7}
    if mrope:
        rope.update(mrope_section=[24, 20, 20], mrope_interleaved=True)
    return SimpleNamespace(
        architectures=[arch],
        image_token_id=151655,
        vision_config=(
            SimpleNamespace(spatial_merge_size=2, patch_size=16, temporal_patch_size=2, in_channels=3)
            if vision
            else None
        ),
        text_config=SimpleNamespace(rope_parameters=rope, vocab_size=248320),
    )


def _grid_item(h, w, offsets):
    return MMItem(
        modality="image", hash=1, pad_value=1, offsets=offsets,
        feature=torch.zeros(1), model_specific_data={"grid_thw": [1, h, w]},
    )


def test_qwen_processor_reads_config():
    proc = QwenVLMMProcessor(_hf_config(), "/nonexistent")
    assert proc.image_token_id == 151655 and proc.is_mrope and proc.merge == 2
    assert proc.image_grid(_grid_item(8, 6, [[0, 12]])) == (4, 3)
    repl = proc.prompt_replacement(_grid_item(8, 6, []))
    assert proc.placeholder == [151655] and repl.full == [151655] * 12 and repl.embed_spans() == [[0, 12]]
    (dummy,) = proc.dummy_items(torch.bfloat16, torch.device("cpu"))
    dummy.validate()
    assert dummy.feature.shape == (4, 3 * 2 * 16 * 16) and dummy.num_tokens == 1
    assert dummy.grid_thw == [1, 2, 2]


def test_qwen_processor_positions_follow_the_grid():
    proc = QwenVLMMProcessor(_hf_config(), "/nonexistent")
    # 3 text tokens, one 8x6-patch image (4x3 = 12 llm tokens), 2 text tokens
    pos, delta = proc.positions(17, [_grid_item(8, 6, [[3, 15]])])
    assert pos.shape == (3, 17)
    assert pos[:, :3].tolist() == [[0, 1, 2]] * 3
    assert pos[0, 3:15].tolist() == [3] * 12
    assert pos[1, 3:15].tolist() == [3 + i // 3 for i in range(12)]
    assert pos[2, 3:15].tolist() == [3 + i % 3 for i in range(12)]
    assert pos[:, 15:].tolist() == [[7, 8]] * 3  # resumes at 3 + max(4, 3)
    assert delta == 8 + 1 - 17


def test_registry_resolves_by_architecture(monkeypatch):
    import freetoken.utils
    from freetoken.mm.processor import get_mm_processor

    configs = {
        "/qwen": _hf_config(),
        "/text-only": _hf_config(vision=False),
        "/unknown-vlm": _hf_config(arch="SomeOtherForConditionalGeneration"),
    }
    monkeypatch.setattr(freetoken.utils, "cached_load_hf_config", lambda p: configs[p])
    get_mm_processor.cache_clear()
    assert isinstance(get_mm_processor("/qwen"), QwenVLMMProcessor)
    assert get_mm_processor("/text-only") is None
    assert get_mm_processor("/unknown-vlm") is None
    assert get_mm_processor("/missing") is None  # a config that fails to load means no vision
