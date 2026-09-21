"""MMProcessor: placeholder expansion on synthetic families, the Qwen VL processor and its positions, the registry (no checkpoint)."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
import torch

from freetoken.message import MMItem
from freetoken.mm import MM_PAD_SHIFT_VALUE
from freetoken.mm.config import MultimodalConfig
from freetoken.mm.processor import MMProcessor, PromptReplacement, image_positions
from freetoken.mm.processors.gemma4 import Gemma4MMProcessor, Gemma4UnifiedMMProcessor
from freetoken.mm.processors.glm5_next import Glm5NextMMProcessor
from freetoken.mm.processors.muse_glimmer import MuseGlimmerMMProcessor
from freetoken.mm.processors.minimax_m3 import IMAGE_END_ID, IMAGE_START_ID, MiniMaxM3MMProcessor
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
        super().__init__("/fake", MultimodalConfig())
        self.n_soft = n_soft

    def process(self, images):
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
    r = _Family(n_soft=3).apply(ids, [_png(), _png()])
    a, b = r.mm_items
    assert r.input_ids.tolist() == [1, 2, BOI, a.pad_value, a.pad_value, a.pad_value, EOI, 3, BOI, b.pad_value, b.pad_value, b.pad_value, EOI, 4]
    assert a.offsets == [[3, 6]] and b.offsets == [[9, 12]]
    assert a.num_tokens == 3 and r.mrope_positions is None and r.mrope_delta == 0


def test_replacement_with_several_embedding_runs_yields_several_spans():
    ids = torch.tensor([PLACEHOLDER, 5], dtype=torch.int32)
    r = _Tiled(n_soft=0).apply(ids, [_png()])
    (item,) = r.mm_items
    p = item.pad_value
    assert r.input_ids.tolist() == [p, p, BOI, p, p, BOI, p, p, BOI, 5]
    assert item.offsets == [[0, 2], [3, 5], [6, 8]] and item.num_tokens == 6


def test_placeholder_count_is_checked_before_any_image_is_decoded():
    ids = torch.tensor([PLACEHOLDER, PLACEHOLDER], dtype=torch.int32)
    with pytest.raises(ValueError, match="2 image placeholders but the request carries 1"):
        _Family(n_soft=1).apply(ids, [b"not an image"])


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
    proc = QwenVLMMProcessor(_hf_config(), "/nonexistent", MultimodalConfig())
    assert proc.image_token_id == 151655 and proc.is_mrope and proc.merge == 2
    assert proc.image_grid(_grid_item(8, 6, [[0, 12]])) == (4, 3)
    repl = proc.prompt_replacement(_grid_item(8, 6, []))
    assert proc.placeholder == [151655] and repl.full == [151655] * 12 and repl.embed_spans() == [[0, 12]]
    (dummy,) = proc.dummy_items(torch.bfloat16, torch.device("cpu"))
    dummy.validate()
    assert dummy.feature.shape == (4, 3 * 2 * 16 * 16) and dummy.num_tokens == 1
    assert dummy.grid_thw == [1, 2, 2]


def test_qwen_token_budget_and_kwargs_reach_the_image_processor():
    from freetoken.mm.config import MultimodalConfig

    calls = []

    class _FakeImageProcessor:
        size = {"shortest_edge": 65536, "longest_edge": 16777216}

        def __call__(self, images, **kwargs):
            calls.append(kwargs)
            return {"pixel_values": torch.zeros(4, 3 * 2 * 16 * 16), "image_grid_thw": torch.tensor([[1, 2, 2]])}

    mm = MultimodalConfig(image_min_tokens=100, image_max_tokens=1000, processor_kwargs={"do_convert_rgb": False})
    proc = QwenVLMMProcessor(_hf_config(), "/nonexistent", mm)
    proc._image_processor = lambda: _FakeImageProcessor()
    (item,) = proc.process([object()])
    assert item.grid_thw == [1, 2, 2]
    # one token is a 32x32 patch of the resized image; the budget travels as pixel areas in size, the extra kwarg rides along
    assert calls == [{"return_tensors": "pt", "size": {"shortest_edge": 100 * 1024, "longest_edge": 1000 * 1024}, "do_convert_rgb": False}]
    proc = QwenVLMMProcessor(_hf_config(), "/nonexistent", MultimodalConfig())
    proc._image_processor = lambda: _FakeImageProcessor()
    proc.process([object()])
    assert calls[-1] == {"return_tensors": "pt"}  # no budget, no size override


def test_qwen_processor_positions_follow_the_grid():
    proc = QwenVLMMProcessor(_hf_config(), "/nonexistent", MultimodalConfig())
    # 3 text tokens, one 8x6-patch image (4x3 = 12 llm tokens), 2 text tokens
    pos, delta = proc.positions(17, [_grid_item(8, 6, [[3, 15]])])
    assert pos.shape == (3, 17)
    assert pos[:, :3].tolist() == [[0, 1, 2]] * 3
    assert pos[0, 3:15].tolist() == [3] * 12
    assert pos[1, 3:15].tolist() == [3 + i // 3 for i in range(12)]
    assert pos[2, 3:15].tolist() == [3 + i % 3 for i in range(12)]
    assert pos[:, 15:].tolist() == [[7, 8]] * 3  # resumes at 3 + max(4, 3)
    assert delta == 8 + 1 - 17


def _gemma_config():
    return SimpleNamespace(
        architectures=["Gemma4ForConditionalGeneration"],
        image_token_id=258880,
        boi_token_id=255999,
        eoi_token_id=258882,
        vision_soft_tokens_per_image=280,
        vision_config=SimpleNamespace(patch_size=16, pooling_kernel_size=3),
        text_config=SimpleNamespace(vocab_size=262144),
    )


class _FakeGemmaImageProcessor:
    """Two soft tokens (18 valid patches) padded out to a 2520-row batch, like the real processor."""

    def __init__(self):
        self.calls = []

    def __call__(self, images, return_tensors, max_soft_tokens=None):
        self.calls.append(max_soft_tokens)
        rows = 2520
        pixels = torch.zeros(1, rows, 768)
        pixels[0, :18] = 0.75
        positions = torch.full((1, rows, 2), -1, dtype=torch.int64)
        positions[0, :18, 0] = torch.arange(18) % 6
        positions[0, :18, 1] = torch.arange(18) // 6
        return {"pixel_values": pixels, "image_position_ids": positions, "num_soft_tokens_per_image": torch.tensor([2])}


def _gemma_processor(mm=None):
    proc = Gemma4MMProcessor(_gemma_config(), "/nonexistent", mm or MultimodalConfig())
    fake = _FakeGemmaImageProcessor()
    proc._image_processor = lambda: fake
    return proc, fake


def test_gemma_items_strip_padding_and_wrap_the_soft_tokens_in_boi_eoi():
    proc, _ = _gemma_processor()
    (item,) = proc.process([object()])
    assert item.feature.shape == (18, 768) and item.feature.dtype == torch.float32
    assert item.feature[0, 0].item() == 0.75  # the [0, 1] pixels of the reference processor; the tower maps them to [-1, 1]
    assert item.position_ids.shape == (18, 2) and item.position_ids[7].tolist() == [1, 1]
    assert item.num_soft_tokens == 2 and item.pad_value >= MM_PAD_SHIFT_VALUE
    repl = proc.prompt_replacement(item)
    assert proc.placeholder == [258880]
    assert repl.full == [255999, 258880, 258880, 258882] and repl.embed_spans() == [[1, 3]]
    assert proc.positions(10, [item]) is None


def test_gemma_image_max_tokens_picks_an_accepted_soft_token_budget():
    for budget, tier in ((910, 560), (70, 70), (None, None)):  # the largest accepted budget within the limit
        proc, fake = _gemma_processor(MultimodalConfig(image_max_tokens=budget))
        proc.process([object()])
        assert fake.calls == [tier]
    with pytest.raises(ValueError, match="below the smallest"):  # a maximum no budget honors is refused at start-up instead of rounded up
        _gemma_processor(MultimodalConfig(image_max_tokens=60))
    proc, _ = _gemma_processor(MultimodalConfig(image_min_tokens=500, processor_kwargs={"max_soft_tokens": 1120}))
    # every image is scaled to its budget, so the minimum has no effect; an explicit kwarg wins over the budget
    assert proc.get_mm_processor_kwargs(proc.mm) == {"return_tensors": "pt", "max_soft_tokens": 1120}


def test_gemma_dummy_item_is_one_pooled_soft_token():
    proc, _ = _gemma_processor()
    (dummy,) = proc.dummy_items(torch.bfloat16, torch.device("cpu"))
    dummy.validate()
    assert dummy.feature.shape == (9, 768) and dummy.num_tokens == 1 and dummy.num_soft_tokens == 1
    assert dummy.position_ids.tolist() == [[x, y] for y in range(3) for x in range(3)]


def _gemma_unified_config():
    return SimpleNamespace(
        architectures=["Gemma4UnifiedForConditionalGeneration"],
        image_token_id=258880,
        boi_token_id=255999,
        eoi_token_id=258882,
        vision_config=SimpleNamespace(patch_size=16, pooling_kernel_size=3, model_patch_size=48, num_soft_tokens=280),
        text_config=SimpleNamespace(vocab_size=262144),
    )


class _FakeGemmaUnifiedImageProcessor:
    """Six super-patches padded out to the budget's rows, like the real processor: one row per soft token."""

    def __call__(self, images, return_tensors, max_soft_tokens=None):
        rows = max_soft_tokens or 280
        pixels = torch.zeros(1, rows, 6912)
        pixels[0, :6] = 0.25
        positions = torch.full((1, rows, 2), -1, dtype=torch.int64)
        positions[0, :6, 0] = torch.arange(6) % 3
        positions[0, :6, 1] = torch.arange(6) // 3
        return {"pixel_values": pixels, "image_position_ids": positions, "num_soft_tokens_per_image": torch.tensor([6])}


def test_gemma_unified_items_are_one_row_per_soft_token():
    proc = Gemma4UnifiedMMProcessor(_gemma_unified_config(), "/nonexistent", MultimodalConfig(image_max_tokens=100))
    proc._image_processor = lambda: _FakeGemmaUnifiedImageProcessor()
    (item,) = proc.process([object()])
    assert item.feature.shape == (6, 6912) and item.num_soft_tokens == 6 and item.position_ids[4].tolist() == [1, 1]
    assert proc.prompt_replacement(item).full == [255999] + [258880] * 6 + [258882]
    assert proc.get_mm_processor_kwargs(proc.mm) == {"return_tensors": "pt", "max_soft_tokens": 70}  # the same budgets as the tower release
    (dummy,) = proc.dummy_items(torch.bfloat16, torch.device("cpu"))
    dummy.validate()
    assert dummy.feature.shape == (1, 6912) and dummy.num_tokens == 1 and dummy.position_ids.tolist() == [[0, 0]]


def _glm_config():
    return SimpleNamespace(
        architectures=["Glm5NextForConditionalGeneration"],
        image_token_id=154854,
        vision_config=SimpleNamespace(spatial_merge_size=2, patch_size=14, temporal_patch_size=2, in_channels=3),
        text_config=SimpleNamespace(vocab_size=154880),
    )


def test_glm_token_budget_reaches_the_image_processor_and_rope_stays_one_dimensional():
    calls = []

    class _FakeImageProcessor:
        def __call__(self, images, **kwargs):
            calls.append(kwargs)
            return {"pixel_values": torch.zeros(16, 1176), "image_grid_thw": torch.tensor([[1, 4, 4]])}

    mm = MultimodalConfig(image_min_tokens=32, image_max_tokens=1024, processor_kwargs={"do_convert_rgb": False})
    proc = Glm5NextMMProcessor(_glm_config(), "/nonexistent", mm)
    proc._image_processor = lambda: _FakeImageProcessor()
    (item,) = proc.process([object()])
    # the budget is already in tokens; the extra kwarg rides along
    assert calls == [{"return_tensors": "pt", "min_image_tokens": 32, "max_image_tokens": 1024, "do_convert_rgb": False}]
    assert item.grid_thw == [1, 4, 4] and item.feature.dtype == torch.bfloat16
    # the template renders the begin/end wrapper tokens itself; only <|image|> expands
    assert proc.placeholder == [154854] and proc.prompt_replacement(item).full == [154854] * 4
    assert proc.positions(10, [item]) is None
    proc = Glm5NextMMProcessor(_glm_config(), "/nonexistent", MultimodalConfig())
    proc._image_processor = lambda: _FakeImageProcessor()
    proc.process([object()])
    assert calls[-1] == {"return_tensors": "pt"}
    (dummy,) = proc.dummy_items(torch.bfloat16, torch.device("cpu"))
    dummy.validate()
    assert dummy.feature.shape == (4, 1176) and dummy.num_tokens == 1


def _muse_config():
    return SimpleNamespace(
        architectures=["MuseGlimmerForConditionalGeneration"],
        image_token_id=200092,
        vision_config=SimpleNamespace(merge_size=2, patch_temporal=2, patch_size=14),
        text_config=SimpleNamespace(vocab_size=202048),
    )


def test_muse_wraps_the_pads_in_image_start_end_and_takes_only_a_token_maximum():
    calls = []

    class _FakeImageProcessor:
        def __call__(self, images, **kwargs):
            calls.append(kwargs)
            return {"pixel_values": torch.zeros(16, 1176), "image_grid_thw": torch.tensor([[1, 4, 4]])}

    mm = MultimodalConfig(image_min_tokens=32, image_max_tokens=1024, processor_kwargs={"do_convert_rgb": False})
    proc = MuseGlimmerMMProcessor(_muse_config(), "/nonexistent", mm)
    proc._image_processor = lambda: _FakeImageProcessor()
    (item,) = proc.process([object()])
    # the image processor has no minimum knob: only the maximum reaches it, the extra kwarg rides along
    assert calls == [{"return_tensors": "pt", "max_image_tokens": 1024, "do_convert_rgb": False}]
    assert item.grid_thw == [1, 4, 4] and item.feature.dtype == torch.bfloat16
    # the template renders one <|patch|>; the replacement adds the start/end wrapper the checkpoint processor renders
    repl = proc.prompt_replacement(item)
    assert proc.placeholder == [200092] and repl.full == [200080, *[200092] * 4, 200081]
    assert repl.embed_spans() == [[1, 5]] and proc.positions(10, [item]) is None
    proc = MuseGlimmerMMProcessor(_muse_config(), "/nonexistent", MultimodalConfig())
    proc._image_processor = lambda: _FakeImageProcessor()
    proc.process([object()])
    assert calls[-1] == {"return_tensors": "pt"}
    (dummy,) = proc.dummy_items(torch.bfloat16, torch.device("cpu"))
    dummy.validate()
    assert dummy.feature.shape == (4, 1176) and dummy.num_tokens == 1


def _minimax_config():
    # the checkpoint's own config shape: image_token_index, merge sizes under img_token_compression_config
    return SimpleNamespace(
        architectures=["MiniMaxM3SparseForConditionalGeneration"],
        image_token_index=200025,
        vision_config=SimpleNamespace(patch_size=14, num_channels=3, img_token_compression_config={"spatial_merge_size": 2, "temporal_patch_size": 2}),
        text_config=SimpleNamespace(vocab_size=200064),
    )


def test_minimax_token_budget_becomes_pixel_bounds_and_the_span_gets_start_end_tokens():
    calls = []

    class _FakeImageProcessor:
        size = {"shortest_edge": 3136, "longest_edge": 451584}

        def __call__(self, images, **kwargs):
            calls.append(kwargs)
            return {"pixel_values": torch.zeros(16, 1176), "image_grid_thw": torch.tensor([[1, 4, 4]])}

    mm = MultimodalConfig(image_min_tokens=8, image_max_tokens=256, processor_kwargs={"do_convert_rgb": False})
    proc = MiniMaxM3MMProcessor(_minimax_config(), "/nonexistent", mm)
    proc._image_processor = lambda: _FakeImageProcessor()
    (item,) = proc.process([object()])
    # 784 pixels per token: patch 14 x merge 2, squared
    assert calls == [{"return_tensors": "pt", "size": {"shortest_edge": 8 * 784, "longest_edge": 256 * 784}, "do_convert_rgb": False}]
    assert item.grid_thw == [1, 4, 4] and item.feature.dtype == torch.bfloat16
    repl = proc.prompt_replacement(item)
    assert proc.placeholder == [200025] and repl.full == [IMAGE_START_ID] + [200025] * 4 + [IMAGE_END_ID] and repl.embed_spans() == [[1, 5]]
    assert proc.positions(10, [item]) is None
    proc = MiniMaxM3MMProcessor(_minimax_config(), "/nonexistent", MultimodalConfig())
    proc._image_processor = lambda: _FakeImageProcessor()
    proc.process([object()])
    assert calls[-1] == {"return_tensors": "pt"}
    (dummy,) = proc.dummy_items(torch.bfloat16, torch.device("cpu"))
    dummy.validate()
    assert dummy.feature.shape == (4, 1176) and dummy.num_tokens == 1


def test_registry_resolves_by_architecture(monkeypatch):
    import freetoken.utils
    from freetoken.mm.processor import get_mm_processor

    configs = {
        "/qwen": _hf_config(),
        "/gemma": _gemma_config(),
        "/gemma-unified": _gemma_unified_config(),
        "/glm": _glm_config(),
        "/muse": _muse_config(),
        "/minimax": _minimax_config(),
        "/text-only": _hf_config(vision=False),
        "/unknown-vlm": _hf_config(arch="SomeOtherForConditionalGeneration"),
    }
    monkeypatch.setattr(freetoken.utils, "cached_load_hf_config", lambda p: configs[p])
    assert isinstance(get_mm_processor("/qwen"), QwenVLMMProcessor)
    assert get_mm_processor("/qwen", MultimodalConfig(disabled_encoders=frozenset({"vision"}))) is None  # --mm-disable vision
    assert isinstance(get_mm_processor("/gemma"), Gemma4MMProcessor)
    assert isinstance(get_mm_processor("/gemma-unified"), Gemma4UnifiedMMProcessor)
    assert isinstance(get_mm_processor("/glm"), Glm5NextMMProcessor)
    assert isinstance(get_mm_processor("/muse"), MuseGlimmerMMProcessor)
    assert isinstance(get_mm_processor("/minimax"), MiniMaxM3MMProcessor)
    assert get_mm_processor("/text-only") is None
    assert get_mm_processor("/unknown-vlm") is None
    assert get_mm_processor("/missing") is None  # a config that fails to load means no vision
