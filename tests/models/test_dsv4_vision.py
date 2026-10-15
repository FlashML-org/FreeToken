"""DeepSeek-V4 image preprocessing, block layout and the tiny vision tower (no checkpoint).

The block layout is what makes an image's token count depend on where it lands in the
prompt, so it is pinned here against hand-written expectations; the checkpoint-gated test
at the bottom checks the same functions against the reference ``inference/`` code.
"""

from __future__ import annotations

import io
import os

import numpy as np
import pytest
import torch
from PIL import Image

from freetoken.models.deepseek_v4.config import DSV4VisionConfig, parse_config, parse_vision_config
from freetoken.models.deepseek_v4.image_processor import load_image
from freetoken.models.deepseek_v4.vision import (
    COMPRESS_PAD_TO,
    DSV4Aligner,
    DSV4VisionTower,
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PAD,
    IMAGE_START,
    assemble_block,
    build_image_block,
)

checkpoint_path = os.environ.get("FREETOKEN_DSV4_VISION_CKPT")
needs_checkpoint = pytest.mark.skipif(
    not checkpoint_path, reason="needs FREETOKEN_DSV4_VISION_CKPT pointing at the vision checkpoint"
)

_HF_CONFIG = {
    "vision_n_layers": 32,
    "vision_dim": 1024,
    "vision_n_heads": 16,
    "vision_inter_dim": 2816,
    "vision_patch_size": 14,
    "vision_rope_theta": 10000.0,
    "vision_downsample_ratio": 3,
    "vision_max_n_token": 384,
    "vision_min_pixels": 147456,
    "vision_max_wh_ratio": 8,
    "hidden_size": 4096,
}


def _hf_config(**over):
    from types import SimpleNamespace

    return SimpleNamespace(**{**_HF_CONFIG, **over})


def _vc(**over) -> DSV4VisionConfig:
    """A tower small enough to run: 2 patch-2 blocks over a 2x2 merge grid."""
    base = dict(
        vision_n_layers=2,
        vision_dim=16,
        vision_n_heads=2,
        vision_inter_dim=32,
        vision_patch_size=4,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=2,
        vision_max_n_token=384,
        vision_min_pixels=147456,
        vision_max_wh_ratio=8,
        text_dim=64,
    )
    return DSV4VisionConfig(**{**base, **over})


def _png(width, height, seed=0):
    rng = np.random.default_rng(seed)
    array = (rng.random((height, width, 3)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(array, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_a_vision_checkpoint_config_becomes_the_tower_dims_and_a_text_one_does_not():
    vc = parse_vision_config(_hf_config())
    assert (vc.vision_n_layers, vc.vision_dim, vc.text_dim) == (32, 1024, 4096)
    # the engine nulls the section on the copy it hands the parser when it serves text-only
    assert parse_vision_config(_hf_config(vision_n_layers=None)) is None
    assert parse_vision_config(_hf_config(vision_n_layers=0)) is None


def test_block_layout_interleaves_rows_and_pads_to_the_compression_stride():
    types, perm = build_image_block(2, 2, 0)
    # 3 lead pads put the first row token on a COMPRESS_PAD_TO (4) boundary, then START,
    # the N-layout rows (column pairs: 2 rows of 2 IMAGE tokens plus their row markers),
    # the trailing parity pad, END.
    assert types.tolist() == (
        [IMAGE_PAD] * 3
        + [IMAGE_START]
        + [IMAGE, IMAGE, IMAGE, IMAGE, IMAGE_NEW_LINE, IMAGE_NEW_LINE]
        + [IMAGE_PAD] * 2
        + [IMAGE_END]
    )
    # row-major aligner outputs 0..3 land in the IMAGE slots in N-layout order
    assert perm.tolist() == [0, 2, 1, 3]


def test_the_block_shrinks_by_one_token_per_lead_pad_the_offset_absorbs():
    # the lead pads are what aligns the block, so the same image at a later offset is a
    # SHORTER span: the count cannot be a property of the image alone
    lengths = [len(build_image_block(2, 2, start)[0]) for start in range(4)]
    assert lengths == [13, 12, 11, 10]
    # an odd row count adds a marker row, which is not pad bytes
    assert [len(build_image_block(3, 2, start)[0]) for start in range(4)] == [17, 16, 15, 14]


def test_assemble_block_puts_the_sentinels_and_the_aligner_rows_where_the_types_say():
    types, perm = build_image_block(2, 2, 0)
    sentinels = torch.stack([torch.full((4,), float(i)) for i in range(5)])
    aligned = torch.arange(4 * 4, dtype=torch.float32).view(4, 4) * 10
    block = assemble_block(aligned, types, perm, sentinels)
    assert block.shape == (len(types), 4)
    # sentinels index the type enum, so START/PAD/END rows carry 0/1/4 and the IMAGE rows
    # carry the aligner output perm says they take
    assert block[0].tolist() == [1.0] * 4  # lead pad
    assert block[3].tolist() == [0.0] * 4  # IMAGE_START
    assert block[-1].tolist() == [4.0] * 4  # IMAGE_END
    image_rows = block[types == IMAGE]
    assert image_rows.tolist() == aligned[perm].tolist()


def test_preprocessing_resizes_to_the_merge_grid_and_normalizes_like_the_reference():
    # min_pixels 0 keeps the input size: the reference upscales below it
    vc = _vc(vision_min_pixels=0)
    patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(
        Image.open(io.BytesIO(_png(80, 64))), vc
    )
    assert (n_vit_h, n_vit_w, n_llm_h, n_llm_w) == (16, 20, 8, 10)
    assert patches.shape == (n_vit_h * n_vit_w, 3, 4, 4)
    assert patches.dtype == torch.bfloat16
    assert float(patches.min()) >= -1.0 and float(patches.max()) <= 1.0
    # identical bytes, identical patches: the content hash and the radix key ride on these
    again, *_ = load_image(Image.open(io.BytesIO(_png(80, 64))), vc)
    assert torch.equal(patches, again)


def test_the_token_budget_caps_a_tall_image():
    vc = _vc(vision_max_n_token=64, vision_min_pixels=0)
    _patches, _n_vit_h, _n_vit_w, n_llm_h, n_llm_w = load_image(
        Image.open(io.BytesIO(_png(64, 1024))), vc
    )
    types, _perm = build_image_block(n_llm_h, n_llm_w, 0)
    assert len(types) <= vc.vision_max_n_token


def test_tiny_tower_runs_and_its_aligner_merges_every_merge_grid():
    vc = _vc()
    tower = DSV4VisionTower(vc)
    aligner = DSV4Aligner(vc)
    patches = torch.zeros(20 * 24, 3, 4, 4, dtype=torch.bfloat16)
    out = tower.forward(patches, 20, 24)
    assert out.shape == (20 * 24, vc.vision_dim)
    assert aligner.forward(out, 20, 24).shape == (10 * 12, vc.text_dim)


def _reference_modules(path):
    """Import the checkpoint's own ``inference/image_processor.py`` (PIL + torch only)."""
    import importlib.util

    module_path = os.path.join(path, "inference", "image_processor.py")
    if not os.path.exists(module_path):
        pytest.skip(f"{module_path} not found")
    spec = importlib.util.spec_from_file_location("_dsv4_reference_image_processor", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_checkpoint
def test_preprocessing_and_block_layout_match_the_reference():
    from types import SimpleNamespace

    reference = _reference_modules(checkpoint_path)
    vc = parse_vision_config(_hf_config())
    args = SimpleNamespace(
        vision_patch_size=vc.vision_patch_size,
        vision_downsample_ratio=vc.vision_downsample_ratio,
        vision_max_n_token=vc.vision_max_n_token,
        vision_min_pixels=vc.vision_min_pixels,
        vision_max_wh_ratio=vc.vision_max_wh_ratio,
    )
    for width, height in ((640, 480), (100, 100), (3000, 400), (1600, 1200)):
        raw = _png(width, height)
        mine = load_image(Image.open(io.BytesIO(raw)), vc)
        theirs = reference.load_image({"data": raw}, args)
        assert torch.equal(mine[0], theirs[0])
        assert mine[1:] == theirs[1:]
        for start in (0, 1, 3, 7):
            types, perm = build_image_block(mine[3], mine[4], start)
            ref_types, ref_perm = reference.build_image_block(theirs[3], theirs[4], start)
            assert torch.equal(types, ref_types) and torch.equal(perm, ref_perm)


_WINDOW, _MAX_IMG = 128, 384


def _ref_get_image_visible(input_ids: torch.Tensor, vocab_size: int, max_image_tokens: int):
    """Verbatim port of the checkpoint's ``inference/model.py`` pair (not vendored in this
    repo): ``get_image_visible`` derives spans from the virtual ids ``vocab + type``, so the
    lead pads before IMAGE_START stay outside; only the START/END comparisons matter, our
    type enum values index the same way."""
    seqlen = input_ids.size(1)
    idx = torch.arange(seqlen, dtype=torch.int32).unsqueeze(0)
    is_start = input_ids == vocab_size + IMAGE_START
    is_end = input_ids == vocab_size + IMAGE_END
    valid = (is_start.cumsum(1) > is_end.cumsum(1)) | is_end
    starts = torch.where(is_start, idx, 0).cummax(1)[0]
    left = (idx - starts) * valid
    ends = torch.where(is_end, idx, seqlen).flip(1).cummin(1)[0].flip(1)
    right = (ends - idx) * valid
    return left.clamp(max=max_image_tokens - 1), right.clamp(max=max_image_tokens)


def _ref_get_window_topk_idxs_visible(window_size, seqlen, left, right, max_image_tokens):
    width = min(seqlen, window_size + max_image_tokens)
    idx = torch.arange(seqlen).unsqueeze(0)
    left_add = (left - (window_size - 1)).clamp(min=0)
    starts = (idx - (window_size - 1) - left_add).clamp(min=0)
    matrix = starts.unsqueeze(-1) + torch.arange(width)
    return torch.where(matrix > (idx + right).unsqueeze(-1), -1, matrix).int().contiguous()


def _reference_candidates(prefix: int, n_llm_h: int, n_llm_w: int, vocab: int = 100_000):
    """The reference pair's window candidates for a text prefix followed by one image block."""
    types, _perm = build_image_block(n_llm_h, n_llm_w, prefix)
    ids = torch.full((1, prefix + len(types)), vocab + 50, dtype=torch.int64)
    ids[0, prefix:] = vocab + types
    n = ids.size(1)
    left, right = _ref_get_image_visible(ids, vocab, _MAX_IMG)
    block = (prefix, prefix + len(types))
    return block, types, _ref_get_window_topk_idxs_visible(_WINDOW, n, left, right, _MAX_IMG)[0]


def test_model_spans_match_the_reference_visibility_pair():
    from types import SimpleNamespace

    from freetoken.models.deepseek_v4.attention import Attention
    from freetoken.models.deepseek_v4.model import _image_spans_by_ti

    for prefix, h, w in ((0, 3, 2), (1, 3, 2), (2, 3, 2), (3, 3, 2), (5, 4, 3), (201, 20, 16), (402, 2, 2)):
        (lo, hi), types, ref = _reference_candidates(prefix, h, w)
        req = SimpleNamespace(table_idx=7, mm_items=[SimpleNamespace(offsets=[[lo, hi]])])
        spans = _image_spans_by_ti([req])
        assert spans == {7: [(lo + types.tolist().index(IMAGE_START), hi)]}
        cols = Attention.visible_window_cols(
            spans[7], 0, hi, _WINDOW, _MAX_IMG, torch.device("cpu"),
            addr_lo=0, width=min(hi, _WINDOW + _MAX_IMG),
        )
        assert torch.equal(cols, ref)
    text = SimpleNamespace(table_idx=0, mm_items=None)
    assert _image_spans_by_ti([text]) is None


def test_a_span_started_at_the_first_lead_pad_breaks_reference_parity():
    from freetoken.models.deepseek_v4.attention import Attention

    # the maintainer's case: a 201-token prefix puts 2 lead pads in front of a 20x16 grid,
    # and a span covering the whole block diverges from the reference on 218 rows
    (lo, hi), _types, ref = _reference_candidates(201, 20, 16)
    assert lo + COMPRESS_PAD_TO - 1 - lo % COMPRESS_PAD_TO - lo == 2  # the two lead pads
    cols = Attention.visible_window_cols(
        [(lo, hi)], 0, hi, _WINDOW, _MAX_IMG, torch.device("cpu"),
        addr_lo=0, width=min(hi, _WINDOW + _MAX_IMG),
    )
    assert int((cols != ref).any(dim=1).sum()) == 218


def test_a_vision_checkpoint_keeps_image_blocks_whole_across_chunk_boundaries(tmp_path):
    """Image spans attend bidirectionally, so a vision checkpoint sets
    bidirectional_mm_blocks on its attention group and the scheduler ends a chunk before a
    block instead of splitting it; text-only defaults to False and keeps plain chunking."""
    from types import SimpleNamespace

    from freetoken.core import Context, SamplingParams, get_global_ctx, set_global_ctx
    from freetoken.message import MMItem
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.mm import cut_image_spans
    from freetoken.scheduler.prefill import PrefillManager
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    # DeepseekV4Args defaults fill an empty inference/config.json; the hf namespace carries the tower
    (tmp_path / "inference").mkdir()
    (tmp_path / "inference" / "config.json").write_text("{}")
    vision = parse_config(SimpleNamespace(_name_or_path=str(tmp_path), **_HF_CONFIG))
    text = parse_config(SimpleNamespace(_name_or_path=str(tmp_path), **{**_HF_CONFIG, "vision_n_layers": None}))
    assert vision.attention_groups[0].bidirectional_mm_blocks is True
    assert text.attention_groups[0].bidirectional_mm_blocks is False

    prefix = 10
    types, _perm = build_image_block(2, 2, prefix)
    lo, hi = prefix, prefix + len(types)
    image = MMItem(modality="image", hash=1, pad_value=1, offsets=[[lo, hi]], feature=torch.zeros(1))
    assert lo < 15 < hi  # a plain 15-token chunk budget ends inside the block

    def chunk_lens(keep_images_whole):
        try:
            get_global_ctx()
        except AssertionError:
            set_global_ctx(Context(page_size=1))
        pt = torch.zeros((5, 64), dtype=torch.int32)
        cm = CacheManager(num_pages=64, page_size=1, page_table=pt, type="radix")
        pm = PrefillManager(
            cm, TableManager(max_running_reqs=4, page_table=pt), DecodeManager(1),
            keep_images_whole=keep_images_whole,
        )
        pm.pending_list = [PendingReq(uid=7, input_ids=torch.arange(30, dtype=torch.int32),
                                      sampling_params=SamplingParams(max_tokens=4), mm_items=[image])]
        lens, cuts = [], []
        while pm.runnable:
            batch = pm.schedule_next_batch(15)
            assert batch is not None
            cm.allocate_paged(batch.reqs)
            cuts.extend(cut_image_spans(batch.reqs))
            for r in batch.reqs:
                lens.append(r.extend_len)
                r.complete_one()
        return lens, cuts

    lens, cuts = chunk_lens(vision.attention_groups[0].bidirectional_mm_blocks)
    assert lens == [lo, 15, 30 - lo - 15] and cuts == []  # first chunk stops at the block, the next holds it whole
    lens, cuts = chunk_lens(text.attention_groups[0].bidirectional_mm_blocks)
    assert lens == [15, 15] and cuts == [(lo, hi)]  # the block is split into two spans; the cut warning would fire
