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

from freetoken.models.deepseek_v4.config import DSV4VisionConfig, parse_vision_config
from freetoken.models.deepseek_v4.image_processor import load_image
from freetoken.models.deepseek_v4.vision import (
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
