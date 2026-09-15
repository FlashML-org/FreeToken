"""Muse-Glimmer vision tower: layout against a tiny reference tower and host streaming on random weights, parity with the reference on a real checkpoint."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from freetoken.models.muse_glimmer import MuseGlimmerVisionModel, VisionConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CHECKPOINT = os.environ.get("FREETOKEN_MUSE_MODEL", "")
needs_checkpoint = pytest.mark.skipif(not os.path.exists(os.path.join(CHECKPOINT, "config.json")), reason="FREETOKEN_MUSE_MODEL not set")


@pytest.fixture(autouse=True)
def _single_rank():
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _tiny_vc():
    # pos_emb_side 4 makes the window 4 patches wide, so a 6x6 grid has ragged 4/2 windows
    return VisionConfig(
        hidden_size=64, intermediate_size=128, num_layers=4, num_heads=4,
        layer_types=("window_attention", "window_attention", "window_attention", "full_attention"),
        patch_size=4, temporal_patch_size=2, merge_size=2, pos_emb_side=4, layer_norm_eps=1e-5, rope_theta=1e4,
        projector_hidden_size=48, text_hidden_size=40, text_rms_norm_eps=1e-5,
    )


def _build(vc):
    # bf16 like the engine: the weightless norm kernel takes no fp32
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            tower = MuseGlimmerVisionModel(vc)
    finally:
        torch.set_default_dtype(torch.float32)
    for p in tower.state_dict().values():
        p.normal_(0, 0.02)
    return tower


def _reference(vc, state, dtype):
    """The reference's get_image_features on our tower's weights (attn.qkv split back into q/k/v)."""
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerVisionConfig
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerRMSNorm, MuseGlimmerVisionAdapter
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerVisionModel as HFVision

    config = MuseGlimmerVisionConfig(
        hidden_size=vc.hidden_size, intermediate_size=vc.intermediate_size, num_hidden_layers=vc.num_layers,
        num_attention_heads=vc.num_heads, layer_types=list(vc.layer_types), patch_size=vc.patch_size,
        patch_temporal=vc.temporal_patch_size, merge_size=vc.merge_size, pos_emb_height=vc.pos_emb_side,
        pos_emb_width=vc.pos_emb_side, layer_norm_eps=vc.layer_norm_eps, max_position_embeddings=vc.pos_emb_side**2,
        rope_parameters={"rope_theta": vc.rope_theta, "rope_type": "default"},
    )
    config._attn_implementation = "sdpa"
    wrapper = SimpleNamespace(out_hidden_size=vc.out_hidden_size, projector_hidden_size=vc.projector_hidden_size, projector_hidden_act="gelu")
    # the default dtype sets the parameters; the rotary table is built fp32 regardless, as from_pretrained keeps it
    torch.set_default_dtype(dtype)
    try:
        tower = HFVision(config).to("cuda").eval()
        adapter = MuseGlimmerVisionAdapter(wrapper).to("cuda")
        projection = torch.nn.Linear(vc.projector_hidden_size, vc.text_hidden_size, bias=False).to("cuda")
    finally:
        torch.set_default_dtype(torch.float32)
    norm = MuseGlimmerRMSNorm(eps=vc.text_rms_norm_eps, with_scale=False)
    hf_state = {}
    for key, value in state.items():
        if ".attn.qkv." in key:
            for name, part in zip(("q_proj", "k_proj", "v_proj"), value.chunk(3, dim=0)):
                hf_state[key.replace("attn.qkv", f"attn.{name}")] = part.to(dtype)
        else:
            hf_state[key] = value.to(dtype)
    tower.load_state_dict({k: v for k, v in hf_state.items() if not k.startswith(("adapter.", "projection."))}, strict=True)
    adapter.load_state_dict({k[len("adapter.") :]: v for k, v in hf_state.items() if k.startswith("adapter.")}, strict=True)
    projection.load_state_dict({"weight": hf_state["projection.weight"]}, strict=True)
    return lambda pixels, grid: norm(projection(adapter(tower(pixels.to(dtype), grid_thw=grid).last_hidden_state)))


def test_tiny_tower_matches_the_reference_layout():
    """Rope layout, window permutation, pixel shuffle, adapter and norm against the reference on random weights."""
    vc = _tiny_vc()
    tower = _build(vc)
    reference = _reference(vc, tower.state_dict(), torch.bfloat16)
    pixels = torch.randn(36 + 16, vc.patch_dim, device="cuda", dtype=torch.bfloat16)
    grid = [[1, 6, 6], [1, 4, 4]]
    with torch.inference_mode():
        ref = reference(pixels, torch.tensor(grid, device="cuda")).float()
        ours = tower.forward(pixels, grid).float()
    assert ours.shape == ref.shape == (9 + 4, 40)
    cos = F.cosine_similarity(ref, ours, dim=-1)
    rel = ((ref - ours).norm() / ref.norm()).item()
    print(f"muse_glimmer tiny tower vs reference: min cos {cos.min():.5f}, relative error {rel:.4f}")
    assert cos.min() > 0.99 and rel < 5e-2
    keys = tower.state_dict()
    assert keys["layers.0.attn.qkv.bias"].shape == (192,) and keys["patch_embedder.position_embedding_table.weight"].shape == (16, 64)
    assert keys["adapter.fc1.weight"].shape == (48, 256) and keys["projection.weight"].shape == (40, 48)
    assert "adapter.fc1.bias" not in keys and "projection.bias" not in keys and "perception_emb_norm.weight" not in keys


def test_packed_images_match_the_single_image_calls():
    vc = _tiny_vc()
    tower = _build(vc)
    big = torch.randn(36, vc.patch_dim, device="cuda", dtype=torch.bfloat16)
    small = torch.randn(16, vc.patch_dim, device="cuda", dtype=torch.bfloat16)
    packed = tower.forward(torch.cat([big, small]), [[1, 6, 6], [1, 4, 4]]).float()
    singles = torch.cat([tower.forward(big, [[1, 6, 6]]), tower.forward(small, [[1, 4, 4]])]).float()
    # windows and positions are per image; only the GEMM tiling differs between the two calls
    assert ((packed - singles).norm() / singles.norm()).item() < 1e-2


def test_host_streamed_weights_compute_the_same_output():
    vc = _tiny_vc()
    tower = _build(vc)
    feature = torch.randn(64, vc.patch_dim, device="cuda", dtype=torch.bfloat16)
    resident = tower.forward(feature, [[1, 8, 8]])
    keys = tower.state_dict()
    tower.place_weights("host")
    assert tower.state_dict().keys() == keys.keys()
    assert tower.state_dict()["layers.0.attn.qkv.weight"].device.type == "cpu"
    assert tower.state_dict()["adapter.fc1.weight"].device.type == "cuda"
    for _ in range(2):  # a second forward reuses the staging buffers behind the release events
        assert torch.equal(tower.forward(feature, [[1, 8, 8]]), resident)
    tower.place_weights("gpu")
    assert tower.state_dict()["layers.0.attn.qkv.weight"].device.type == "cuda"
    assert torch.equal(tower.forward(feature, [[1, 8, 8]]), resident)


def _textured_image(width, height):
    """Gradient, shapes and mild noise: on flat colour fields the bf16 reference itself drifts far from its fp32 result."""
    from PIL import Image, ImageDraw

    xs = torch.linspace(0, 255, width).expand(height, width)
    ys = torch.linspace(0, 255, height)[:, None].expand(height, width)
    rgb = torch.stack([xs, ys, 127 + (ys - xs) / 2], dim=-1)
    rgb = rgb + torch.randn(height, width, 3, generator=torch.Generator().manual_seed(0)) * 12
    img = Image.fromarray(rgb.clamp(0, 255).to(torch.uint8).numpy())
    draw = ImageDraw.Draw(img)
    draw.ellipse((width * 0.2, height * 0.15, width * 0.7, height * 0.7), fill=(255, 215, 0))
    draw.rectangle((width * 0.1, height * 0.8, width * 0.9, height * 0.9), fill=(220, 20, 60))
    return img


@pytest.mark.needs_weights
@needs_checkpoint
def test_tower_matches_the_reference_on_a_checkpoint():
    """The bf16 tower stays within the reference's own bf16 noise around its fp32 get_image_features."""
    import json

    from safetensors import safe_open
    from transformers import AutoConfig, AutoImageProcessor

    from freetoken.mm.config import MultimodalConfig
    from freetoken.mm.processors.muse_glimmer import MuseGlimmerMMProcessor
    from freetoken.models.muse_glimmer import parse_config
    from freetoken.models.muse_glimmer.weight import _vision_name, _vision_tensors

    torch.set_float32_matmul_precision("highest")
    hf_config = AutoConfig.from_pretrained(CHECKPOINT)
    config = parse_config(hf_config)
    index = json.load(open(os.path.join(CHECKPOINT, "model.safetensors.index.json")))["weight_map"]
    state, buf = {}, {}
    for name, file in index.items():
        if _vision_name(name) is not None:
            with safe_open(os.path.join(CHECKPOINT, file), "pt") as f:
                for key, value in _vision_tensors(_vision_name(name)[len("vision_tower.") :], f.get_tensor(name).cuda(), buf):
                    state[key] = value
    tower = _build(config.vision_config)
    tower.load_state_dict(dict(state))

    # 896x616 resizes to a 64x44 patch grid: four windows, the bottom row of them 12 patches high
    img = _textured_image(896, 616)
    out = AutoImageProcessor.from_pretrained(CHECKPOINT)(images=img, return_tensors="pt")
    pixels, grid = out["pixel_values"].cuda(), out["image_grid_thw"].cuda()
    with torch.inference_mode():
        ref32 = _reference(config.vision_config, state, torch.float32)(pixels, grid)
        ref16 = _reference(config.vision_config, state, torch.bfloat16)(pixels, grid).float()
        (item,) = MuseGlimmerMMProcessor(hf_config, CHECKPOINT, MultimodalConfig()).process([img])
        ours = tower.forward(item.feature, [item.grid_thw]).float()
    assert item.grid_thw == [1, 44, 64]
    assert ours.shape == ref32.shape == (44 * 64 // 4, config.hidden_size)
    noise, error = (ref32 - ref16).norm(), (ref32 - ours).norm()
    print(f"muse_glimmer tower: ours vs fp32 mean cos {F.cosine_similarity(ref32, ours, dim=-1).mean():.5f}, error/noise {error / noise:.3f}")
    assert F.cosine_similarity(ref32, ours, dim=-1).mean() > 0.99
    assert error <= 1.2 * noise
