"""Gemma 4 vision: shapes and host streaming on a tiny random tower, the unified release's embedder, parity with the reference on real checkpoints."""

from __future__ import annotations

import glob
import os

import pytest
import torch

from freetoken.models.gemma4.config import UnifiedVisionConfig, VisionConfig
from freetoken.models.gemma4.vision import Gemma4MultimodalEmbedder, Gemma4UnifiedVisionEmbedder, Gemma4VisionModel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CHECKPOINT = os.environ.get("FREETOKEN_GEMMA4_MODEL", "")
UNIFIED_CHECKPOINT = os.environ.get("FREETOKEN_GEMMA4_UNIFIED_MODEL", "")
needs_checkpoint = pytest.mark.skipif(not os.path.exists(os.path.join(CHECKPOINT, "config.json")), reason="FREETOKEN_GEMMA4_MODEL not set")
needs_unified_checkpoint = pytest.mark.skipif(not os.path.exists(os.path.join(UNIFIED_CHECKPOINT, "config.json")), reason="FREETOKEN_GEMMA4_UNIFIED_MODEL not set")


def _tiny_vc():
    return VisionConfig(
        hidden_size=32, num_layers=2, num_heads=2, num_kv_heads=2, head_dim=16, intermediate_size=64, patch_size=16,
        position_embedding_size=64, pooling_kernel_size=3, rms_norm_eps=1e-6, rope_theta=100.0, hidden_act="gelu_tanh",
        standardize=True, use_clipped_linears=False, text_hidden_size=48,
    )


def _build(vc):
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            tower, embedder = Gemma4VisionModel(vc), Gemma4MultimodalEmbedder(vc)
    finally:
        torch.set_default_dtype(torch.float32)
    for p in list(tower.state_dict().values()) + [embedder.embedding_projection.weight]:
        p.normal_(0, 0.02)
    return tower, embedder


def _grid_positions(width, height):
    ys, xs = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)


def test_tiny_tower_pools_nine_patches_per_soft_token():
    tower, embedder = _build(_tiny_vc())
    feature = torch.randn(1, 18, 768, dtype=torch.bfloat16, device="cuda")
    positions = _grid_positions(6, 3)[None].cuda()
    out = embedder.forward(tower.forward(feature, positions))
    assert out.shape == (2, 48) and out.dtype == torch.bfloat16


def test_host_streamed_weights_compute_the_same_output():
    tower, _ = _build(_tiny_vc())
    feature = torch.randn(1, 18, 768, dtype=torch.bfloat16, device="cuda")
    positions = _grid_positions(6, 3)[None].cuda()
    resident = tower.forward(feature, positions)
    keys = tower.state_dict().keys()
    tower.place_weights("host")
    assert tower.state_dict().keys() == keys
    assert tower.state_dict()["encoder.layers.0.self_attn.q_proj.weight"].device.type == "cpu"
    assert tower.state_dict()["patch_embedder.input_proj.weight"].device.type == "cuda"
    for _ in range(2):  # a second forward reuses the staging buffers behind the release events
        assert torch.equal(tower.forward(feature, positions), resident)
    tower.place_weights("gpu")
    assert tower.state_dict()["encoder.layers.0.self_attn.q_proj.weight"].device.type == "cuda"
    assert torch.equal(tower.forward(feature, positions), resident)


@pytest.mark.needs_weights
@needs_checkpoint
def test_vision_mode_marks_only_the_sliding_layers_bidirectional():
    from transformers import AutoConfig

    from freetoken.models.config import FullAttentionGroupConfig, SWAAttentionGroupConfig
    from freetoken.models.gemma4 import parse_config

    groups = {type(g): g for g in parse_config(AutoConfig.from_pretrained(CHECKPOINT)).attention_groups}
    assert groups[SWAAttentionGroupConfig].bidirectional_mm_blocks
    assert not hasattr(groups[FullAttentionGroupConfig], "bidirectional_mm_blocks")


@pytest.mark.needs_weights
@needs_checkpoint
def test_tower_matches_the_reference_on_a_checkpoint():
    """The bf16 tower stays within the reference's own bf16 noise around its fp32 result."""
    import json

    from PIL import Image, ImageDraw
    from safetensors import safe_open
    from transformers import AutoConfig, AutoImageProcessor
    from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder as HFEmbedder
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel as HFVision
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionRotaryEmbedding

    from freetoken.mm.config import MultimodalConfig
    from freetoken.mm.processors.gemma4 import Gemma4MMProcessor
    from freetoken.models.gemma4 import parse_config

    torch.set_float32_matmul_precision("highest")
    hf_config = AutoConfig.from_pretrained(CHECKPOINT)
    hf_config.vision_config._attn_implementation = "sdpa"
    index = json.load(open(os.path.join(CHECKPOINT, "model.safetensors.index.json")))["weight_map"]
    tower_weights, embed_weights = {}, {}
    for name, file in index.items():
        if name.startswith(("model.vision_tower.", "model.embed_vision.")):
            with safe_open(os.path.join(CHECKPOINT, file), "pt") as f:
                (tower_weights if "vision_tower" in name else embed_weights)[name.split(".", 2)[2]] = f.get_tensor(name)

    def reference(dtype):
        vision = HFVision(hf_config.vision_config).to("cuda", dtype).eval()
        vision.load_state_dict({k: v.to(dtype) for k, v in tower_weights.items()}, strict=True)
        # .to(dtype) also casts the rotary table, which from_pretrained keeps in fp32
        vision.encoder.rotary_emb.inv_freq = Gemma4VisionRotaryEmbedding.compute_default_rope_parameters(hf_config.vision_config)[0].cuda()
        embed = HFEmbedder(hf_config.vision_config, hf_config.text_config).to("cuda", dtype).eval()
        embed.load_state_dict({k: v.to(dtype) for k, v in embed_weights.items()}, strict=True)
        return lambda pixels, positions: embed(vision(pixel_values=pixels, pixel_position_ids=positions).last_hidden_state)

    config = parse_config(hf_config)
    tower, embedder = _build(config.vision_config)
    tower.load_state_dict({k.replace(".linear.", "."): v.cuda() for k, v in tower_weights.items()})
    embedder.load_state_dict({k: v.cuda() for k, v in embed_weights.items()})

    img = Image.new("RGB", (640, 400), (30, 144, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse((100, 60, 340, 300), fill=(255, 215, 0))
    draw.rectangle((40, 340, 600, 380), fill=(220, 20, 60))
    out = AutoImageProcessor.from_pretrained(CHECKPOINT)(images=img, return_tensors="pt")
    pixels, positions = out["pixel_values"].cuda(), out["image_position_ids"].cuda()
    with torch.inference_mode():
        ref32 = reference(torch.float32)(pixels, positions)
        ref16 = reference(torch.bfloat16)(pixels, positions).float()
        (item,) = Gemma4MMProcessor(hf_config, CHECKPOINT, MultimodalConfig()).process([img])
        ours = embedder.forward(tower.forward(item.feature.cuda()[None], item.position_ids.cuda().long()[None])).float()
    assert ours.shape == ref32.shape == (item.num_soft_tokens, config.hidden_size)
    cos = torch.nn.functional.cosine_similarity
    noise, error = (ref32 - ref16).norm(), (ref32 - ours).norm()
    print(f"gemma4 tower: ours vs fp32 mean cos {cos(ref32, ours, dim=-1).mean():.5f}, error/noise {error / noise:.3f}")
    assert cos(ref32, ours, dim=-1).mean() > 0.99
    assert error <= 1.2 * noise


# the 12B gemma4_unified release: a linear embedder in place of the tower


def _tiny_unified_vc():
    return UnifiedVisionConfig(hidden_size=32, patch_dim=48, posemb_size=16, layer_norm_eps=1e-5, rms_norm_eps=1e-6, text_hidden_size=40)


def _build_unified(vc):
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            embedder, proj = Gemma4UnifiedVisionEmbedder(vc), Gemma4MultimodalEmbedder(vc)
    finally:
        torch.set_default_dtype(torch.float32)
    for p in list(embedder.state_dict().values()) + [proj.embedding_projection.weight]:
        p.normal_(0, 0.02)
    return embedder, proj


def test_tiny_unified_embedder_maps_one_super_patch_to_one_soft_token():
    embedder, proj = _build_unified(_tiny_unified_vc())
    feature = torch.rand(1, 6, 48, device="cuda")
    positions = torch.tensor([[[x, y] for y in range(2) for x in range(3)]], device="cuda")
    out = proj.forward(embedder.forward(feature, positions))
    assert out.shape == (1, 6, 40) and out.dtype == torch.bfloat16


@pytest.mark.needs_weights
@needs_unified_checkpoint
def test_unified_embedder_matches_the_reference_on_a_checkpoint():
    """The bf16 embedder stays within the reference's own bf16 noise around its fp32 result."""
    from PIL import Image, ImageDraw
    from safetensors import safe_open
    from transformers import AutoConfig, AutoImageProcessor
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedVisionEmbedder as HFEmbedder

    from freetoken.mm.config import MultimodalConfig
    from freetoken.mm.processors.gemma4 import Gemma4UnifiedMMProcessor
    from freetoken.models.gemma4 import parse_config

    torch.set_float32_matmul_precision("highest")
    hf_config = AutoConfig.from_pretrained(UNIFIED_CHECKPOINT)
    weights = {}
    for file in glob.glob(os.path.join(UNIFIED_CHECKPOINT, "*.safetensors")):
        with safe_open(file, "pt") as f:
            for name in f.keys():
                if name.startswith(("model.vision_embedder.", "model.embed_vision.")):
                    weights[name[len("model."):]] = f.get_tensor(name)
    embedder_weights = {k[len("vision_embedder."):]: v for k, v in weights.items() if k.startswith("vision_embedder.")}
    projection = weights["embed_vision.embedding_projection.weight"]

    def reference(dtype):
        ref = HFEmbedder(hf_config.vision_config, hf_config.text_config).to("cuda", dtype).eval()
        ref.load_state_dict({**{k: v.to(dtype) for k, v in embedder_weights.items()}, "multimodal_embedder.embedding_projection.weight": projection.to(dtype)}, strict=True)
        def run(pixels, positions):
            out = ref(pixel_values=pixels, image_position_ids=positions)
            return out.pooler_output if hasattr(out, "pooler_output") else out  # a plain tensor on transformers 5.15

        return run

    config = parse_config(hf_config)
    embedder, proj = _build_unified(config.vision_config)
    embedder.load_state_dict({k: v.cuda() for k, v in embedder_weights.items()})
    proj.load_state_dict({"embedding_projection.weight": projection.cuda()})

    img = Image.new("RGB", (640, 400), (30, 144, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse((100, 60, 340, 300), fill=(255, 215, 0))
    draw.rectangle((40, 340, 600, 380), fill=(220, 20, 60))
    out = AutoImageProcessor.from_pretrained(UNIFIED_CHECKPOINT)(images=img, return_tensors="pt")
    pixels, positions = out["pixel_values"].cuda(), out["image_position_ids"].cuda()
    valid = (positions[0] != -1).all(dim=-1)
    with torch.inference_mode():
        ref32 = reference(torch.float32)(pixels, positions)[0][valid]
        ref16 = reference(torch.bfloat16)(pixels, positions)[0][valid].float()
        (item,) = Gemma4UnifiedMMProcessor(hf_config, UNIFIED_CHECKPOINT, MultimodalConfig()).process([img])
        ours = proj.forward(embedder.forward(item.feature.cuda()[None], item.position_ids.cuda().long()[None]))[0].float()
    assert ours.shape == ref32.shape == (item.num_soft_tokens, hf_config.text_config.hidden_size)
    cos = torch.nn.functional.cosine_similarity
    noise, error = (ref32 - ref16).norm(), (ref32 - ours).norm()
    print(f"gemma4 unified embedder: ours vs fp32 mean cos {cos(ref32, ours, dim=-1).mean():.5f}, error/noise {error / noise:.3f}")
    assert cos(ref32, ours, dim=-1).mean() > 0.99
    assert error <= 1.2 * noise
