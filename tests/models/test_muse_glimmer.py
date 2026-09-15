"""Muse-Glimmer-30B config parsing and serving-mode resolution.

Drives ``muse_glimmer.parse_config`` off a synthetic HF config shaped exactly like
meta-models/Muse-Glimmer-30B's (multimodal wrapper: text tower in ``text_config``,
weights under ``model.language_model.``) and checks every decision the plan pinned:
the [SWA x3, full] layer split with NoPE full layers, the folded qk scale, the two
norm eps values, the logit post-processing scalars, the compressed-tensors NVFP4
quant modes, and the pool/backend resolution.
"""

from __future__ import annotations

import pytest

from freetoken.attention.base import AttnType
from freetoken.models.muse_glimmer.config import parse_config


class _Cfg:
    """Attribute-access shim over a dict (what AutoConfig hands parse_config)."""

    def __init__(self, data: dict):
        for k, v in data.items():
            setattr(self, k, _Cfg(v) if isinstance(v, dict) and k in ("text_config", "vision_config") else v)


def _vision_section() -> dict:
    """A tiny vision_config in the checkpoint's shape (the real tower is 50 x 1536)."""
    return {
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "hidden_act": "gelu",
        "layer_types": ["window_attention"] * 3 + ["full_attention"],
        "patch_size": 2,
        "patch_temporal": 2,
        "merge_size": 2,
        "pos_emb_height": 4,
        "pos_emb_width": 4,
        "layer_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "max_position_embeddings": 16,
        "model_type": "muse_glimmer_vision",
    }


def _hf_config(num_layers: int = 52, quantized: bool = False, vision: bool = False) -> _Cfg:
    pattern = ["sliding_attention"] * 3 + ["full_attention"]
    thetas = [500000.0] * 3 + [0.0]
    reps = (num_layers + 3) // 4
    data = {
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "model_type": "muse_glimmer",
        "image_token_id": 200092,
        "text_config": {
            "hidden_size": 6656,
            "intermediate_size": 19968,
            "num_hidden_layers": num_layers,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "vocab_size": 202048,
            "max_position_embeddings": 131072,
            "hidden_activation": "silu",
            "rms_norm_eps": 1e-5,
            "post_norm_eps": 1e-8,
            "qk_scale_factor": 3.87,
            "final_logit_softcapping": 20.0,
            "output_multiplier": 0.19611613513818404,
            "sliding_window": 2048,
            "tie_word_embeddings": False,
            "layer_types": (pattern * reps)[:num_layers],
            "layer_rope_theta": (thetas * reps)[:num_layers],
            "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
            "model_type": "muse_glimmer_text",
        },
    }
    if vision:
        data |= {
            "vision_config": _vision_section(),
            "out_hidden_size": 16 * 4,
            "projector_hidden_size": 24,
            "projector_hidden_act": "gelu",
        }
    if quantized:
        data["quantization_config"] = {
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "ignore": ["lm_head"],
            "config_groups": {
                "group_0": {
                    "format": "nvfp4-pack-quantized",
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                        "strategy": "tensor_group",
                    },
                }
            },
        }
    return _Cfg(data)


def test_parse_config_full_model():
    cfg = parse_config(_hf_config())

    assert cfg.num_layers == 52
    assert (cfg.num_qo_heads, cfg.num_kv_heads, cfg.head_dim) == (32, 2, 128)
    assert cfg.hidden_size == 6656 and cfg.intermediate_size == 19968
    assert cfg.vocab_size == 202048 and not cfg.tie_word_embeddings
    assert cfg.hidden_act == "silu" and not cfg.is_moe
    assert cfg.rms_norm_eps == 1e-5 and cfg.post_norm_eps == 1e-8
    assert cfg.use_qk_norm
    # q = qk_norm(q) * 3.87 with standard 1/sqrt(d) attention: folded into sm_scale.
    assert cfg.attn_sm_scale == pytest.approx(3.87 * 128**-0.5)
    assert cfg.final_logit_softcapping == 20.0
    assert cfg.output_multiplier == pytest.approx(0.19611613513818404)
    assert cfg.embedding_scale is None  # NormedEmbedding, not Gemma's sqrt(hidden) scale
    assert cfg.vision_config is None  # no vision_config section: the engine asked for no tower

    groups = {g.name: g for g in cfg.attention_groups}
    assert groups["swa"].layer_ids == tuple(i for i in range(52) if (i + 1) % 4 != 0)
    assert groups["full"].layer_ids == tuple(i for i in range(52) if (i + 1) % 4 == 0)
    assert groups["swa"].sliding_window == 2048
    assert groups["swa"].rotary_config.base == 500000.0
    # NoPE full layers: base 0.0 is the skip-rope marker read by the attention module.
    assert groups["full"].rotary_config.base == 0.0
    assert cfg.rotary_config.base == 500000.0  # top-level config carries the real rope

    specs = {s.name: s for s in cfg.kv_cache_group_specs()}
    assert specs["swa"].attn_type == AttnType.SWA and specs["swa"].sliding_window == 2048
    assert specs["full"].attn_type == AttnType.FULL and specs["full"].sliding_window is None
    assert all((s.num_kv_heads, s.head_dim) == (2, 128) for s in specs.values())

    # BF16 checkpoint: nothing quantized.
    assert cfg.attn_quant == "none" and cfg.dense_quant == "none"
    assert cfg.lm_head_quant == "none"


def test_parse_vision_config():
    from freetoken.models.muse_glimmer.config import parse_vision_config

    vc = parse_config(_hf_config(vision=True)).vision_config
    assert (vc.hidden_size, vc.intermediate_size, vc.num_layers, vc.num_heads) == (16, 32, 4, 2)
    assert vc.layer_types == ("window_attention",) * 3 + ("full_attention",)
    assert (vc.patch_size, vc.temporal_patch_size, vc.merge_size, vc.pos_emb_side) == (2, 2, 2, 4)
    assert vc.patch_dim == 2 * 3 * 2 * 2 and vc.out_hidden_size == 64
    assert (vc.projector_hidden_size, vc.text_hidden_size, vc.text_rms_norm_eps) == (24, 6656, 1e-5)
    assert vc.layer_norm_eps == 1e-5 and vc.rope_theta == 10000.0

    hf = _hf_config(vision=True)
    hf.vision_config.hidden_act = "silu"
    with pytest.raises(NotImplementedError, match="only gelu"):
        parse_vision_config(hf)
    hf = _hf_config(vision=True)
    hf.out_hidden_size = 16  # the adapter must take the pixel-shuffled width
    with pytest.raises(AssertionError, match="pixel-shuffled"):
        parse_vision_config(hf)


def test_parse_config_nvfp4_checkpoint():
    cfg = parse_config(_hf_config(quantized=True))
    # compressed-tensors NVFP4 quantizes every text Linear (attention incl. the gate,
    # and the MLP); lm_head / embeddings / norms stay bf16 (the ignore list).
    assert cfg.attn_quant == "nvfp4" and cfg.dense_quant == "nvfp4"
    assert cfg.lm_head_quant == "none"


def test_non_nvfp4_4bit_scheme_is_a_clear_error():
    # A compressed-tensors MXFP4 checkpoint (group_size 32, real since LLM Compressor
    # 0.9) must fail with a clear "unsupported scheme" error instead of routing into
    # the NVFP4 loader and dying in a shape assert.
    hf = _hf_config(quantized=True)
    weights = hf.quantization_config["config_groups"]["group_0"]["weights"]
    weights["group_size"] = 32
    weights["strategy"] = "group"
    with pytest.raises(ValueError, match="unsupported compressed-tensors"):
        parse_config(hf)


def test_per_layer_theta_beats_shared_rope_theta():
    # layer_rope_theta is the source of truth: a hypothetical checkpoint giving the
    # full layers a real theta must not be forced to NoPE.
    hf = _hf_config(num_layers=8)
    hf.text_config.layer_rope_theta = [1e6] * 8
    cfg = parse_config(hf)
    groups = {g.name: g for g in cfg.attention_groups}
    assert groups["full"].rotary_config.base == 1e6
    assert groups["swa"].rotary_config.base == 1e6


def test_pool_family_and_backend_resolution():
    from freetoken.engine.engine import _required_attn_types, _resolve_auto_attention_backend
    from freetoken.kvcache import resolve_pool_class
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache

    cfg = parse_config(_hf_config())
    assert resolve_pool_class(cfg) is HybridSWAKVCache
    required = _required_attn_types(cfg)
    assert required == frozenset({AttnType.FULL, AttnType.SWA})
    # SWA restricts serving to the triton backend (the only one in the capability
    # matrix that consumes per-call sliding windows), same as gemma4.
    assert _resolve_auto_attention_backend(required) == "triton"


def test_registry_resolves_architecture():
    from freetoken.models.register import get_model_spec

    spec = get_model_spec("MuseGlimmerForConditionalGeneration")
    assert spec.module == "freetoken.models.muse_glimmer"
    assert spec.model_cls == "MuseGlimmerForConditionalGeneration"
    assert spec.mm_processor == "freetoken.mm.processors.muse_glimmer:MuseGlimmerMMProcessor"
    assert [(e.kind, e.config_key, e.modalities) for e in spec.encoders] == [("vision", "vision_config", ("image",))]


def test_aot_table_covers_the_checkpoints():
    from freetoken.kernel.aot_models import SUPPORTED_MODELS

    entry = next(m for m in SUPPORTED_MODELS if m.architecture == "MuseGlimmerForConditionalGeneration")
    assert entry.hidden_size == 6656
    assert entry.kv_groups == ((2, 128),)
    assert entry.expert_formats == ()  # dense
    assert "RedHatAI/Muse-Glimmer-30B-NVFP4" in entry.aliases


def test_weight_rename_and_fusion():
    import torch

    from freetoken.models.loader import ct_bf16_fuse
    from freetoken.models.muse_glimmer.weight import _FUSIONS, _rename, _vision_name, _vision_tensors

    # Text tower renamed, lm_head untouched; the image path lands under the wrapper's vision_tower.
    assert _rename("model.language_model.layers.0.self_attn.q_proj.weight") == (
        "model.layers.0.self_attn.q_proj.weight"
    )
    assert _rename("model.language_model.embed_tokens.weight") == "model.embed_tokens.weight"
    assert _rename("lm_head.weight") == "lm_head.weight"
    assert _vision_name("model.language_model.norm.weight") is None
    assert _vision_name("model.vision_tower.layers.0.attn.q_proj.weight") == "vision_tower.layers.0.attn.q_proj.weight"
    assert _vision_name("model.vision_adapter.fc1.weight") == "vision_tower.adapter.fc1.weight"
    assert _vision_name("model.vision_projection.weight") == "vision_tower.projection.weight"

    # vision q/k/v fuse into attn.qkv in that order, biases like weights; other tensors pass through
    vbuf: dict = {}
    assert _vision_tensors("vision_tower.layers.0.attn.q_proj.bias", torch.zeros(2), vbuf) == []
    assert _vision_tensors("vision_tower.layers.0.attn.v_proj.bias", torch.full((2,), 2.0), vbuf) == []
    ((key, bias),) = _vision_tensors("vision_tower.layers.0.attn.k_proj.bias", torch.ones(2), vbuf)
    assert key == "vision_tower.layers.0.attn.qkv.bias" and bias.tolist() == [0, 0, 1, 1, 2, 2] and not vbuf
    ((key, weight),) = _vision_tensors("vision_tower.layers.0.attn.proj.weight", torch.zeros(2, 2), vbuf)
    assert key == "vision_tower.layers.0.attn.proj.weight" and torch.equal(weight, torch.zeros(2, 2))

    # q/k/v + the attention gate fuse into qkvg_proj in declaration order.
    buf: dict = {}
    parts = {
        "q_proj": torch.full((4, 2), 0.0),
        "k_proj": torch.full((2, 2), 1.0),
        "v_proj": torch.full((2, 2), 2.0),
        "gate_proj": torch.full((4, 2), 3.0),
    }
    fused = None
    for name, tensor in parts.items():
        out = ct_bf16_fuse(f"model.layers.0.self_attn.{name}", tensor, buf, _FUSIONS)
        assert out is not None
        if out:
            fused = out[0]
    assert fused is not None and not buf
    key, tensor = fused
    assert key == "model.layers.0.self_attn.qkvg_proj.weight"
    assert tensor.shape == (12, 2)
    assert tensor[0, 0] == 0.0 and tensor[4, 0] == 1.0 and tensor[6, 0] == 2.0
    assert tensor[8, 0] == 3.0

    # The MLP's own gate_proj is a different fusion (gate|up), not the attention one.
    buf2: dict = {}
    assert ct_bf16_fuse("model.layers.0.mlp.gate_proj", torch.zeros(3, 2), buf2, _FUSIONS) == []
    (out,) = ct_bf16_fuse("model.layers.0.mlp.up_proj", torch.ones(3, 2), buf2, _FUSIONS)
    key2, tensor2 = out
    assert key2 == "model.layers.0.mlp.gate_up_proj.weight" and tensor2.shape == (6, 2)


def _write_shards(tmp_path, shards: dict[str, dict]):
    """Write synthetic safetensors shards + an index, real checkpoint naming."""
    import safetensors.torch

    weight_map = {}
    for shard, tensors in shards.items():
        safetensors.torch.save_file(tensors, str(tmp_path / shard))
        for name in tensors:
            weight_map[name] = shard
    (tmp_path / "model.safetensors.index.json").write_text(
        __import__("json").dumps({"weight_map": weight_map})
    )


def _vision_checkpoint_tensors(hf) -> dict:
    """The image path in the checkpoint's naming: separate q/k/v with biases, an nn.Embedding position table, bias-free adapter and projection."""
    import torch

    vc = hf.vision_config
    H, I, P = vc.hidden_size, vc.intermediate_size, hf.projector_hidden_size
    bf16 = torch.bfloat16
    tensors = {
        "model.vision_tower.patch_embedder.patch_embedding.weight": torch.randn(H, vc.patch_temporal * 3 * vc.patch_size**2, dtype=bf16),
        "model.vision_tower.patch_embedder.position_embedding_table.weight": torch.randn(vc.pos_emb_height * vc.pos_emb_width, H, dtype=bf16),
        "model.vision_tower.ln_pre.weight": torch.randn(H, dtype=bf16),
        "model.vision_tower.ln_pre.bias": torch.randn(H, dtype=bf16),
        "model.vision_tower.ln_post.weight": torch.randn(H, dtype=bf16),
        "model.vision_tower.ln_post.bias": torch.randn(H, dtype=bf16),
        "model.vision_adapter.fc1.weight": torch.randn(P, hf.out_hidden_size, dtype=bf16),
        "model.vision_adapter.fc2.weight": torch.randn(P, P, dtype=bf16),
        "model.vision_projection.weight": torch.randn(hf.text_config.hidden_size, P, dtype=bf16),
    }
    for i in range(vc.num_hidden_layers):
        lp = f"model.vision_tower.layers.{i}."
        for name in ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.proj"):
            tensors[lp + name + ".weight"] = torch.randn(H, H, dtype=bf16)
            tensors[lp + name + ".bias"] = torch.randn(H, dtype=bf16)
        tensors |= {
            lp + "mlp.fc1.weight": torch.randn(I, H, dtype=bf16),
            lp + "mlp.fc1.bias": torch.randn(I, dtype=bf16),
            lp + "mlp.fc2.weight": torch.randn(H, I, dtype=bf16),
            lp + "mlp.fc2.bias": torch.randn(H, dtype=bf16),
        }
        for name in ("norm1", "norm2"):
            tensors[lp + name + ".weight"] = torch.randn(H, dtype=bf16)
            tensors[lp + name + ".bias"] = torch.randn(H, dtype=bf16)
    return tensors


def _bf16_checkpoint_tensors(hf) -> dict:
    import torch

    text = hf.text_config
    H, I = text.hidden_size, text.intermediate_size
    q = text.num_attention_heads * text.head_dim
    kv = text.num_key_value_heads * text.head_dim
    p = "model.language_model."
    tensors = {
        p + "embed_tokens.weight": torch.randn(text.vocab_size, H, dtype=torch.bfloat16),
        p + "norm.weight": torch.randn(H, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(text.vocab_size, H, dtype=torch.bfloat16),
    } | _vision_checkpoint_tensors(hf)
    for i in range(text.num_hidden_layers):
        lp = f"{p}layers.{i}."
        tensors |= {
            lp + "self_attn.q_proj.weight": torch.randn(q, H, dtype=torch.bfloat16),
            lp + "self_attn.k_proj.weight": torch.randn(kv, H, dtype=torch.bfloat16),
            lp + "self_attn.v_proj.weight": torch.randn(kv, H, dtype=torch.bfloat16),
            lp + "self_attn.gate_proj.weight": torch.randn(q, H, dtype=torch.bfloat16),
            lp + "self_attn.o_proj.weight": torch.randn(H, q, dtype=torch.bfloat16),
            lp + "mlp.gate_proj.weight": torch.randn(I, H, dtype=torch.bfloat16),
            lp + "mlp.up_proj.weight": torch.randn(I, H, dtype=torch.bfloat16),
            lp + "mlp.down_proj.weight": torch.randn(H, I, dtype=torch.bfloat16),
            lp + "input_layernorm.weight": torch.randn(H, dtype=torch.bfloat16),
            lp + "post_attention_layernorm.weight": torch.randn(H, dtype=torch.bfloat16),
            lp + "pre_feedforward_layernorm.weight": torch.randn(H, dtype=torch.bfloat16),
            lp + "post_feedforward_layernorm.weight": torch.randn(H, dtype=torch.bfloat16),
        }
    return tensors


@pytest.mark.parametrize("include_vision", [False, True])
def test_iter_weights_bf16_matches_model_state_dict(tmp_path, monkeypatch, include_vision):
    """The BF16 loader must produce exactly the model's state-dict keys with the
    right shapes (rename + qkvg / gate_up fusion, norms raw); the image path only
    when an encoder is built, then with q/k/v fused into attn.qkv."""
    import torch

    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    from freetoken.models.muse_glimmer.model import MuseGlimmerForConditionalGeneration
    from freetoken.models.muse_glimmer.weight import iter_weights

    hf = _hf_config(num_layers=4, vision=True)
    tensors = _bf16_checkpoint_tensors(hf)
    _write_shards(tmp_path, {"model-00001-of-00001.safetensors": tensors})
    import freetoken.models.muse_glimmer.weight as w

    monkeypatch.setattr(w, "cached_load_hf_config", lambda _p: hf)

    loaded = dict(iter_weights(
        str(tmp_path), torch.device("cpu"), include_moe_experts=False, include_non_moe=True, include_vision=include_vision,
    ))
    if not include_vision:
        hf.vision_config = None  # what the engine's text-only model_config looks like
    model = MuseGlimmerForConditionalGeneration(parse_config(hf))
    expected = model.state_dict()
    assert set(loaded) == set(expected)
    assert any(k.startswith("vision_tower.") for k in expected) is include_vision
    for k in expected:
        assert loaded[k].shape == expected[k].shape, k
    if include_vision:
        v = "model.vision_tower.layers.1.attn."
        assert torch.equal(loaded["vision_tower.layers.1.attn.qkv.weight"], torch.cat([tensors[v + f"{n}_proj.weight"] for n in "qkv"]))
        assert torch.equal(loaded["vision_tower.layers.1.attn.qkv.bias"], torch.cat([tensors[v + f"{n}_proj.bias"] for n in "qkv"]))
        assert torch.equal(loaded["vision_tower.projection.weight"], tensors["model.vision_projection.weight"])
    # fusion order [q, k, v, gate] against the raw parts
    fused = loaded["model.layers.0.self_attn.qkvg_proj.weight"]
    p = "model.language_model.layers.0.self_attn."
    q_dim = tensors[p + "q_proj.weight"].shape[0]
    kv_dim = tensors[p + "k_proj.weight"].shape[0]
    assert torch.equal(fused[:q_dim], tensors[p + "q_proj.weight"])
    assert torch.equal(fused[q_dim : q_dim + kv_dim], tensors[p + "k_proj.weight"])
    assert torch.equal(fused[-q_dim:], tensors[p + "gate_proj.weight"])


def test_iter_weights_nvfp4_cross_shard_scales(tmp_path, monkeypatch):
    """compressed-tensors loader: native FP4 parts fused with per-part scales, the
    reciprocal global, and sibling scales resolved through the index even when they
    land in a different shard than their weight_packed (the real checkpoint splits
    layer 49's down_proj across the shard boundary); the bf16 image path rides along
    with its q/k/v fused across shards."""
    import torch

    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    from freetoken.models.muse_glimmer import MuseGlimmerVisionModel, parse_vision_config
    from freetoken.models.muse_glimmer.weight import iter_weights

    hf = _hf_config(num_layers=1, quantized=True, vision=True)
    text = hf.text_config
    text.hidden_size, text.intermediate_size = 64, 96
    text.num_attention_heads, text.num_key_value_heads, text.head_dim = 4, 2, 16
    text.vocab_size = 128
    H, I = 64, 96
    q, kv = 4 * 16, 2 * 16
    fp8 = torch.float8_e4m3fn

    def nvfp4(base, out_f, in_f, global_scale, shard_for_scales=None):
        t = {
            base + ".weight_packed": torch.randint(0, 256, (out_f, in_f // 2), dtype=torch.uint8),
            base + ".weight_scale": torch.randn(out_f, in_f // 16).abs().to(fp8),
            base + ".weight_global_scale": torch.tensor([global_scale]),
            base + ".input_global_scale": torch.tensor([1.0]),  # W4A4 scale: skipped
        }
        return t

    p = "model.language_model.layers.0."
    shard1 = {}
    for base, (o, i) in {
        p + "self_attn.q_proj": (q, H), p + "self_attn.k_proj": (kv, H),
        p + "self_attn.v_proj": (kv, H), p + "self_attn.gate_proj": (q, H),
        p + "self_attn.o_proj": (q, H)[::-1], p + "mlp.gate_proj": (I, H),
        p + "mlp.up_proj": (I, H),
    }.items():
        shard1 |= nvfp4(base, o, i, 2.0)
    # down_proj straddles the boundary: packed weight in shard 1, scales in shard 2.
    down = nvfp4(p + "mlp.down_proj", H, I, 4.0)
    shard1[p + "mlp.down_proj.weight_packed"] = down[p + "mlp.down_proj.weight_packed"]
    shard2 = {k: v for k, v in down.items() if not k.endswith(".weight_packed")}
    shard2 |= {
        "model.language_model.embed_tokens.weight": torch.randn(128, H, dtype=torch.bfloat16),
        "model.language_model.norm.weight": torch.randn(H, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(128, H, dtype=torch.bfloat16),
    }
    for name in (
        "input_layernorm", "post_attention_layernorm",
        "pre_feedforward_layernorm", "post_feedforward_layernorm",
    ):
        shard1[p + name + ".weight"] = torch.randn(H, dtype=torch.bfloat16)
    # the image path stays bf16; layer 0's q_proj lands in shard 1 and its k/v in shard 2
    vision = _vision_checkpoint_tensors(hf)
    q_keys = {k for k in vision if ".layers.0.attn.q_proj." in k}
    shard1 |= {k: vision[k] for k in q_keys}
    shard2 |= {k: v for k, v in vision.items() if k not in q_keys}
    _write_shards(tmp_path, {
        "model-00001-of-00002.safetensors": shard1,
        "model-00002-of-00002.safetensors": shard2,
    })
    import freetoken.models.muse_glimmer.weight as w

    monkeypatch.setattr(w, "cached_load_hf_config", lambda _p: hf)

    loaded = dict(
        iter_weights(str(tmp_path), torch.device("cpu"), include_moe_experts=False, include_non_moe=True)
    )
    qkvg = "model.layers.0.self_attn.qkvg_proj"
    assert loaded[qkvg + ".weight"].shape == (2 * q + 2 * kv, H // 2)
    assert loaded[qkvg + ".weight_scale"].shape == (2 * q + 2 * kv, H // 16)
    assert loaded[qkvg + ".weight_global"].shape == (2 * q + 2 * kv,)
    # global is the reciprocal of the stored quant-side scale
    assert loaded[qkvg + ".weight_global"][0].item() == pytest.approx(0.5)
    dp = "model.layers.0.mlp.down_proj"
    assert loaded[dp + ".weight"].shape == (H, I // 2)  # cross-shard scales resolved
    assert loaded[dp + ".weight_global"][0].item() == pytest.approx(0.25)
    # the quant-side activation global lands as the dequant-side input_scale the layer declares
    assert loaded[dp + ".input_scale"].item() == pytest.approx(1.0)
    assert loaded[qkvg + ".input_scale"].shape == ()
    assert not any(k.endswith(".input_global_scale") for k in loaded)
    expected_vision = {"vision_tower." + k for k in MuseGlimmerVisionModel(parse_vision_config(hf)).state_dict()}
    assert {k for k in loaded if k.startswith("vision_tower.")} == expected_vision
    v = "model.vision_tower.layers.0.attn."
    assert torch.equal(loaded["vision_tower.layers.0.attn.qkv.weight"], torch.cat([vision[v + f"{n}_proj.weight"] for n in "qkv"]))
    assert torch.equal(loaded["vision_tower.layers.0.attn.qkv.bias"], torch.cat([vision[v + f"{n}_proj.bias"] for n in "qkv"]))
    assert loaded["vision_tower.projection.weight"].dtype == torch.bfloat16
    text_only = dict(iter_weights(
        str(tmp_path), torch.device("cpu"), include_moe_experts=False, include_non_moe=True, include_vision=False,
    ))
    assert not any(k.startswith("vision_tower.") for k in text_only) and qkvg + ".weight" in text_only


def test_raw_config_shim_serves_unknown_model_type(tmp_path):
    """A checkpoint whose model_type is newer than the installed transformers must
    still parse: cached_load_hf_config falls back to the raw config.json."""
    import json as _json

    from freetoken.utils.hf import RawConfigShim, _load_hf_config, cached_load_hf_config

    raw = {
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "model_type": "some_model_type_from_the_future",
        "dtype": "bfloat16",
        "text_config": {"hidden_size": 64, "rope_parameters": {"rope_theta": 1e4}},
    }
    (tmp_path / "config.json").write_text(_json.dumps(raw))
    _load_hf_config.cache_clear()
    try:
        cfg = cached_load_hf_config(str(tmp_path))
    finally:
        _load_hf_config.cache_clear()
    assert isinstance(cfg, RawConfigShim)
    assert cfg.architectures == ["MuseGlimmerForConditionalGeneration"]
    assert cfg.text_config.hidden_size == 64
    # sub-config wrapped for attribute access; plain dicts stay dicts
    assert cfg.text_config.rope_parameters.get("rope_theta") == 1e4
    assert getattr(cfg, "missing_field", None) is None
    assert cfg.to_dict()["dtype"] == "bfloat16"
    # _name_or_path survives the underscore guard (DSV4's parse_config reads it
    # to locate inference/config.json) and the cached copy keeps it.
    assert cfg._name_or_path == str(tmp_path)


def test_model_state_dict_matches_loader_keys():
    import torch

    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    from freetoken.models.muse_glimmer.model import MuseGlimmerForCausalLM

    cfg = parse_config(_hf_config(num_layers=4))
    model = MuseGlimmerForCausalLM(cfg)
    keys = set(model.state_dict().keys())
    layer0 = {k for k in keys if k.startswith("model.layers.0.")}
    assert layer0 == {
        "model.layers.0.self_attn.qkvg_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.pre_feedforward_layernorm.weight",
        "model.layers.0.post_feedforward_layernorm.weight",
    }
    # Weightless norms (embed_norm, qk_norm) must not demand checkpoint tensors.
    assert "model.embed_norm.weight" not in keys
    assert not any("qk_norm" in k for k in keys)
    assert {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"} <= keys

    # Rope built only where a real theta exists: sliding layers yes, NoPE full no.
    layers = model.model.layers.op_list
    assert layers[0].self_attn.rotary is not None and layers[0].self_attn.is_swa
    assert layers[3].self_attn.rotary is None and not layers[3].self_attn.is_swa
    assert layers[0].self_attn.attn_spec.sliding_window == 2048
    assert layers[3].self_attn.attn_spec.sliding_window is None
    assert layers[0].self_attn.attn_spec.sm_scale == pytest.approx(3.87 * 128**-0.5)

    # NVFP4 build: every text Linear the config targets gets the W4A16 method.
    from freetoken.layers.quantization import NameMap, QuantConfig
    from freetoken.layers.quantization.linear import Nvfp4LinearMethod
    from freetoken.layers.quantization.linear import UnquantizedLinearMethod
    from freetoken.models.register import get_model_spec

    hf_q = _hf_config(num_layers=4, quantized=True)
    qcfg = parse_config(hf_q)
    spec = get_model_spec("MuseGlimmerForConditionalGeneration")
    object.__setattr__(qcfg, "quant", QuantConfig.from_hf(
        hf_q, name_map=NameMap(roots=spec.checkpoint_roots, packed=spec.packed_modules_mapping)
    ))
    qmodel = MuseGlimmerForCausalLM(qcfg)
    attn = qmodel.model.layers.op_list[0].self_attn
    mlp = qmodel.model.layers.op_list[0].mlp
    for proj in (attn.qkvg_proj, attn.o_proj, mlp.gate_up_proj, mlp.down_proj):
        assert type(proj.quant_method) is Nvfp4LinearMethod
    assert type(qmodel.lm_head.quant_method) is UnquantizedLinearMethod  # lm_head stays bf16
    del model, qmodel, torch
