"""mimo_v2 weight loading against a synthetic checkpoint shaped like the release.

Real key names and dtypes: F32 fused qkv + F32 dense layer-0 MLP (yielded in source dtype;
the engine's single conversion to EngineConfig.dtype is the only cast -- DR-04), bf16
o_proj/embed/norms/router gate/lm_head, BF16 SWA-only sinks (cast to fp32), F32 selection
bias, and modelopt NVFP4 experts (packed u8 + fp8-e4m3 block-16 scales + F32 scalar
weight_scale_2) read through the offload-cache expert spec.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.models.mimo_v2.config import parse_config
from freetoken.models.mimo_v2.weight import iter_weights, nvfp4_expert_spec
from freetoken.models.nvfp4_banks import iter_nvfp4_expert_pieces
from freetoken.utils import cached_load_hf_config

H, V = 64, 128  # hidden, vocab
QH, KHF, KHS, HD, VHD = 4, 2, 4, 192, 128  # q heads, full/swa kv heads, head dims
I, MI, E = 128, 64, 2  # dense / expert intermediate, routed experts
FP8 = torch.float8_e4m3fn

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="session", autouse=True)
def _tp_info():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _bf16(*shape: int) -> torch.Tensor:
    return torch.randn(*shape).to(torch.bfloat16)


def _f32(*shape: int) -> torch.Tensor:
    return torch.randn(*shape)


# layers: 0 = full+dense, 1 = swa+MoE, 2 = full+MoE (pattern / moe_layer_freq in _config_json)
MOE_LAYERS = (1, 2)
SWA_LAYERS = (1,)
FULL_KV_ROWS = QH * HD + KHF * HD + KHF * VHD  # 768 + 384 + 256
SWA_KV_ROWS = QH * HD + KHS * HD + KHS * VHD  # 768 + 768 + 512


def _raw(nvfp4_experts: bool = True) -> dict[str, torch.Tensor]:
    raw: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": _bf16(V, H),
        "model.norm.weight": _bf16(H),
        "lm_head.weight": _bf16(V, H),
    }
    for layer in (0, 1, 2):
        a = f"model.layers.{layer}.self_attn"
        rows = SWA_KV_ROWS if layer in SWA_LAYERS else FULL_KV_ROWS
        # fused qkv is F32 on both layer types in this checkpoint
        raw[f"{a}.qkv_proj.weight"] = _f32(rows, H)
        raw[f"{a}.o_proj.weight"] = _bf16(H, QH * VHD)
        if layer in SWA_LAYERS:
            raw[f"{a}.attention_sink_bias"] = _bf16(QH)
        raw[f"model.layers.{layer}.input_layernorm.weight"] = _bf16(H)
        raw[f"model.layers.{layer}.post_attention_layernorm.weight"] = _bf16(H)
        m = f"model.layers.{layer}.mlp"
        if layer in MOE_LAYERS:
            raw[f"{m}.gate.weight"] = _bf16(E, H)
            raw[f"{m}.gate.e_score_correction_bias"] = _f32(E)
            for e in range(E):
                for proj, shape in (("gate_proj", (MI, H)), ("up_proj", (MI, H)), ("down_proj", (H, MI))):
                    base = f"{m}.experts.{e}.{proj}"
                    if not nvfp4_experts:
                        # unquantized release layout: plain bf16 per-expert weights
                        raw[f"{base}.weight"] = _bf16(*shape)
                        continue
                    raw[f"{base}.weight"] = torch.randint(0, 256, (shape[0], shape[1] // 2), dtype=torch.uint8)
                    raw[f"{base}.weight_scale"] = (torch.rand(shape[0], shape[1] // 16) + 0.5).to(FP8)
                    raw[f"{base}.weight_scale_2"] = (torch.rand(1) + 0.5).reshape(()).to(torch.float32)
        else:  # dense layer 0, F32 in the checkpoint
            raw[f"{m}.gate_proj.weight"] = _f32(I, H)
            raw[f"{m}.up_proj.weight"] = _f32(I, H)
            raw[f"{m}.down_proj.weight"] = _f32(H, I)
    # DROP-list noise: mtp / vision / audio / speech must never be yielded
    raw.update(
        {
            "model.mtp.layers.0.self_attn.qkv_proj.weight": _bf16(FULL_KV_ROWS, H),
            "model.mtp.layers.0.self_attn.attention_sink_bias": _bf16(QH),
            "visual.patch_embed.proj.weight": _bf16(H, H),
            "audio_encoder.input_local_transformer.norm.weight": _bf16(H),
            "speech_embeddings.3.weight": _bf16(8, H),
        }
    )
    return raw


def _quantization_config() -> dict:
    return {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "group_size": 16,
        "kv_cache_quant_algo": None,
        "exclude_modules": [],
        "quantized_layers": {
                    f"model.layers.{layer}.mlp.experts.{e}.{proj}": {
                        "quant_algo": "W4A16_NVFP4", "group_size": 16
                    }
            for layer in MOE_LAYERS
            for e in range(E)
            for proj in ("gate_proj", "up_proj", "down_proj")
        },
    }


def _config_json(quantized: bool = True, *, tie: bool = False) -> dict:
    return {
        "architectures": ["MiMoV2ForCausalLM"],
        "model_type": "mimo_v2",
        "hidden_size": H,
        "intermediate_size": I,
        "num_hidden_layers": 3,
        "num_attention_heads": QH,
        "num_key_value_heads": KHF,
        "swa_num_attention_heads": QH,
        "swa_num_key_value_heads": KHS,
        "head_dim": HD,
        "swa_head_dim": HD,
        "v_head_dim": VHD,
        "swa_v_head_dim": VHD,
        "vocab_size": V,
        "tie_word_embeddings": tie,
        "hidden_act": "silu",
        "layernorm_epsilon": 1e-6,
        "max_position_embeddings": 256,
        "attention_bias": False,
        "attention_value_scale": 0.707,
        "sliding_window": 128,
        "hybrid_layer_pattern": [0, 1, 0],
        "moe_layer_freq": [0, 1, 1],
        "rope_theta": 10000000.0,
        "swa_rope_theta": 10000.0,
        "partial_rotary_factor": 0.334,
        "rope_parameters": {"partial_rotary_factor": 0.334, "rope_theta": 10000000.0, "rope_type": "default", "type": "default"},
        "add_swa_attention_sink_bias": True,
        "add_full_attention_sink_bias": False,
        "n_routed_experts": E,
        "num_experts_per_tok": 2,
        "n_shared_experts": None,
        "moe_intermediate_size": MI,
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": None,
        "quantization_config": _quantization_config() if quantized else None,
    }


def _write(
    folder, raw: dict[str, torch.Tensor], quantized: bool = True, *, indexed: bool = True, tie: bool = False
) -> str:
    save_file({k: v.clone() for k, v in raw.items()}, str(folder / "model.safetensors"))
    if indexed:
        (folder / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": {k: "model.safetensors" for k in raw}})
        )
    (folder / "config.json").write_text(json.dumps(_config_json(quantized, tie=tie)))
    return str(folder)


def _load(folder: str) -> dict[str, torch.Tensor]:
    return {
        n: t.clone()
        for n, t in iter_weights(folder, torch.device("cpu"), include_moe_experts=False, include_non_moe=True)
    }


def _build_model(folder: str, strategy: str = "offload", *, device: torch.device | None = None,
                 dtype: torch.dtype = torch.bfloat16):
    """The engine model for ``folder`` (experts per ``strategy``); meta unless a real build is needed."""
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import _decode_target
    from freetoken.layers import rotary
    from freetoken.mm.config import ENCODER_KINDS, MultimodalConfig
    from freetoken.models import create_model
    from freetoken.utils.torch_utils import torch_dtype

    config = EngineConfig(
        model_path=folder,
        tp_info=try_get_tp_info(),
        dtype=dtype,
        moe_strategy=strategy,
        mm=MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS)),
    )
    object.__setattr__(config.model_config, "moe_strategy", strategy)
    object.__setattr__(config.model_config, "decode_target", _decode_target(config))
    saved = rotary._ROPE_DEVICE
    rotary.set_rope_device(torch.device("cpu"))  # get_rope refuses to build on meta
    rotary.get_rope.cache_clear()
    try:
        with torch.device(device or torch.device("meta")), torch_dtype(dtype):
            return create_model(config.model_config)
    finally:
        rotary.set_rope_device(saved)
        rotary.get_rope.cache_clear()


def _meta_state_dict(folder: str, strategy: str = "offload") -> dict[str, torch.Tensor]:
    """State dict of the model the engine builds for ``folder`` (experts per ``strategy``), on the meta device."""
    return _build_model(folder, strategy).state_dict()


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(7)
    raw = _raw()
    return _write(tmp_path_factory.mktemp("mimo_v2"), raw), raw


@pytest.fixture(scope="module")
def bf16_checkpoint(tmp_path_factory):
    """Same release layout, bf16 experts, no quantization_config (the tiny-validation dialect)."""
    torch.manual_seed(11)
    raw = _raw(nvfp4_experts=False)
    return _write(tmp_path_factory.mktemp("mimo_v2_bf16"), raw, quantized=False, indexed=False), raw


def test_emitted_tensors_are_the_model_state_dict(checkpoint):
    """The reader fills exactly the buffers the engine builds, with their shapes; the engine's
    single conversion (DR-04) lands every emitted tensor on the buffer dtype."""
    folder, _raw_w = checkpoint
    loaded, state = _load(folder), _meta_state_dict(folder)
    assert set(loaded) == set(state)
    for key, tensor in loaded.items():
        assert tensor.shape == state[key].shape, key
    materialized = _engine_state(folder, torch.bfloat16, "offload")
    assert set(materialized) == set(state)
    for key, tensor in materialized.items():
        assert tensor.dtype is state[key].dtype, key


def test_f32_linears_pass_through_and_sinks_land_fp32(checkpoint):
    """DR-04 re-anchor: the F32 qkv / layer-0 MLP are emitted verbatim in source dtype (no
    pre-rounding); the engine's single conversion equals the direct cast to the model dtype,
    and sinks still land fp32."""
    folder, raw = checkpoint
    loaded = _load(folder)
    for layer in (0, 1, 2):
        key = f"model.layers.{layer}.self_attn.qkv_proj.weight"
        assert loaded[key].dtype is torch.float32
        assert torch.equal(loaded[key], raw[key])  # value-identical source pass-through
        assert torch.equal(
            loaded[f"model.layers.{layer}.self_attn.o_proj.weight"],
            raw[f"model.layers.{layer}.self_attn.o_proj.weight"],
        )
    assert loaded["model.layers.0.self_attn.qkv_proj.weight"].shape == (FULL_KV_ROWS, H)
    assert loaded["model.layers.1.self_attn.qkv_proj.weight"].shape == (SWA_KV_ROWS, H)
    # layer-0 dense MLP: F32 in the checkpoint, emitted verbatim
    for proj in ("gate_proj", "up_proj", "down_proj"):
        key = f"model.layers.0.mlp.{proj}.weight"
        assert loaded[key].dtype is torch.float32
        assert torch.equal(loaded[key], raw[key])
    # engine path: materialized state == the direct single cast to the target dtype
    state = _engine_state(folder, torch.bfloat16, "offload")
    for layer in (0, 1, 2):
        key = f"model.layers.{layer}.self_attn.qkv_proj.weight"
        assert state[key].dtype is torch.bfloat16
        assert torch.equal(state[key], raw[key].to(torch.bfloat16))
    for proj in ("gate_proj", "up_proj", "down_proj"):
        key = f"model.layers.0.mlp.{proj}.weight"
        assert state[key].dtype is torch.bfloat16
        assert torch.equal(state[key], raw[key].to(torch.bfloat16))
    sinks = state["model.layers.1.self_attn.sinks"]
    assert sinks.dtype is torch.float32
    assert torch.equal(sinks, raw["model.layers.1.self_attn.attention_sink_bias"].to(torch.float32))


def test_sinks_are_swa_only_and_selection_bias_is_fp32(checkpoint):
    folder, raw = checkpoint
    loaded = _load(folder)
    assert "model.layers.1.self_attn.sinks" in loaded
    assert "model.layers.0.self_attn.sinks" not in loaded  # add_full_attention_sink_bias=False
    assert "model.layers.2.self_attn.sinks" not in loaded
    sinks = loaded["model.layers.1.self_attn.sinks"]
    assert sinks.dtype is torch.float32 and sinks.shape == (QH,)
    assert torch.equal(sinks, raw["model.layers.1.self_attn.attention_sink_bias"].to(torch.float32))
    bias = loaded["model.layers.1.mlp.e_score_correction_bias"]
    assert bias.dtype is torch.float32
    assert torch.equal(bias, raw["model.layers.1.mlp.gate.e_score_correction_bias"])
    assert not any(k.endswith("attention_sink_bias") for k in loaded)


@pytest.mark.parametrize(
    ("sink_change", "message"),
    [("missing", "missing="), ("unexpected", "unexpected=")],
)
def test_sink_key_placement_is_validated_before_loading(sink_change, message, tmp_path):
    raw = _raw()
    if sink_change == "missing":
        raw.pop("model.layers.1.self_attn.attention_sink_bias")
    else:
        raw["model.layers.0.self_attn.attention_sink_bias"] = _bf16(QH)
    folder = _write(tmp_path, raw)
    with pytest.raises(ValueError, match=message):
        _load(folder)


def test_sink_weight_shape_is_validated_during_loading(tmp_path):
    raw = _raw()
    raw["model.layers.1.self_attn.attention_sink_bias"] = _bf16(QH - 1)
    folder = _write(tmp_path, raw)
    with pytest.raises(ValueError, match=r"attention_sink_bias has shape \(3,\), expected \(4,\)"):
        _load(folder)


def test_dropped_checkpoint_prefixes_never_leak(checkpoint):
    folder, _raw_w = checkpoint
    loaded = _load(folder)
    assert not any(
        k.startswith(("model.mtp.", "visual.", "audio_encoder.", "speech_embeddings.")) for k in loaded
    )
    assert not any(".mlp.experts." in k for k in loaded)


def test_nvfp4_expert_pieces_read_the_modelopt_dialect(checkpoint):
    """Bank == checkpoint layer - 1 (first_k_dense_replace); modelopt weight_scale_2 lands verbatim (no reciprocal)."""
    folder, raw = checkpoint
    config = parse_config(cached_load_hf_config(folder))
    assert config.expert_quant == "nvfp4" and config.num_moe_layers == 2
    spec = nvfp4_expert_spec(folder, config)
    pieces = {(layer, expert): piece for layer, expert, _n, piece in iter_nvfp4_expert_pieces(folder, config, spec)}
    assert len(pieces) == 2 * E  # 2 MoE layers x 2 experts, one piece per expert
    assert {(0, 0), (1, E - 1)} <= set(pieces)  # checkpoint layers 1 and 2 -> banks 0 and 1

    base = "model.layers.1.mlp.experts.0"  # bank layer 0 == checkpoint layer 1
    piece = pieces[(0, 0)]
    assert torch.equal(piece["gate"][0], raw[f"{base}.gate_proj.weight"])
    assert torch.equal(piece["down_scale"][0], raw[f"{base}.down_proj.weight_scale"])
    assert piece["gate_global"].dtype is torch.float16
    assert torch.equal(
        piece["gate_global"].reshape(-1), raw[f"{base}.gate_proj.weight_scale_2"].reshape(-1).to(torch.float16)
    )


def test_nvfp4_weight_map_rejects_a_non_modelopt_scale_suffix(checkpoint):
    from freetoken.models.mimo_v2.weight import _validate_weight_map

    config = parse_config(cached_load_hf_config(checkpoint[0]))
    weight_map = {
        "model.layers.1.self_attn.attention_sink_bias": "model.safetensors",
        "model.layers.1.mlp.experts.0.gate_proj.weight_global_scale": "model.safetensors",
    }
    with pytest.raises(ValueError, match="expects ModelOpt weight_scale_2"):
        _validate_weight_map(weight_map, config)


def test_public_nvfp4_piece_route_rejects_an_alternate_scale_suffix(tmp_path):
    raw = _raw()
    raw["model.layers.1.mlp.experts.0.gate_proj.weight_global_scale"] = torch.ones(())
    folder = _write(tmp_path, raw)
    config = parse_config(cached_load_hf_config(folder))
    spec = nvfp4_expert_spec(folder, config)
    with pytest.raises(ValueError, match="expects ModelOpt weight_scale_2"):
        iter_nvfp4_expert_pieces(folder, config, spec)


def test_nvfp4_layout_rejects_dense_expert_loading(checkpoint):
    """Resident and offload NVFP4 experts both come from pieces, never dense tensors."""
    folder, _raw_w = checkpoint
    assert parse_config(cached_load_hf_config(folder)).expert_quant == "nvfp4"
    with pytest.raises(AssertionError, match="NVFP4"):
        dict(iter_weights(folder, torch.device("cpu"), include_moe_experts=True, include_non_moe=True))
    with pytest.raises(AssertionError, match="NVFP4"):
        dict(iter_weights(folder, torch.device("cpu"), include_moe_experts=True, include_non_moe=False))


def test_nvfp4_resident_bank_loading_preserves_modelopt_rows(checkpoint):
    """Packing/attachment only: CPU assertions do not claim GPU dequantization parity."""
    folder, raw = checkpoint
    model = _build_model(folder, "fused", device=torch.device("cpu"))
    _engine_state(folder, torch.bfloat16, "fused", model=model)
    for layer in MOE_LAYERS:
        experts = model.model.layers.op_list[layer].mlp.experts
        assert experts.layer_id == layer - 1
        view = experts.quant_method.resident_view(experts)
        assert set(view.tensors) == {
            "gate_up", "gate_up_scale", "gate_up_global", "down", "down_scale", "down_global",
        }
        assert view.tensors["gate_up"].shape == (E, 2 * MI, H // 2)
        assert view.tensors["down"].shape == (E, H, MI // 2)
        assert view.tensors["gate_up"].dtype == torch.uint8
        assert view.tensors["gate_up_scale"].dtype == FP8
        assert view.tensors["gate_up_global"].dtype == torch.float16
        for e in range(E):
            prefix = f"model.layers.{layer}.mlp.experts.{e}"
            for proj, role, rows in (("gate", "gate_up", slice(0, MI)), ("up", "gate_up", slice(MI, 2 * MI)),
                                     ("down", "down", slice(0, H))):
                key = f"{prefix}.{proj}_proj"
                assert torch.equal(view.tensors[role][e, rows], raw[key + ".weight"])
                assert torch.equal(view.tensors[role + "_scale"][e, rows].view(torch.uint8), raw[key + ".weight_scale"].view(torch.uint8))
                expected_global = raw[key + ".weight_scale_2"].to(torch.float16)
                assert torch.equal(view.tensors[role + "_global"][e, rows], expected_global.expand(view.tensors[role + "_global"][e, rows].shape))


# ---------------------------------------------------------------- unquantized (resident experts) layout


def test_bf16_config_reports_no_expert_quant(bf16_checkpoint):
    folder, _raw_w = bf16_checkpoint
    from pathlib import Path

    assert not (Path(folder) / "model.safetensors.index.json").exists()
    config = parse_config(cached_load_hf_config(folder))
    assert config.expert_quant == "none" and config.num_moe_layers == 2


def test_bf16_single_safetensors_without_index_loads(bf16_checkpoint):
    folder, _raw_w = bf16_checkpoint
    loaded = _load(folder)
    assert "model.embed_tokens.weight" in loaded
    assert not any(".mlp.experts." in name for name in loaded)


def test_tp_greater_than_one_is_rejected_before_weight_discovery(monkeypatch):
    from types import SimpleNamespace

    import freetoken.models.mimo_v2.weight as mimo_weight

    monkeypatch.setattr(mimo_weight, "get_tp_info", lambda: SimpleNamespace(size=2))
    monkeypatch.setattr(
        mimo_weight, "cached_load_hf_config", lambda _path: pytest.fail("config must not load")
    )
    with pytest.raises(ValueError, match="tensor parallel size 1 only; got TP=2"):
        next(
            mimo_weight.iter_weights(
                "unused", torch.device("cpu"), include_moe_experts=False, include_non_moe=True
            )
        )


def test_bf16_experts_yield_stacked_resident_names(bf16_checkpoint):
    """include_moe_experts=True on the bf16 dialect yields EXACTLY the resident experts.gate_up_proj/.down_proj."""
    folder, raw = bf16_checkpoint
    experts = dict(
        iter_weights(folder, torch.device("cpu"), include_moe_experts=True, include_non_moe=False)
    )
    assert set(experts) == {
        f"model.layers.{layer}.mlp.experts.{proj}" for layer in MOE_LAYERS for proj in ("gate_up_proj", "down_proj")
    }
    for layer in MOE_LAYERS:
        gate_up = experts[f"model.layers.{layer}.mlp.experts.gate_up_proj"]
        down = experts[f"model.layers.{layer}.mlp.experts.down_proj"]
        assert gate_up.shape == (E, 2 * MI, H) and gate_up.dtype is torch.bfloat16
        assert down.shape == (E, H, MI) and down.dtype is torch.bfloat16
        for e in range(E):
            m = f"model.layers.{layer}.mlp.experts.{e}"
            assert torch.equal(gate_up[e, :MI], raw[f"{m}.gate_proj.weight"])  # gate rows first
            assert torch.equal(gate_up[e, MI:], raw[f"{m}.up_proj.weight"])
            assert torch.equal(down[e], raw[f"{m}.down_proj.weight"])


def test_bf16_resident_load_fills_dense_state_and_expert_banks(bf16_checkpoint):
    """The production load sequence fills dense buffers and attaches every sparse bank."""
    folder, raw = bf16_checkpoint
    loaded = _load(folder)
    state = _meta_state_dict(folder, strategy="fused")
    assert set(loaded) == set(state)
    for key, tensor in loaded.items():
        assert tensor.shape == state[key].shape, key
    model = _build_model(folder, "fused", device=torch.device("cpu"))
    materialized = _engine_state(folder, torch.bfloat16, "fused", model=model)
    assert set(materialized) == set(state)
    for key, tensor in materialized.items():
        assert tensor.dtype is state[key].dtype, key
    assert not any(".experts." in key for key in materialized)
    for layer in MOE_LAYERS:
        experts = model.model.layers.op_list[layer].mlp.experts
        assert experts.layer_id == layer - 1
        view = experts.quant_method.resident_view(experts)
        assert set(view.tensors) == {"gate_up", "down"} and view.n == E
        assert view.tensors["gate_up"].shape == (E, 2 * MI, H)
        assert view.tensors["down"].shape == (E, H, MI)
        for e in range(E):
            prefix = f"model.layers.{layer}.mlp.experts.{e}"
            assert torch.equal(view.tensors["gate_up"][e, :MI], raw[f"{prefix}.gate_proj.weight"])
            assert torch.equal(view.tensors["gate_up"][e, MI:], raw[f"{prefix}.up_proj.weight"])
            assert torch.equal(view.tensors["down"][e], raw[f"{prefix}.down_proj.weight"])
        assert all(t.dtype == torch.bfloat16 and t.device.type == "cpu" for t in view.tensors.values())


# ---------------------------------------------------------------- device placement (GPU sinks crash)


def test_iter_weights_yields_tensors_on_the_requested_device(checkpoint):
    """The gpt_oss device mechanism mirrored (gpt_oss weight.py:218 opens safetensors ON the
    engine's device and BaseOP.load_state_dict SETATTRS the yielded tensors over the model's
    placeholder attrs, layers/base.py:46): every yielded tensor's .device must be exactly the
    requested one. CPU CI asks for cpu; the engine asks for CUDA, which is what lets
    kernel/triton/attention.py:810 (`assert sinks.is_cuda`) hold on GPU."""
    folder, _raw_w = checkpoint
    device = torch.device("cpu")
    for name, tensor in iter_weights(
        folder, device, include_moe_experts=False, include_non_moe=True
    ):
        assert tensor.device == device, name


def test_loader_casts_never_re_home_tensors_off_the_reader_device(checkpoint, monkeypatch):
    """The bf16->fp32 sinks cast (weight.py:189, like gpt_oss's clone at weight.py:112) must not
    drop tensors to CPU: a CPU tensor here becomes the CUDA-graph-captured pointer, since
    load_state_dict replaces the attr with the yielded tensor verbatim (base.py:46). Serve the
    shards from ``meta`` -- the same device-preservation check cuda needs, runnable on CPU CI."""
    import freetoken.models.mimo_v2.weight as mw

    real = mw._ShardReader

    class _MetaHome(real):  # stand-in for a reader mmapping straight onto the engine device
        def __init__(self, folder, weight_map, device):
            super().__init__(folder, weight_map, torch.device("cpu"))  # safetensors opens real devices only

        def get(self, name):
            return super().get(name).to("meta")

    monkeypatch.setattr(mw, "_ShardReader", _MetaHome)
    loaded = dict(
        iter_weights(
            checkpoint[0], torch.device("meta"), include_moe_experts=False, include_non_moe=True
        )
    )
    assert any(k.endswith(".sinks") for k in loaded)
    for name, tensor in loaded.items():
        assert tensor.device == torch.device("meta"), name


def test_sinks_land_on_the_layer_attribute_before_the_backend_reads_them(checkpoint, monkeypatch):
    """The GPU crash (triton attention.py:810 `assert sinks.is_cuda` at warmup): the engine loads
    sinks by SETATTRing the yielded tensor over the __init__ placeholder (base.py:46), so
    MiMoV2Attention.forward must hand the backend the CURRENT self.sinks -- building the
    AttentionSpec at construction pinned the placeholder object and shipped a CPU pointer to the
    kernel (gpt_oss builds it per call, gpt_oss/attention.py:88-92). Meta-built layer + a real
    replacement tensor reproduces the stale-object bug on CPU CI."""
    import contextlib
    import types

    import torch.cuda.nvtx

    import freetoken.models.mimo_v2.attention as attn_mod
    from freetoken.layers import rotary
    from freetoken.models.mimo_v2.attention import MiMoV2Attention
    from freetoken.utils.torch_utils import torch_dtype

    # the CPU test image has no NVTX; @nvtx_annotate resolves torch.cuda.nvtx.range at call time
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda _name: contextlib.nullcontext())

    folder, _raw_w = checkpoint
    cfg = parse_config(cached_load_hf_config(folder))

    saved = rotary._ROPE_DEVICE
    rotary.set_rope_device(torch.device("cpu"))
    rotary.get_rope.cache_clear()
    try:
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            swa = MiMoV2Attention(cfg, layer_id=1, prefix="model.layers.1.self_attn")
            full = MiMoV2Attention(cfg, layer_id=0, prefix="model.layers.0.self_attn")
    finally:
        rotary.set_rope_device(saved)
        rotary.get_rope.cache_clear()

    assert swa.sinks is not None and full.sinks is None  # SWA-only declaration placeholder

    loaded_sinks = torch.linspace(-1.0, 1.0, swa.num_qo_heads, dtype=torch.float32)
    captured: dict = {}

    class _Backend:
        def forward(self, q, k, v, layer_id, batch, attn_spec=None):
            captured[layer_id] = attn_spec
            return torch.zeros(q.shape, dtype=q.dtype)

    T = 2
    ctx = types.SimpleNamespace(
        batch=types.SimpleNamespace(positions=torch.zeros(T, dtype=torch.int32)),
        attn_backend=_Backend(),
    )
    monkeypatch.setattr(attn_mod, "get_global_ctx", lambda: ctx)

    for attn, loaded in ((swa, loaded_sinks), (full, None)):
        # stub the sub-OPs so load_state_dict (and forward) touch only the sinks-attr path
        attn.qkv_proj = types.SimpleNamespace(
            output_sizes=[attn.qo_attn_dim, attn.kv_attn_dim, attn.v_attn_dim],
            forward=lambda _x: torch.zeros(T, attn.qo_attn_dim + attn.kv_attn_dim + attn.v_attn_dim,
                                           dtype=torch.bfloat16),
        )
        attn.o_proj = types.SimpleNamespace(forward=lambda o: o)
        attn.rotary = types.SimpleNamespace(forward=lambda _pos, q, k: (q, k))
        if loaded is not None:
            attn.load_state_dict({"sinks": loaded})
            assert attn.sinks is loaded  # base.py:46 replaces the attr with the LOADED tensor

        out = attn.forward(torch.zeros(T, cfg.hidden_size, dtype=torch.bfloat16))
        assert out.shape == (T, attn.num_qo_heads * attn.v_head_dim)

        spec = captured[attn.layer_id]
        assert spec.sinks is loaded  # the BACKEND sees the load-time tensor, not a stale placeholder
        assert spec.sm_scale == cfg.attn_sm_scale
    assert captured[1].sliding_window is not None
    assert captured[0].sliding_window is None


def test_loaded_selection_bias_is_read_live_by_routing(bf16_checkpoint, monkeypatch):
    """moe.py:35 audit: the fp32 selection-bias placeholder has the SAME load-time setattr
    lifecycle as sinks, and _route (moe.py:62) reads the live attr -- assert the replacement
    tensor really drives selection (no init-time capture to go stale like the sinks spec did)."""
    import types

    from freetoken.layers import rotary
    from freetoken.models.mimo_v2.moe import MiMoV2SparseBlock
    from freetoken.utils.torch_utils import torch_dtype

    folder, _raw_w = bf16_checkpoint
    cfg = parse_config(cached_load_hf_config(folder))

    saved = rotary._ROPE_DEVICE
    rotary.set_rope_device(torch.device("cpu"))
    rotary.get_rope.cache_clear()
    try:
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            block = MiMoV2SparseBlock(cfg, layer_id=1, prefix="model.layers.1.mlp")
    finally:
        rotary.set_rope_device(saved)
        rotary.get_rope.cache_clear()

    torch.manual_seed(5)
    gate_w = torch.randn(cfg.num_experts, cfg.hidden_size)
    bias = torch.zeros(cfg.num_experts, dtype=torch.float32)
    bias[0] = 10.0  # strongly favour expert 0 once loaded
    block.gate = types.SimpleNamespace(weight=gate_w)
    block.experts = types.SimpleNamespace()  # not BaseOP anymore: load visits only the bias attr
    block.load_state_dict({"e_score_correction_bias": bias})
    assert block.e_score_correction_bias is bias

    hidden = torch.randn(3, cfg.hidden_size)
    _weights, ids = MiMoV2SparseBlock._route(block, hidden)
    scores = torch.sigmoid(hidden.float() @ gate_w.float().T) + bias
    for row in range(hidden.shape[0]):
        assert set(ids[row].tolist()) == set(torch.topk(scores[row], block.top_k).indices.tolist())


def test_bf16_layout_dense_only_load_still_skips_experts(bf16_checkpoint):
    """include_moe_experts=False on the bf16 dialect keeps the offload-cache expert-only reader happy: dense-only yields no experts."""
    folder, _raw_w = bf16_checkpoint
    loaded = {
        n: t.clone()
        for n, t in iter_weights(
            folder, torch.device("cpu"), include_moe_experts=False, include_non_moe=True
        )
    }
    assert not any(".mlp.experts." in k for k in loaded)
    assert set(loaded) == set(_meta_state_dict(folder, strategy="fused")) - {
        f"model.layers.{layer}.mlp.experts.{proj}"
        for layer in MOE_LAYERS
        for proj in ("gate_up_proj", "down_proj")
    }


# ---------------------------------------------------------------- tied lm head selection (DR-06)


def test_tied_checkpoint_without_lm_head_loads_the_embedding_as_head(tmp_path, monkeypatch):
    """tie_word_embeddings=true checkpoints omit the redundant head tensor: the dense reader
    must select the embedding for the head, and that tensor must drive the actual logits."""
    import contextlib
    import types

    import torch.cuda.nvtx
    import torch.nn.functional as F
    from freetoken.layers import embedding as embedding_mod

    raw = _raw(nvfp4_experts=False)
    raw.pop("lm_head.weight")
    folder = _write(tmp_path, raw, quantized=False, indexed=False, tie=True)
    loaded = {
        n: t.clone()
        for n, t in iter_weights(
            folder, torch.device("cpu"), include_moe_experts=False, include_non_moe=True
        )
    }
    # state tree: exactly the engine's buffers (the tied head's state_dict is empty)
    assert set(loaded) == set(_meta_state_dict(folder, strategy="fused"))
    assert torch.equal(loaded["model.embed_tokens.weight"], raw["model.embed_tokens.weight"])

    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda _name: contextlib.nullcontext())
    # the head reads the engine batch ctx to slice prefill rows; a decode batch passes through
    monkeypatch.setattr(
        embedding_mod,
        "get_global_ctx",
        lambda: types.SimpleNamespace(batch=types.SimpleNamespace(is_prefill=False, size=1)),
    )

    model = _build_model(folder, strategy="fused", device=torch.device("cpu"))
    # engine path (DR-04 re-anchor): materialize to the buffer dtype before the strict mount,
    # exactly as engine.py does -- the reader no longer pre-casts to the model dtype
    state = _engine_state(folder, torch.bfloat16, "fused", model=model)
    embed = state["model.embed_tokens.weight"]
    assert model.lm_head.tied_embedding is model.model.embed_tokens
    assert model.model.embed_tokens.weight is embed
    torch.manual_seed(3)
    x = torch.randn(2, H, dtype=torch.bfloat16)
    assert torch.equal(model.lm_head.forward(x), F.linear(x, raw["model.embed_tokens.weight"]))


def test_tied_nvfp4_offload_checkpoint_without_lm_head_loads_dense(tmp_path):
    """Same tied selection on the NVFP4 dialect: the offload dense reader (experts are never
    include_moe_experts there) must not require lm_head.weight either."""
    raw = _raw()
    raw.pop("lm_head.weight")
    folder = _write(tmp_path, raw, tie=True)
    assert parse_config(cached_load_hf_config(folder)).expert_quant == "nvfp4"
    loaded = _load(folder)  # KeyError('lm_head.weight') before the fix
    assert set(loaded) == set(_meta_state_dict(folder, strategy="offload"))
    assert torch.equal(loaded["model.embed_tokens.weight"], raw["model.embed_tokens.weight"])


def test_tied_checkpoint_with_redundant_lm_head_ignores_it(tmp_path):
    """A tied checkpoint that still serializes lm_head.weight loads with the embedding as the
    selected head; the redundant tensor is neither yielded nor loadable into the head."""
    raw = _raw(nvfp4_experts=False)
    # unmistakable non-embed payload: if it ever reached the head, logits would be all zeros
    raw["lm_head.weight"] = torch.zeros(V, H, dtype=torch.bfloat16)
    folder = _write(tmp_path, raw, quantized=False, indexed=False, tie=True)
    loaded = dict(
        iter_weights(folder, torch.device("cpu"), include_moe_experts=False, include_non_moe=True)
    )
    assert "lm_head.weight" not in loaded
    assert set(loaded) == set(_meta_state_dict(folder, strategy="fused"))
    assert torch.equal(loaded["model.embed_tokens.weight"], raw["model.embed_tokens.weight"])
    assert loaded["model.embed_tokens.weight"].count_nonzero() > 0
    model = _build_model(folder, strategy="fused", device=torch.device("cpu"))
    # strict load via the engine path (DR-04 re-anchor): the redundant tensor must not be an
    # extra key, and every loaded tensor is materialized to the buffer dtype first
    _engine_state(folder, torch.bfloat16, "fused", model=model)


def test_untied_checkpoint_missing_lm_head_still_fails(tmp_path):
    """Guard the other direction: an untied config declares lm_head.weight (head load_state_dict
    pops a tensor, BaseOP raises on the missing key), so the reader must keep requiring it."""
    raw = _raw()
    raw.pop("lm_head.weight")
    offload_dir = tmp_path / "offload"
    offload_dir.mkdir()
    with pytest.raises(KeyError, match="lm_head.weight"):
        _load(_write(offload_dir, raw))
    resident_raw = _raw(nvfp4_experts=False)
    resident_raw.pop("lm_head.weight")
    resident_dir = tmp_path / "resident"
    resident_dir.mkdir()
    with pytest.raises(KeyError, match="lm_head.weight"):
        dict(
            iter_weights(
                _write(resident_dir, resident_raw, quantized=False, indexed=False),
                torch.device("cpu"),
                include_moe_experts=True,
                include_non_moe=True,
            )
        )


# ---------------------------------------------------------------- DR-04: no BF16 pre-rounding (folded from tests/models/test_dr04_bf16_pre_rounding.py)
#
# The reader used to cast the F32 source projections (fused qkv, dense-layer MLP) and the
# unquantized stacked experts to BF16 BEFORE the engine converted to the requested parameter
# dtype (engine.py ``_materialize_loaded_weight_state_dict`` -> ``weight.to(device,
# dtype=expected.dtype)``; the model is built under ``torch_dtype(config.dtype)``, so the
# target is EngineConfig.dtype -- admitted as float16/bfloat16/float32). Admitted executions
# therefore double-rounded through BF16: FP16 saw an F32 65504.0 become inf (65504 -> BF16
# 65536 -> FP16 inf) and FP32 saw an F32 1.001953125 become 1.0. These tests pin the
# single-rounding contract on both the NVFP4 offload dialect and the unquantized
# resident-expert dialect, driven through the engine's own load path. The NVFP4 quantized
# experts themselves are not asserted here: their packed u8 codes / fp8 scales / F32
# ``weight_scale_2`` never traverse the removed cast sites (models/nvfp4_banks.py reads them
# verbatim, with its own independent fp16 global-scale cast).

# sentinel source values: 65504 is FP16's max finite; 1.001953125 == 1 + 2**-9 exists in
# F32 only (BF16 rounds it to 1.0).
_DR04_F16_SENTINEL = 65504.0
_DR04_F32_SENTINEL = 1.001953125
_DR04_ROW_F16, _DR04_COL_F16 = 32, 17
_DR04_ROW_F32, _DR04_COL_F32 = 50, 11

# the cast-site tensors: F32 in the release checkpoint (qkv + dense-layer MLP) and F32
# fixtures here for the unquantized stacked experts
_DR04_QKV_KEY = "model.layers.0.self_attn.qkv_proj.weight"
_DR04_DENSE_KEYS = tuple(f"model.layers.0.mlp.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj"))
_DR04_EXPERT_KEY = "model.layers.1.mlp.experts.0.gate_proj.weight"


def _raw_dr04(nvfp4: bool) -> dict[str, torch.Tensor]:
    """The release layout with F32 payloads in the cast-site tensors."""
    raw = _raw(nvfp4_experts=nvfp4)
    keys = (_DR04_QKV_KEY, *_DR04_DENSE_KEYS)
    if not nvfp4:
        # the unquantized dialect's per-expert projections: make them F32 so the
        # stacked-expert site is exercised on F32 sources too
        raw[_DR04_EXPERT_KEY] = torch.randn(MI, H)
        keys = (*keys, _DR04_EXPERT_KEY)
    for key in keys:
        raw[key][_DR04_ROW_F16, _DR04_COL_F16] = _DR04_F16_SENTINEL
        raw[key][_DR04_ROW_F32, _DR04_COL_F32] = _DR04_F32_SENTINEL
    return raw


@pytest.fixture(scope="module")
def dr04_nvfp4_checkpoint(tmp_path_factory):
    """Release NVFP4 dialect: the qkv + dense-MLP sites apply (experts are packed pieces)."""
    torch.manual_seed(13)
    raw = _raw_dr04(nvfp4=True)
    return _write(tmp_path_factory.mktemp("dr04_nvfp4"), raw), raw


@pytest.fixture(scope="module")
def dr04_bf16_experts_checkpoint(tmp_path_factory):
    """Tiny-validation dialect: same layout, quantization_config dropped (resident experts)."""
    torch.manual_seed(19)
    raw = _raw_dr04(nvfp4=False)
    folder = _write(tmp_path_factory.mktemp("dr04_bf16"), raw, quantized=False, indexed=False)
    return folder, raw


def _engine_state(folder: str, dtype: torch.dtype, strategy: str, *, model=None) -> dict[str, torch.Tensor]:
    """The engine's exact load sequence on CPU (engine.py:318 + :540): reader tensors run
    through ``_materialize_loaded_weight_state_dict`` (the dtype=expected.dtype conversion)
    and are strict-mounted onto the real-dtype model."""
    from freetoken.engine.engine import _materialize_loaded_weight_state_dict
    from freetoken.engine.engine import Engine
    from freetoken.layers import iter_moe_layers
    from freetoken.models.weight import load_weight
    from types import SimpleNamespace

    if model is None:
        model = _build_model(folder, strategy, device=torch.device("cpu"), dtype=dtype)
    state = _materialize_loaded_weight_state_dict(
        model.state_dict(),
        load_weight(folder, torch.device("cpu"), include_moe_experts=False, include_vision=False),
        device=torch.device("cpu"),
    )
    model.load_state_dict(dict(state))  # BaseOP.load_state_dict pops from its argument
    if strategy == "fused":
        config = SimpleNamespace(
            model_path=folder, model_config=parse_config(cached_load_hf_config(folder)),
            use_dummy_weight=False, expert_load="serial",
        )
        Engine._load_resident_experts(
            SimpleNamespace(device=torch.device("cpu"), dtype=dtype), config, list(iter_moe_layers(model))
        )
    return state


# ---------------------------------------------------------------- folded repro (fp16)


@pytest.mark.parametrize(
    ("fixture", "strategy"),
    [("dr04_nvfp4_checkpoint", "offload"), ("dr04_bf16_experts_checkpoint", "fused")],
)
def test_dr04_fp16_fused_qkv_converts_directly_from_f32(fixture, strategy, request):
    """65504.0 F32 must reach FP16 execution as 65504.0 (the removed BF16 stopover made it inf)."""
    folder, raw = request.getfixturevalue(fixture)
    state = _engine_state(folder, torch.float16, strategy)
    qkv = state[_DR04_QKV_KEY]
    assert qkv.dtype is torch.float16
    assert torch.isfinite(qkv[_DR04_ROW_F16, _DR04_COL_F16]), "admitted fp16 execution read inf from a finite F32 source"
    assert qkv[_DR04_ROW_F16, _DR04_COL_F16].item() == _DR04_F16_SENTINEL
    # full-tensor fidelity: engine value == direct F32->FP16 conversion, element-exact
    assert torch.equal(qkv, raw[_DR04_QKV_KEY].to(torch.float16))
    assert tuple(qkv.shape) == (FULL_KV_ROWS, H)


@pytest.mark.parametrize(
    ("fixture", "strategy"),
    [("dr04_nvfp4_checkpoint", "offload"), ("dr04_bf16_experts_checkpoint", "fused")],
)
def test_dr04_fp16_dense_mlp_converts_directly_from_f32(fixture, strategy, request):
    folder, raw = request.getfixturevalue(fixture)
    state = _engine_state(folder, torch.float16, strategy)
    for key in _DR04_DENSE_KEYS:
        got = state[key]
        assert got.dtype is torch.float16
        assert torch.isfinite(got[_DR04_ROW_F16, _DR04_COL_F16]), f"{key}: fp16 read inf from a finite F32 source"
        assert got[_DR04_ROW_F16, _DR04_COL_F16].item() == _DR04_F16_SENTINEL
        assert torch.equal(got, raw[key].to(torch.float16))


def test_dr04_fp16_unquantized_expert_converts_directly_from_f32(dr04_bf16_experts_checkpoint):
    """The stacked-expert site hits unquantized experts: an F32 gate projection with 65504.0
    must survive fp16 execution (gate rows [0, MI) of expert 0's stack)."""
    folder, raw = dr04_bf16_experts_checkpoint
    model = _build_model(folder, "fused", device=torch.device("cpu"), dtype=torch.float16)
    _engine_state(folder, torch.float16, "fused", model=model)
    stack = model.model.layers.op_list[1].mlp.experts._expert_banks["gate_up"]
    assert stack.dtype is torch.float16
    gate = stack[0, :MI]  # expert 0's merged gate piece
    assert torch.isfinite(gate[_DR04_ROW_F16, _DR04_COL_F16]), "fp16 execution read inf from a finite F32 expert source"
    assert gate[_DR04_ROW_F16, _DR04_COL_F16].tolist() == _DR04_F16_SENTINEL
    assert torch.equal(gate, raw[_DR04_EXPERT_KEY].to(torch.float16))
    for layer in MOE_LAYERS:  # every stacked expert lands for the fused strategy
        assert set(model.model.layers.op_list[layer].mlp.experts._expert_banks) == {"gate_up", "down"}


# ---------------------------------------------------------------- folded repro (fp32)


@pytest.mark.parametrize(
    ("fixture", "strategy"),
    [("dr04_nvfp4_checkpoint", "offload"), ("dr04_bf16_experts_checkpoint", "fused")],
)
def test_dr04_fp32_fused_qkv_keeps_full_f32_precision(fixture, strategy, request):
    """1.001953125 F32 must reach FP32 execution unchanged (the removed BF16 stopover rounded it to 1.0)."""
    folder, raw = request.getfixturevalue(fixture)
    state = _engine_state(folder, torch.float32, strategy)
    qkv = state[_DR04_QKV_KEY]
    assert qkv.dtype is torch.float32
    assert qkv[_DR04_ROW_F32, _DR04_COL_F32].item() == _DR04_F32_SENTINEL
    assert torch.equal(qkv, raw[_DR04_QKV_KEY])  # F32 source == FP32 parameter, element-exact


@pytest.mark.parametrize(
    ("fixture", "strategy"),
    [("dr04_nvfp4_checkpoint", "offload"), ("dr04_bf16_experts_checkpoint", "fused")],
)
def test_dr04_fp32_dense_mlp_keeps_full_f32_precision(fixture, strategy, request):
    folder, raw = request.getfixturevalue(fixture)
    state = _engine_state(folder, torch.float32, strategy)
    for key in _DR04_DENSE_KEYS:
        got = state[key]
        assert got.dtype is torch.float32
        assert got[_DR04_ROW_F32, _DR04_COL_F32].item() == _DR04_F32_SENTINEL
        assert torch.equal(got, raw[key])


def test_dr04_fp32_unquantized_expert_keeps_full_f32_precision(dr04_bf16_experts_checkpoint):
    folder, raw = dr04_bf16_experts_checkpoint
    model = _build_model(folder, "fused", device=torch.device("cpu"), dtype=torch.float32)
    _engine_state(folder, torch.float32, "fused", model=model)
    stack = model.model.layers.op_list[1].mlp.experts._expert_banks["gate_up"]
    assert stack.dtype is torch.float32
    gate = stack[0, :MI]
    assert gate[_DR04_ROW_F32, _DR04_COL_F32].tolist() == _DR04_F32_SENTINEL
    assert torch.equal(gate, raw[_DR04_EXPERT_KEY])
