"""qwen3_5_moe weight loading against synthetic checkpoints shaped like the released ones.

Tiny tensors, real key names, dtypes and quantization_config blocks. Each layout is read by
``iter_weights`` and compared with the state dict of the model the engine builds from the same
config; the dense pass has to fill exactly those buffers whatever the checkpoint quantized.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.layers.quantization import QuantKind, set_quant_config
from freetoken.models.nvfp4_banks import iter_nvfp4_expert_pieces
from freetoken.models.qwen3_5_moe.config import parse_config
from freetoken.models.qwen3_5_moe.weight import iter_expert_pieces, iter_weights, nvfp4_expert_spec
from freetoken.models.register import checkpoint_quant_config, get_model_spec
from freetoken.utils import cached_load_hf_config

H, V = 128, 256  # hidden_size (every block-fp8 projection needs in/out multiples of 128), vocab
KH, VH, HD = 2, 4, 64  # GDN key / value heads, head dim: qkv rows 512, z rows 256, b|a rows 4 each
QH, KVH, AHD = 4, 2, 64  # attention q / kv heads, head dim: q rows 512 (gated), k / v rows 128
I, MI, E = 128, 128, 4  # shared / dense MLP width, routed expert width, routed experts
BLOCK = 128
LM = "model.language_model"
FP8 = torch.float8_e4m3fn

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="session", autouse=True)
def _tp_info():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _bf16(*shape: int) -> torch.Tensor:
    return torch.randn(*shape).to(torch.bfloat16)


# --------------------------------------------------------------------------- checkpoints


def _dense_bf16(moe: bool) -> dict[str, torch.Tensor]:
    """Every non-expert tensor in bf16: layer 0 = GDN, layer 1 = attention, plus mtp / visual noise."""
    raw = {f"{LM}.embed_tokens.weight": _bf16(V, H), f"{LM}.norm.weight": _bf16(H), "lm_head.weight": _bf16(V, H)}
    for layer in (0, 1):
        pre = f"{LM}.layers.{layer}"
        raw.update({f"{pre}.input_layernorm.weight": _bf16(H), f"{pre}.post_attention_layernorm.weight": _bf16(H)})
        mlp = f"{pre}.mlp.shared_expert" if moe else f"{pre}.mlp"
        raw.update({f"{mlp}.gate_proj.weight": _bf16(I, H), f"{mlp}.up_proj.weight": _bf16(I, H), f"{mlp}.down_proj.weight": _bf16(H, I)})
        if moe:
            raw.update({f"{pre}.mlp.gate.weight": _bf16(E, H), f"{pre}.mlp.shared_expert_gate.weight": _bf16(1, H)})
    gdn = f"{LM}.layers.0.linear_attn"
    raw.update({
        f"{gdn}.in_proj_qkv.weight": _bf16(2 * KH * HD + VH * HD, H), f"{gdn}.in_proj_z.weight": _bf16(VH * HD, H),
        f"{gdn}.in_proj_b.weight": _bf16(VH, H), f"{gdn}.in_proj_a.weight": _bf16(VH, H),
        f"{gdn}.conv1d.weight": _bf16(2 * KH * HD + VH * HD, 1, 4), f"{gdn}.A_log": torch.randn(VH),
        f"{gdn}.dt_bias": torch.randn(VH), f"{gdn}.norm.weight": _bf16(HD), f"{gdn}.out_proj.weight": _bf16(H, VH * HD),
    })
    attn = f"{LM}.layers.1.self_attn"
    raw.update({
        f"{attn}.q_proj.weight": _bf16(2 * QH * AHD, H), f"{attn}.k_proj.weight": _bf16(KVH * AHD, H),
        f"{attn}.v_proj.weight": _bf16(KVH * AHD, H), f"{attn}.o_proj.weight": _bf16(H, QH * AHD),
        f"{attn}.q_norm.weight": _bf16(AHD), f"{attn}.k_norm.weight": _bf16(AHD),
    })
    raw.update({
        "mtp.layers.0.self_attn.q_proj.weight": _bf16(2 * QH * AHD, H),
        "model.visual.blocks.0.attn.qkv.weight": _bf16(3 * H, H),
    })
    return raw


def _nvfp4(weight: torch.Tensor, *, ct: bool) -> dict[str, torch.Tensor]:
    """Packed NVFP4 tensors of one module under the ModelOpt or the llm-compressor names."""
    rows, cols = weight.shape
    packed = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8)
    scale = (torch.rand(rows, cols // 16) + 0.5).to(FP8)
    glob = torch.rand(1) + 0.5
    if ct:
        return {"weight_packed": packed, "weight_scale": scale, "weight_global_scale": glob, "input_global_scale": torch.rand(1)}
    return {"weight": packed, "weight_scale": scale, "weight_scale_2": glob.reshape(()), "input_scale": torch.rand(())}


def _fp8_tensor(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    """ModelOpt FP8: one fp32 scale for the whole weight plus the calibrated activation scale."""
    return {"weight": weight.to(FP8), "weight_scale": torch.rand(()) + 0.5, "input_scale": torch.rand(()) + 0.5}


def _fp8_tensor_ct(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    """llm-compressor ``strategy: tensor`` with static activations: bf16 scalar weight and input scales."""
    return {"weight": weight.to(FP8), "weight_scale": (torch.rand(1) + 0.5).to(torch.bfloat16), "input_scale": (torch.rand(1) + 0.5).to(torch.bfloat16)}


def _fp8_channel(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    """llm-compressor ``strategy: channel``: one bf16 scale per output row."""
    return {"weight": weight.to(FP8), "weight_scale": (torch.rand(weight.shape[0], 1) + 0.5).to(torch.bfloat16)}


def _fp8_block(weight: torch.Tensor, *, ct: bool) -> dict[str, torch.Tensor]:
    """128x128 block-fp8: ``weight_scale_inv`` bf16 (HF fp8) or ``weight_scale`` fp32 (llm-compressor)."""
    scale = torch.rand(weight.shape[0] // BLOCK, weight.shape[1] // BLOCK) + 0.5
    return {"weight": weight.to(FP8), "weight_scale": scale} if ct else {"weight": weight.to(FP8), "weight_scale_inv": scale.to(torch.bfloat16)}


def _quantize(raw: dict[str, torch.Tensor], modules: list[str], quantize) -> None:
    for module in modules:
        weight = raw.pop(f"{module}.weight")
        raw.update({f"{module}.{suffix}": t for suffix, t in quantize(weight).items()})


def _experts(raw: dict[str, torch.Tensor], quantize=None) -> None:
    """Routed experts: stacked bf16 per layer, or per-expert tensors under ``quantize``."""
    for layer in (0, 1):
        pre = f"{LM}.layers.{layer}.mlp.experts"
        if quantize is None:
            raw[f"{pre}.gate_up_proj"] = _bf16(E, 2 * MI, H)
            raw[f"{pre}.down_proj"] = _bf16(E, H, MI)
            continue
        for expert in range(E):
            for proj, shape in (("gate_proj", (MI, H)), ("up_proj", (MI, H)), ("down_proj", (H, MI))):
                raw.update({f"{pre}.{expert}.{proj}.{suffix}": t for suffix, t in quantize(_bf16(*shape)).items()})


GDN_QKVZ_OUT = [f"{LM}.layers.0.linear_attn.{p}" for p in ("in_proj_qkv", "in_proj_z", "out_proj")]
GDN_BA = [f"{LM}.layers.0.linear_attn.{p}" for p in ("in_proj_b", "in_proj_a")]
ATTN = [f"{LM}.layers.1.self_attn.{p}_proj" for p in "qkvo"]
SHARED = [f"{LM}.layers.{l}.mlp.shared_expert.{p}_proj" for l in (0, 1) for p in ("gate", "up", "down")]
DENSE_MLP = [f"{LM}.layers.{l}.mlp.{p}_proj" for l in (0, 1) for p in ("gate", "up", "down")]
ROUTERS = [f"{LM}.layers.{l}.mlp.{p}" for l in (0, 1) for p in ("gate", "shared_expert_gate")]

NVFP4_GROUP = {
    "weights": {"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16, "symmetric": True},
    "input_activations": {"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16, "dynamic": "local"},
    "format": "nvfp4-pack-quantized",
}
FP8_CHANNEL_GROUP = {
    "weights": {"num_bits": 8, "type": "float", "strategy": "channel", "group_size": None, "symmetric": True},
    "input_activations": {"num_bits": 8, "type": "float", "strategy": "token", "dynamic": True},
    "format": "float-quantized",
}
FP8_TENSOR_STATIC_GROUP = {
    "weights": {"num_bits": 8, "type": "float", "strategy": "tensor", "group_size": None, "symmetric": True},
    "input_activations": {"num_bits": 8, "type": "float", "strategy": "tensor", "dynamic": False},
    "format": "float-quantized",
}
FP8_BLOCK_GROUP = {
    "weights": {"num_bits": 8, "type": "float", "strategy": "block", "block_structure": [128, 128], "symmetric": True},
    "input_activations": {"num_bits": 8, "type": "float", "strategy": "token", "dynamic": True},
    "format": "float-quantized",
}


def _ct(config_groups: dict, ignore: list[str], fmt: str) -> dict:
    return {"quant_method": "compressed-tensors", "format": fmt, "config_groups": config_groups, "ignore": ignore,
            "quantization_status": "compressed"}


# Qwen/Qwen3.6-35B-A3B-FP8: 128x128 block-fp8 everywhere but the listed modules
QWEN_FP8 = {
    "quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3", "weight_block_size": [128, 128],
    "modules_to_not_convert": ["lm_head", "model.embed_tokens", *GDN_BA, *ROUTERS,
                               *[f"{LM}.layers.{l}.{n}" for l in (0, 1) for n in ("input_layernorm", "post_attention_layernorm")]],
}
# nvidia/Qwen3.6-35B-A3B-NVFP4: per-tensor FP8 attention / GDN, NVFP4 shared expert, experts and lm_head
MODELOPT_MIXED = {
    "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "ignore": ["mtp*"],
    "quantized_layers": {
        **{m: {"quant_algo": "FP8"} for m in GDN_QKVZ_OUT + ATTN},
        **{m: {"quant_algo": "W4A16_NVFP4", "group_size": 16} for m in SHARED + ["lm_head"]},
        **{f"{LM}.layers.{l}.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16} for l in (0, 1)},
    },
}
# sakamakismile/Qwen3.6-27B-NVFP4: every Linear of the dense model, GDN in_proj included
CT_NVFP4_DENSE = _ct({"group_0": {**NVFP4_GROUP, "targets": ["Linear"]}}, ["lm_head"], "nvfp4-pack-quantized")
# RedHatAI/Qwen3.6-35B-A3B-NVFP4: every Linear but the GDN, the routers and lm_head
CT_NVFP4_MOE = _ct({"group_0": {**NVFP4_GROUP, "targets": ["Linear"]}}, ["lm_head", f"{LM}.embed_tokens", *GDN_QKVZ_OUT, *GDN_BA, *ROUTERS], "nvfp4-pack-quantized")
# unsloth/Qwen3.6-35B-A3B-NVFP4-Fast: channel-fp8 attention / GDN / lm_head, NVFP4 experts and shared expert; the ignore list names the linear_attn container too
CT_MIXED_FAST = _ct({
    "group_0": {**FP8_CHANNEL_GROUP, "targets": [r"re:.*self_attn\.(q|k|v|o)_proj$", r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$", "re:.*lm_head"]},
    "group_1": {**NVFP4_GROUP, "targets": [r"re:.*mlp\.experts\.\d+\.(gate|up|down)_proj$", r"re:.*shared_expert\.(gate|up|down)_proj$"]},
}, [f"{LM}.layers.0.linear_attn", f"{LM}.layers.0.linear_attn.norm", *GDN_BA, *ROUTERS], "mixed-precision")
# primitive-ai/Ornith-1.5-35B-A3B-mixed-NVFP4-FP8: static per-tensor fp8 attention / GDN / shared expert, NVFP4 experts; the ignore list names every ``experts.N`` container
CT_TENSOR_FP8_MOE = _ct({
    "group_0": {**FP8_TENSOR_STATIC_GROUP, "targets": [r"re:.*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$", r"re:.*\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$", r"re:.*\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)$"]},
    "group_1": {**NVFP4_GROUP, "targets": [r"re:.*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"]},
}, ["lm_head", f"{LM}.layers.0.linear_attn", *GDN_BA, *ROUTERS, *[f"{LM}.layers.{l}.mlp.experts.{e}" for l in (0, 1) for e in range(E)]], "mixed-precision")
# block-fp8 dense side with NVFP4 experts (the JIAQI13 / kyaky export)
CT_BLOCK_MOE = _ct({
    "group_0": {**FP8_BLOCK_GROUP, "targets": [r"re:.*self_attn\.(q|k|v|o)_proj$", r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$", r"re:.*shared_expert\.(gate|up|down)_proj$"]},
    "group_1": {**NVFP4_GROUP, "targets": [r"re:.*mlp\.experts\.\d+\.(gate|up|down)_proj$"]},
}, ["lm_head", *GDN_BA, *ROUTERS], "mixed-precision")
# block-fp8 everywhere, experts included, under llm-compressor's names (``weight_scale`` for the block scale)
CT_BLOCK_EXPERTS = _ct({"group_0": {**FP8_BLOCK_GROUP, "targets": ["Linear"]}}, ["lm_head", f"{LM}.embed_tokens", *GDN_BA, *ROUTERS], "float-quantized")


def _layout(name: str) -> tuple[bool, dict | None, dict[str, torch.Tensor]]:
    """``(moe, quantization_config, raw tensors)`` of one released layout."""
    moe = name != "ct_nvfp4_dense"
    raw = _dense_bf16(moe)
    if name == "bf16":
        _experts(raw)
        return moe, None, raw
    if name == "fp8_block":
        _quantize(raw, GDN_QKVZ_OUT + ATTN + SHARED, lambda w: _fp8_block(w, ct=False))
        _experts(raw, lambda w: _fp8_block(w, ct=False))
        return moe, QWEN_FP8, raw
    if name == "modelopt_mixed":
        _quantize(raw, GDN_QKVZ_OUT + ATTN, _fp8_tensor)
        _quantize(raw, SHARED + ["lm_head"], lambda w: _nvfp4(w, ct=False))
        _experts(raw, lambda w: _nvfp4(w, ct=False))
        return moe, MODELOPT_MIXED, raw
    if name == "ct_nvfp4_dense":
        _quantize(raw, GDN_QKVZ_OUT + GDN_BA + ATTN + DENSE_MLP, lambda w: _nvfp4(w, ct=True))
        return moe, CT_NVFP4_DENSE, raw
    if name == "ct_nvfp4_moe":
        _quantize(raw, ATTN + SHARED, lambda w: _nvfp4(w, ct=True))
        _experts(raw, lambda w: _nvfp4(w, ct=True))
        return moe, CT_NVFP4_MOE, raw
    if name == "ct_mixed_fast":
        _quantize(raw, GDN_QKVZ_OUT + ATTN + ["lm_head"], _fp8_channel)
        _quantize(raw, SHARED, lambda w: _nvfp4(w, ct=True))
        _experts(raw, lambda w: _nvfp4(w, ct=True))
        raw.update({f"{LM}.layers.1.self_attn.k_scale": torch.rand(()), f"{LM}.layers.1.self_attn.v_scale": torch.rand(())})
        return moe, CT_MIXED_FAST, raw
    if name == "ct_tensor_fp8_moe":
        _quantize(raw, GDN_QKVZ_OUT + ATTN + SHARED, _fp8_tensor_ct)
        _experts(raw, lambda w: _nvfp4(w, ct=True))
        return moe, CT_TENSOR_FP8_MOE, raw
    if name == "ct_block_moe":
        _quantize(raw, GDN_QKVZ_OUT + ATTN + SHARED, lambda w: _fp8_block(w, ct=True))
        _experts(raw, lambda w: _nvfp4(w, ct=True))
        return moe, CT_BLOCK_MOE, raw
    if name == "ct_block_experts":
        _quantize(raw, GDN_QKVZ_OUT + ATTN + SHARED, lambda w: _fp8_block(w, ct=True))
        _experts(raw, lambda w: _fp8_block(w, ct=True))
        return moe, CT_BLOCK_EXPERTS, raw
    raise KeyError(name)


LAYOUTS = ["bf16", "fp8_block", "modelopt_mixed", "ct_nvfp4_dense", "ct_nvfp4_moe", "ct_mixed_fast", "ct_tensor_fp8_moe", "ct_block_moe", "ct_block_experts"]


def _config_json(moe: bool, quantization_config) -> dict:
    text = {
        "model_type": "qwen3_5_moe_text" if moe else "qwen3_5_text", "num_hidden_layers": 2, "hidden_size": H, "vocab_size": V,
        "head_dim": AHD, "num_attention_heads": QH, "num_key_value_heads": KVH, "intermediate_size": I,
        "layer_types": ["linear_attention", "full_attention"],
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
        "max_position_embeddings": 4096, "rms_norm_eps": 1e-6, "hidden_act": "silu", "tie_word_embeddings": False,
        "linear_num_key_heads": KH, "linear_num_value_heads": VH, "linear_key_head_dim": HD, "linear_value_head_dim": HD,
        "linear_conv_kernel_dim": 4,
    }
    if moe:
        text.update(num_experts=E, num_experts_per_tok=2, moe_intermediate_size=MI, shared_expert_intermediate_size=I)
    return {
        "architectures": ["Qwen3_5MoeForConditionalGeneration" if moe else "Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5_moe" if moe else "qwen3_5", "text_config": text, "quantization_config": quantization_config,
    }


def _write(folder, moe: bool, quantization_config, raw: dict[str, torch.Tensor], *, shards: int = 2) -> str:
    """Spread the tensors over ``shards`` files, without an index, so fusions cross a file boundary."""
    names = sorted(raw)
    for i in range(shards):
        save_file({n: raw[n] for n in names[i::shards]}, str(folder / f"model-{i:05d}.safetensors"))
    (folder / "config.json").write_text(json.dumps(_config_json(moe, quantization_config)))
    return str(folder)


def _install(folder: str) -> None:
    """Install the folder's QuantConfig process-wide, as EngineConfig does before the reader runs."""
    hf = cached_load_hf_config(folder)
    set_quant_config(checkpoint_quant_config(folder, hf, get_model_spec(hf.architectures[0])))


def _load(folder: str, *, experts: bool = False) -> dict[str, torch.Tensor]:
    _install(folder)
    return {n: t.clone() for n, t in iter_weights(folder, torch.device("cpu"), include_moe_experts=experts, include_non_moe=True)}


def _meta_state_dict(folder: str) -> dict[str, torch.Tensor]:
    """State dict of the model the engine builds for ``folder`` (routed experts offloaded), on the meta device."""
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import _decode_target
    from freetoken.layers import rotary
    from freetoken.models import create_model
    from freetoken.utils.torch_utils import torch_dtype

    strategy = "offload" if cached_load_hf_config(folder).architectures[0].startswith("Qwen3_5Moe") else "auto"
    config = EngineConfig(model_path=folder, tp_info=try_get_tp_info(), dtype=torch.bfloat16, moe_strategy=strategy)
    object.__setattr__(config.model_config, "moe_strategy", strategy)
    object.__setattr__(config.model_config, "decode_target", _decode_target(config))
    saved = rotary._ROPE_DEVICE
    rotary.set_rope_device(torch.device("cpu"))  # get_rope refuses to build on meta
    rotary.get_rope.cache_clear()
    try:
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            return create_model(config.model_config).state_dict()
    finally:
        rotary.set_rope_device(saved)
        rotary.get_rope.cache_clear()


@pytest.fixture(scope="module", params=LAYOUTS)
def checkpoint(request, tmp_path_factory):
    torch.manual_seed(LAYOUTS.index(request.param))
    moe, quant, raw = _layout(request.param)
    return request.param, _write(tmp_path_factory.mktemp(request.param), moe, quant, raw), raw


# --------------------------------------------------------------------------- the reader against the engine's model


def test_emitted_keys_are_the_model_state_dict(checkpoint):
    """Every layout fills exactly the buffers the engine builds from the same config, with the buffers' shapes and (for the weights) dtypes."""
    _name, folder, _raw = checkpoint
    loaded, state = _load(folder), _meta_state_dict(folder)
    assert set(loaded) == set(state)
    for key, tensor in loaded.items():
        assert tensor.shape == state[key].shape, key
        if key.endswith(".weight"):
            assert tensor.dtype is state[key].dtype, key
    assert not any(k.endswith((".input_global_scale", ".weight_scale_2", ".weight_global_scale", ".k_scale")) for k in loaded)
    assert not any(".mlp.experts." in k or k.startswith(("mtp.", "model.visual.")) for k in loaded)


def test_expert_quant_tag_follows_the_config(checkpoint):
    name, folder, _raw = checkpoint
    config = parse_config(cached_load_hf_config(folder))
    expected = {"bf16": "none", "fp8_block": "fp8_block", "ct_block_experts": "fp8_block", "ct_nvfp4_dense": "none"}.get(name, "nvfp4")
    assert config.expert_quant == expected
    assert config.weight_block_size == ((128, 128) if expected == "fp8_block" else None)


def _slices(fused: torch.Tensor, parts: list[torch.Tensor]) -> list[torch.Tensor]:
    return list(torch.split(fused, [p.shape[0] for p in parts], dim=0))


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype is b.dtype and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_bf16_fusions_slice_back_and_norms_get_plus_one(checkpoint):
    name, folder, raw = checkpoint
    if name != "bf16":
        pytest.skip("bf16 layout only")
    loaded = _load(folder)
    gdn, attn = f"{LM}.layers.0.linear_attn", f"{LM}.layers.1.self_attn"
    parts = [raw[f"{gdn}.in_proj_{p}.weight"] for p in ("qkv", "z", "b", "a")]
    assert all(_same(p, s) for p, s in zip(parts, _slices(loaded["model.layers.0.linear_attn.in_proj.weight"], parts)))
    parts = [raw[f"{attn}.{p}_proj.weight"] for p in "qkv"]
    assert all(_same(p, s) for p, s in zip(parts, _slices(loaded["model.layers.1.self_attn.qkv_proj.weight"], parts)))
    merged = loaded["model.layers.1.mlp.shared_expert.gate_up_proj.weight"]
    assert _same(merged[:I], raw[f"{LM}.layers.1.mlp.shared_expert.gate_proj.weight"])
    assert _same(merged[I:], raw[f"{LM}.layers.1.mlp.shared_expert.up_proj.weight"])
    assert torch.equal(loaded["model.norm.weight"], raw[f"{LM}.norm.weight"] + 1.0)
    assert torch.equal(loaded["model.layers.1.self_attn.q_norm.weight"], raw[f"{attn}.q_norm.weight"] + 1.0)
    assert torch.equal(loaded["model.layers.0.linear_attn.norm.weight"], raw[f"{gdn}.norm.weight"])
    assert loaded["model.layers.0.linear_attn.A_log"].dtype is torch.float32


def test_bf16_stacked_experts_pass_through_only_when_asked(checkpoint):
    name, folder, raw = checkpoint
    if name != "bf16":
        pytest.skip("bf16 layout only")
    _install(folder)
    experts = dict(iter_weights(folder, torch.device("cpu"), include_moe_experts=True, include_non_moe=False))
    assert set(experts) == {f"model.layers.{l}.mlp.experts.{p}" for l in (0, 1) for p in ("gate_up_proj", "down_proj")}
    assert torch.equal(experts["model.layers.0.mlp.experts.gate_up_proj"], raw[f"{LM}.layers.0.mlp.experts.gate_up_proj"])
    assert not any(".experts." in k for k in _load(folder))


def test_block_fp8_fuses_weight_and_scale_per_kind(checkpoint):
    name, folder, raw = checkpoint
    if name not in ("fp8_block", "ct_block_moe", "ct_block_experts"):
        pytest.skip("block-fp8 layouts only")
    loaded = _load(folder)
    scale = "weight_scale_inv" if name == "fp8_block" else "weight_scale"
    gdn, attn = f"{LM}.layers.0.linear_attn", f"{LM}.layers.1.self_attn"
    for fused, sources in (
        ("model.layers.1.self_attn.qkv_proj", [f"{attn}.{p}_proj" for p in "qkv"]),
        ("model.layers.0.linear_attn.in_proj_qkvz", [f"{gdn}.in_proj_qkv", f"{gdn}.in_proj_z"]),
        ("model.layers.0.mlp.shared_expert.gate_up_proj", [f"{LM}.layers.0.mlp.shared_expert.{p}_proj" for p in ("gate", "up")]),
    ):
        parts = [raw[f"{s}.weight"] for s in sources]
        assert all(_same(p, b) for p, b in zip(parts, _slices(loaded[f"{fused}.weight"], parts)))
        scales = [raw[f"{s}.{scale}"] for s in sources]
        assert all(_same(p, b) for p, b in zip(scales, _slices(loaded[f"{fused}.weight_scale_inv"], scales)))
    assert loaded["model.layers.0.linear_attn.in_proj_qkvz.weight_scale_inv"].shape == (6, 1)
    ba = loaded["model.layers.0.linear_attn.in_proj_ba.weight"]
    assert ba.dtype is torch.bfloat16 and torch.equal(ba, torch.cat([raw[f"{gdn}.in_proj_b.weight"], raw[f"{gdn}.in_proj_a.weight"]]))
    assert loaded["lm_head.weight"].dtype is torch.bfloat16


def test_modelopt_fp8_scales_broadcast_per_part_and_input_scale_is_the_max(checkpoint):
    name, folder, raw = checkpoint
    if name != "modelopt_mixed":
        pytest.skip("modelopt layout only")
    loaded = _load(folder)
    attn = f"{LM}.layers.1.self_attn"
    scale = loaded["model.layers.1.self_attn.qkv_proj.weight_scale"]
    assert scale.dtype is torch.float32 and scale.shape == (2 * QH * AHD + 2 * KVH * AHD,)
    expected = torch.cat([raw[f"{attn}.{p}_proj.weight_scale"].expand(raw[f"{attn}.{p}_proj.weight"].shape[0]) for p in "qkv"])
    assert torch.equal(scale, expected)
    assert torch.equal(loaded["model.layers.1.self_attn.qkv_proj.input_scale"], torch.stack([raw[f"{attn}.{p}_proj.input_scale"] for p in "qkv"]).max())
    assert loaded["model.layers.1.self_attn.o_proj.input_scale"].shape == ()
    # NVFP4 shared expert: each part keeps its own block scales and global; the input scale is the max of the parts
    shared = f"{LM}.layers.0.mlp.shared_expert"
    fused = "model.layers.0.mlp.shared_expert.gate_up_proj"
    assert _same(loaded[f"{fused}.weight"][:I], raw[f"{shared}.gate_proj.weight"])
    assert _same(loaded[f"{fused}.weight_scale"][I:], raw[f"{shared}.up_proj.weight_scale"])
    glob = loaded[f"{fused}.weight_global"]
    assert glob.dtype is torch.float16 and torch.equal(glob[:I], raw[f"{shared}.gate_proj.weight_scale_2"].to(torch.float16).expand(I))
    assert torch.equal(loaded[f"{fused}.input_scale"], torch.stack([raw[f"{shared}.{p}_proj.input_scale"] for p in ("gate", "up")]).max())
    assert loaded["lm_head.input_scale"].shape == ()
    assert loaded["lm_head.weight"].dtype is torch.uint8 and loaded["lm_head.weight_global"].shape == (V,)


def test_compressed_tensors_nvfp4_global_is_the_reciprocal(checkpoint):
    name, folder, raw = checkpoint
    if name not in ("ct_nvfp4_dense", "ct_nvfp4_moe"):
        pytest.skip("llm-compressor NVFP4 layouts only")
    loaded = _load(folder)
    attn = f"{LM}.layers.1.self_attn"
    glob = loaded["model.layers.1.self_attn.qkv_proj.weight_global"]
    expected = torch.cat([(1.0 / raw[f"{attn}.{p}_proj.weight_global_scale"]).to(torch.float16).expand(raw[f"{attn}.{p}_proj.weight_packed"].shape[0]) for p in "qkv"])
    assert glob.dtype is torch.float16 and torch.equal(glob, expected)
    assert _same(loaded["model.layers.1.self_attn.o_proj.weight_scale"], raw[f"{attn}.o_proj.weight_scale"])
    # ``dynamic: local`` ships the quant-side activation global too; it lands as the dequant-side input_scale
    assert torch.equal(loaded["model.layers.1.self_attn.o_proj.input_scale"], (1.0 / raw[f"{attn}.o_proj.input_global_scale"]).reshape(()))
    assert torch.equal(loaded["model.layers.1.self_attn.qkv_proj.input_scale"], torch.stack([1.0 / raw[f"{attn}.{p}_proj.input_global_scale"].reshape(()) for p in "qkv"]).max())
    assert loaded["lm_head.weight"].dtype is torch.bfloat16
    gdn = f"{LM}.layers.0.linear_attn"
    if name == "ct_nvfp4_dense":
        # the GDN projections are quantized too, so qkv|z and b|a are both native NVFP4
        assert loaded["model.layers.0.linear_attn.in_proj_qkvz.weight"].dtype is torch.uint8
        ba = loaded["model.layers.0.linear_attn.in_proj_ba.weight"]
        assert ba.dtype is torch.uint8 and _same(ba[VH:], raw[f"{gdn}.in_proj_a.weight_packed"])
        assert loaded["model.layers.0.linear_attn.in_proj_ba.weight_global"].shape == (2 * VH,)
    else:
        parts = [raw[f"{gdn}.in_proj_{p}.weight"] for p in ("qkv", "z", "b", "a")]
        assert all(_same(p, s) for p, s in zip(parts, _slices(loaded["model.layers.0.linear_attn.in_proj.weight"], parts)))


def test_channel_fp8_keeps_row_order_and_an_ignored_container_does_not_shield_its_children(checkpoint):
    name, folder, raw = checkpoint
    if name != "ct_mixed_fast":
        pytest.skip("unsloth layout only")
    loaded = _load(folder)
    attn, gdn = f"{LM}.layers.1.self_attn", f"{LM}.layers.0.linear_attn"
    scale = loaded["model.layers.1.self_attn.qkv_proj.weight_scale"]
    assert scale.dtype is torch.float32
    assert torch.equal(scale, torch.cat([raw[f"{attn}.{p}_proj.weight_scale"].reshape(-1) for p in "qkv"]).to(torch.float32))
    assert "model.layers.1.self_attn.qkv_proj.input_scale" not in loaded  # dynamic per-token fp8: no static activation scale
    # ``ignore`` names ``layers.0.linear_attn`` itself; its quantized projections still load as fp8
    qkvz = loaded["model.layers.0.linear_attn.in_proj_qkvz.weight"]
    assert qkvz.dtype is FP8 and _same(qkvz[: 2 * KH * HD + VH * HD], raw[f"{gdn}.in_proj_qkv.weight"])
    assert loaded["model.layers.0.linear_attn.in_proj_ba.weight"].dtype is torch.bfloat16
    assert loaded["lm_head.weight"].dtype is FP8 and loaded["lm_head.weight_scale"].shape == (V,)
    assert loaded["model.layers.0.mlp.shared_expert.down_proj.weight"].dtype is torch.uint8


def test_static_fp8_keeps_the_activation_scale_and_ignored_expert_containers_do_not_shield_the_experts(checkpoint):
    name, folder, raw = checkpoint
    if name != "ct_tensor_fp8_moe":
        pytest.skip("Ornith layout only")
    loaded = _load(folder)
    attn = f"{LM}.layers.1.self_attn"
    assert loaded["model.layers.1.self_attn.qkv_proj.weight_scale"].shape == (2 * QH * AHD + 2 * KVH * AHD,)
    expected = torch.stack([raw[f"{attn}.{p}_proj.input_scale"].reshape(()) for p in "qkv"]).max().to(torch.float32)
    assert torch.equal(loaded["model.layers.1.self_attn.qkv_proj.input_scale"], expected)
    assert loaded["model.layers.0.mlp.shared_expert.down_proj.input_scale"].dtype is torch.float32
    assert loaded["lm_head.weight"].dtype is torch.bfloat16
    assert parse_config(cached_load_hf_config(folder)).expert_quant == "nvfp4"


# --------------------------------------------------------------------------- the expert reader shares the dialect names


def test_nvfp4_expert_pieces_read_either_dialect_from_a_single_file(checkpoint):
    name, folder, raw = checkpoint
    if name not in ("modelopt_mixed", "ct_nvfp4_moe", "ct_tensor_fp8_moe"):
        pytest.skip("NVFP4 expert layouts only")
    _install(folder)
    config = parse_config(cached_load_hf_config(folder))
    spec = nvfp4_expert_spec(folder, config)
    pieces = list(iter_nvfp4_expert_pieces(folder, config, spec))
    assert len(pieces) == 2 * E
    layer, e0, e1, piece = next(p for p in pieces if p[0] == 1 and p[1] == 2)
    assert (e0, e1) == (2, 3)
    base = f"{LM}.layers.1.mlp.experts.2.gate_proj"
    if name == "modelopt_mixed":
        assert _same(piece["gate"][0], raw[f"{base}.weight"])
        assert torch.equal(piece["gate_global"].reshape(-1), raw[f"{base}.weight_scale_2"].reshape(-1).to(torch.float16))
    else:
        assert _same(piece["gate"][0], raw[f"{base}.weight_packed"])
        assert _same(piece["down_scale"][0], raw[f"{LM}.layers.1.mlp.experts.2.down_proj.weight_scale"])
        assert torch.equal(piece["gate_global"].reshape(-1), (1.0 / raw[f"{base}.weight_global_scale"]).to(torch.float16))


@pytest.mark.parametrize("parallel", [False, True])
def test_block_fp8_expert_pieces_read_either_dialect(checkpoint, parallel):
    """The block-fp8 expert reader takes the scale's name from the dialect: ``weight_scale_inv`` (HF fp8) or ``weight_scale`` (llm-compressor)."""
    name, folder, raw = checkpoint
    if name not in ("fp8_block", "ct_block_experts"):
        pytest.skip("block-fp8 expert layouts only")
    _install(folder)
    config = parse_config(cached_load_hf_config(folder))
    pieces = list(iter_expert_pieces(folder, config, QuantKind.FP8_BLOCK, parallel=parallel))
    assert len(pieces) == 2 * E
    layer, e0, e1, piece = next(p for p in pieces if p[0] == 1 and p[1] == 2)
    base = f"{LM}.layers.1.mlp.experts.2"
    scale = "weight_scale_inv" if name == "fp8_block" else "weight_scale"
    assert _same(piece["gate"][0], raw[f"{base}.gate_proj.weight"])
    assert torch.equal(piece["down_scale"][0].float(), raw[f"{base}.down_proj.{scale}"].float())


# --------------------------------------------------------------------------- the family's unquantized modules


def _nvfp4_reference(packed: torch.Tensor, scale: torch.Tensor, global_rows: torch.Tensor) -> torch.Tensor:
    """e2m1 codes (low nibble first) x e4m3 block scale x per-row global, as the kernel computes it."""
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(packed.shape[0], -1).long()
    values = table[codes & 7] * torch.where(codes & 8 > 0, -1.0, 1.0)
    return (values * scale.float().repeat_interleave(16, dim=1) * global_rows.float()[:, None]).to(torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the NVFP4 dequant kernel runs on CUDA")
def test_a_quantized_router_is_dequantized_because_the_family_serves_it_bf16(tmp_path):
    """``targets: ["Linear"]`` with no ignore entry for ``mlp.gate`` quantizes the router; the model builds it bf16 (register.py unquantized_modules), so the reader dequantizes."""
    torch.manual_seed(11)
    moe, quant, raw = _layout("ct_nvfp4_moe")
    quant = {**quant, "ignore": [i for i in quant["ignore"] if not i.endswith(".mlp.gate")]}
    gates = [f"{LM}.layers.{l}.mlp.gate" for l in (0, 1)]
    _quantize(raw, gates, lambda w: _nvfp4(w, ct=True))
    loaded = _load(_write(tmp_path, moe, quant, raw))
    for layer, gate in enumerate(gates):
        got = loaded[f"model.layers.{layer}.mlp.gate.weight"]
        assert got.dtype is torch.bfloat16 and got.shape == (E, H)
        expected = _nvfp4_reference(raw[f"{gate}.weight_packed"], raw[f"{gate}.weight_scale"], (1.0 / raw[f"{gate}.weight_global_scale"]).to(torch.float16).expand(E))
        assert torch.equal(got, expected)


# --------------------------------------------------------------------------- checkpoints that disagree with their config


ATTN1 = f"{LM}.layers.1.self_attn"
REJECTED = [
    pytest.param(None, lambda w: {f"{ATTN1}.q_proj.weight": w, f"{ATTN1}.q_proj.weight_scale": torch.rand(())},
                 r"q_proj\.weight_scale: .*declares .*q_proj unquantized", id="scale the config does not declare"),
    pytest.param(None, lambda w: {f"{ATTN1}.o_proj.weight": w.to(FP8)},
                 r"o_proj\.weight is torch\.float8", id="fp8 weight the config declares bf16"),
    pytest.param(MODELOPT_MIXED, lambda w: {f"{ATTN1}.o_proj.weight": w, f"{ATTN1}.o_proj.weight_scale": torch.rand(()), f"{ATTN1}.o_proj.input_scale": torch.rand(())},
                 r"o_proj: weight is torch\.bfloat16", id="bf16 weight the config declares fp8"),
    pytest.param(CT_MIXED_FAST, lambda w: {f"{ATTN1}.o_proj.weight": w.to(FP8), f"{ATTN1}.o_proj.weight_scale": torch.rand(3, 1)},
                 "expected 1 or", id="per-channel scale with the wrong row count"),
    pytest.param(CT_BLOCK_MOE, lambda w: {f"{ATTN1}.o_proj.weight": w.to(FP8), f"{ATTN1}.o_proj.weight_scale": torch.rand(2, 1)},
                 r"weight_scale_inv is \(2, 1\), expected \(1, 2\)", id="block scale of the wrong shape"),
    pytest.param(CT_NVFP4_MOE, lambda w: {f"{ATTN1}.o_proj.weight_packed": torch.zeros(H, QH * AHD // 2, dtype=torch.uint8), f"{ATTN1}.o_proj.weight_global_scale": torch.rand(1)},
                 "missing tensors of .*o_proj", id="quantized module without its block scale"),
]


@pytest.mark.parametrize("quantization_config, tensors, match", REJECTED)
def test_a_checkpoint_disagreeing_with_its_quant_config_is_rejected(tmp_path, quantization_config, tensors, match):
    save_file(tensors(_bf16(H, QH * AHD)), str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(_config_json(True, quantization_config)))
    with pytest.raises(ValueError, match=match):
        _load(str(tmp_path))
