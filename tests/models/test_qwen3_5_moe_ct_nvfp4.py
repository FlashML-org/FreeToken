"""qwen3_5_moe compressed-tensors (llm-compressor) NVFP4 support.

Qwen3.6-35B-A3B-class MoE checkpoints exported with ``quant_method:
compressed-tensors`` store their routed experts as ``weight_packed | weight_scale |
weight_global_scale`` (quant-side global), either per-expert
(``...experts.E.{gate,up,down}_proj``, AEON Qwen3.6-35B) or stacked per-layer
(``...experts.{gate_up,down}_proj`` [E*rows, cols], Kwaipilot-KAT-Coder). These tests
pin the fixes that let such checkpoints convert/serve with NVFP4 offload banks: config
detection (config_groups, format-only + ``recipe``), the CT expert-source specs, the
single-file no-index + stacked bank loaders, the shared-expert gate/up fusion, the fp8
GDN (``gdn:fp8`` recipe) and NVFP4 embed_tokens dequantization, and lm_head kept native
NVFP4 when the export quantized it.
"""

from __future__ import annotations

import pytest

from freetoken.models.qwen3_5_moe.config import parse_config


class _Cfg:
    """Attribute-access shim over a dict (what cached_load_hf_config hands parse_config).
    Only ``text_config`` is recursed; ``quantization_config`` stays a plain dict so
    ``quant.get`` accessors fire."""

    def __init__(self, data: dict):
        for k, v in data.items():
            setattr(self, k, _Cfg(v) if k == "text_config" and isinstance(v, dict) else v)


def _text_config(num_layers: int, num_experts: int) -> dict:
    return {
        "hidden_size": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "num_hidden_layers": num_layers,
        "vocab_size": 32,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 1000,
        "tie_word_embeddings": False,
        "num_experts": num_experts,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 8,
        "layer_types": ["full_attention"] * num_layers,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 2,
    }


def _ct_nvfp4_quant() -> dict:
    """Shape of the Qwen3.6-35B-A3B-NVFP4 exports: one generic Linear group in the
    NVFP4 geometry (no ``quant_algo``; ``quant_method`` carries it) plus an ``ignore``
    list that leaves the GDN (linear_attn.*), the routers, lm_head and vision bf16."""
    ignore = [
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
        "model.language_model.layers.0.linear_attn.in_proj_b",
        "model.language_model.layers.0.linear_attn.in_proj_a",
        "model.language_model.layers.1.linear_attn.out_proj",
        "model.language_model.layers.0.mlp.gate",
        "model.language_model.layers.0.mlp.shared_expert_gate",
        "model.language_model.layers.1.mlp.gate",
        "model.language_model.layers.1.mlp.shared_expert_gate",
        "lm_head",
    ]
    return {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "ignore": ignore,
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


def _ct_mixed_quant() -> dict:
    """A hypothetical mixed export: NVFP4 dense groups but fp8 routed experts. The
    expert-targeted group decides -- experts are NOT nvfp4."""
    return {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "group_0": {
                "targets": ["re:.*self_attn.*_proj$"],
                "weights": {"num_bits": 4, "type": "float", "group_size": 16, "strategy": "tensor_group"},
                "format": "nvfp4-pack-quantized",
            },
            "group_1": {
                "targets": ["re:.*mlp\\.experts\\..*(gate|up|down)_proj$"],
                "weights": {"num_bits": 8, "type": "float", "strategy": "block"},
                "format": "float-quantized",
            },
        },
    }


def _ct_format_only_quant() -> dict:
    """Shape of doth4580/Kwaipilot-KAT-Coder-V2.5-Dev-NVFP4-MIXED: no config_groups, no
    ignore -- just ``format`` and a ``recipe`` that quantizes everything to NVFP4 except
    the GDN (per-tensor fp8) and the routers (skipped)."""
    return {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "recipe": "all,gdn:fp8,-router",
    }


def _hf_config(num_layers: int = 2, num_experts: int = 4, quant: dict | None = None) -> _Cfg:
    data = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": _text_config(num_layers, num_experts),
    }
    if quant is not None:
        data["quantization_config"] = quant
    return _Cfg(data)


# -----------------------------------------------------------------------------------
# config detection
# -----------------------------------------------------------------------------------


def test_parse_config_ct_moe_nvfp4():
    """compressed-tensors MoE checkpoint: the routed experts resolve to nvfp4 (offload
    banks), and the dense/attention weights keep native FP4 too -- but the GDN out_proj
    stays bf16 because the export's ``ignore`` list skipped the whole linear_attn."""
    cfg = parse_config(_hf_config(quant=_ct_nvfp4_quant()))
    assert cfg.expert_quant == "nvfp4"
    assert cfg.dense_quant == "nvfp4"
    assert cfg.attn_quant == "nvfp4"
    assert cfg.gdn_quant == "none"
    assert cfg.is_moe


def test_ct_linear_attn_ignored_detection():
    """The GDN-out_proj signal is read from the ignore list: exact per-module names
    (AEON Qwen3.6-35B) and regexes both match; an export that only skipped in_proj_*
    (dense Qwen3.6-27B) must NOT match."""
    from freetoken.models.qwen3_5_moe.config import _ct_linear_attn_ignored

    moe = _hf_config(quant=_ct_nvfp4_quant())
    assert _ct_linear_attn_ignored(moe)
    # regex form of the same skip
    regex_quant = dict(_ct_nvfp4_quant())
    regex_quant["ignore"] = ["re:.*linear_attn\\..*", "re:.*mlp\\.gate.*", "lm_head"]
    assert _ct_linear_attn_ignored(_hf_config(quant=regex_quant))
    # only in_proj_* skipped -> out_proj stayed quantized (Qwen3.6-27B)
    only_in = dict(_ct_nvfp4_quant())
    only_in["ignore"] = [x for x in only_in["ignore"] if "out_proj" not in x]
    assert not _ct_linear_attn_ignored(_hf_config(quant=only_in))
    # no ignore list at all -> everything Linear quantized
    no_ignore = dict(_ct_nvfp4_quant())
    no_ignore.pop("ignore", None)
    assert not _ct_linear_attn_ignored(_hf_config(quant=no_ignore))


def test_parse_config_ct_moe_gdn_quantized_when_not_ignored():
    """A CT export that quantizes the GDN (no linear_attn in its ignore list, e.g. the
    dense Qwen3.6-27B) keeps out_proj native FP4 (gdn_quant == "nvfp4")."""
    quant = dict(_ct_nvfp4_quant())
    quant["ignore"] = [x for x in quant["ignore"] if "linear_attn" not in x]
    cfg = parse_config(_hf_config(quant=quant))
    assert cfg.attn_quant == "nvfp4"
    assert cfg.gdn_quant == "nvfp4"


def test_parse_config_ct_format_only_moe():
    """Format-only export (Kwaipilot-KAT-Coder): no config_groups/ignore, just
    ``format: nvfp4-pack-quantized`` + ``recipe: all,gdn:fp8,-router``. Everything is
    NVFP4 except the GDN (fp8 -> gdn_quant none) and lm_head (kept NVFP4)."""
    cfg = parse_config(_hf_config(quant=_ct_format_only_quant()))
    assert cfg.expert_quant == "nvfp4"
    assert cfg.attn_quant == "nvfp4"
    assert cfg.dense_quant == "nvfp4"
    assert cfg.gdn_quant == "none"
    assert cfg.lm_head_quant == "nvfp4"


def test_ct_recipe_gdn_override():
    """The GDN-out_proj NVFP4 signal honors the ``recipe`` string: ``gdn:fp8`` leaves it
    non-NVFP4; a gdn:nvfp4 token or a recipe that quantizes the GDN keeps it NVFP4."""
    from freetoken.models.qwen3_5_moe.config import _ct_gdn_nvfp4

    base = dict(_ct_format_only_quant())
    assert not _ct_gdn_nvfp4(_hf_config(quant=base))  # gdn:fp8
    fp4 = dict(base)
    fp4["recipe"] = "all,gdn:nvfp4,-router"
    assert _ct_gdn_nvfp4(_hf_config(quant=fp4))
    no_recipe = dict(base)
    no_recipe.pop("recipe", None)
    assert _ct_gdn_nvfp4(_hf_config(quant=no_recipe))  # default: nvfp4


def test_parse_config_ct_dense_keeps_experts_none():
    """Dense compressed-tensors export (e.g. Qwen3.6-27B, num_experts==0) must keep
    expert_quant "none" -- the dense reader owns all of its weights."""
    cfg = parse_config(_hf_config(num_experts=0, quant=_ct_nvfp4_quant()))
    assert cfg.expert_quant == "none"
    assert cfg.dense_quant == "nvfp4"  # dense MLP is still native FP4
    assert cfg.num_experts == 0
    assert not cfg.moe_enabled


def test_parse_config_ct_mixed_fp8_experts_keep_none():
    """A mixed export whose routed experts are fp8 (not nvfp4) must not route them
    into the NVFP4 bank loader: the expert-targeted config group decides."""
    cfg = parse_config(_hf_config(quant=_ct_mixed_quant()))
    assert cfg.expert_quant == "none"


# -----------------------------------------------------------------------------------
# expert source spec selection
# -----------------------------------------------------------------------------------


def test_ct_expert_source_spec():
    from freetoken.models.qwen3_5_moe.weight import _NVFP4_CT_SOURCE_SPEC, _NVFP4_SOURCE_SPEC

    ct = _NVFP4_CT_SOURCE_SPEC
    m = ct.key_pattern.match(
        "model.language_model.layers.5.mlp.experts.7.gate_proj.weight_packed"
    )
    assert m and m.group("kind") == "weight_packed"
    assert ct.kind_map["weight_packed"] == "weight"
    assert ct.kind_map["weight_global_scale"] == "weight_scale_2"
    assert ct.global_reciprocal
    # W4A16 serving never consumes the calibrated activation scale.
    assert ct.key_pattern.match(
        "model.language_model.layers.5.mlp.experts.7.gate_proj.input_global_scale"
    ) is None
    assert _NVFP4_SOURCE_SPEC.kind_map is None
    assert not _NVFP4_SOURCE_SPEC.global_reciprocal


def test_select_expert_source_spec(monkeypatch):
    import freetoken.models.qwen3_5_moe.weight as w
    from freetoken.models.qwen3_5_moe.weight import (
        _NVFP4_CT_SOURCE_SPEC,
        _NVFP4_SOURCE_SPEC,
        _select_expert_source_spec,
    )

    monkeypatch.setattr(w, "_stacked_expert_keying", lambda _p: False)
    monkeypatch.setattr(
        w, "cached_load_hf_config",
        lambda _p: _Cfg({"quantization_config": {"quant_method": "compressed-tensors"}}),
    )
    assert _select_expert_source_spec("x") is _NVFP4_CT_SOURCE_SPEC
    monkeypatch.setattr(
        w, "cached_load_hf_config",
        lambda _p: _Cfg({"quantization_config": {"quant_algo": "NVFP4"}}),
    )
    assert _select_expert_source_spec("x") is _NVFP4_SOURCE_SPEC
    monkeypatch.setattr(w, "cached_load_hf_config", lambda _p: _Cfg({}))
    assert _select_expert_source_spec("x") is _NVFP4_SOURCE_SPEC


def test_ct_stacked_expert_spec():
    """The stacked (per-layer) CT spec matches ``...experts.gate_up_proj.weight_packed``
    (one tensor per layer) but NOT the per-expert keying, and vice versa."""
    from freetoken.models.qwen3_5_moe.weight import (
        _NVFP4_CT_SOURCE_SPEC,
        _NVFP4_CT_STACKED_SOURCE_SPEC,
    )

    stacked = _NVFP4_CT_STACKED_SOURCE_SPEC
    m = stacked.key_pattern.match(
        "model.language_model.layers.5.mlp.experts.gate_up_proj.weight_packed"
    )
    assert m and m.group("proj") == "gate_up_proj" and m.group("kind") == "weight_packed"
    assert stacked.stacked
    assert stacked.kind_map["weight_global_scale"] == "weight_scale_2"
    # per-expert keying must NOT match the stacked spec, and stacked not match per-expert
    assert stacked.key_pattern.match(
        "model.language_model.layers.5.mlp.experts.7.gate_proj.weight_packed"
    ) is None
    assert _NVFP4_CT_SOURCE_SPEC.key_pattern.match(
        "model.language_model.layers.5.mlp.experts.gate_up_proj.weight_packed"
    ) is None


def test_stacked_expert_keying_detection(tmp_path):
    """``_stacked_expert_keying`` tells the per-expert from the stacked layout by probing
    the weight map (down_proj key first must still resolve stacked)."""
    import torch

    import freetoken.models.qwen3_5_moe.weight as w

    p = "model.language_model."
    tensors = {
        p + "layers.0.mlp.experts.down_proj.weight_packed": torch.zeros(4, 4, dtype=torch.uint8),
    }
    _write_single_file(tmp_path, tensors)
    assert w._stacked_expert_keying(str(tmp_path))
    tensors2 = {
        p + "layers.0.mlp.experts.0.down_proj.weight_packed": torch.zeros(4, 4, dtype=torch.uint8),
    }
    _write_single_file(tmp_path, tensors2)
    assert not w._stacked_expert_keying(str(tmp_path))


# -----------------------------------------------------------------------------------
# single-file, no-index bank loading
# -----------------------------------------------------------------------------------


def _ct_expert_tensors(p: str, layer: int, expert: int, H: int, I: int,
                       g_scale, up_scale, down_scale) -> dict:
    import torch

    fp8 = torch.float8_e4m3fn
    base = f"{p}layers.{layer}.mlp.experts.{expert}."
    return {
        base + "gate_proj.weight_packed": torch.randint(0, 256, (I, H // 2), dtype=torch.uint8),
        base + "gate_proj.weight_scale": torch.randn(I, H // 16).abs().to(fp8),
        base + "gate_proj.weight_global_scale": torch.tensor([g_scale]),
        base + "gate_proj.input_global_scale": torch.tensor([1.0]),
        base + "up_proj.weight_packed": torch.randint(0, 256, (I, H // 2), dtype=torch.uint8),
        base + "up_proj.weight_scale": torch.randn(I, H // 16).abs().to(fp8),
        base + "up_proj.weight_global_scale": torch.tensor([up_scale]),
        base + "up_proj.input_global_scale": torch.tensor([1.0]),
        base + "down_proj.weight_packed": torch.randint(0, 256, (H, I // 2), dtype=torch.uint8),
        base + "down_proj.weight_scale": torch.randn(H, I // 16).abs().to(fp8),
        base + "down_proj.weight_global_scale": torch.tensor([down_scale]),
        base + "down_proj.input_global_scale": torch.tensor([1.0]),
    }


def _write_single_file(tmp_path, tensors: dict) -> None:
    """Write one ``model.safetensors`` with NO index -- the llm-compressor single-file
    layout (the checkpoints that broke the old unconditional index read)."""
    import safetensors.torch

    safetensors.torch.save_file(tensors, str(tmp_path / "model.safetensors"))


def test_load_nvfp4_expert_source_banks_ct_single_file(tmp_path):
    """The CT spec + the no-index weight-map fallback must place every (layer, expert)
    into the six native banks with the reciprocal globals. Fails on the old code
    (unconditional ``model.safetensors.index.json`` open -> FileNotFoundError)."""
    import torch
    from types import SimpleNamespace

    from freetoken.models.nvfp4_banks import load_nvfp4_expert_source_banks
    from freetoken.models.qwen3_5_moe.weight import _NVFP4_CT_SOURCE_SPEC

    L, E, H, I = 2, 2, 32, 32
    p = "model.language_model."
    tensors = {}
    for li in range(L):
        for ei in range(E):
            tensors |= _ct_expert_tensors(p, li, ei, H, I, 4.0, 2.0, 3.0)
    _write_single_file(tmp_path, tensors)

    cfg = SimpleNamespace(num_experts=E, hidden_size=H, moe_intermediate_size=I,
                          num_layers=L, num_moe_layers=L, first_k_dense_replace=0)
    collected: dict[int, dict[str, torch.Tensor]] = {}

    class _Sink:
        def __call__(self, layer_id: int, banks: dict) -> None:
            collected[layer_id] = {k: v.tensor.clone() for k, v in banks.items()}

    banks = load_nvfp4_expert_source_banks(
        str(tmp_path), cfg, _NVFP4_CT_SOURCE_SPEC,
        drop_page_cache=lambda _path: None, primary=True, layer_sink=_Sink(),
    )
    assert banks.keys() == {
        "gate_up_packed", "gate_up_scale", "gate_up_global",
        "down_packed", "down_scale", "down_global",
    }
    assert set(collected) == {0, 1}
    gu = collected[0]
    assert gu["gate_up_packed"].shape == (E, 2 * I, H // 2)
    assert gu["gate_up_scale"].shape == (E, 2 * I, H // 16)
    assert gu["gate_up_global"].shape == (E, 2 * I)
    assert gu["down_packed"].shape == (E, H, I // 2)
    assert gu["down_scale"].shape == (E, H, I // 16)
    assert gu["down_global"].shape == (E, H)

    # gate fills the fused rows [0:I), up the rows [I:2I) -- exact placement.
    src = tensors
    e0 = f"{p}layers.0.mlp.experts.0."
    assert torch.equal(gu["gate_up_packed"][0, :I], src[e0 + "gate_proj.weight_packed"])
    assert torch.equal(gu["gate_up_packed"][0, I:], src[e0 + "up_proj.weight_packed"])
    assert torch.equal(gu["down_packed"][0], src[e0 + "down_proj.weight_packed"])
    # globals are the reciprocal of the stored quant-side scales, per output row.
    assert gu["gate_up_global"][0, 0].item() == pytest.approx(0.25)  # 1 / 4.0
    assert gu["gate_up_global"][0, I].item() == pytest.approx(0.5)  # 1 / 2.0
    assert gu["down_global"][0, 0].item() == pytest.approx(
        torch.tensor(1 / 3).to(torch.float16).item()  # fp16-rounded reciprocal
    )


def _ct_stacked_expert_tensors(p: str, layer: int, E: int, H: int, I: int,
                               g_scale: float, down_scale: float | None = None) -> dict:
    """One layer's stacked (per-layer) expert tensors, real llm-compressor layout."""
    import torch

    fp8 = torch.float8_e4m3fn
    if down_scale is None:
        down_scale = g_scale
    base = f"{p}layers.{layer}.mlp.experts."
    return {
        base + "gate_up_proj.weight_packed": torch.randint(0, 256, (E * 2 * I, H // 2), dtype=torch.uint8),
        base + "gate_up_proj.weight_scale": torch.randn(E * 2 * I, H // 16).abs().to(fp8),
        base + "gate_up_proj.weight_global_scale": torch.tensor([g_scale]),
        base + "down_proj.weight_packed": torch.randint(0, 256, (E * H, I // 2), dtype=torch.uint8),
        base + "down_proj.weight_scale": torch.randn(E * H, I // 16).abs().to(fp8),
        base + "down_proj.weight_global_scale": torch.tensor([down_scale]),
    }


def test_load_nvfp4_stacked_expert_sources(tmp_path):
    """The stacked loader reshapes each [E*rows, cols] tensor into [E, rows, cols] and
    broadcasts the layer-global (reciprocated) scale into the *_global banks."""
    import torch
    from types import SimpleNamespace

    from freetoken.models.nvfp4_banks import load_nvfp4_stacked_expert_sources
    from freetoken.models.qwen3_5_moe.weight import _NVFP4_CT_STACKED_SOURCE_SPEC

    L, E, H, I = 2, 2, 32, 32
    p = "model.language_model."
    tensors = {}
    for li in range(L):
        tensors |= _ct_stacked_expert_tensors(p, li, E, H, I, 4.0, down_scale=8.0)
    _write_single_file(tmp_path, tensors)

    cfg = SimpleNamespace(num_experts=E, hidden_size=H, moe_intermediate_size=I,
                          num_layers=L, num_moe_layers=L, first_k_dense_replace=0)
    collected: dict[int, dict[str, torch.Tensor]] = {}

    class _Sink:
        def __call__(self, layer_id: int, banks: dict) -> None:
            collected[layer_id] = {k: v.tensor.clone() for k, v in banks.items()}

    banks = load_nvfp4_stacked_expert_sources(
        str(tmp_path), cfg, _NVFP4_CT_STACKED_SOURCE_SPEC,
        drop_page_cache=lambda _path: None, primary=True, layer_sink=_Sink(),
    )
    assert banks.keys() == {
        "gate_up_packed", "gate_up_scale", "gate_up_global",
        "down_packed", "down_scale", "down_global",
    }
    assert set(collected) == {0, 1}
    gu = collected[0]
    assert gu["gate_up_packed"].shape == (E, 2 * I, H // 2)
    assert gu["gate_up_scale"].shape == (E, 2 * I, H // 16)
    assert gu["gate_up_global"].shape == (E, 2 * I)
    assert gu["down_packed"].shape == (E, H, I // 2)
    assert gu["down_scale"].shape == (E, H, I // 16)
    assert gu["down_global"].shape == (E, H)

    src = tensors
    l0 = f"{p}layers.0.mlp.experts."
    assert torch.equal(
        gu["gate_up_packed"],
        src[l0 + "gate_up_proj.weight_packed"].view(E, 2 * I, H // 2),
    )
    assert torch.equal(
        gu["down_packed"],
        src[l0 + "down_proj.weight_packed"].view(E, H, I // 2),
    )
    # the layer-globals are reciprocated and broadcast to every expert row, per proj.
    assert gu["gate_up_global"][0, 0].item() == pytest.approx(0.25)  # 1 / 4.0
    assert gu["gate_up_global"].unique().numel() == 1
    assert gu["down_global"][0, 0].item() == pytest.approx(
        torch.tensor(1 / 8).to(torch.float16).item()  # 1 / 8.0, fp16-rounded
    )
    assert gu["down_global"].unique().numel() == 1


def test_dequant_fp8_weight_per_row():
    """Per-tensor fp8 (per-output-row scale, e.g. a ``gdn:fp8`` recipe) must broadcast
    the row scale, not fail on the row count vs 1."""
    import torch

    from freetoken.models.qwen3_5_moe.weight import _dequant_fp8_weight

    w8 = torch.randint(-128, 128, (6, 4), dtype=torch.int8).to(torch.float8_e4m3fn)
    rows = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = _dequant_fp8_weight(w8, rows)
    assert out.shape == (6, 4)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out[2].float(), (w8[2].float() * 3.0), atol=0.5)
    # a scalar scale still works
    out2 = _dequant_fp8_weight(w8, torch.tensor(2.0))
    assert out2.shape == (6, 4)
    assert torch.allclose(out2[0].float(), (w8[0].float() * 2.0), atol=0.5)


# -----------------------------------------------------------------------------------
# dense pass: expert exclusion + shared-expert gate/up fusion
# -----------------------------------------------------------------------------------


def test_iter_weights_ct_moe_dense_pass(tmp_path, monkeypatch):
    """compressed-tensors MoE dense pass: routed experts must NOT leak into the dense
    weights, and the shared expert's gate/up must fuse into the native ``gate_up_proj``
    (W4A16, per-part globals) the model's Nvfp4DenseColMerged expects. Fails on the old
    code: experts were emitted as dense weights and shared_expert gate/up stayed split."""
    import torch

    import freetoken.models.qwen3_5_moe.weight as w
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.qwen3_5_moe.weight import iter_weights

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    L, E, H, I, Q, KV, SHARED, VOCAB = 2, 4, 32, 16, 32, 16, 8, 32
    p = "model.language_model."
    fp8 = torch.float8_e4m3fn
    tensors = {}

    def nvfp4(base: str, o: int, i: int) -> None:
        tensors[base + ".weight_packed"] = torch.randint(0, 256, (o, i // 2), dtype=torch.uint8)
        tensors[base + ".weight_scale"] = torch.randn(o, i // 16).abs().to(fp8)
        tensors[base + ".weight_global_scale"] = torch.tensor([2.0])
        tensors[base + ".input_global_scale"] = torch.tensor([1.0])

    for li in range(L):
        lp = f"{p}layers.{li}."
        nvfp4(lp + "self_attn.q_proj", Q, H)
        nvfp4(lp + "self_attn.k_proj", KV, H)
        nvfp4(lp + "self_attn.v_proj", KV, H)
        nvfp4(lp + "self_attn.o_proj", H, Q)
        nvfp4(lp + "mlp.shared_expert.gate_proj", SHARED, H)
        nvfp4(lp + "mlp.shared_expert.up_proj", SHARED, H)
        nvfp4(lp + "mlp.shared_expert.down_proj", H, SHARED)
        tensors[lp + "mlp.gate.weight"] = torch.randn(E, H, dtype=torch.bfloat16)
        tensors[lp + "mlp.shared_expert_gate.weight"] = torch.randn(1, H, dtype=torch.bfloat16)
        tensors[lp + "input_layernorm.weight"] = torch.randn(H, dtype=torch.bfloat16)
        tensors[lp + "post_attention_layernorm.weight"] = torch.randn(H, dtype=torch.bfloat16)
        for ei in range(E):
            tensors |= _ct_expert_tensors(p, li, ei, H, I, 2.0, 2.0, 2.0)
    tensors[p + "embed_tokens.weight"] = torch.randn(VOCAB, H, dtype=torch.bfloat16)
    tensors[p + "norm.weight"] = torch.randn(H, dtype=torch.bfloat16)
    tensors["lm_head.weight"] = torch.randn(VOCAB, H, dtype=torch.bfloat16)
    _write_single_file(tmp_path, tensors)

    monkeypatch.setattr(w, "cached_load_hf_config", lambda _p: _hf_config(num_layers=L, num_experts=E, quant=_ct_nvfp4_quant()))

    loaded = dict(
        iter_weights(str(tmp_path), torch.device("cpu"), include_moe_experts=False, include_non_moe=True)
    )
    # routed experts never reach the dense pass.
    assert not any(".mlp.experts." in k for k in loaded)

    gu = "model.layers.0.mlp.shared_expert.gate_up_proj"
    lp = f"{p}layers.0."
    assert loaded[gu + ".weight"].shape == (2 * SHARED, H // 2)
    assert loaded[gu + ".weight_scale"].shape == (2 * SHARED, H // 16)
    assert loaded[gu + ".weight_global"].shape == (2 * SHARED,)
    assert torch.equal(loaded[gu + ".weight"][:SHARED],
                       tensors[lp + "mlp.shared_expert.gate_proj.weight_packed"])
    assert torch.equal(loaded[gu + ".weight"][SHARED:],
                       tensors[lp + "mlp.shared_expert.up_proj.weight_packed"])
    assert loaded[gu + ".weight_global"][0].item() == pytest.approx(0.5)  # 1 / 2.0

    dp = "model.layers.0.mlp.shared_expert.down_proj"
    assert torch.equal(loaded[dp + ".weight"],
                       tensors[lp + "mlp.shared_expert.down_proj.weight_packed"])

    qkv = "model.layers.0.self_attn.qkv_proj"
    assert loaded[qkv + ".weight"].shape == (Q + 2 * KV, H // 2)


def test_iter_weights_ct_format_only_mixed(tmp_path, monkeypatch):
    """Kwaipilot-style export (format-only + recipe): the GDN is per-tensor fp8
    (dequantized to bf16, fused in_proj), stacked experts never reach the dense pass,
    embed_tokens dequantizes to bf16, lm_head stays native NVFP4 (the model wants it),
    and the full-attn/shared-expert projections stay native."""
    import torch

    import freetoken.models.qwen3_5_moe.weight as w
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.qwen3_5_moe.weight import iter_weights

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    # _dequant_nvfp4_weight runs on CUDA; fake it (CPU bf16 of the right shape).
    monkeypatch.setattr(
        w, "_dequant_nvfp4_weight",
        lambda weight, scale, g: torch.zeros(
            weight.shape[0], weight.shape[1] * 2, dtype=torch.bfloat16
        ),
    )

    L, E, H, I, Q, KV, SHARED, VOCAB = 2, 4, 32, 16, 32, 16, 8, 64
    CONV, VAL = 24, 8  # conv_dim = 2*num_k*key_dim + value_dim; value_dim = num_v*value_head_dim
    p = "model.language_model."
    fp8 = torch.float8_e4m3fn
    tensors = {}

    def nvfp4(base: str, o: int, i: int) -> None:
        tensors[base + ".weight_packed"] = torch.randint(0, 256, (o, i // 2), dtype=torch.uint8)
        tensors[base + ".weight_scale"] = torch.randn(o, i // 16).abs().to(fp8)
        tensors[base + ".weight_global_scale"] = torch.tensor([2.0])
        tensors[base + ".input_global_scale"] = torch.tensor([1.0])

    def pt_fp8(base: str, o: int, i: int) -> None:
        # per-tensor fp8: fp8 weight + per-output-row F32 scale
        tensors[base + ".weight"] = torch.randint(-128, 128, (o, i), dtype=torch.int8).to(fp8)
        tensors[base + ".weight_scale"] = torch.rand(o).abs() + 0.5

    for li in range(L):
        lp = f"{p}layers.{li}."
        # GDN is fp8 (recipe gdn:fp8): in_proj_* + out_proj.
        pt_fp8(lp + "linear_attn.in_proj_qkv", CONV, H)
        pt_fp8(lp + "linear_attn.in_proj_z", VAL, H)
        pt_fp8(lp + "linear_attn.in_proj_b", 2, H)
        pt_fp8(lp + "linear_attn.in_proj_a", 2, H)
        pt_fp8(lp + "linear_attn.out_proj", VAL, H)
        tensors[lp + "linear_attn.conv1d.weight"] = torch.randn(CONV, 1, 2, dtype=torch.bfloat16)
        tensors[lp + "linear_attn.A_log"] = torch.randn(2, dtype=torch.float32)
        tensors[lp + "linear_attn.dt_bias"] = torch.randn(2, dtype=torch.float32)
        tensors[lp + "linear_attn.norm.weight"] = torch.randn(4, dtype=torch.bfloat16)
        # full-attn projections stay NVFP4.
        nvfp4(lp + "self_attn.q_proj", Q, H)
        nvfp4(lp + "self_attn.k_proj", KV, H)
        nvfp4(lp + "self_attn.v_proj", KV, H)
        nvfp4(lp + "self_attn.o_proj", H, Q)
        tensors[lp + "self_attn.q_norm.weight"] = torch.randn(4, dtype=torch.bfloat16)
        tensors[lp + "self_attn.k_norm.weight"] = torch.randn(4, dtype=torch.bfloat16)
        # shared expert NVFP4.
        nvfp4(lp + "mlp.shared_expert.gate_proj", SHARED, H)
        nvfp4(lp + "mlp.shared_expert.up_proj", SHARED, H)
        nvfp4(lp + "mlp.shared_expert.down_proj", H, SHARED)
        tensors[lp + "mlp.gate.weight"] = torch.randn(E, H, dtype=torch.bfloat16)
        tensors[lp + "mlp.shared_expert_gate.weight"] = torch.randn(1, H, dtype=torch.bfloat16)
        tensors[lp + "input_layernorm.weight"] = torch.randn(H, dtype=torch.bfloat16)
        tensors[lp + "post_attention_layernorm.weight"] = torch.randn(H, dtype=torch.bfloat16)
        # stacked (per-layer) experts -- must be skipped by the dense pass.
        tensors |= _ct_stacked_expert_tensors(p, li, E, H, I, 2.0)
    # embed_tokens NVFP4 (dequantized), lm_head NVFP4 (kept native), norm bf16.
    nvfp4(p + "embed_tokens", VOCAB, H)
    nvfp4("lm_head", VOCAB, H)
    tensors[p + "norm.weight"] = torch.randn(H, dtype=torch.bfloat16)
    _write_single_file(tmp_path, tensors)

    monkeypatch.setattr(
        w, "cached_load_hf_config",
        lambda _p: _hf_config(num_layers=L, num_experts=E, quant=_ct_format_only_quant()),
    )

    loaded = dict(
        iter_weights(str(tmp_path), torch.device("cpu"), include_moe_experts=False, include_non_moe=True)
    )
    # stacked experts never reach the dense pass.
    assert not any(".mlp.experts." in k for k in loaded)

    lp = f"{p}layers.0."
    # GDN: bf16 fused in_proj + bf16 out_proj (dequantized from fp8).
    in_proj = "model.layers.0.linear_attn.in_proj"
    assert loaded[in_proj + ".weight"].dtype == torch.bfloat16
    assert loaded[in_proj + ".weight"].shape == (CONV + VAL + 4, H)
    assert loaded["model.layers.0.linear_attn.out_proj.weight"].dtype == torch.bfloat16
    assert loaded["model.layers.0.linear_attn.out_proj.weight"].shape == (VAL, H)

    # embed_tokens dequantized to bf16.
    assert loaded["model.embed_tokens.weight"].dtype == torch.bfloat16
    assert loaded["model.embed_tokens.weight"].shape == (VOCAB, H)

    # lm_head kept native NVFP4 (lm_head_quant == "nvfp4").
    lh = "lm_head"
    assert loaded[lh + ".weight"].dtype == torch.uint8
    assert loaded[lh + ".weight"].shape == (VOCAB, H // 2)
    assert loaded[lh + ".weight_scale"].shape == (VOCAB, H // 16)
    assert loaded[lh + ".weight_global"].shape == (VOCAB,)

    # full-attn qkv + shared expert native.
    qkv = "model.layers.0.self_attn.qkv_proj"
    assert loaded[qkv + ".weight"].dtype == torch.uint8
    gu = "model.layers.0.mlp.shared_expert.gate_up_proj"
    assert loaded[gu + ".weight"].dtype == torch.uint8
    assert loaded[gu + ".weight"].shape == (2 * SHARED, H // 2)