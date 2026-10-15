"""FTW conversion stores each dense tensor in the dtype its model parameter declares.

Regression for the DR-04 follow-up review: the converter first relied on the reader's bf16
pre-cast, then briefly blanket-applied EngineConfig.dtype to every dense tensor -- both
corrupted the per-tensor fp32 stalwarts (attention sinks, router selection bias) that the
reader preserves on purpose (moe.py: "Keep selection bias in fp32: rounding can change the
selected experts"). The conversion now mirrors the engine's load-time materialization
(engine.py ``_materialize_loaded_weight_state_dict``): one conversion per tensor, to the
dtype the model declares for that parameter; undeclared keys store verbatim.

Runs the real ``convert_checkpoint`` on a tiny MiMo fixture (the unquantized dialect uses
resident banks); only the CUDA
context init is skipped, everything else (meta model build, dense load, FTW write) is real.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.checkpoint.convert import convert_checkpoint
from freetoken.checkpoint.ftw import iter_ftw_weights, load_ftw_banks_to_device

from tests.models.test_mimo_v2_weight import E, MI, _raw, _write


@pytest.fixture(scope="module")
def unquantized_checkpoint(tmp_path_factory):
    """Tiny-validation MiMo dialect (bf16 experts, no quantization_config)."""
    torch.manual_seed(23)
    raw = _raw(nvfp4_experts=False)
    return _write(tmp_path_factory.mktemp("convert_src"), raw, quantized=False, indexed=False), raw


@pytest.fixture()
def _cpu_convert(monkeypatch):
    """Skip convert_checkpoint's CUDA-context init only (the fused/CPU path needs nothing else)."""
    real_set_device = torch.cuda.set_device

    def set_device(dev):
        if getattr(dev, "type", dev) != "cpu":
            real_set_device(dev)

    monkeypatch.setattr(torch.cuda, "set_device", set_device)


def test_default_conversion_stores_the_declared_param_dtypes(unquantized_checkpoint, tmp_path, _cpu_convert):
    folder, raw = unquantized_checkpoint
    out = tmp_path / "ftw_bf16"
    index = convert_checkpoint(folder, str(out), moe_backend="fused", device="cpu")
    assert index["counts"]["weight"] > 0
    stored = dict(iter_ftw_weights(str(out)))
    # (a) attention sinks: fp32 end to end (a blanket EngineConfig.dtype cast rounds them)
    sinks = stored["model.layers.1.self_attn.sinks"]
    assert sinks.dtype is torch.float32
    assert torch.equal(sinks, raw["model.layers.1.self_attn.attention_sink_bias"].to(torch.float32))
    # (b) router selection bias: fp32 end to end
    bias = stored["model.layers.1.mlp.e_score_correction_bias"]
    assert bias.dtype is torch.float32
    assert torch.equal(bias, raw["model.layers.1.mlp.gate.e_score_correction_bias"])
    # (c) F32 qkv: one rounding to the declared (model) dtype
    qkv = stored["model.layers.0.self_attn.qkv_proj.weight"]
    assert qkv.dtype is torch.bfloat16
    assert torch.equal(qkv, raw["model.layers.0.self_attn.qkv_proj.weight"].to(torch.bfloat16))
    # (d) bf16-sourced dense tensor: bit-identical (no-op conversion)
    embed = stored["model.embed_tokens.weight"]
    assert embed.dtype is torch.bfloat16
    assert torch.equal(embed, raw["model.embed_tokens.weight"])
    # Experts are bank entries, not dense weights; dtype and row order stay exact.
    assert not any(".experts." in key for key in stored)
    banks = load_ftw_banks_to_device(str(out), num_layers=2, device=torch.device("cpu"))
    gate_up = banks.sources["gate_up"][0]
    assert gate_up.dtype is torch.bfloat16
    for e in range(E):
        m = f"model.layers.1.mlp.experts.{e}"
        assert torch.equal(gate_up[e, :MI], raw[f"{m}.gate_proj.weight"])
        assert torch.equal(gate_up[e, MI:], raw[f"{m}.up_proj.weight"])


def test_fp32_conversion_keeps_f32_linears_value_identical(unquantized_checkpoint, tmp_path, _cpu_convert):
    folder, raw = unquantized_checkpoint
    out = tmp_path / "ftw_fp32"
    convert_checkpoint(folder, str(out), dtype=torch.float32, moe_backend="fused", device="cpu")
    stored = dict(iter_ftw_weights(str(out)))
    qkv = stored["model.layers.0.self_attn.qkv_proj.weight"]
    assert qkv.dtype is torch.float32
    assert torch.equal(qkv, raw["model.layers.0.self_attn.qkv_proj.weight"])  # identity, no rounding
    # the fp32 stalwarts stay fp32 under a non-default dtype too
    assert stored["model.layers.1.self_attn.sinks"].dtype is torch.float32
    assert stored["model.layers.1.mlp.e_score_correction_bias"].dtype is torch.float32
    # a bf16 source upcasts to the fp32 model dtype exactly as the engine load would
    embed = stored["model.embed_tokens.weight"]
    assert embed.dtype is torch.float32
    assert torch.equal(embed, raw["model.embed_tokens.weight"].to(torch.float32))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_resident_conversion_and_replay_never_preround_experts(tmp_path, _cpu_convert, dtype):
    from tests.models.test_mimo_v2_weight import _raw_dr04, _DR04_EXPERT_KEY, _build_model, _engine_state

    raw = _raw_dr04(nvfp4=False)
    (tmp_path / "source").mkdir()
    folder = _write(tmp_path / "source", raw, quantized=False, indexed=False)
    out = tmp_path / "ftw"
    convert_checkpoint(folder, str(out), dtype=dtype, moe_backend="fused", device="cpu")
    banks = load_ftw_banks_to_device(str(out), num_layers=2, device=torch.device("cpu"))
    gate = banks.sources["gate_up"][0][0, :MI]
    assert gate.dtype is dtype
    assert torch.isfinite(gate).all()
    assert torch.equal(gate, raw[_DR04_EXPERT_KEY].to(dtype))
    model = _build_model(str(out), "fused", device=torch.device("cpu"), dtype=dtype)
    state = _engine_state(str(out), dtype, "fused", model=model)
    assert state["model.layers.1.self_attn.sinks"].dtype is torch.float32
    assert state["model.layers.1.mlp.e_score_correction_bias"].dtype is torch.float32
    assert torch.equal(model.model.layers.op_list[1].mlp.experts._expert_banks["gate_up"][0, :MI], gate)


def test_fp32_ftw_bank_replays_directly_to_fp16(tmp_path, _cpu_convert):
    from tests.models.test_mimo_v2_weight import _raw_dr04, _DR04_EXPERT_KEY, _build_model, _engine_state

    raw = _raw_dr04(nvfp4=False)
    (tmp_path / "source").mkdir()
    folder = _write(tmp_path / "source", raw, quantized=False, indexed=False)
    out = tmp_path / "ftw"
    convert_checkpoint(folder, str(out), dtype=torch.float32, moe_backend="fused", device="cpu")
    model = _build_model(str(out), "fused", device=torch.device("cpu"), dtype=torch.float16)
    _engine_state(str(out), torch.float16, "fused", model=model)
    gate = model.model.layers.op_list[1].mlp.experts._expert_banks["gate_up"][0, :MI]
    assert gate.dtype is torch.float16 and torch.isfinite(gate).all()
    assert torch.equal(gate, raw[_DR04_EXPERT_KEY].to(torch.float16))
