"""e8m0 scale code 0xFF (NaN) is refused when the MXFP4 expert pieces are read.

The helper is exercised on its own, and through both MXFP4 readers on synthetic
checkpoints: DeepSeek-V4 ``ds_fp4`` (``F8_E8M0`` scales, one tensor per expert and
projection, as the published shards store them) and gpt-oss (``U8`` scales, one stacked
tensor per layer). No GPU, no kernel: the readers stop at the piece.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.layers.quantization import QuantKind
from safetensors.torch import save_file

E8M0 = torch.float8_e8m0fnu
ONE = 127  # e8m0 code for 2**0


@pytest.fixture(scope="module", autouse=True)
def _tp_info():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _codes(*shape: int, dtype=torch.uint8) -> torch.Tensor:
    return torch.full(shape, ONE, dtype=torch.uint8).view(dtype)


def _write(folder, shards: dict[str, dict[str, torch.Tensor]]) -> None:
    weight_map = {}
    for shard, tensors in shards.items():
        save_file(tensors, os.path.join(folder, shard))
        weight_map.update({name: shard for name in tensors})
    with open(os.path.join(folder, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)


# --------------------------------------------------------------------------- the helper


def test_one_nan_code_is_refused_naming_the_tensor_the_count_and_the_file():
    from freetoken.moe.expert_pieces import check_e8m0_scales

    codes = _codes(4, 8)
    codes[2, 5] = 0xFF
    with pytest.raises(ValueError) as info:
        check_e8m0_scales("layers.3.ffn.experts.7.w1.scale", codes, "model-00002-of-00046.safetensors")
    text = str(info.value)
    assert "layers.3.ffn.experts.7.w1.scale" in text
    assert "model-00002-of-00046.safetensors" in text
    assert "1 NaN scale code" in text and "out of 32" in text

    e8m0 = codes.view(E8M0)
    with pytest.raises(ValueError, match="carries 1 NaN scale code"):
        check_e8m0_scales("x.scale", e8m0, "f")
    codes[0, 0] = 0xFF
    with pytest.raises(ValueError, match="carries 2 NaN scale code"):
        check_e8m0_scales("x.scale", codes, "f")


def test_every_finite_code_passes_and_the_tensor_comes_back_unchanged():
    from freetoken.moe.expert_pieces import check_e8m0_scales

    codes = torch.arange(0, 255, dtype=torch.uint8).reshape(5, 51)  # 0x00 .. 0xFE, 0xFE = 2**127
    assert check_e8m0_scales("x.scale", codes, "f") is codes
    e8m0 = codes.view(E8M0)
    assert check_e8m0_scales("x.scale", e8m0, "f") is e8m0
    strided = codes[:, ::3]  # a sliced (non-contiguous) view scans too, as the gpt-oss TP slice is
    assert check_e8m0_scales("x.scale", strided, "f") is strided


# --------------------------------------------------------------------------- DeepSeek-V4 ds_fp4

DSV4_L, DSV4_E, DSV4_H, DSV4_I = 1, 2, 64, 32


def _dsv4_checkpoint(folder, *, bad: tuple[int, int, str] | None = None) -> str:
    """One MoE layer, two routed experts, plus an MTP layer the reader must skip. ``bad`` =
    (expert, proj, 'serial'|'parallel' shard tag) marks one scale byte 0xFF."""
    os.makedirs(os.path.join(folder, "inference"))
    with open(os.path.join(folder, "inference", "config.json"), "w") as f:
        json.dump({"n_layers": DSV4_L, "n_routed_experts": DSV4_E, "n_mtp_layers": 1, "dim": DSV4_H, "moe_inter_dim": DSV4_I}, f)
    shards: dict[str, dict[str, torch.Tensor]] = {"model-00001-of-00002.safetensors": {}, "model-00002-of-00002.safetensors": {}}
    for layer in range(DSV4_L + 1):  # layer DSV4_L is the MTP layer
        shard = shards["model-00001-of-00002.safetensors" if layer == 0 else "model-00002-of-00002.safetensors"]
        for e in range(DSV4_E):
            for proj, (n, k) in (("w1", (DSV4_I, DSV4_H)), ("w3", (DSV4_I, DSV4_H)), ("w2", (DSV4_H, DSV4_I))):
                base = f"layers.{layer}.ffn.experts.{e}.{proj}"
                shard[f"{base}.weight"] = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
                scale = _codes(n, k // 32)
                if layer == 0 and bad is not None and bad[:2] == (e, proj):
                    scale[n - 1, 0] = 0xFF
                shard[f"{base}.scale"] = scale.view(E8M0)
    _write(folder, shards)
    return str(folder)


@pytest.mark.parametrize("parallel", [False, True])
def test_dsv4_pieces_refuse_a_nan_scale_by_expert_and_shard(tmp_path, parallel):
    from freetoken.models.deepseek_v4.weight import iter_expert_pieces

    path = _dsv4_checkpoint(tmp_path, bad=(1, "w2", ""))
    with pytest.raises(ValueError) as info:
        list(iter_expert_pieces(path, None, QuantKind.MXFP4, parallel=parallel))
    text = str(info.value)
    assert "layers.0.ffn.experts.1.w2.scale" in text
    assert "model-00001-of-00002.safetensors" in text
    assert "1 NaN scale code" in text


@pytest.mark.parametrize("parallel", [False, True])
def test_dsv4_clean_pieces_pass_and_each_scale_tensor_is_scanned_once(tmp_path, parallel, monkeypatch):
    from freetoken.models.deepseek_v4.weight import iter_expert_pieces

    calls: list[int] = []
    real = torch.count_nonzero

    def counting(t, *a, **k):
        calls.append(t.numel())
        return real(t, *a, **k)

    monkeypatch.setattr(torch, "count_nonzero", counting)
    path = _dsv4_checkpoint(tmp_path)
    pieces = list(iter_expert_pieces(path, None, QuantKind.MXFP4, parallel=parallel))
    assert [(layer, e0, e1) for layer, e0, e1, _ in sorted(pieces, key=lambda p: p[:3])] == [(0, 0, 1), (0, 1, 2)]
    assert all(set(piece) == {"gate", "gate_scale", "up", "up_scale", "down", "down_scale"} for *_, piece in pieces)
    # one pass per scale tensor of the real layer (2 experts x 3 projections), each over exactly that
    # tensor's bytes; the weights and the MTP layer are never scanned
    assert calls == [DSV4_I * DSV4_H // 32] * (DSV4_E * 3)


# --------------------------------------------------------------------------- gpt-oss

OSS_L, OSS_E, OSS_H, OSS_I = 1, 2, 64, 32


def _gpt_oss_checkpoint(folder, *, bad: str | None = None) -> str:
    """One layer of HF gpt-oss experts: stacked ``[E, ...]`` blocks / U8 scales / biases."""
    pre = "model.layers.0.mlp.experts"
    gate_up_scales = _codes(OSS_E, 2 * OSS_I, OSS_H // 32)
    down_scales = _codes(OSS_E, OSS_H, OSS_I // 32)
    if bad == "gate_up":
        gate_up_scales[1, 3, 0] = 0xFF
    if bad == "down":
        down_scales[0, 0, 0] = 0xFF
        down_scales[1, OSS_H - 1, 0] = 0xFF
    tensors = {
        f"{pre}.gate_up_proj_blocks": torch.randint(0, 256, (OSS_E, 2 * OSS_I, OSS_H // 32, 16), dtype=torch.uint8),
        f"{pre}.gate_up_proj_scales": gate_up_scales,
        f"{pre}.gate_up_proj_bias": torch.zeros(OSS_E, 2 * OSS_I, dtype=torch.bfloat16),
        f"{pre}.down_proj_blocks": torch.randint(0, 256, (OSS_E, OSS_H, OSS_I // 32, 16), dtype=torch.uint8),
        f"{pre}.down_proj_scales": down_scales,
        f"{pre}.down_proj_bias": torch.zeros(OSS_E, OSS_H, dtype=torch.bfloat16),
        "model.embed_tokens.weight": torch.zeros(4, OSS_H, dtype=torch.bfloat16),
    }
    _write(folder, {"model-00000-of-00001.safetensors": tensors})
    return str(folder)


def _gpt_oss_config():
    return SimpleNamespace(moe_weight_format="mxfp4", moe_intermediate_size=OSS_I, num_layers=OSS_L, num_experts=OSS_E)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("bad, count", [("gate_up", 1), ("down", 2)])
def test_gpt_oss_pieces_refuse_a_nan_scale_by_tensor_and_shard(tmp_path, parallel, bad, count):
    from freetoken.models.gpt_oss.weight import iter_expert_pieces

    path = _gpt_oss_checkpoint(tmp_path, bad=bad)
    with pytest.raises(ValueError) as info:
        list(iter_expert_pieces(path, _gpt_oss_config(), QuantKind.MXFP4, parallel=parallel))
    text = str(info.value)
    assert f"model.layers.0.mlp.experts.{bad}_proj_scales" in text
    assert "model-00000-of-00001.safetensors" in text
    assert f"{count} NaN scale code" in text


@pytest.mark.parametrize("parallel", [False, True])
def test_gpt_oss_clean_pieces_pass(tmp_path, parallel):
    from freetoken.models.gpt_oss.weight import iter_expert_pieces

    path = _gpt_oss_checkpoint(tmp_path)
    pieces = list(iter_expert_pieces(path, _gpt_oss_config(), QuantKind.MXFP4, parallel=parallel))
    assert [(layer, e0, e1) for layer, e0, e1, _ in pieces] == [(0, 0, OSS_E)]
    assert set(pieces[0][3]) == {"gate_up", "gate_up_scale", "gate_up_bias", "down", "down_scale", "down_bias"}
