"""qwen4_exp weight loading against synthetic checkpoints shaped like the released ones.

The tensors are tiny but the key names, dtypes and the fusion geometry that matters
(hc_lowrank=320 + hc_count=4 -> a 12-row zero pad; 128-row block scales) are the real ones.
"""

from __future__ import annotations

import dataclasses
import json
import random
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kernel.aot_models import SUPPORTED_MODELS, expert_bank_row_bytes
import freetoken.models.qwen4_exp.weight as qwen4_exp_weight
from freetoken.models.nvfp4_banks import decode_csf_scales, iter_nvfp4_expert_pieces
from freetoken.models.qwen4_exp.weight import (
    _ZERO_CENTERED_NORM_SUFFIXES,
    _DenseReader,
    iter_weights,
    load_ple_table,
    nvfp4_expert_spec,
)
from freetoken.models.register import get_model_spec
from freetoken.moe.host_banks import HostBank, read_range_into

from .common import LM, RADIXARK_NVFP4, hf_config, install_quant_config, meta_state_dict, mixed_precision_quant

H = 128  # hidden_size; every block-fp8 projection needs in/out multiples of 128
HC = 4  # hc_count
LR = 320  # hc_lowrank; kept real so the merged HC pad is the real (-(320+4)) % 16 = 12
HCH = HC * H  # hyper-connection stream width
KH, VH, HD = 2, 4, 32  # GDN key / value heads, head dim: qkv rows 256, z rows 128
QH, KVH, AHD = 4, 2, 64  # QSA q / kv heads, head dim: q rows 512, k / v rows 128
IHD = 64  # indexer head dim
BLOCK = 128
E, I = 3, 6  # routed experts, moe_intermediate_size
NGRAM_DIM, NGRAM_ROWS, NGRAM_SHARDS = 4, 7, 4


@pytest.fixture(scope="session", autouse=True)
def _tp_info():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _bf16(*shape: int) -> torch.Tensor:
    return torch.randn(*shape).to(torch.bfloat16)


def _hc_weights(prefix: str, inject: bool) -> dict[str, torch.Tensor]:
    w = {
        f"{prefix}.hc_norm.weight": _bf16(HCH),
        f"{prefix}.input_mix_weight_down.weight": _bf16(LR, HCH),
        f"{prefix}.input_mix_weight_up.weight": _bf16(HCH, LR),
    }
    if inject:
        w[f"{prefix}.block_inject_weight.weight"] = _bf16(HC, HCH)
    return w


def _fp8_scale(weight: torch.Tensor) -> torch.Tensor:
    return torch.rand(weight.shape[0] // BLOCK, weight.shape[1] // BLOCK) + 0.5


def _csf_target(rows: int, cols: int, seed: int = 0) -> torch.Tensor:
    """An E4M3 scale matrix whose rows share a high nibble (so base + 4-bit offset is exact)."""
    rng = random.Random(seed)
    high = torch.tensor([[rng.randrange(16)] for _ in range(rows)], dtype=torch.uint8)
    off = torch.tensor([[rng.randrange(16) for _ in range(cols)] for _ in range(rows)], dtype=torch.uint8)
    return high << 4 | off


def _csf_encode(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CSF slab pair of ``target``: fixed slabs (per-row high-nibble bases + 4-bit offsets) and a
    few absolute exception records. The reference the decoder test compares against."""
    rows, cols = target.shape
    assert rows % 16 == 0 and cols % 2 == 0
    row_bytes = cols // 2
    fixed = torch.zeros(rows // 16, 16 * (1 + row_bytes), dtype=torch.uint8)
    exceptions: list[int] = []
    for r in range(rows):
        slab, row = divmod(r, 16)
        base = target[r, 0] & 0xF0
        fixed[slab, row] = base
        for j in range(row_bytes):
            fixed[slab, 16 + row * row_bytes + j] = (target[r, 2 * j + 1] & 0xF) << 4 | (target[r, 2 * j] & 0xF)
    # absolute records: (value << 24) | row-major position, U32 bit patterns stored as int32
    for pos in (3, rows * cols - 1):
        value = 0xFF - target.ravel()[pos].item()
        rec = (value << 24) | pos
        exceptions.append(rec - 0x100000000 if rec >= 0x80000000 else rec)
    exc = torch.tensor(exceptions, dtype=torch.int32)
    ref = target.clone()
    for rec in exceptions:
        ref.view(-1)[rec & 0xFFFFFF] = (rec >> 24) & 0xFF
    return fixed, exc, ref


# --------------------------------------------------------------------------- CSF slab scales


def test_csf_decode_roundtrip():
    for rows, cols in ((32, 8), (128, 160), (40 * 16, 40)):
        target = _csf_target(rows, cols, seed=rows)
        fixed, exc, ref = _csf_encode(target)
        out = decode_csf_scales(fixed, exc)
        assert out.shape == (rows, cols) and out.dtype is torch.float8_e4m3fn
        assert torch.equal(out.view(torch.uint8), ref)
        # the same records as a U32 byte stream
        assert torch.equal(decode_csf_scales(fixed, exc.view(torch.uint8)).view(torch.uint8), ref)
        # the same records as their true uint32 dtype (sign-bit records included)
        assert torch.equal(decode_csf_scales(fixed, exc.to(torch.uint32)).view(torch.uint8), ref)
        # with no exceptions the slab decode is the target itself (rows share a high nibble)
        no_exc = decode_csf_scales(fixed, torch.zeros(0, dtype=torch.int32))
        assert torch.equal(no_exc.view(torch.uint8), target)


def test_csf_decode_rejects_streams_of_the_wrong_dtype():
    target = _csf_target(32, 8, seed=1)
    fixed, exc, _ref = _csf_encode(target)
    with pytest.raises(ValueError, match="U32 records"):
        decode_csf_scales(fixed, exc.to(torch.float32))
    with pytest.raises(ValueError, match="expected U8"):
        decode_csf_scales(fixed.to(torch.int32), exc)


def test_rename_scaled_maps_the_checkpoint_scale_names_to_the_model_roles(monkeypatch):
    """FP4-Local quantises the vision tower: attention as MXFP8, the MLP as W4A16 NVFP4.

    The checkpoint names the block scale ``weight_scale`` while the engine's buffers are
    ``weight_scale_inv`` (and the per-tensor global ``weight_scale_2``), so the reader must
    rename along the module prefix -- and only for modules the quant config actually names.
    """
    from freetoken.layers.quantization import ModelOptConfig

    quant = ModelOptConfig({
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "model.visual.blocks.0.attn.qkv": {"quant_algo": "MXFP8"},
            "model.visual.blocks.0.mlp.linear_fc1": {"quant_algo": "W4A16_NVFP4"},
            "model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4"},
        },
    })
    monkeypatch.setattr(qwen4_exp_weight, "get_quant_config", lambda: quant)

    # MXFP8 block scale: checkpoint role name -> the engine's buffer name
    got = qwen4_exp_weight._rename_scaled("model.visual.blocks.0.attn.qkv.weight_scale")
    assert got == "visual.blocks.0.attn.qkv.weight_scale_inv"
    # W4A16 per-tensor global: the vision prefix lands under visual., the role under weight_global
    got = qwen4_exp_weight._rename_scaled("model.visual.blocks.0.mlp.linear_fc1.weight_scale_2")
    assert got == "visual.blocks.0.mlp.linear_fc1.weight_global"
    # a scale whose module the config leaves unquantized stays noise
    assert qwen4_exp_weight._rename_scaled("model.visual.blocks.1.attn.qkv.weight_scale") is None
    # routed experts read their scales through the bank readers, never the dense path
    assert qwen4_exp_weight._rename_scaled(
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale.nvfp4_csf_fixed"
    ) is None


def test_global_expander_broadcasts_the_scalar_once_its_weight_is_seen():
    """The W4A16 global is a scalar fp32; the model buffer is one fp32 entry per output row.

    The pair may arrive in either order, so the expansion is flushed only once both are known.
    """
    g = qwen4_exp_weight._GlobalExpander()
    module = "visual.blocks.0.mlp.linear_fc1"
    scalar = torch.tensor(0.0625)
    # global first: buffered until the weight fixes the row count
    assert g.on_global(module + ".weight_global", scalar) is None
    name, tensor = g.on_weight(module, 256)
    assert name == module + ".weight_global"
    assert tensor.shape == (256,) and tensor.dtype is torch.float32
    assert torch.all(tensor == 0.0625)
    # weight first: the global flushes on the spot
    g2 = qwen4_exp_weight._GlobalExpander()
    assert g2.on_weight(module, 64) is None
    name, tensor = g2.on_global(module + ".weight_global", scalar)
    assert name == module + ".weight_global" and tensor.shape == (64,)
    # a full-width global passes through untouched; a miscounted one raises
    full = torch.ones(64, dtype=torch.float16)
    assert g2.on_global(module + ".weight_global", full) == (module + ".weight_global", full)
    row = torch.arange(3, dtype=torch.float16)
    with pytest.raises(ValueError, match="holds 3 elements"):
        g2.on_global(module + ".weight_global", row)
    g.check_complete()
    g2.check_complete()
    # a scalar that never met its weight fails the completeness assert
    g3 = qwen4_exp_weight._GlobalExpander()
    g3.on_global(module + ".weight_global", scalar)
    with pytest.raises(ValueError, match="without a weight"):
        g3.check_complete()


def _csf_expert_raw() -> tuple[dict[str, torch.Tensor], dict[tuple[int, str], torch.Tensor]]:
    """One CSF-quantised MoE layer: packed codes, slab scale pairs, globals and input scales."""
    lm = "model.language_model"
    H_, I_ = 128, 64  # hidden / expert width; K multiples of 16, rows multiples of 16
    raw: dict[str, torch.Tensor] = {}
    refs: dict[tuple[int, str], torch.Tensor] = {}
    for expert in range(2):
        for proj, rows, k in (("gate_proj", I_, H_), ("up_proj", I_, H_), ("down_proj", H_, I_)):
            base = f"{lm}.layers.0.mlp.experts.{expert}.{proj}"
            raw[f"{base}.weight"] = torch.randint(0, 256, (rows, k // 2), dtype=torch.uint8)
            raw[f"{base}.weight_scale_2"] = torch.rand(()) + 0.5
            raw[f"{base}.input_scale"] = torch.rand(()) + 0.5
            fixed, exc, ref = _csf_encode(_csf_target(rows, k // 16, seed=expert + k))
            raw[f"{base}.weight_scale.nvfp4_csf_fixed"] = fixed
            raw[f"{base}.weight_scale.nvfp4_csf_exceptions"] = exc
            refs[(expert, proj)] = ref
    return raw, refs


def test_nvfp4_expert_spec_selects_the_csf_checkpoint(tmp_path):
    """A weight map carrying CSF slab pairs switches the spec onto the shared decoder."""
    quant = mixed_precision_quant(gdn_layers=(), attn_layers=(), moe_layers=(0,))
    raw, _refs = _csf_expert_raw()
    (tmp_path / "csf").mkdir()
    folder, _full = _write_checkpoint(tmp_path / "csf", raw, quant)
    install_quant_config(folder)
    spec = nvfp4_expert_spec(folder, None)
    assert spec.scale_pair == qwen4_exp_weight._CSF_KINDS
    assert spec.scale_decoder is qwen4_exp_weight.decode_csf_scales
    assert "CSF" in spec.desc
    plain = {k.replace(".weight_scale.nvfp4_csf_fixed", ".weight_scale"): v for k, v in raw.items()}
    (tmp_path / "plain").mkdir()
    folder, _full = _write_checkpoint(tmp_path / "plain", plain, quant)
    install_quant_config(folder)
    spec = nvfp4_expert_spec(folder, None)
    assert spec.scale_pair is None and spec.scale_decoder is None


def test_csf_expert_reader_expands_the_scale_pieces(tmp_path):
    """A CSF checkpoint streams slab pairs; the reader expands them into the plain bank roles."""
    from freetoken.models.loader import drop_page_cache

    lm = "model.language_model"
    raw, refs = _csf_expert_raw()
    quant = mixed_precision_quant(gdn_layers=(), attn_layers=(), moe_layers=(0,))
    (tmp_path / "reader").mkdir()
    folder, _full = _write_checkpoint(tmp_path / "reader", raw, quant)
    install_quant_config(folder)
    spec = nvfp4_expert_spec(folder, None)
    assert spec.scale_pair is not None
    config = SimpleNamespace(num_experts=2, num_layers=1, first_k_dense_replace=0)
    pieces = list(iter_nvfp4_expert_pieces(folder, config, spec, drop_page_cache=drop_page_cache))
    assert sorted(p[1] for p in pieces) == [0, 1]
    for layer, e0, e1, piece in pieces:
        assert (layer, e0, e1) == (0, e0, e0 + 1)
        for proj, role in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
            base = f"{lm}.layers.0.mlp.experts.{e0}.{proj}"
            assert torch.equal(piece[role][0], raw[f"{base}.weight"])
            assert piece[f"{role}_global"].dtype is torch.float32 and torch.equal(
                piece[f"{role}_global"].reshape(-1), raw[f"{base}.weight_scale_2"].reshape(-1).float()
            )
            assert piece[f"{role}_input"].dtype is torch.float32 and torch.equal(
                piece[f"{role}_input"].reshape(-1), raw[f"{base}.input_scale"].reshape(-1).float()
            )
            assert piece[f"{role}_scale"].shape == (1, raw[f"{base}.weight"].shape[0], raw[f"{base}.weight"].shape[1] // 8)
            assert torch.equal(piece[f"{role}_scale"].view(torch.uint8)[0], refs[(e0, proj)])


def test_csf_pieces_decoder_errors_surface_to_the_consumer(tmp_path):
    """A raising scale decoder must fail the stream, not silently drop pieces."""
    raw, _refs = _csf_expert_raw()
    quant = mixed_precision_quant(gdn_layers=(), attn_layers=(), moe_layers=(0,))
    folder, _full = _write_checkpoint(tmp_path, raw, quant)
    install_quant_config(folder)
    spec = nvfp4_expert_spec(folder, None)
    calls = iter(range(99))

    def boom(fixed, exceptions):
        if next(calls) == 4:
            raise RuntimeError("decoder boom")
        return decode_csf_scales(fixed, exceptions)

    spec = dataclasses.replace(spec, scale_decoder=boom)
    config = SimpleNamespace(num_experts=2, num_layers=1, first_k_dense_replace=0)
    with pytest.raises(RuntimeError, match="decoder boom"):
        list(iter_nvfp4_expert_pieces(folder, config, spec))


def test_csf_expert_reader_parallel_streams_csf_pieces(tmp_path):
    """The O_DIRECT parallel reader (the production path) expands CSF pairs identically."""
    raw, refs = _csf_expert_raw()
    quant = mixed_precision_quant(gdn_layers=(), attn_layers=(), moe_layers=(0,))
    folder, _full = _write_checkpoint(tmp_path, raw, quant)
    install_quant_config(folder)
    spec = nvfp4_expert_spec(folder, None)
    config = SimpleNamespace(num_experts=2, num_layers=1, first_k_dense_replace=0)
    pieces = list(iter_nvfp4_expert_pieces(folder, config, spec, parallel=True, workers=2, chunk=1 << 20))
    assert sorted(p[1] for p in pieces) == [0, 1]
    for _layer, e0, e1, piece in pieces:
        assert (e0, e1) == (0, 1) or (e0, e1) == (1, 2)
        for proj, role in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
            base = f"model.language_model.layers.0.mlp.experts.{e0}.{proj}"
            assert piece[f"{role}_scale"].shape == (1, raw[f"{base}.weight"].shape[0], raw[f"{base}.weight"].shape[1] // 8)
            assert torch.equal(piece[f"{role}_scale"].view(torch.uint8)[0], refs[(e0, proj)])


def test_csf_pieces_read_activation_scales_out_of_band(tmp_path, monkeypatch):
    """input_scale rides no part of the expert stream.

    The FP4-Local checkpoint keeps every input_scale in one trailing shard; streaming
    them made the first piece wait on that shard with all weights already in hand.
    """
    from freetoken.models import weight as models_weight

    probed: list = []
    real = models_weight.iter_expert_tensors_parallel

    def spy(folder, is_expert, **kwargs):
        probed.append(is_expert)
        yield from real(folder, is_expert, **kwargs)

    monkeypatch.setattr(models_weight, "iter_expert_tensors_parallel", spy)
    quant = mixed_precision_quant(gdn_layers=(), attn_layers=(), moe_layers=(0,))
    raw, _refs = _csf_expert_raw()
    # mimic the real checkpoint: the whole input_scale set lives in its own shard
    act_tensors = {name: raw.pop(name) for name in list(raw) if name.endswith(".input_scale")}
    assert len(act_tensors) == 6
    folder, _full = _write_checkpoint(tmp_path, raw, quant)
    save_file(act_tensors, str(tmp_path / "model-act-00000.safetensors"))
    assert folder == str(tmp_path)
    install_quant_config(folder)
    spec = nvfp4_expert_spec(folder, None)
    config = SimpleNamespace(num_experts=2, num_layers=1, first_k_dense_replace=0)
    pieces = list(iter_nvfp4_expert_pieces(folder, config, spec, parallel=True, workers=2, chunk=1 << 20))
    assert len(pieces) == 2
    for _layer, _e0, _e1, piece in pieces:
        for role in ("gate", "up", "down"):
            assert piece[f"{role}_input"].dtype is torch.float32
    assert probed
    want = probed[0]
    for expert in range(2):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            base = f"model.language_model.layers.0.mlp.experts.{expert}.{proj}"
            assert not want(f"{base}.input_scale")
            assert want(f"{base}.weight")


def _raw_checkpoint(dense_fp8: bool = False, channel_fp8: bool = False, mxfp8_scale: str | None = None) -> dict[str, torch.Tensor]:
    """Layer 0 = GDN + PLE, layer 1 = QSA; plus the mtp / visual / routed-expert noise.

    ``dense_fp8`` stores the attention and GDN qkv|z / out projections as 128x128 block-fp8 (e4m3 ``.weight`` + fp32 ``.weight_scale_inv``) like the community NVFP4-FP8 requants.
    ``channel_fp8`` stores those and the shared experts as per-channel fp8 (e4m3 ``.weight`` + bf16 ``[rows, 1]`` ``.weight_scale``) like the llm-compressor requants.
    ``mxfp8_scale`` stores the attention and GDN qkv|z / out projections as MXFP8 (e4m3 ``.weight`` + uint8 e8m0 codes per 32 inputs) under that scale suffix.
    """
    lm = "model.language_model"
    raw: dict[str, torch.Tensor] = {
        f"{lm}.embed_tokens.weight": _bf16(11, H),
        "lm_head.weight": _bf16(11, H),
    }
    raw.update(_hc_weights(f"{lm}.hyper_connection_mixer", inject=False))
    for layer in (0, 1):
        raw.update(_hc_weights(f"{lm}.layers.{layer}.attn_hyper_connection", inject=True))
        raw.update(_hc_weights(f"{lm}.layers.{layer}.mlp_hyper_connection", inject=True))
        raw.update({
            f"{lm}.layers.{layer}.mlp.gate.weight": _bf16(E, H),
            f"{lm}.layers.{layer}.mlp.shared_expert.gate_proj.weight": _bf16(I, H),
            f"{lm}.layers.{layer}.mlp.shared_expert.up_proj.weight": _bf16(I, H),
            f"{lm}.layers.{layer}.mlp.shared_expert.down_proj.weight": _bf16(H, I),
            f"{lm}.layers.{layer}.mlp.shared_expert_gate.weight": _bf16(1, H),
        })
        for expert in range(E):
            base = f"{lm}.layers.{layer}.mlp.experts.{expert}"
            for proj, out, inn in (("gate_proj", I, H), ("up_proj", I, H), ("down_proj", H, I)):
                raw[f"{base}.{proj}.weight"] = torch.randint(
                    0, 256, (out, inn // 2), dtype=torch.uint8
                )
                raw[f"{base}.{proj}.weight_scale"] = torch.ones(
                    out, inn // 16 or 1, dtype=torch.float8_e4m3fn
                )
                raw[f"{base}.{proj}.weight_scale_2"] = torch.tensor(0.5)
                raw[f"{base}.{proj}.input_scale"] = torch.tensor(0.25)
    gdn = f"{lm}.layers.0.linear_attn"
    raw.update({
        f"{gdn}.in_proj_qkv.weight": _bf16(2 * KH * HD + VH * HD, H),
        f"{gdn}.in_proj_z.weight": _bf16(VH * HD, H),
        f"{gdn}.in_proj_b.weight": _bf16(VH, H),
        f"{gdn}.in_proj_a.weight": _bf16(VH, H),
        f"{gdn}.conv1d.weight": _bf16(2 * KH * HD + VH * HD, 1, 4),
        f"{gdn}.A_log": _bf16(VH),
        f"{gdn}.dt_bias": _bf16(VH),
        f"{gdn}.norm.weight": _bf16(HD),
        f"{gdn}.out_proj.weight": _bf16(H, VH * HD),
    })
    ple = f"{lm}.layers.0.ple"
    raw.update({
        f"{ple}.key_proj.weight": _bf16(HCH, H),
        f"{ple}.value_proj.weight": _bf16(H, H),
        f"{ple}.norm_key.weight": _bf16(HCH),
        f"{ple}.norm_query.weight": _bf16(HCH),
        f"{ple}.norm_conv.weight": _bf16(HCH),
        f"{ple}.conv1d.weight": _bf16(HCH, 1, 4),
        f"{ple}.ple_embedding.layer_multipliers": torch.randint(1, 1 << 40, (3,)),
        f"{ple}.ple_embedding.ngram_heads_offsets": torch.arange(4),
        f"{ple}.ple_embedding.ngram_heads_vocab_sizes": torch.full((4,), 5),
    })
    attn = f"{lm}.layers.1.self_attn"
    raw.update({
        f"{attn}.q_proj.weight": _bf16(2 * QH * AHD, H),
        f"{attn}.k_proj.weight": _bf16(KVH * AHD, H),
        f"{attn}.v_proj.weight": _bf16(KVH * AHD, H),
        f"{attn}.o_proj.weight": _bf16(H, QH * AHD),
        f"{attn}.q_norm.weight": _bf16(AHD),
        f"{attn}.k_norm.weight": _bf16(AHD),
        f"{attn}.indexer.index_qk_proj.weight": _bf16(5 * IHD, H),
        f"{attn}.indexer.q_layernorm.weight": _bf16(IHD),
        f"{attn}.indexer.k_layernorm.weight": _bf16(IHD),
    })
    raw.update({
        "mtp.hyper_connection_mixer.hc_norm.weight": _bf16(HCH),
        "mtp.layers.0.self_attn.q_proj.weight": _bf16(2 * QH * AHD, H),
        "mtp.layers.0.mlp.experts.gate_up_proj": _bf16(E, 2 * I, H),
        "mtp.layers.0.mlp.experts.down_proj": _bf16(E, H, I),
        "model.visual.blocks.0.attn.qkv.weight": _bf16(3 * H, H),
        "model.visual.merger.norm.weight": _bf16(H),
    })
    if dense_fp8:
        for module in (f"{gdn}.in_proj_qkv", f"{gdn}.in_proj_z", f"{gdn}.out_proj",
                       *(f"{attn}.{p}_proj" for p in "qkvo")):
            weight = raw[f"{module}.weight"]
            raw[f"{module}.weight"] = weight.to(torch.float8_e4m3fn)
            raw[f"{module}.weight_scale_inv"] = _fp8_scale(weight)
    if channel_fp8:
        for module in _channel_fp8_modules(gdn_layers=(0,), attn_layers=(1,), mlp_layers=(0, 1)):
            weight = raw[f"{module}.weight"]
            raw[f"{module}.weight"] = weight.to(torch.float8_e4m3fn)
            raw[f"{module}.weight_scale"] = (torch.rand(weight.shape[0], 1) + 0.5).to(torch.bfloat16)
    if mxfp8_scale is not None:
        for module in (f"{gdn}.in_proj_qkv", f"{gdn}.in_proj_z", f"{gdn}.out_proj", *(f"{attn}.{p}_proj" for p in "qkvo")):
            weight = raw[f"{module}.weight"]
            raw[f"{module}.weight"] = weight.to(torch.float8_e4m3fn)
            raw[f"{module}.{mxfp8_scale}"] = torch.randint(120, 135, (weight.shape[0], weight.shape[1] // 32), dtype=torch.uint8)
    return raw


FP8_DENSE_QUANT = mixed_precision_quant(gdn_layers=(0,), attn_layers=(1,), moe_layers=(0, 1))


# ModelOpt MIXED_PRECISION with MXFP8 attention / GDN projections, as in nvidia/Qwen3.8-Flash-Next-FP4-Local
MXFP8_DENSE_QUANT = {
    **FP8_DENSE_QUANT,
    "quantized_layers": {
        k: ({"quant_algo": "MXFP8", "group_size": 32} if v["quant_algo"] == "FP8_PB_WO" else v)
        for k, v in FP8_DENSE_QUANT["quantized_layers"].items()
    },
}


def _channel_fp8_modules(gdn_layers, attn_layers, mlp_layers) -> list[str]:
    return (
        [f"{LM}.layers.{i}.linear_attn.{p}" for i in gdn_layers for p in ("in_proj_qkv", "in_proj_z", "out_proj")]
        + [f"{LM}.layers.{i}.self_attn.{p}_proj" for i in attn_layers for p in "qkvo"]
        + [f"{LM}.layers.{i}.mlp.shared_expert.{p}_proj" for i in mlp_layers for p in ("gate", "up", "down")]
    )


# llm-compressor mixed-precision as in orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4: per-channel fp8 dense, weight-only NVFP4 experts
CT_CHANNEL_FP8_QUANT = {
    "quant_method": "compressed-tensors",
    "format": "mixed-precision",
    "quantization_status": "compressed",
    "config_groups": {
        "group_0": {
            "format": "naive-quantized",
            "targets": _channel_fp8_modules(gdn_layers=(0,), attn_layers=(1,), mlp_layers=(0, 1)),
            "weights": {"num_bits": 8, "type": "float", "strategy": "channel", "symmetric": True, "dynamic": False},
            "input_activations": None,
        },
        "group_1": {
            "format": "nvfp4-pack-quantized",
            "targets": [r"re:.*mlp\.experts\..*proj$"],
            "weights": {"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16, "symmetric": True, "dynamic": False},
            "input_activations": None,
        },
    },
    "ignore": ["lm_head"],
}


def _config_json(quantization_config) -> dict:
    cfg = hf_config(
        num_layers=2, head_dim=AHD, num_q=QH, num_kv=KVH, index_head_dim=IHD, index_heads=2,
        budget=16, hidden=H, max_position=4096, rope_theta=10000.0,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=KH, linear_num_value_heads=VH,
        linear_key_head_dim=HD, linear_value_head_dim=HD,
        hc_lowrank=LR, ple_layer_ids=[1],
        num_experts=E, moe_intermediate_size=I, shared_expert_intermediate_size=I,
    )
    return {**vars(cfg), "text_config": vars(cfg.text_config), "quantization_config": quantization_config}


def _ngram_table() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    shards = {
        f"{prefix}.shard_{i}.weight": (
            torch.arange(i * NGRAM_ROWS * NGRAM_DIM, (i + 1) * NGRAM_ROWS * NGRAM_DIM)
            .remainder(200).to(torch.uint8).view(NGRAM_ROWS, NGRAM_DIM).view(torch.float8_e4m3fn)
        )
        for i in range(NGRAM_SHARDS)
    }
    scale = torch.tensor([0.125], dtype=torch.bfloat16)
    shards[f"{prefix}.weight_scale"] = scale
    return shards, scale


def _write_checkpoint(folder, raw: dict[str, torch.Tensor], quantization_config) -> tuple[str, dict[str, torch.Tensor]]:
    table, _scale = _ngram_table()
    # Spread the dense tensors over two shards so the fusion buffer has to survive a file
    # boundary, and put the n-gram table in its own shards like the real checkpoint does.
    names = sorted(raw)
    save_file({n: raw[n] for n in names[::2]}, str(folder / "model-bf16-00001.safetensors"))
    save_file({n: raw[n] for n in names[1::2]}, str(folder / "model-bf16-00002.safetensors"))
    shard_names = sorted(table)
    save_file({n: table[n] for n in shard_names[:2]}, str(folder / "model-plefp8-00000.safetensors"))
    save_file({n: table[n] for n in shard_names[2:]}, str(folder / "model-plefp8-00001.safetensors"))
    (folder / "config.json").write_text(json.dumps(_config_json(quantization_config)))
    return str(folder), {**raw, **table}


def _load(folder: str, *, vision: bool = True) -> dict[str, torch.Tensor]:
    install_quant_config(folder)
    return {
        name: tensor.clone()
        for name, tensor in iter_weights(
            folder, torch.device("cpu"), include_moe_experts=True, include_non_moe=True, include_vision=vision
        )
    }


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> tuple[str, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    return _write_checkpoint(tmp_path_factory.mktemp("qwen4_exp_ckpt"), _raw_checkpoint(), None)


@pytest.fixture(scope="module")
def loaded(checkpoint) -> dict[str, torch.Tensor]:
    return _load(checkpoint[0])


@pytest.fixture(scope="module")
def checkpoint_fp8(tmp_path_factory) -> tuple[str, dict[str, torch.Tensor]]:
    torch.manual_seed(1)
    return _write_checkpoint(
        tmp_path_factory.mktemp("qwen4_exp_fp8_ckpt"), _raw_checkpoint(dense_fp8=True), FP8_DENSE_QUANT
    )


@pytest.fixture(scope="module")
def loaded_fp8(checkpoint_fp8) -> dict[str, torch.Tensor]:
    return _load(checkpoint_fp8[0])


@pytest.fixture(scope="module")
def checkpoint_fp8_channel(tmp_path_factory) -> tuple[str, dict[str, torch.Tensor]]:
    torch.manual_seed(3)
    return _write_checkpoint(
        tmp_path_factory.mktemp("qwen4_exp_fp8_channel_ckpt"), _raw_checkpoint(channel_fp8=True), CT_CHANNEL_FP8_QUANT
    )


def test_tower_keys_come_out_under_the_prefix_load_weight_filters(loaded):
    assert {n for n in loaded if "visual" in n} == {"visual.blocks.0.attn.qkv.weight", "visual.merger.norm.weight"}


def test_mtp_experts_and_table_never_loaded(loaded):
    for name in loaded:
        assert not name.startswith("mtp.")
        assert ".mlp.experts." not in name
        assert "ngram_embedding" not in name
        assert not name.endswith((".weight_scale", ".weight_scale_2", ".input_scale", ".weight_scale_inv"))


def test_hc_merge_is_down_then_inject_then_zero_pad(loaded, checkpoint):
    _folder, raw = checkpoint
    key = "model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight"
    merged = loaded[key]
    assert merged.shape == (LR + HC + 12, HCH)  # pad = (-(320 + 4)) % 16
    down = raw["model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight"]
    inject = raw["model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight"]
    assert torch.equal(merged[:LR], down)
    assert torch.equal(merged[LR:LR + HC], inject)
    assert torch.equal(merged[LR + HC:], torch.zeros(12, HCH, dtype=merged.dtype))


def test_top_level_mixer_keeps_the_unmerged_down(loaded, checkpoint):
    _folder, raw = checkpoint
    got = loaded["model.hyper_connection_mixer.input_mix_weight_down.weight"]
    assert got.shape == (LR, HCH)
    assert torch.equal(
        got, raw["model.language_model.hyper_connection_mixer.input_mix_weight_down.weight"]
    )
    assert torch.equal(
        loaded["model.hyper_connection_mixer.input_mix_weight_up.weight"],
        raw["model.language_model.hyper_connection_mixer.input_mix_weight_up.weight"],
    )


def test_qkv_fusion_slices_back_to_q_k_v(loaded, checkpoint):
    _folder, raw = checkpoint
    attn = "model.language_model.layers.1.self_attn"
    parts = [raw[f"{attn}.{p}_proj.weight"] for p in ("q", "k", "v")]
    fused = loaded["model.layers.1.self_attn.qkv_proj.weight"]
    assert fused.shape == (2 * QH * AHD + 2 * KVH * AHD, H)  # q carries the output gate
    for part, back in zip(parts, torch.split(fused, [p.shape[0] for p in parts], dim=0)):
        assert torch.equal(part, back)


def test_gdn_in_proj_slices_round_trip(loaded, checkpoint):
    _folder, raw = checkpoint
    gdn = "model.language_model.layers.0.linear_attn"
    parts = [raw[f"{gdn}.in_proj_{p}.weight"] for p in ("qkv", "z", "b", "a")]
    fused = loaded["model.layers.0.linear_attn.in_proj.weight"]
    assert fused.shape == (sum(p.shape[0] for p in parts), H)
    splits = torch.split(fused, [p.shape[0] for p in parts], dim=0)
    for part, back in zip(parts, splits):
        assert torch.equal(part, back)


def test_shared_expert_gate_up_merge(loaded, checkpoint):
    _folder, raw = checkpoint
    base = "model.language_model.layers.1.mlp.shared_expert"
    merged = loaded["model.layers.1.mlp.shared_expert.gate_up_proj.weight"]
    assert torch.equal(merged[:I], raw[f"{base}.gate_proj.weight"])
    assert torch.equal(merged[I:], raw[f"{base}.up_proj.weight"])


ZERO_CENTERED = (
    "model.layers.0.attn_hyper_connection.hc_norm.weight",
    "model.layers.0.mlp_hyper_connection.hc_norm.weight",
    "model.hyper_connection_mixer.hc_norm.weight",
    "model.layers.0.ple.norm_key.weight",
    "model.layers.0.ple.norm_query.weight",
    "model.layers.0.ple.norm_conv.weight",
    "model.layers.1.self_attn.q_norm.weight",
    "model.layers.1.self_attn.k_norm.weight",
    "model.layers.1.self_attn.indexer.q_layernorm.weight",
    "model.layers.1.self_attn.indexer.k_layernorm.weight",
)


def test_zero_centered_norms_are_loaded_raw(loaded, checkpoint):
    """(1+w) is applied at runtime in fp32, so the loader must not fold it into the bf16 weight."""
    _folder, raw = checkpoint
    for name in ZERO_CENTERED:
        raw_name = name.replace("model.", "model.language_model.", 1)
        assert torch.equal(loaded[name], raw[raw_name]), name


def test_the_zero_centered_suffix_list_covers_every_such_norm():
    assert {n for n in ZERO_CENTERED if n.endswith(_ZERO_CENTERED_NORM_SUFFIXES)} == set(ZERO_CENTERED)
    assert not "model.layers.0.linear_attn.norm.weight".endswith(_ZERO_CENTERED_NORM_SUFFIXES)


def test_gdn_gated_norm_passes_through(loaded, checkpoint):
    _folder, raw = checkpoint
    assert torch.equal(
        loaded["model.layers.0.linear_attn.norm.weight"],
        raw["model.language_model.layers.0.linear_attn.norm.weight"],
    )


def test_hash_constants_stay_int64(loaded):
    for leaf in ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"):
        assert loaded[f"model.layers.0.ple.ple_embedding.{leaf}"].dtype is torch.int64


def test_load_ple_table_concatenates_shards_in_index_order(checkpoint):
    folder, raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=NGRAM_DIM)
    table = load_ple_table(folder, args, pin=False)
    assert table.tensor.shape == (NGRAM_SHARDS * NGRAM_ROWS, NGRAM_DIM)
    assert table.tensor.dtype is torch.float8_e4m3fn
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    for shard in range(NGRAM_SHARDS):
        rows = table.tensor[shard * NGRAM_ROWS: (shard + 1) * NGRAM_ROWS]
        assert torch.equal(rows.view(torch.uint8),
                           raw[f"{prefix}.shard_{shard}.weight"].view(torch.uint8))
    assert table.weight_scale.dtype is torch.bfloat16
    assert float(table.weight_scale) == 0.125


def test_load_ple_table_reads_an_unscaled_bf16_table(tmp_path):
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    shards = {f"{prefix}.shard_{i}.weight": _bf16(NGRAM_ROWS, NGRAM_DIM) for i in range(NGRAM_SHARDS)}
    save_file(shards, str(tmp_path / "model.safetensors"))
    table = load_ple_table(str(tmp_path), SimpleNamespace(split_ngram_parts=NGRAM_SHARDS, ngram_head_dim=NGRAM_DIM), pin=False)
    assert torch.equal(table.tensor, torch.cat([shards[f"{prefix}.shard_{i}.weight"] for i in range(NGRAM_SHARDS)]))
    assert float(table.weight_scale) == 1.0


def test_load_ple_table_rejects_a_shard_count_mismatch(checkpoint):
    folder, _raw = checkpoint
    args = SimpleNamespace(split_ngram_parts=NGRAM_SHARDS + 1, ngram_head_dim=NGRAM_DIM)
    with pytest.raises(ValueError, match="shards 0"):
        load_ple_table(folder, args, pin=False)


def _nvfp4_ngram_table() -> tuple[dict[str, torch.Tensor], torch.Tensor, int, int, int]:
    """The FP4-Local table: packed e2m1 values (u8, two per byte) with per-16 e4m3 scale rows.

    head_dim is 16 so a packed row is 8 bytes with a single per-16 scale.
    """
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    rows, parts, dim = 5, 3, 16
    tensors = {
        f"{prefix}.shard_{i}.weight": (
            torch.arange(i * rows * dim // 2, (i + 1) * rows * dim // 2)
            .remainder(255).to(torch.uint8).view(rows, dim // 2)
        )
        for i in range(parts)
    }
    tensors.update(
        {
            f"{prefix}.shard_{i}.weight_scale": (
                torch.arange(i * rows * dim // 16, (i + 1) * rows * dim // 16)
                .remainder(127).to(torch.uint8).view(rows, dim // 16).view(torch.float8_e4m3fn)
            )
            for i in range(parts)
        }
    )
    scale = torch.tensor([0.25], dtype=torch.float32)
    tensors[f"{prefix}.weight_scale_2"] = scale
    return tensors, scale, rows, dim, parts


def test_load_ple_table_nvfp4_concatenates_packed_shards(tmp_path):
    """The FP4-Local table: packed e2m1 value rows, per-16 e4m3 scale rows, global weight_scale_2."""
    tensors, _scale, rows, dim, parts = _nvfp4_ngram_table()
    names = sorted(tensors)
    save_file({n: tensors[n] for n in names[:2]}, str(tmp_path / "model-plefp4-00000.safetensors"))
    save_file({n: tensors[n] for n in names[2:]}, str(tmp_path / "model-plefp4-00001.safetensors"))
    args = SimpleNamespace(split_ngram_parts=parts, ngram_head_dim=dim, ple_embedding_dtype="nvfp4")
    table = load_ple_table(str(tmp_path), args, pin=False)
    assert table.tensor.shape == (parts * rows, dim // 2)
    assert table.tensor.dtype is torch.uint8
    assert table.scale_bank is not None
    assert table.scale_bank.tensor.shape == (parts * rows, dim // 16)
    assert table.scale_bank.tensor.dtype is torch.float8_e4m3fn
    assert table.weight_scale.dtype is torch.float32
    assert float(table.weight_scale) == 0.25
    prefix = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding"
    for shard in range(parts):
        assert torch.equal(
            table.tensor[shard * rows : (shard + 1) * rows], tensors[f"{prefix}.shard_{shard}.weight"]
        )
        assert torch.equal(
            table.scale_bank.tensor[shard * rows : (shard + 1) * rows],
            tensors[f"{prefix}.shard_{shard}.weight_scale"],
        )


def test_load_ple_table_rejects_a_nvfp4_table_under_fp8(tmp_path):
    """A packed table's U8 value shards must not load as an FP8 table."""
    tensors, _scale, _rows, dim, parts = _nvfp4_ngram_table()
    for i, name in enumerate(sorted(tensors)):
        save_file({name: tensors[name]}, str(tmp_path / f"model-plefp4-{i:05d}.safetensors"))
    args = SimpleNamespace(split_ngram_parts=parts, ngram_head_dim=dim, ple_embedding_dtype="fp8")
    with pytest.raises(ValueError, match="expected F8_E4M3"):
        load_ple_table(str(tmp_path), args, pin=False)


# ======================================================================================
# read_range_into: the O_DIRECT byte-range read the PLE table load is built on
# ======================================================================================


@pytest.fixture(scope="module")
def blob(tmp_path_factory) -> tuple[str, bytes]:
    data = random.Random(7).randbytes(5_000_003)
    path = tmp_path_factory.mktemp("blob") / "data.bin"
    path.write_bytes(data)
    return str(path), data


@pytest.mark.parametrize("file_offset, nbytes, dest_offset", [
    (1, 4095, 0),                 # sub-block, unaligned source
    (2239, 1_000_000, 0),         # the real checkpoint's header-end phase
    (4095, 4097, 1),              # straddles two block boundaries
    (4_999_000, 1003, 123_456),   # runs to EOF
])
def test_read_range_into_matches_the_file(blob, file_offset, nbytes, dest_offset):
    path, data = blob
    bank = HostBank((6_000_000,), torch.uint8)
    view = bank.memoryview()
    got = read_range_into(view, path, file_offset=file_offset, nbytes=nbytes,
                          dest_offset=dest_offset, chunk=1 << 20)
    assert got == nbytes
    assert bytes(view[dest_offset:dest_offset + nbytes]) == data[file_offset:file_offset + nbytes]


def test_read_range_into_is_chunk_and_thread_safe(blob):
    path, data = blob
    bank = HostBank((6_000_000,), torch.uint8)
    view = bank.memoryview()
    read_range_into(view, path, file_offset=2239, nbytes=4_000_000, dest_offset=1024,
                    workers=8, chunk=64 << 10)
    assert bytes(view[1024:1024 + 4_000_000]) == data[2239:2239 + 4_000_000]


def test_read_range_into_rejects_a_short_destination(blob):
    path, _data = blob
    bank = HostBank((1024,), torch.uint8)
    with pytest.raises(ValueError, match="destination holds"):
        read_range_into(bank.memoryview(), path, file_offset=0, nbytes=1 << 20)


# ======================================================================================
# AOT shape table
# ======================================================================================


def test_aot_entry_carries_the_checkpoint_geometry():
    entry = next(m for m in SUPPORTED_MODELS
                 if m.architecture == "Qwen4ExpForConditionalGeneration")
    assert (entry.hidden_size, entry.moe_intermediate_size, entry.top_k) == (2560, 640, 10)
    assert entry.kv_groups == ((2, 256),)
    rows = expert_bank_row_bytes("nvfp4", entry.hidden_size, entry.moe_intermediate_size)
    assert set(rows) == {"gate_up_packed", "gate_up_scale", "gate_up_global",
                         "down_packed", "down_scale", "down_global"}
    for name, nbytes in rows.items():
        assert nbytes % 16 == 0, name  # fused multi-bank copy only engages on 16B multiples


def test_every_registry_architecture_is_claimed_by_an_aot_entry():
    from freetoken.models.register import _MODEL_REGISTRY

    claimed = {m.architecture for m in SUPPORTED_MODELS}
    claimed |= {a for m in SUPPORTED_MODELS for a in m.arch_aliases}
    assert "Qwen4ExpForConditionalGeneration" in claimed
    assert set(_MODEL_REGISTRY) - claimed == set()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_fusion_pad_rides_the_tensor_device():
    """safetensors loads straight to cuda; a cpu-allocated pad row would break torch.cat."""
    reader = _DenseReader(None, get_model_spec("Qwen4ExpForConditionalGeneration").packed_modules_mapping)
    down = torch.randn(320, 64, device="cuda", dtype=torch.bfloat16)
    inject = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    assert reader.add("model.layers.0.attn_hyper_connection.input_mix_weight_down.weight", down) == []
    [(key, fused)] = reader.add("model.layers.0.attn_hyper_connection.block_inject_weight.weight", inject)
    assert key == "model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight"
    assert fused.device.type == "cuda" and fused.shape[0] == 336
    assert torch.equal(fused[324:], torch.zeros(12, 64, device="cuda", dtype=torch.bfloat16))


# ======================================================================================
# the reader against the model the engine builds, for each released quant layout
# ======================================================================================


@pytest.fixture(scope="module")
def checkpoint_nvfp4(tmp_path_factory) -> tuple[str, dict[str, torch.Tensor]]:
    """bf16 dense tensors under a real ModelOptConfig whose ignore list covers them (RadixArk)."""
    torch.manual_seed(2)
    return _write_checkpoint(tmp_path_factory.mktemp("qwen4_exp_nvfp4_ckpt"), _raw_checkpoint(), RADIXARK_NVFP4)


FP8_MODULES = (
    "model.layers.1.self_attn.qkv_proj", "model.layers.1.self_attn.o_proj",
    "model.layers.0.linear_attn.in_proj_qkvz", "model.layers.0.linear_attn.out_proj",
)


@pytest.mark.parametrize("fixture", ["checkpoint", "checkpoint_nvfp4", "checkpoint_fp8", "checkpoint_fp8_channel"])
def test_emitted_keys_are_the_model_state_dict(fixture, request):
    """The reader fills exactly the buffers the engine builds from the same config, fp8 ones with the stored weight dtype and fp32 scales."""
    folder, _raw = request.getfixturevalue(fixture)
    loaded, state = _load(folder, vision=False), meta_state_dict(folder)
    assert set(loaded) == set(state)
    if fixture in ("checkpoint", "checkpoint_nvfp4"):
        assert loaded["model.layers.0.linear_attn.in_proj.weight"].dtype is torch.bfloat16
        return
    scale = ".weight_scale_inv" if fixture == "checkpoint_fp8" else ".weight_scale"
    for module in FP8_MODULES:
        for kind in (".weight", scale):
            assert loaded[module + kind].shape == state[module + kind].shape, module + kind
        assert loaded[module + ".weight"].dtype is state[module + ".weight"].dtype is torch.float8_e4m3fn
        assert loaded[module + scale].dtype is torch.float32  # the engine casts it to the buffer dtype at load


def test_channel_fp8_scales_fuse_as_fp32_rows(checkpoint_fp8_channel):
    folder, raw = checkpoint_fp8_channel
    loaded = _load(folder)
    gdn, shared = f"{LM}.layers.0.linear_attn", f"{LM}.layers.1.mlp.shared_expert"
    for fused, parts in (
        ("model.layers.0.linear_attn.in_proj_qkvz", [f"{gdn}.in_proj_qkv", f"{gdn}.in_proj_z"]),
        ("model.layers.1.mlp.shared_expert.gate_up_proj", [f"{shared}.gate_proj", f"{shared}.up_proj"]),
    ):
        assert torch.equal(loaded[f"{fused}.weight_scale"], torch.cat([raw[f"{p}.weight_scale"].reshape(-1).float() for p in parts]))
        assert torch.equal(loaded[f"{fused}.weight"].view(torch.uint8), torch.cat([raw[f"{p}.weight"].view(torch.uint8) for p in parts]))


@pytest.mark.parametrize("scale_name", ["weight_scale_inv", "weight_scale"])
def test_mxfp8_scales_load_under_either_modelopt_name(tmp_path, scale_name):
    folder, raw = _write_checkpoint(tmp_path, _raw_checkpoint(mxfp8_scale=scale_name), MXFP8_DENSE_QUANT)
    loaded, state = _load(folder, vision=False), meta_state_dict(folder)
    assert set(loaded) == set(state)
    gdn = f"{LM}.layers.0.linear_attn"
    assert torch.equal(
        loaded["model.layers.0.linear_attn.in_proj_qkvz.weight_scale_inv"],
        torch.cat([raw[f"{gdn}.{p}.{scale_name}"] for p in ("in_proj_qkv", "in_proj_z")]),
    )


def test_expert_spec_reads_the_llm_compressor_names(checkpoint_fp8_channel):
    folder, _raw = checkpoint_fp8_channel
    install_quant_config(folder)
    spec = nvfp4_expert_spec(folder, None)
    base = f"{LM}.layers.1.mlp.experts.2.down_proj"
    kinds = {spec.kind_map[spec.key_pattern.match(f"{base}.{k}")["kind"]] for k in ("weight_packed", "weight_scale", "weight_global_scale")}
    assert kinds == {"weight", "weight_scale", "weight_scale_2"}
    assert spec.global_reciprocal


def _assert_fused_per_kind(loaded, raw, fused: str, parts: list[str]) -> None:
    for kind in (".weight", ".weight_scale_inv"):
        sources = [raw[f"{p}{kind}"].view(torch.uint8) for p in parts]
        merged = loaded[fused + kind]
        assert merged.dtype is raw[f"{parts[0]}{kind}"].dtype
        for source, back in zip(sources, torch.split(merged.view(torch.uint8), [s.shape[0] for s in sources], dim=0)):
            assert torch.equal(source, back)


def test_fp8_projections_fuse_per_kind(loaded_fp8, checkpoint_fp8):
    _folder, raw = checkpoint_fp8
    attn, gdn = f"{LM}.layers.1.self_attn", f"{LM}.layers.0.linear_attn"
    _assert_fused_per_kind(loaded_fp8, raw, "model.layers.1.self_attn.qkv_proj", [f"{attn}.{p}_proj" for p in "qkv"])
    _assert_fused_per_kind(loaded_fp8, raw, "model.layers.0.linear_attn.in_proj_qkvz", [f"{gdn}.in_proj_qkv", f"{gdn}.in_proj_z"])
    assert torch.equal(loaded_fp8["model.layers.0.linear_attn.out_proj.weight_scale_inv"], raw[f"{gdn}.out_proj.weight_scale_inv"])
    assert torch.equal(
        loaded_fp8["model.layers.0.linear_attn.in_proj_ba.weight"],
        torch.cat([raw[f"{gdn}.in_proj_b.weight"], raw[f"{gdn}.in_proj_a.weight"]], dim=0),
    )
    for name in ("model.layers.0.linear_attn.in_proj_ba.weight", "model.layers.1.mlp.shared_expert.gate_up_proj.weight",
                 "model.layers.1.self_attn.indexer.index_qk_proj.weight", "lm_head.weight",
                 "model.hyper_connection_mixer.input_mix_weight_down.weight",
                 "model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight"):
        assert loaded_fp8[name].dtype is torch.bfloat16


ATTN = f"{LM}.layers.1.self_attn"
REJECTED = [
    pytest.param(None, lambda w: {f"{ATTN}.q_proj.weight": w, f"{ATTN}.q_proj.weight_scale_inv": _fp8_scale(w)},
                 r"q_proj\.weight_scale_inv", id="scale the config does not declare"),
    pytest.param(None, lambda w: {f"{ATTN}.o_proj.weight": w.to(torch.float8_e4m3fn)},
                 r"o_proj\.weight is torch\.float8", id="fp8 weight the config declares bf16"),
    pytest.param(FP8_DENSE_QUANT, lambda w: {f"{ATTN}.{p}_proj.weight": w.clone() for p in "qkv"},
                 r"[qkv]_proj\.weight is torch\.bfloat16", id="bf16 weight the config declares fp8"),
    pytest.param(FP8_DENSE_QUANT, lambda w: {f"{ATTN}.q_proj.weight": w[:-64].to(torch.float8_e4m3fn), f"{ATTN}.q_proj.weight_scale_inv": _fp8_scale(w)},
                 "128x128", id="part that is not a 128-row multiple"),
]


@pytest.mark.parametrize("quantization_config, tensors, match", REJECTED)
def test_checkpoint_disagreeing_with_its_quant_config_is_rejected(tmp_path, quantization_config, tensors, match):
    save_file(tensors(_bf16(2 * QH * AHD, H)), str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(_config_json(quantization_config)))
    with pytest.raises(ValueError, match=match):
        _load(str(tmp_path))
