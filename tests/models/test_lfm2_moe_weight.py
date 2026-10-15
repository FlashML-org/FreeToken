"""lfm2_moe weight loading against synthetic checkpoints shaped like the released ones.

Tiny tensors, real key names, dtypes and quantization_config blocks: each layout is read by
``iter_weights`` and compared with the state dict of the model the engine builds from the same
config, so the dense pass has to fill exactly those buffers whatever the checkpoint quantized.
The routed experts are per-expert in every LFM2 release, so they load through the stacked form
(bf16) or the expert pieces (block-fp8), never the dense pass.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.layers.quantization import QuantKind, set_quant_config
from freetoken.models.lfm2_moe.config import parse_config
from freetoken.models.lfm2_moe.weight import iter_expert_pieces, iter_weights, iter_weights_parallel
from freetoken.models.register import checkpoint_quant_config, get_model_spec
from freetoken.moe.expert_pieces import iter_expert_pieces as stream_expert_pieces
from freetoken.utils import cached_load_hf_config

H, V = 128, 256  # hidden_size (128 keeps every block-fp8 projection on the 128 block edge), vocab
QH, KVH, HD = 4, 2, 64  # attention heads / kv heads / head dim: q rows 256, k|v rows 128
DI, MI, E = 128, 128, 4  # dense MLP width, routed expert width, routed experts
LAYER_TYPES = ["conv", "full_attention", "conv"]  # conv+dense, attention+MoE, conv+MoE
DENSE_LAYERS = 1
FP8 = torch.float8_e4m3fn
LAYOUTS = ("bf16", "fp8_block")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="session", autouse=True)
def _tp_info():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _bf16(*shape: int) -> torch.Tensor:
    return torch.randn(*shape).to(torch.bfloat16)


def _config_json(name: str) -> dict:
    config = {
        "architectures": ["Lfm2MoeForCausalLM"],
        "model_type": "lfm2_moe",
        "hidden_size": H,
        "intermediate_size": DI,
        "num_hidden_layers": len(LAYER_TYPES),
        "layer_types": list(LAYER_TYPES),
        "num_attention_heads": QH,
        "num_key_value_heads": KVH,
        "head_dim": HD,
        "vocab_size": V,
        "moe_intermediate_size": MI,
        "num_experts": E,
        "num_experts_per_tok": 2,
        "num_dense_layers": DENSE_LAYERS,
        "norm_eps": 1e-5,
        "conv_L_cache": 3,
        "conv_bias": False,
        "tie_word_embeddings": True,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
        "use_expert_bias": True,
        "max_position_embeddings": 512,
        "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
    }
    if name == "fp8_block":
        config["quantization_config"] = {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "ignore": ["lm_head", "model.embed_tokens", "re:.*\\.gate$", "re:.*\\.(?:conv|short_conv)\\..*"],
            "config_groups": {
                "FP8_BLOCK": {
                    "targets": ["Linear"],
                    "input_activations": {"dynamic": True, "num_bits": 8, "type": "float", "strategy": "group"},
                    "weights": {
                        "num_bits": 8, "type": "float", "strategy": "block",
                        "block_structure": [128, 128], "dynamic": False,
                    },
                }
            },
        }
    return config


def _fp8_pair(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    """llm-compressor block-fp8: fp8-e4m3 weight + a bf16 128x128 block scale (the released dtype)."""
    rows, cols = weight.shape
    return {
        "weight": weight.to(FP8),
        "weight_scale": (torch.rand(rows // 128, cols // 128) + 0.5).to(torch.bfloat16),
    }


def _quantize(raw: dict[str, torch.Tensor], modules: list[str]) -> None:
    for module in modules:
        raw.update({f"{module}.{suffix}": t for suffix, t in _fp8_pair(raw.pop(f"{module}.weight")).items()})


def _raw(name: str) -> dict[str, torch.Tensor]:
    """Every checkpoint tensor in the released layout, quantized for the fp8 layout."""
    raw = {
        "model.embed_tokens.weight": _bf16(V, H),
        "model.embedding_norm.weight": _bf16(H),
    }
    quantizable = []
    for layer, kind in enumerate(LAYER_TYPES):
        pre = f"model.layers.{layer}"
        raw.update({f"{pre}.operator_norm.weight": _bf16(H), f"{pre}.ffn_norm.weight": _bf16(H)})
        if kind == "conv":
            raw.update({
                f"{pre}.conv.conv.weight": _bf16(H, 1, 3),
                f"{pre}.conv.in_proj.weight": _bf16(3 * H, H),
                f"{pre}.conv.out_proj.weight": _bf16(H, H),
            })
        else:
            attn = f"{pre}.self_attn"
            raw.update({
                f"{attn}.q_proj.weight": _bf16(QH * HD, H),
                f"{attn}.k_proj.weight": _bf16(KVH * HD, H),
                f"{attn}.v_proj.weight": _bf16(KVH * HD, H),
                f"{attn}.out_proj.weight": _bf16(H, QH * HD),
                f"{attn}.q_layernorm.weight": _bf16(HD),
                f"{attn}.k_layernorm.weight": _bf16(HD),
            })
            quantizable += [f"{attn}.{proj}_proj" for proj in ("q", "k", "v", "out")]
        ff = f"{pre}.feed_forward"
        if layer < DENSE_LAYERS:
            raw.update({
                f"{ff}.w1.weight": _bf16(DI, H),
                f"{ff}.w3.weight": _bf16(DI, H),
                f"{ff}.w2.weight": _bf16(H, DI),
            })
            quantizable += [f"{ff}.{proj}" for proj in ("w1", "w2", "w3")]
        else:
            raw.update({
                f"{ff}.gate.weight": _bf16(E, H),
                f"{ff}.expert_bias": torch.randn(E, dtype=torch.float32),
            })
            for expert in range(E):
                base = f"{ff}.experts.{expert}"
                raw.update({
                    f"{base}.w1.weight": _bf16(MI, H),
                    f"{base}.w3.weight": _bf16(MI, H),
                    f"{base}.w2.weight": _bf16(H, MI),
                })
                quantizable += [f"{base}.{proj}" for proj in ("w1", "w2", "w3")]
    if name == "fp8_block":
        _quantize(raw, quantizable)
    return raw


def _write(folder, name: str, raw: dict[str, torch.Tensor], *, shards: int = 2) -> str:
    """Spread the tensors over ``shards`` files, without an index, so a module's weight and its
    scale can cross a file boundary."""
    names = sorted(raw)
    for i in range(shards):
        save_file({n: raw[n] for n in names[i::shards]}, str(folder / f"model-{i:05d}.safetensors"))
    (folder / "config.json").write_text(json.dumps(_config_json(name)))
    return str(folder)


@pytest.fixture(scope="module", params=LAYOUTS)
def checkpoint(request, tmp_path_factory):
    torch.manual_seed(LAYOUTS.index(request.param))
    raw = _raw(request.param)
    return request.param, _write(tmp_path_factory.mktemp(request.param), request.param, raw), raw


def _install(folder: str) -> None:
    """Install the folder's QuantConfig process-wide, as EngineConfig does before the reader runs."""
    hf = cached_load_hf_config(folder)
    set_quant_config(checkpoint_quant_config(folder, hf, get_model_spec(hf.architectures[0])))


def _load(folder: str, *, experts: bool = False) -> dict[str, torch.Tensor]:
    _install(folder)
    return {
        n: t.clone()
        for n, t in iter_weights(
            folder, torch.device("cpu"), include_moe_experts=experts, include_non_moe=True
        )
    }


def _meta_state_dict(folder: str) -> dict[str, torch.Tensor]:
    """State dict of the model the engine builds for ``folder`` (experts offloaded), on the meta device."""
    from freetoken.engine.config import EngineConfig
    from freetoken.engine.engine import _decode_target
    from freetoken.layers import rotary
    from freetoken.models import create_model
    from freetoken.utils.torch_utils import torch_dtype

    strategy = "offload"
    config = EngineConfig(
        model_path=folder, tp_info=try_get_tp_info(), dtype=torch.bfloat16, moe_strategy=strategy
    )
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


# --------------------------------------------------------------------------- the reader against the engine's model


def test_emitted_keys_are_the_model_state_dict(checkpoint):
    """Every layout fills exactly the buffers the engine builds from the same config, with the
    buffers' shapes and dtypes."""
    _name, folder, _raw_tensors = checkpoint
    loaded, state = _load(folder), _meta_state_dict(folder)
    assert set(loaded) == set(state)
    for key, tensor in loaded.items():
        assert tensor.shape == state[key].shape, key
        assert tensor.dtype == state[key].dtype, key
    assert not any(".experts." in k for k in loaded)
    assert loaded["model.layers.1.feed_forward.expert_bias"].dtype is torch.float32
    assert loaded["model.layers.2.feed_forward.gate.weight"].shape == (E, H)


def test_fp8_block_scales_are_renamed_to_the_layer_buffers(checkpoint):
    name, folder, raw = checkpoint
    loaded = _load(folder)
    proj = "model.layers.1.self_attn.q_proj"
    if name == "bf16":
        assert f"{proj}.weight" in loaded and set(k for k in loaded if k.startswith(proj)) == {f"{proj}.weight"}
        return
    assert loaded[f"{proj}.weight"].dtype is FP8
    assert torch.equal(loaded[f"{proj}.weight_scale_inv"], raw[f"{proj}.weight_scale"])
    # the block edge (128) is checked against the weight's own shape
    assert loaded[f"{proj}.weight_scale_inv"].shape == (QH * HD // 128, H // 128)
    assert not any(k.endswith("weight_scale") for k in loaded)


def test_a_wrong_block_scale_shape_is_refused(checkpoint, tmp_path):
    name, folder, raw = checkpoint
    if name != "fp8_block":
        pytest.skip("block-fp8 layout only")
    broken = {k: v.clone() for k, v in raw.items()}
    broken["model.layers.1.self_attn.k_proj.weight_scale"] = torch.rand(2, 2, dtype=torch.bfloat16)
    folder2 = _write(tmp_path, "fp8_block", broken)
    _install(folder2)
    with pytest.raises(ValueError, match="weight_scale_inv is"):
        list(iter_weights(folder2, torch.device("cpu"), include_moe_experts=False, include_non_moe=True))


def test_a_quantized_bf16_module_is_refused(checkpoint, tmp_path):
    """The family serves the conv projections and the router bf16; a checkpoint that quantized
    one must fail loudly rather than load a scale the model has no buffer for."""
    name, folder, raw = checkpoint
    if name != "fp8_block":
        pytest.skip("block-fp8 layout only")
    broken = {k: v.clone() for k, v in raw.items()}
    conv = "model.layers.0.conv.in_proj"
    broken.update({f"{conv}.{suffix}": t for suffix, t in _fp8_pair(broken.pop(f"{conv}.weight")).items()})
    folder2 = _write(tmp_path, "fp8_block", broken)
    _install(folder2)
    with pytest.raises(ValueError, match="serves in bf16"):
        list(iter_weights(folder2, torch.device("cpu"), include_moe_experts=False, include_non_moe=True))


def test_bf16_stacked_experts_are_emitted_only_when_asked(checkpoint):
    name, folder, raw = checkpoint
    if name != "bf16":
        pytest.skip("bf16 layout only")
    _install(folder)
    experts = dict(
        iter_weights(folder, torch.device("cpu"), include_moe_experts=True, include_non_moe=False)
    )
    assert set(experts) == {
        f"model.layers.{layer}.feed_forward.experts.{leaf}"
        for layer in (1, 2)
        for leaf in ("gate_up_proj", "down_proj")
    }
    for layer in (1, 2):
        base = f"model.layers.{layer}.feed_forward.experts"
        gate_up = experts[f"{base}.gate_up_proj"]
        assert gate_up.shape == (E, 2 * MI, H)
        assert torch.equal(gate_up[:, :MI], torch.stack([raw[f"{base}.{e}.w1.weight"] for e in range(E)]))
        assert torch.equal(gate_up[:, MI:], torch.stack([raw[f"{base}.{e}.w3.weight"] for e in range(E)]))
        assert torch.equal(
            experts[f"{base}.down_proj"], torch.stack([raw[f"{base}.{e}.w2.weight"] for e in range(E)])
        )


# --------------------------------------------------------------------------- expert pieces into the banks


def _expected_piece_raw(raw, layer: int, expert: int, name: str) -> dict[str, torch.Tensor]:
    base = f"model.layers.{layer}.feed_forward.experts.{expert}"
    out = {
        "gate": raw[f"{base}.w1.weight"][None],
        "up": raw[f"{base}.w3.weight"][None],
        "down": raw[f"{base}.w2.weight"][None],
    }
    if name == "fp8_block":
        out.update({
            "gate_scale": raw[f"{base}.w1.weight_scale"][None],
            "up_scale": raw[f"{base}.w3.weight_scale"][None],
            "down_scale": raw[f"{base}.w2.weight_scale"][None],
        })
    return out


@pytest.mark.parametrize("parallel", [False, True])
def test_expert_pieces_land_in_the_moe_bank_layout(checkpoint, parallel):
    """Both expert kinds stream the checkpoint's per-expert tensors into the kernel's banks."""
    name, folder, raw = checkpoint
    _install(folder)
    config = parse_config(cached_load_hf_config(folder))
    kind = QuantKind.NONE if name == "bf16" else QuantKind.FP8_BLOCK

    pieces = list(stream_expert_pieces(folder, config, kind, parallel=parallel))
    assert len(pieces) == 2 * E  # two MoE layers x E experts, one piece each
    for layer, e0, e1, piece in pieces:
        assert (e0, e1) == (e0, e0 + 1)
        expected = _expected_piece_raw(raw, layer + DENSE_LAYERS, e0, name)
        assert set(piece) == set(expected)
        for role, tensor in expected.items():
            assert torch.equal(piece[role].view(torch.uint8), tensor.view(torch.uint8)), (layer, e0, role)

    # the pieces pack into the kernel's banks as the checkpoint stored them
    from freetoken.layers import iter_moe_layers
    from freetoken.engine.config import EngineConfig
    from freetoken.layers import rotary
    from freetoken.models import create_model
    from freetoken.moe.expert_banks import build_expert_banks
    from freetoken.utils.torch_utils import torch_dtype

    strategy = "offload"
    engine_config = EngineConfig(
        model_path=folder, tp_info=try_get_tp_info(), dtype=torch.bfloat16, moe_strategy=strategy
    )
    object.__setattr__(engine_config.model_config, "moe_strategy", strategy)
    saved = rotary._ROPE_DEVICE
    rotary.set_rope_device(torch.device("cpu"))
    rotary.get_rope.cache_clear()
    try:
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            model = create_model(engine_config.model_config)
    finally:
        rotary.set_rope_device(saved)
        rotary.get_rope.cache_clear()
    layer = next(iter(iter_moe_layers(model)))
    banks = build_expert_banks(layer.quant_method, 2, pieces, device=torch.device("cpu"))
    for bank_layer in range(2):
        ckpt_layer = bank_layer + DENSE_LAYERS
        for e in range(E):
            base = f"model.layers.{ckpt_layer}.feed_forward.experts.{e}"
            w1, w3, w2 = (raw[f"{base}.{proj}.weight"] for proj in ("w1", "w3", "w2"))
            if name == "bf16":
                assert torch.equal(banks.sources["gate_up"][bank_layer][e], torch.cat([w1, w3]))
                assert torch.equal(banks.sources["down"][bank_layer][e], w2)
                continue
            # block-fp8 banks: gate|up fp8 rows + their block scales (trailing dim 16B-padded)
            assert torch.equal(banks.sources["gate_up"][bank_layer][e].view(torch.uint8), torch.cat([w1, w3]).view(torch.uint8))
            assert torch.equal(banks.sources["down"][bank_layer][e].view(torch.uint8), w2.view(torch.uint8))
            down_scale = banks.sources["down_scale"][bank_layer][e]
            stored_down = raw[f"{base}.w2.weight_scale"]
            assert down_scale.shape[1] >= stored_down.shape[1]
            assert torch.equal(down_scale[:, : stored_down.shape[1]], stored_down)
            gate_up_scale = banks.sources["gate_up_scale"][bank_layer][e]
            cols = raw[f"{base}.w1.weight_scale"].shape[1]
            assert torch.equal(gate_up_scale[:1, :cols], raw[f"{base}.w1.weight_scale"])
            assert torch.equal(gate_up_scale[1:, :cols], raw[f"{base}.w3.weight_scale"])


def test_parallel_weight_reader_is_experts_only(checkpoint):
    name, folder, raw = checkpoint
    if name != "bf16":
        pytest.skip("bf16 layout only")
    _install(folder)
    stacked = dict(iter_weights_parallel(
        folder, torch.device("cpu"), include_moe_experts=True, include_non_moe=False
    ))
    serial = dict(iter_weights(folder, torch.device("cpu"), include_moe_experts=True, include_non_moe=False))
    assert set(stacked) == set(serial)
    for key, tensor in stacked.items():
        assert torch.equal(tensor, serial[key])


def test_tp_above_one_is_refused(checkpoint, monkeypatch):
    _name, folder, _raw_tensors = checkpoint
    _install(folder)
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr("freetoken.distributed.info._TP_INFO", DistributedInfo(rank=0, size=2))
    config = parse_config(cached_load_hf_config(folder))
    with pytest.raises(NotImplementedError, match="TP=1 only"):
        list(iter_weights(folder, torch.device("cpu"), include_moe_experts=False, include_non_moe=True))
    with pytest.raises(NotImplementedError, match="TP=1 only"):
        list(stream_expert_pieces(folder, config, QuantKind.NONE))
    with pytest.raises(NotImplementedError, match="TP=1 only"):
        list(iter_expert_pieces(folder, config, QuantKind.FP8_BLOCK))
    with pytest.raises(NotImplementedError, match="TP=1 only"):
        list(iter_weights_parallel(folder, torch.device("cpu"), include_moe_experts=True, include_non_moe=False))
