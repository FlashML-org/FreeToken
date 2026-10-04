"""Coverage for the explicit Gemma 4 resident-Q4 expert path."""

from math import prod
from types import SimpleNamespace

import pytest
import torch
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.engine.engine import _explicit_q4_resident_ok
from freetoken.models.gemma4.moe import Gemma4ResidentQ4MoELayer
from freetoken.models.gguf.dequant import GGML_Q4_0, row_bytes


def _config(**overrides):
    values = {
        "moe_strategy": "fused",
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "hidden_size": 64,
        "moe_intermediate_size": 32,
        "norm_topk_prob": True,
        "decode_target": "gpu",
        "moe_weight_format": "q4_0",
        "quant": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _ensure_tp1() -> None:
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


@pytest.mark.parametrize(
    ("model_type", "expert_quant", "weight_format", "supported"),
    [
        ("gemma4", "q4_0", "q4_0", True),
        ("qwen3", "q4_0", "q4_0", False),
        ("gemma4", "q4_k", "q4_0", False),
        ("gemma4", "q4_0", "q4_k", False),
    ],
)
def test_explicit_q4_resident_capability_is_narrow(
    model_type, expert_quant, weight_format, supported
) -> None:
    config = SimpleNamespace(
        model_type=model_type,
        expert_quant=expert_quant,
        moe_weight_format=weight_format,
    )
    assert _explicit_q4_resident_ok(config) is supported


def _engine_config(*, model_type="gemma4", expert_quant="q4_0", weight_format="q4_0"):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/gemma4-q4.gguf",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        attention_backend="triton",
        moe_strategy="fused",
    )
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(
            model_type=model_type,
            single_stream_only=False,
            is_moe=True,
            expert_quant=expert_quant,
            moe_weight_format=weight_format,
            has_swa_attention=False,
            has_linear_attention=False,
            num_layers=30,
            num_moe_layers=30,
            num_experts=128,
            rotary_config=SimpleNamespace(max_position=8192),
        ),
    )
    return config


def test_adjust_config_admits_explicit_gemma4_q4_resident() -> None:
    from freetoken.engine.engine import _adjust_config

    config = _engine_config()
    _adjust_config(config)
    assert config.moe_strategy == "fused"
    assert config.model_config.moe_strategy == "fused"


@pytest.mark.parametrize(
    ("model_type", "expert_quant", "weight_format"),
    [
        ("qwen3", "q4_0", "q4_0"),
        ("gemma4", "q4_0", "q4_k"),
    ],
)
def test_adjust_config_rejects_unsupported_q4_resident(
    model_type, expert_quant, weight_format
) -> None:
    from freetoken.engine.engine import _adjust_config

    config = _engine_config(
        model_type=model_type,
        expert_quant=expert_quant,
        weight_format=weight_format,
    )
    with pytest.raises(
        ValueError,
        match=r"q4_0 experts require --moe-strategy offload or cpu, got 'fused'",
    ):
        _adjust_config(config)


def test_gemma4_resident_q4_declares_packed_banks_without_dense_weights() -> None:
    _ensure_tp1()
    with torch.device("meta"):
        layer = Gemma4ResidentQ4MoELayer(
            num_experts=4,
            top_k=2,
            hidden_size=64,
            intermediate_size=32,
            activation="gelu_tanh",
            quant_config=None,
            prefix="model.layers.0.feed_forward.experts",
        )

    assert layer.gate_up_q.shape == (4, 64, 36)
    assert layer.down_q.shape == (4, 64, 18)
    assert layer.quant_method is None
    assert layer.gate_up_q.dtype == torch.uint8
    assert layer.down_q.dtype == torch.uint8


def test_gemma4_resident_q4_rejects_non_block_geometry() -> None:
    _ensure_tp1()
    with (
        pytest.raises(ValueError, match="divisible by the 32-value GGUF block size"),
        torch.device("meta"),
    ):
        Gemma4ResidentQ4MoELayer(
            num_experts=4,
            top_k=2,
            hidden_size=48,
            intermediate_size=32,
            activation="gelu_tanh",
            quant_config=None,
        )


class _PackedTensor:
    def __init__(
        self,
        name: str,
        packed_shape: tuple[int, ...],
        logical_shape: tuple[int, ...],
        ggml_type: int = GGML_Q4_0,
    ):
        self.name = name
        self.shape = logical_shape
        self.ggml_type = ggml_type
        values = torch.arange(prod(packed_shape), dtype=torch.int64)
        self._packed = values.to(torch.uint8).reshape(packed_shape)

    def packed(self) -> torch.Tensor:
        return self._packed


def _loader_config(num_layers: int = 30):
    return SimpleNamespace(
        num_layers=num_layers,
        num_experts=4,
        hidden_size=64,
        moe_intermediate_size=32,
        is_multimodal=False,
        attention_group_for_layer=lambda _layer: None,
    )


def _expert_tensors(config, *, ggml_type: int = GGML_Q4_0):
    tensors = []
    for layer in range(config.num_layers):
        tensors.extend(
            [
                _PackedTensor(
                    f"blk.{layer}.ffn_gate_up_exps.weight",
                    (
                        config.num_experts,
                        2 * config.moe_intermediate_size,
                        row_bytes(config.hidden_size, GGML_Q4_0),
                    ),
                    (
                        config.num_experts,
                        2 * config.moe_intermediate_size,
                        config.hidden_size,
                    ),
                    ggml_type,
                ),
                _PackedTensor(
                    f"blk.{layer}.ffn_down_exps.weight",
                    (
                        config.num_experts,
                        config.hidden_size,
                        row_bytes(config.moe_intermediate_size, GGML_Q4_0),
                    ),
                    (
                        config.num_experts,
                        config.hidden_size,
                        config.moe_intermediate_size,
                    ),
                    ggml_type,
                ),
            ]
        )
    return tensors


def _patch_loader(monkeypatch, config, tensors) -> None:
    import freetoken.models.gemma4.gguf as gemma_gguf
    from freetoken import utils
    from freetoken.models.gguf import reader

    monkeypatch.setattr(gemma_gguf, "parse_gguf_config", lambda _shim: config)
    monkeypatch.setattr(reader, "iter_gguf_tensors", lambda _path: iter(tensors))
    monkeypatch.setattr(utils, "cached_load_hf_config", lambda _path: object())


def test_resident_loader_yields_all_packed_expert_banks(monkeypatch) -> None:
    from freetoken.models.gemma4.gguf import iter_gguf_weights

    config = _loader_config()
    tensors = _expert_tensors(config)
    _patch_loader(monkeypatch, config, tensors)

    loaded = dict(
        iter_gguf_weights(
            "gemma4-q4.gguf",
            torch.device("cpu"),
            include_moe_experts=True,
            include_non_moe=True,
        )
    )

    expected_names = {
        f"model.layers.{layer}.feed_forward.experts.{role}"
        for layer in range(30)
        for role in ("gate_up_q", "down_q")
    }
    assert set(loaded) == expected_names
    assert all(tensor.dtype == torch.uint8 for tensor in loaded.values())
    gate_up = loaded["model.layers.0.feed_forward.experts.gate_up_q"]
    down = loaded["model.layers.29.feed_forward.experts.down_q"]
    assert gate_up.shape == (4, 64, 36)
    assert down.shape == (4, 64, 18)
    assert gate_up.data_ptr() == tensors[0].packed().data_ptr()


def test_gemma4_mlp_factory_and_loader_use_resident_expert_state_seam(
    monkeypatch,
) -> None:
    from freetoken.models.config import ModelConfig, RotaryConfig
    from freetoken.models.gemma4.gguf import iter_gguf_weights
    from freetoken.models.gemma4.moe import Gemma4MLP

    _ensure_tp1()
    config = ModelConfig(
        num_layers=1,
        num_qo_heads=2,
        num_kv_heads=1,
        head_dim=32,
        hidden_size=64,
        vocab_size=128,
        intermediate_size=32,
        rms_norm_eps=1e-6,
        rotary_config=RotaryConfig(32, 32, 8192, 1e4, None),
        hidden_act="gelu_tanh",
        tie_word_embeddings=True,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        norm_topk_prob=True,
        model_type="gemma4",
        architectures=["Gemma4ForConditionalGeneration"],
        moe_strategy="fused",
        moe_enabled=True,
        expert_quant="q4_0",
        moe_weight_format="q4_0",
    )
    prefix = "model.layers.0.feed_forward"
    with torch.device("meta"):
        mlp = Gemma4MLP(config, layer_id=0, prefix=prefix)

    assert isinstance(mlp.experts, Gemma4ResidentQ4MoELayer)
    state = mlp.state_dict(prefix=prefix)
    expert_keys = {key for key in state if key.startswith(f"{prefix}.experts.")}
    assert expert_keys == {
        f"{prefix}.experts.gate_up_q",
        f"{prefix}.experts.down_q",
    }

    tensors = _expert_tensors(config)
    _patch_loader(monkeypatch, config, tensors)
    emitted = dict(
        iter_gguf_weights(
            "gemma4-q4.gguf",
            torch.device("cpu"),
            include_moe_experts=True,
            include_non_moe=True,
        )
    )
    state.update(emitted)
    mlp.load_state_dict(state, prefix=prefix)

    assert state == {}
    assert mlp.experts.gate_up_q is emitted[f"{prefix}.experts.gate_up_q"]
    assert mlp.experts.down_q is emitted[f"{prefix}.experts.down_q"]


def test_offload_loader_still_skips_resident_expert_banks(monkeypatch) -> None:
    from freetoken.models.gemma4.gguf import iter_gguf_weights

    config = _loader_config(num_layers=1)
    _patch_loader(monkeypatch, config, _expert_tensors(config))
    loaded = list(
        iter_gguf_weights(
            "gemma4-q4.gguf",
            torch.device("cpu"),
            include_moe_experts=False,
            include_non_moe=True,
        )
    )
    assert loaded == []


def test_resident_loader_rejects_non_q4_experts(monkeypatch) -> None:
    from freetoken.models.gemma4.gguf import iter_gguf_weights

    config = _loader_config(num_layers=1)
    _patch_loader(monkeypatch, config, _expert_tensors(config, ggml_type=GGML_Q4_0 + 1))
    with pytest.raises(ValueError, match="require GGUF Q4_0"):
        list(
            iter_gguf_weights(
                "gemma4-q8.gguf",
                torch.device("cpu"),
                include_moe_experts=True,
                include_non_moe=True,
            )
        )


def test_resident_loader_rejects_noncanonical_expert_name(monkeypatch) -> None:
    from freetoken.models.gemma4.gguf import iter_gguf_weights

    config = _loader_config(num_layers=1)
    tensors = _expert_tensors(config)
    tensors[0].name = "blk.0.extra.ffn_gate_up_exps.weight"
    _patch_loader(monkeypatch, config, tensors)
    with pytest.raises(
        ValueError, match="noncanonical Gemma 4 resident expert tensor name"
    ):
        list(
            iter_gguf_weights(
                "gemma4-q4.gguf",
                torch.device("cpu"),
                include_moe_experts=True,
                include_non_moe=True,
            )
        )


def test_resident_loader_rejects_duplicate_expert_bank(monkeypatch) -> None:
    from freetoken.models.gemma4.gguf import iter_gguf_weights

    config = _loader_config(num_layers=1)
    tensors = _expert_tensors(config)
    _patch_loader(monkeypatch, config, [*tensors, tensors[0]])
    with pytest.raises(ValueError, match="duplicate Gemma 4 resident expert tensor"):
        list(
            iter_gguf_weights(
                "gemma4-q4.gguf",
                torch.device("cpu"),
                include_moe_experts=True,
                include_non_moe=True,
            )
        )


def test_resident_loader_rejects_missing_expert_bank(monkeypatch) -> None:
    from freetoken.models.gemma4.gguf import iter_gguf_weights

    config = _loader_config(num_layers=1)
    tensors = _expert_tensors(config)
    _patch_loader(monkeypatch, config, tensors[:1])
    with pytest.raises(
        ValueError,
        match="require one gate/up and down Q4_0 bank per layer",
    ):
        list(
            iter_gguf_weights(
                "gemma4-q4.gguf",
                torch.device("cpu"),
                include_moe_experts=True,
                include_non_moe=True,
            )
        )


def test_resident_loader_rejects_wrong_packed_byte_geometry(monkeypatch) -> None:
    from freetoken.models.gemma4.gguf import iter_gguf_weights

    config = _loader_config(num_layers=1)
    tensors = _expert_tensors(config)
    tensors[0]._packed = tensors[0]._packed[..., :-1]
    _patch_loader(monkeypatch, config, tensors)
    with pytest.raises(ValueError, match="has .* packed bytes; expected"):
        list(
            iter_gguf_weights(
                "gemma4-q4.gguf",
                torch.device("cpu"),
                include_moe_experts=True,
                include_non_moe=True,
            )
        )


def test_resident_loader_rejects_wrong_logical_shape_with_same_byte_count(
    monkeypatch,
) -> None:
    from freetoken.models.gemma4.gguf import iter_gguf_weights

    config = _loader_config(num_layers=1)
    tensors = _expert_tensors(config)
    tensors[0].shape = (config.num_experts, config.moe_intermediate_size, 128)
    _patch_loader(monkeypatch, config, tensors)
    with pytest.raises(ValueError, match="has logical shape .*; expected"):
        list(
            iter_gguf_weights(
                "gemma4-q4.gguf",
                torch.device("cpu"),
                include_moe_experts=True,
                include_non_moe=True,
            )
        )


def _pack_q4_0(nibbles: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    slots, out_features, in_features = nibbles.shape
    blocks = in_features // 32
    values = nibbles.reshape(slots, out_features, blocks, 32)
    packed = values[..., :16] | (values[..., 16:] << 4)
    scale_bytes = scale.to(torch.float16).view(torch.uint8).reshape(
        slots, out_features, blocks, 2
    )
    return torch.cat([scale_bytes, packed], dim=-1).reshape(
        slots, out_features, blocks * 18
    ).contiguous()


def _random_q4_bank(slots: int, out_features: int, in_features: int) -> torch.Tensor:
    nibbles = torch.randint(0, 16, (slots, out_features, in_features), dtype=torch.uint8)
    scales = 0.02 + 0.03 * torch.rand(slots, out_features, in_features // 32)
    return _pack_q4_0(nibbles, scales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA or ROCm")
def test_resident_q4_eager_and_graph_replay_match_packed_oracle() -> None:
    from freetoken.moe.fused_q4_0 import fused_experts_gguf_q4_0

    _ensure_tp1()
    torch.manual_seed(29)
    device = torch.device("cuda")
    layer = Gemma4ResidentQ4MoELayer(
        num_experts=4,
        top_k=2,
        hidden_size=64,
        intermediate_size=32,
        activation="gelu_tanh",
        quant_config=None,
    )
    layer.gate_up_q = _random_q4_bank(4, 64, 64).to(device)
    layer.down_q = _random_q4_bank(4, 64, 32).to(device)
    hidden = torch.randn(3, 64, device=device, dtype=torch.bfloat16)
    weights = torch.rand(3, 2, device=device, dtype=torch.float32)
    ids = torch.randint(0, 4, (3, 2), device=device, dtype=torch.int32)

    eager = layer.routed_forward(hidden, weights, ids)
    oracle = fused_experts_gguf_q4_0(
        hidden, layer.gate_up_q, layer.down_q, weights, ids, "gelu_tanh"
    )
    torch.testing.assert_close(eager, oracle, rtol=0, atol=0)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = layer.routed_forward(hidden, weights, ids)
    torch.cuda.synchronize()
    # Let PyTorch initialize its graph replay bookkeeping before measuring steady replay.
    graph.replay()
    torch.cuda.synchronize()

    for seed in (31, 37, 41):
        torch.manual_seed(seed)
        hidden.copy_(torch.randn(3, 64, dtype=torch.bfloat16))
        weights.copy_(torch.rand(3, 2, dtype=torch.float32))
        ids.copy_(torch.randint(0, 4, (3, 2), dtype=torch.int32))
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        expected = fused_experts_gguf_q4_0(
            hidden, layer.gate_up_q, layer.down_q, weights, ids, "gelu_tanh"
        )
        torch.testing.assert_close(replayed, expected, rtol=0, atol=0)
        del expected
