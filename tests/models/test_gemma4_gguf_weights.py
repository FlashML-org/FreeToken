from types import SimpleNamespace

import torch

from freetoken.models.gemma4 import model as gemma4_model
from freetoken.models.gemma4.gguf import (
    _kv_head_counts,
    _native_qweight_rows,
    _rename_hf_nvfp4_tensor_name,
    _shared_kv_sources,
    _to_bf16,
    parse_gguf_config,
    to_gguf,
)
from freetoken.models.gguf.dequant import GGML_F32, GGML_Q8_0


def test_causal_lm_applies_gguf_module_conversion(monkeypatch):
    calls = []
    monkeypatch.setattr(
        gemma4_model,
        "Gemma4Model",
        lambda _: SimpleNamespace(embed_tokens=SimpleNamespace()),
    )
    monkeypatch.setattr(gemma4_model, "ParallelLMHead", lambda **_: SimpleNamespace())
    monkeypatch.setattr(
        "freetoken.models.gemma4.gguf.convert_gemma4_to_gguf",
        lambda model, config: calls.append((model, config)),
    )
    config = SimpleNamespace(
        vocab_size=8,
        hidden_size=4,
        embedding_scale=None,
        quant=None,
        tie_word_embeddings=True,
        final_logit_softcapping=None,
        moe_weight_format="nvfp4",
    )

    model = gemma4_model.Gemma4ForCausalLM(config)

    assert calls == [(model, config)]


def test_to_bf16_converts_float_gguf_tensor():
    source = torch.tensor([1.25, -2.0], dtype=torch.float32)
    tensor = SimpleNamespace(
        packed=lambda: source.view(torch.uint8),
        ggml_type=GGML_F32,
        shape=source.shape,
    )

    result = _to_bf16(tensor)

    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result, source.to(torch.bfloat16))


def test_hf_nvfp4_weight_loader_maps_router_scales():
    assert _rename_hf_nvfp4_tensor_name("blk.0.ffn_gate_inp.scale") == (
        "model.layers.0.feed_forward.router.scale"
    )
    assert _rename_hf_nvfp4_tensor_name("blk.0.ffn_down_exps.scale") == (
        "model.layers.0.feed_forward.router.per_expert_scale"
    )


def test_q8_expert_scale_check_uses_shared_gguf_helper():
    from freetoken.models.gemma4 import gguf

    assert gguf.weight_scale(SimpleNamespace(ggml_type=GGML_Q8_0)) is None


def test_gguf_expert_loader_reports_completed_layers(monkeypatch):
    from freetoken import distributed
    from freetoken.models.gemma4 import gguf
    from freetoken.models.gguf import reader
    from freetoken.models.gguf.dequant import GGML_Q4_0

    class TensorInfo:
        def __init__(self, name, rows):
            self.name = name
            self.row_bytes = 2
            self.ggml_type = GGML_Q4_0
            self.ne = (256, rows, 2)
            self.path = "missing-test-file.gguf"
            self._packed = torch.zeros(2 * rows * self.row_bytes, dtype=torch.uint8)

        def packed(self):
            return self._packed

    class Progress:
        def __init__(self, iterable=None, *, total=None, **kwargs):
            self.iterable = iterable
            self.total = total
            self.n = 0

        def __enter__(self):
            progress_bars.append(self)
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return iter(self.iterable)

        def update(self, count=1):
            self.n += count

    progress_bars = []
    infos = []
    index = {}
    for layer_id in range(2):
        gate_up = TensorInfo(f"blk.{layer_id}.ffn_gate_up_exps.weight", 4)
        down = TensorInfo(f"blk.{layer_id}.ffn_down_exps.weight", 4)
        infos.extend((gate_up, down))
        index[gate_up.name] = gate_up
        index[down.name] = down

    monkeypatch.setattr(
        distributed,
        "get_tp_info",
        lambda: SimpleNamespace(is_primary=lambda: True),
    )
    monkeypatch.setattr(gguf, "_require_tp1", lambda _: None)
    monkeypatch.setattr(gguf, "gguf_tensor_index", lambda _: index)
    monkeypatch.setattr(reader, "iter_gguf_tensors", lambda _: iter(infos))
    monkeypatch.setattr(gguf, "tqdm", Progress)

    sources = gguf.load_q4_0_expert_sources(
        "test.gguf",
        SimpleNamespace(
            num_layers=2,
            num_experts=2,
            hidden_size=4,
            moe_intermediate_size=2,
        ),
        layer_sink=lambda *_: None,
    )

    assert len(sources["gate_up"]) == len(sources["down"]) == 2
    assert len(progress_bars) == 1
    assert progress_bars[0].total == progress_bars[0].n == 2


def test_gemma4_gguf_scalar_and_per_layer_kv_head_counts():
    assert _kv_head_counts(2, 3) == (2, 2, 2)
    assert _kv_head_counts([2, 4, 2], 3) == (2, 4, 2)


def test_shared_kv_uses_last_preceding_layer_of_each_attention_type():
    assert _shared_kv_sources([True, True, False, True, False, True], 3) == (
        None,
        None,
        None,
        1,
        2,
        1,
    )
    assert _shared_kv_sources([True, False], 0) == ()


def test_native_float_qweight_preserves_gguf_bytes():
    source = torch.tensor([[1.25, -2.0]], dtype=torch.bfloat16)
    packed = _native_qweight_rows(source)

    assert packed.dtype == torch.uint8
    torch.testing.assert_close(packed.contiguous().view(torch.bfloat16), source)


def test_to_gguf_maps_per_layer_embeddings_and_projections():
    tensors = {
        "per_layer_token_embd.weight": 14,
        "per_layer_model_proj.weight": 30,
        "per_layer_proj_norm.weight": 0,
        "blk.3.inp_gate.weight": 14,
        "blk.3.proj.weight": 14,
        "blk.3.post_norm.weight": 0,
    }

    assert to_gguf("model.embed_tokens_per_layer", tensors) == (
        "per_layer_token_embd",
    )
    assert to_gguf("model.per_layer_model_projection", tensors) == (
        "per_layer_model_proj",
    )
    assert to_gguf("model.per_layer_projection_norm", tensors) == (
        "per_layer_proj_norm",
    )
    assert to_gguf("model.layers.3.per_layer_input_gate", tensors) == (
        "blk.3.inp_gate",
    )
    assert to_gguf("model.layers.3.per_layer_projection", tensors) == (
        "blk.3.proj",
    )
    assert to_gguf("model.layers.3.post_per_layer_input_norm", tensors) == (
        "blk.3.post_norm",
    )


def test_parse_gguf_config_supports_dense_ple_and_shared_kv(monkeypatch):
    from freetoken.models.gemma4 import gguf

    metadata = {
        "gemma4.block_count": 4,
        "gemma4.embedding_length": 8,
        "gemma4.attention.head_count": 2,
        "gemma4.attention.head_count_kv": 1,
        "gemma4.attention.sliding_window_pattern": [True, False, True, False],
        "gemma4.attention.key_length": 4,
        "gemma4.attention.key_length_swa": 2,
        "gemma4.attention.shared_kv_layers": 2,
        "gemma4.embedding_length_per_layer_input": 2,
        "gemma4.vocab_size_per_layer_input": 16,
        "gemma4.context_length": 32,
        "gemma4.rope.freq_base": 10000.0,
        "gemma4.rope.dimension_count_swa": 2,
        "gemma4.rope.freq_base_swa": 10000.0,
        "gemma4.feed_forward_length": 16,
        "gemma4.attention.layer_norm_rms_epsilon": 1e-6,
        "gemma4.attention.sliding_window": 4,
        "gemma4.final_logit_softcapping": 30.0,
    }
    tensor_shapes = {
        "per_layer_token_embd.weight": (16, 8),
        "per_layer_model_proj.weight": (8, 8),
        "per_layer_proj_norm.weight": (2,),
    }
    for layer_id in range(4):
        tensor_shapes.update(
            {
                f"blk.{layer_id}.inp_gate.weight": (2, 8),
                f"blk.{layer_id}.proj.weight": (8, 2),
                f"blk.{layer_id}.post_norm.weight": (8,),
            }
        )
    tensor_index = {
        name: SimpleNamespace(shape=shape, ggml_type=14)
        for name, shape in tensor_shapes.items()
    }
    shim = SimpleNamespace(
        metadata=metadata,
        model_path="test.gguf",
        vocab_size=16,
        tie_word_embeddings=True,
        architectures=["Gemma4GGUFForCausalLM"],
        is_hf_nvfp4=False,
    )
    monkeypatch.setattr(gguf, "check_scales", lambda *args, **kwargs: None)
    monkeypatch.setattr(gguf, "_full_rotary_dim", lambda *_: 2)
    monkeypatch.setattr(gguf, "gguf_tensor_index", lambda _: tensor_index)
    monkeypatch.setattr(gguf, "_gguf_expert_quant", lambda _: "q4_0")
    monkeypatch.setattr(gguf, "iter_gguf_tensors", lambda _: [])

    config = parse_gguf_config(shim)

    assert not config.moe_enabled
    assert config.num_experts == 0
    assert config.per_layer_input_dim == 2
    assert config.gguf_per_layer_embedding_type == 14
    assert config.shared_kv_source_layers == (None, None, 0, 1)
