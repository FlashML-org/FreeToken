from __future__ import annotations

import pytest
import torch


def test_gguf_dispatch_uses_vec_for_small_batches_and_mat_for_large(monkeypatch):
    import freetoken.kernel.gguf as kernel
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_Q4_0

    calls = []

    def vec(weight, x, quant_type, row):
        calls.append(("vec", quant_type, row))
        return torch.zeros(x.shape[0], row)

    def mat(weight, x, quant_type, row):
        calls.append(("mat", quant_type, row))
        return torch.zeros(x.shape[0], row)

    monkeypatch.setattr(kernel, "ggml_mul_mat_vec_a8", vec)
    monkeypatch.setattr(kernel, "ggml_mul_mat_a8", mat)
    weight = torch.zeros(3, 18, dtype=torch.uint8)
    fused_mul_mat_gguf(torch.zeros(2, 32), weight, GGML_Q4_0)
    fused_mul_mat_gguf(torch.zeros(8, 32), weight, GGML_Q4_0)
    assert calls == [("vec", GGML_Q4_0, 3), ("mat", GGML_Q4_0, 3)]


def test_gguf_merged_linear_preserves_mixed_row_strides():
    from freetoken.layers.gguf import GGUFMergedLinear
    from freetoken.models.gguf.dequant import GGML_Q4_0, GGML_Q6_K

    merged = GGUFMergedLinear(256, [32, 64], [GGML_Q4_0, GGML_Q6_K])
    assert merged._mixed is True
    assert merged.qweight_0.shape == (32, 144)
    assert merged.qweight_1.shape == (64, 210)
    assert set(merged.state_dict()) == {"qweight_0", "qweight_1"}


def test_gguf_lm_head_gathers_explicit_last_tokens(monkeypatch):
    import freetoken.layers.gguf as layers
    from freetoken.models.gguf.dequant import GGML_Q4_0

    calls = []

    def fake_mat(x, qweight, quant_type):
        calls.append((x.shape, qweight.shape, quant_type))
        return torch.zeros(x.shape[0], qweight.shape[0])

    monkeypatch.setattr(layers, "fused_mul_mat_gguf", fake_mat)
    head = layers.GGUFLMHead(5, 32, GGML_Q4_0)
    x = torch.zeros(4, 32)
    out = head.forward(x, torch.tensor([1, 3]))
    assert out.shape == (2, 5)
    assert calls == [((2, 32), (5, 18), GGML_Q4_0)]


def test_unsupported_gguf_dispatch_fails_before_kernel_call():
    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_I8

    with pytest.raises(NotImplementedError, match="unsupported GGUF type I8"):
        fused_mul_mat_gguf(torch.zeros(1, 32), torch.zeros(2, 32, dtype=torch.uint8), GGML_I8)


def test_qwen35moe_gguf_architecture_dispatches_to_adapter(monkeypatch):
    from freetoken.models.gguf import config

    monkeypatch.setattr(config, "gguf_architecture", lambda _path: "qwen35moe")
    monkeypatch.setattr(config, "gguf_tensor_names", lambda _path: {"token_embd.weight", "output.weight"})
    monkeypatch.setattr(config, "_vocab_size", lambda _path: 2)
    monkeypatch.setattr(
        config,
        "load_gguf_metadata",
        lambda _path: {"general.architecture": "qwen35moe", "tokenizer.ggml.tokens": ["a", "b"]},
    )

    shim = config.build_gguf_shim("model.gguf")

    assert shim.architectures == ["Qwen35moeGGUFForCausalLM"]
    assert shim.vocab_size == 2
    assert shim.tie_word_embeddings is False
