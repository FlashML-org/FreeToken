"""Flint contracts checked against independent tensor math and checkpoint layouts."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from freetoken.models.flint.config import parse_config
from freetoken.models.flint.moe import FlintMoE


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch):
    from freetoken.distributed import info
    monkeypatch.setattr(info, "_TP_INFO", info.DistributedInfo(rank=0, size=1))


def _hf(**overrides):
    return SimpleNamespace(**(dict(
        architectures=["FlintForCausalLM"], model_type="flint", num_hidden_layers=3,
        hidden_size=64, head_dim=128, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=128, moe_intermediate_size=64,
        shared_expert_intermediate_size=128, vocab_size=128, max_position_embeddings=2048,
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        no_rope_layer_indices=[0, 2], mlp_only_layers=[0],
        hidden_act="silu", router_score_function="sigmoid", softmax_type="learnable",
        norm_topk_prob=True, num_experts=8, num_experts_per_tok=2,
        rms_norm_eps=1e-6, sliding_window=513, rope_theta=10000., tie_word_embeddings=False,
    ) | overrides))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused CUDA router")
@pytest.mark.parametrize("renormalize", [False, True])
def test_sigmoid_routing_bias_affects_selection_only(renormalize):
    from freetoken.layers import LinearReplicated
    torch.manual_seed(42)
    block = FlintMoE.__new__(FlintMoE)
    block.top_k, block.renormalize = 6, renormalize
    block.gate = LinearReplicated(128, 256, has_bias=False)
    block.gate.weight = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) * .1
    block.gate.expert_bias = torch.randn(256, device="cuda") * .05
    x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
    weights, ids = block._route(x)
    logits = F.linear(x, block.gate.weight).float()
    scores = 1. / (1. + torch.exp(-logits))
    expected_ids = torch.argsort(scores + block.gate.expert_bias, dim=-1, descending=True)[:, :6]
    assert [set(row) for row in ids.tolist()] == [set(row) for row in expected_ids.tolist()]
    expected = scores.gather(1, ids.long())
    if renormalize:
        expected = expected / expected.sum(1, keepdim=True)
    torch.testing.assert_close(weights, expected, rtol=1e-6, atol=1e-7)
    assert ids.dtype == torch.int32


def test_weight_reader_merges_cross_shard_projections_and_preserves_sinks(tmp_path):
    from safetensors.torch import save_file
    from freetoken.models.flint.weight import iter_weights
    hf = _hf()
    (tmp_path / "config.json").write_text(json.dumps(vars(hf)))
    q = torch.randn(512, 64, dtype=torch.bfloat16)
    k = torch.randn(256, 64, dtype=torch.bfloat16)
    v = torch.randn(256, 64, dtype=torch.bfloat16)
    pre = "model.layers.0.self_attn"
    sink = torch.tensor([.125, .25, .5, 1.])
    save_file({f"{pre}.q_proj.weight": q, f"{pre}.softmax_offset": sink}, tmp_path / "a.safetensors")
    save_file({f"{pre}.k_proj.weight": k, f"{pre}.v_proj.weight": v}, tmp_path / "b.safetensors")
    weights = dict(iter_weights(str(tmp_path), torch.device("cpu"), include_moe_experts=False, include_non_moe=True))
    torch.testing.assert_close(weights[f"{pre}.qkv_proj.weight"], torch.cat([q, k, v]))
    torch.testing.assert_close(weights[f"{pre}.softmax_offset"], sink)
    for i in (1, 2):
        assert not weights[f"model.layers.{i}.mlp.gate.expert_bias"].any()


def test_gguf_explicit_end_of_generation_ids_are_not_lost():
    from freetoken.models.gguf.tokenizer import gguf_eos_token_ids
    tokens = ["x"] * 19
    for i, token in [(1, "<|end_of_text|>"), (3, "<|user|>"), (5, "<|observation|>"), (18, "<|developer|>")]:
        tokens[i] = token
    meta = {"tokenizer.ggml.tokens": tokens, "tokenizer.ggml.eos_token_id": 1,
            "tokenizer.ggml.eog_token_ids": [1, 3, 5, 18]}
    with patch("freetoken.models.gguf.tokenizer.load_gguf_metadata", return_value=meta):
        assert gguf_eos_token_ids("test.gguf") == {1, 3, 5, 18}


@pytest.mark.needs_weights
def test_official_gguf_matches_hf_tokenizer_and_config():
    from transformers import AutoTokenizer
    from freetoken.models.gguf.config import build_gguf_shim
    from freetoken.models.gguf.tokenizer import load_gguf_tokenizer, gguf_eos_token_ids
    from freetoken.models.flint.gguf import parse_gguf_config
    from freetoken.utils import cached_load_hf_config
    gguf_path = os.environ.get("FREETOKEN_FLINT_GGUF")
    hf_path = os.environ.get("FREETOKEN_FLINT_MODEL")
    if not gguf_path or not hf_path:
        pytest.skip("set FREETOKEN_FLINT_GGUF and FREETOKEN_FLINT_MODEL")
    hf = AutoTokenizer.from_pretrained(hf_path, local_files_only=True)
    gg = load_gguf_tokenizer(gguf_path)
    texts = [
        "HelloWorld camelCase XMLHTTPRequest don't we're I'LL",
        "1234567890 3.1415926535 -42 1e-10",
        "漢字かなカナ 中文测试 café e\u0301 👩‍💻",
        "def f(x):\n\treturn {'value': x + 1}\n\n# EOF   ",
        "<|user|>hello<|assistant|><think>check</think>42<|end_of_text|>",
    ]
    for text in texts:
        assert gg.encode(text, add_special_tokens=False) == hf.encode(text, add_special_tokens=False), text
    tools = [{"type": "function", "function": {"name": "read", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"],
    }}}]
    messages = [
        {"role": "user", "content": "Read the file."},
        {"role": "assistant", "content": "", "reasoning_content": "Inspect it first.",
         "tool_calls": [{"type": "function", "function": {"name": "read", "arguments": {"path": "/tmp/a.py"}}}]},
        {"role": "tool", "content": "print(42)"},
    ]
    assert gg.apply_chat_template(messages, tools=tools, add_generation_prompt=True) == hf.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True,
    )
    assert gguf_eos_token_ids(gguf_path) == {150001, 150003, 150005, 150018}
    cfg_hf = parse_config(cached_load_hf_config(hf_path))
    cfg_gg = parse_gguf_config(build_gguf_shim(gguf_path))
    for name in ("head_dim", "num_qo_heads", "num_kv_heads", "num_layers", "hidden_size",
                 "num_experts", "num_experts_per_tok", "moe_intermediate_size",
                 "shared_expert_intermediate_size", "first_k_dense_replace", "attention_groups"):
        assert getattr(cfg_hf, name) == getattr(cfg_gg, name), name
