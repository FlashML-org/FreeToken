"""End-to-end Laguna GGUF weight loading over a synthetic tiny checkpoint.

Writes a real (tiny) laguna GGUF with gguf-py -- Q8_0 quantized projections and
expert banks, F32 norms/router -- then exercises config parsing, the name map,
``iter_gguf_weights`` (every tensor consumed exactly once, fused buffers built),
deferred materialization, and the mixed-type expert-bank loader.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import gguf

import freetoken.distributed.info as di
from freetoken.models.gguf.dequant import GGML_Q8_0, row_bytes

# Tiny geometry: 4 layers (full at 0), heads 4 full / 6 swa, kv 2, head_dim 32.
L, H, FF = 4, 64, 96
HEADS = [4, 6, 6, 6]
KV, HD = 2, 32
E, TOPK, I, SHI = 8, 3, 32, 32
VOCAB = 128


@pytest.fixture(scope="module")
def tiny_gguf(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("laguna") / "tiny-laguna.gguf")
    w = gguf.GGUFWriter(path, "laguna")
    w.add_block_count(L)
    w.add_context_length(4096)
    w.add_embedding_length(H)
    w.add_feed_forward_length(FF)
    w.add_head_count(HEADS)
    w.add_head_count_kv(KV)
    w.add_key_length(HD)
    w.add_value_length(HD)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_sliding_window(512)
    w.add_rope_freq_base(500000.0)
    w.add_rope_dimension_count(16)
    # SWA rope mirrors + yarn keys (raw kv names, mirroring the real file).
    w.add_float32("laguna.rope.freq_base_swa", 10000.0)
    w.add_uint32("laguna.rope.dimension_count_swa", HD)
    w.add_string("laguna.rope.scaling.type", "yarn")
    w.add_float32("laguna.rope.scaling.factor", 32.0)
    w.add_uint32("laguna.rope.scaling.original_context_length", 8192)
    w.add_float32("laguna.rope.scaling.yarn_attn_factor", 1.0)
    w.add_float32("laguna.rope.scaling.yarn_beta_fast", 32.0)
    w.add_float32("laguna.rope.scaling.yarn_beta_slow", 1.0)
    w.add_expert_count(E)
    w.add_expert_used_count(TOPK)
    w.add_expert_feed_forward_length(I)
    w.add_expert_shared_feed_forward_length(SHI)
    w.add_bool("laguna.expert_weights_norm", True)
    w.add_float32("laguna.expert_weights_scale", 2.5)
    w.add_uint32("laguna.expert_gating_func", 2)
    w.add_uint32("laguna.leading_dense_block_count", 1)
    w.add_uint32("laguna.vocab_size", VOCAB)
    # Minimal gpt2 tokenizer metadata so the shim can size the vocab.
    w.add_tokenizer_model("gpt2")
    w.add_token_list([f"<t{i}>" for i in range(VOCAB)])
    w.add_token_types([1] * VOCAB)
    w.add_token_merges([])
    w.add_bos_token_id(2)
    w.add_eos_token_id(2)
    w.add_uint32("tokenizer.ggml.eot_token_id", 24)

    rng = np.random.default_rng(0)
    q8 = gguf.GGMLQuantizationType.Q8_0

    def quant(name, rows, cols):
        data = rng.standard_normal((rows, cols)).astype(np.float32)
        w.add_tensor(name, gguf.quants.quantize(data, q8), raw_dtype=q8)

    def f32(name, *shape):
        w.add_tensor(name, rng.standard_normal(shape).astype(np.float32))

    quant("token_embd.weight", VOCAB, H)
    quant("output.weight", VOCAB, H)
    f32("output_norm.weight", H)
    for i in range(L):
        p = f"blk.{i}."
        nh = HEADS[i]
        f32(p + "attn_norm.weight", H)
        f32(p + "attn_q_norm.weight", HD)
        f32(p + "attn_k_norm.weight", HD)
        f32(p + "ffn_norm.weight", H)
        quant(p + "attn_q.weight", nh * HD, H)
        quant(p + "attn_k.weight", KV * HD, H)
        quant(p + "attn_v.weight", KV * HD, H)
        quant(p + "attn_output.weight", H, nh * HD)
        quant(p + "attn_gate.weight", nh, H)
        if i == 0:
            quant(p + "ffn_gate.weight", FF, H)
            quant(p + "ffn_up.weight", FF, H)
            quant(p + "ffn_down.weight", H, FF)
        else:
            f32(p + "ffn_gate_inp.weight", E, H)
            f32(p + "exp_probs_b.bias", E)
            quant(p + "ffn_gate_shexp.weight", SHI, H)
            quant(p + "ffn_up_shexp.weight", SHI, H)
            quant(p + "ffn_down_shexp.weight", H, SHI)
            for role, rows, cols in (
                ("ffn_gate_exps", I, H),
                ("ffn_up_exps", I, H),
                ("ffn_down_exps", H, I),
            ):
                data = rng.standard_normal((E, rows, cols)).astype(np.float32)
                w.add_tensor(
                    p + role + ".weight",
                    gguf.quants.quantize(data.reshape(E * rows, cols), q8).reshape(E, rows, -1),
                    raw_dtype=q8,
                )
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


@pytest.fixture(autouse=True)
def _tp1():
    try:
        di.get_tp_info()
    except RuntimeError:
        di.set_tp_info(0, 1)


def _config(path):
    from freetoken.models.gguf.config import build_gguf_shim
    from freetoken.models.laguna.gguf import parse_gguf_config

    return parse_gguf_config(build_gguf_shim(path))


def test_config_and_expert_types(tiny_gguf):
    cfg = _config(tiny_gguf)
    assert cfg.num_layers == L and cfg.num_qo_heads == 6
    assert cfg.gguf_embed_quant == GGML_Q8_0
    assert cfg.expert_quant == "gguf" and cfg.moe_weight_format == "gguf"
    assert cfg.gguf_expert_types == ((GGML_Q8_0, GGML_Q8_0),) * (L - 1)


def test_iter_weights_complete_and_fused(tiny_gguf):
    from freetoken.models.laguna.gguf import iter_gguf_weights

    got = dict(iter_gguf_weights(tiny_gguf, "cpu", include_moe_experts=False, include_non_moe=True))
    # split q/k/v per layer (mixed-type files forbid fusing)
    for i, nh in enumerate(HEADS):
        assert got[f"model.layers.{i}.self_attn.q_proj.qweight"].shape == (nh * HD, row_bytes(H, GGML_Q8_0))
        assert got[f"model.layers.{i}.self_attn.k_proj.qweight"].shape == (KV * HD, row_bytes(H, GGML_Q8_0))
        assert got[f"model.layers.{i}.self_attn.v_proj.qweight"].shape == (KV * HD, row_bytes(H, GGML_Q8_0))
    t = got["model.layers.0.mlp.gate_up_proj.qweight"]
    assert t.shape == (2 * FF, row_bytes(H, GGML_Q8_0))
    t = got["model.layers.1.mlp.shared_experts.gate_up_proj.qweight"]
    assert t.shape == (2 * SHI, row_bytes(H, GGML_Q8_0))
    assert got["model.layers.1.mlp.gate.weight"].dtype == torch.float32
    assert got["model.layers.1.mlp.e_score_correction_bias"].dtype == torch.float32
    assert got["model.layers.0.ffn_norm.weight"].dtype == torch.bfloat16
    assert got["lm_head.qweight"].shape == (VOCAB, row_bytes(H, GGML_Q8_0))


def test_unknown_tensor_rejected(tiny_gguf, tmp_path):
    from freetoken.models.laguna.gguf import iter_gguf_weights

    w = gguf.GGUFWriter(str(tmp_path / "bad.gguf"), "laguna")
    w.add_block_count(1)
    w.add_tensor("blk.0.mystery.weight", np.zeros((4, 4), dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    with pytest.raises(ValueError, match="mystery"):
        list(iter_gguf_weights(str(tmp_path / "bad.gguf"), "cpu",
                               include_moe_experts=False, include_non_moe=True))


def test_expert_bank_loader(tiny_gguf):
    from freetoken.models.laguna.gguf import _expert_bank_geometry, load_gguf_expert_sources

    cfg = _config(tiny_gguf)
    banks = load_gguf_expert_sources(tiny_gguf, cfg)
    gu_s, dn_s = _expert_bank_geometry(cfg)
    assert len(banks["gate_up"]) == L - 1 and len(banks["down"]) == L - 1
    for t in banks["gate_up"]:
        assert t.shape == (E, gu_s) and t.dtype == torch.uint8
    # payload bytes decode to the source values via gguf-py
    half = I * row_bytes(H, GGML_Q8_0)
    blob = banks["gate_up"][0][:, : 2 * half]
    dec = gguf.quants.dequantize(
        np.ascontiguousarray(blob.reshape(E * 2 * I, -1).numpy()),
        gguf.GGMLQuantizationType.Q8_0,
    )
    assert np.isfinite(dec).all() and dec.std() > 0.5  # real data, not padding


def test_deferred_materialization(tiny_gguf):
    cfg = _config(tiny_gguf)
    from freetoken.models.laguna.gguf import DeferredGGUFLinear

    # conversion materializes from the file's tensor table; emulate on one module
    mod = DeferredGGUFLinear(H, 6 * HD)
    mod.materialize(GGML_Q8_0)
    assert mod.qweight.shape == (6 * HD, row_bytes(H, GGML_Q8_0))
    assert cfg.gguf_model_path == tiny_gguf
