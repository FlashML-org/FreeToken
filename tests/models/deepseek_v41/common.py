"""Shared fixtures for the deepseek_v41 tests: a tiny V4.1-shaped config (2 window layers, a ratio-2
encoder group with one Full + one Reuse layer, a ratio-1 decoder group with a Full candidate source and
a Reindex layer) and synthetic bf16 and FP8/MXFP4 checkpoints in the real tensor naming. Holds no tests itself."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

VOCAB = 512
DIM = 256
N_LAYERS = 6
RATIOS = [0, 0, 2, 2, 1, 1, 0]  # + one trailing MTP layer the model ignores
KV_SOURCES = [2, 4]
INDEX_SOURCES = [2, 4, 5]
ENGRAM_LAYERS = [1]
ENGRAM_VOCAB = 500  # bucket prime start; 3 orders x 2 heads ~= 3000 rows
ENGRAM_ROWS = 4096


def tiny_text_config(**over) -> dict:
    text = dict(
        model_type="deepseek_v41_text",
        vocab_size=VOCAB, hidden_size=DIM, moe_intermediate_size=64, num_hidden_layers=N_LAYERS, num_attention_heads=4,
        num_key_value_heads=1, head_dim=64, qk_rope_head_dim=16, q_lora_rank=64, o_lora_rank=32, o_groups=2,
        hidden_act="silu", swiglu_limit=10.0, rms_norm_eps=1e-20, max_position_embeddings=16384,
        rope_theta=10000, rope_scaling=dict(rope_type="yarn", factor=16, beta_fast=32, beta_slow=1, original_max_position_embeddings=1024),
        n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2, scoring_func="sqrtsoftplus", topk_method="noaux_tc",
        norm_topk_prob=True, routed_scaling_factor=1.5, sliding_window=128,
        compress_ratios=list(RATIOS), compress_rope_theta=160000, kv_source_layer_ids=list(KV_SOURCES),
        index_source_layer_ids=list(INDEX_SOURCES), index_n_heads=2, index_head_dim=32, index_topk=8,
        candidate_source_layer_id=4, candidate_topk_blocks=2, candidate_block_size=4,
        hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
        engram_layer_ids=list(ENGRAM_LAYERS), engram_num_embeddings=[ENGRAM_ROWS], engram_max_ngram_size=4,
        engram_vocab_size=ENGRAM_VOCAB, engram_n_heads=2, engram_head_dim=32, engram_pad_token_id=2,
        engram_compressed_vocab_size=VOCAB,  # the synthetic tokenizer maps every id to itself
        num_nextn_predict_layers=1, dspark_block_size=0, dspark_target_layer_ids=[],
    )
    text.update(over)
    return text


def tiny_hf_config(**over) -> SimpleNamespace:
    return SimpleNamespace(architectures=["DeepseekV41ForCausalLM"], model_type="deepseek_v41", text_config=tiny_text_config(**over))


def checkpoint_tensors(text: dict, seed: int = 0) -> dict[str, torch.Tensor]:
    """Random bf16 weights under the checkpoint's names (the reference ``convert.py`` dialect without
    quantization: no ``.scale`` tensors). hc / sink / router-bias / engram gates in fp32 as shipped."""
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale=0.02, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=g) * scale).to(dtype)

    dim, heads, hd, rd = text["hidden_size"], text["num_attention_heads"], text["head_dim"], text["qk_rope_head_dim"]
    qr, ol, og = text["q_lora_rank"], text["o_lora_rank"], text["o_groups"]
    inter, E, hc = text["moe_intermediate_size"], text["n_routed_experts"], text["hc_mult"]
    ih, ihd = text["index_n_heads"], text["index_head_dim"]
    mix = (2 + hc) * hc
    t: dict[str, torch.Tensor] = {
        "embed.weight": rnd(text["vocab_size"], dim, scale=1.0),
        "norm.weight": torch.ones(dim, dtype=torch.bfloat16),
        "head.weight": rnd(text["vocab_size"], dim),
    }
    for L in range(text["num_hidden_layers"]):
        a = f"layers.{L}.attn"
        t[f"{a}.wq_a.weight"] = rnd(qr, dim)
        t[f"{a}.q_norm.weight"] = torch.ones(qr, dtype=torch.bfloat16)
        t[f"{a}.wq_b.weight"] = rnd(heads * hd, qr)
        t[f"{a}.wkv.weight"] = rnd(hd, dim)
        t[f"{a}.kv_norm.weight"] = torch.ones(hd, dtype=torch.bfloat16)
        t[f"{a}.wo_a.weight"] = rnd(og * ol, heads * hd // og)
        t[f"{a}.wo_b.weight"] = rnd(dim, og * ol)
        t[f"{a}.attn_sink"] = rnd(heads, scale=0.5, dtype=torch.float32)
        ratio = text["compress_ratios"][L]
        if L in text["kv_source_layer_ids"]:
            t[f"{a}.compressor.wkv.weight"] = rnd(hd, dim)
            if ratio > 1:
                t[f"{a}.compressor.wgate.weight"] = rnd(hd, dim)
            t[f"{a}.compressor.norm.weight"] = torch.ones(hd, dtype=torch.bfloat16)
            t[f"{a}.indexer.wk.weight"] = rnd(ihd, hd)
            t[f"{a}.indexer.k_norm.weight"] = torch.ones(ihd, dtype=torch.bfloat16)
        if L in text["index_source_layer_ids"]:
            t[f"{a}.indexer.wq_b.weight"] = rnd(ih * ihd, qr)
            t[f"{a}.indexer.weights_proj.weight"] = rnd(ih, dim)
        p = f"layers.{L}"
        t[f"{p}.attn_norm.weight"] = torch.ones(dim, dtype=torch.bfloat16)
        t[f"{p}.ffn_norm.weight"] = torch.ones(dim, dtype=torch.bfloat16)
        t[f"{p}.ffn.gate.weight"] = rnd(E, dim)
        t[f"{p}.ffn.gate.bias"] = rnd(E, scale=0.1, dtype=torch.float32)
        t[f"{p}.ffn.gate.bias_vl"] = rnd(E, scale=0.1, dtype=torch.float32)
        for w, shape in (("w1", (inter, dim)), ("w2", (dim, inter)), ("w3", (inter, dim))):
            t[f"{p}.ffn.shared_experts.{w}.weight"] = rnd(*shape)
            for e in range(E):
                t[f"{p}.ffn.experts.{e}.{w}.weight"] = rnd(*shape)
        for nm in ("hc_attn_fn", "hc_ffn_fn"):
            t[f"{p}.{nm}"] = rnd(mix, hc * dim, scale=0.05, dtype=torch.float32)
        for nm in ("hc_attn_base", "hc_ffn_base"):
            t[f"{p}.{nm}"] = rnd(mix, scale=0.3, dtype=torch.float32)
        for nm in ("hc_attn_scale", "hc_ffn_scale"):
            t[f"{p}.{nm}"] = (torch.rand(3, generator=g) + 0.5).to(torch.float32)
        if L in text["engram_layer_ids"]:
            e = f"{p}.engram"
            cols = (text["engram_max_ngram_size"] - 1) * text["engram_n_heads"]
            t[f"{e}.wkv.weight"] = rnd(dim * (hc + 1), cols * text["engram_head_dim"])
            t[f"{e}.q_weight"] = torch.ones(hc, dim, dtype=torch.bfloat16)
            t[f"{e}.k_weight"] = torch.ones(hc, dim, dtype=torch.bfloat16)
            rows = text["engram_num_embeddings"][text["engram_layer_ids"].index(L)]
            t[f"{e}.embed.weight"] = rnd(rows, text["engram_head_dim"], scale=1.0).to(torch.float8_e4m3fn)
            t[f"{e}.embed.scale"] = torch.full((rows, text["engram_head_dim"] // 32), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    # the DSpark stage and the vision tower: present in a real checkpoint, dropped by the loader
    t["mtp.0.attn_norm.weight"] = torch.ones(dim, dtype=torch.bfloat16)
    t["vision.norm.weight"] = torch.ones(8, dtype=torch.bfloat16)
    return t


def write_tiny_checkpoint(folder: str, seed: int = 0, *, quantized: bool = False, **over) -> dict[str, torch.Tensor]:
    """A loadable checkpoint directory: HF-style config.json + one safetensors shard + index."""
    from safetensors.torch import save_file

    if quantized:
        over.setdefault("moe_intermediate_size", 256)
    text = tiny_text_config(**over)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "config.json"), "w") as f:
        config = {"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41", "text_config": text}
        if quantized:
            config["quantization_config"] = {
                "quant_method": "fp8", "activation_scheme": "dynamic",
                "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4",
            }
        json.dump(config, f)
    tensors = checkpoint_tensors(text, seed)
    if quantized:
        tensors = _quantize_checkpoint(tensors)
    save_file(tensors, os.path.join(folder, "model-00001-of-00001.safetensors"))
    with open(os.path.join(folder, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}}, f)
    return tensors


def _quantize_checkpoint(tensors):
    from .reference.kernel import _round_fp4

    out = dict(tensors)
    fp8_suffixes = (
        ".attn.wq_a", ".attn.wq_b", ".attn.wkv", ".attn.wo_a", ".attn.wo_b",
        ".indexer.wq_b", ".shared_experts.w1", ".shared_experts.w2", ".shared_experts.w3", ".engram.wkv",
    )
    grid = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    for name, weight in tensors.items():
        if not name.endswith(".weight"):
            continue
        stem = name.removesuffix(".weight")
        if ".ffn.experts." in name:
            groups = weight.float().unflatten(-1, (-1, 32))
            scale = torch.exp2(torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-12) / 6)))
            q = _round_fp4(groups / scale.unsqueeze(-1)).flatten(-2)
            codes = (q.abs().unsqueeze(-1) == grid).to(torch.int32).argmax(-1).to(torch.uint8)
            codes |= torch.signbit(q).to(torch.uint8) * 8
            out[name] = codes[:, 0::2] | (codes[:, 1::2] << 4)
            out[stem + ".scale"] = scale.to(torch.float8_e8m0fnu)
        elif stem.endswith(fp8_suffixes):
            n, k = weight.shape
            blocks = weight.float().reshape(n // 32, 32, k // 32, 32).permute(0, 2, 1, 3)
            scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax((-1, -2)).clamp_min(1e-4) / 448)))
            expanded = scale.repeat_interleave(32, 0).repeat_interleave(32, 1)
            out[name] = (weight.float() / expanded).to(torch.float8_e4m3fn)
            out[stem + ".scale"] = scale.to(torch.float8_e8m0fnu)
    return out
