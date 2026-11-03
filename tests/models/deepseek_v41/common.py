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


def checkpoint_tensors(text: dict, seed: int = 0, *, random_gate: bool = False) -> dict[str, torch.Tensor]:
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
            t[f"{e}.q_weight"] = rnd(hc, dim, scale=1.0) if random_gate else torch.ones(hc, dim, dtype=torch.bfloat16)
            t[f"{e}.k_weight"] = rnd(hc, dim, scale=1.0) if random_gate else torch.ones(hc, dim, dtype=torch.bfloat16)
            rows = text["engram_num_embeddings"][text["engram_layer_ids"].index(L)]
            t[f"{e}.embed.weight"] = rnd(rows, text["engram_head_dim"], scale=1.0).to(torch.float8_e4m3fn)
            t[f"{e}.embed.scale"] = torch.full((rows, text["engram_head_dim"] // 32), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    # the DSpark stage and the vision tower: present in a real checkpoint, dropped by the loader
    t["mtp.0.attn_norm.weight"] = torch.ones(dim, dtype=torch.bfloat16)
    t["vision.norm.weight"] = torch.ones(8, dtype=torch.bfloat16)
    return t


def write_tiny_checkpoint(folder: str, seed: int = 0, *, quantized: bool = False, random_gate: bool = False, **over) -> dict[str, torch.Tensor]:
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
    tensors = checkpoint_tensors(text, seed, random_gate=random_gate)
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


_GGUF_ATTN = {
    "wq_a.weight": "attn_q_a", "q_norm.weight": "attn_q_a_norm", "wq_b.weight": "attn_q_b", "wkv.weight": "attn_kv",
    "kv_norm.weight": "attn_kv_a_norm", "wo_a.weight": "attn_output_a", "wo_b.weight": "attn_output_b", "attn_sink": "attn_sinks",
    "compressor.wkv.weight": "attn_compressor_kv", "compressor.wgate.weight": "attn_compressor_gate", "compressor.norm.weight": "attn_compressor_norm",
    "indexer.wk.weight": "indexer.attn_k", "indexer.k_norm.weight": "indexer.k_norm", "indexer.wq_b.weight": "indexer.attn_q_b", "indexer.weights_proj.weight": "indexer.proj",
}
_GGUF_LAYER = {
    "attn_norm.weight": "attn_norm", "ffn_norm.weight": "ffn_norm", "ffn.gate.weight": "ffn_gate_inp", "ffn.gate.bias": "exp_probs_b.bias", "ffn.gate.bias_vl": "exp_probs_b_vl.bias",
    "ffn.shared_experts.w1.weight": "ffn_gate_shexp", "ffn.shared_experts.w3.weight": "ffn_up_shexp", "ffn.shared_experts.w2.weight": "ffn_down_shexp",
    "hc_attn_fn": "hc_attn_fn", "hc_attn_base": "hc_attn_base", "hc_attn_scale": "hc_attn_scale", "hc_ffn_fn": "hc_ffn_fn", "hc_ffn_base": "hc_ffn_base", "hc_ffn_scale": "hc_ffn_scale",
    "engram.wkv.weight": "engram_kv",
}


def _pack_q8_0(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[n, k]`` floats to ggml Q8_0 blocks of 32 values: ``[n, k // 32 * 34]`` bytes (fp16 scale, then int8s), and the values they decode to."""
    n = rows.shape[0]
    blocks = rows.float().view(-1, 32)
    d = (blocks.abs().amax(-1, keepdim=True).clamp_min(1e-3) / 127).to(torch.float16)
    q = torch.round(blocks / d.float()).clamp(-127, 127).to(torch.int8)
    packed = torch.cat([d.view(torch.uint8).view(-1, 2), q.view(torch.uint8)], dim=-1).view(n, -1)
    return packed, (q.float() * d.float()).view(n, -1)


def _write_gguf(path: str, kvs: list[tuple[str, int, object, int | None]], tensors: list[tuple[str, tuple[int, ...], int, bytes]]) -> None:
    """A little-endian GGUF v3 file. ``kvs``: (key, value type, value, array element type); ``tensors``: (name, torch-order shape, ggml type, bytes)."""
    import struct

    from freetoken.third_party.gguf import GGUFValueType as K

    fmt = {K.UINT32: "<I", K.INT32: "<i", K.FLOAT32: "<f", K.BOOL: "<?", K.UINT64: "<Q"}

    def string(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    def value(kind, val, sub) -> bytes:
        if kind == K.STRING:
            return string(val)
        if kind == K.ARRAY:
            return struct.pack("<IQ", sub, len(val)) + b"".join(value(sub, v, None) for v in val)
        return struct.pack(fmt[kind], val)

    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs))
    for key, kind, val, sub in kvs:
        head += string(key) + struct.pack("<I", kind) + value(kind, val, sub)
    offset, blobs = 0, []
    for name, shape, ggml_type, data in tensors:
        head += string(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", x) for x in reversed(shape))
        head += struct.pack("<IQ", ggml_type, offset)
        pad = -len(data) % 32
        blobs.append(data + b"\0" * pad)
        offset += len(data) + pad
    with open(path, "wb") as f:
        f.write(head + b"\0" * (-len(head) % 32))
        for blob in blobs:
            f.write(blob)


def write_tiny_gguf(path: str, seed: int = 0, *, table_type: str = "BF16", random_gate: bool = False, kv_over: dict | None = None, **over) -> dict[int, torch.Tensor]:
    """The tiny checkpoint as llama.cpp's converter lays it out (conversion/deepseek41.py), every tensor float except an optional Q8_0 ``table_type`` table;
    ``kv_over`` replaces array KVs. Returns each Engram layer's rows as they decode, keyed by layer."""
    import numpy as np

    from freetoken.models.deepseek_v41.args import DeepseekV41Args
    from freetoken.models.deepseek_v41.engram import bucket_primes, hash_multipliers
    from freetoken.third_party.gguf import ExpertGatingFuncType, GGMLQuantizationType as Q, GGUFValueType as K

    text = tiny_text_config(**over)
    tensors = checkpoint_tensors(text, seed, random_gate=random_gate)
    args = DeepseekV41Args.from_hf(SimpleNamespace(text_config=text))
    sizes = [p for layer in bucket_primes(args) for per_ngram in layer for p in per_ngram]
    rs = text["rope_scaling"]
    limit = [float(text["swiglu_limit"])] * text["num_hidden_layers"]
    kvs: list[tuple[str, int, object, int | None]] = [("general.architecture", K.STRING, "deepseek41", None)]

    def add_kv(key, kind, value, sub=None):
        kvs.append((f"deepseek41.{key}", kind, (kv_over or {}).get(key, value), sub))

    for key, value in {
        "block_count": text["num_hidden_layers"], "context_length": text["max_position_embeddings"], "embedding_length": text["hidden_size"],
        "expert_feed_forward_length": text["moe_intermediate_size"], "expert_count": text["n_routed_experts"], "expert_used_count": text["num_experts_per_tok"],
        "expert_shared_count": text["n_shared_experts"], "expert_gating_func": int(ExpertGatingFuncType.SQRTSOFTPLUS), "hash_layer_count": 0,
        "attention.head_count": text["num_attention_heads"], "attention.head_count_kv": 1, "attention.key_length": text["head_dim"], "attention.value_length": text["head_dim"],
        "attention.q_lora_rank": text["q_lora_rank"], "attention.sliding_window": text["sliding_window"], "attention.output_group_count": text["o_groups"],
        "attention.output_lora_rank": text["o_lora_rank"], "rope.dimension_count": text["qk_rope_head_dim"], "rope.scaling.original_context_length": rs["original_max_position_embeddings"],
        "attention.indexer.head_count": text["index_n_heads"], "attention.indexer.key_length": text["index_head_dim"], "attention.indexer.top_k": text["index_topk"],
        "hyper_connection.count": text["hc_mult"], "hyper_connection.sinkhorn_iterations": text["hc_sinkhorn_iters"],
        "attention.indexer.candidate_top_k_blocks": text["candidate_topk_blocks"], "attention.indexer.candidate_block_size": text["candidate_block_size"],
        "ple.ngram_size": text["engram_max_ngram_size"], "ple.heads_per_ngram": text["engram_n_heads"], "embedding_length_per_layer_input": text["engram_head_dim"],
        "engram.pad_token_id": text["engram_pad_token_id"],
    }.items():
        add_kv(key, K.UINT32, value)
    for key, value in {
        "expert_weights_scale": text["routed_scaling_factor"], "attention.layer_norm_rms_epsilon": text["rms_norm_eps"], "rope.freq_base": text["rope_theta"],
        "rope.scaling.factor": rs["factor"], "rope.scaling.yarn_beta_fast": rs["beta_fast"], "rope.scaling.yarn_beta_slow": rs["beta_slow"],
        "attention.compress_rope_freq_base": text["compress_rope_theta"], "hyper_connection.epsilon": text["hc_eps"],
    }.items():
        add_kv(key, K.FLOAT32, float(value))
    add_kv("attention.indexer.candidate_source_layer", K.INT32, text["candidate_source_layer_id"])
    add_kv("expert_weights_norm", K.BOOL, text["norm_topk_prob"])
    add_kv("rope.scaling.type", K.STRING, "yarn")
    add_kv("swiglu_clamp_exp", K.ARRAY, limit, K.FLOAT32)
    add_kv("swiglu_clamp_shexp", K.ARRAY, limit, K.FLOAT32)
    add_kv("attention.compress_ratios", K.ARRAY, text["compress_ratios"], K.UINT32)
    add_kv("attention.kv_source_layers", K.ARRAY, text["kv_source_layer_ids"], K.UINT32)
    add_kv("attention.indexer.source_layers", K.ARRAY, text["index_source_layer_ids"], K.UINT32)
    add_kv("ple.layers", K.ARRAY, text["engram_layer_ids"], K.UINT32)
    add_kv("ple.layer_multipliers", K.ARRAY, hash_multipliers(args.engram_layer_ids, args.engram_max_ngram_size, args.engram_compressed_vocab_size).flatten().tolist(), K.UINT64)
    add_kv("ple.head_vocab_sizes", K.ARRAY, sizes, K.UINT64)
    add_kv("ple.head_offsets", K.ARRAY, np.cumsum([0, *sizes[:-1]]).tolist(), K.UINT64)
    add_kv("engram.token_map", K.ARRAY, list(range(text["vocab_size"])), K.UINT32)

    out: list[tuple[str, tuple[int, ...], int, bytes]] = []

    def add(name: str, t: torch.Tensor) -> None:
        kind = Q.BF16 if t.dtype is torch.bfloat16 else Q.F32
        raw = t.view(torch.uint16) if t.dtype is torch.bfloat16 else t.float()
        out.append((name, tuple(t.shape), int(kind), raw.numpy().tobytes()))

    def weight_name(gguf_name: str) -> str:
        return gguf_name if gguf_name.endswith(".bias") else f"{gguf_name}.weight"

    add("token_embd.weight", tensors["embed.weight"])
    add("output_norm.weight", tensors["norm.weight"])
    add("output.weight", tensors["head.weight"])
    table: dict[int, torch.Tensor] = {}
    for L in range(text["num_hidden_layers"]):
        attn, layer = f"layers.{L}.attn.", f"layers.{L}."
        for prefix, leaves in ((attn, _GGUF_ATTN), (layer, _GGUF_LAYER)):
            for leaf, gguf_name in leaves.items():
                if prefix + leaf in tensors:
                    add(f"blk.{L}.{weight_name(gguf_name)}", tensors[prefix + leaf])
        for proj, gguf_name in (("w1", "ffn_gate_exps"), ("w3", "ffn_up_exps"), ("w2", "ffn_down_exps")):
            add(f"blk.{L}.{gguf_name}.weight", torch.stack([tensors[f"{layer}ffn.experts.{e}.{proj}.weight"] for e in range(text["n_routed_experts"])]))
        if L in text["engram_layer_ids"]:
            add(f"blk.{L}.engram_gate.weight", tensors[f"{layer}engram.q_weight"].float() * tensors[f"{layer}engram.k_weight"].float())
            e = text["engram_layer_ids"].index(L)
            rows = sum(sizes[e * args.engram_hash_cols : (e + 1) * args.engram_hash_cols])
            table[L] = (tensors[f"{layer}engram.embed.weight"][:rows].float() * torch.exp2(tensors[f"{layer}engram.embed.scale"][:rows].view(torch.uint8).float() - 127.0)).to(torch.bfloat16)
    packed = torch.cat(list(table.values()))
    if table_type == "Q8_0":
        data, decoded = _pack_q8_0(packed)
        out.append(("per_layer_token_embd.weight", (packed.shape[0], packed.shape[1]), int(Q.Q8_0), data.numpy().tobytes()))
        table = dict(zip(table, decoded.to(torch.bfloat16).split([t.shape[0] for t in table.values()])))
    else:
        add("per_layer_token_embd.weight", packed)
    _write_gguf(path, kvs, out)
    return table
