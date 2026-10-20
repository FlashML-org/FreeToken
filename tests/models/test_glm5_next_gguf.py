"""glm5-next GGUF: config parse and weight-reader mapping against a synthetic checkpoint.

Builds a tiny two-layer (KDA + DSA) glm5-next GGUF in tmp -- geometry scaled down but
every tensor, KV key and converter quirk of the real writer present (A_log stored as
``-exp(A_log)``, the per-head ``attn_k_b``/``attn_v_b`` split with k transposed, the
per-block ``head_count_kv`` list, the trailing MTP block) -- then checks the config the
shim derives and the state dict the reader yields: names complete against the model's,
kv_b fused to the module's dense layout, A_log inverted, conv1d concatenation.
"""

from __future__ import annotations

import struct

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info

# GGUF v3 value types (the vendored gguf-py GGUFValueType numbering)
_T_UINT32, _T_FLOAT32, _T_BOOL, _T_STRING, _T_ARRAY, _T_UINT64 = 4, 6, 7, 8, 9, 10


def _tp1():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _pack_val(v):
    if isinstance(v, str):
        b = v.encode()
        return struct.pack("<Q", len(b)) + b
    if isinstance(v, bool):
        return struct.pack("<B", int(v))
    if isinstance(v, int):
        return struct.pack("<I", v)
    if isinstance(v, float):
        return struct.pack("<f", v)
    if isinstance(v, (list, tuple)):
        assert v and all(isinstance(x, type(v[0])) for x in v), "homogeneous arrays only"
        elem_type = {int: _T_UINT32, float: _T_FLOAT32, bool: _T_BOOL, str: _T_STRING}[type(v[0])]
        body = b"".join(_pack_val(x) for x in v)
        return struct.pack("<I", elem_type) + struct.pack("<Q", len(v)) + body
    raise TypeError(v)


def _gguf_writer(path, kvs, tensors):
    """A minimal GGUF v3 writer: header + KV section + tensor infos + data, little-endian."""
    kv_blob = b"".join(
        struct.pack("<Q", len(k.encode())) + k.encode() + struct.pack("<I", t) + _pack_val(v)
        for k, (t, v) in kvs.items()
    )

    infos, data, offset = b"", b"", 0
    ALIGN = 32
    for name, (ggml_type, ne, payload) in tensors:
        ne = tuple(ne)
        infos += struct.pack("<Q", len(name.encode())) + name.encode()
        infos += struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", d) for d in ne)
        infos += struct.pack("<I", ggml_type)
        pad = (-offset) % ALIGN
        infos += struct.pack("<Q", offset + pad)
        offset += pad + len(payload)
        data += b"\x00" * pad + payload

    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3))
        f.write(struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kvs)))
        f.write(kv_blob)
        f.write(infos)
        f.write(b"\x00" * ((-f.tell()) % ALIGN))
        f.write(data)


# ggml type ids
_F32, _Q8_0, _IQ3_S, _IQ4_XS, _BF16 = 0, 8, 21, 23, 30


def _make_gguf(path, *, n_layers=2, n_mtp=1, arch="glm5-next", head_kv_list=None, kv_b_q8_0=False,
                 head_kv_scalar=False, conv_unit_dim=False):
    """A miniature glm5-next layout: layer 0 KDA, layer 1 DSA, one trailing MTP block."""
    import numpy as np

    # hidden/q_lora keep Q8_0 rows a multiple of 32, expert rows a multiple of 256 (IQ3_S/IQ4_XS
    # blocks); the kv_b halves block on nope/v head rows, so D doubles when they store Q8_0
    H, D, KDA_H, KDA_D = 8, 64 if kv_b_q8_0 else 32, 4, 8
    QKV = KDA_H * KDA_D  # 32
    I, E, V = 512, 4, 64

    rng = np.random.default_rng(0)
    f32 = lambda *shape: rng.standard_normal(shape).astype(np.float32)
    bf16 = lambda n: (rng.standard_normal(n).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()
    q8 = lambda k, rows: np.zeros(rows * (k // 32) * 34, dtype=np.uint8).tobytes()  # placeholder packed rows
    iq3 = lambda k, rows, e: np.zeros(rows * e * (k // 256) * 110, dtype=np.uint8).tobytes()
    iq4 = lambda k, rows, e: np.zeros(rows * e * (k // 256) * 144, dtype=np.uint8).tobytes()

    kv_heads = head_kv_list if head_kv_list is not None else ([0, 1] + [1] * n_mtp)
    hidden = H * D  # 128
    q_lora = 32
    kvs = {
        "general.architecture": (_T_STRING, arch),
        "general.name": (_T_STRING, "glm5-next-test"),
        f"{arch}.block_count": (_T_UINT32, n_layers + n_mtp),
        f"{arch}.context_length": (_T_UINT32, 512),
        f"{arch}.embedding_length": (_T_UINT32, hidden),
        f"{arch}.feed_forward_length": (_T_UINT32, I),
        f"{arch}.attention.head_count": (_T_UINT32, H),
        # the generic GGUF spelling is a scalar (all blocks); the new writer's list carries the layout
        f"{arch}.attention.head_count_kv": (_T_UINT32, 1) if head_kv_scalar else (_T_ARRAY, kv_heads),
        f"{arch}.attention.layer_norm_rms_epsilon": (_T_FLOAT32, 1e-6),
        f"{arch}.attention.q_lora_rank": (_T_UINT32, q_lora),
        f"{arch}.attention.kv_lora_rank": (_T_UINT32, D),
        f"{arch}.attention.key_length_mla": (_T_UINT32, D // 2),
        f"{arch}.attention.value_length_mla": (_T_UINT32, D // 2),
        f"{arch}.attention.indexer.head_count": (_T_UINT32, 2),
        f"{arch}.attention.indexer.key_length": (_T_UINT32, 8),
        f"{arch}.attention.indexer.top_k": (_T_UINT32, 64),
        f"{arch}.attention.indexer.kpool": (_T_UINT32, 4),
        f"{arch}.attention.indexer.types": (_T_ARRAY, [True] * (n_layers + n_mtp)),
        f"{arch}.ssm.conv_kernel": (_T_UINT32, 4),
        f"{arch}.kda.head_dim": (_T_UINT32, KDA_D),
        f"{arch}.kda.gate_lower_bound": (_T_FLOAT32, -5.0),
        f"{arch}.leading_dense_block_count": (_T_UINT32, 0),
        f"{arch}.expert_count": (_T_UINT32, E),
        f"{arch}.expert_used_count": (_T_UINT32, 2),
        f"{arch}.expert_shared_count": (_T_UINT32, 1),
        f"{arch}.expert_feed_forward_length": (_T_UINT32, I // 2),
        f"{arch}.expert_weights_norm": (_T_BOOL, True),
        f"{arch}.expert_weights_scale": (_T_FLOAT32, 2.5),
        f"{arch}.hyper_connection.count": (_T_UINT32, 2),
        f"{arch}.hyper_connection.sinkhorn_iterations": (_T_UINT32, 20),
        f"{arch}.hyper_connection.epsilon": (_T_FLOAT32, 1e-6),
        f"{arch}.swiglu_clamp_exp": (_T_ARRAY, [10.0] * (n_layers + n_mtp)),
        f"{arch}.nextn_predict_layers": (_T_UINT32, n_mtp),
        "tokenizer.ggml.tokens": (_T_ARRAY, ["<unk>"] * V),
        "tokenizer.ggml.model": (_T_STRING, "gpt2"),
    }

    tensors = [
        ("token_embd.weight", (_Q8_0, (hidden, V), q8(hidden, V))),
        ("output.weight", (_Q8_0, (hidden, V), q8(hidden, V))),
        ("output_norm.weight", (_F32, (hidden,), f32(hidden).tobytes())),
    ]
    for blk in range(n_layers + n_mtp):
        is_dsa = kv_heads[blk] == 1
        tensors.append((f"blk.{blk}.attn_norm.weight", (_F32, (hidden,), f32(hidden).tobytes())))
        tensors.append((f"blk.{blk}.ffn_norm.weight", (_F32, (hidden,), f32(hidden).tobytes())))
        for sub in ("attn", "ffn"):
            tensors.append((f"blk.{blk}.hc_{sub}_fn.weight", (_F32, (2 * hidden, 8), f32(8, 2 * hidden).tobytes())))
            tensors.append((f"blk.{blk}.hc_{sub}_base.weight", (_F32, (8,), f32(8).tobytes())))
            tensors.append((f"blk.{blk}.hc_{sub}_scale.weight", (_F32, (3,), f32(3).tobytes())))
        if is_dsa:
            tensors.append((f"blk.{blk}.attn_q_a.weight", (_Q8_0, (hidden, q_lora), q8(hidden, q_lora))))
            tensors.append((f"blk.{blk}.attn_q_b.weight", (_Q8_0, (q_lora, H * (D // 2)), q8(q_lora, H * (D // 2)))))
            tensors.append((f"blk.{blk}.attn_kv_a_mqa.weight", (_BF16, (hidden, D), bf16(hidden * D))))
            if kv_b_q8_0:
                tensors.append((f"blk.{blk}.attn_k_b.weight", (_Q8_0, (D // 2, D, H), q8(D // 2, D * H))))
                tensors.append((f"blk.{blk}.attn_v_b.weight", (_Q8_0, (D, D // 2, H), q8(D, (D // 2) * H))))
            else:
                tensors.append((f"blk.{blk}.attn_k_b.weight", (_BF16, (D // 2, D, H), bf16(D * (D // 2) * H))))
                tensors.append((f"blk.{blk}.attn_v_b.weight", (_BF16, (D, D // 2, H), bf16(D * (D // 2) * H))))
            tensors.append((f"blk.{blk}.attn_output.weight", (_Q8_0, (H * (D // 2), hidden), q8(H * (D // 2), hidden))))
            tensors.append((f"blk.{blk}.attn_q_a_norm.weight", (_F32, (q_lora,), f32(q_lora).tobytes())))
            tensors.append((f"blk.{blk}.attn_kv_a_norm.weight", (_F32, (D,), f32(D).tobytes())))
            tensors.append((f"blk.{blk}.indexer.attn_q_b.weight", (_BF16, (q_lora, 16), bf16(q_lora * 16))))
            tensors.append((f"blk.{blk}.indexer.attn_k.weight", (_BF16, (hidden, 8), bf16(hidden * 8))))
            tensors.append((f"blk.{blk}.indexer.k_norm.weight", (_F32, (8,), f32(8).tobytes())))
            tensors.append((f"blk.{blk}.indexer.k_norm.bias", (_F32, (8,), f32(8).tobytes())))
            tensors.append((f"blk.{blk}.indexer.proj.weight", (_F32, (hidden, 2), f32(hidden, 2).tobytes())))
            tensors.append((f"blk.{blk}.indexer_compressor_gate.weight", (_F32, (hidden, 8), f32(hidden, 8).tobytes())))
            tensors.append((f"blk.{blk}.indexer_compressor_ape.weight", (_F32, (8, 4), f32(8, 4).tobytes())))
        else:
            for name in ("attn_q", "attn_k", "attn_v", "ssm_beta", "ssm_f_a", "ssm_g_a"):
                out_rows = QKV if name in ("attn_q", "attn_k", "attn_v") else (KDA_H if name == "ssm_beta" else KDA_D)
                gt = _Q8_0 if name in ("attn_q", "attn_k", "attn_v") else _BF16
                payload = q8(hidden, out_rows) if gt == _Q8_0 else bf16(out_rows * hidden)
                tensors.append((f"blk.{blk}.{name}.weight", (gt, (hidden, out_rows), payload)))
            tensors.append((f"blk.{blk}.ssm_f_b.weight", (_BF16, (KDA_D, QKV), bf16(KDA_D * QKV))))
            tensors.append((f"blk.{blk}.ssm_g_b.weight", (_BF16, (KDA_D, QKV), bf16(KDA_D * QKV))))
            tensors.append((f"blk.{blk}.attn_output.weight", (_Q8_0, (QKV, hidden), q8(QKV, hidden))))
            for q in ("q", "k", "v"):
                conv_ne = (4, 1, QKV, 1) if conv_unit_dim else (4, 1, QKV)
                tensors.append((f"blk.{blk}.ssm_conv1d_{q}.weight", (_F32, conv_ne, f32(4, 1, QKV).tobytes())))
            tensors.append((f"blk.{blk}.ssm_a", (_F32, (KDA_H,), (-np.abs(f32(KDA_H)) - 0.1).tobytes())))
            tensors.append((f"blk.{blk}.ssm_dt.bias", (_F32, (QKV,), f32(QKV).tobytes())))
            tensors.append((f"blk.{blk}.ssm_norm.weight", (_F32, (KDA_D,), f32(KDA_D).tobytes())))
        # every decoder block here is MoE (leading_dense_block_count 0)
        tensors.append((f"blk.{blk}.ffn_gate_inp.weight", (_F32, (hidden, E), f32(hidden, E).tobytes())))
        tensors.append((f"blk.{blk}.exp_probs_b.bias", (_F32, (E,), f32(E).tobytes())))
        tensors.append((f"blk.{blk}.ffn_gate_exps.weight", (_IQ3_S, (hidden, I // 2, E), iq3(hidden, I // 2, E))))
        tensors.append((f"blk.{blk}.ffn_up_exps.weight", (_IQ3_S, (hidden, I // 2, E), iq3(hidden, I // 2, E))))
        tensors.append((f"blk.{blk}.ffn_down_exps.weight", (_IQ4_XS, (I // 2, hidden, E), iq4(I // 2, hidden, E))))
        for p in ("gate", "up"):
            tensors.append((f"blk.{blk}.ffn_{p}_shexp.weight", (_Q8_0, (hidden, I // 2), q8(hidden, I // 2))))
        tensors.append((f"blk.{blk}.ffn_down_shexp.weight", (_Q8_0, (I // 2, hidden), q8(I // 2, hidden))))
    _gguf_writer(path, kvs, tensors)


@pytest.fixture(scope="module")
def glm_gguf(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm5gguf") / "glm5-test.gguf"
    _make_gguf(str(path))
    return str(path)


def _shim(path):
    from freetoken.models.gguf.config import build_gguf_shim

    return build_gguf_shim(path)


def test_shim_routes_the_architecture(glm_gguf):
    shim = _shim(glm_gguf)
    assert shim.architectures == ["Glm5NextGGUFForCausalLM"]
    assert shim.vocab_size == 64
    assert shim.tie_word_embeddings is False


def test_parse_gguf_config_matches_layer_census(glm_gguf):
    from freetoken.models.glm5_next.args import DSA_LAYER, KDA_LAYER
    from freetoken.models.glm5_next.gguf import parse_gguf_config

    config = parse_gguf_config(_shim(glm_gguf))
    args = config.glm5_args
    # the trailing MTP block is dropped, and head_count_kv lays out KDA|DSA
    assert config.num_layers == 2
    assert args.layer_types == (KDA_LAYER, DSA_LAYER)
    assert config.first_k_dense_replace == 0
    assert config.num_experts == 4 and config.num_experts_per_tok == 2
    assert args.qk_nope_head_dim == 16 and args.v_head_dim == 16 and args.qk_rope_head_dim == 0
    assert args.index_kpool == 4 and args.index_kpool_compress and args.index_kpool_always_select_tail
    assert args.indexer_rope_interleave  # the served fact the GGUF KV predates
    assert config.expert_quant == "gguf"
    assert config.swiglu_limit == 10.0


def test_legacy_arch_spelling_routes_identically(tmp_path):
    path = tmp_path / "glm5-legacy.gguf"
    _make_gguf(str(path), arch="glm5next")
    assert _shim(str(path)).architectures == ["Glm5NextGGUFForCausalLM"]


def test_head_count_kv_fallback_uses_tensor_census(tmp_path):
    """The pre-list writer (unsloth's first release) is laid out from the tensor names."""
    from freetoken.models.glm5_next.args import DSA_LAYER, KDA_LAYER
    from freetoken.models.glm5_next.gguf import _layer_types

    path = tmp_path / "glm5-nolist.gguf"
    _make_gguf(str(path))
    shim = _shim(str(path))
    stripped = type(shim)(
        architectures=shim.architectures,
        model_path=shim.model_path,
        model_type=shim.model_type,
        metadata={k: v for k, v in shim.metadata.items() if not k.endswith("attention.head_count_kv")},
        vocab_size=shim.vocab_size,
        tie_word_embeddings=shim.tie_word_embeddings,
    )
    assert _layer_types(stripped, 2) == (KDA_LAYER, DSA_LAYER)


def test_reader_state_dict_matches_model(glm_gguf):
    _tp1()
    from dataclasses import replace

    from freetoken.models.glm5_next import Glm5NextForCausalLM
    from freetoken.models.glm5_next.gguf import iter_gguf_weights, parse_gguf_config
    from freetoken.models.register import checkpoint_quant_config, get_model_spec

    shim = _shim(glm_gguf)
    config = parse_gguf_config(shim)
    spec = get_model_spec(shim.architectures[0])
    config = replace(config, quant=checkpoint_quant_config(glm_gguf, shim, spec))
    model = Glm5NextForCausalLM(config)
    want = model.state_dict()
    got = dict(iter_gguf_weights(glm_gguf, torch.device("cpu"), include_moe_experts=False, include_non_moe=True))
    assert set(want) == set(got), f"missing={sorted(set(want) - set(got))[:6]} extra={sorted(set(got) - set(want))[:6]}"
    for k in sorted(want):
        assert tuple(want[k].shape) == tuple(got[k].shape), k


def test_quantized_kv_b_halves_serve_dense(tmp_path):
    """Dynamic quants (unsloth UD) store the per-head kv_b halves Q8_0, not bf16: the
    module stays dense -- the reader dequantizes and fuses -- instead of refusing it."""
    _tp1()
    from dataclasses import replace

    from freetoken.models.glm5_next import Glm5NextForCausalLM
    from freetoken.models.glm5_next.gguf import iter_gguf_weights, parse_gguf_config
    from freetoken.models.register import checkpoint_quant_config, get_model_spec

    path = str(tmp_path / "glm5-kvq8.gguf")
    _make_gguf(path, kv_b_q8_0=True)
    shim = _shim(path)
    spec = get_model_spec(shim.architectures[0])
    config = replace(parse_gguf_config(shim), quant=checkpoint_quant_config(path, shim, spec))
    model = Glm5NextForCausalLM(config)
    assert config.quant.scheme_for("model.layers.1.self_attn.kv_b_proj") is None
    got = dict(iter_gguf_weights(path, torch.device("cpu"), include_moe_experts=False, include_non_moe=True))
    want = model.state_dict()
    assert set(want) == set(got), f"missing={sorted(set(want) - set(got))[:6]} extra={sorted(set(got) - set(want))[:6]}"
    for k in sorted(want):
        assert tuple(want[k].shape) == tuple(got[k].shape), k


def test_converter_transforms_are_undone(glm_gguf):
    from freetoken.models.gguf.reader import gguf_tensor_index
    from freetoken.models.gguf.tensors import float_tensor
    from freetoken.models.glm5_next.gguf import iter_gguf_weights

    got = dict(iter_gguf_weights(glm_gguf, torch.device("cpu"), include_moe_experts=False, include_non_moe=True))
    index = gguf_tensor_index(glm_gguf)
    # A_log: the converter stores -exp(A_log); the reader inverts to the raw log
    stored = float_tensor(index["blk.0.ssm_a"])
    assert torch.allclose(got["model.layers.0.self_attn.A_log"], torch.log(-stored))
    # kv_b: per-head k [lora, nope] (transposed) + v [v, lora] fuse to [heads*(nope+v), lora]
    k = float_tensor(index["blk.1.attn_k_b.weight"]).transpose(1, 2)
    v = float_tensor(index["blk.1.attn_v_b.weight"])
    expect = torch.cat([k, v], dim=1).reshape(-1, 32).to(torch.bfloat16)
    assert torch.equal(got["model.layers.1.self_attn.kv_b_proj.weight"], expect)
    # conv1d: q|k|v conv kernels concatenate on the channel axis
    conv = torch.cat([float_tensor(index[f"blk.0.ssm_conv1d_{q}.weight"]) for q in ("q", "k", "v")], dim=0)
    assert torch.equal(got["model.layers.0.self_attn.conv1d.weight"], conv.to(torch.bfloat16))


def test_to_gguf_maps_the_fused_and_packed_modules(glm_gguf):
    from freetoken.models.glm5_next.gguf import to_gguf

    assert to_gguf("model.language_model.embed_tokens", {}) == ("token_embd",)
    assert to_gguf("lm_head", {}) == ("output",)
    for leaf, gguf_name in (
        ("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"),
        ("b_proj", "ssm_beta"), ("f_a_proj", "ssm_f_a"), ("g_a_proj", "ssm_g_a"),
    ):
        assert to_gguf(f"model.language_model.layers.0.self_attn.{leaf}", {}) == (f"blk.0.{gguf_name}",)
    assert to_gguf("model.language_model.layers.3.mlp.experts.0.gate_proj", {}) == ("blk.3.ffn_gate_exps",)
    assert to_gguf("model.language_model.layers.3.mlp.gate", {}) == ()


def test_fused_covers_the_mmq_tail():
    """A packed segment must cover the bytes MMQ reads past the last row: the KDA f_b / g_b
    (K = linear head dim 128) are the family's first K below half a padding unit."""
    from types import SimpleNamespace

    from freetoken.layers.quantization.ggml_types import mmq_tail_bytes, row_bytes
    from freetoken.models.gguf.tensors import fused

    k = 128
    part = torch.zeros(96, row_bytes(8, k), dtype=torch.uint8)  # Q8_0
    out = dict(fused("l0.self_attn.f_b_proj", [SimpleNamespace(ggml_type=8)], [part], k))
    assert out["l0.self_attn.f_b_proj.weight"].numel() >= 96 * row_bytes(8, k) + mmq_tail_bytes(8, k)


def test_scalar_head_count_kv_falls_back_to_census(tmp_path):
    """The generic GGUF spelling of head_count_kv is a scalar (all blocks), which is wrong for the
    hybrid -- llama.cpp broadcasts it and then fails tensor creation; the census recovers the layout."""
    from freetoken.models.glm5_next.args import DSA_LAYER, KDA_LAYER
    from freetoken.models.glm5_next.gguf import _layer_types, parse_gguf_config

    path = str(tmp_path / "glm5-scalar.gguf")
    _make_gguf(path, head_kv_scalar=True)
    shim = _shim(path)
    assert _layer_types(shim, 2) == (KDA_LAYER, DSA_LAYER)
    assert parse_gguf_config(shim).num_layers == 2


def test_conv_kernels_with_a_trailing_unit_dim(tmp_path):
    """Some writers ship the conv kernels 4-D; llama.cpp's glm5-next loader tries that shape first."""
    _tp1()
    from dataclasses import replace

    from freetoken.models.glm5_next import Glm5NextForCausalLM
    from freetoken.models.glm5_next.gguf import iter_gguf_weights, parse_gguf_config
    from freetoken.models.register import checkpoint_quant_config, get_model_spec

    path = str(tmp_path / "glm5-conv4d.gguf")
    _make_gguf(path, conv_unit_dim=True)
    shim = _shim(path)
    spec = get_model_spec(shim.architectures[0])
    config = replace(parse_gguf_config(shim), quant=checkpoint_quant_config(path, shim, spec))
    model = Glm5NextForCausalLM(config)
    got = dict(iter_gguf_weights(path, torch.device("cpu"), include_moe_experts=False, include_non_moe=True))
    want = model.state_dict()
    assert set(want) == set(got)
    assert tuple(want["model.layers.0.self_attn.conv1d.weight"].shape) == tuple(got["model.layers.0.self_attn.conv1d.weight"].shape)
