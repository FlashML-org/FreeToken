"""LFM2-MoE GGUF adapter: llama.cpp's ``lfm2moe`` metadata -> ModelConfig, the packed weight
mapping, and the native quantized expert banks.

The fixture is a real (tiny) GGUF written with gguf-py, so the config, the reader and the bank
loader run over genuine gguf block bytes. ``iter_gguf_weights`` must land exactly on a model
built from that config, every expert bank row must equal the file's row (with the alignment
padding provably zero), and the bank GEMM must reproduce the file's own dequantized experts.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from freetoken.models.gguf.config import GgufConfigShim, build_gguf_shim
from freetoken.models.gguf.dequant import GGML_Q4_0, GGML_Q8_0, padded_row_bytes, row_bytes
from freetoken.models.gguf.reader import (
    GGUF_TENSOR_TYPES_KV,
    load_gguf_metadata,
    write_metadata_gguf,
)
from freetoken.models.weight import load_weight

_TYPE_IDS = {"Q4_0": GGML_Q4_0, "Q8_0": GGML_Q8_0}

HIDDEN, DENSE_INTER, MOE_INTER, VOCAB = 256, 256, 128, 320
LAYERS = 3  # two short-conv layers, one attention layer
EXPERTS, TOP_K = 4, 2
HEADS, KV_HEADS, HEAD_DIM = 2, 1, 128
CONV_K = 3
KV_LAYER = 2

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _quant(values: np.ndarray, name: str) -> np.ndarray:
    """Real gguf block bytes for a dense float array (ggml order, i.e. fastest dim first)."""
    import gguf

    return gguf.quantize(np.ascontiguousarray(values, dtype=np.float32), getattr(gguf.GGMLQuantizationType, name))


def write_tiny_lfm2_gguf(path, *, drop: tuple[str, ...] = (), extra: dict | None = None) -> None:
    """A minimal llama.cpp-style ``lfm2moe`` file: Q4_0/Q8_0 mix, one MoE layer."""
    import gguf

    rng = np.random.default_rng(0)
    w = gguf.GGUFWriter(path, "lfm2moe")
    w.add_block_count(LAYERS)
    w.add_embedding_length(HIDDEN)
    w.add_feed_forward_length(DENSE_INTER)
    w.add_head_count(HEADS)
    w.add_head_count_kv([0] * (LAYERS - 1) + [KV_HEADS])
    w.add_context_length(512)
    w.add_rope_freq_base(1e7)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_expert_count(EXPERTS)
    w.add_expert_used_count(TOP_K)
    w.add_expert_feed_forward_length(MOE_INTER)
    w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
    w.add_leading_dense_block_count(2)
    w.add_shortconv_l_cache(CONV_K)
    w.add_uint32("lfm2moe.vocab_size", VOCAB)
    w.add_tokenizer_model("gpt2")
    w.add_token_list([f"tok{i}" for i in range(VOCAB)])

    def f32(shape):
        return (rng.standard_normal(shape).astype(np.float32) * 0.05)

    def add(name: str, values, qtype: str | None = None):
        """``values`` is in torch order (the quantized row is the last axis).

        gguf-py takes the *packed byte* shape for a quantized tensor and stores ggml's
        fastest-dim-first dims, so passing ``[rows, row_bytes]`` round-trips to this torch
        shape through ``iter_gguf_tensors``.
        """
        if name in drop:
            return
        if qtype is None:
            w.add_tensor(name, values.astype(np.float32))
            return
        rows = int(np.prod(values.shape[:-1])) if values.ndim > 1 else 1
        row = row_bytes(values.shape[-1], _TYPE_IDS[qtype])
        w.add_tensor(name, _quant(values, qtype), raw_shape=[rows, row],
                     raw_dtype=getattr(gguf.GGMLQuantizationType, qtype))

    add("token_embd.weight", f32((VOCAB, HIDDEN)), "Q4_0")
    add("token_embd_norm.weight", f32(HIDDEN) + 1.0)

    for layer in range(LAYERS):
        b = f"blk.{layer}"
        add(f"{b}.attn_norm.weight", f32(HIDDEN) + 1.0)
        add(f"{b}.ffn_norm.weight", f32(HIDDEN) + 1.0)
        if layer < 2:
            add(f"{b}.shortconv.conv.weight", f32((HIDDEN, CONV_K)))
            add(f"{b}.shortconv.in_proj.weight", f32((3 * HIDDEN, HIDDEN)), "Q4_0")
            add(f"{b}.shortconv.out_proj.weight", f32((HIDDEN, HIDDEN)), "Q4_0")
            add(f"{b}.ffn_gate.weight", f32((DENSE_INTER, HIDDEN)), "Q4_0")
            add(f"{b}.ffn_up.weight", f32((DENSE_INTER, HIDDEN)), "Q4_0")
            add(f"{b}.ffn_down.weight", f32((HIDDEN, DENSE_INTER)), "Q4_0")
            continue
        add(f"{b}.attn_q.weight", f32((HEADS * HEAD_DIM, HIDDEN)), "Q4_0")
        add(f"{b}.attn_k.weight", f32((KV_HEADS * HEAD_DIM, HIDDEN)), "Q4_0")
        # a second quant type on one tensor of the layer (the K-quant files mix per tensor)
        add(f"{b}.attn_v.weight", f32((KV_HEADS * HEAD_DIM, HIDDEN)), "Q8_0")
        add(f"{b}.attn_output.weight", f32((HIDDEN, HEADS * HEAD_DIM)), "Q4_0")
        add(f"{b}.attn_q_norm.weight", f32(HEAD_DIM) + 1.0)
        add(f"{b}.attn_k_norm.weight", f32(HEAD_DIM) + 1.0)
        add(f"{b}.ffn_gate_inp.weight", f32((EXPERTS, HIDDEN)))
        add(f"{b}.exp_probs_b.bias", f32(EXPERTS))
        add(f"{b}.ffn_gate_exps.weight", f32((EXPERTS, MOE_INTER, HIDDEN)), "Q4_0")
        add(f"{b}.ffn_up_exps.weight", f32((EXPERTS, MOE_INTER, HIDDEN)), "Q4_0")
        # Q8_0 rows of 128 elements are 136 bytes: the bank has to pad them to a 16-byte row
        add(f"{b}.ffn_down_exps.weight", f32((EXPERTS, HIDDEN, MOE_INTER)), "Q8_0")
    for name, values in (extra or {}).items():
        add(name, values)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


@pytest.fixture(scope="module")
def gguf_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("lfm2gguf") / "tiny-lfm2moe-q4_0.gguf"
    write_tiny_lfm2_gguf(str(path))
    return str(path)


@pytest.fixture(scope="module")
def config(gguf_path):
    return build_gguf_shim(gguf_path), _parse(build_gguf_shim(gguf_path))


def _parse(shim: GgufConfigShim):
    from freetoken.models.lfm2_moe.gguf import parse_gguf_config

    return parse_gguf_config(shim)


def _model(config):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.lfm2_moe.model import Lfm2MoeForCausalLM

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    return Lfm2MoeForCausalLM(config)


def test_config_from_llama_cpp_metadata(config):
    shim, cfg = config
    assert shim.architectures == ["Lfm2MoeGGUFForCausalLM"]
    assert (cfg.num_layers, cfg.num_moe_layers) == (LAYERS, 1)
    assert cfg.short_conv_group().layer_ids == (0, 1)
    assert cfg.attention_group_for_layer(KV_LAYER).layer_ids == (KV_LAYER,)
    assert cfg.is_short_conv_layer(0) and not cfg.is_short_conv_layer(KV_LAYER)
    assert cfg.has_state_layers and not cfg.has_linear_attention
    assert cfg.state_group() is cfg.short_conv_group()
    assert (cfg.num_qo_heads, cfg.num_kv_heads, cfg.head_dim) == (HEADS, KV_HEADS, HEAD_DIM)
    assert (cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size) == (
        EXPERTS, TOP_K, MOE_INTER)
    assert (cfg.first_k_dense_replace, cfg.tie_word_embeddings) == (2, True)
    assert cfg.rms_norm_eps == pytest.approx(1e-5)
    assert cfg.expert_quant == cfg.moe_weight_format == "gguf_k"
    assert cfg.rotary_config.base == pytest.approx(1e7)
    # the per-tensor types travel in the config (the model classes need them to build the ops)
    types = dict(cfg.gguf_tensor_types)
    assert types[f"blk.{KV_LAYER}.attn_v.weight"] == GGML_Q8_0
    assert types[f"blk.{KV_LAYER}.ffn_gate_exps.weight"] == GGML_Q4_0
    # expert bank bytes are exact, padding included
    gate_up_row = padded_row_bytes(HIDDEN, GGML_Q4_0)[0]
    down_row = padded_row_bytes(MOE_INTER, GGML_Q8_0)[0]
    assert down_row == 272  # 136 -> whole-block padding
    assert cfg.expert_bank_bytes == EXPERTS * (2 * MOE_INTER * gate_up_row + HIDDEN * down_row)


def test_iter_weights_lands_on_the_model_state_dict(gguf_path, config):
    from freetoken.models.lfm2_moe.gguf import iter_gguf_weights

    _shim, cfg = config
    model = _model(cfg)
    state = model.state_dict()
    loaded = dict(iter_gguf_weights(gguf_path, torch.device("cpu"),
                                    include_moe_experts=False, include_non_moe=True))
    assert set(loaded) == set(state), sorted(set(state) ^ set(loaded))
    for name, tensor in state.items():
        assert tuple(loaded[name].shape) == tuple(tensor.shape), name
    # the checkpoint's packed rows are what the GGUF ops read, and the norms stay dense
    assert loaded["model.embed_tokens.qweight"].dtype is torch.uint8
    assert loaded["model.layers.0.operator_norm.weight"].dtype is torch.bfloat16
    assert loaded["model.layers.0.conv.conv.weight"].shape == (HIDDEN, 1, CONV_K)
    assert loaded[f"model.layers.{KV_LAYER}.feed_forward.expert_bias"].dtype is torch.float32
    assert loaded[f"model.layers.{KV_LAYER}.feed_forward.gate.weight"].shape == (EXPERTS, HIDDEN)


def test_expert_banks_are_the_file_rows_with_zero_padding(gguf_path, config):
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.models.lfm2_moe.gguf import load_gguf_k_expert_sources

    _shim, cfg = config
    banks, tags = load_gguf_k_expert_sources(gguf_path, cfg)
    assert tags == [(GGML_Q4_0, GGML_Q8_0)]
    gate_up, down = banks["gate_up"][0], banks["down"][0]
    assert tuple(gate_up.shape) == (EXPERTS, 2 * MOE_INTER, padded_row_bytes(HIDDEN, GGML_Q4_0)[0])
    assert tuple(down.shape) == (EXPERTS, HIDDEN, padded_row_bytes(MOE_INTER, GGML_Q8_0)[0])

    tensors = {t.name: t for t in iter_gguf_tensors(gguf_path)}
    b = f"blk.{KV_LAYER}"
    gate = tensors[f"{b}.ffn_gate_exps.weight"].packed().reshape(EXPERTS, MOE_INTER, -1)
    up = tensors[f"{b}.ffn_up_exps.weight"].packed().reshape(EXPERTS, MOE_INTER, -1)
    dn = tensors[f"{b}.ffn_down_exps.weight"].packed().reshape(EXPERTS, HIDDEN, -1)
    torch.testing.assert_close(gate_up[:, :MOE_INTER, : gate.shape[-1]], gate, rtol=0, atol=0)
    torch.testing.assert_close(gate_up[:, MOE_INTER:, : up.shape[-1]], up, rtol=0, atol=0)
    torch.testing.assert_close(down[:, :, : dn.shape[-1]], dn, rtol=0, atol=0)
    assert down[:, :, dn.shape[-1] :].numel() == EXPERTS * HIDDEN * (272 - dn.shape[-1])
    assert not down[:, :, dn.shape[-1] :].any(), "alignment padding must stay zero"
    assert sum(t.numel() for t in banks["down"]) + sum(t.numel() for t in banks["gate_up"]) == cfg.expert_bank_bytes


@_CUDA
def test_bank_gemm_reproduces_the_file_experts(gguf_path, config):
    from freetoken.models.gguf.dequant import dequantize
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.models.lfm2_moe.gguf import load_gguf_k_expert_sources
    from freetoken.moe.fused_gguf import fused_experts_gguf

    _shim, cfg = config
    banks, tags = load_gguf_k_expert_sources(gguf_path, cfg)
    picked = [0, 2]
    gate_up = torch.stack([banks["gate_up"][0][e] for e in picked]).cuda()
    down = torch.stack([banks["down"][0][e] for e in picked]).cuda()

    tensors = {t.name: t for t in iter_gguf_tensors(gguf_path)}
    b = f"blk.{KV_LAYER}"

    def rows(t, count):
        flat = dequantize(t.packed().reshape(-1), t.ggml_type, torch.float32)
        return flat.reshape(t.rows, -1)[:count]

    gate = rows(tensors[f"{b}.ffn_gate_exps.weight"], len(picked) * MOE_INTER).reshape(
        len(picked), MOE_INTER, HIDDEN).cuda()
    up = rows(tensors[f"{b}.ffn_up_exps.weight"], len(picked) * MOE_INTER).reshape(
        len(picked), MOE_INTER, HIDDEN).cuda()
    dn = rows(tensors[f"{b}.ffn_down_exps.weight"], len(picked) * HIDDEN).reshape(
        len(picked), HIDDEN, MOE_INTER).cuda()

    tokens, top_k = 3, TOP_K
    x = (torch.randn(tokens, HIDDEN).cuda() * 0.1).to(torch.bfloat16)
    # every slot takes a different bank row, so the reference must follow the routing
    ids = (torch.arange(top_k, dtype=torch.int32, device="cuda").repeat(tokens, 1) % len(picked))
    weights = torch.full((tokens, top_k), 1.0 / top_k, device="cuda")
    got = fused_experts_gguf(x, gate_up, down, weights, ids, "silu", *tags[0])
    want = torch.zeros_like(got)
    for i in range(tokens):
        h = x[i].float()
        for k in range(top_k):
            e = int(ids[i, k])
            act = torch.nn.functional.silu(h @ gate[e].T) * (h @ up[e].T)
            want[i] += (act @ dn[e].T) / top_k
    # bf16 activations against an fp32 reference: the tolerance rides the activation scale
    scale = (x.float().abs().max() * gate.abs().max() * MOE_INTER ** 0.5).item()
    assert (got.float() - want).abs().max().item() < 1e-2 * max(scale, 1.0)


def test_mixed_gate_up_types_are_rejected(gguf_path):
    from freetoken.models.lfm2_moe.gguf import expert_layer_types

    # same file, but the up experts re-written as Q8_0: the fused gate_up bank has one row
    # geometry, so the loader must refuse rather than read the wrong rows
    shim = build_gguf_shim(gguf_path)
    types = dict(_parse(shim).gguf_tensor_types)
    types[f"blk.{KV_LAYER}.ffn_up_exps.weight"] = GGML_Q8_0
    cfg = _parse(shim)
    object.__setattr__(cfg, "gguf_tensor_types", tuple(sorted(types.items())))
    with pytest.raises(ValueError, match="one row geometry"):
        expert_layer_types(cfg)


def test_incomplete_layer_and_unmapped_tensor_are_rejected(tmp_path):
    from freetoken.models.lfm2_moe.gguf import iter_gguf_weights

    missing = tmp_path / "missing.gguf"
    write_tiny_lfm2_gguf(str(missing), drop=(f"blk.{KV_LAYER}.attn_k_norm.weight",))
    _shim, cfg = config_for(missing)
    with pytest.raises(ValueError, match="incomplete"):
        list(iter_gguf_weights(str(missing), torch.device("cpu"),
                               include_moe_experts=False, include_non_moe=True))

    unknown = tmp_path / "unknown.gguf"
    write_tiny_lfm2_gguf(str(unknown), extra={"blk.1.mystery.weight": np.zeros(HIDDEN, dtype=np.float32)})
    with pytest.raises(ValueError, match="unmapped"):
        list(iter_gguf_weights(str(unknown), torch.device("cpu"),
                               include_moe_experts=False, include_non_moe=True))


def config_for(path):
    shim = build_gguf_shim(str(path))
    return shim, _parse(shim)


def test_tp_above_one_is_refused(gguf_path, monkeypatch):
    from freetoken.models.lfm2_moe.gguf import iter_gguf_weights

    class _Tp(dict):
        size = 2

    monkeypatch.setattr("freetoken.distributed.get_tp_info", lambda: _Tp())
    with pytest.raises(NotImplementedError, match="TP=1"):
        list(iter_gguf_weights(gguf_path, torch.device("cpu"),
                               include_moe_experts=False, include_non_moe=True))


def test_metadata_only_gguf_keeps_the_type_map(gguf_path, config, tmp_path):
    """A converted FTW dir carries only the KV section: the type map must survive it."""
    _shim, cfg = config
    dest = tmp_path / "source_metadata.gguf"
    write_metadata_gguf(gguf_path, str(dest))
    md = load_gguf_metadata(str(dest))
    assert len(md[GGUF_TENSOR_TYPES_KV]) == len(cfg.gguf_tensor_types)
    shim = build_gguf_shim(str(dest))
    converted = _parse(shim)
    assert converted.gguf_tensor_types == cfg.gguf_tensor_types
    assert converted.expert_bank_bytes == cfg.expert_bank_bytes
    assert converted.attention_groups == cfg.attention_groups


@_CUDA
def test_gguf_converts_to_ftw_keeping_the_bank_tags_and_weights(gguf_path, config, tmp_path):
    """The FTW has one on-disk bank layout per file, so the per-layer types must travel in
    its index: without them a converted directory would read K-quant rows with the wrong
    kernel."""
    from freetoken.checkpoint.convert import convert_checkpoint
    from freetoken.checkpoint.ftw import load_ftw_banks

    _shim, cfg = config
    out = tmp_path / "converted"
    index = convert_checkpoint(gguf_path, str(out), device="cuda:0")
    assert index["quant_format"] == "gguf_k"
    assert index["expert_layer_tags"] == [[GGML_Q4_0, GGML_Q8_0]]

    banks = load_ftw_banks(str(out), num_layers=1)
    assert banks.layer_tags == [(GGML_Q4_0, GGML_Q8_0)]
    assert banks.kind is None and banks.kernel is None  # no QuantKind method serves these

    # the converted directory is a self-contained checkpoint: config, weights and banks
    shim = build_gguf_shim(os.path.join(str(out), "source_metadata.gguf"))
    converted = _parse(shim)
    assert converted.gguf_tensor_types == cfg.gguf_tensor_types
    model = _model(converted)
    state = model.state_dict()
    loaded = dict(load_weight(str(out), torch.device("cpu"), include_moe_experts=False))
    assert set(loaded) == set(state)
    for name, tensor in state.items():
        assert tuple(loaded[name].shape) == tuple(tensor.shape), name
    for layer_id, (gate_up, down) in enumerate(zip(banks.sources["gate_up"], banks.sources["down"])):
        assert tuple(gate_up.shape) == tuple(banks_raw_shape(cfg, layer_id)[0])
        assert tuple(down.shape) == tuple(banks_raw_shape(cfg, layer_id)[1])


def banks_raw_shape(cfg, layer_id):
    from freetoken.models.lfm2_moe.gguf import expert_layer_types

    gate_up_type, down_type = expert_layer_types(cfg)[layer_id]
    return (
        (EXPERTS, 2 * MOE_INTER, padded_row_bytes(HIDDEN, gate_up_type)[0]),
        (EXPERTS, HIDDEN, padded_row_bytes(MOE_INTER, down_type)[0]),
    )


REAL_GLOB = os.environ.get("FREETOKEN_LFM2_GGUF_GLOB", "")


@pytest.mark.skipif(
    not REAL_GLOB, reason="FREETOKEN_LFM2_GGUF_GLOB not set to a local LFM2-MoE .gguf"
)
def test_real_q4_k_m_geometry_and_bank_plan():
    """llama.cpp's Q4_K_M recipe: 30 conv + 10 attention layers, Q4_K gate/up experts with
    Q6_K down experts in half the layers, every bank row a 16-byte multiple after padding."""
    import glob as _glob

    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.models.lfm2_moe.gguf import expert_layer_types

    path = _glob.glob(REAL_GLOB)[0]
    cfg = _parse(build_gguf_shim(path))
    assert (cfg.num_layers, len(cfg.short_conv_group().layer_ids)) == (40, 30)
    assert (cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size) == (64, 4, 1536)
    assert (cfg.hidden_size, cfg.head_dim, cfg.num_kv_heads) == (2048, 64, 8)
    assert (cfg.first_k_dense_replace, cfg.tie_word_embeddings) == (2, True)
    assert cfg.expert_quant == "gguf_k"

    types = dict(cfg.gguf_tensor_types)
    assert types["token_embd.weight"] == 14  # Q6_K embedding (also the tied head)
    v_types = {
        types[f"blk.{i}.attn_v.weight"] for i in cfg.attention_group_for_layer(2).layer_ids
    }
    assert v_types == {12, 14}  # llama.cpp alternates Q6_K v_proj across the attention layers
    assert types["blk.0.shortconv.in_proj.weight"] == 12
    assert types["blk.2.ffn_gate_inp.weight"] == 0  # the router of the first MoE layer is F32

    tags = expert_layer_types(cfg)
    assert len(tags) == cfg.num_moe_layers == 38
    assert {t[0] for t in tags} == {12}  # every gate_up bank is Q4_K
    assert {t[1] for t in tags} == {12, 14}  # down experts mix Q4_K and Q6_K
    assert all(t[1] in (12, 14) for t in tags)

    # the config's bank-byte figure is exactly the padded rows the loader allocates
    planned = cfg.expert_bank_bytes
    row_bytes_sum = 0
    for gate_up_type, down_type in tags:
        gate_up_row = padded_row_bytes(cfg.hidden_size, gate_up_type)[0]
        down_row = padded_row_bytes(cfg.moe_intermediate_size, down_type)[0]
        assert gate_up_row % 16 == 0 and down_row % 16 == 0
        row_bytes_sum += cfg.num_experts * (
            2 * cfg.moe_intermediate_size * gate_up_row + cfg.hidden_size * down_row
        )
    assert planned == row_bytes_sum
    # and it equals the file's expert tensor bytes plus the alignment padding
    file_expert_bytes = sum(
        t.rows * t.row_bytes
        for t in iter_gguf_tensors(path)
        if t.name.endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"))
    )
    assert file_expert_bytes < planned
    assert planned - file_expert_bytes < file_expert_bytes * 0.2  # padding is a small tax


def test_dummy_banks_match_the_loader_geometry(config):
    """``--use-dummy-weight`` fabricates banks the kernel can read: same roles, shapes and
    per-layer tags as the real loader (the types come from the config, not the file)."""
    from freetoken.models.lfm2_moe.gguf import (
        dummy_gguf_k_expert_sources,
        load_gguf_k_expert_sources,
    )

    _shim, cfg = config
    dummy, dummy_tags = dummy_gguf_k_expert_sources(cfg)
    real, real_tags = load_gguf_k_expert_sources(_shim.model_path, cfg)
    assert dummy_tags == real_tags == [(GGML_Q4_0, GGML_Q8_0)]
    assert set(dummy) == set(real) == {"gate_up", "down"}
    for name in ("gate_up", "down"):
        assert len(dummy[name]) == len(real[name]) == cfg.num_moe_layers
        for fake, true in zip(dummy[name], real[name]):
            assert fake.dtype is true.dtype is torch.uint8
            assert tuple(fake.shape) == tuple(true.shape)
