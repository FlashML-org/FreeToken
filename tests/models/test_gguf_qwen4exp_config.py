"""Qwen4Exp GGUF config adapter: ``qwen4exp.*`` KVs -> ``ModelConfig``.

Mirrors ``qwen4_exp.config.parse_config`` (the HF path). The metadata below is a released
Qwen3.8-Flash-Next GGUF (IQ4_XS); the PLE arithmetic is cross-checked against
``ple.derive_ngram_hash_constants`` so the GGUF-derived ``ngram_vocab_size_base`` is proven
to reproduce the checkpoint's per-head vocab sizes and offsets.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from freetoken.models.qwen4_exp.gguf import parse_gguf_config

_HEAD_OFFSETS = [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]
_HEAD_VOCAB_SIZES = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

_METADATA = {
    "general.architecture": "qwen4exp",
    "qwen4exp.block_count": 48,
    "qwen4exp.context_length": 262144,
    "qwen4exp.embedding_length": 2560,
    "qwen4exp.attention.head_count": 24,
    "qwen4exp.attention.head_count_kv": 2,
    "qwen4exp.attention.key_length": 256,
    "qwen4exp.attention.value_length": 256,
    "qwen4exp.attention.layer_norm_rms_epsilon": 9.999999974752427e-07,
    "qwen4exp.attention.compress_ratios": [4 if (i + 1) % 4 == 0 else 0 for i in range(48)],
    "qwen4exp.attention.indexer.head_count": 4,
    "qwen4exp.attention.indexer.key_length": 128,
    "qwen4exp.attention.indexer.top_k": 2048,
    "qwen4exp.rope.freq_base": 10000000.0,
    "qwen4exp.rope.dimension_count": 64,
    "qwen4exp.rope.dimension_sections": [11, 11, 10, 0],
    "qwen4exp.full_attention_interval": 4,
    "qwen4exp.expert_count": 512,
    "qwen4exp.expert_used_count": 10,
    "qwen4exp.expert_feed_forward_length": 640,
    "qwen4exp.expert_shared_feed_forward_length": 640,
    "qwen4exp.ssm.conv_kernel": 4,
    "qwen4exp.ssm.group_count": 16,
    "qwen4exp.ssm.inner_size": 6144,
    "qwen4exp.ssm.state_size": 128,
    "qwen4exp.ssm.time_step_rank": 48,
    "qwen4exp.hyper_connection.count": 4,
    "qwen4exp.hyper_connection.low_rank": 320,
    "qwen4exp.ple.layers": [1],
    "qwen4exp.ple.ngram_size": 3,
    "qwen4exp.ple.heads_per_ngram": 8,
    "qwen4exp.ple.conv_kernel": 4,
    "qwen4exp.ple.eos_token_id": 248044,
    "qwen4exp.ple.image_token_id": 248056,
    "qwen4exp.embedding_length_per_layer_input": 160,
    "qwen4exp.ple.head_offsets": list(_HEAD_OFFSETS),
    "qwen4exp.ple.head_vocab_sizes": list(_HEAD_VOCAB_SIZES),
}


def _shim(model_path: str = "x"):
    return SimpleNamespace(
        architectures=["Qwen4ExpGGUFForCausalLM"],
        model_path=model_path,
        model_type="qwen4exp",
        metadata=_METADATA,
        vocab_size=248320,
        tie_word_embeddings=False,
    )


def _parse(monkeypatch):
    from freetoken.models.qwen4_exp import gguf as qgguf

    monkeypatch.setattr(qgguf, "_expert_format", lambda path: "iq4_xs")
    return qgguf.parse_gguf_config(_shim())


def test_base_config(monkeypatch) -> None:
    cfg = _parse(monkeypatch)
    assert cfg.num_layers == 48
    assert cfg.hidden_size == 2560
    assert cfg.num_qo_heads == 24
    assert cfg.num_kv_heads == 2
    assert cfg.head_dim == 256
    assert cfg.vocab_size == 248320
    assert cfg.num_experts == 512 and cfg.num_experts_per_tok == 10
    assert cfg.moe_intermediate_size == 640 == cfg.shared_expert_intermediate_size
    assert cfg.moe_enabled and cfg.use_qk_norm and not cfg.tie_word_embeddings
    assert cfg.rotary_config.rotary_dim == 64
    assert cfg.rotary_config.max_position == 262144
    assert cfg.expert_quant == "iq4_xs" == cfg.moe_weight_format
    assert cfg.architectures == ["Qwen4ExpGGUFForCausalLM"]
    # model_path "x" is not a file -> no tensor table -> no op-swap plan (metadata-only)
    assert cfg.gguf_quant_types is None


def test_attention_groups(monkeypatch) -> None:
    cfg = _parse(monkeypatch)
    full = next(g for g in cfg.attention_groups if g.kind == "full")
    linear = next(g for g in cfg.attention_groups if g.kind == "linear_gated_delta")
    assert full.layer_ids == tuple(range(3, 48, 4))  # every 4th layer (1-indexed) is QSA
    assert linear.layer_ids == tuple(i for i in range(48) if i % 4 != 3)
    assert full.num_kv_heads == 2 and full.head_dim == 256
    assert full.index_head_dim == 128 and full.num_index_layers == 12 and full.index_ratio == 4
    assert linear.num_key_heads == 16 and linear.num_value_heads == 48
    assert linear.key_head_dim == 128 and linear.value_head_dim == 128
    assert linear.conv_kernel_dim == 4
    # The GDN output gate is the architecture's sigmoid, not hidden_act (silu, which is only
    # the MoE activation); the GGUF carries no KV for it.
    assert linear.output_gate == "sigmoid"


def test_qwen4_args(monkeypatch) -> None:
    cfg = _parse(monkeypatch)
    args = cfg.qwen4_args
    assert args.hidden_size == 2560
    assert args.hc_count == 4 and args.hc_lowrank == 320
    assert args.ple_layer_ids == (1,)
    # per-head width is the n-gram table row (embedding_length_per_layer_input=160); the
    # concatenated embedding key_proj consumes is num_ngram_heads * that = n_embd = 2560.
    assert args.ple_embed_dim == 2560 == cfg.hidden_size
    assert args.num_ngram_heads == 16 and args.ngram_head_dim == 160
    assert args.ngram_head_dim == _METADATA["qwen4exp.embedding_length_per_layer_input"]
    assert args.index_n_heads == 4 and args.index_kv_heads == 1
    assert args.index_head_dim == 128 and args.index_budget == 2048 and args.index_ratio == 4
    assert args.ngram_boundary_token_id == 248044
    assert args.image_token_id == 248056
    assert len(cfg.slot_states) == 2


def test_ngram_base_reproduces_checkpoint_sizes() -> None:
    """The GGUF-derived base must yield the checkpoint's own per-head vocab sizes/offsets."""
    from freetoken.models.qwen4_exp.ple import derive_ngram_hash_constants

    _, sizes, offsets = derive_ngram_hash_constants(
        vocab_size=248320,
        ngram_size=3,
        num_ngram_heads=16,
        ngram_vocab_size_base=_HEAD_VOCAB_SIZES[0],
        ple_layer_index=0,
    )
    assert sizes == _HEAD_VOCAB_SIZES
    assert offsets == _HEAD_OFFSETS


_REAL_GGUF = os.environ.get("FREETOKEN_QWEN4EXP_GGUF", "")


@pytest.mark.needs_weights
@pytest.mark.skipif(
    not os.path.isfile(_REAL_GGUF),
    reason="FREETOKEN_QWEN4EXP_GGUF not set to a local qwen4exp GGUF",
)
def test_parse_gguf_config_on_a_real_checkpoint() -> None:
    from freetoken.models.gguf.config import GgufConfigShim
    from freetoken.models.gguf.reader import gguf_tensor_names, iter_gguf_tensors, load_gguf_metadata

    metadata = load_gguf_metadata(_REAL_GGUF)
    names = gguf_tensor_names(_REAL_GGUF)
    vocab = next(t.shape[-1] for t in iter_gguf_tensors(_REAL_GGUF) if t.name == "token_embd.weight")
    shim = GgufConfigShim(
        architectures=["Qwen4ExpGGUFForCausalLM"],
        model_path=_REAL_GGUF,
        model_type="qwen4exp",
        metadata=metadata,
        vocab_size=int(vocab),
        tie_word_embeddings="output.weight" not in names,
    )
    from freetoken.models.qwen4_exp.gguf import is_gguf_model

    cfg = parse_gguf_config(shim)
    assert cfg.num_layers == 48 and cfg.hidden_size == 2560 and cfg.num_experts == 512
    assert cfg.qwen4_args.ngram_head_dim == 160
    assert is_gguf_model(cfg)
    plan = cfg.gguf_quant_types
    assert plan["embed"] == 23 and plan["lm_head"] == 14
    assert plan["L0.inproj_qkvz"] == 23 and "L3.qkv" not in plan  # mixed q=Q6_K k/v=Q8_0


@pytest.mark.needs_weights
@pytest.mark.skipif(
    not os.path.isfile(_REAL_GGUF),
    reason="FREETOKEN_QWEN4EXP_GGUF not set to a local qwen4exp GGUF",
)
def test_iter_gguf_weights_covers_every_tensor() -> None:
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    from freetoken.models.qwen4_exp.gguf import iter_gguf_weights

    names = [
        n for n, _ in iter_gguf_weights(
            _REAL_GGUF, "cpu", include_moe_experts=False, include_non_moe=True
        )
    ]
    # Every non-expert tensor maps (an unmapped one raises), exactly once.
    assert len(names) == len(set(names))
    assert "model.embed_tokens.qweight" in names
    assert "lm_head.qweight" in names
    assert "model.layers.0.linear_attn.in_proj_qkvz.qweight" in names
    assert "model.layers.0.linear_attn.in_proj_ba.weight" in names
    assert "model.layers.1.ple.key_proj.qweight" in names
    # Mixed QSA q/k/v (Q6_K + Q8_0) cannot share a packed buffer -> dense bf16.
    assert "model.layers.3.self_attn.qkv_proj.weight" in names


def test_resolve_ple_source(tmp_path) -> None:
    from freetoken.models.qwen4_exp.gguf import resolve_ple_source

    gguf_file = tmp_path / "m.gguf"
    gguf_file.write_bytes(b"")  # is_gguf_path only checks isfile + .gguf suffix
    with pytest.raises(ValueError, match="ple-source"):
        resolve_ple_source(SimpleNamespace(model_path=str(gguf_file), ple_source=None))
    assert resolve_ple_source(SimpleNamespace(model_path=str(gguf_file), ple_source="/orig")) == "/orig"
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    assert resolve_ple_source(SimpleNamespace(model_path=str(hf_dir), ple_source=None)) == str(hf_dir)


def test_resolve_ple_source_uses_ftw_side_files(tmp_path) -> None:
    from freetoken.checkpoint.ftw import INDEX_NAME
    from freetoken.models.qwen4_exp.gguf import resolve_ple_source

    ftw = tmp_path / "ftw"
    ftw.mkdir()
    (ftw / INDEX_NAME).write_text("{}")
    (ftw / "ple-table-0.safetensors").write_text("")  # side file written by ftw_side_files
    assert resolve_ple_source(SimpleNamespace(model_path=str(ftw), ple_source=None)) == str(ftw)

    empty = tmp_path / "ftw-empty"
    empty.mkdir()
    (empty / INDEX_NAME).write_text("{}")
    with pytest.raises(ValueError, match="ple-source"):
        resolve_ple_source(SimpleNamespace(model_path=str(empty), ple_source=None))


def test_compress_ratios_absent_or_zero_defaults_to_four(monkeypatch) -> None:
    """A converter that drops/zeroes compress_ratios must not flip the full layers to BSA."""
    from freetoken.models.qwen4_exp import gguf as qgguf

    monkeypatch.setattr(qgguf, "_expert_format", lambda path: "iq4_xs")
    for value in ([0] * 48, None):
        metadata = dict(_METADATA)
        if value is None:
            metadata.pop("qwen4exp.attention.compress_ratios")
        else:
            metadata["qwen4exp.attention.compress_ratios"] = value
        shim = _shim()
        shim.metadata = metadata
        assert qgguf.parse_gguf_config(shim).qwen4_args.index_ratio == 4
