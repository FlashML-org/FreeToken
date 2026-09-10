"""parse_gguf_config must handle dense Gemma-4 GGUFs (issue #357).

Dense checkpoints (gemma-4-12B-it, gemma-4-31B-it) carry no gemma4.expert_* keys
in their GGUF metadata. The old parser unconditionally required expert_count and
hardcoded moe_enabled=True, so dense GGUFs could not load at all.
"""

import struct

import pytest

from freetoken.models.gguf.config import GgufConfigShim
from freetoken.models.gemma4.gguf import parse_gguf_config


def make_shim(metadata_overrides: dict, tmp_path) -> GgufConfigShim:
    """A metadata-only shim over an empty file; _full_rotary_dim falls back to head_dim//4."""
    empty = tmp_path / "meta_only.gguf"
    # Minimal GGUF: magic + version 3 + kv_count=0 + tensor_count=0. GGUFReader opens
    # it fine with no fields/tensors, so _full_rotary_dim takes its metadata-only
    # fallback (head_dim//4) without needing a real checkpoint.
    empty.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 0))
    base = {
        "gemma4.block_count": 4,
        "gemma4.embedding_length": 256,
        "gemma4.attention.head_count": 4,
        "gemma4.attention.head_count_kv": [2, 2, 2, 2],
        # One SWA layer + three full layers.
        "gemma4.attention.sliding_window_pattern": [True, False, False, False],
        "gemma4.attention.key_length_swa": 128,
        "gemma4.attention.key_length": 256,
        "gemma4.attention.sliding_window": 512,
        "gemma4.context_length": 1024,
        "gemma4.rope.freq_base": 1000000.0,
        "gemma4.rope.freq_base_swa": 1000000.0,
        "gemma4.rope.dimension_count_swa": 128,
        "gemma4.feed_forward_length": 512,
        "gemma4.attention.layer_norm_rms_epsilon": 1e-6,
        "gemma4.final_logit_softcapping": 30.0,
    }
    base.update(metadata_overrides)
    return GgufConfigShim(
        architectures=["Gemma4GGUFForCausalLM"],
        model_path=str(empty),
        model_type="gemma4",
        metadata=base,
        vocab_size=262144,
        tie_word_embeddings=True,
    )


def test_moe_gguf_keeps_moe_path(tmp_path):
    """A MoE GGUF (has expert_* keys) keeps moe_enabled + q4_0 expert quant."""
    shim = make_shim({
        "gemma4.expert_count": 128,
        "gemma4.expert_used_count": 8,
        "gemma4.expert_feed_forward_length": 1024,
    }, tmp_path)
    cfg = parse_gguf_config(shim)
    assert cfg.moe_enabled is True
    assert cfg.num_experts == 128
    assert cfg.num_experts_per_tok == 8
    assert cfg.moe_intermediate_size == 1024
    assert cfg.expert_quant == "q4_0"


def test_dense_gguf_loads_without_expert_keys(tmp_path):
    """A dense GGUF (no expert_* keys) must parse and route to the dense path."""
    cfg = parse_gguf_config(make_shim({}, tmp_path))
    assert cfg.moe_enabled is False
    assert cfg.num_experts == 0
    assert cfg.num_experts_per_tok == 0
    assert cfg.moe_intermediate_size == 0
    assert cfg.expert_quant == "none"
    # Sanity: the rest of the geometry still parses.
    assert cfg.num_layers == 4
    assert cfg.model_type == "gemma4"


def test_explicit_zero_experts_is_dense(tmp_path):
    """expert_count=0 in metadata must behave like an absent key."""
    shim = make_shim({
        "gemma4.expert_count": 0,
        "gemma4.expert_used_count": 0,
        "gemma4.expert_feed_forward_length": 0,
    }, tmp_path)
    cfg = parse_gguf_config(shim)
    assert cfg.moe_enabled is False
    assert cfg.expert_quant == "none"
