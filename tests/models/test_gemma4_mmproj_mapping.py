"""Pure mapping regression coverage for the separate Gemma4 projector GGUF."""

import pytest

from freetoken.models.gemma4.gguf import _gemma4_image_token_id, gemma4_mmproj_param_name


def test_gemma4_mmproj_tensor_mapping() -> None:
    """Every projector tensor family maps into its corresponding vision module."""
    assert gemma4_mmproj_param_name("mm.input_projection.weight") == "embed_vision.embedding_projection.weight"
    assert gemma4_mmproj_param_name("v.patch_embd.weight") == "vision_tower.patch_embedder.input_proj.weight"
    assert gemma4_mmproj_param_name("v.blk.7.attn_q.weight") == "vision_tower.encoder.layers.7.self_attn.q_proj.weight"
    assert gemma4_mmproj_param_name("v.blk.7.ffn_down.weight") == "vision_tower.encoder.layers.7.mlp.down_proj.weight"
    assert gemma4_mmproj_param_name("v.blk.7.ln1.weight") == "vision_tower.encoder.layers.7.input_layernorm.weight"


def test_gemma4_mmproj_rejects_unknown_names() -> None:
    """Unexpected projector data cannot silently bind to an unrelated parameter."""
    assert gemma4_mmproj_param_name("v.blk.bad.attn_q.weight") is None
    assert gemma4_mmproj_param_name("unrelated.weight") is None


def test_gemma4_gguf_image_placeholder_uses_checkpoint_token() -> None:
    """Gemma's actual ``<|image>`` placeholder is not inferred from a fixed id."""
    assert _gemma4_image_token_id({"tokenizer.ggml.tokens": ["x", "<|image>"]}) == 1


def test_gemma4_gguf_image_placeholder_rejects_ambiguous_tokenizers() -> None:
    """A conversion that carries multiple candidate placeholders must fail closed."""
    with pytest.raises(ValueError, match="exactly one image placeholder"):
        _gemma4_image_token_id({"tokenizer.ggml.tokens": ["<image>", "<|image>"]})
