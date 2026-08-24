"""Qwen3.5-MoE GGUF adapter: build the FreeToken ``ModelConfig`` and stream weights
from a llama.cpp ``qwen35moe`` checkpoint.

The geometry is identical to the HF qwen3_5_moe model (hybrid GDN/full attention on a
``full_attention_interval`` stride, 256 routed experts + a shared expert, NextN/MTP head),
so this produces the *same* ``ModelConfig`` as ``qwen3_5_moe.config.parse_config`` -- only
the source is GGUF KV metadata instead of a HF config object.

Tensor-name mapping is the inverse of llama.cpp's ``gguf-py/gguf/tensor_mapping.py``.
The one non-obvious part is the GDN projections. llama.cpp's *qwen3.5* mapping splits
what qwen3next fused::

    attn_qkv    <- model.layers.{i}.linear_attn.in_proj_qkv
    attn_gate   <- model.layers.{i}.linear_attn.in_proj_z
    ssm_beta    <- model.layers.{i}.linear_attn.in_proj_b
    ssm_alpha   <- model.layers.{i}.linear_attn.in_proj_a

FreeToken's HF loader already knows how to put those back together -- see ``_PT_FP8_FUSE``
and ``_PT_BF16_FUSE`` in ``weight.py``, which fuse ``(in_proj_qkv, in_proj_z) ->
in_proj_qkvz`` and ``(in_proj_b, in_proj_a) -> in_proj_ba`` in that order. We emit the
same fused buffers here so the model code sees one representation regardless of source.

Verified against vcruz305/Ornith-1.5-35B-A3B-GGUF (IQ3_M), whose metadata gives
block_count=41 (40 decoder layers + 1 NextN block), embedding_length=2048,
head_count=16, head_count_kv=2, key_length=value_length=256, expert_count=256,
expert_used_count=8, expert_feed_forward_length=512, full_attention_interval=4,
ssm.conv_kernel=4, ssm.state_size=128, ssm.group_count=16, ssm.time_step_rank=32,
ssm.inner_size=4096. Those are self-consistent: the packed ``attn_qkv`` output width of
8192 is exactly q(16*128) + k(16*128) + v(32*128), i.e. num_k_heads == group_count and
num_v_heads == time_step_rank, both with head_dim == state_size == 128.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

import torch

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)
from freetoken.models.gguf.dequant import GGML_NAME, dequantize

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim

_ARCH = "qwen35moe"


def _kv(shim: "GgufConfigShim", key: str, default: Any = None) -> Any:
    """Read ``qwen35moe.<key>`` from the GGUF metadata."""
    val = shim.metadata.get(f"{_ARCH}.{key}", default)
    if val is None and default is None:
        raise ValueError(f"GGUF {shim.model_path}: missing required key {_ARCH}.{key}")
    return val


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    block_count = int(_kv(shim, "block_count"))
    # llama.cpp appends the NextN/MTP block to the decoder stack. FreeToken serves
    # text-only without speculative decoding, so the MTP block is not a decoder layer.
    nextn = int(_kv(shim, "nextn_predict_layers", 0))
    num_layers = block_count - nextn

    hidden_size = int(_kv(shim, "embedding_length"))
    num_qo_heads = int(_kv(shim, "attention.head_count"))
    num_kv_heads = int(_kv(shim, "attention.head_count_kv"))
    head_dim = int(_kv(shim, "attention.key_length"))
    rms_eps = float(_kv(shim, "attention.layer_norm_rms_epsilon"))
    rope_base = float(_kv(shim, "rope.freq_base"))
    rotary_dim = int(_kv(shim, "rope.dimension_count"))
    max_pos = int(_kv(shim, "context_length"))

    num_experts = int(_kv(shim, "expert_count", 0))
    experts_per_tok = int(_kv(shim, "expert_used_count", 0))
    moe_inter = int(_kv(shim, "expert_feed_forward_length", 0))
    shared_inter = int(_kv(shim, "expert_shared_feed_forward_length", 0))

    # GDN geometry. state_size is the per-head dim; group_count is the number of k heads
    # and time_step_rank the number of v heads (see module docstring for the arithmetic
    # that pins this down against the packed attn_qkv width).
    conv_kernel = int(_kv(shim, "ssm.conv_kernel"))
    state_size = int(_kv(shim, "ssm.state_size"))
    num_k_heads = int(_kv(shim, "ssm.group_count"))
    num_v_heads = int(_kv(shim, "ssm.time_step_rank"))
    inner_size = int(_kv(shim, "ssm.inner_size"))
    if num_v_heads * state_size != inner_size:
        raise ValueError(
            f"GGUF {shim.model_path}: ssm.time_step_rank({num_v_heads}) * "
            f"ssm.state_size({state_size}) != ssm.inner_size({inner_size}); the GDN head "
            "layout assumed by this adapter does not hold for this checkpoint"
        )

    # llama.cpp writes the stride, not a per-layer list: layer i is full attention when
    # (i + 1) % interval == 0. For Ornith (interval=4, 40 layers) that is 3,7,...,39.
    interval = int(_kv(shim, "full_attention_interval"))
    full_ids = tuple(i for i in range(num_layers) if (i + 1) % interval == 0)
    linear_ids = tuple(i for i in range(num_layers) if i not in set(full_ids))

    full_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=max_pos,
        base=rope_base,
        scaling=None,
    )
    groups = tuple(
        sorted(
            (
                FullAttentionGroupConfig(
                    name="full",
                    layer_ids=full_ids,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    rotary_config=full_rotary,
                ),
                LinearGatedDeltaGroupConfig(
                    name="linear",
                    layer_ids=linear_ids,
                    num_key_heads=num_k_heads,
                    num_value_heads=num_v_heads,
                    key_head_dim=state_size,
                    value_head_dim=state_size,
                    conv_kernel_dim=conv_kernel,
                    output_gate=True,
                ),
            ),
            key=lambda g: g.layer_ids[0] if g.layer_ids else 1 << 30,
        )
    )

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        vocab_size=shim.vocab_size,
        intermediate_size=0,  # every layer is MoE in qwen35moe
        hidden_act="silu",
        rms_norm_eps=rms_eps,
        tie_word_embeddings=shim.tie_word_embeddings,
        rotary_config=full_rotary,
        num_experts=num_experts,
        num_experts_per_tok=experts_per_tok,
        moe_intermediate_size=moe_inter,
        shared_expert_intermediate_size=shared_inter,
        norm_topk_prob=True,
        moe_enabled=num_experts > 0,
        use_qk_norm=True,
        model_type=_ARCH,
        architectures=["Qwen35MoeGGUFForCausalLM"],
        vision_config=None,
        image_token_id=None,
        attention_groups=groups,
        expert_quant="gguf",
        weight_block_size=None,
        attn_quant="gguf",
        dense_quant="gguf",
        lm_head_quant="gguf",
    )


# --------------------------------------------------------------------------------------
# Tensor-name mapping (inverse of llama.cpp gguf-py/gguf/tensor_mapping.py for qwen3.5)
# --------------------------------------------------------------------------------------

# Per-layer 1:1 renames that need no reshaping or fusing.
_LAYER_MAP: dict[str, str] = {
    # shared by both layer kinds
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    # full-attention layers
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    # GDN (linear-attention) layers
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    # MoE router + shared expert
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
}

# Pairs llama.cpp splits that FreeToken's model code wants fused, in concat order.
# Mirrors _PT_FP8_FUSE / _PT_BF16_FUSE in weight.py.
_FUSE: dict[str, tuple[str, str]] = {
    "linear_attn.in_proj_qkvz.weight": ("attn_qkv.weight", "attn_gate.weight"),
    "linear_attn.in_proj_ba.weight": ("ssm_beta.weight", "ssm_alpha.weight"),
}

# Routed-expert stacks: [num_experts, out, in] packed blocks, handled by the offload
# expert-bank loader rather than yielded as ordinary parameters.
_EXPERT_SUFFIXES = (
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
)

_GLOBAL_MAP: dict[str, str] = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}


def gguf_name_to_freetoken(name: str, num_layers: int) -> str | None:
    """Map one llama.cpp tensor name to its FreeToken parameter name.

    Returns ``None`` for tensors FreeToken does not consume (the NextN/MTP block, and
    the routed-expert stacks, which the expert-bank loader reads directly).
    """
    if name in _GLOBAL_MAP:
        return _GLOBAL_MAP[name]
    if not name.startswith("blk."):
        return None
    _, idx, suffix = name.split(".", 2)
    layer = int(idx)
    if layer >= num_layers:
        return None  # the trailing NextN/MTP block: served text-only, no speculation
    if suffix.startswith("nextn."):
        return None
    if suffix in _EXPERT_SUFFIXES:
        return None
    mapped = _LAYER_MAP.get(suffix)
    if mapped is None:
        return None
    return f"model.layers.{layer}.{mapped}"


__all__ = [
    "parse_gguf_config",
    "gguf_name_to_freetoken",
    "_FUSE",
    "_EXPERT_SUFFIXES",
]
