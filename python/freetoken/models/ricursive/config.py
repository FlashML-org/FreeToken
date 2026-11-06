from __future__ import annotations

from dataclasses import replace
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
)

# The model card serves up to 1M tokens; the config's 2M rows would pin 1 GiB of VRAM in the fp32 cos/sin table.
_MAX_POSITION = 1 << 20


def parse_config(hf: Any) -> ModelConfig:
    layers = int(hf.num_hidden_layers)
    types = tuple(hf.layer_types)
    if len(types) != layers or set(types) != {"full_attention", "sliding_attention"}:
        raise ValueError("Ricursive requires one full_attention/sliding_attention type per layer")
    if hf.hidden_act != "silu" or not (hf.use_gated_attention and hf.gate_full_precision and hf.use_valres):
        raise ValueError("Ricursive requires SwiGLU, fp32 attention gates and value residuals")
    head_dim, num_kv_heads = int(hf.head_dim), int(hf.num_key_value_heads)
    rope = RotaryConfig(
        head_dim=head_dim, rotary_dim=head_dim,
        max_position=min(int(hf.max_position_embeddings), _MAX_POSITION), base=float(hf.rope_theta), scaling=None,
    )
    return ModelConfig(
        num_layers=layers, num_qo_heads=int(hf.num_attention_heads), num_kv_heads=num_kv_heads,
        head_dim=head_dim, hidden_size=int(hf.hidden_size), vocab_size=int(hf.vocab_size),
        intermediate_size=int(hf.intermediate_size), hidden_act="silu",
        rms_norm_eps=float(hf.rms_norm_eps), tie_word_embeddings=bool(hf.tie_word_embeddings),
        rotary_config=rope, num_experts=0, num_experts_per_tok=0, moe_intermediate_size=0,
        norm_topk_prob=False, model_type="ricursive", architectures=list(hf.architectures),
        use_qk_norm=True, embedding_scale=hf.embed_scale, swiglu_limit=float(hf.ffn_clamp_threshold or 0) or None,
        attention_groups=(
            SWAAttentionGroupConfig(
                name="swa", layer_ids=tuple(i for i, kind in enumerate(types) if kind == "sliding_attention"),
                num_kv_heads=num_kv_heads, head_dim=head_dim, rotary_config=rope,
                sliding_window=int(hf.sliding_window),
            ),
            FullAttentionGroupConfig(
                name="full", layer_ids=tuple(i for i, kind in enumerate(types) if kind == "full_attention"),
                num_kv_heads=num_kv_heads, head_dim=head_dim, rotary_config=replace(rope, rotary_dim=0),
            ),
        ),
    )
