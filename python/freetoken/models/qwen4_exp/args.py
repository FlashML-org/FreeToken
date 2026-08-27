from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen4VisionConfig:
    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_position_embeddings: int
    out_hidden_size: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    in_channels: int
    hidden_act: str
    deepstack_visual_indexes: tuple[int, ...]


@dataclass(frozen=True)
class Qwen4ExpArgs:
    hc_count: int
    hc_lowrank: int
    ple_layer_ids: tuple[int, ...]
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    split_ngram_parts: int
    eos_token_id: int
    indexer_n_heads: int
    indexer_kv_heads: int
    indexer_head_dim: int
    indexer_budget: int
    indexer_compress_ratio: int
    output_gate_type: str
    mrope_section: tuple[int, int, int]
    mrope_interleaved: bool


__all__ = ["Qwen4ExpArgs", "Qwen4VisionConfig"]
