from __future__ import annotations

from dataclasses import dataclass


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
    indexer_budget: int
    indexer_compress_ratio: int
    output_gate_type: str


__all__ = ["Qwen4ExpArgs"]
