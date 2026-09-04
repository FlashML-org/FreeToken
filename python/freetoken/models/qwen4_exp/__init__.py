"""Qwen3.8-Flash-Next (model_type qwen4_exp), served text-only.

48 decoder layers on hc_count=4 hyper-connection residual streams R [T, 4*hidden]:
embed -> repeat(1, 4) -> [PLE at zero-based layer 1] -> per layer attn_hc.mix -> (GDN | QSA) -> attn_hc.combine -> mlp_hc.mix -> MoE -> mlp_hc.combine -> top-level mixer.mix -> lm_head.
Layer contract: forward(R [T, 4*hidden], batch) -> R' [T, 4*hidden].

Contracts shared across modules (do not rename):
- The PLE dilated-conv left context lives on the LinearStatePool slots as the declared slot state ``ple_conv`` (config.ple_slot_states -> ModelConfig.slot_states), read back with ``pool.slot_state("ple_conv", layer_id)``; same slot / COW / snapshot lifecycle as conv_states / recurrent_states.
- kvcache.qsa_pool.QSAKVCache(MHAKVCache): ``cmp_k_cache(slot) -> [rows, index_head_dim]`` (compressed index keys, row = kv slot // index_ratio), ``pending_ring(slot) -> [num_req_slots, ring_capacity, index_head_dim]`` (per-request pre-RoPE index-k tail indexed by table_idx, never cleared), ``cmp_scratch_base`` (int, first scratch row for non-closing decode writes). ``slot`` is the sparse layer's order in the attention backend.
"""

from .config import parse_config
from .model import Qwen4ExpForCausalLM
from .weight import (
    iter_weights,
    nvfp4_expert_source_spec,
    load_nvfp4_expert_sources,
    load_nvfp4_expert_sources_parallel,
    load_ple_table,
)

# Official FP8 checkpoints share qwen3_5_moe's block-fp8 expert layout (same
# model.language_model.layers.* keys), so reuse its bank hook; for every other
# expert_quant it defers to the per-quant providers, which resolve this module's
# load_nvfp4_expert_sources via the model spec.
from freetoken.models.qwen3_5_moe.weight import setup_offload_expert_banks

__all__ = ["nvfp4_expert_source_spec", 
    "Qwen4ExpForCausalLM",
    "iter_weights",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
    "load_ple_table",
    "parse_config",
    "setup_offload_expert_banks",
]
