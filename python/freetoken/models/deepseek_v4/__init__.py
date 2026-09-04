"""DeepSeek-V4-Flash support for FreeToken.

This package ports the official ``inference/model.py`` reference (MLA + CSA/HCA
compressors + Lightning Indexer + manifold-constrained Hyper-Connections) onto
FreeToken's primitives. The exotic ops are reimplemented as Triton kernels (see
``freetoken/kernel/triton/dsv4_*.py``); the routed FP4 experts are served from
FreeToken's :class:`~freetoken.moe.offload_cache.OffloadMoeCache` so only a subset
of experts is resident on the GPU (the framework's core acceleration).

DeepSeek-V4-Flash is a first-class registered model on the shared paged-KV engine:
its window / compressed-attention / compressed-index KV live in DSV4-owned pools
addressed by page tables, and sparse attention is a physical-slot gather (see
:mod:`freetoken.attention.dsv4_sparse` and :mod:`freetoken.kvcache.dsv4_paged_pool`).
"""

from .args import DeepseekV4Args, load_args
from .config import parse_config
from .gguf import (
    convert_deepseek4_to_gguf,
    is_gguf_model,
    iter_gguf_weights,
    parse_gguf_config,
)
from .gguf_experts import gguf_expert_types, load_gguf_expert_sources
from .model import DeepseekV4ForCausalLM
from .weight import iter_weights, load_dsfp4_expert_sources

__all__ = [
    "DeepseekV4Args",
    "load_args",
    "parse_config",
    "DeepseekV4ForCausalLM",
    "iter_weights",
    "load_dsfp4_expert_sources",
    "parse_gguf_config",
    "iter_gguf_weights",
    "convert_deepseek4_to_gguf",
    "is_gguf_model",
    "gguf_expert_types",
    "load_gguf_expert_sources",
]
