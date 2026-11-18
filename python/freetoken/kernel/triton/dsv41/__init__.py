"""Triton kernels for DeepSeek-V4.1 Compressed Sparse Attention 2 over packed KV pools.

The packed row formats themselves (``row_format``, ``pack``) live in ``kernel/triton/dsv4``, shared
with the DeepSeek-V4 pool.

``sparse_attn`` paged sparse attention gathering window (fp8) and compressed (fp4) rows.
``indexer``     Lightning-indexer logits over fp4 index keys, full-range or candidate-restricted.
"""
