"""Triton kernels for DeepSeek-V4.1 Compressed Sparse Attention 2 over packed KV pools.

``row_format``  device helpers: load a tile of packed rows (bf16 / fp8+e8m0 / fp4+e4m3 / fp4+e8m0)
                and dequantize it to fp32, encode / decode e2m1 nibbles.
``pack``        host wrappers: quantize rows into a packed pool (``pack_rows``) and read them back
                (``unpack_rows``), matching the reference ``act_quant`` / ``fp4_act_quant`` numerics.
``sparse_attn`` paged sparse attention gathering window (fp8) and compressed (fp4) rows.
``indexer``     Lightning-indexer logits over fp4 index keys, full-range or candidate-restricted.
"""
