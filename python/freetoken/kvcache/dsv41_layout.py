"""Storage row sizes shared by the V4.1 pool and its budget planner."""

from __future__ import annotations


def dsv41_row_bytes(args, kv_quant="none") -> tuple[int, int, int]:
    if kv_quant not in ("none", "fp8-fp4"):
        raise ValueError(f"Unsupported DeepSeek-V4.1 KV storage format: {kv_quant}")
    head_dim, index_dim = args.head_dim, args.index_head_dim
    if head_dim <= 0 or index_dim <= 0:
        raise ValueError("DeepSeek-V4.1 KV dimensions must be positive")
    if kv_quant == "none":
        return head_dim * 2, head_dim * 2, index_dim * 2
    if head_dim % 32 or index_dim % 32:
        raise ValueError("DeepSeek-V4.1 fp8-fp4 KV dimensions must be divisible by 32")
    return (head_dim + head_dim // 32,
            head_dim // 2 + head_dim // 16,
            index_dim // 2 + index_dim // 32)
