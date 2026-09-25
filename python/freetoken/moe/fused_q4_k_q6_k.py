"""Exact Q4_K/Q6_K routed-expert execution for exceptional Qwen GGUF layers.

Qwen3.6 Q4_K_M stores almost every routed down projection as Q5_K, but a few
late layers use Q6_K.  The primary Q4_K/Q5_K cache cannot store both row sizes,
so this kernel accepts Q4_K gate/up rows from that cache and Q6_K down rows from
the small auxiliary cache.  Both id tensors address the same routed experts,
but they intentionally name slots in their respective caches.
"""

from __future__ import annotations

import torch

from freetoken.layers.activation import silu_and_mul
from freetoken.models.gguf.dequant import GGML_Q4_K, GGML_Q6_K


def fused_experts_gguf_q4_k_q6_k(
    hidden_states: torch.Tensor,
    gate_up_q4_k: torch.Tensor,
    down_q6_k: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_ids: torch.Tensor,
    down_ids: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    """Run Q4_K gate/up and Q6_K down using their independent cache slots."""
    if activation != "silu":
        raise ValueError(
            "Qwen mixed GGUF experts require the checkpoint's silu SwiGLU activation, "
            f"got {activation!r}"
        )
    if gate_up_ids.shape != down_ids.shape:
        raise ValueError("Q4_K and Q6_K routed id tensors must have the same shape")
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    tokens = hidden_states.shape[0]
    top_k = gate_up_ids.shape[1]
    fused_width = gate_up_q4_k.shape[1]
    hidden_size = down_q6_k.shape[1]
    gate_up = ggml_moe_a8_vec(
        hidden_states, gate_up_q4_k, gate_up_ids, top_k,
        int(GGML_Q4_K), fused_width, tokens,
    )
    intermediate = silu_and_mul(gate_up)
    output = ggml_moe_a8_vec(
        intermediate, down_q6_k, down_ids, 1,
        int(GGML_Q6_K), hidden_size, tokens * top_k,
    )
    output = output.reshape(tokens, top_k, hidden_size)
    return (output * topk_weights.reshape(tokens, top_k, 1).to(output.dtype)).sum(dim=1)


__all__ = ["fused_experts_gguf_q4_k_q6_k"]
