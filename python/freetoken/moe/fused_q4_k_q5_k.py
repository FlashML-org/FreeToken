"""Mixed Q4_K/Q5_K GGUF routed-expert execution for Qwen3.6 MoE checkpoints.

The GGUF model recipe names this combination ``Q4_K_M``, but its tensor table
stores the gate and up expert projections as Q4_K and the down projection as
Q5_K.  The borrowed HIP GGML kernels dispatch one quant type per matrix, so
this module intentionally launches one packed Q4_K MoE GEMV followed by one
packed Q5_K MoE GEMV.  Neither weight is dequantized to a persistent bf16 copy.
"""

from __future__ import annotations

import torch

from freetoken.layers.activation import silu_and_mul
from freetoken.models.gguf.dequant import GGML_Q4_K, GGML_Q5_K


def fused_experts_gguf_q4_k_q5_k(
    hidden_states: torch.Tensor,
    gate_up_q4_k: torch.Tensor,
    down_q5_k: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    """Run packed Q4_K gate/up then packed Q5_K down over routed experts.

    ``topk_ids`` already name the materialized GGUF expert-cache slots.  Qwen
    uses SwiGLU, so only ``silu`` is accepted here.  Explicit validation prevents
    a future model family from silently receiving Qwen's activation semantics.
    """
    if activation != "silu":
        raise ValueError(
            "Qwen mixed GGUF experts require the checkpoint's silu SwiGLU activation, "
            f"got {activation!r}"
        )
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    tokens = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    fused_width = gate_up_q4_k.shape[1]
    hidden_size = down_q5_k.shape[1]
    gate_up = ggml_moe_a8_vec(
        hidden_states, gate_up_q4_k, topk_ids, top_k, int(GGML_Q4_K), fused_width, tokens
    )
    intermediate = silu_and_mul(gate_up)
    output = ggml_moe_a8_vec(
        intermediate, down_q5_k, topk_ids, 1, int(GGML_Q5_K), hidden_size, tokens * top_k
    )
    output = output.reshape(tokens, top_k, hidden_size)
    return (output * topk_weights.reshape(tokens, top_k, 1).to(output.dtype)).sum(dim=1)


__all__ = ["fused_experts_gguf_q4_k_q5_k"]
