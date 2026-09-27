"""Grouped expert GEMM over native GGUF Q4_0 banks (borrowed ggml MoE kernels).

Thin compatibility wrapper over :mod:`freetoken.moe.gguf_experts`, which now carries
the generic, quant-type-parameterized implementation. Kept so the original gemma4
Q4_0 entry point keeps working.
"""

from __future__ import annotations

import torch

from freetoken.models.gguf.dequant import GGML_Q4_0


def fused_experts_gguf_q4_0(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, 2I, H//32*18] uint8
    down_q: torch.Tensor,  # [num_slots, H, I//32*18] uint8
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    from freetoken.moe.gguf_experts import fused_experts_gguf

    return fused_experts_gguf(
        hidden_states, gate_up_q, down_q, topk_weights, topk_ids, activation,
        int(GGML_Q4_0), int(GGML_Q4_0),
    )


__all__ = ["fused_experts_gguf_q4_0"]
