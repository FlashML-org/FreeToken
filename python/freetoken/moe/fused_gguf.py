"""Grouped expert GEMM over native GGUF Q4_K gate/up + Q8_0 down banks.

Ports vLLM/sglang's ``_fused_moe_gguf`` MMVQ path onto FreeToken's offload-cache
interface: experts are streamed to the GPU as packed block bytes and dequantized
*inside* ``ggml_moe_a8_vec`` -- no bf16 expert copy is materialized. ``gate_up`` stays
native Q4_K; ``down`` is stored as Q8_0 (re-quantized at load from the GGUF's
Q5_K/Q6_K -- 8-bit, >= the source precision, so no quality loss) because the offload
cache needs a single uniform per-bank format. We use the MMVQ (vector) kernel for both
prefill and decode, mirroring ``fused_experts_gguf_q4_0``.
"""

from __future__ import annotations

import torch

from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
from freetoken.models.gguf.dequant import GGML_Q4_K, GGML_Q8_0

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}


def fused_experts_gguf(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, 2I, row_bytes(H, Q4_K)] uint8
    down_q: torch.Tensor,  # [num_slots, H, row_bytes(I, Q8_0)] uint8
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    act_fn = _ACT.get(activation)
    if act_fn is None:
        raise ValueError(f"unsupported MoE activation {activation!r}")

    num_tokens = hidden_states.shape[0]
    n2 = gate_up_q.shape[1]  # 2 * intermediate
    h = down_q.shape[1]  # hidden
    top_k = topk_ids.shape[1]

    # "moe_gate_up" / "moe_down" record_function labels = the profiler-segmented
    # expert-GEMM halves of the fused MoE forward (Inc 2, .plans/rocm-perf-parity).
    with torch.profiler.record_function("moe_gate_up"):
        gate_up = ggml_moe_a8_vec(
            hidden_states, gate_up_q, topk_ids, top_k, int(GGML_Q4_K), n2, num_tokens
        )
    inter = act_fn(gate_up)
    with torch.profiler.record_function("moe_down"):
        out = ggml_moe_a8_vec(
            inter, down_q, topk_ids, 1, int(GGML_Q8_0), h, num_tokens * top_k
        )
    out = out.reshape(num_tokens, top_k, h) * topk_weights.reshape(num_tokens, top_k, 1).to(
        out.dtype
    )
    return out.sum(dim=1)


__all__ = ["fused_experts_gguf"]
