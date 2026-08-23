"""Grouped expert GEMM over mixed-type GGUF banks (borrowed ggml MoE kernels).

The generalization of :mod:`freetoken.moe.fused_q4_0` for checkpoints whose
routed-expert quant type varies per layer (Unsloth Dynamic laguna: gate/up
IQ1_S or IQ2_XXS, down IQ3_XXS or IQ4_XS). Because per-expert byte sizes then
differ across layers, the banks are FLAT padded slots -- ``[num_slots,
stride_bytes]`` uint8 with each expert's real payload in the leading bytes --
and the kernels read them via ``expert_stride_bytes``. Geometry (quant type,
output rows) rides in per-call arguments; MMVQ serves prefill and decode like
the q4_0 path.
"""

from __future__ import annotations

import torch

from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}


def fused_experts_gguf(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, gu_stride] uint8 (flat padded slots)
    down_q: torch.Tensor,  # [num_slots, dn_stride] uint8
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    *,
    gate_up_type: int,
    down_type: int,
    gate_up_rows: int,  # 2 * intermediate
    down_rows: int,  # hidden
) -> torch.Tensor:
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    act_fn = _ACT.get(activation)
    if act_fn is None:
        raise ValueError(f"unsupported MoE activation {activation!r}")

    num_tokens = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    assert gate_up_q.dim() == 2 and down_q.dim() == 2, "gguf banks are flat padded slots"

    gate_up = ggml_moe_a8_vec(
        hidden_states, gate_up_q, topk_ids, top_k, int(gate_up_type),
        gate_up_rows, num_tokens, gate_up_q.shape[1],
    )
    inter = act_fn(gate_up)
    out = ggml_moe_a8_vec(
        inter, down_q, topk_ids, 1, int(down_type),
        down_rows, num_tokens * top_k, down_q.shape[1],
    )
    out = out.reshape(num_tokens, top_k, down_rows) * topk_weights.reshape(
        num_tokens, top_k, 1
    ).to(out.dtype)
    return out.sum(dim=1)


__all__ = ["fused_experts_gguf"]
