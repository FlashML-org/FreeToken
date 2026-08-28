"""Grouped expert GEMM over native GGUF Q4_0 banks (borrowed ggml MoE kernels).

Ports vLLM/sglang's ``_fused_moe_gguf`` MMVQ path onto FreeToken's offload-cache
interface: the experts are streamed to the GPU as packed Q4_0 block bytes and
dequantized *inside* ``ggml_moe_a8_vec`` -- no bf16 expert copy is materialized. We
use the MMVQ (vector) kernel for both prefill and decode: it consumes ``topk_ids``
directly (no ``moe_align_block_size`` needed) and on small batches it is the right
choice anyway. ``topk_ids`` already index the streamed cache slots (decode) or the
materialized layer positions (prefill).
"""

from __future__ import annotations

import os

import torch

from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
from freetoken.models.gguf.dequant import GGML_Q4_0

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}


def _use_fp32_intermediate() -> bool:
    """Return whether the explicitly experimental Q4_0 FP32 path was requested.

    The flag is deliberately strict and opt-in.  A normal service launch never
    changes precision or throughput behavior merely because the environment
    contains an unrelated truthy-looking value.
    """
    return os.environ.get("FREETOKEN_GGUF_MOE_FP32_INTERMEDIATE") == "1"


def fused_experts_gguf_q4_0(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, 2I, H//32*18] uint8
    down_q: torch.Tensor,  # [num_slots, H, I//32*18] uint8
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
    qt = int(GGML_Q4_0)
    use_fp32_intermediate = _use_fp32_intermediate()

    # gate_up: [num_tokens*top_k, 2I] -> activation -> [num_tokens*top_k, I].
    # The opt-in temporary is only for an AMD HIP experiment.  It mirrors the
    # FP32 vector destination used by llama.cpp without changing the public
    # result dtype returned to the transformer layer.
    gate_up = ggml_moe_a8_vec(
        hidden_states,
        gate_up_q,
        topk_ids,
        top_k,
        qt,
        n2,
        num_tokens,
        output_fp32=use_fp32_intermediate,
    )
    inter = act_fn(gate_up)
    # down: each of the num_tokens*top_k intermediate rows uses its own expert id.
    out = ggml_moe_a8_vec(
        inter,
        down_q,
        topk_ids,
        1,
        qt,
        h,
        num_tokens * top_k,
        output_fp32=use_fp32_intermediate,
    )
    out = out.reshape(num_tokens, top_k, h) * topk_weights.reshape(num_tokens, top_k, 1).to(
        out.dtype
    )
    # Preserve the original BF16 caller contract even when the temporary
    # candidate computed its two quantized vector products in FP32.
    return out.sum(dim=1).to(hidden_states.dtype)


__all__ = ["fused_experts_gguf_q4_0"]
