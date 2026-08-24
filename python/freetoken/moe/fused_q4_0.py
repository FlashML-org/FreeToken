"""Grouped expert GEMM over native GGUF banks (borrowed ggml MoE kernels).

Ports vLLM/sglang's ``_fused_moe_gguf`` MMVQ path onto FreeToken's offload-cache
interface: the experts are streamed to the GPU as packed block bytes and
dequantized *inside* ``ggml_moe_a8_vec`` -- no bf16 expert copy is materialized. We
use the MMVQ (vector) kernel for both prefill and decode: it consumes ``topk_ids``
directly (no ``moe_align_block_size`` needed) and on small batches it is the right
choice anyway. ``topk_ids`` already index the streamed cache slots (decode) or the
materialized layer positions (prefill).

This module is general over any quantization type supported by the ``ggml_moe_a8_vec``
kernel (all types in ``MOE_VEC_TYPES``, which includes all 19 quantized types). Q4_0
is currently the only type the rest of the pipeline plumbs through; support for other
types is added by parametrizing the quant type at the MoE bank loader, dequant.py, and
moe/expert_banks.py level.
"""

from __future__ import annotations

import torch

from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
from freetoken.models.gguf.dequant import GGML_Q4_0, MOE_VEC_TYPES

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}


def fused_experts_gguf(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, 2I, H//32*18] uint8 (or other quant format)
    down_q: torch.Tensor,  # [num_slots, H, I//32*18] uint8 (or other quant format)
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    quant_type: int,
    down_quant_type: int | None = None,
) -> torch.Tensor:
    """Fused GGUF MoE expert compute over any MMVQ-supported quantization type.

    This kernel operates directly on packed quantized weights (no materialization to bf16);
    dequantization happens inside the ``ggml_moe_a8_vec`` CUDA kernel. ``quant_type`` must be
    in ``MOE_VEC_TYPES``, which mirrors the supported types in ``ggml_moe_a8_vec``
    (gguf_kernel.cu:559).

    ``quant_type`` is the gate_up bank's type; ``down_quant_type`` defaults to it. They may
    differ because gate_up and down are separate banks with separate slot pools, and
    llama.cpp routinely quantizes the down projection differently from gate/up. What may
    NOT differ is the type *within* one bank across layers -- that pool is one allocation.
    """
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    if down_quant_type is None:
        down_quant_type = quant_type
    for label, qt in (("gate_up", quant_type), ("down", down_quant_type)):
        if qt not in MOE_VEC_TYPES:
            from freetoken.models.gguf.dequant import GGML_NAME
            raise NotImplementedError(
                f"fused GGUF MoE kernel does not support quant type "
                f"{GGML_NAME.get(qt, qt)} for the {label} bank "
                f"(only {sorted(MOE_VEC_TYPES)} supported)"
            )

    act_fn = _ACT.get(activation)
    if act_fn is None:
        raise ValueError(f"unsupported MoE activation {activation!r}")

    num_tokens = hidden_states.shape[0]
    n2 = gate_up_q.shape[1]  # 2 * intermediate
    h = down_q.shape[1]  # hidden
    top_k = topk_ids.shape[1]
    qt = int(quant_type)

    # gate_up: [num_tokens*top_k, 2I] -> activation -> [num_tokens*top_k, I]
    gate_up = ggml_moe_a8_vec(hidden_states, gate_up_q, topk_ids, top_k, qt, n2, num_tokens)
    inter = act_fn(gate_up)
    # down: each of the num_tokens*top_k intermediate rows uses its own expert id.
    out = ggml_moe_a8_vec(inter, down_q, topk_ids, 1, int(down_quant_type), h, num_tokens * top_k)
    out = out.reshape(num_tokens, top_k, h) * topk_weights.reshape(num_tokens, top_k, 1).to(
        out.dtype
    )
    return out.sum(dim=1)


def fused_experts_gguf_q4_0(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, 2I, H//32*18] uint8
    down_q: torch.Tensor,  # [num_slots, H, I//32*18] uint8
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    """GGUF Q4_0 MoE (backward-compat wrapper).

    This is a thin wrapper around ``fused_experts_gguf`` that hardcodes the Q4_0 type.
    All existing callers use this for now; the general function is available for future
    multi-quant pipelines.
    """
    return fused_experts_gguf(hidden_states, gate_up_q, down_q, topk_weights, topk_ids, activation, int(GGML_Q4_0))


__all__ = ["fused_experts_gguf", "fused_experts_gguf_q4_0"]
