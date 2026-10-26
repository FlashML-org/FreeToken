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

import torch

from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
from freetoken.models.gguf.dequant import GGML_Q4_0, GGML_Q8_0, GGML_Q4_K

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}


def fused_experts_gguf(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, 2I, H//32*18] uint8
    down_q: torch.Tensor,  # [num_slots, H, I//32*18] uint8
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    quant_type: int,
) -> torch.Tensor:
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    act_fn = _ACT.get(activation)
    if act_fn is None:
        raise ValueError(f"unsupported MoE activation {activation!r}")

    num_tokens = hidden_states.shape[0]
    n2 = gate_up_q.shape[1]  # 2 * intermediate
    h = down_q.shape[1]  # hidden
    top_k = topk_ids.shape[1]
    qt = int(quant_type)

    props = torch.cuda.get_device_properties(hidden_states.device)
    shared_memory = int(getattr(props, "shared_memory_per_block_optin", 0))
    if shared_memory <= 0:
        shared_memory = int(props.shared_memory_per_block)
    # ggml's expert-index helper stores one int32 per routed row in shared memory.
    chunk_tokens = max(1, shared_memory // 4 // top_k)

    def run_chunk(start: int, end: int) -> torch.Tensor:
        chunk_ids = topk_ids[start:end]
        chunk_weights = topk_weights[start:end]
        chunk_size = end - start
        gate_up = ggml_moe_a8_vec(
            hidden_states[start:end], gate_up_q, chunk_ids, top_k, qt, n2, chunk_size
        )
        inter = act_fn(gate_up)
        # The down projection indexes one expert per routed row, so its index table
        # contains chunk_size * top_k entries.
        out = ggml_moe_a8_vec(
            inter, down_q, chunk_ids, 1, qt, h, chunk_size * top_k
        )
        out = out.reshape(chunk_size, top_k, h) * chunk_weights.reshape(
            chunk_size, top_k, 1
        ).to(out.dtype)
        return out.sum(dim=1)

    if num_tokens <= chunk_tokens:
        return run_chunk(0, num_tokens)
    return torch.cat(
        [
            run_chunk(start, min(start + chunk_tokens, num_tokens))
            for start in range(0, num_tokens, chunk_tokens)
        ],
        dim=0,
    )


def fused_experts_gguf_q4_0(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,
    down_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    return fused_experts_gguf(
        hidden_states,
        gate_up_q,
        down_q,
        topk_weights,
        topk_ids,
        activation,
        GGML_Q4_0,
    )


def fused_experts_gguf_q8_0(
    hidden_states, gate_up_q, down_q, topk_weights, topk_ids, activation
):
    return fused_experts_gguf(
        hidden_states, gate_up_q, down_q, topk_weights, topk_ids, activation, GGML_Q8_0
    )


def fused_experts_gguf_q4_k(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,
    down_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    return fused_experts_gguf(
        hidden_states,
        gate_up_q,
        down_q,
        topk_weights,
        topk_ids,
        activation,
        GGML_Q4_K,
    )


__all__ = ["fused_experts_gguf", "fused_experts_gguf_q4_0", "fused_experts_gguf_q8_0", "fused_experts_gguf_q4_k"]
