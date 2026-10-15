"""Grouped expert GEMM over native GGUF packed banks (borrowed ggml MoE kernels).

Ports vLLM/sglang's ``_fused_moe_gguf`` MMVQ path onto FreeToken's offload-cache
interface: the experts are streamed to the GPU as packed block bytes of the file's own
ggml type and dequantized *inside* ``ggml_moe_a8_vec`` -- no bf16 expert copy is
materialized. We use the MMVQ (vector) kernel for both prefill and decode: it consumes
``topk_ids`` directly (no ``moe_align_block_size`` needed) and on small batches it is the
right choice anyway. ``topk_ids`` already index the streamed cache slots (decode) or the
materialized layer positions (prefill).

A K-quant bank row may be padded with whole super-blocks to a 16-byte multiple
(:func:`freetoken.models.gguf.dequant.padded_row_bytes`), because the offload cache moves
rows with 16-byte-granular kernels while a Q6_K row of 1536 elements is 1260 bytes. The
kernel is told the row's element width (padding included) and the activation is zero-padded
to match: an all-zero weight block dequantizes to exactly zero, and the matching zero
activation block contributes exactly zero to the dot product.
"""

from __future__ import annotations

import torch

from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
from freetoken.models.gguf.dequant import row_numel

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}


def _pad_cols(x: torch.Tensor, cols: int) -> torch.Tensor:
    """Zero-pad the last dim of ``x`` to ``cols`` (no-op when it already matches).

    The padding must be whole blocks of zeros: the GEMM quantizes the activation in 32-element
    blocks, so an all-zero block scales to zero and leaves the real blocks' quantization and
    dot products untouched. The returned tensor is contiguous -- the kernel walks each row of
    ``x`` by ``cols`` elements from the base pointer.
    """
    if x.shape[-1] == cols:
        return x.contiguous()
    assert x.shape[-1] < cols, (x.shape, cols)
    out = x.new_zeros(*x.shape[:-1], cols)
    out[..., : x.shape[-1]] = x
    return out


def fused_experts_gguf(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, 2I, row_bytes(gate_up_type)]
    down_q: torch.Tensor,  # [num_slots, H, row_bytes(down_type)]
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    gate_up_type: int,
    down_type: int,
) -> torch.Tensor:
    """Routed expert compute over native GGUF banks, one quant type per bank.

    ``gate_up_type``/``down_type`` are the ggml types the banks were packed with (a K-quant
    file mixes them: Q4_K attention, Q6_K v_proj and down experts), so they are per layer.
    """
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    act_fn = _ACT.get(activation)
    if act_fn is None:
        raise ValueError(f"unsupported MoE activation {activation!r}")

    num_tokens = hidden_states.shape[0]
    n2 = gate_up_q.shape[1]  # 2 * intermediate
    h = down_q.shape[1]  # hidden
    top_k = topk_ids.shape[1]

    # gate_up: [num_tokens*top_k, 2I] -> activation -> [num_tokens*top_k, I]
    gate_up = ggml_moe_a8_vec(
        _pad_cols(hidden_states, row_numel(gate_up_q.shape[-1], gate_up_type)),
        gate_up_q,
        topk_ids,
        top_k,
        int(gate_up_type),
        n2,
        num_tokens,
    )
    inter = act_fn(gate_up)
    # down: each of the num_tokens*top_k intermediate rows uses its own expert id.
    out = ggml_moe_a8_vec(
        _pad_cols(inter, row_numel(down_q.shape[-1], down_type)),
        down_q,
        topk_ids,
        1,
        int(down_type),
        h,
        num_tokens * top_k,
    )
    out = out.reshape(num_tokens, top_k, h) * topk_weights.reshape(num_tokens, top_k, 1).to(
        out.dtype
    )
    return out.sum(dim=1)


__all__ = ["fused_experts_gguf"]
