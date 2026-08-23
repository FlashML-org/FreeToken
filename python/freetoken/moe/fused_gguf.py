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

# moe_vec's CUDA grid puts (tokens * top_k) rows in grid.z, which CUDA caps at
# 65535. Large prefill chunks (e.g. 16384 tokens * top_8 = 131072) exceed that,
# so calls are split into row-count-bounded pieces. The down projection already
# runs at "top_k=1, tokens=num_tokens*top_k" (one row per selected expert), so
# both calls share one chunking helper keyed off total (rows, top_k) pairs.
_MAX_GRID_Z = 65535

# Transient memory bound: each call materializes [rows_in_flight, out_rows] plus a
# q8_1 copy of its activations. On a VRAM-tight offload setup (expert cache eats
# everything the KV pool leaves) a 16k-token prefill chunk at top_8 would allocate
# ~1 GiB in one shot and fault asynchronously, so cap rows well below the grid limit.
_MAX_ROWS_IN_FLIGHT = 16384


def _moe_vec_chunked(x, weight, topk_ids, top_k, quant_type, rows, tokens, stride):
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    limit = min(_MAX_GRID_Z, _MAX_ROWS_IN_FLIGHT)
    if tokens * top_k <= limit:
        return ggml_moe_a8_vec(x, weight, topk_ids, top_k, quant_type, rows, tokens, stride)

    chunk = max(1, limit // top_k)
    outs = []
    for start in range(0, tokens, chunk):
        end = min(start + chunk, tokens)
        outs.append(
            ggml_moe_a8_vec(
                x[start:end], weight, topk_ids[start:end], top_k, quant_type, rows, end - start, stride
            )
        )
    return torch.cat(outs, dim=0)


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
    act_fn = _ACT.get(activation)
    if act_fn is None:
        raise ValueError(f"unsupported MoE activation {activation!r}")

    num_tokens = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    assert gate_up_q.dim() == 2 and down_q.dim() == 2, "gguf banks are flat padded slots"

    gate_up = _moe_vec_chunked(
        hidden_states, gate_up_q, topk_ids, top_k, int(gate_up_type),
        gate_up_rows, num_tokens, gate_up_q.shape[1],
    )
    inter = act_fn(gate_up)
    # Down pass: one selected-expert row per (token, k) -- already flat, so it's a
    # top_k=1 call over num_tokens*top_k "tokens". topk_ids must flatten the same
    # way (row-major [num_tokens, top_k] -> contiguous [num_tokens*top_k, 1]).
    flat_ids = topk_ids.reshape(-1, 1)
    out = _moe_vec_chunked(
        inter, down_q, flat_ids, 1, int(down_type),
        down_rows, num_tokens * top_k, down_q.shape[1],
    )
    out = out.reshape(num_tokens, top_k, down_rows) * topk_weights.reshape(
        num_tokens, top_k, 1
    ).to(out.dtype)
    return out.sum(dim=1)


__all__ = ["fused_experts_gguf"]
