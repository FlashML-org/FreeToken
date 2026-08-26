"""Grouped expert GEMM over mixed-type GGUF banks (borrowed ggml MoE kernels).

The generalization of :mod:`freetoken.moe.fused_q4_0` for checkpoints whose
routed-expert quant type varies per layer (Unsloth Dynamic laguna: gate/up
IQ1_S or IQ2_XXS, down IQ3_XXS or IQ4_XS). Because per-expert byte sizes then
differ across layers, the banks are FLAT padded slots -- ``[num_slots,
stride_bytes]`` uint8 with each expert's real payload in the leading bytes --
and the kernels read them via ``expert_stride_bytes``. Geometry (quant type,
output rows) rides in per-call arguments. Decode uses the low-latency MMVQ
kernel; sufficiently large prefills use the grouped MMQ kernel.
"""

from __future__ import annotations

import torch

from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
from freetoken.models.gguf.dequant import GGML_BF16

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

# MMQ has alignment/setup overhead and its donated kernel is not safe for every
# tiny routed batch.  On the RTX 2000 Ada used for the Qwen3.5 GGUF bring-up it
# is already faster at 32 input tokens (Q4_K top-8: 0.61 ms vs 0.71 ms) and the
# advantage grows to 3.1x at 2K.  Keep decode and short tails on MMVQ.
_MMQ_MIN_TOKENS = 32
# On sm_120 (RTX 5080, Ornith blk.0 banks E=256 top-8) grouped MMQ already wins
# at 16 tokens (0.314 vs 0.324 ms) and pulls away from there (0.382 vs 0.475 at
# 24), so prefill tail chunks switch over earlier.
_MMQ_MIN_TOKENS_SM120 = 16


def mmq_min_tokens(compute_capability: tuple[int, int] | None) -> int:
    """Token count from which grouped MMQ beats chunked MMVQ for routed experts."""
    if compute_capability is not None and compute_capability >= (12, 0):
        return _MMQ_MIN_TOKENS_SM120
    return _MMQ_MIN_TOKENS


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


# mm_ids_helper in the int8-MMA extension keeps one 4-byte record per token in
# shared memory; stay safely under the ~99KB sm_120 opt-in limit.
_MMA_MAX_TOKENS = 16384

# Below this the DP4A grouped MMQ wins: with E=256 top-8 the per-expert row
# count is tiny and mul_mat_q's per-expert MMA tiles waste work (sm_120,
# Ornith geometry: 256 tokens mma 1.64 vs dp4a 1.48 ms; 320 tokens 1.55 vs
# 1.76; 8192 tokens 9.1 vs 38.5 -- the full-prefill-chunk regime is the win).
_MMA_MOE_MIN_TOKENS = 320


def _use_mma_moe(quant_type, stride, x, capability) -> bool:
    """Upstream int8-MMA grouped MMQ: sm_120-gated, Q4_K/Q6_K slots only.

    The slot byte stride must be a multiple of the quant block size so the
    kernel can address experts in whole blocks (true for Ornith's banks, where
    payloads are already 64-byte aligned).
    """
    if capability is None or capability < (12, 0):
        return False
    from freetoken.kernel.gguf import mma_mmq_supported
    from freetoken.models.gguf.dequant import BLOCK_SHAPE

    if not mma_mmq_supported(int(quant_type)):
        return False
    if stride % BLOCK_SHAPE[int(quant_type)][1] != 0:
        return False
    from freetoken.layers.gguf import _mma_mmq_ok

    return _mma_mmq_ok()


def _moe_matmul(x, weight, topk_ids, top_k, quant_type, rows, tokens, stride, *, broadcast=True):
    """Choose grouped MMQ for prefill and MMVQ for decode/small tails.

    ``broadcast=True``: ``x[tokens, in]`` shared by each token's top_k experts
    (gate/up). ``broadcast=False``: ``x[tokens*top_k, in]`` with row
    ``t*top_k + k`` belonging to ``topk_ids[t][k]`` (down).

    ``ggml_moe_a8`` reads experts using ``weight.stride(0)``.  A mixed-GGUF
    bank is uint8 ``[experts, padded_slot_bytes]``, so that stride is exactly
    the byte stride expected by the donated kernel even though the real packed
    payload occupies only the beginning of each slot.
    """
    from freetoken.layers.gguf import _device_capability

    capability = _device_capability(x.device.index) if x.is_cuda else None
    if (
        _MMA_MOE_MIN_TOKENS <= tokens <= _MMA_MAX_TOKENS
        and _use_mma_moe(quant_type, stride, x, capability)
    ):
        from freetoken.kernel.gguf import ggml_moe_a8_mma

        out = ggml_moe_a8_mma(
            x, weight, topk_ids.contiguous(), top_k, int(quant_type),
            rows, tokens, stride, broadcast,
        )
        return out.to(x.dtype)
    if not broadcast:
        # The DP4A/vec kernels take per-slot rows as a flat top_k=1 call.
        topk_ids = topk_ids.reshape(-1, 1)
        tokens = tokens * top_k
        top_k = 1
    if tokens >= mmq_min_tokens(capability):
        from freetoken.kernel.gguf import ggml_moe_a8, ggml_moe_get_block_size
        from freetoken.moe.fused import moe_align_block_size

        block_size = ggml_moe_get_block_size(int(quant_type))
        if block_size:
            sorted_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                topk_ids, block_size, weight.shape[0]
            )
            return ggml_moe_a8(
                x,
                weight,
                sorted_ids,
                expert_ids,
                num_tokens_post_padded,
                int(quant_type),
                rows,
                top_k,
                tokens,
            )
    return _moe_vec_chunked(
        x, weight, topk_ids, top_k, quant_type, rows, tokens, stride
    )


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

    # Safetensors Laguna-S keeps its last expert layers in BF16.  Variable-size
    # cache rows place the real payload at the start of each padded slot, so expose
    # that prefix as ordinary dense expert tensors and reuse the native BF16 MoE.
    if gate_up_type == down_type == GGML_BF16:
        from freetoken.moe.fused import fused_experts_impl

        hidden = hidden_states.shape[-1]
        intermediate = gate_up_rows // 2
        gu_elems = gate_up_rows * hidden
        dn_elems = down_rows * intermediate
        # The leading payload is contiguous within each slot, while the slot-to-slot
        # stride includes padding for the largest layer. ``view`` preserves that outer
        # stride, giving the dense kernel an exact zero-copy 3-D view.
        gate_up = gate_up_q[:, : gu_elems * 2].view(torch.bfloat16).view(
            gate_up_q.shape[0], gate_up_rows, hidden
        )
        down = down_q[:, : dn_elems * 2].view(torch.bfloat16).view(
            down_q.shape[0], down_rows, intermediate
        )
        return fused_experts_impl(
            # fused_experts_impl writes its input in place. Laguna evaluates the
            # shared expert afterwards from the original hidden states, so preserve
            # that input just as the quantized path does.
            hidden_states.clone(),
            gate_up,
            down,
            topk_weights,
            topk_ids,
            activation,
            False,
        )
    if gate_up_type == GGML_BF16 or down_type == GGML_BF16:
        raise ValueError("mixed BF16/quantized projections within one expert layer are unsupported")

    gate_up = _moe_matmul(
        hidden_states, gate_up_q, topk_ids, top_k, int(gate_up_type),
        gate_up_rows, num_tokens, gate_up_q.shape[1],
    )
    inter = act_fn(gate_up)
    # Down pass: one selected-expert row per (token, k). _moe_matmul flattens to
    # a top_k=1 call for the DP4A/vec kernels (row-major [num_tokens, top_k] ->
    # contiguous [num_tokens*top_k, 1]); the MMA path keeps the 2-D ids.
    out = _moe_matmul(
        inter, down_q, topk_ids, top_k, int(down_type),
        down_rows, num_tokens, down_q.shape[1], broadcast=False,
    )
    out = out.reshape(num_tokens, top_k, down_rows) * topk_weights.reshape(
        num_tokens, top_k, 1
    ).to(out.dtype)
    return out.sum(dim=1)


__all__ = ["fused_experts_gguf"]
