"""Expert GEMV over native GGUF gate/up + K-quant down banks.

Ports vLLM/sglang's ``_fused_moe_gguf`` MMVQ path onto FreeToken's offload-cache
interface: experts are streamed to the GPU as packed block bytes and dequantized
*inside* GGUF kernels -- no bf16 expert copy is materialized. ``gate_up`` is usually
Q4_K but can be Q8_0 on terminal full-attention layers; legacy offload ``down`` is
Q8_0, while native GPU/resident paths retain each layer's Q4_K/Q5_K/Q6_K/Q8_0 type.
Decode uses MMVQ; larger prefill uses aligned grouped MMQ where its shape contract is
proven.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
from freetoken.models.gguf.dequant import GGML_Q4_K, GGML_Q8_0, row_bytes

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}


@dataclass
class MoeDecodeWork:
    """Fixed-shape GGUF MoE ABI and reusable decode scratch.

    ``id_space`` is part of call state: ``raw`` IDs address resident expert
    banks, ``slot`` IDs address offload-cache rows. Kernels never infer one
    space from tensor provenance. Buffers grow only during warmup; graph replay
    sees stable addresses after the requested shape has been provisioned.
    """

    phase: str
    buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    id_space: str | None = None
    quant_type: int | None = None
    down_quant_type: int | None = None
    gate_expert_stride_bytes: int | None = None
    gate_row_stride_bytes: int | None = None
    down_expert_stride_bytes: int | None = None
    down_row_stride_bytes: int | None = None

    def bind(
        self,
        hidden_states: torch.Tensor,
        gate_up_q: torch.Tensor,
        down_q: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        id_space: str,
        down_quant_type: int,
        gate_quant_type: int = GGML_Q4_K,
    ) -> None:
        if self.phase not in {"moe_decode", "moe_prefill"}:
            raise ValueError(f"invalid GGUF MoE phase {self.phase!r}")
        if id_space not in {"raw", "slot"}:
            raise ValueError(f"invalid GGUF expert ID space {id_space!r}")
        if hidden_states.ndim != 2 or gate_up_q.ndim != 3 or down_q.ndim != 3:
            raise ValueError("GGUF MoE expects hidden [T,H] and 3D packed banks")
        if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
            raise ValueError("top-k IDs and weights must have equal [T,K] shape")
        if topk_ids.dtype != torch.int32:
            raise ValueError(f"GGUF MoE IDs must be int32, got {topk_ids.dtype}")
        if gate_up_q.dtype != torch.uint8 or down_q.dtype != torch.uint8:
            raise ValueError("GGUF MoE banks must use packed uint8 storage")
        devices = {
            str(t.device)
            for t in (hidden_states, gate_up_q, down_q, topk_weights, topk_ids)
        }
        if len(devices) != 1:
            raise ValueError(f"GGUF MoE tensors must share device, got {sorted(devices)}")
        if hidden_states.shape[0] != topk_ids.shape[0]:
            raise ValueError("GGUF MoE hidden/token and route shapes disagree")
        if topk_weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"unsupported top-k weight dtype {topk_weights.dtype}")
        if gate_up_q.stride(2) != 1 or down_q.stride(2) != 1:
            raise ValueError("packed GGUF bank innermost stride must be one byte")
        self.id_space = id_space
        self.quant_type = int(gate_quant_type)
        self.down_quant_type = int(down_quant_type)
        self.gate_expert_stride_bytes = int(gate_up_q.stride(0))
        self.gate_row_stride_bytes = int(gate_up_q.stride(1))
        self.down_expert_stride_bytes = int(down_q.stride(0))
        self.down_row_stride_bytes = int(down_q.stride(1))

    def reserve(self, name: str, shape: tuple[int, ...], dtype: torch.dtype, device) -> torch.Tensor:
        current = self.buffers.get(name)
        if current is None or tuple(current.shape) != tuple(shape) or current.dtype != dtype or current.device != device:
            current = torch.empty(shape, dtype=dtype, device=device)
            self.buffers[name] = current
        return current

    def tensor(self, name: str) -> torch.Tensor | None:
        return self.buffers.get(name)


def _work_buffer(work: MoeDecodeWork | dict[str, torch.Tensor] | None, name: str):
    if isinstance(work, MoeDecodeWork):
        return work.tensor(name)
    return work.get(name) if work is not None else None


def _reserve_work_buffer(
    work: MoeDecodeWork | dict[str, torch.Tensor] | None,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device,
):
    if isinstance(work, MoeDecodeWork):
        return work.reserve(name, shape, dtype, device)
    if work is None:
        return None
    value = work.get(name)
    if value is None or tuple(value.shape) != tuple(shape) or value.dtype != dtype or value.device != device:
        value = torch.empty(shape, dtype=dtype, device=device)
        work[name] = value
    return value


def _reduce_routes(
    routes: torch.Tensor,
    output: torch.Tensor | None,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weight/reduce [T,K,H] routes with one graph-safe launch when on GPU."""
    if weights is not None and weights.shape != routes.shape[:2]:
        raise ValueError("route weights must match the first two route dimensions")
    if not routes.is_cuda:
        value = routes if weights is None else routes * weights.to(routes.dtype).unsqueeze(-1)
        reduced = value.sum(dim=1)
        if output is not None:
            output.copy_(reduced)
            return output
        return reduced
    if output is None:
        output = torch.empty(
            (routes.shape[0], routes.shape[2]), dtype=routes.dtype, device=routes.device
        )
    if weights is None:
        from freetoken.kernel import moe_sum_reduce_triton

        moe_sum_reduce_triton(routes, output)
    else:
        from freetoken.kernel import moe_weighted_sum_reduce_triton

        moe_weighted_sum_reduce_triton(routes, weights, output)
    return output


def _gguf_moe_matmul(
    x: torch.Tensor,
    weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type: int,
    row: int,
    dispatch: dict | None,
    output: torch.Tensor | None = None,
    weight_stride_bytes: int | None = None,
    weight_row_stride_bytes: int | None = None,
    work: MoeDecodeWork | dict[str, torch.Tensor] | None = None,
    quant_x: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run vector MMVQ or b10434-style aligned grouped MMQ, preserving route order."""
    from freetoken.kernel.gguf import (
        ggml_moe_a8,
        ggml_moe_a8_vec,
        ggml_moe_a8_vec_strided,
        ggml_moe_get_block_size,
    )

    if work is not None and isinstance(work, MoeDecodeWork):
        if work.id_space not in {"raw", "slot"}:
            raise ValueError("MoeDecodeWork must be bound before GGUF dispatch")

    if dispatch is not None and dispatch.get("implementation") == "rdna3_mmvdq":
        from freetoken.kernel.gguf import ggml_moe_mmvdq_id

        if quant_x is not None:
            raise ValueError("rdna3_mmvdq must not receive Q8_1 scratch")
        if weight_stride_bytes is None:
            weight_stride_bytes = int(weights.stride(0))
        if weight_row_stride_bytes is None:
            weight_row_stride_bytes = int(weights.stride(1))
        direct_id_space = work.id_space if isinstance(work, MoeDecodeWork) else "slot"
        args = (
            x, weights, topk_ids, int(topk_ids.shape[1]), quant_type, row, x.shape[0],
            weight_stride_bytes, weight_row_stride_bytes, direct_id_space,
        )
        return ggml_moe_mmvdq_id(*args, output=output)

    if dispatch is not None and dispatch.get("implementation") == "rdna3_mmid":
        from freetoken.kernel.gguf import ggml_moe_mmvq_id

        if weight_stride_bytes is None:
            weight_stride_bytes = int(weights.stride(0))
        if weight_row_stride_bytes is None:
            weight_row_stride_bytes = int(weights.stride(1))
        if work is not None and isinstance(work, MoeDecodeWork):
            candidate_id_space = work.id_space
        else:
            candidate_id_space = "slot"
        args = (
            x, weights, topk_ids, int(topk_ids.shape[1]), quant_type, row, x.shape[0],
            weight_stride_bytes, weight_row_stride_bytes, candidate_id_space,
        )
        if output is not None and quant_x is not None:
            return ggml_moe_mmvq_id(*args, output=output, quant_x=quant_x)
        if output is not None or quant_x is not None:
            raise ValueError("rdna3_mmid requires output and quant_x together")
        return ggml_moe_mmvq_id(*args)

    if (
        dispatch is not None
        and dispatch.get("implementation") == "ggml_moe_a8"
        and weight_stride_bytes is not None
    ):
        # Grouped strided ABI is not available yet. Keep selection visible and
        # fall back explicitly; do not silently relabel strided MMVQ as grouped.
        dispatch["reason"] = "grouped strided ABI unavailable; vector fallback"

    # Native mixed-Q5_K/Q6_K cache rows use a Q6_K-sized expert stride.
    if weight_stride_bytes is not None:
        if weight_row_stride_bytes is None:
            weight_row_stride_bytes = int(weights.stride(1))
        args = (x, weights, topk_ids, int(topk_ids.shape[1]), quant_type, row, x.shape[0])
        if output is not None and quant_x is not None:
            from freetoken.kernel.gguf import ggml_moe_a8_vec_strided_workspace

            return ggml_moe_a8_vec_strided_workspace(
                *args, weight_stride_bytes, weight_row_stride_bytes, output, quant_x
            )
        return (
            ggml_moe_a8_vec_strided(*args, weight_stride_bytes, weight_row_stride_bytes)
            if output is None
            else ggml_moe_a8_vec_strided(
                *args, weight_stride_bytes, weight_row_stride_bytes, output
            )
        )
    if dispatch is None or dispatch.get("implementation") != "ggml_moe_a8":
        args = (x, weights, topk_ids, int(topk_ids.shape[1]), quant_type, row, x.shape[0])
        if output is not None and quant_x is not None:
            from freetoken.kernel.gguf import ggml_moe_a8_vec_workspace

            return ggml_moe_a8_vec_workspace(*args, output, quant_x)
        return ggml_moe_a8_vec(*args) if output is None else ggml_moe_a8_vec(*args, output)

    from freetoken.moe.fused import moe_align_block_size

    block_size = ggml_moe_get_block_size(quant_type)
    sorted_ids, expert_ids, tokens_post_padded = moe_align_block_size(
        topk_ids, block_size, weights.shape[0]
    )
    return ggml_moe_a8(
        x,
        weights,
        sorted_ids,
        expert_ids,
        tokens_post_padded,
        quant_type,
        row,
        int(topk_ids.shape[1]),
        x.shape[0],
    )


def _maybe_packed_strides(weight: torch.Tensor, quant_type: int, cols: int):
    """Return strides only when bank rows contain padding for ``quant_type``."""
    try:
        tight_row = row_bytes(cols, quant_type)
    except (AssertionError, ValueError):
        # Shape-only dispatch tests use tiny placeholder rows that are not
        # legal GGUF K-quant geometry; preserve compact-path behavior there.
        return None, None
    if weight.stride(1) == tight_row and weight.stride(0) == weight.shape[1] * tight_row:
        return None, None
    return int(weight.stride(0)), int(weight.stride(1))


def fused_experts_gguf(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, 2I, row_bytes(H, Q4_K)] uint8
    down_q: torch.Tensor,  # [num_slots, H, row_bytes(I, Q8_0)] uint8
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    *,
    is_prefill: bool = False,
    dispatch_metadata: dict | None = None,
    gate_quant_type: int = GGML_Q4_K,
    down_quant_type: int = GGML_Q8_0,
    gate_stride_bytes: int | None = None,
    gate_row_stride_bytes: int | None = None,
    down_stride_bytes: int | None = None,
    down_row_stride_bytes: int | None = None,
    work: MoeDecodeWork | None = None,
    id_space: str = "slot",
) -> torch.Tensor:
    act_fn = _ACT.get(activation)
    if act_fn is None:
        raise ValueError(f"unsupported MoE activation {activation!r}")

    num_tokens = hidden_states.shape[0]
    n2 = gate_up_q.shape[1]  # 2 * intermediate
    h = down_q.shape[1]  # hidden
    top_k = topk_ids.shape[1]
    gate_dispatch = (dispatch_metadata or {}).get("gate_up")
    down_dispatch = (dispatch_metadata or {}).get("down")
    # Fused gate/up is an opt-in candidate optimization.  Its BF16 activation
    # and SiLU rounding differ from proven legacy MMVQ; keep correctness path
    # on separate ID-aware MMVQ until model-level greedy parity is proven.
    fused_gate_up = (
        gate_dispatch is not None
        and gate_dispatch.get("implementation") == "rdna3_mmid"
        and activation == "silu"
        and os.environ.get("FREETOKEN_GGUF_FUSED_GATE_UP", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )

    if work is not None:
        work.bind(
            hidden_states, gate_up_q, down_q, topk_weights, topk_ids,
            id_space=id_space, down_quant_type=down_quant_type,
            gate_quant_type=gate_quant_type,
        )
        gate_out = None
        if not fused_gate_up:
            gate_out = work.reserve(
                "gate_up", (num_tokens * top_k, n2), hidden_states.dtype, hidden_states.device
            )
        inter = work.reserve(
            "inter", (num_tokens * top_k, n2 // 2), hidden_states.dtype, hidden_states.device
        )
        down_out = work.reserve(
            "down", (num_tokens * top_k, h), hidden_states.dtype, hidden_states.device
        )
        result = work.reserve(
            "output", (num_tokens, h), hidden_states.dtype, hidden_states.device
        )
        gate_quant_x = None
        if gate_dispatch is None or gate_dispatch.get("implementation") != "rdna3_mmvdq":
            gate_quant_x = work.reserve(
                "quant_x_gate",
                (num_tokens, ((hidden_states.shape[1] + 511) // 512) * 144),
                torch.int32,
                hidden_states.device,
            )
        down_quant_x = None
        if down_dispatch is None or down_dispatch.get("implementation") != "rdna3_mmvdq":
            down_quant_x = work.reserve(
                "quant_x_down",
                (num_tokens * top_k, ((n2 // 2 + 511) // 512) * 144),
                torch.int32,
                hidden_states.device,
            )
    else:
        gate_out = inter = down_out = result = gate_quant_x = down_quant_x = None

    # "moe_gate_up" / "moe_down" labels segment both vector decode and grouped prefill.
    if fused_gate_up:
        from freetoken.kernel.gguf import ggml_moe_gate_up_swiglu_id

        with torch.profiler.record_function("moe_gate_up_swiglu"):
            inter = ggml_moe_gate_up_swiglu_id(
                hidden_states,
                gate_up_q,
                topk_ids,
                top_k,
                n2 // 2,
                num_tokens,
                int(gate_up_q.stride(0)),
                int(gate_up_q.stride(1)),
                work.id_space if isinstance(work, MoeDecodeWork) else id_space,
                output=inter,
                quant_x=gate_quant_x,
            )
    else:
        with torch.profiler.record_function("moe_gate_up"):
            gate_up = _gguf_moe_matmul(
                hidden_states, gate_up_q, topk_ids, int(gate_quant_type), n2, gate_dispatch,
                output=gate_out, quant_x=gate_quant_x, work=work,
                weight_stride_bytes=gate_stride_bytes,
                weight_row_stride_bytes=gate_row_stride_bytes,
            )
        with torch.profiler.record_function("moe_activation"):
            inter = act_fn(gate_up, out=inter)
    with torch.profiler.record_function("moe_down"):
        route_ids = topk_ids.reshape(-1, 1)
        out = _gguf_moe_matmul(
            inter, down_q, route_ids, int(down_quant_type), h, down_dispatch,
            weight_stride_bytes=down_stride_bytes,
            weight_row_stride_bytes=down_row_stride_bytes,
            output=down_out,
            quant_x=down_quant_x,
            work=work,
        )
    out = out.reshape(num_tokens, top_k, h)
    return _reduce_routes(out, result, topk_weights.reshape(num_tokens, top_k))


def fused_experts_gguf_native(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,
    down_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    *,
    down_quant_type: int | None,
    gate_quant_type: int = GGML_Q4_K,
    is_prefill: bool = False,
    workspace: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Resident native expert path; no cache slot remap or Q8 conversion."""
    from freetoken.kernel.gguf import gguf_dispatch, gguf_runtime_metadata
    from freetoken.models.gguf.dequant import GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0

    if gate_quant_type not in (GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0):
        raise ValueError(f"unsupported resident GGUF gate/up type {gate_quant_type!r}")
    if down_quant_type not in (GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0):
        raise ValueError(f"unsupported resident GGUF down type {down_quant_type!r}")
    act_fn = _ACT.get(activation)
    if act_fn is None:
        raise ValueError(f"unsupported MoE activation {activation!r}")
    num_tokens = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    phase = "moe_prefill" if is_prefill else "moe_decode"
    arch = gguf_runtime_metadata().get("arch")
    gate_dispatch = gguf_dispatch(
        phase, gate_quant_type, gate_up_q.shape[1], hidden_states.shape[1], num_tokens, arch
    )
    down_dispatch = gguf_dispatch(
        phase, down_quant_type, down_q.shape[1], gate_up_q.shape[1] // 2,
        num_tokens * top_k, arch
    )
    gate_stride_bytes, gate_row_stride_bytes = _maybe_packed_strides(
        gate_up_q, gate_quant_type, hidden_states.shape[1]
    )
    down_stride_bytes, down_row_stride_bytes = _maybe_packed_strides(
        down_q, down_quant_type, gate_up_q.shape[1] // 2
    )
    if isinstance(workspace, MoeDecodeWork):
        return fused_experts_gguf(
            hidden_states,
            gate_up_q,
            down_q,
            topk_weights,
            topk_ids,
            activation,
            is_prefill=is_prefill,
            dispatch_metadata={"gate_up": gate_dispatch, "down": down_dispatch},
            gate_quant_type=gate_quant_type,
            down_quant_type=down_quant_type,
            gate_stride_bytes=gate_stride_bytes,
            gate_row_stride_bytes=gate_row_stride_bytes,
            down_stride_bytes=down_stride_bytes,
            down_row_stride_bytes=down_row_stride_bytes,
            work=workspace,
            id_space="raw",
        )
    with torch.profiler.record_function("moe_gate_up_native"):
        gate_out = None
        if workspace is not None:
            gate_out = workspace.get("gate_up")
            shape = (num_tokens * top_k, gate_up_q.shape[1])
            if gate_out is None or tuple(gate_out.shape) != shape or gate_out.dtype != hidden_states.dtype:
                gate_out = torch.empty(shape, dtype=hidden_states.dtype, device=hidden_states.device)
                workspace["gate_up"] = gate_out
        gate_up = _gguf_moe_matmul(
            hidden_states, gate_up_q, topk_ids, gate_quant_type,
            gate_up_q.shape[1], gate_dispatch, gate_out,
            weight_stride_bytes=gate_stride_bytes,
            weight_row_stride_bytes=gate_row_stride_bytes,
        )
    with torch.profiler.record_function("moe_activation_native"):
        inter = None
        if workspace is not None:
            inter = workspace.get("inter")
            shape = (num_tokens * top_k, gate_up_q.shape[1] // 2)
            if inter is None or tuple(inter.shape) != shape or inter.dtype != hidden_states.dtype:
                inter = torch.empty(shape, dtype=hidden_states.dtype, device=hidden_states.device)
                workspace["inter"] = inter
        inter = act_fn(gate_up, out=inter)
    with torch.profiler.record_function("moe_down_native"):
        down_out = None
        if workspace is not None:
            down_out = workspace.get("down")
            shape = (num_tokens * top_k, down_q.shape[1])
            if down_out is None or tuple(down_out.shape) != shape or down_out.dtype != hidden_states.dtype:
                down_out = torch.empty(shape, dtype=hidden_states.dtype, device=hidden_states.device)
                workspace["down"] = down_out
        out = _gguf_moe_matmul(
            inter, down_q, topk_ids.reshape(-1, 1), down_quant_type,
            down_q.shape[1], down_dispatch, down_out,
            weight_stride_bytes=down_stride_bytes,
            weight_row_stride_bytes=down_row_stride_bytes,
        )
    out = out.reshape(num_tokens, top_k, down_q.shape[1])
    return _reduce_routes(out, None, topk_weights.reshape(num_tokens, top_k))


__all__ = ["MoeDecodeWork", "fused_experts_gguf", "fused_experts_gguf_native"]
