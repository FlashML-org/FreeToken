"""marlin_sm75.py — Python dispatch layer for the Marlin WNA16 sm_75 extension.

Wraps ``freetoken.kernel._marlin_sm75`` (the pybind11 extension built from
csrc/marlin_wna16/) and exposes a single ``fused_experts_marlin_sm75`` function
that mirrors the signature of ``fused_experts_impl`` for INT4 (AWQ/GPTQ) and
INT8 weight formats on sm_75 (Turing) GPUs.

Key facts borrowed from vLLM-2080Ti-Definitive:
- Marlin WNA16 works on sm_75 with ``stages=2`` (vs ``stages=4`` on sm_80+).
  Turing lacks ``cp.async`` (sm_80+ async-copy instruction), so the 4-stage
  prefetch pipeline is replaced with a 2-stage synchronous shared-memory pipeline.
  The generated ``sm75_kernel_*.cu`` files use ``stages=2`` throughout.
- Only FP16 and INT8 activations with FP16 output are supported on sm_75.
  BF16 activation is not built for sm_75 (Turing has no BF16 ALUs).
  FP8 activation requires sm_89+ and is never attempted here.
- The extension is optional: if it wasn't built (``ImportError``), callers fall
  back to the Triton dequant path transparently.
"""
from __future__ import annotations

import functools

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)


@functools.cache
def _marlin_sm75_available() -> bool:
    """True if the compiled _marlin_sm75 extension exists and its symbols probe ok."""
    try:
        from freetoken.kernel import _marlin_sm75  # noqa: F401
        # Quick symbol probe — fail early rather than crashing at first decode.
        _ = _marlin_sm75.marlin_mm
        return True
    except (ImportError, AttributeError):
        return False


def is_marlin_sm75_applicable(
    device: torch.device,
    w1: torch.Tensor,
    activation: str,
) -> bool:
    """Return True when all conditions for the sm_75 Marlin path are met.

    Conditions:
    - Device is sm_75 (Turing); sm_80+ should use the standard Marlin path.
    - Weight dtype is int32 (packed INT4 — AWQ/GPTQ layout used by FreeToken).
    - Activation is silu/gelu/gelu_tanh (all map to FP16 epilogue in Marlin).
    - The compiled extension is present.
    """
    if device.type != "cuda":
        return False
    cc = torch.cuda.get_device_capability(device)
    # Only target sm_75; sm_80+ uses the existing vLLM Marlin path.
    if cc != (7, 5):
        return False
    if w1.dtype != torch.int32:
        return False
    if activation not in {"silu", "gelu", "gelu_tanh"}:
        return False
    return _marlin_sm75_available()


def fused_experts_marlin_sm75(
    hidden_states: torch.Tensor,       # [M, K] float16
    w1: torch.Tensor,                  # [E, N//8, K//16] int32  (Marlin-packed gate+up)
    w2: torch.Tensor,                  # [E, K//8, N//16] int32  (Marlin-packed down)
    w1_scales: torch.Tensor,           # [E, N//group, K//16] float16
    w2_scales: torch.Tensor,           # [E, K//group, N//16] float16
    topk_weights: torch.Tensor,        # [M, top_k] float32
    topk_ids: torch.Tensor,            # [M, top_k] int32
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """Run fused MoE via the sm_75 Marlin WNA16 (stages=2) extension.

    The weight tensors must already be in Marlin-tiled layout (produced by
    ``marlin_permute`` / ``marlin_permute_scales`` at load time). This function
    is the hot-path forward; weight repacking happens once at model load, not here.

    Returns ``[M, K]`` float16 — same shape and dtype as ``hidden_states``.
    """
    from freetoken.kernel import _marlin_sm75
    from freetoken.moe.fused import moe_align_block_size

    M, K = hidden_states.shape
    E = w1.shape[0]
    top_k = topk_ids.shape[1]

    # Flatten topk_ids for the Marlin MoE dispatch kernel.
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size=16, num_experts=E
    )

    # Marlin MoE forward: gate+up projection
    # Output: [M * top_k, 2 * intermediate_size] float16
    gate_up_out = _marlin_sm75.marlin_mm(
        hidden_states,          # A: [M, K]
        w1,                     # B: packed INT4 gate+up weights
        w1_scales,              # scales
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        top_k,
        mul_topk_weights=apply_router_weight_on_input,
    )  # → [M * top_k, 2 * intermediate_size]

    # Activation (silu/gelu splits gate_up_out in half along last dim)
    from freetoken.layers import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
    _ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}
    intermediate_size = gate_up_out.shape[-1] // 2
    intermediate = torch.empty(
        (M * top_k, intermediate_size), dtype=torch.float16, device=hidden_states.device
    )
    _ACT[activation](gate_up_out, intermediate)

    # Down projection
    # Output: [M * top_k, K] float16
    down_sorted_ids, down_expert_ids, down_num_tokens = moe_align_block_size(
        topk_ids.reshape(-1, 1), block_size=16, num_experts=E
    )
    out_flat = _marlin_sm75.marlin_mm(
        intermediate,           # A: [M * top_k, intermediate_size]
        w2,                     # B: packed INT4 down weights
        w2_scales,
        down_sorted_ids,
        down_expert_ids,
        down_num_tokens,
        topk_weights,
        1,                      # top_k=1 for down projection (already expanded)
        mul_topk_weights=not apply_router_weight_on_input,
    )  # → [M * top_k, K]

    # Weighted sum over top_k dimension → [M, K]
    out = out_flat.view(M, top_k, K)
    if not apply_router_weight_on_input:
        out = out * topk_weights.unsqueeze(-1).to(torch.float16)
    return out.sum(dim=1)


__all__ = [
    "is_marlin_sm75_applicable",
    "fused_experts_marlin_sm75",
    "_marlin_sm75_available",
]
