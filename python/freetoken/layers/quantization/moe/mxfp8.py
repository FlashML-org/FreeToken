"""fp8 e4m3 experts with an e8m0 scale per 32 inputs of a row (OCP MXFP8)."""

from __future__ import annotations

import torch

from ..registry import LayerKind, register_method
from ..scheme import MX_GROUP as GROUP, QuantKind
from .base import BankSpec, ExpertView, fused_piece, gated_epilogue_reason, limit_or_inf, MoEConfig, MoEKernel, MoEMethod

FP8 = torch.float8_e4m3fn


class TritonMxfp8MoEKernel(MoEKernel):
    """W8A16: the kernels dequantize the routed rows in their K loop, activations stay bf16."""

    name = "triton"
    cpu_format = "mxfp8"
    supports_inactive_slots = True

    def unusable_reason(self, cfg: MoEConfig) -> str | None:
        reason = self._common_reject(cfg, tp_ok=False, cpu_ok=True, plain_silu_only=False)
        if reason:
            return reason
        if cfg.dtype == torch.float16:
            return "mxfp8 MoE kernel folds the e8m0 scale into the weight in the activation dtype, which overflows float16"
        if cfg.hidden % GROUP or cfg.intermediate % GROUP:
            return f"mxfp8 experts need hidden and intermediate sizes divisible by {GROUP}, got {cfg.hidden} and {cfg.intermediate}"
        reason = gated_epilogue_reason(cfg)
        if reason:
            return f"mxfp8 MoE kernel: {reason}"
        if cfg.apply_router_weight_on_input:
            return "mxfp8 MoE kernel cannot apply the router weight on the input"
        return None

    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]:
        i, h = cfg.intermediate, cfg.hidden
        return {
            "gate_up": BankSpec((2 * i, h), FP8),
            "gate_up_scale": BankSpec((2 * i, h // GROUP), torch.uint8),
            "down": BankSpec((h, i), FP8),
            "down_scale": BankSpec((h, i // GROUP), torch.uint8),
        }

    def pack(self, pieces, cfg: MoEConfig, out):
        out["gate_up"].copy_(fused_piece(pieces, "gate_up"))
        out["gate_up_scale"].copy_(fused_piece(pieces, "gate_up_scale").view(torch.uint8))
        out["down"].copy_(pieces["down"])
        out["down_scale"].copy_(pieces["down_scale"].view(torch.uint8))
        return {}

    def apply(self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool):
        from freetoken.kernel.triton.mxfp8_moe import fused_experts_decode_mxfp8, fused_experts_mxfp8

        t = view.tensors
        banks = (t["gate_up"], t["gate_up_scale"], t["down"], t["down_scale"])
        alpha, limit = float(layer.alpha), limit_or_inf(layer)
        if is_prefill:
            n = view.n if view.n is not None else layer.num_experts
            return fused_experts_mxfp8(x, *banks, topk_weights, topk_ids, n, layer.activation, alpha, limit)
        return fused_experts_decode_mxfp8(x, *banks, topk_weights, topk_ids, layer.activation, alpha, limit)


@register_method(QuantKind.MXFP8, LayerKind.MOE)
class Mxfp8MoEMethod(MoEMethod):
    candidates = (TritonMxfp8MoEKernel,)
