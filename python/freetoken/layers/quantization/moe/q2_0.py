"""GSQ Q2_0 routed experts, kept in their native 64-element / 18-byte blocks."""
from __future__ import annotations

import torch

from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import BankSpec, MoEConfig, MoEKernel, MoEMethod


class TritonQ20MoEKernel(MoEKernel):
    name = "triton"

    def unusable_reason(self, cfg: MoEConfig):
        reason = self._common_reject(cfg, tp_ok=False, cpu_ok=False, plain_silu_only=True)
        if reason:
            return reason
        if cfg.interleaved or cfg.apply_router_weight_on_input:
            return "Q2_0 requires contiguous gate/up halves and output-side routing weights"
        if cfg.hidden % 64 or cfg.intermediate % 64:
            return "Q2_0 input dimensions must be divisible by 64"
        if cfg.strategy != "offload":
            return "Q2_0 currently requires pure GPU offload"
        if cfg.dtype != torch.bfloat16:
            return "Q2_0 mixed-precision path currently requires BF16 activations"
        return None

    def layout(self, cfg):
        return {
            "gate_up": BankSpec((2 * cfg.intermediate, cfg.hidden // 64 * 18), torch.uint8),
            "down": BankSpec((cfg.hidden, cfg.intermediate // 64 * 18), torch.uint8),
        }

    def pack(self, pieces, cfg, out):
        # Copy packed bytes directly; do not expand a full expert layer to BF16.
        out["gate_up"][:, :cfg.intermediate].copy_(pieces["gate"])
        out["gate_up"][:, cfg.intermediate:].copy_(pieces["up"])
        out["down"].copy_(pieces["down"])
        return {}

    def apply(self, layer, x, topk_weights, topk_ids, view, *, is_prefill):
        from freetoken.moe.fused_q2_0 import fused_experts_q2_0

        return fused_experts_q2_0(x, view.tensors["gate_up"], view.tensors["down"], topk_weights, topk_ids)


@register_method(QuantKind.Q2_0, LayerKind.MOE)
class Q20MoEMethod(MoEMethod):
    candidates = (TritonQ20MoEKernel,)
