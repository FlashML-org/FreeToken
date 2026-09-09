"""fp8 e4m3 experts with 128x128 block scales."""

from __future__ import annotations

import torch

from ..registry import LayerKind, register_method
from ..scheme import FP8_BLOCK as BLOCK, QuantKind
from .base import BankSpec, ExpertView, fused_piece, gated_epilogue_reason, limit_or_inf, MoEConfig, MoEKernel, MoEMethod

FP8 = torch.float8_e4m3fn


def _pad_scale(rows: int, cols: int) -> int:
    from freetoken.kernel.aot_models import fp8_block_scale_pad

    return fp8_block_scale_pad(rows, cols)


class TritonFp8BlockMoEKernel(MoEKernel):
    name = "triton"

    def unusable_reason(self, cfg: MoEConfig) -> str | None:
        reason = self._common_reject(cfg, resident_ok=True, tp_ok=False, cpu_ok=False, plain_silu_only=False)
        if reason:
            return reason
        reason = gated_epilogue_reason(cfg)
        if reason:
            return f"fp8 block MoE kernel: {reason}"
        if cfg.apply_router_weight_on_input:
            return "fp8 block MoE kernel cannot apply the router weight on the input"
        return None

    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]:
        i, h, b = cfg.intermediate, cfg.hidden, BLOCK
        return {
            "gate_up": BankSpec((2 * i, h), FP8),
            "gate_up_scale": BankSpec((2 * i // b, _pad_scale(2 * i // b, h // b)), torch.bfloat16),
            "down": BankSpec((h, i), FP8),
            "down_scale": BankSpec((h // b, _pad_scale(h // b, i // b)), torch.bfloat16),
        }

    def pack(self, pieces, cfg: MoEConfig, out):
        out["gate_up"].copy_(fused_piece(pieces, "gate_up"))
        out["down"].copy_(pieces["down"])
        # the scale banks carry 16-byte padding on the last dim
        gs = fused_piece(pieces, "gate_up_scale")
        out["gate_up_scale"][:, :, : gs.shape[2]].copy_(gs)
        ds = pieces["down_scale"]
        out["down_scale"][:, :, : ds.shape[2]].copy_(ds)
        return {}

    def apply(self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool):
        from freetoken.moe.fused_fp8_block import fused_experts_decode_fp8_block, fused_experts_fp8_block

        t = view.tensors
        alpha, limit = float(layer.alpha), limit_or_inf(layer)
        if is_prefill:
            n = view.n if view.n is not None else layer.num_experts
            return fused_experts_fp8_block(x, t["gate_up"], t["gate_up_scale"], t["down"], t["down_scale"], topk_weights, topk_ids, n, layer.activation, layer.apply_router_weight_on_input, alpha, limit)
        return fused_experts_decode_fp8_block(x, t["gate_up"], t["gate_up_scale"], t["down"], t["down_scale"], topk_weights, topk_ids, layer.activation, layer.apply_router_weight_on_input, alpha, limit)


@register_method(QuantKind.FP8_BLOCK, LayerKind.MOE)
class Fp8BlockMoEMethod(MoEMethod):
    candidates = (TritonFp8BlockMoEKernel,)

    def create_weights(self, layer) -> None:
        g = self.cfg
        e, i, h, b = g.num_experts, g.intermediate, g.hidden, BLOCK
        layer.gate_up_proj = torch.empty(e, 2 * i, h, dtype=FP8)
        layer.gate_up_scale_inv = torch.empty(e, 2 * i // b, h // b, dtype=torch.bfloat16)
        layer.down_proj = torch.empty(e, h, i, dtype=FP8)
        layer.down_scale_inv = torch.empty(e, h // b, i // b, dtype=torch.bfloat16)

    def resident_view(self, layer) -> ExpertView:
        return ExpertView({
            "gate_up": layer.gate_up_proj, "gate_up_scale": layer.gate_up_scale_inv,
            "down": layer.down_proj, "down_scale": layer.down_scale_inv,
        })
