"""Reuse FreeToken packing, expert views and CPU formats; replace decode only."""

from freetoken.layers.quantization.linear.fp8_block import TritonFp8BlockLinearKernel
from freetoken.layers.quantization.moe.base import limit_or_inf
from freetoken.layers.quantization.moe.fp8_block import TritonFp8BlockMoEKernel


class ArkFp8MoEKernel(TritonFp8BlockMoEKernel):
    name = "ark_r9700"
    # FT fallbacks require the caller's safe slot-zero/zero-weight CPU routes.
    supports_inactive_slots = False

    def apply(self, layer, x, topk_weights, topk_ids, view, *, is_prefill):
        if is_prefill:
            return super().apply(layer, x, topk_weights, topk_ids, view, is_prefill=True)
        from .moe_decode_fp8.adapter import fused_experts_decode_fp8_block

        t = view.tensors
        return fused_experts_decode_fp8_block(
            x,
            t["gate_up"],
            t["gate_up_scale"],
            t["down"],
            t["down_scale"],
            topk_weights,
            topk_ids,
            layer.activation,
            layer.apply_router_weight_on_input,
            float(layer.alpha),
            limit_or_inf(layer),
        )


class ArkFp8LinearKernel(TritonFp8BlockLinearKernel):
    name = "ark_r9700"

    def apply(self, layer, x):
        from .dense_fp8_decode.adapter import block_fp8_linear

        return block_fp8_linear(x, layer.weight, layer.weight_scale_inv, layer.bias)
