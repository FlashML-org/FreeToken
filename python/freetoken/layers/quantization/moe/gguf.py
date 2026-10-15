"""GGUF experts: packed ggml rows in two banks (gate|up and down), multiplied by llama.cpp's mul_mat_id kernels; float experts run on the bf16 fused MoE."""

from __future__ import annotations

import math

import torch

from ..ggml_types import float_dtype, is_quantized, row_bytes, type_name
from ..registry import LayerKind, register_method
from ..scheme import QuantKind, gguf_types
from .base import BankSpec, ExpertView, MoEConfig, MoEKernel, MoEMethod, gated_epilogue_reason, is_resident, limit_or_inf



def fused_glu(layer) -> tuple[str, float] | None:
    """The MMVQ epilogue (``ggml.mul_mat_id_glu``) that computes this layer's gated activation, or None to run it as a separate kernel."""
    act, alpha, limit = layer.activation, float(layer.alpha), limit_or_inf(layer)
    if act == "silu" and alpha == 1.0 and limit == math.inf:
        return "swiglu", 0.0
    if act == "gelu_tanh":
        return "geglu", 0.0
    if act == "swiglu_clamp" and alpha == 1.0 and limit != math.inf:
        return "swiglu_clamp", limit
    # ggml's fused swiglu_oai keeps its default alpha and limit
    if act == "swigluoai" and math.isclose(alpha, 1.702, rel_tol=1e-6) and math.isclose(limit, 7.0, rel_tol=1e-6):
        return "swiglu_oai", 0.0
    return None


def expert_types(cfg: MoEConfig) -> tuple[int, int]:
    gate, up, down = gguf_types(cfg.scheme)
    if gate != up:
        raise NotImplementedError(f"GGUF experts with gate {type_name(gate)} and up {type_name(up)} need one gate_up bank type")
    if is_quantized(gate) != is_quantized(down):
        raise NotImplementedError(f"GGUF experts with gate_up {type_name(gate)} and down {type_name(down)} mix float and quantized rows")
    return gate, down



class GgmlMoEKernel(MoEKernel):
    name = "ggml"
    cpu_format = "gguf"

    def unusable_reason(self, cfg: MoEConfig) -> str | None:
        if not is_quantized(expert_types(cfg)[0]):
            reason = gated_epilogue_reason(cfg)
            if reason:
                return f"float GGUF experts run on the bf16 fused MoE: {reason}"
        return self._common_reject(cfg, tp_ok=False, cpu_ok=True, plain_silu_only=False)

    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]:
        gate_up, down = expert_types(cfg)
        i, h = cfg.local_intermediate, cfg.hidden
        dtype = float_dtype(gate_up)
        if dtype is not None:
            return {"gate_up": BankSpec((2 * i, h), dtype), "down": BankSpec((h, i), dtype)}
        return {
            "gate_up": BankSpec((2 * i, row_bytes(gate_up, h)), torch.uint8),
            "down": BankSpec((h, row_bytes(down, i)), torch.uint8),
        }

    def pack(self, pieces, cfg: MoEConfig, out):
        i = cfg.local_intermediate
        out["gate_up"][:, :i].copy_(pieces["gate"])
        out["gate_up"][:, i:].copy_(pieces["up"])
        out["down"].copy_(pieces["down"])
        return {}

    def apply(self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool):
        from freetoken.kernel import ggml
        from freetoken.layers.activation import gated_act_and_mul

        gate_up_type, down_type = expert_types(layer.quant_method.cfg)
        if not is_quantized(gate_up_type):
            return self._apply_float(layer, x, topk_weights, topk_ids, view, gate_up_type, is_prefill=is_prefill)
        assert not layer.apply_router_weight_on_input, "GGUF experts apply the router weights on the output"
        ids = topk_ids.to(torch.int32)
        gate_up = view.tensors["gate_up"]
        glu = fused_glu(layer)
        small = x.shape[0] <= min(ggml.mmvq_max_tokens(gate_up_type), ggml.mmvq_max_tokens(down_type))
        if glu is not None and small:
            inter = ggml.mul_mat_id_glu(gate_up, gate_up_type, x, ids, *glu)
        else:
            h = ggml.mul_mat_id(gate_up, gate_up_type, x, ids)
            t, k, two_i = h.shape
            inter = torch.empty(t, k, two_i // 2, dtype=h.dtype, device=h.device)
            gated_act_and_mul(layer.activation, h.view(-1, two_i), inter.view(-1, two_i // 2), alpha=float(layer.alpha), limit=limit_or_inf(layer))
            del h  # free the gate_up output before the down GEMM allocates its own
        y = ggml.mul_mat_id(view.tensors["down"], down_type, inter, ids)
        return ggml.moe_weighted_sum(y, topk_weights)

    @staticmethod
    def _apply_float(layer, x, topk_weights, topk_ids, view: ExpertView, ggml_type: int, *, is_prefill: bool):
        """Float experts: their banks in their own dtype, through the bf16 fused MoE (the unquantized kernel's path)."""
        from freetoken.moe.fused import fused_experts_decode_impl, fused_experts_impl

        gate_up, down = view.tensors["gate_up"], view.tensors["down"]
        if gate_up.dtype != x.dtype:
            raise NotImplementedError(f"GGUF {type_name(ggml_type)} experts need {gate_up.dtype} activations, got {x.dtype}")
        impl = fused_experts_impl if is_prefill or is_resident(layer) else fused_experts_decode_impl
        return impl(x, gate_up, down, topk_weights, topk_ids, layer.activation, layer.apply_router_weight_on_input, float(layer.alpha), limit_or_inf(layer))


@register_method(QuantKind.GGUF, LayerKind.MOE)
class GGUFMoEMethod(MoEMethod):
    candidates = (GgmlMoEKernel,)

    @property
    def cpu_format(self) -> str | None:
        # the CPU executor reads Q4_0 experts in their GGUF rows and BF16 experts as its bf16 banks; other ggml types have no CPU kernel yet
        gate_up, down = expert_types(self.cfg)
        names = {type_name(gate_up), type_name(down)}
        if names == {"Q4_0"}:
            return "q4_0"
        if names == {"BF16"}:
            return "bf16"
        return self.kernel.cpu_format
