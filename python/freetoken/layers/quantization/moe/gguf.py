"""GGUF experts: packed ggml rows in two banks (gate|up and down), multiplied by llama.cpp's mul_mat_id kernels; float experts run on the bf16 fused MoE."""

from __future__ import annotations

import math

import torch

from ..ggml_types import NVFP4, float_dtype, is_quantized, row_bytes, type_name
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
    return gate, down


def _bank_spec(ggml_type: int, rows: int, k: int) -> BankSpec:
    dtype = float_dtype(ggml_type)
    return BankSpec((rows, k), dtype) if dtype is not None else BankSpec((rows, row_bytes(ggml_type, k)), torch.uint8)


_DECODE_CONFIG = {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "num_warps": 8}


def float_mul_mat_id(w: torch.Tensor, x: torch.Tensor, ids: torch.Tensor, topk_weights: torch.Tensor, config: dict | None) -> torch.Tensor:
    """``ggml.mul_mat_id`` for float experts ``w[E, N, K]``: same ``x`` and ``ids``, same ``[T, k, N]`` output in ``x``'s dtype, no router weights.

    ``config`` runs the sorted-route kernel (prefill, resident layers); None runs the per-route decode kernel."""
    from freetoken.kernel import fused_moe_decode_kernel_triton, fused_moe_kernel_triton
    from freetoken.moe.fused import moe_align_block_size

    t, top_k = ids.shape
    per_route = x.dim() == 3
    # f16 experts take f16 activations, as llama.cpp's f16 path does
    a = x.reshape(-1, x.shape[-1]).to(w.dtype).contiguous()
    out = torch.empty(t, top_k, w.shape[1], dtype=x.dtype, device=x.device)
    if config is None:
        fused_moe_decode_kernel_triton(a, w, out, topk_weights, ids, False, per_route, _DECODE_CONFIG, compute_type=x.dtype)
    else:
        sorted_ids, expert_ids, num_post_padded = moe_align_block_size(ids, config["BLOCK_SIZE_M"], w.shape[0])
        fused_moe_kernel_triton(
            a, w, out, topk_weights, ids, sorted_ids, expert_ids, num_post_padded, False, 1 if per_route else top_k, config, compute_type=x.dtype
        )
    return out



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
        banks = {"gate_up": _bank_spec(gate_up, 2 * i, h), "down": _bank_spec(down, h, i)}
        if cfg.scheme.weight.scale is not None:
            # each expert's NVFP4 weight_scale_2 as (gate, up, down, 1): a 16-byte row, which the slot cache's copies need
            banks["global"] = BankSpec((4,), torch.float32)
        return banks

    def pack(self, pieces, cfg: MoEConfig, out):
        i = cfg.local_intermediate
        out["gate_up"][:, :i].copy_(pieces["gate"])
        out["gate_up"][:, i:].copy_(pieces["up"])
        out["down"].copy_(pieces["down"])
        return {}

    def apply(self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool):
        from freetoken.kernel import ggml
        from freetoken.layers.activation import gated_act_and_mul

        cfg = layer.quant_method.cfg
        gate_up_type, down_type = expert_types(cfg)
        if not is_quantized(gate_up_type) and not is_quantized(down_type):
            return self._apply_float(layer, x, topk_weights, topk_ids, view, gate_up_type, is_prefill=is_prefill)
        assert not layer.apply_router_weight_on_input, "GGUF experts apply the router weights on the output"
        ids = topk_ids.to(torch.int32)
        gate_up = view.tensors["gate_up"]
        # a float half runs on the bf16 fused MoE's grouped GEMM, with its config for this layer's shape
        config = None
        if not (is_quantized(gate_up_type) and is_quantized(down_type)) and (is_prefill or is_resident(layer)):
            from freetoken.moe.fused import try_get_optimal_moe_config

            e, inter_size, hidden = gate_up.shape[0], cfg.local_intermediate, cfg.hidden
            config = try_get_optimal_moe_config((e, 2 * inter_size, hidden), (e, hidden, inter_size), ids.shape[1], x.shape[0])

        def mul_mat_id(w: torch.Tensor, ggml_type: int, a: torch.Tensor) -> torch.Tensor:
            if is_quantized(ggml_type):
                return ggml.mul_mat_id(w, ggml_type, a, ids)
            return float_mul_mat_id(w, a, ids, topk_weights, config)

        # llama.cpp's build_moe_ffn: each NVFP4 GEMM result times the expert's weight_scale_2, gate and up before the glu, down before the routing weight
        scales = view.tensors.get("global")
        gate_up_scales = scales if scales is not None and gate_up_type == NVFP4 else None
        down_scales = scales if scales is not None and down_type == NVFP4 else None
        glu = fused_glu(layer)
        small = x.shape[0] <= min(ggml.mmvq_max_tokens(t) for t in (gate_up_type, down_type) if is_quantized(t))
        if glu is not None and small and is_quantized(gate_up_type):
            gu = (None, None) if gate_up_scales is None else (gate_up_scales[:, 0], gate_up_scales[:, 1])
            inter = ggml.mul_mat_id_glu(gate_up, gate_up_type, x, ids, *glu, *gu)
        else:
            h = mul_mat_id(gate_up, gate_up_type, x)
            t, k, two_i = h.shape
            if gate_up_scales is not None:
                h.view(t, k, 2, two_i // 2).mul_(gate_up_scales[ids, :2].unsqueeze(-1))
            inter = torch.empty(t, k, two_i // 2, dtype=h.dtype, device=h.device)
            gated_act_and_mul(layer.activation, h.view(-1, two_i), inter.view(-1, two_i // 2), alpha=float(layer.alpha), limit=limit_or_inf(layer))
            del h  # free the gate_up output before the down GEMM allocates its own
        if down_scales is not None and x.shape[0] <= ggml.mmvq_max_tokens(down_type):
            y = ggml.mul_mat_id(view.tensors["down"], down_type, inter, ids, scale=down_scales[:, 2])
        else:
            y = mul_mat_id(view.tensors["down"], down_type, inter)
            if down_scales is not None:
                topk_weights = topk_weights * down_scales[ids, 2]
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

    def cpu_spec(self) -> tuple[str | None, dict[str, int]]:
        gate_up, down = expert_types(self.cfg)
        # BF16 experts keep the executor's bf16 kernels, which use AVX512-BF16 where the CPU has it; ggml-cpu is built for AVX2 only
        if float_dtype(gate_up) is torch.bfloat16 and float_dtype(down) is torch.bfloat16:
            return "bf16", {}
        # an NVFP4 layer of a file with weight scales reads the layout's global bank too
        scaled = self.cfg.scheme.weight.scale is not None and NVFP4 in (gate_up, down)
        return "gguf_scaled" if scaled else "gguf", {"gate_up_type": gate_up, "down_type": down}
