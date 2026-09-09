"""MXFP4 experts: the standard OCP form (DeepSeek-V4) and the gpt-oss variant with biases."""

from __future__ import annotations

import torch

from ..registry import LayerKind, register_method
from ..scheme import MX_GROUP as GROUP, QuantKind
from .base import BankSpec, ExpertView, fused_piece, is_resident, limit_or_inf, MoEConfig, MoEKernel, MoEMethod

E8M0 = torch.float8_e8m0fnu


class TritonMxfp4MoEKernel(MoEKernel):
    """Standard OCP MXFP4 experts (DeepSeek-V4 ds_fp4): e2m1 pairs + e8m0 scales, no bias."""

    name = "triton"
    cpu_format = "ds_fp4"

    def unusable_reason(self, cfg: MoEConfig) -> str | None:
        if cfg.interleaved:
            return "standard MXFP4 kernel reads the concatenated gate|up row order"
        if (cfg.alpha, cfg.beta) != (1.0, 0.0):
            return "standard MXFP4 kernel has no alpha / beta in its swiglu"
        return self._common_reject(cfg, resident_ok=False, tp_ok=False, cpu_ok=True, plain_silu_only=False)

    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]:
        i, h = cfg.intermediate, cfg.hidden
        return {
            "gate_up": BankSpec((2 * i, h // 2), torch.uint8),
            "gate_up_scale": BankSpec((2 * i, h // GROUP), E8M0),
            "down": BankSpec((h, i // 2), torch.uint8),
            "down_scale": BankSpec((h, i // GROUP), E8M0),
        }

    def pack(self, pieces, cfg: MoEConfig, out):
        out["gate_up"].copy_(fused_piece(pieces, "gate_up").view(torch.uint8))
        out["gate_up_scale"].copy_(fused_piece(pieces, "gate_up_scale").view(E8M0))
        out["down"].copy_(pieces["down"].view(torch.uint8))
        out["down_scale"].copy_(pieces["down_scale"].view(E8M0))
        return {}

    def apply(self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool):
        t = view.tensors
        banks = (t["gate_up"], t["gate_up_scale"], t["down"], t["down_scale"])
        limit = limit_or_inf(layer)
        if is_prefill and view.n is not None:
            from freetoken.moe.fused_ds_fp4 import routed_experts_fp4_prefill

            return routed_experts_fp4_prefill(x, topk_ids, topk_weights, *banks, limit, view.n)
        from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

        return routed_experts_fp4(x, topk_ids, topk_weights, *banks, limit)


class TritonGptossMxfp4MoEKernel(MoEKernel):
    """gpt-oss MXFP4: bias, alpha / limit swiglu, interleaved gate|up rows, transposed split-K banks."""

    name = "triton_gptoss"
    cpu_format = "mxfp4_triton"

    def unusable_reason(self, cfg: MoEConfig) -> str | None:
        if not cfg.has_bias:
            return "gpt-oss kernel needs the expert biases"
        return None

    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]:
        i, h = cfg.local_intermediate, cfg.hidden
        return {
            "gate_up": BankSpec((h // 2, 2 * i), torch.uint8),
            "gate_up_scale": BankSpec((h // GROUP, 2 * i), torch.uint8),
            "gate_up_bias": BankSpec((2 * i,), torch.bfloat16),
            "down": BankSpec((i // 2, h), torch.uint8),
            "down_scale": BankSpec((i // GROUP, h), torch.uint8),
            "down_bias": BankSpec((h,), torch.bfloat16),
        }

    def pack(self, pieces, cfg: MoEConfig, out):
        # HF [e, N, K//32, 16] blocks / [e, N, K//32] scales -> split-K layout with N innermost
        def blocks_t(t):
            e, n = t.shape[0], t.shape[1]
            return t.reshape(e, n, -1).permute(0, 2, 1)

        out["gate_up"].copy_(blocks_t(fused_piece(pieces, "gate_up")))
        out["gate_up_scale"].copy_(fused_piece(pieces, "gate_up_scale").permute(0, 2, 1))
        out["gate_up_bias"].copy_(fused_piece(pieces, "gate_up_bias"))
        out["down"].copy_(blocks_t(pieces["down"]))
        out["down_scale"].copy_(pieces["down_scale"].permute(0, 2, 1))
        out["down_bias"].copy_(pieces["down_bias"])
        return {}

    def apply(self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool):
        from freetoken.moe.fused_mxfp4 import MXFP4_DECODE_MAX_TOKENS, run_mxfp4_prefill_experts_t, run_mxfp4_splitk_decode_experts

        t = view.tensors
        # the resident layer picks the split-K decode kernel by token count, not by phase
        decode = x.shape[0] <= MXFP4_DECODE_MAX_TOKENS if is_resident(layer) else not is_prefill
        run = run_mxfp4_splitk_decode_experts if decode else run_mxfp4_prefill_experts_t
        return run(
            x, topk_weights, topk_ids,
            t["gate_up"], t["gate_up_scale"], t["gate_up_bias"], t["down"], t["down_scale"], t["down_bias"],
            top_k=layer.top_k, hidden_act_alpha=float(layer.alpha), swiglu_limit=layer.limit,
        )


@register_method(QuantKind.MXFP4, LayerKind.MOE)
class Mxfp4MoEMethod(MoEMethod):
    candidates = (TritonMxfp4MoEKernel, TritonGptossMxfp4MoEKernel)

    def create_weights(self, layer) -> None:
        if not self.cfg.has_bias:
            raise NotImplementedError("standard MXFP4 experts are served from the offload cache, not resident")
        g = self.cfg
        e, i, h = g.num_experts, g.local_intermediate, g.hidden
        if h % GROUP:
            raise ValueError(f"MXFP4 hidden size must be divisible by {GROUP}")
        layer.gate_up_proj_blocks = torch.empty(e, 2 * i, h // GROUP, 16, dtype=torch.uint8)
        layer.gate_up_proj_scales = torch.empty(e, 2 * i, h // GROUP, dtype=torch.uint8)
        layer.gate_up_proj_bias = torch.empty(e, 2 * i, dtype=torch.bfloat16)
        layer.down_proj_blocks = torch.empty(e, h, i // GROUP, 16, dtype=torch.uint8)
        layer.down_proj_scales = torch.empty(e, h, i // GROUP, dtype=torch.uint8)
        layer.down_proj_bias = torch.empty(e, h, dtype=torch.bfloat16)

    def finalize(self, layer) -> None:
        if getattr(layer, "gate_up_proj_blocks", None) is None:
            return
        from freetoken.moe.fused_mxfp4 import _transpose_mxfp4_for_decode

        # one transposed copy serves prefill and decode; the HF blocks are freed so 120B fits
        layer._gu_blocks_t, layer._gu_scales_t = _transpose_mxfp4_for_decode(layer.gate_up_proj_blocks, layer.gate_up_proj_scales)
        layer._dn_blocks_t, layer._dn_scales_t = _transpose_mxfp4_for_decode(layer.down_proj_blocks, layer.down_proj_scales)
        layer.gate_up_proj_blocks = None
        layer.gate_up_proj_scales = None
        layer.down_proj_blocks = None
        layer.down_proj_scales = None
        torch.cuda.empty_cache()

    def resident_view(self, layer) -> ExpertView:
        return ExpertView({
            "gate_up": layer._gu_blocks_t, "gate_up_scale": layer._gu_scales_t, "gate_up_bias": layer.gate_up_proj_bias,
            "down": layer._dn_blocks_t, "down_scale": layer._dn_scales_t, "down_bias": layer.down_proj_bias,
        })
