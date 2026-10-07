"""DeepSeek-V4.1 MoE block: sqrtsoftplus router with a selection-only correction bias, one shared
clamped-SwiGLU expert, and the routed MXFP4 experts on the shared offload cache.

Router (reference ``Gate``): ``scores = sqrt(softplus(x @ W^T))`` in fp32; the bias picks the experts,
the unbiased scores weight them (renormalized, ``+1e-20`` as in training), times ``route_scale``.
Image-span tokens select experts using the checkpoint's separate ``bias_vl``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
from freetoken.kernel.triton.dsv4.router import router_select
from freetoken.kernel.triton.dsv4.swiglu import fused_swiglu
from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer

from .args import DeepseekV41Args

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

TopK = Tuple[torch.Tensor, torch.Tensor]


class Router(BaseOP):
    def __init__(self, args: DeepseekV41Args):
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.route_scale
        self.weight = torch.empty(args.n_routed_experts, args.dim, dtype=torch.bfloat16)
        self.bias = torch.empty(args.n_routed_experts, dtype=torch.float32)
        self._bias_vl = None  # bound to the visual stack's per-layer routing bias after loading

    def forward(self, x: torch.Tensor, *, image_mask: torch.Tensor | None = None) -> TopK:
        if image_mask is not None:
            assert self._bias_vl is not None, "image routing bias was not loaded"
        scores = bf16_linear_fp32(x, self.weight)
        return router_select(
            scores.view(-1, scores.shape[-1]), self.bias, self.topk,
            score_func=self.score_func, normalize=self.norm_topk_prob, route_scale=self.route_scale,
            alternate_bias=self._bias_vl, alternate_mask=image_mask,
        )


class SharedExpert(BaseOP):
    """The dense clamped-SwiGLU expert every token passes through (fp8 block-32 linears)."""

    def __init__(self, args: DeepseekV41Args, *, quant_config=None, prefix: str = ""):
        self.w1 = LinearReplicated(args.dim, args.moe_inter_dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.w1")
        self.w2 = LinearReplicated(args.moe_inter_dim, args.dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.w2")
        self.w3 = LinearReplicated(args.dim, args.moe_inter_dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.w3")
        self.limit = args.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = fused_swiglu(self.w1.forward(x), self.w3.forward(x), self.limit, x.dtype)
        return self.w2.forward(h)


class MoE(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        args: DeepseekV41Args = config.dsv41_args
        self.dim = args.dim
        self.gate = Router(args)
        self.shared_experts = SharedExpert(args, quant_config=config.quant, prefix=f"{prefix}.shared_experts")
        # every layer is MoE: the offload cache indexes experts by layer id directly
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            activation="swiglu_clamp",
            renormalize=args.norm_topk_prob,
            limit=args.swiglu_limit,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )

    def forward(self, x: torch.Tensor, *, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.shape
        x = x.view(-1, self.dim)
        weights, indices = self.gate.forward(x, image_mask=image_mask)
        # the shared expert runs before the routed experts (same stream, so it does not overlap the
        # hybrid decode's CPU work, whose submit follows it -- reordering that is follow-up work)
        shared = self.shared_experts.forward(x)
        routed = self.experts.routed_forward(x, weights, indices)
        return (routed + shared).view(shape)


__all__ = ["MoE", "Router", "SharedExpert"]
