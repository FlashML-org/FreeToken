"""V4.1 bias-only routing and clamped SwiGLU, with native NVFP4 offload banks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.layers import OffloadMoELayer

from .layers import Linear


class Gate(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.gate_temp = args.gate_temp
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim,
                                               dtype=torch.bfloat16), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32), requires_grad=False)
        self.bias_vl = nn.Parameter(torch.empty_like(self.bias), requires_grad=False) if args.vision_enabled else None

    def forward(self, x, image_mask=None):
        scores = F.linear(x.float(), self.weight.float()) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        bias = self.bias
        if image_mask is not None and self.bias_vl is not None:
            bias = torch.where(image_mask.reshape(-1, 1), self.bias_vl, bias)
        indices = (scores + bias).topk(self.topk, dim=-1).indices
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return weights * self.route_scale, indices


def clamped_swiglu(gate, up, limit):
    gate, up = gate.float(), up.float()
    if limit > 0:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    return F.silu(gate) * up


class Expert(nn.Module):
    def __init__(self, dim, inter_dim, swiglu_limit):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)
        self.swiglu_limit = swiglu_limit

    def forward(self, x):
        return self.w2(clamped_swiglu(self.w1(x), self.w3(x), self.swiglu_limit).to(x.dtype))


class DSV41OffloadMoELayer(OffloadMoELayer):
    def __init__(self, layer_id, args, *, strategy="offload", decode_target="gpu", quant_config=None):
        if quant_config is None:
            from .config import DeepseekV41QuantConfig

            quant_config = DeepseekV41QuantConfig(args)
        super().__init__(layer_id=layer_id, num_experts=args.n_routed_experts,
                         top_k=args.n_activated_experts, hidden_size=args.dim,
                         intermediate_size=args.moe_inter_dim,
                         renormalize=args.norm_topk_prob, activation="swiglu_clamp",
                         alpha=1.0, limit=args.swiglu_limit,
                         strategy=strategy, decode_target=decode_target,
                         quant_config=quant_config, prefix=f"layers.{layer_id}.ffn.experts")


class MoE(nn.Module):
    def __init__(self, layer_id, args, *, strategy="offload", decode_target="gpu", quant_config=None):
        super().__init__()
        self.dim = args.dim
        self.gate = Gate(layer_id, args)
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, args.swiglu_limit)
        self.experts = DSV41OffloadMoELayer(layer_id, args, strategy=strategy,
                                          decode_target=decode_target, quant_config=quant_config)

    def forward(self, x, image_mask=None):
        shape = x.shape
        x = x.reshape(-1, self.dim)
        weights, indices = self.gate(x, image_mask)
        shared = self.shared_experts(x)
        routed = self.experts.routed_forward(x, weights.float().contiguous(),
                                             indices.to(torch.int32).contiguous())
        return (routed.float() + shared.float()).to(x.dtype).view(shape)


__all__ = ["Gate", "Expert", "MoE", "DSV41OffloadMoELayer"]
