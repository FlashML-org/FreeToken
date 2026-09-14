from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.layers import BaseOP, LinearReplicated, MoELayer, OffloadMoELayer, make_moe_layer
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


def _route(router_logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    from freetoken.kernel import gpt_oss_fused_routing

    return gpt_oss_fused_routing(router_logits.contiguous(), top_k)


class GptOssMoELayer(MoELayer):
    """Resident experts routed by the fused softmax-over-top-k kernel."""

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor | None = None):
        assert router_logits is not None
        topk_weights, topk_ids = _route(router_logits, self.top_k)
        return self.routed_forward(hidden_states.contiguous(), topk_weights, topk_ids)


class GptOssOffloadMoELayer(OffloadMoELayer):
    """Offloaded experts routed by the fused softmax-over-top-k kernel."""

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor | None = None):
        assert router_logits is not None
        topk_weights, topk_ids = _route(router_logits, self.top_k)
        return self.routed_forward(hidden_states, topk_weights, topk_ids)


class GptOssMLP(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int | None = None, *, prefix: str = ""):
        self.router = LinearReplicated(
            config.hidden_size,
            config.num_experts,
            has_bias=config.has_router_bias,
        )
        if config.moe_weight_format != "mxfp4":
            raise ValueError(
                f"gpt-oss supports only mxfp4 expert weights, got "
                f"moe_weight_format={config.moe_weight_format!r}"
            )
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            activation="gpt_oss_swiglu",
            alpha=config.hidden_act_alpha,
            limit=config.swiglu_limit,
            interleaved=True,
            has_bias=True,
            resident_cls=GptOssMoELayer,
            offload_cls=GptOssOffloadMoELayer,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )
        self._layer_id = layer_id

    @nvtx_annotate("MoE")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.router.forward(hidden_states)
        final_hidden_states = self.experts.forward(hidden_states, router_logits)
        return final_hidden_states.view(num_tokens, hidden_dim)

    def prepare_for_runtime(self) -> None:
        if hasattr(self.experts, "prepare_for_runtime"):
            self.experts.prepare_for_runtime()


__all__ = [
    "GptOssMLP",
    "GptOssMoELayer",
    "GptOssOffloadMoELayer",
]
