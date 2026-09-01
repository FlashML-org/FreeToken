"""GLM-5.3 sparse MoE block: glm_moe_dsa's router/experts (activation resolved to
swiglu_clamp via config.swiglu_limit) with the SHARED expert also clamped."""

from __future__ import annotations

from freetoken.layers.moe import make_moe_layer
from freetoken.models.config import ModelConfig
from freetoken.models.glm_moe_dsa.moe import GlmMoeDsaSparseBlock

from .experts_resident import ResidentNvfp4Experts
from .mlp import Glm5ClampedMLP


def offload_moe_layers(config: ModelConfig):
    """MoE layers served from the offload cache (non-resident), in order. The host
    bank builder (weight.py) and the per-layer bank ids below share this list, so the
    two can never disagree."""
    res = frozenset(config.glm5_args.resident_layer_ids)
    return [
        l for l in range(config.first_k_dense_replace, config.num_layers) if l not in res
    ]


class Glm5SparseBlock(GlmMoeDsaSparseBlock):
    def _make_experts(self, config: ModelConfig, layer_id: int):
        activation = "swiglu_clamp" if getattr(config, "swiglu_limit", None) else "silu"
        if layer_id in config.glm5_args.resident_layer_ids:
            return ResidentNvfp4Experts(config, activation, layer_id=layer_id)
        # Offload bank ids are dense over the NON-resident MoE layers.
        return make_moe_layer(
            config,
            layer_id=offload_moe_layers(config).index(layer_id),
            renormalize=config.norm_topk_prob,
            activation=activation,
        )

    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__(config, layer_id)
        # Same attribute name/weights; only the activation gains the HF clamp.
        self.shared_experts = Glm5ClampedMLP(
            config.hidden_size,
            config.moe_intermediate_size * max(1, config.n_shared_experts),
            quant=config.dense_quant,
            limit=float(config.swiglu_limit or 10.0),
        )


__all__ = ["Glm5SparseBlock"]
