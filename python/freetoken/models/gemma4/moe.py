from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.kernel.triton.gemma4_fused import gemma_dual_rmsnorm_residual_scalar
from freetoken.layers import BaseOP, GemmaRMSNorm, LinearReplicated, make_moe_layer
from freetoken.models.blocks import GatedMLP

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Gemma4Router(BaseOP):
    """Gemma 4 MoE router with per-expert output rescaling."""

    def __init__(self, config: ModelConfig):
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, with_scale=False)
        self.proj = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.scale = torch.empty(config.hidden_size)
        self.per_expert_scale = torch.empty(config.num_experts)
        self._scalar_root = config.hidden_size ** -0.5
        self._top_k = config.num_experts_per_tok
        from freetoken.kernel.triton.gemma4_fused import gemma4_fused_routing

        self.gemma4_fused_routing = gemma4_fused_routing

    def forward(self, x: torch.Tensor):
        h = self.norm.forward(x)
        h = h * self.scale * self._scalar_root
        logits = self.proj.forward(h)
        return self.gemma4_fused_routing(
            logits,
            self.per_expert_scale,
            self._top_k,
        )


class Gemma4MLP(BaseOP):
    """Gemma 4 feed-forward sandwich: shared MLP plus routed MoE branch."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self.shared_mlp = GatedMLP(config)
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            activation="gelu_tanh",
        )
        self.router = Gemma4Router(config)

        eps = config.rms_norm_eps
        hidden_size = config.hidden_size
        self.pre_feedforward_layernorm_2 = GemmaRMSNorm(hidden_size, eps=eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(hidden_size, eps=eps)
        self.post_feedforward_layernorm_1 = GemmaRMSNorm(hidden_size, eps=eps)
        self.post_feedforward_layernorm_2 = GemmaRMSNorm(hidden_size, eps=eps)
        self.layer_scalar = torch.empty(1)

    def forward(self, pre_ff: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        residual = x
        shared = self.shared_mlp.forward(pre_ff)

        topk_weights, topk_ids = self.router.forward(x)
        routed_in = self.pre_feedforward_layernorm_2.forward(x)
        # routed_forward may mutate the ids in place (offload decode slot remap);
        # keep the router's output intact.
        routed = self.experts.routed_forward(routed_in, topk_weights, topk_ids.clone())

        return gemma_dual_rmsnorm_residual_scalar(
            shared,
            self.post_feedforward_layernorm_1.weight,
            routed,
            self.post_feedforward_layernorm_2.weight,
            self.post_feedforward_layernorm.weight,
            residual,
            self.layer_scalar,
            self.post_feedforward_layernorm_1.eps,
            self.post_feedforward_layernorm_2.eps,
            self.post_feedforward_layernorm.eps,
        )


class _Nvfp4GatedMLP(BaseOP):
    """W4A16 NVFP4 gated MLP (fused gate_up + down, gelu-tanh gated). Drop-in for the bf16
    GatedMLP inside Gemma4DenseMLP on modelopt-NVFP4 dense checkpoints; same attribute names
    (gate_up_proj / down_proj) so the loader targets it identically."""

    def __init__(self, config: ModelConfig):
        from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseColMerged, Nvfp4DenseLinear
        from freetoken.layers import gelu_tanh_and_mul

        self.gate_up_proj = Nvfp4DenseColMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
        )
        self.down_proj = Nvfp4DenseLinear(
            config.intermediate_size, config.hidden_size, has_bias=False
        )
        self._act = gelu_tanh_and_mul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(self._act(self.gate_up_proj.forward(x)))


class Gemma4DenseMLP(BaseOP):
    """Gemma 4 dense feed-forward: a single gated MLP branch (no routed experts),
    RMS-normed then residual-added, scaled by the per-layer ``layer_scalar``.

    Mirrors transformers' Gemma4TextDecoderLayer with ``enable_moe_block=False``:
    ``out = (x + post_feedforward_layernorm(shared_mlp(pre_ff))) * layer_scalar``.
    Attribute names match the MoE Gemma4MLP so weight.py's rename machinery
    (``mlp.* -> feed_forward.shared_mlp.*``, bare ``layer_scalar`` /
    ``post_feedforward_layernorm.``) resolves the dense checkpoint unchanged. The gated
    MLP is bf16, or W4A16 NVFP4 when the checkpoint quantizes the dense MLP."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        intermediate_size: int | None = None,
        apply_scalar: bool = True,
    ):
        # E-series double-wide MLP (KV-shared layers): 2x intermediate. `apply_scalar=False`
        # lets the decoder run the Per-Layer-Embedding step before the per-layer scale.
        if intermediate_size is not None and intermediate_size != config.intermediate_size:
            import dataclasses

            mlp_cfg = dataclasses.replace(config, intermediate_size=intermediate_size)
        else:
            mlp_cfg = config
        self.shared_mlp = (
            _Nvfp4GatedMLP(mlp_cfg)
            if getattr(config, "dense_quant", "none") == "nvfp4"
            else GatedMLP(mlp_cfg)
        )
        self.post_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_scalar = torch.empty(1)
        self._apply_scalar = apply_scalar

    def forward(self, pre_ff: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = self.shared_mlp.forward(pre_ff)
        h = self.post_feedforward_layernorm.forward(h)
        out = x + h
        if self._apply_scalar:
            out = out * self.layer_scalar
        return out


__all__ = [
    "Gemma4MLP",
    "Gemma4DenseMLP",
    "Gemma4Router",
]
