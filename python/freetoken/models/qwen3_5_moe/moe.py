from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer, silu_and_mul

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class _SharedExpert(BaseOP):
    """Always-present shared SwiGLU expert of width ``shared_expert_intermediate_size``.
    The quant type follows ``dense_quant`` (NVFP4 W4A16 for modelopt/shared_expert NVFP4
    and dense Qwen3.6-27B; per-tensor FP8 W8A8 for llm-compressor mixed-precision Ornith),
    then ``attn_quant`` (per-tensor FP8 when shared_expert is in the FP8 group), then
    ``expert_quant`` (block FP8), else bf16. The dispatch lives in models/quant_linear.py
    so every dense col-merged linear (attn, GDN, shared_expert) picks the same kernel."""

    def __init__(self, config: ModelConfig, hidden_size: int, intermediate_size: int):
        from freetoken.models.quant_linear import make_col_merged_quant, make_replicated_quant

        # ``dense_quant`` wins for the NVFP4 path (shared_expert NVFP4 in modelopt mixed
        # precision and in dense Qwen3.6-27B); ``attn_quant`` for FP8 paths.
        # The factory's pre-NVFP4 check (``expert_quant == "fp8_block"``) still fires for
        # block-fp8 models; per-tensor FP8 takes the attn_quant path; everything else bf16.
        dense_quant = getattr(config, "dense_quant", "none")
        attn_quant = getattr(config, "attn_quant", "none")
        expert_quant = getattr(config, "expert_quant", "none")
        # For NVFP4 dense, the factory's check is on ``attn_quant``; promote ``dense_quant``
        # to ``attn_quant`` for the call so a single dispatch covers both. Block-fp8 and
        # per-tensor-fp8 are unchanged.
        fused_attn = "nvfp4" if dense_quant == "nvfp4" else attn_quant
        self.gate_up_proj = make_col_merged_quant(
            expert_quant, fused_attn, hidden_size,
            [intermediate_size, intermediate_size], has_bias=False,
        )
        self.down_proj = make_replicated_quant(
            expert_quant, fused_attn, intermediate_size, hidden_size, has_bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class Qwen3_5DenseMLP(_SharedExpert):
    """Dense (non-MoE) SwiGLU MLP for dense Qwen3.x checkpoints (e.g. 27B): ``gate_up_proj``
    (fused gate|up) + ``down_proj`` at full ``intermediate_size``. Same structure (and quant
    dispatch) as the shared expert -- NVFP4 (W4A16) when ``dense_quant=="nvfp4"``, else bf16 --
    so it reuses ``_SharedExpert`` directly and keeps the state-dict keys flat
    (``...layers.N.mlp.{gate_up_proj,down_proj}``)."""

    def __init__(self, config: ModelConfig):
        super().__init__(config, config.hidden_size, config.intermediate_size)


class Qwen3_5MoE(BaseOP):
    """Routed MoE (256 experts, top-8) plus a gated shared expert:

        out = routed(x) + sigmoid(shared_expert_gate(x)) * shared_expert(x)

    Router softmaxes over all experts, takes top-k, and renormalizes (HF semantics).
    """

    def __init__(self, config: ModelConfig, layer_id: int | None = None):
        weight_format = (
            "fp8_block" if getattr(config, "expert_quant", "none") == "fp8_block" else "bf16"
        )
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            weight_format=weight_format,
        )
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _SharedExpert(
            config, config.hidden_size, config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # Compute the router + shared expert BEFORE the routed experts: the fused MoE
        # kernel may write into ``hidden_states`` in place, which would corrupt the
        # shared expert's input (HF also evaluates the shared expert first).
        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states)
        shared = shared * torch.sigmoid(self.shared_expert_gate.forward(hidden_states))
        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        return (routed + shared).view(num_tokens, hidden_dim)


__all__ = ["Qwen3_5MoE", "Qwen3_5DenseMLP"]
