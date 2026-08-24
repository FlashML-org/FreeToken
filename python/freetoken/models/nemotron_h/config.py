from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)


@dataclass(frozen=True)
class NemotronHArgs:
    layer_types: tuple[str, ...]
    mamba_num_heads: int
    mamba_head_dim: int
    ssm_state_size: int
    n_groups: int
    conv_kernel: int
    chunk_size: int
    mamba_intermediate_size: int
    moe_latent_size: int
    shared_intermediate_size: int
    fp8_modules: frozenset[str]
    nvfp4_dense_modules: frozenset[str]

    def module_quant(self, name: str) -> str:
        if name in self.fp8_modules:
            return "fp8_pertensor"
        if name in self.nvfp4_dense_modules:
            # These rare dense FP4 matrices are dequantized by the loader.  The large
            # routed-expert matrices stay native NVFP4 in the offload banks.
            return "dequant_bf16"
        return "none"


def _quantized_modules(hf_config: Any) -> tuple[frozenset[str], frozenset[str]]:
    quant = getattr(hf_config, "quantization_config", None) or {}
    get = quant.get if isinstance(quant, dict) else lambda k, d=None: getattr(quant, k, d)
    layers = get("quantized_layers", {}) or {}
    fp8, nvfp4 = set(), set()
    for name, spec in layers.items():
        if ".experts." in name:
            continue
        algo = str((spec or {}).get("quant_algo", "")).lower()
        if algo == "fp8":
            fp8.add(name)
        elif "fp4" in algo:
            nvfp4.add(name)
    return frozenset(fp8), frozenset(nvfp4)


def parse_config(hf_config: Any) -> ModelConfig:
    layer_types = tuple(
        "mamba" if kind in ("mamba", "linear_attention") else kind
        for kind in hf_config.layers_block_type
    )
    mamba_ids = tuple(i for i, kind in enumerate(layer_types) if kind == "mamba")
    attention_ids = tuple(i for i, kind in enumerate(layer_types) if kind == "attention")
    moe_ids = tuple(i for i, kind in enumerate(layer_types) if kind == "moe")
    fp8_modules, nvfp4_dense_modules = _quantized_modules(hf_config)

    head_dim = int(getattr(hf_config, "head_dim", 128))
    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=head_dim,
        max_position=int(hf_config.max_position_embeddings),
        base=float(getattr(hf_config, "rope_theta", 10000.0)),
        scaling=None,
    )
    groups = (
        LinearGatedDeltaGroupConfig(
            name="mamba",
            layer_ids=mamba_ids,
            # LinearStatePool is [value_heads, key_dim, value_dim]. Map those axes to
            # Mamba's [heads, state_dim, head_dim].
            num_key_heads=int(hf_config.n_groups),
            num_value_heads=int(hf_config.mamba_num_heads),
            key_head_dim=int(hf_config.ssm_state_size),
            value_head_dim=int(hf_config.mamba_head_dim),
            conv_kernel_dim=int(hf_config.conv_kernel),
            output_gate=True,
        ),
        FullAttentionGroupConfig(
            name="full",
            layer_ids=attention_ids,
            num_kv_heads=int(hf_config.num_key_value_heads),
            head_dim=head_dim,
            rotary_config=rotary,
        ),
    )
    args = NemotronHArgs(
        layer_types=layer_types,
        mamba_num_heads=int(hf_config.mamba_num_heads),
        mamba_head_dim=int(hf_config.mamba_head_dim),
        ssm_state_size=int(hf_config.ssm_state_size),
        n_groups=int(hf_config.n_groups),
        conv_kernel=int(hf_config.conv_kernel),
        chunk_size=int(hf_config.chunk_size),
        mamba_intermediate_size=int(hf_config.mamba_num_heads * hf_config.mamba_head_dim),
        moe_latent_size=int(hf_config.moe_latent_size),
        shared_intermediate_size=int(hf_config.moe_shared_expert_intermediate_size),
        fp8_modules=fp8_modules,
        nvfp4_dense_modules=nvfp4_dense_modules,
    )
    return ModelConfig(
        num_layers=int(hf_config.num_hidden_layers),
        num_qo_heads=int(hf_config.num_attention_heads),
        num_kv_heads=int(hf_config.num_key_value_heads),
        head_dim=head_dim,
        hidden_size=int(hf_config.hidden_size),
        vocab_size=int(hf_config.vocab_size),
        intermediate_size=int(hf_config.intermediate_size),
        rms_norm_eps=float(hf_config.layer_norm_epsilon),
        rotary_config=rotary,
        hidden_act="relu2",
        tie_word_embeddings=bool(hf_config.tie_word_embeddings),
        num_experts=int(hf_config.n_routed_experts),
        num_experts_per_tok=int(hf_config.num_experts_per_tok),
        moe_intermediate_size=int(hf_config.moe_intermediate_size),
        norm_topk_prob=bool(hf_config.norm_topk_prob),
        model_type=str(hf_config.model_type),
        architectures=list(hf_config.architectures),
        moe_enabled=True,
        expert_quant="nvfp4",
        attn_quant="fp8_pertensor" if fp8_modules else "none",
        shared_expert_intermediate_size=int(hf_config.moe_shared_expert_intermediate_size),
        routed_scaling_factor=float(hf_config.routed_scaling_factor),
        n_group=int(hf_config.n_group),
        topk_group=int(hf_config.topk_group),
        attention_groups=groups,
        moe_layer_ids=moe_ids,
        expert_hidden_size=int(hf_config.moe_latent_size),
        expert_gated=False,
        nemotron_h_args=args,
        # One live Mamba state is ~160 MiB at fp32. Keep the initial implementation
        # bounded to one session; concurrency can be enabled once state paging lands.
        single_stream_only=True,
    )


__all__ = ["NemotronHArgs", "parse_config"]
