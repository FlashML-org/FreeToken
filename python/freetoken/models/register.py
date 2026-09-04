from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import ModelConfig


@dataclass(frozen=True)
class ModelSpec:
    module: str
    model_cls: str
    parse_config: str = "parse_config"
    iter_weights: str = "iter_weights"


_MODEL_REGISTRY: dict[str, ModelSpec] = {
    "LlamaForCausalLM": ModelSpec(
        "freetoken.models.llama",
        "LlamaForCausalLM",
    ),
    "Qwen2ForCausalLM": ModelSpec(
        "freetoken.models.qwen2",
        "Qwen2ForCausalLM",
    ),
    "Qwen3ForCausalLM": ModelSpec(
        "freetoken.models.qwen3",
        "Qwen3ForCausalLM",
    ),
    "Qwen3MoeForCausalLM": ModelSpec(
        "freetoken.models.qwen3_moe",
        "Qwen3MoeForCausalLM",
    ),
    "MiniMaxM2ForCausalLM": ModelSpec(
        "freetoken.models.minimax_m2",
        "MiniMaxM2ForCausalLM",
    ),
    # MiniMax-M3 (model_type minimax_m3_vl): multimodal wrapper config (text tower in
    # text_config, weights under language_model.); served text-only. GQA + block-sparse
    # attention (lightning indexer, top-k 128-token blocks) on the trailing layers,
    # sigmoid/bias-routed NVFP4 experts + MXFP8 shared expert, swigluoai activation.
    "MiniMaxM3SparseForConditionalGeneration": ModelSpec(
        "freetoken.models.minimax_m3",
        "MiniMaxM3ForCausalLM",
    ),
    # Text-only sibling (the text_config's own architectures entry).
    "MiniMaxM3SparseForCausalLM": ModelSpec(
        "freetoken.models.minimax_m3",
        "MiniMaxM3ForCausalLM",
    ),
    "DeepseekV4ForCausalLM": ModelSpec(
        "freetoken.models.deepseek_v4",
        "DeepseekV4ForCausalLM",
    ),
    "Qwen3_5MoeForConditionalGeneration": ModelSpec(
        "freetoken.models.qwen3_5_moe",
        "Qwen3_5MoEForCausalLM",
    ),
    # Qwen3.8-Flash-Next (model_type qwen4_exp): multimodal wrapper config (text tower in
    # text_config, weights under model.language_model.); served text-only. 36 GDN + 12 QSA
    # compressed-sparse attention layers on 4 hyper-connection residual streams, a PLE
    # n-gram embedding layer, 512 NVFP4 routed experts top-10 + a gated shared expert.
    "Qwen4ExpForConditionalGeneration": ModelSpec(
        "freetoken.models.qwen4_exp",
        "Qwen4ExpForCausalLM",
    ),
    # Dense Qwen3.x (no "Moe" in the arch name, num_experts==0, e.g. Qwen3.6-27B). Shares the
    # qwen3_5_moe package: the decoder routes its MLP through the dense Qwen3_5DenseMLP and the
    # loader handles the compressed-tensors NVFP4 layout.
    "Qwen3_5ForConditionalGeneration": ModelSpec(
        "freetoken.models.qwen3_5_moe",
        "Qwen3_5MoEForCausalLM",
    ),
    # Muse-Glimmer-30B (model_type muse_glimmer): multimodal wrapper config (text tower in
    # text_config, weights under model.language_model.); served text-only. Dense gated GQA
    # with a [SWA x3, full] pattern -- full layers are NoPE -- weightless qk norms, centered
    # (1+w) sandwich norms and softcapped logits; the NVFP4 release is compressed-tensors
    # W4A16 on every text Linear.
    "MuseGlimmerForConditionalGeneration": ModelSpec(
        "freetoken.models.muse_glimmer",
        "MuseGlimmerForCausalLM",
    ),
    "MistralForCausalLM": ModelSpec(
        "freetoken.models.mistral",
        "MistralForCausalLM",
    ),
    "Mistral3ForConditionalGeneration": ModelSpec(
        "freetoken.models.mistral",
        "MistralForCausalLM",
    ),
    "Gemma4ForConditionalGeneration": ModelSpec(
        "freetoken.models.gemma4",
        "Gemma4ForCausalLM",
    ),
    "Gemma4ForCausalLM": ModelSpec(
        "freetoken.models.gemma4",
        "Gemma4ForCausalLM",
    ),
    # Dense text tower of the gemma-4-12B "Unified"/omni model (model_type gemma4_unified_text).
    # Same decoder as gemma4; the dense feed-forward is selected via config.is_moe.
    "Gemma4UnifiedForConditionalGeneration": ModelSpec(
        "freetoken.models.gemma4",
        "Gemma4ForCausalLM",
    ),
    "Gemma4UnifiedForCausalLM": ModelSpec(
        "freetoken.models.gemma4",
        "Gemma4ForCausalLM",
    ),
    # GGUF (native Q4_0/Q6_K) gemma4: same model classes, GGUF config + weight loaders.
    "Gemma4GGUFForCausalLM": ModelSpec(
        "freetoken.models.gemma4",
        "Gemma4ForCausalLM",
        parse_config="parse_gguf_config",
        iter_weights="iter_gguf_weights",
    ),
    "GptOssForCausalLM": ModelSpec(
        "freetoken.models.gpt_oss",
        "GptOssForCausalLM",
    ),
    "Glm4MoeForCausalLM": ModelSpec(
        "freetoken.models.glm4_moe",
        "Glm4MoeForCausalLM",
    ),
    # GLM-5.2 (model_type glm_moe_dsa): DeepSeek-V3.2-class MLA + DSA sparse attention
    # with GLM-4-style sigmoid/noaux_tc MoE routing; NVFP4 routed experts served from
    # the offload cache.
    "GlmMoeDsaForCausalLM": ModelSpec(
        "freetoken.models.glm_moe_dsa",
        "GlmMoeDsaForCausalLM",
    ),
    # GLM-5.3-Flash (model_type glm5_next): hybrid KDA linear attention (34/45 layers)
    # + NoPE-MLA/DSA with a kpool-compressed indexer (11/45), mHC x4 residual streams,
    # 288-expert sigmoid/noaux_tc MoE; natively-multimodal wrapper config (text tower
    # in text_config, weights under model.language_model.), served text-only.
    "Glm5NextForConditionalGeneration": ModelSpec(
        "freetoken.models.glm5_next",
        "Glm5NextForCausalLM",
    ),
    # Text-only sibling (the text_config's own architectures entry).
    "Glm5NextForCausalLM": ModelSpec(
        "freetoken.models.glm5_next",
        "Glm5NextForCausalLM",
    ),
}


def get_model_spec(model_architecture: str) -> ModelSpec:
    try:
        return _MODEL_REGISTRY[model_architecture]
    except KeyError as exc:
        raise ValueError(f"Model architecture {model_architecture} not supported") from exc


def _load_attr(module_path: str, attr_name: str) -> Any:
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def get_model_class(model_architecture: str, model_config: ModelConfig):
    spec = get_model_spec(model_architecture)
    model_cls = _load_attr(spec.module, spec.model_cls)
    return model_cls(model_config)


__all__ = ["ModelSpec", "get_model_spec", "get_model_class"]
