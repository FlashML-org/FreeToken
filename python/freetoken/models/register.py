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
    # attribute-path root -> checkpoint root, for quantization_config lookups
    checkpoint_roots: tuple[tuple[str, str], ...] = ()
    # inner attribute path -> checkpoint path (feed_forward.shared_mlp -> mlp)
    checkpoint_segments: tuple[tuple[str, str], ...] = ()
    # fused leaf -> source leaves (qkv_proj -> q_proj, k_proj, v_proj)
    packed_modules_mapping: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # checkpoint-name globs the family serves in bf16 although the quantization_config covers them
    unquantized_modules: tuple[str, ...] = ()


# Multimodal wrappers store the text tower under model.language_model.
_LANGUAGE_MODEL_ROOT = (("model", "model.language_model"),)
# Fused projections split back into their HF leaves so the quantization_config lookup sees the stored names.
_DENSE_PACKED = (
    ("qkv_proj", ("q_proj", "k_proj", "v_proj")),
    ("gate_up_proj", ("gate_proj", "up_proj")),
)
# per-expert checkpoints: probe the first expert's projections for the experts container
_EXPERTS_PACKED = (("experts", ("experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj")),)
_EXPERTS_W123_PACKED = (("experts", ("experts.0.w1", "experts.0.w2", "experts.0.w3")),)
_QWEN3_5_PACKED = _DENSE_PACKED + (
    ("in_proj_qkvz", ("in_proj_qkv", "in_proj_z")),
    ("in_proj_ba", ("in_proj_b", "in_proj_a")),
    ("in_proj", ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")),
) + _EXPERTS_PACKED
# Qwen3.8's per-layer hyper-connections fuse the down projection with the block-inject rows.
_QWEN4_EXP_PACKED = _QWEN3_5_PACKED + (
    ("input_mix_weight_down_block_inject", ("input_mix_weight_down", "block_inject_weight")),
)
# GLM-5.3-Flash's KDA fuses its six input projections.
_GLM5_NEXT_PACKED = _EXPERTS_PACKED + (
    ("in_proj", ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj")),
)
# Gemma 4 keeps HF's flat layer children (mlp / experts / router) under one feed_forward block.
_GEMMA4_SEGMENTS = (
    ("feed_forward.shared_mlp", "mlp"),
    ("feed_forward.experts", "experts"),
    ("feed_forward.router", "router"),
)
_GEMMA4_PACKED = _DENSE_PACKED + _EXPERTS_PACKED
_MINIMAX_M3_PACKED = _DENSE_PACKED + (
    ("index_qk_proj", ("index_q_proj", "index_k_proj")),
) + _EXPERTS_W123_PACKED

_MODEL_REGISTRY: dict[str, ModelSpec] = {
    "LlamaForCausalLM": ModelSpec(
        "freetoken.models.llama",
        "LlamaForCausalLM",
        packed_modules_mapping=_DENSE_PACKED,
    ),
    "Qwen2ForCausalLM": ModelSpec(
        "freetoken.models.qwen2",
        "Qwen2ForCausalLM",
        packed_modules_mapping=_DENSE_PACKED,
    ),
    "Qwen3ForCausalLM": ModelSpec(
        "freetoken.models.qwen3",
        "Qwen3ForCausalLM",
        packed_modules_mapping=_DENSE_PACKED,
    ),
    "Qwen3MoeForCausalLM": ModelSpec(
        "freetoken.models.qwen3_moe",
        "Qwen3MoeForCausalLM",
        packed_modules_mapping=_DENSE_PACKED + _EXPERTS_PACKED,
    ),
    "MiniMaxM2ForCausalLM": ModelSpec(
        "freetoken.models.minimax_m2",
        "MiniMaxM2ForCausalLM",
        packed_modules_mapping=_DENSE_PACKED + _EXPERTS_W123_PACKED,
    ),
    # MiniMax-M3 (model_type minimax_m3_vl): multimodal wrapper config (text tower in
    # text_config, weights under language_model.); served text-only. GQA + block-sparse
    # attention (lightning indexer, top-k 128-token blocks) on the trailing layers,
    # sigmoid/bias-routed NVFP4 experts + MXFP8 shared expert, swigluoai activation.
    "MiniMaxM3SparseForConditionalGeneration": ModelSpec(
        "freetoken.models.minimax_m3",
        "MiniMaxM3ForCausalLM",
        checkpoint_roots=(("model", "language_model.model"), ("lm_head", "language_model.lm_head")),
        packed_modules_mapping=_MINIMAX_M3_PACKED,
    ),
    # Text-only sibling (the text_config's own architectures entry).
    "MiniMaxM3SparseForCausalLM": ModelSpec(
        "freetoken.models.minimax_m3",
        "MiniMaxM3ForCausalLM",
        packed_modules_mapping=_MINIMAX_M3_PACKED,
    ),
    "DeepseekV4ForCausalLM": ModelSpec(
        "freetoken.models.deepseek_v4",
        "DeepseekV4ForCausalLM",
        # the checkpoint has no ``model.`` root
        checkpoint_roots=(("model.layers", "layers"), ("model.head", "head")),
        packed_modules_mapping=_EXPERTS_W123_PACKED,
        # the head, the KV compressors and the indexer's scorer ship bf16; the fp8 config has no modules_to_not_convert
        unquantized_modules=("head", "*.compressor.wkv", "*.compressor.wgate", "*.indexer.weights_proj"),
    ),
    "Qwen3_5MoeForConditionalGeneration": ModelSpec(
        "freetoken.models.qwen3_5_moe",
        "Qwen3_5MoEForCausalLM",
        checkpoint_roots=_LANGUAGE_MODEL_ROOT,
        packed_modules_mapping=_QWEN3_5_PACKED,
    ),
    # Qwen3.8-Flash-Next (model_type qwen4_exp): multimodal wrapper config (text tower in
    # text_config, weights under model.language_model.); served text-only. 36 GDN + 12 QSA
    # compressed-sparse attention layers on 4 hyper-connection residual streams, a PLE
    # n-gram embedding layer, 512 NVFP4 routed experts top-10 + a gated shared expert.
    "Qwen4ExpForConditionalGeneration": ModelSpec(
        "freetoken.models.qwen4_exp",
        "Qwen4ExpForCausalLM",
        checkpoint_roots=_LANGUAGE_MODEL_ROOT,
        packed_modules_mapping=_QWEN4_EXP_PACKED,
    ),
    # Dense Qwen3.x (no "Moe" in the arch name, num_experts==0, e.g. Qwen3.6-27B). Shares the
    # qwen3_5_moe package: the decoder routes its MLP through the dense Qwen3_5DenseMLP and the
    # loader handles the compressed-tensors NVFP4 layout.
    "Qwen3_5ForConditionalGeneration": ModelSpec(
        "freetoken.models.qwen3_5_moe",
        "Qwen3_5MoEForCausalLM",
        checkpoint_roots=_LANGUAGE_MODEL_ROOT,
        packed_modules_mapping=_QWEN3_5_PACKED,
    ),
    # Muse-Glimmer-30B (model_type muse_glimmer): multimodal wrapper config (text tower in
    # text_config, weights under model.language_model.); served text-only. Dense gated GQA
    # with a [SWA x3, full] pattern -- full layers are NoPE -- weightless qk norms, centered
    # (1+w) sandwich norms and softcapped logits; the NVFP4 release is compressed-tensors
    # W4A16 on every text Linear.
    "MuseGlimmerForConditionalGeneration": ModelSpec(
        "freetoken.models.muse_glimmer",
        "MuseGlimmerForCausalLM",
        checkpoint_roots=_LANGUAGE_MODEL_ROOT,
        packed_modules_mapping=(
            ("qkvg_proj", ("q_proj", "k_proj", "v_proj", "gate_proj")),
            ("gate_up_proj", ("gate_proj", "up_proj")),
        ),
    ),
    "MistralForCausalLM": ModelSpec(
        "freetoken.models.mistral",
        "MistralForCausalLM",
        packed_modules_mapping=_DENSE_PACKED,
    ),
    "Mistral3ForConditionalGeneration": ModelSpec(
        "freetoken.models.mistral",
        "MistralForCausalLM",
    ),
    "Gemma4ForConditionalGeneration": ModelSpec(
        "freetoken.models.gemma4",
        "Gemma4ForCausalLM",
        checkpoint_roots=_LANGUAGE_MODEL_ROOT,
        checkpoint_segments=_GEMMA4_SEGMENTS,
        packed_modules_mapping=_GEMMA4_PACKED,
    ),
    "Gemma4ForCausalLM": ModelSpec(
        "freetoken.models.gemma4",
        "Gemma4ForCausalLM",
        checkpoint_segments=_GEMMA4_SEGMENTS,
        packed_modules_mapping=_GEMMA4_PACKED,
    ),
    # Dense text tower of the gemma-4-12B "Unified"/omni model (model_type gemma4_unified_text).
    # Same decoder as gemma4; the dense feed-forward is selected via config.is_moe.
    "Gemma4UnifiedForConditionalGeneration": ModelSpec(
        "freetoken.models.gemma4",
        "Gemma4ForCausalLM",
        checkpoint_roots=_LANGUAGE_MODEL_ROOT,
        checkpoint_segments=_GEMMA4_SEGMENTS,
        packed_modules_mapping=_GEMMA4_PACKED,
    ),
    "Gemma4UnifiedForCausalLM": ModelSpec(
        "freetoken.models.gemma4",
        "Gemma4ForCausalLM",
        checkpoint_segments=_GEMMA4_SEGMENTS,
        packed_modules_mapping=_GEMMA4_PACKED,
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
        packed_modules_mapping=_DENSE_PACKED,
    ),
    "Glm4MoeForCausalLM": ModelSpec(
        "freetoken.models.glm4_moe",
        "Glm4MoeForCausalLM",
        packed_modules_mapping=_EXPERTS_PACKED,
    ),
    # GLM-5.2 (model_type glm_moe_dsa): DeepSeek-V3.2-class MLA + DSA sparse attention
    # with GLM-4-style sigmoid/noaux_tc MoE routing; NVFP4 routed experts served from
    # the offload cache.
    "GlmMoeDsaForCausalLM": ModelSpec(
        "freetoken.models.glm_moe_dsa",
        "GlmMoeDsaForCausalLM",
        packed_modules_mapping=_EXPERTS_PACKED,
    ),
    # GLM-5.3-Flash (model_type glm5_next): hybrid KDA linear attention (34/45 layers)
    # + NoPE-MLA/DSA with a kpool-compressed indexer (11/45), mHC x4 residual streams,
    # 288-expert sigmoid/noaux_tc MoE; natively-multimodal wrapper config (text tower
    # in text_config, weights under model.language_model.), served text-only.
    "Glm5NextForConditionalGeneration": ModelSpec(
        "freetoken.models.glm5_next",
        "Glm5NextForCausalLM",
        checkpoint_roots=_LANGUAGE_MODEL_ROOT,
        packed_modules_mapping=_GLM5_NEXT_PACKED,
    ),
    # Text-only sibling (the text_config's own architectures entry).
    "Glm5NextForCausalLM": ModelSpec(
        "freetoken.models.glm5_next",
        "Glm5NextForCausalLM",
        packed_modules_mapping=_GLM5_NEXT_PACKED,
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


def checkpoint_quant_config(model_path: str, hf_config: Any, spec: ModelSpec):
    """The checkpoint's QuantConfig under the family's naming, or None for GGUF, whose native-quant ops the shared parser does not model yet."""
    from freetoken.layers.quantization import NameMap, QuantConfig
    from freetoken.utils.hf import optional_hf_file

    if spec.parse_config == "parse_gguf_config":
        return None
    # NOTE: ModelOpt exports before 0.41 keep the quantization config only in hf_quant_config.json, and the weight download fetches nothing but the safetensors shards, so this sidecar is fetched on its own.
    hf_quant_config = None
    sidecar = optional_hf_file(model_path, "hf_quant_config.json")
    if sidecar is not None:
        import json

        with open(sidecar) as f:
            hf_quant_config = json.load(f)
    return QuantConfig.from_hf(
        hf_config,
        name_map=NameMap(roots=spec.checkpoint_roots, segments=spec.checkpoint_segments, packed=spec.packed_modules_mapping),
        unquantized=spec.unquantized_modules,
        hf_quant_config=hf_quant_config,
    )


def get_model_class(model_architecture: str, model_config: ModelConfig):
    spec = get_model_spec(model_architecture)
    model_cls = _load_attr(spec.module, spec.model_cls)
    return model_cls(model_config)


__all__ = ["ModelSpec", "checkpoint_quant_config", "get_model_spec", "get_model_class"]
