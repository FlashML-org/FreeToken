from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import ModelConfig


@dataclass(frozen=True)
class EncoderSpec:
    kind: str  # "vision" | "audio", the name --mm-disable takes
    config_key: str  # the checkpoint config section that builds the tower
    modalities: tuple[str, ...]  # inputs it serves, as implemented


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
    # "module:Class" turning the checkpoint's media into items; None: the family takes no multimodal input
    mm_processor: str | None = None
    encoders: tuple[EncoderSpec, ...] = ()


# Multimodal wrappers store the text tower under model.language_model.
_LANGUAGE_MODEL_ROOT = (("model", "model.language_model"),)
_QWEN_VL_ROOTS = _LANGUAGE_MODEL_ROOT + (("visual", "model.visual"),)
_QWEN_VL_PROCESSOR = "freetoken.mm.processors.qwen_vl:QwenVLMMProcessor"
# TODO: video once the processor samples frames; the tower already takes grid_thw with t > 1
_QWEN_VL_ENCODERS = (EncoderSpec("vision", "vision_config", ("image",)),)
# Fused projections split back into their HF leaves so the quantization_config lookup sees the stored names.
_DENSE_PACKED = (
    ("qkv_proj", ("q_proj", "k_proj", "v_proj")),
    ("gate_up_proj", ("gate_proj", "up_proj")),
)
# per-expert checkpoints: probe the first expert's projections for the experts container
_EXPERTS_PACKED = (("experts", ("experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj")),)
_EXPERTS_W123_PACKED = (("experts", ("experts.0.w1", "experts.0.w2", "experts.0.w3")),)
# the GGUF scheme reads expert types in (gate, up, down) order
_EXPERTS_W132_PACKED = (("experts", ("experts.0.w1", "experts.0.w3", "experts.0.w2")),)
# pre-stacked expert checkpoints (Qwen3-VL-MoE): the container's own two tensors are the probe
_STACKED_EXPERTS_PACKED = (("experts", ("experts.gate_up_proj", "experts.down_proj")),)
_QWEN3_5_PACKED = _DENSE_PACKED + (
    ("in_proj_qkvz", ("in_proj_qkv", "in_proj_z")),
    ("in_proj_ba", ("in_proj_b", "in_proj_a")),
    ("in_proj", ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")),
) + _EXPERTS_PACKED
# routers the family builds without a quant config (qwen3_5_moe/moe.py), so a checkpoint that quantized them is dequantized at load
_QWEN3_5_UNQUANTIZED = ("*.mlp.gate", "*.mlp.shared_expert_gate")
# Qwen3.8's per-layer hyper-connections fuse the down projection with the block-inject rows.
_QWEN4_EXP_PACKED = _QWEN3_5_PACKED + (
    ("input_mix_weight_down_block_inject", ("input_mix_weight_down", "block_inject_weight")),
)
# the llama.cpp converter splits the indexer projection into q and k
_QWEN4_EXP_GGUF_PACKED = _QWEN4_EXP_PACKED + (("index_qk_proj", ("index_q_proj", "index_k_proj")),)
# GLM-5.3-Flash's KDA fuses its six input projections.
_GLM5_NEXT_PACKED = _EXPERTS_PACKED + (
    ("in_proj", ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj")),
)
_GLM5_NEXT_PROCESSOR = "freetoken.mm.processors.glm5_next:Glm5NextMMProcessor"
_GLM5_NEXT_ENCODERS = (EncoderSpec("vision", "vision_config", ("image",)),)
# Gemma 4 keeps HF's flat layer children (mlp / experts / router) under one feed_forward block.
_GEMMA4_SEGMENTS = (
    ("feed_forward.shared_mlp", "mlp"),
    ("feed_forward.experts", "experts"),
    ("feed_forward.router", "router"),
)
_GEMMA4_PACKED = _DENSE_PACKED + _EXPERTS_PACKED
_GEMMA4_PROCESSOR = "freetoken.mm.processors.gemma4:Gemma4MMProcessor"
_GEMMA4_UNIFIED_PROCESSOR = "freetoken.mm.processors.gemma4:Gemma4UnifiedMMProcessor"
_GEMMA4_ENCODERS = (EncoderSpec("vision", "vision_config", ("image",)),)
_MUSE_GLIMMER_PROCESSOR = "freetoken.mm.processors.muse_glimmer:MuseGlimmerMMProcessor"
_MUSE_GLIMMER_ENCODERS = (EncoderSpec("vision", "vision_config", ("image",)),)
_MINIMAX_M3_PACKED = _DENSE_PACKED + (
    ("index_qk_proj", ("index_q_proj", "index_k_proj")),
) + _EXPERTS_W123_PACKED
_MINIMAX_M3_GGUF_PACKED = _DENSE_PACKED + (("index_qk_proj", ("index_q_proj", "index_k_proj")),) + _EXPERTS_W132_PACKED
_MINIMAX_M3_PROCESSOR = "freetoken.mm.processors.minimax_m3:MiniMaxM3MMProcessor"
_MINIMAX_M3_ENCODERS = (EncoderSpec("vision", "vision_config", ("image",)),)
# DeepSeek-V4's vision dims sit at the top level of the checkpoint config, next to the
# language model's rather than under a section of their own.
_DSV4_PROCESSOR = "freetoken.mm.processors.deepseek_v4:DSV4MMProcessor"
_DSV4_ENCODERS = (EncoderSpec("vision", "vision_n_layers", ("image",)),)

_MODEL_REGISTRY: dict[str, ModelSpec] = {
    "LlamaForCausalLM": ModelSpec(
        "freetoken.models.llama",
        "LlamaForCausalLM",
        packed_modules_mapping=_DENSE_PACKED,
    ),
    # llama.cpp llama GGUF (Llama, Mistral): the llama model classes with every tensor in its packed ggml type
    "LlamaGGUFForCausalLM": ModelSpec(
        "freetoken.models.llama",
        "LlamaForCausalLM",
        parse_config="parse_gguf_config",
        iter_weights="iter_gguf_weights",
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
    # llama.cpp qwen2 / qwen3 / qwen3moe GGUF: the same model classes with every tensor in its packed ggml type
    "Qwen2GGUFForCausalLM": ModelSpec(
        "freetoken.models.qwen2",
        "Qwen2ForCausalLM",
        parse_config="parse_gguf_config",
        iter_weights="iter_gguf_weights",
        packed_modules_mapping=_DENSE_PACKED,
    ),
    "Qwen3GGUFForCausalLM": ModelSpec(
        "freetoken.models.qwen3",
        "Qwen3ForCausalLM",
        parse_config="parse_gguf_config",
        iter_weights="iter_gguf_weights",
        packed_modules_mapping=_DENSE_PACKED,
    ),
    "Qwen3MoeGGUFForCausalLM": ModelSpec(
        "freetoken.models.qwen3_moe",
        "Qwen3MoeForCausalLM",
        parse_config="parse_gguf_config",
        iter_weights="iter_gguf_weights",
        packed_modules_mapping=_DENSE_PACKED + _EXPERTS_PACKED,
    ),
    # Qwen3-VL: the Qwen3 text tower under model.language_model. plus the shared Qwen VL vision tower with DeepStack; the MoE variant ships its experts pre-stacked.
    "Qwen3VLForConditionalGeneration": ModelSpec(
        "freetoken.models.qwen3_vl",
        "Qwen3VLForConditionalGeneration",
        checkpoint_roots=_QWEN_VL_ROOTS,
        mm_processor=_QWEN_VL_PROCESSOR,
        encoders=_QWEN_VL_ENCODERS,
        packed_modules_mapping=_DENSE_PACKED,
    ),
    "Qwen3VLMoeForConditionalGeneration": ModelSpec(
        "freetoken.models.qwen3_vl",
        "Qwen3VLMoeForConditionalGeneration",
        checkpoint_roots=_QWEN_VL_ROOTS,
        mm_processor=_QWEN_VL_PROCESSOR,
        encoders=_QWEN_VL_ENCODERS,
        packed_modules_mapping=_DENSE_PACKED + _STACKED_EXPERTS_PACKED,
    ),
    "MiniMaxM2ForCausalLM": ModelSpec(
        "freetoken.models.minimax_m2",
        "MiniMaxM2ForCausalLM",
        packed_modules_mapping=_DENSE_PACKED + _EXPERTS_W123_PACKED,
    ),
    # llama.cpp minimax-m2 GGUF: the same model classes with every tensor in its packed ggml type
    "MiniMaxM2GGUFForCausalLM": ModelSpec(
        "freetoken.models.minimax_m2",
        "MiniMaxM2ForCausalLM",
        parse_config="parse_gguf_config",
        iter_weights="iter_gguf_weights",
        packed_modules_mapping=_DENSE_PACKED + _EXPERTS_W132_PACKED,
    ),
    # MiniMax-M3 (model_type minimax_m3_vl): multimodal wrapper config (text tower in
    # text_config, weights under language_model.); the CLIP-style tower under vision_tower.
    # serves image input. GQA + block-sparse attention (lightning indexer, top-k 128-token
    # blocks) on the trailing layers, sigmoid/bias-routed NVFP4 experts + MXFP8 shared
    # expert, swigluoai activation.
    "MiniMaxM3SparseForConditionalGeneration": ModelSpec(
        "freetoken.models.minimax_m3",
        "MiniMaxM3ForConditionalGeneration",
        checkpoint_roots=(("model", "language_model.model"), ("lm_head", "language_model.lm_head")),
        packed_modules_mapping=_MINIMAX_M3_PACKED,
        mm_processor=_MINIMAX_M3_PROCESSOR,
        encoders=_MINIMAX_M3_ENCODERS,
    ),
    # Text-only sibling (the text_config's own architectures entry).
    "MiniMaxM3SparseForCausalLM": ModelSpec(
        "freetoken.models.minimax_m3",
        "MiniMaxM3ForCausalLM",
        packed_modules_mapping=_MINIMAX_M3_PACKED,
    ),
    # llama.cpp minimax-m3 GGUF (text only): the same model classes with every tensor in its packed ggml type
    "MiniMaxM3GGUFForCausalLM": ModelSpec(
        "freetoken.models.minimax_m3",
        "MiniMaxM3ForCausalLM",
        parse_config="parse_gguf_config",
        iter_weights="iter_gguf_weights",
        packed_modules_mapping=_MINIMAX_M3_GGUF_PACKED,
    ),
    "DeepseekV4ForCausalLM": ModelSpec(
        "freetoken.models.deepseek_v4",
        "DeepseekV4ForCausalLM",
        # the checkpoint has no ``model.`` root
        checkpoint_roots=(("model.layers", "layers"), ("model.head", "head")),
        packed_modules_mapping=_EXPERTS_W123_PACKED,
        # the head, the KV compressors, the indexer's scorer and the vision tower + aligner ship bf16;
        # the fp8 config has no modules_to_not_convert (patterns match checkpoint names, like V4.1)
        unquantized_modules=(
            "head", "*.compressor.wkv", "*.compressor.wgate", "*.indexer.weights_proj", "vision.*", "aligner.*",
        ),
        mm_processor=_DSV4_PROCESSOR,
        encoders=_DSV4_ENCODERS,
    ),
    # DeepSeek-V4.1-Flash (model_type deepseek_v41): multimodal wrapper config (text tower in
    # text_config). The checkpoint has no ``model.`` root; the compressors, the indexer's key projection /
    # scorer, the head and the vision tower + aligner ship bf16 (the fp8 config has no modules_to_not_convert).
    "DeepseekV41ForCausalLM": ModelSpec(
        "freetoken.models.deepseek_v41",
        "DeepseekV41ForCausalLM",
        mm_processor="freetoken.mm.processors.deepseek_v41:DeepseekV41MMProcessor",
        encoders=(EncoderSpec("vision", "vision_config", ("image",)),),
        checkpoint_roots=(("model.layers", "layers"), ("model.embed", "embed"), ("model.norm", "norm")),
        packed_modules_mapping=_EXPERTS_W123_PACKED,
        unquantized_modules=(
            "head", "*.compressor.wkv", "*.compressor.wgate", "*.indexer.wk", "*.indexer.weights_proj", "vision.*", "aligner.*",
        ),
    ),
    "Qwen3_5MoeForConditionalGeneration": ModelSpec(
        "freetoken.models.qwen3_5_moe",
        "Qwen3_5MoeForConditionalGeneration",
        checkpoint_roots=_QWEN_VL_ROOTS,
        mm_processor=_QWEN_VL_PROCESSOR,
        encoders=_QWEN_VL_ENCODERS,
        packed_modules_mapping=_QWEN3_5_PACKED,
        unquantized_modules=_QWEN3_5_UNQUANTIZED,
    ),
    # text-only Qwen3.5 releases: text_config is the top-level config and there is no tower
    "Qwen3_5MoeForCausalLM": ModelSpec(
        "freetoken.models.qwen3_5_moe",
        "Qwen3_5MoeForCausalLM",
        packed_modules_mapping=_QWEN3_5_PACKED,
        unquantized_modules=_QWEN3_5_UNQUANTIZED,
    ),
    "Qwen3_5ForCausalLM": ModelSpec(
        "freetoken.models.qwen3_5_moe",
        "Qwen3_5ForCausalLM",
        packed_modules_mapping=_QWEN3_5_PACKED,
        unquantized_modules=_QWEN3_5_UNQUANTIZED,
    ),
    # llama.cpp qwen35 / qwen35moe GGUF (text only): the same model classes with every tensor in its packed ggml type
    "Qwen3_5GGUFForCausalLM": ModelSpec(
        "freetoken.models.qwen3_5_moe",
        "Qwen3_5ForCausalLM",
        parse_config="parse_gguf_config",
        iter_weights="iter_gguf_weights",
        checkpoint_roots=_LANGUAGE_MODEL_ROOT,
        packed_modules_mapping=_QWEN3_5_PACKED,
        unquantized_modules=_QWEN3_5_UNQUANTIZED,
    ),
    "Qwen3_5MoeGGUFForCausalLM": ModelSpec(
        "freetoken.models.qwen3_5_moe",
        "Qwen3_5MoeForCausalLM",
        parse_config="parse_gguf_config",
        iter_weights="iter_gguf_weights",
        checkpoint_roots=_LANGUAGE_MODEL_ROOT,
        packed_modules_mapping=_QWEN3_5_PACKED,
        unquantized_modules=_QWEN3_5_UNQUANTIZED,
    ),
    # Qwen3.8-Flash-Next (model_type qwen4_exp): multimodal wrapper config (text tower in
    # text_config, weights under model.language_model.). 36 GDN + 12 QSA
    # compressed-sparse attention layers on 4 hyper-connection residual streams, a PLE
    # n-gram embedding layer, 512 NVFP4 routed experts top-10 + a gated shared expert.
    "Qwen4ExpForConditionalGeneration": ModelSpec(
        "freetoken.models.qwen4_exp",
        "Qwen4ExpForConditionalGeneration",
        checkpoint_roots=_QWEN_VL_ROOTS,
        mm_processor=_QWEN_VL_PROCESSOR,
        encoders=_QWEN_VL_ENCODERS,
        packed_modules_mapping=_QWEN4_EXP_PACKED,
    ),
    # llama.cpp qwen4exp GGUF (text only): the same model classes with every tensor in its packed ggml type
    "Qwen4ExpGGUFForCausalLM": ModelSpec(
        "freetoken.models.qwen4_exp",
        "Qwen4ExpForCausalLM",
        parse_config="parse_gguf_config",
        iter_weights="iter_gguf_weights",
        checkpoint_roots=_LANGUAGE_MODEL_ROOT,
        packed_modules_mapping=_QWEN4_EXP_GGUF_PACKED,
    ),
    # Dense Qwen3.x (no "Moe" in the arch name, num_experts==0, e.g. Qwen3.6-27B). Shares the
    # qwen3_5_moe package: the decoder routes its MLP through the dense Qwen3_5DenseMLP and the
    # loader handles the compressed-tensors NVFP4 layout.
    "Qwen3_5ForConditionalGeneration": ModelSpec(
        "freetoken.models.qwen3_5_moe",
        "Qwen3_5ForConditionalGeneration",
        checkpoint_roots=_QWEN_VL_ROOTS,
        mm_processor=_QWEN_VL_PROCESSOR,
        encoders=_QWEN_VL_ENCODERS,
        packed_modules_mapping=_QWEN3_5_PACKED,
        unquantized_modules=_QWEN3_5_UNQUANTIZED,
    ),
    # Muse-Glimmer-30B (model_type muse_glimmer): multimodal wrapper config (text tower in
    # text_config, weights under model.language_model.); the windowed ViT under
    # model.vision_tower. serves image input. Dense gated GQA with a [SWA x3, full]
    # pattern -- full layers are NoPE -- weightless qk norms, centered (1+w) sandwich norms
    # and softcapped logits; the NVFP4 release is compressed-tensors W4A16 on every text
    # Linear.
    "MuseGlimmerForConditionalGeneration": ModelSpec(
        "freetoken.models.muse_glimmer",
        "MuseGlimmerForConditionalGeneration",
        checkpoint_roots=_LANGUAGE_MODEL_ROOT,
        packed_modules_mapping=_DENSE_PACKED,
        mm_processor=_MUSE_GLIMMER_PROCESSOR,
        encoders=_MUSE_GLIMMER_ENCODERS,
    ),
    # llama.cpp muse-glimmer GGUF (text only): the same model classes with every tensor in its packed ggml type
    "MuseGlimmerGGUFForCausalLM": ModelSpec(
        "freetoken.models.muse_glimmer",
        "MuseGlimmerForCausalLM",
        parse_config="parse_gguf_config",
        iter_weights="iter_gguf_weights",
        packed_modules_mapping=_DENSE_PACKED,
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
        "Gemma4ForConditionalGeneration",
        checkpoint_roots=_LANGUAGE_MODEL_ROOT,
        checkpoint_segments=_GEMMA4_SEGMENTS,
        packed_modules_mapping=_GEMMA4_PACKED,
        mm_processor=_GEMMA4_PROCESSOR,
        encoders=_GEMMA4_ENCODERS,
    ),
    "Gemma4ForCausalLM": ModelSpec(
        "freetoken.models.gemma4",
        "Gemma4ForCausalLM",
        checkpoint_segments=_GEMMA4_SEGMENTS,
        packed_modules_mapping=_GEMMA4_PACKED,
    ),
    # the gemma-4-12B "Unified" release (model_type gemma4_unified): the gemma4 decoder with a dense feed-forward and a linear vision embedder in place of the ViT tower; audio is not wired
    "Gemma4UnifiedForConditionalGeneration": ModelSpec(
        "freetoken.models.gemma4",
        "Gemma4UnifiedForConditionalGeneration",
        checkpoint_roots=_LANGUAGE_MODEL_ROOT,
        checkpoint_segments=_GEMMA4_SEGMENTS,
        packed_modules_mapping=_GEMMA4_PACKED,
        mm_processor=_GEMMA4_UNIFIED_PROCESSOR,
        encoders=_GEMMA4_ENCODERS,
    ),
    "Gemma4UnifiedForCausalLM": ModelSpec(
        "freetoken.models.gemma4",
        "Gemma4ForCausalLM",
        checkpoint_segments=_GEMMA4_SEGMENTS,
        packed_modules_mapping=_GEMMA4_PACKED,
    ),
    # llama.cpp gemma4 GGUF (text only): the same model classes with every tensor in its packed ggml type
    "Gemma4GGUFForCausalLM": ModelSpec(
        "freetoken.models.gemma4",
        "Gemma4ForCausalLM",
        parse_config="parse_gguf_config",
        iter_weights="iter_gguf_weights",
        checkpoint_segments=_GEMMA4_SEGMENTS,
        packed_modules_mapping=_GEMMA4_PACKED,
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
    # llama.cpp glm4moe GGUF (GLM-4.5, 4.5-Air, 4.6, 4.7): the same model classes with every tensor in its packed ggml type
    "Glm4MoeGGUFForCausalLM": ModelSpec(
        "freetoken.models.glm4_moe",
        "Glm4MoeForCausalLM",
        parse_config="parse_gguf_config",
        iter_weights="iter_gguf_weights",
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
    # in text_config, weights under model.language_model.); the patch-grid vision tower
    # under model.visual. serves image input.
    "Glm5NextForConditionalGeneration": ModelSpec(
        "freetoken.models.glm5_next",
        "Glm5NextForConditionalGeneration",
        checkpoint_roots=_LANGUAGE_MODEL_ROOT,
        packed_modules_mapping=_GLM5_NEXT_PACKED,
        mm_processor=_GLM5_NEXT_PROCESSOR,
        encoders=_GLM5_NEXT_ENCODERS,
    ),
    # Text-only sibling (the text_config's own architectures entry).
    "Glm5NextForCausalLM": ModelSpec(
        "freetoken.models.glm5_next",
        "Glm5NextForCausalLM",
        packed_modules_mapping=_GLM5_NEXT_PACKED,
    ),
    # llama.cpp glm5-next GGUF (unsloth GLM-5.3-Flash-GGUF; text only): the same model
    # classes with every tensor in its packed ggml type
    "Glm5NextGGUFForCausalLM": ModelSpec(
        "freetoken.models.glm5_next",
        "Glm5NextForCausalLM",
        parse_config="parse_gguf_config",
        iter_weights="iter_gguf_weights",
        checkpoint_roots=_LANGUAGE_MODEL_ROOT,
        packed_modules_mapping=_GLM5_NEXT_PACKED,
        # the MLA kv_b ships as two per-head tensors (attn_k_b transposed + attn_v_b), quantized in
        # dynamic quants (unsloth UD); the reader dequantizes and fuses them into the dense module
        unquantized_modules=("*.self_attn.kv_b_proj",),
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
    """The checkpoint's QuantConfig under the family's naming.

    A GGUF family gets the GGUF dialect over the file's tensor table, named through the ``to_gguf`` its module exports."""
    from freetoken.layers.quantization import GGUFConfig, NameMap, QuantConfig

    name_map = NameMap(roots=spec.checkpoint_roots, segments=spec.checkpoint_segments, packed=spec.packed_modules_mapping)
    if spec.parse_config == "parse_gguf_config":
        to_gguf = _load_attr(spec.module, "to_gguf")
        from freetoken.models.gguf.reader import gguf_tensor_index

        types = {name: info.ggml_type for name, info in gguf_tensor_index(hf_config.model_path).items()}
        return GGUFConfig(types, to_gguf, name_map=name_map, unquantized=spec.unquantized_modules)
    return QuantConfig.from_hf(hf_config, name_map=name_map, unquantized=spec.unquantized_modules)


def get_model_class(model_architecture: str, model_config: ModelConfig):
    spec = get_model_spec(model_architecture)
    model_cls = _load_attr(spec.module, spec.model_cls)
    return model_cls(model_config)


__all__ = ["ModelSpec", "checkpoint_quant_config", "get_model_spec", "get_model_class"]
