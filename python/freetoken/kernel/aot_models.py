"""Explicit per-model AOT kernel-shape support table.

The prebuilt kernel cache serves a runtime JIT lookup only on a byte-identical
spec name, so the shape lists in :mod:`freetoken.kernel.aot` must cover every
(model, weight-format) pair FreeToken serves. Mirroring flashinfer's ``aot``
module, the supported set is declared explicitly here -- one entry per
checkpoint family, carrying the config.json fields its kernel shapes derive
from -- and the aot spec lists are generated as the union over entries.

Derivations, which must be kept in lockstep with the runtime call sites by hand
(a drifted derivation misses the prebuilt cache by spec name and falls back to
JIT, which needs nvcc):

- store: ``element_size = num_kv_heads * head_dim * 2`` (bf16 KV row), one per
  paged-KV attention group (kvcache/mha_pool.py, kvcache/hybrid_swa_pool.py).
  DSV4 writes its MLA latent via torch scatter and contributes nothing.
- index: ``element_size = hidden_size * 2`` (bf16 embedding row) paired with
  the runtime ``num_splits_for`` rule (layers/embedding.py -> kernel/index.py).
  DSV4 (plain nn.Embedding) and GGUF embeddings (GGUFEmbedding) bypass it.

The whole table targets the shipped serving configuration: TP=1 (TP>1 shards
kv heads, shrinking the store row) and a 2-byte compute dtype (``--dtype
float32`` doubles the KV/embedding/bf16-bank rows). Both fall back to JIT.
- fast_index_copy: per-expert row bytes of every offload bank the checkpoint's
  expert format registers (moe/offload_cache.py ``_BANK_SCHEMAS``); this is the
  per-bank fallback of ``copy_missing`` plus any caller that copies single
  banks. Geometry is TP=1 (the offload path's supported configuration).
"""

from __future__ import annotations

from dataclasses import dataclass

from .index import num_splits_for

KV_CACHE_DTYPE_BYTES = 2  # every current model allocates bf16 paged KV
EMBED_DTYPE_BYTES = 2  # embedding weights stay bf16 on the indexing() path


@dataclass(frozen=True)
class AotModel:
    """One supported checkpoint family and the config fields shapes need.

    ``kv_groups`` holds ``(num_kv_heads, head_dim)`` per paged-KV attention
    group (hybrid-SWA models have two); empty means the model bypasses the
    store kernel. ``expert_formats`` are ``_BANK_SCHEMAS`` keys this checkpoint
    is served with; empty for dense models. ``aliases`` are sibling checkpoints
    with identical shapes that this entry also covers.
    """

    name: str
    architecture: str
    hidden_size: int
    kv_groups: tuple[tuple[int, int], ...]
    top_k: int | None = None
    moe_intermediate_size: int | None = None
    expert_formats: tuple[str, ...] = ()
    embed_indexing: bool = True
    aliases: tuple[str, ...] = ()
    # Further register.py keys this entry's checkpoints load under (text-only /
    # GGUF config variants of the same model); every registry key must be
    # claimed by some entry's architecture or arch_aliases.
    arch_aliases: tuple[str, ...] = ()


def expert_bank_row_bytes(fmt: str, hidden_size: int, moe_intermediate_size: int) -> dict[str, int]:
    """Per-expert row bytes for each offload bank a format registers.

    Bank names and layouts follow moe/offload_cache.py ``_BANK_SCHEMAS`` and the
    per-format loaders cited inline; every value must stay a multiple of 16 for
    the fused multi-bank copy to engage.
    """
    H, I = hidden_size, moe_intermediate_size
    if fmt == "bf16":
        # models/loader.py stream_moe_expert_sources: gate_up [E, 2I, H], down [E, H, I], bf16
        return {"gate_up": 2 * I * H * 2, "down": H * I * 2}
    if fmt == "fp8_block":
        # qwen3_5_moe/weight.py _build_fp8_expert_banks: fp8 weights + bf16 128x128 block scales
        B = 128
        return {
            "gate_up": 2 * I * H,
            "gate_up_scale": (2 * I // B) * (H // B) * 2,
            "down": H * I,
            "down_scale": (H // B) * (I // B) * 2,
        }
    if fmt == "q4_0":
        # gemma4/gguf.py _q4_0_expert_specs: GGML Q4_0 rows, 32 elems -> 18 bytes
        return {"gate_up": 2 * I * (H // 32 * 18), "down": H * (I // 32 * 18)}
    if fmt in ("nvfp4", "nvfp4_marlin", "nvfp4_b12x"):
        # models/nvfp4_banks.py: packed e2m1 pairs + per-16 fp8-e4m3 scales + fp16
        # per-row globals; marlin/b12x repacks are byte-identical with the globals
        # folded into GPU-resident alphas (moe/nvfp4_backends.py), so no global banks.
        banks = {
            "gate_up_packed": 2 * I * (H // 2),
            "gate_up_scale": 2 * I * (H // 16),
            "down_packed": H * (I // 2),
            "down_scale": H * (I // 16),
        }
        if fmt == "nvfp4":
            banks["gate_up_global"] = 2 * I * 2
            banks["down_global"] = H * 2
        return banks
    if fmt == "mxfp4_triton":
        # gpt_oss/weight.py _empty_mxfp4_triton_banks: transposed split-K blocks/scales + bf16 bias
        return {
            "gate_up_blocks": (H // 2) * (2 * I),
            "gate_up_scales": (H // 32) * (2 * I),
            "gate_up_bias": 2 * I * 2,
            "down_blocks": (I // 2) * H,
            "down_scales": (I // 32) * H,
            "down_bias": H * 2,
        }
    if fmt == "ds_fp4":
        # deepseek_v4/weight.py load_dsfp4_expert_sources: packed e2m1 + e8m0 per-32 scales
        return {
            "gate_up_packed": 2 * I * (H // 2),
            "gate_up_scale": 2 * I * (H // 32),
            "down_packed": H * (I // 2),
            "down_scale": H * (I // 32),
        }
    raise ValueError(f"Unknown expert bank format {fmt!r}")


_NVFP4_FORMATS = ("nvfp4", "nvfp4_marlin", "nvfp4_b12x")

# Config fields come from each checkpoint's config.json (text_config for the
# multimodal wrappers); checkpoints without a local copy were recorded from
# https://huggingface.co/<name>/raw/main/config.json.
SUPPORTED_MODELS: tuple[AotModel, ...] = (
    # ---- MoE checkpoints (offload expert banks) ----
    AotModel(
        name="Qwen/Qwen3.8-Flash-Next",
        architecture="Qwen4ExpForConditionalGeneration",
        hidden_size=2560,
        kv_groups=((2, 256),),
        top_k=10,
        moe_intermediate_size=640,
        expert_formats=("nvfp4",),
        aliases=("RadixArk/Qwen3.8-Flash-Next-NVFP4",),
    ),
    AotModel(
        name="Qwen/Qwen3-30B-A3B",
        architecture="Qwen3MoeForCausalLM",
        hidden_size=2048,
        kv_groups=((4, 128),),
        top_k=8,
        moe_intermediate_size=768,
        expert_formats=("bf16",),
        aliases=("Qwen/Qwen3-30B-A3B-Thinking-2507",),
    ),
    AotModel(
        name="Qwen/Qwen3.5-35B-A3B",
        architecture="Qwen3_5MoeForConditionalGeneration",
        hidden_size=2048,
        kv_groups=((2, 256),),  # full-attention group; GDN layers hold no paged KV
        top_k=8,
        moe_intermediate_size=512,
        expert_formats=("bf16",),
    ),
    AotModel(
        name="Qwen/Qwen3.5-35B-A3B-FP8",
        architecture="Qwen3_5MoeForConditionalGeneration",
        hidden_size=2048,
        kv_groups=((2, 256),),
        top_k=8,
        moe_intermediate_size=512,
        expert_formats=("fp8_block",),
    ),
    AotModel(
        name="Qwen/Qwen3.6-35B-A3B",
        architecture="Qwen3_5MoeForConditionalGeneration",
        hidden_size=2048,
        kv_groups=((2, 256),),
        top_k=8,
        moe_intermediate_size=512,
        expert_formats=("bf16",),
    ),
    AotModel(
        name="Qwen/Qwen3.6-35B-A3B-FP8",
        architecture="Qwen3_5MoeForConditionalGeneration",
        hidden_size=2048,
        kv_groups=((2, 256),),
        top_k=8,
        moe_intermediate_size=512,
        expert_formats=("fp8_block",),
    ),
    AotModel(
        name="nvidia/Qwen3.6-35B-A3B-NVFP4",
        architecture="Qwen3_5MoeForConditionalGeneration",
        hidden_size=2048,
        kv_groups=((2, 256),),
        top_k=8,
        moe_intermediate_size=512,
        expert_formats=_NVFP4_FORMATS,
    ),
    AotModel(
        name="google/gemma-4-26B-A4B-it",
        architecture="Gemma4ForConditionalGeneration",
        hidden_size=2816,
        # full group: num_global_key_value_heads x global_head_dim; swa group:
        # num_key_value_heads x head_dim (gemma4/config.py attention_groups)
        kv_groups=((2, 512), (8, 256)),
        top_k=8,
        moe_intermediate_size=704,
        # q4_0 = the Q4_0 GGUF release (Gemma4GGUFForCausalLM); its GGUFEmbedding
        # path skips indexing(), but the bf16 safetensors path needs the variant.
        expert_formats=("bf16", "q4_0"),
        arch_aliases=("Gemma4ForCausalLM", "Gemma4GGUFForCausalLM"),
    ),
    AotModel(
        name="nvidia/Gemma-4-26B-A4B-NVFP4",
        architecture="Gemma4ForConditionalGeneration",
        hidden_size=2816,
        kv_groups=((2, 512), (8, 256)),
        top_k=8,
        moe_intermediate_size=704,
        expert_formats=_NVFP4_FORMATS,
    ),
    AotModel(
        name="openai/gpt-oss-120b",
        architecture="GptOssForCausalLM",
        hidden_size=2880,
        kv_groups=((8, 64),),  # sliding and full layers share the same geometry
        top_k=4,
        moe_intermediate_size=2880,
        expert_formats=("mxfp4_triton",),
    ),
    AotModel(
        name="openai/gpt-oss-20b",
        architecture="GptOssForCausalLM",
        hidden_size=2880,
        kv_groups=((8, 64),),
        top_k=4,
        moe_intermediate_size=2880,
        expert_formats=("mxfp4_triton",),
    ),
    AotModel(
        name="zai-org/GLM-4.7",
        architecture="Glm4MoeForCausalLM",
        hidden_size=5120,
        kv_groups=((8, 128),),
        top_k=8,
        moe_intermediate_size=1536,
        expert_formats=("bf16",),
    ),
    AotModel(
        name="nvidia/GLM-4.7-NVFP4",
        architecture="Glm4MoeForCausalLM",
        hidden_size=5120,
        kv_groups=((8, 128),),
        top_k=8,
        moe_intermediate_size=1536,
        expert_formats=_NVFP4_FORMATS,
    ),
    AotModel(
        # MLA latent-KV: the dsa backend writes the latent via torch scatter,
        # not store_cache (same as DSV4 below), so no paged-KV store groups.
        name="zai-org/GLM-5.2-NVFP4",
        architecture="GlmMoeDsaForCausalLM",
        hidden_size=6144,
        kv_groups=(),
        top_k=8,
        moe_intermediate_size=2048,
        expert_formats=_NVFP4_FORMATS,
    ),
    AotModel(
        # MiniMaxAI/MiniMax-M2.5 ships block-fp8, which has no expert-bank
        # provider for this arch on main -- the NVFP4 release is the servable
        # offload path, and both share the same attention/embedding shapes.
        name="nvidia/MiniMax-M2.5-NVFP4",
        architecture="MiniMaxM2ForCausalLM",
        hidden_size=3072,
        kv_groups=((8, 128),),
        top_k=8,
        moe_intermediate_size=1536,  # experts reuse the dense intermediate_size
        expert_formats=_NVFP4_FORMATS,
        aliases=("MiniMaxAI/MiniMax-M2.5",),
    ),
    AotModel(
        # GQA block-sparse (BSA): the BSAKVCache stores K/V through store_cache
        # (4 kv heads x 128 head_dim), the index-key slab writes via torch scatter.
        # Routed experts are NVFP4 with swigluoai, which restricts the expert GEMM
        # to the Triton kernels -- but the banks keep the native "nvfp4" layout,
        # so its bank rows are what fast_index_copy sees.
        # The upstream MiniMaxAI/MiniMax-M3 alias covers the attention/embedding
        # shapes only: its experts are MXFP8, which has no expert-bank provider on
        # main -- the NVFP4 release is the servable offload path (M2.5 precedent).
        name="nvidia/MiniMax-M3-NVFP4",
        architecture="MiniMaxM3SparseForConditionalGeneration",
        hidden_size=6144,
        kv_groups=((4, 128),),
        top_k=4,
        moe_intermediate_size=3072,
        expert_formats=("nvfp4",),
        aliases=("MiniMaxAI/MiniMax-M3",),
        arch_aliases=("MiniMaxM3SparseForCausalLM",),
    ),
    AotModel(
        name="deepseek-ai/DeepSeek-V4-Flash",
        architecture="DeepseekV4ForCausalLM",
        hidden_size=4096,
        kv_groups=(),  # MLA latent pool writes via torch scatter, not store_cache
        top_k=6,
        moe_intermediate_size=2048,
        expert_formats=("ds_fp4",),
        embed_indexing=False,  # plain nn.Embedding
    ),
    # ---- dense checkpoints (store/index only, no expert banks) ----
    AotModel(
        name="Qwen/Qwen3.6-27B",
        architecture="Qwen3_5ForConditionalGeneration",
        hidden_size=5120,
        kv_groups=((4, 256),),
        aliases=("Qwen/Qwen3.6-27B-FP8", "nvidia/Qwen3.6-27B-NVFP4"),
    ),
    AotModel(
        name="google/gemma-4-12B-it",
        architecture="Gemma4UnifiedForConditionalGeneration",
        hidden_size=3840,
        kv_groups=((1, 512), (8, 256)),
        arch_aliases=("Gemma4UnifiedForCausalLM",),
    ),
    AotModel(
        name="nvidia/Gemma-4-31B-IT-NVFP4",
        architecture="Gemma4ForConditionalGeneration",
        hidden_size=5376,
        kv_groups=((4, 512), (16, 256)),
    ),
    AotModel(
        name="meta-models/Muse-Glimmer-30B",
        architecture="MuseGlimmerForConditionalGeneration",
        hidden_size=6656,
        kv_groups=((2, 128),),  # sliding and full (NoPE) layers share the same geometry
        aliases=("RedHatAI/Muse-Glimmer-30B-NVFP4",),
    ),
    AotModel(
        name="meta-llama/Llama-3.1-8B-Instruct",
        architecture="LlamaForCausalLM",
        hidden_size=4096,
        kv_groups=((8, 128),),
    ),
    AotModel(
        name="Qwen/Qwen2-7B",
        architecture="Qwen2ForCausalLM",
        hidden_size=3584,
        kv_groups=((4, 128),),
    ),
    AotModel(
        name="Qwen/Qwen3-8B",
        architecture="Qwen3ForCausalLM",
        hidden_size=4096,
        kv_groups=((8, 128),),
    ),
    AotModel(
        name="mistralai/Mistral-7B-Instruct-v0.3",
        architecture="MistralForCausalLM",
        hidden_size=4096,
        kv_groups=((8, 128),),
    ),
    AotModel(
        name="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        architecture="Mistral3ForConditionalGeneration",
        hidden_size=5120,
        kv_groups=((8, 128),),
    ),
)


def store_element_sizes(model: AotModel) -> set[int]:
    return {kv * hd * KV_CACHE_DTYPE_BYTES for kv, hd in model.kv_groups}


def index_variants(model: AotModel) -> set[tuple[int, int]]:
    if not model.embed_indexing:
        return set()
    element_size = model.hidden_size * EMBED_DTYPE_BYTES
    return {(element_size, num_splits_for(element_size))}


def fast_index_copy_feature_sizes(model: AotModel) -> set[int]:
    sizes: set[int] = set()
    for fmt in model.expert_formats:
        assert model.moe_intermediate_size is not None
        sizes.update(
            expert_bank_row_bytes(fmt, model.hidden_size, model.moe_intermediate_size).values()
        )
    return sizes



def aggregate_store_element_sizes() -> tuple[int, ...]:
    sizes: set[int] = set()
    for model in SUPPORTED_MODELS:
        sizes.update(store_element_sizes(model))
    return tuple(sorted(sizes))


def aggregate_index_variants() -> tuple[tuple[int, int], ...]:
    variants: set[tuple[int, int]] = set()
    for model in SUPPORTED_MODELS:
        variants.update(index_variants(model))
    return tuple(sorted(variants))


# Synthetic per-bank sizes tests/moe/test_fused_copy.py exercises (its FEATS list);
# prebuilt so the suite also runs under FREETOKEN_DISABLE_JIT=1 release validation.
TEST_FEATURE_SIZES = (512, 8192)


def aggregate_fast_index_copy_feature_sizes() -> tuple[int, ...]:
    sizes: set[int] = set(TEST_FEATURE_SIZES)
    for model in SUPPORTED_MODELS:
        sizes.update(fast_index_copy_feature_sizes(model))
    return tuple(sorted(sizes))


__all__ = [
    "AotModel",
    "SUPPORTED_MODELS",
    "TEST_FEATURE_SIZES",
    "aggregate_fast_index_copy_feature_sizes",
    "aggregate_index_variants",
    "aggregate_store_element_sizes",
    "expert_bank_row_bytes",
    "fast_index_copy_feature_sizes",
    "index_variants",
    "store_element_sizes",
]
