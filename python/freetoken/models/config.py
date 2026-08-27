from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Literal, Tuple, TypeAlias

from freetoken.attention.base import AttnType

# State-dict key prefixes for the (optional) vision stack. Used both to drop the vision
# config (so the tower is never built) and to skip the matching tensors in the FTW reader.
VISION_KEY_PREFIXES = ("vision_tower.", "embed_vision.")
_VISION_TRUE = {"1", "true", "yes", "on"}


def vision_load_enabled() -> bool:
    """Vision is opt-in (default OFF). The vision tower + multimodal embedder are ~1 GiB of
    resident, never-quantized (bf16) GPU weights that text-only serving never touches, so we
    skip building and loading them unless ``FREETOKEN_LOAD_VISION=1`` is set."""
    return os.getenv("FREETOKEN_LOAD_VISION", "0").strip().lower() in _VISION_TRUE


def detect_expert_quant(hf_config: Any) -> str:
    """Routed-expert quantization from a checkpoint's ``quantization_config``: ``"nvfp4"`` for
    a ModelOpt FP4 build, else the lowercased algo string (``"none"`` when unquantized). Models
    with mixed-precision configs (e.g. qwen3_5_moe) need their own detector."""
    quant = getattr(hf_config, "quantization_config", None)
    if quant is None:
        return "none"
    if isinstance(quant, dict):
        algo = quant.get("quant_algo") or quant.get("quant_method")
    else:
        algo = getattr(quant, "quant_algo", None) or getattr(quant, "quant_method", None)
    if algo is None:
        return "none"
    return "nvfp4" if "fp4" in str(algo).lower() else str(algo).lower()


def detect_compressed_tensors_nvfp4(hf_config: Any) -> bool:
    """Detect a compressed-tensors NVFP4 (W4A16) checkpoint (dense Qwen3.6-27B,
    Muse-Glimmer-30B-NVFP4, ...).

    llm-compressor stores ``quant_method == "compressed-tensors"`` with a ``config_groups``
    map whose ``weights`` are ``num_bits=4, type="float", group_size=16,
    strategy="tensor_group"`` -- the NVFP4 layout (``weight_packed`` uint8 +
    ``weight_scale`` fp8 block + scalar ``weight_global_scale``). All ``targets:
    ["Linear"]`` are NVFP4 except the per-module ``ignore`` list. A 4-bit float scheme
    with a DIFFERENT geometry (MXFP4: group_size 32, real since LLM Compressor 0.9)
    raises instead of routing into the NVFP4 loader and dying in a shape assert."""
    quant = getattr(hf_config, "quantization_config", None)
    if quant is None:
        return False
    get = quant.get if isinstance(quant, dict) else (lambda k, d=None: getattr(quant, k, d))
    if str(get("quant_method") or "").lower() != "compressed-tensors":
        return False
    groups = get("config_groups") or {}
    # Verdicts are collected across ALL groups before returning: an early return on
    # the first NVFP4 group would accept a mixed {nvfp4, mxfp4} checkpoint (and the
    # error would depend on the groups' key order).
    saw_nvfp4 = False
    for g in (groups.values() if isinstance(groups, dict) else []):
        w = (g or {}).get("weights") or {}
        if int(w.get("num_bits", 0) or 0) != 4 or str(w.get("type", "")).lower() != "float":
            continue
        # vLLM's _is_nvfp4_format gates on the same two fields.
        group_size = int(w.get("group_size", 0) or 0)
        strategy = str(w.get("strategy", "")).lower()
        if group_size == 16 and strategy == "tensor_group":
            saw_nvfp4 = True
            continue
        raise ValueError(
            "unsupported compressed-tensors 4-bit float scheme "
            f"(group_size={group_size}, strategy={strategy!r}); FreeToken serves "
            "NVFP4 (group_size=16, strategy=tensor_group) only"
        )
    return saw_nvfp4


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class KVCacheGroupSpec:
    name: str
    layer_ids: Tuple[int, ...]
    num_kv_heads: int
    head_dim: int
    sliding_window: int | None
    # Latent-KV MLA group: the pool stores ONE latent slab per token (no separate V)
    # and the cost model budgets 1x instead of 2x per token. When the model also runs
    # DSA, ``index_head_dim``/``num_index_layers`` size the index-key slab, so the
    # pool factory and the cost model derive the same bytes from the same spec.
    mla: bool = False
    index_head_dim: int = 0
    num_index_layers: int = 0
    index_num_kv_heads: int = 1
    index_compress_ratio: int = 1
    index_token_budget: int = 0
    # Attention-type taxonomy value for this group; drives the backend capability
    # matrix and (with the pool factory) selects the KV pool family.
    attn_type: AttnType = AttnType.FULL

    @property
    def num_layers(self) -> int:
        return len(self.layer_ids)

    @property
    def is_swa(self) -> bool:
        return self.name == "swa"


@dataclass(frozen=True)
class BaseAttentionGroupConfig:
    name: str
    layer_ids: Tuple[int, ...]

    def owns_layer(self, layer_id: int) -> bool:
        return layer_id in self.layer_ids


@dataclass(frozen=True)
class FullAttentionGroupConfig(BaseAttentionGroupConfig):
    kind: ClassVar[Literal["full"]] = "full"
    cache_kind: ClassVar[Literal["paged_kv"]] = "paged_kv"

    num_kv_heads: int
    head_dim: int
    rotary_config: RotaryConfig
    k_eq_v: bool = False
    # Latent-KV MLA (single latent slab, V aliases K). With DSA, ``index_head_dim`` /
    # ``num_index_layers`` size the index-key slab -- the pool factory and the KV cost
    # model both read these, so they can never disagree.
    mla: bool = False
    index_head_dim: int = 0
    num_index_layers: int = 0


@dataclass(frozen=True)
class SWAAttentionGroupConfig(BaseAttentionGroupConfig):
    kind: ClassVar[Literal["swa"]] = "swa"
    cache_kind: ClassVar[Literal["swa_kv"]] = "swa_kv"

    num_kv_heads: int
    head_dim: int
    rotary_config: RotaryConfig
    sliding_window: int


@dataclass(frozen=True)
class QSAAttentionGroupConfig(BaseAttentionGroupConfig):
    """Qwen compressed sparse-attention group.

    Full K/V remains paged at token resolution. The indexer keeps one BF16
    key per ``index_compress_ratio`` tokens, then selects
    ``index_token_budget`` original tokens for exact sparse GQA.
    """

    kind: ClassVar[Literal["qsa"]] = "qsa"
    cache_kind: ClassVar[Literal["qsa_paged"]] = "qsa_paged"

    num_kv_heads: int
    head_dim: int
    rotary_config: RotaryConfig
    index_num_heads: int
    index_num_kv_heads: int
    index_head_dim: int
    index_token_budget: int
    index_compress_ratio: int


@dataclass(frozen=True)
class LinearGatedDeltaGroupConfig(BaseAttentionGroupConfig):
    kind: ClassVar[Literal["linear_gated_delta"]] = "linear_gated_delta"
    cache_kind: ClassVar[Literal["linear_state"]] = "linear_state"

    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_dim: int
    output_gate: bool


@dataclass(frozen=True)
class DSV4AttentionGroupConfig(BaseAttentionGroupConfig):
    """DSV4 sparse attention (window + compressed tiers + Lightning Indexer). Standalone
    class on purpose: subclassing SWAAttentionGroupConfig would flip has_swa_attention
    and reroute DSV4 through the SWA gates. Geometry beyond the latent width lives in
    dsv4_args; the pool prices itself from there, not from this spec."""

    kind: ClassVar[Literal["dsv4"]] = "dsv4"
    cache_kind: ClassVar[Literal["dsv4_paged"]] = "dsv4_paged"

    num_kv_heads: int
    head_dim: int
    sliding_window: int  # the P-token window page


AttentionGroupConfig: TypeAlias = (
    FullAttentionGroupConfig
    | QSAAttentionGroupConfig
    | SWAAttentionGroupConfig
    | LinearGatedDeltaGroupConfig
    | DSV4AttentionGroupConfig
)


def _full_group_attn_type(group: FullAttentionGroupConfig) -> AttnType:
    # Mirrors the pool-factory split: mla + index slab -> DSAKVCache, mla -> MLAKVCache,
    # GQA (non-mla) + index slab -> BSAKVCache (MiniMax-M3 block-sparse attention).
    if not group.mla:
        if group.index_head_dim > 0 and group.num_index_layers > 0:
            return AttnType.BSA
        return AttnType.FULL
    if group.index_head_dim > 0 and group.num_index_layers > 0:
        return AttnType.DSA
    return AttnType.MLA


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]
    moe_backend: str = "fused"
    # ----- optional, model-specific extensions (default keeps other models intact) -----
    moe_enabled: bool = False
    # Weight quantization of the MoE experts only. "none" keeps the default BF16
    # offload/fused path; "nvfp4" stores experts as packed FP4 + block scales;
    # "fp8_block" is DeepSeek-V3-style 128x128 block-fp8 (weight fp8-e4m3 +
    # weight_scale_inv per block), also applied to the dense projections.
    expert_quant: str = "none"
    # NVFP4 routed-expert GEMM backend (--nvfp4-backend); injected from EngineConfig.
    nvfp4_backend: str = "triton"
    # Block size (out, in) for block-wise weight quantization (fp8_block: (128, 128)).
    weight_block_size: tuple[int, int] | None = None
    # Weight quantization of the *dense* attention / GatedDeltaNet projections (separate
    # from the routed experts above). "fp8_pertensor" keeps them fp8-e4m3 + a per-output-row
    # scale and runs a W8A16 kernel (modelopt MIXED_PRECISION); "none" leaves them bf16
    # (dequant-at-load for any other dense quant, e.g. NVFP4 shared_expert/lm_head).
    attn_quant: str = "none"
    # Weight quantization of the *dense* NVFP4 MLP projections -- the shared expert, and dense
    # (non-MoE) MLP layers -- which NVFP4 checkpoints store as packed FP4 like the routed
    # experts. "nvfp4" keeps them packed and runs the W4A16 dense kernels (quartering their
    # decode weight traffic); "none" dequantizes them to bf16 at load. Set independently of the
    # routed experts and lm_head: e.g. pure-NVFP4 Qwen3.5 has bf16 attn + bf16 lm_head but FP4
    # shared experts, so this is "nvfp4" while attn_quant / lm_head_quant are "none".
    dense_quant: str = "none"
    # Weight quantization of the lm_head. "nvfp4" keeps the (untied) FP4 head native (W4A16) --
    # the bf16 dequant of this ~1 GB matrix was the single largest decode kernel; "none" leaves
    # it bf16. Separate from dense_quant because only some NVFP4 checkpoints quantize lm_head
    # (modelopt MIXED_PRECISION does; pure NVFP4 leaves it bf16).
    lm_head_quant: str = "none"
    shared_expert_intermediate_size: int = 0
    use_qk_norm: bool = False
    # ----- DeepSeek/GLM-style MoE extensions (default keeps other models intact) -----
    # The first ``first_k_dense_replace`` decoder layers use a dense MLP instead of the
    # sparse MoE block (GLM-4: 3). Experts (and the offload cache) therefore only exist
    # for layers ``[first_k_dense_replace, num_layers)``.
    first_k_dense_replace: int = 0
    # Always-on shared expert(s) added to every MoE layer's output (GLM-4: 1).
    n_shared_experts: int = 0
    # Selected-expert weights are multiplied by this after renormalization (GLM-4: 2.5).
    routed_scaling_factor: float = 1.0
    # Group-limited routing (DeepSeek-V3 noaux_tc); GLM-4 uses n_group=topk_group=1 so the
    # grouping is a no-op, but we keep the knobs for completeness.
    n_group: int = 1
    topk_group: int = 1
    attn_sm_scale: float | None = None  # None -> 1/sqrt(head_dim)
    final_logit_softcapping: float | None = None
    embedding_scale: float | None = None
    # Second RMSNorm eps for models whose post-sublayer norms use a different eps than the
    # pre-sublayer ones (muse_glimmer: post_attention/post_feedforward at 1e-8 vs 1e-5).
    post_norm_eps: float | None = None
    # Scalar the raw lm_head logits are multiplied by BEFORE final_logit_softcapping
    # (muse_glimmer: 1/sqrt(hidden/256)); None leaves the logits unscaled.
    output_multiplier: float | None = None
    vision_config: Any | None = None
    image_token_id: int | None = None
    attention_groups: Tuple[AttentionGroupConfig, ...] = ()
    has_attn_bias: bool = False
    has_router_bias: bool = False
    moe_weight_format: str | None = None
    swiglu_limit: float | None = None
    hidden_act_alpha: float = 1.702
    # Full DeepseekV4Args payload for the DSV4-specific machinery (MLA sparse attention,
    # CSA/HCA compressors, Lightning Indexer, manifold-constrained Hyper-Connections,
    # hash routing). Opaque to model-agnostic engine code; None for non-DSV4 models.
    dsv4_args: Any | None = None
    # GLM-5.2 (glm_moe_dsa) MLA/DSA payload (GlmMoeDsaArgs): the MLA low-rank dims and the
    # DSA indexer geometry the model module needs. Opaque to model-agnostic engine code;
    # None for every other model.
    glm_dsa_args: Any | None = None
    # MiniMax-M3 (minimax_m3) payload (MiniMaxM3Args): the block-sparse indexer geometry
    # (index heads/dim, top-k blocks, init/local blocks, sparse layer set) plus the
    # swigluoai/dense-MLP scalars the model module needs. Opaque to model-agnostic engine
    # code; None for every other model.
    m3_args: Any | None = None
    # Qwen4-Exp payload: hyper-connections, PLE host embedding geometry, and the
    # QSA exact-context ceiling. Opaque outside the qwen4_exp model package.
    qwen4_args: Any | None = None
    # Generic execution-path capability flags (set by a model's parse_config) so the engine and
    # factories stay model-agnostic instead of branching on dsv4_args:
    single_stream_only: bool = False  # model runs one sequence at a time -> force bs=1
    requires_naive_cache: bool = False  # model owns host/runtime state radix cannot snapshot
    supports_cuda_graph: bool = True  # False when forward performs host-side dynamic work

    @property
    def is_moe(self) -> bool:
        return "moe" in self.model_type or self.moe_enabled

    @property
    def num_moe_layers(self) -> int:
        """Number of layers that own a sparse MoE block (and offload-cache expert slots).

        Models with leading dense layers (``first_k_dense_replace`` > 0, e.g. GLM-4)
        only store experts for the trailing layers; everything else has all layers MoE.
        """
        return self.num_layers - self.first_k_dense_replace

    @property
    def is_multimodal(self) -> bool:
        return self.vision_config is not None

    @property
    def has_hybrid_attention(self) -> bool:
        return len(self.attention_groups) > 1

    @property
    def has_swa_attention(self) -> bool:
        return any(isinstance(group, SWAAttentionGroupConfig) for group in self.attention_groups)

    @property
    def has_linear_attention(self) -> bool:
        return any(
            isinstance(group, LinearGatedDeltaGroupConfig)
            for group in self.attention_groups
        )

    def default_full_attention_group(self) -> FullAttentionGroupConfig:
        return FullAttentionGroupConfig(
            name="full",
            layer_ids=tuple(range(self.num_layers)),
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            rotary_config=self.rotary_config,
        )

    def attention_group_for_layer(self, layer_id: int) -> AttentionGroupConfig:
        groups = self.attention_groups or (self.default_full_attention_group(),)
        matches = [group for group in groups if group.owns_layer(layer_id)]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one attention group for layer {layer_id}, "
                f"got {len(matches)}"
            )
        return matches[0]

    def swa_attention_group(self) -> SWAAttentionGroupConfig | None:
        groups = [
            group
            for group in self.attention_groups
            if isinstance(group, SWAAttentionGroupConfig)
        ]
        if len(groups) > 1:
            raise ValueError("Expected at most one SWA attention group")
        return groups[0] if groups else None

    def linear_attention_group(self) -> LinearGatedDeltaGroupConfig | None:
        groups = [
            group
            for group in self.attention_groups
            if isinstance(group, LinearGatedDeltaGroupConfig)
        ]
        if len(groups) > 1:
            raise ValueError("Expected at most one linear-gated-delta attention group")
        return groups[0] if groups else None

    def is_swa_layer(self, layer_id: int) -> bool:
        return isinstance(self.attention_group_for_layer(layer_id), SWAAttentionGroupConfig)

    def is_linear_layer(self, layer_id: int) -> bool:
        return isinstance(
            self.attention_group_for_layer(layer_id),
            LinearGatedDeltaGroupConfig,
        )

    def attn_type_for_layer(self, layer_id: int) -> AttnType:
        """Canonical per-layer attention-type lookup (the taxonomy is declared
        top-down on the attention groups; this is the layer-granular view)."""
        group = self.attention_group_for_layer(layer_id)
        if isinstance(group, LinearGatedDeltaGroupConfig):
            return AttnType.LINEAR
        if isinstance(group, SWAAttentionGroupConfig):
            return AttnType.SWA
        if isinstance(group, DSV4AttentionGroupConfig):
            return AttnType.DSV4
        if isinstance(group, QSAAttentionGroupConfig):
            return AttnType.QSA
        return _full_group_attn_type(group)

    def kv_cache_group_specs(self) -> Tuple[KVCacheGroupSpec, ...]:
        if not self.attention_groups:
            return (
                KVCacheGroupSpec(
                    name="full",
                    layer_ids=tuple(range(self.num_layers)),
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    sliding_window=None,
                ),
            )

        specs: list[KVCacheGroupSpec] = []
        for group in self.attention_groups:
            if isinstance(group, FullAttentionGroupConfig):
                specs.append(
                    KVCacheGroupSpec(
                        name=group.name,
                        layer_ids=group.layer_ids,
                        num_kv_heads=group.num_kv_heads,
                        head_dim=group.head_dim,
                        sliding_window=None,
                        mla=group.mla,
                        index_head_dim=group.index_head_dim,
                        num_index_layers=group.num_index_layers,
                        attn_type=_full_group_attn_type(group),
                    )
                )
            elif isinstance(group, QSAAttentionGroupConfig):
                specs.append(
                    KVCacheGroupSpec(
                        name=group.name,
                        layer_ids=group.layer_ids,
                        num_kv_heads=group.num_kv_heads,
                        head_dim=group.head_dim,
                        sliding_window=None,
                        index_head_dim=group.index_head_dim,
                        num_index_layers=len(group.layer_ids),
                        index_num_kv_heads=group.index_num_kv_heads,
                        index_compress_ratio=group.index_compress_ratio,
                        index_token_budget=group.index_token_budget,
                        attn_type=AttnType.QSA,
                    )
                )
            elif isinstance(group, SWAAttentionGroupConfig):
                specs.append(
                    KVCacheGroupSpec(
                        name=group.name,
                        layer_ids=group.layer_ids,
                        num_kv_heads=group.num_kv_heads,
                        head_dim=group.head_dim,
                        sliding_window=group.sliding_window,
                        attn_type=AttnType.SWA,
                    )
                )
            elif isinstance(group, DSV4AttentionGroupConfig):
                # Matrix/taxonomy entry only: DSV4 sizing never reads this spec
                # (the pool prices itself from dsv4_args), and is_swa/mla stay
                # False so no generic spec walker treats it as SWA or MLA.
                specs.append(
                    KVCacheGroupSpec(
                        name=group.name,
                        layer_ids=group.layer_ids,
                        num_kv_heads=group.num_kv_heads,
                        head_dim=group.head_dim,
                        sliding_window=group.sliding_window,
                        attn_type=AttnType.DSV4,
                    )
                )
        return tuple(specs)

    def kv_cache_groups(self) -> List[Tuple[int, int, int]]:
        """Return ``[(num_layers, num_kv_heads, head_dim), ...]`` per KV group.

        This legacy memory-accounting helper is derived from the SGLang-style
        group specs used by hybrid full/SWA KV cache storage.
        """
        return [
            (group.num_layers, group.num_kv_heads, group.head_dim)
            for group in self.kv_cache_group_specs()
            if group.num_layers > 0
        ]
