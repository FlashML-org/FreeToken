"""Engine-facing config for DeepSeek-V4-Flash.

``parse_config`` maps the standard transformer fields the engine needs (layer
count, hidden size, vocab, MoE expert counts, etc.) into :class:`ModelConfig`,
and carries the full :class:`DeepseekV4Args` in ``ModelConfig.dsv4_args`` for the
DSV4-specific machinery (MLA sparse attention, CSA/HCA compressors, Lightning
Indexer, manifold-constrained Hyper-Connections, hash routing).

The transformers ``AutoConfig`` for ``deepseek_v4`` drops a few keys (e.g.
``compress_ratios``), so we recover the authoritative args from the checkpoint's
``inference/config.json`` via :func:`load_args`, keyed off ``hf_config._name_or_path``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from freetoken.models.config import DSV4AttentionGroupConfig, ModelConfig, RotaryConfig

from .args import load_args

# ``<｜deepseek_image｜>``: the only vocab-level image token. The reference expands each
# occurrence into a block of ``vocab_size + type`` pseudo ids; here the block keeps this
# one id and the image embeddings are scattered over the span it occupies.
IMAGE_TOKEN_ID = 129264


@dataclass(frozen=True)
class DSV4VisionConfig:
    """The tower's dims, split out of the checkpoint's flat inference config so the engine's
    section nulling and the loader treat DSV4 like every other family with a vision tower.
    Field names follow the reference (``inference/model.py`` ModelArgs) one to one."""

    vision_n_layers: int
    vision_dim: int
    vision_n_heads: int
    vision_inter_dim: int
    vision_patch_size: int
    vision_rope_theta: float
    vision_downsample_ratio: int
    vision_max_n_token: int
    vision_min_pixels: int
    vision_max_wh_ratio: int
    # the language model's hidden size the aligner projects into (ModelArgs.dim)
    text_dim: int


def parse_config(hf_config: Any) -> ModelConfig:
    model_path = getattr(hf_config, "_name_or_path", None) or getattr(
        hf_config, "name_or_path", None
    )
    if not model_path:
        raise ValueError(
            "DeepSeek-V4 parse_config needs the checkpoint path (hf_config._name_or_path)"
        )
    args = load_args(model_path, max_batch_size=1)

    rope_scaling = {
        "rope_type": "yarn",
        "factor": args.rope_factor,
        "beta_fast": args.beta_fast,
        "beta_slow": args.beta_slow,
        "original_max_position_embeddings": args.original_seq_len,
    }

    # Serving ceiling: the HF top-level config's max_position_embeddings (1M on V4-Flash).
    # args.original_seq_len is yarn's PRE-scaling length (64k); the compressed tiers serve
    # original_seq_len * rope_factor positions, so fall back to that product when a checkpoint
    # ships only the inference/ dialect.
    max_position = int(getattr(hf_config, "max_position_embeddings", 0) or 0)
    if max_position <= 0:
        max_position = int(args.original_seq_len * args.rope_factor)

    vision_config = parse_vision_config(hf_config)

    return ModelConfig(
        num_layers=args.n_layers,
        num_qo_heads=args.n_heads,
        num_kv_heads=1,  # MLA: a single shared latent KV head (K == V)
        head_dim=args.head_dim,
        hidden_size=args.dim,
        vocab_size=args.vocab_size,
        intermediate_size=args.moe_inter_dim,
        hidden_act="silu",
        rms_norm_eps=args.norm_eps,
        tie_word_embeddings=False,
        rotary_config=RotaryConfig(
            head_dim=args.head_dim,
            rotary_dim=args.rope_head_dim,
            max_position=max_position,
            base=args.rope_theta,
            scaling=rope_scaling,
        ),
        num_experts=args.n_routed_experts,
        num_experts_per_tok=args.n_activated_experts,
        moe_intermediate_size=args.moe_inter_dim,
        norm_topk_prob=True,
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        moe_enabled=True,
        expert_quant="ds_fp4",
        # NB: DSV4 has MoE on every layer (no dense-replace) and reads its routing /
        # shared-expert / scaling config from dsv4_args, so the generic DeepSeek-family
        # MoE ModelConfig fields (first_k_dense_replace / n_shared_experts /
        # routed_scaling_factor) are intentionally not set here.
        attn_sm_scale=args.head_dim**-0.5,
        dsv4_args=args,
        # Declares attn_type=DSV4 for the backend capability matrix (auto resolves to
        # dsv4_sparse, anything else is rejected at config time). Sizing stays on
        # dsv4_args; nothing generic prices this group.
        attention_groups=(
            DSV4AttentionGroupConfig(
                name="dsv4",
                layer_ids=tuple(range(args.n_layers)),
                num_kv_heads=1,  # MLA-style shared latent (K == V)
                head_dim=args.head_dim,
                sliding_window=args.window_size,
            ),
        ),
        vision_config=vision_config,
        image_token_id=IMAGE_TOKEN_ID if vision_config is not None else None,
    )


def parse_vision_config(hf_config: Any) -> DSV4VisionConfig | None:
    """The tower's dims from the checkpoint config, or None when it carries no vision stack.

    The vision dims sit at the top level of the checkpoint config, next to the language
    model's, and the engine nulls ``vision_n_layers`` on the copy it hands the parser when
    this process serves text-only (--text-model-only / --mm-disable). That null is what
    decides, so the built tower always matches the weights the loader is told to read.
    """
    n_layers = int(getattr(hf_config, "vision_n_layers", 0) or 0)
    if n_layers <= 0:
        return None
    return DSV4VisionConfig(
        vision_n_layers=n_layers,
        vision_dim=int(hf_config.vision_dim),
        vision_n_heads=int(hf_config.vision_n_heads),
        vision_inter_dim=int(hf_config.vision_inter_dim),
        vision_patch_size=int(hf_config.vision_patch_size),
        vision_rope_theta=float(hf_config.vision_rope_theta),
        vision_downsample_ratio=int(hf_config.vision_downsample_ratio),
        vision_max_n_token=int(hf_config.vision_max_n_token),
        vision_min_pixels=int(hf_config.vision_min_pixels),
        vision_max_wh_ratio=int(hf_config.vision_max_wh_ratio),
        text_dim=int(hf_config.hidden_size),
    )


__all__ = ["IMAGE_TOKEN_ID", "DSV4VisionConfig", "parse_config", "parse_vision_config"]
