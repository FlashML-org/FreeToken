from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


def _build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> List[int]:
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [round(start + (i * span) / (num_draft_layers - 1)) for i in range(num_draft_layers)]


@dataclass
class DFlashConfig:
    """Config for DFlash draft model, parsed from checkpoint's ``dflash_config``."""

    block_size: int = 16
    mask_token_id: int = 248077
    target_layer_ids: List[int] = field(default_factory=list)

    # Draft model architecture (from top-level config fields)
    hidden_size: int = 2048
    num_hidden_layers: int = 6
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 6144
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    sliding_window: int = 4096
    max_position_embeddings: int = 32768
    layer_types: List[str] = field(default_factory=list)
    is_causal: bool | None = None
    input_embedding_scale: float = 1.0
    output_multiplier: float = 1.0
    final_logit_softcapping: float | None = None

    @property
    def num_target_layers(self) -> int:
        return len(self.target_layer_ids)

    @property
    def context_dim(self) -> int:
        """Concatenated target hidden states dimension = num_target_layers * hidden_size."""
        return self.num_target_layers * self.hidden_size

    @classmethod
    def from_hf_config(cls, hf_config) -> DFlashConfig:
        # hf_config can be a dict or a HfConfig object
        if isinstance(hf_config, dict):
            cfg = hf_config
        else:
            cfg = hf_config.to_dict() if hasattr(hf_config, 'to_dict') else vars(hf_config)
        dflash_cfg = cfg.get("dflash_config", {})
        layer_types = cfg.get("layer_types", [])
        rope_params = cfg.get("rope_parameters", {})
        target_layer_ids = dflash_cfg.get("target_layer_ids")
        if target_layer_ids is None and "num_target_layers" in cfg:
            target_layer_ids = _build_target_layer_ids(cfg["num_target_layers"], cfg.get("num_hidden_layers", 6))
        return cls(
            block_size=dflash_cfg.get("block_size", 16),
            mask_token_id=dflash_cfg.get("mask_token_id", 248077),
            target_layer_ids=target_layer_ids or [],
            hidden_size=cfg.get("hidden_size", 2048),
            num_hidden_layers=cfg.get("num_hidden_layers", 6),
            num_attention_heads=cfg.get("num_attention_heads", 32),
            num_key_value_heads=cfg.get("num_key_value_heads", 8),
            head_dim=cfg.get("head_dim", 128),
            intermediate_size=cfg.get("intermediate_size", 6144),
            vocab_size=cfg.get("vocab_size", 248320),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            rope_theta=rope_params.get("rope_theta", 10000000.0),
            sliding_window=cfg.get("sliding_window", 4096),
            max_position_embeddings=cfg.get("max_position_embeddings", 32768),
            layer_types=layer_types,
            is_causal=cfg.get("is_causal", dflash_cfg.get("is_causal", None)),
            input_embedding_scale=dflash_cfg.get("input_embedding_scale", cfg.get("input_embedding_scale", 1.0)),
            output_multiplier=dflash_cfg.get("output_multiplier", cfg.get("output_multiplier", 1.0)),
            final_logit_softcapping=dflash_cfg.get(
                "final_logit_softcapping",
                cfg.get("final_logit_softcapping", None),
            ),
        )
