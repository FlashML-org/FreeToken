"""DeepSeek-V4.1 shapes, including the shared CSA2 sources and vision tower."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from typing import Any


def as_dict(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return vars(value)


@dataclass
class DeepseekV41Args:
    max_batch_size: int = 1
    max_seq_len: int = 4096
    dtype: str = "fp8"
    vocab_size: int = 129280
    dim: int = 5120
    moe_inter_dim: int = 2304
    n_layers: int = 40
    n_mtp_layers: int = 3
    n_heads: int = 64
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: str = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    q_lora_rank: int = 1280
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: tuple[int, ...] = (0, 0) + (2,) * 18 + (1,) * 20 + (0,) * 3
    kv_source_layers: tuple[int, ...] = (2, 8, 14, 20)
    index_source_layers: tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    engram_layer_ids: tuple[int, ...] = (1, 14)
    engram_num_embeddings: tuple[int, ...] = (384006168, 384016682)
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 99092
    engram_dtype: str = "fp8"
    engram_block_size: int = 32
    engram_scale_fmt: str = "ue8m0"
    vision_n_layers: int = 0
    vision_dim: int = 1024
    vision_n_heads: int = 16
    vision_inter_dim: int = 2816
    vision_patch_size: int = 14
    vision_rope_theta: float = 10000.0
    vision_downsample_ratio: int = 3
    vision_max_n_token: int = 1024
    vision_min_pixels: int = 295936
    vision_max_wh_ratio: int | None = None
    image_token_id: int = 129264

    def __post_init__(self):
        for name in ("compress_ratios", "kv_source_layers", "index_source_layers",
                     "engram_layer_ids", "engram_num_embeddings"):
            setattr(self, name, tuple(getattr(self, name)))
        if self.n_layers <= 0 or len(self.compress_ratios) < self.n_layers:
            raise ValueError("DeepSeek-V4.1 needs a compression ratio for every backbone layer")
        if self.n_shared_experts != 1:
            raise ValueError("DeepSeek-V4.1 requires exactly one shared expert")
        if not 0 < self.n_activated_experts <= self.n_routed_experts or self.gate_temp <= 0:
            raise ValueError("Invalid DeepSeek-V4.1 expert routing configuration")
        if self.score_func not in {"sqrtsoftplus", "softmax", "sigmoid"}:
            raise ValueError(f"Unsupported DeepSeek-V4.1 routing score: {self.score_func}")
        if self.hc_mult <= 0 or self.hc_mult & (self.hc_mult - 1) or self.hc_sinkhorn_iters < 1:
            raise ValueError("DeepSeek-V4.1 requires power-of-two hc_mult and positive Sinkhorn iterations")
        if self.n_heads % self.o_groups or not 0 <= self.rope_head_dim <= self.head_dim:
            raise ValueError("Invalid DeepSeek-V4.1 attention geometry")
        for name in ("kv_source_layers", "index_source_layers"):
            sources = getattr(self, name)
            if sources != tuple(sorted(set(sources))):
                raise ValueError(f"{name} must contain unique increasing layer IDs")
            if any(i < 0 or i >= self.n_layers or not self.compress_ratios[i] for i in sources):
                raise ValueError(f"{name} must name compressed backbone layers")
            for layer, ratio in enumerate(self.compress_ratios[:self.n_layers]):
                if ratio not in (0, 1, 2):
                    raise ValueError(f"Unsupported CSA2 compression ratio {ratio}")
                source = max((i for i in sources if i <= layer), default=-1)
                if ratio and (source < 0 or self.compress_ratios[source] != ratio):
                    raise ValueError(f"Layer {layer} has no compatible {name} source")
        if not set(self.kv_source_layers).issubset(self.index_source_layers):
            raise ValueError("Every KV source must also own an indexer")
        if self.candidate_source_layer >= 0:
            if self.candidate_source_layer not in self.index_source_layers:
                raise ValueError("Candidate source must own an indexer")
            if self.candidate_block_size <= 0 or self.candidate_topk_blocks <= 0:
                raise ValueError("Candidate selection sizes must be positive")
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError("Engram layers and table sizes must have equal lengths")
        if any(i < 0 or i >= self.n_layers for i in self.engram_layer_ids):
            raise ValueError("Engram layer must belong to the backbone")
        if self.engram_dtype not in {"fp8", "fp4"}:
            raise ValueError("DeepSeek-V4.1 Engram tables require FP8 or FP4 weights")
        if self.engram_block_size != 32 or self.engram_scale_fmt != "ue8m0":
            raise ValueError("DeepSeek-V4.1 Engram tables require block_size=32 and UE8M0 scales")
        if self.engram_layer_ids and (self.engram_head_dim <= 0 or self.engram_head_dim % 32):
            raise ValueError("DeepSeek-V4.1 Engram head dimension must be a positive multiple of 32")

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    @property
    def vision_enabled(self) -> bool:
        return self.vision_n_layers > 0


_TEXT_NAMES = {
    "hidden_size": "dim", "moe_intermediate_size": "moe_inter_dim",
    "num_hidden_layers": "n_layers", "num_nextn_predict_layers": "n_mtp_layers",
    "num_attention_heads": "n_heads", "num_experts_per_tok": "n_activated_experts",
    "scoring_func": "score_func", "routed_scaling_factor": "route_scale",
    "qk_rope_head_dim": "rope_head_dim", "rms_norm_eps": "norm_eps",
    "sliding_window": "window_size", "kv_source_layer_ids": "kv_source_layers",
    "index_source_layer_ids": "index_source_layers",
    "candidate_source_layer_id": "candidate_source_layer", "engram_pad_token_id": "engram_pad_id",
}
_VISION_NAMES = {
    "num_hidden_layers": "vision_n_layers", "hidden_size": "vision_dim",
    "num_attention_heads": "vision_n_heads", "intermediate_size": "vision_inter_dim",
    "patch_size": "vision_patch_size", "rope_theta": "vision_rope_theta",
    "downsample_ratio": "vision_downsample_ratio", "max_image_tokens": "vision_max_n_token",
    "min_pixels": "vision_min_pixels", "max_wh_ratio": "vision_max_wh_ratio",
}


def load_args(config_or_path: Any, **overrides) -> DeepseekV41Args:
    """Read either native inference fields or the HF nested configuration without model code."""
    if isinstance(config_or_path, (str, os.PathLike)):
        path = os.fspath(config_or_path)
        if os.path.isdir(path):
            path = os.path.join(path, "config.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        else:
            from freetoken.utils import cached_load_hf_config

            raw = as_dict(cached_load_hf_config(path))
    else:
        raw = as_dict(config_or_path)
    text = as_dict(raw.get("text_config", raw))
    valid = {field.name for field in fields(DeepseekV41Args)}
    kwargs = {_TEXT_NAMES.get(k, k): v for k, v in text.items()
              if _TEXT_NAMES.get(k, k) in valid}
    scaling = as_dict(text.get("rope_scaling"))
    for src, dst in (("factor", "rope_factor"), ("beta_fast", "beta_fast"),
                     ("beta_slow", "beta_slow"),
                     ("original_max_position_embeddings", "original_seq_len")):
        if src in scaling:
            kwargs[dst] = scaling[src]
    for src, dst in _VISION_NAMES.items():
        if src in as_dict(raw.get("vision_config")):
            kwargs[dst] = as_dict(raw["vision_config"])[src]
    if "vision_config" in raw and raw["vision_config"] is None:
        kwargs["vision_n_layers"] = 0
    if "image_token_id" in raw:
        kwargs["image_token_id"] = raw["image_token_id"]
    quant = as_dict(raw.get("quantization_config"))
    for name in ("engram_dtype", "engram_block_size", "engram_scale_fmt"):
        if name in quant:
            kwargs[name] = quant[name]
    # HF dtype describes activations; resident FP8 storage comes from quantization_config.
    if "text_config" in raw:
        kwargs["dtype"] = "fp8"
    kwargs.update(overrides)
    return DeepseekV41Args(**kwargs)


__all__ = ["DeepseekV41Args", "load_args"]
