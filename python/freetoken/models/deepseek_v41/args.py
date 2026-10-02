"""DeepSeek-V4.1-Flash hyperparameters and the static per-layer attention roles.

``DeepseekV41Args`` mirrors the reference ``ModelArgs`` (``inference/model.py``) field for field,
read from the HF ``config.json``'s ``text_config`` (the ``inference/config.json`` dialect is also
accepted). Everything the engine reconciles at runtime (``max_seq_len``, the replay policy) is
attached by ``parse_config`` / the engine, not read from the checkpoint.

``LayerRole`` is the attention mode table (tech report sec. 2.3.1): every compressing layer is Full
(owns the compressed KV + indexer K), Reindex (own indexer over a shared K) or Reuse (shared
Top-K); the candidate pool of the Hierarchical Sparse Indexer (sec. 2.3.2) is produced by the
first decoder Full layer and consumed by the decoder Reindex layers after it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

# text_config key -> ModelArgs key, where the two dialects differ
_HF_TO_ARGS = {
    "hidden_size": "dim",
    "moe_intermediate_size": "moe_inter_dim",
    "num_hidden_layers": "n_layers",
    "num_nextn_predict_layers": "n_mtp_layers",
    "num_attention_heads": "n_heads",
    "num_experts_per_tok": "n_activated_experts",
    "scoring_func": "score_func",
    "routed_scaling_factor": "route_scale",
    "qk_rope_head_dim": "rope_head_dim",
    "rms_norm_eps": "norm_eps",
    "sliding_window": "window_size",
    "kv_source_layer_ids": "kv_source_layers",
    "index_source_layer_ids": "index_source_layers",
    "candidate_source_layer_id": "candidate_source_layer",
    "engram_pad_token_id": "engram_pad_id",
    "dspark_num_experts_per_tok": "dspark_n_activated_experts",
}


class Mode(str, Enum):
    WINDOW = "window"  # sliding window only (compress_ratio == 0)
    FULL = "full"  # computes main KV + indexer K, runs its indexer
    REINDEX = "reindex"  # reuses main KV + indexer K, runs its own indexer
    REUSE = "reuse"  # reuses main KV and the latest Top-K


@dataclass(frozen=True)
class LayerRole:
    layer_id: int
    ratio: int
    mode: Mode
    kv_source: int | None  # the layer whose main KV / indexer K this layer reads (itself when Full)
    is_candidate_source: bool  # builds the hierarchical candidate pool
    uses_candidates: bool  # indexes only inside the candidate pool

    @property
    def compresses(self) -> bool:
        return self.ratio > 0

    @property
    def has_indexer(self) -> bool:
        return self.mode in (Mode.FULL, Mode.REINDEX)


@dataclass
class DeepseekV41Args:
    vocab_size: int = 129280
    dim: int = 5120
    moe_inter_dim: int = 2304
    n_layers: int = 40
    n_mtp_layers: int = 3
    n_heads: int = 64
    # moe
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    norm_topk_prob: bool = True
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    # attention: latent q, one shared latent kv head, grouped low-rank output projection
    q_lora_rank: int = 1280
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 1024
    # sparse attention
    window_size: int = 128
    compress_ratios: tuple[int, ...] = ()  # one per layer, MTP layers included
    kv_source_layers: tuple[int, ...] = ()
    index_source_layers: tuple[int, ...] = ()
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer: int = -1
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0
    # hyper-connections (single-pass mHC)
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # engram
    engram_layer_ids: tuple[int, ...] = ()
    engram_num_embeddings: tuple[int, ...] = ()
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 0
    engram_n_heads: int = 0
    engram_head_dim: int = 0
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 0
    # dspark (weights are not served in v1; kept so the config round-trips)
    dspark_block_size: int = 0
    dspark_target_layer_ids: tuple[int, ...] = ()
    # ---- runtime (engine-reconciled, never read from the checkpoint) ----
    max_seq_len: int = 4096
    max_batch_size: int = 1
    # "bounded": the decoder runs on the prompt's last window_size tokens with the sliding window
    # truncated there (Decoder SWA Bounded Replay, tech report sec. 3.2.2); "exact": on every token.
    swa_decoder_replay: Literal["bounded", "exact"] = "bounded"
    roles: tuple[LayerRole, ...] = field(default_factory=tuple, repr=False)

    def __post_init__(self) -> None:
        for name in ("compress_ratios", "kv_source_layers", "index_source_layers", "engram_layer_ids",
                     "engram_num_embeddings", "dspark_target_layer_ids"):
            setattr(self, name, tuple(int(v) for v in getattr(self, name)))
        if len(self.compress_ratios) < self.n_layers:
            raise ValueError(f"compress_ratios has {len(self.compress_ratios)} entries for {self.n_layers} layers")
        if self.n_shared_experts != 1:
            raise ValueError("DeepSeek-V4.1 has exactly one shared expert")
        if self.hc_mult < 1:
            raise ValueError("hc_mult must be positive")
        self.roles = tuple(self._role(layer) for layer in range(self.n_layers))
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError("engram_layer_ids and engram_num_embeddings differ in length")

    # ---- derived ----
    def _role(self, layer: int) -> LayerRole:
        ratio = self.compress_ratios[layer]
        if ratio == 0:
            return LayerRole(layer, 0, Mode.WINDOW, None, False, False)
        sources = [s for s in self.kv_source_layers if s <= layer]
        if not sources:
            raise ValueError(f"layer {layer} compresses but no kv source precedes it")
        source = max(sources)
        if self.compress_ratios[source] != ratio:
            raise ValueError(f"layer {layer} (ratio {ratio}) reads kv source {source} (ratio {self.compress_ratios[source]})")
        if layer == source:
            mode = Mode.FULL
        elif layer in self.index_source_layers:
            mode = Mode.REINDEX
        else:
            mode = Mode.REUSE
        cand = self.candidate_source_layer
        is_cand_src = cand >= 0 and layer == cand
        uses = cand >= 0 and cand < layer and mode is Mode.REINDEX
        if uses and self.kv_source_layers and max(s for s in self.kv_source_layers if s <= cand) != source:
            raise ValueError(f"layer {layer} would index the candidate pool of layer {cand} over a different kv source")
        return LayerRole(layer, ratio, mode, source, is_cand_src, uses)

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    @property
    def backbone_kv_sources(self) -> tuple[int, ...]:
        return tuple(s for s in self.kv_source_layers if s < self.n_layers)

    @property
    def decoder_start_layer(self) -> int:
        """The first decoder layer: the last kv source (CED projects the decoder's global KV from
        its input, the final encoder hidden state)."""
        return max(self.backbone_kv_sources) if self.backbone_kv_sources else self.n_layers

    @property
    def engram_hash_cols(self) -> int:
        return (self.engram_max_ngram_size - 1) * self.engram_n_heads

    @property
    def index_softmax_scale(self) -> float:
        return self.index_head_dim**-0.5

    def freqs_params(self, layer: int) -> tuple[int, int, float, float, int, int]:
        """``(rotary_dim, original_seq_len, theta, factor, beta_fast, beta_slow)``: compressing layers
        rotate at the compressed theta under YaRN, window-only layers at the base theta without it."""
        if self.compress_ratios[layer]:
            return self.rope_head_dim, self.original_seq_len, self.compress_rope_theta, self.rope_factor, self.beta_fast, self.beta_slow
        return self.rope_head_dim, 0, self.rope_theta, self.rope_factor, self.beta_fast, self.beta_slow

    @classmethod
    def from_hf(cls, hf_config: Any) -> "DeepseekV41Args":
        """From the HF config (``text_config`` of the multimodal wrapper, or the text config itself,
        or the ``inference/config.json`` dialect)."""
        text = getattr(hf_config, "text_config", None) or hf_config
        if isinstance(text, dict):
            raw = text
        elif hasattr(text, "to_dict"):  # PretrainedConfig / RawConfigShim
            raw = text.to_dict()
        else:
            raw = vars(text)
        names = {f.name for f in cls.__dataclass_fields__.values()} - {"roles", "max_seq_len", "max_batch_size", "swa_decoder_replay"}
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            key = _HF_TO_ARGS.get(key, key)
            if key in names:
                kwargs[key] = value
        # HF nests the YaRN parameters; the inference dialect keeps them flat
        scaling = raw.get("rope_scaling") or raw.get("rope_parameters")
        if isinstance(scaling, dict):
            if "factor" in scaling:
                kwargs["rope_factor"] = scaling["factor"]
            for src, dst in (("beta_fast", "beta_fast"), ("beta_slow", "beta_slow"), ("original_max_position_embeddings", "original_seq_len")):
                if src in scaling:
                    kwargs[dst] = scaling[src]
        return cls(**kwargs)


__all__ = ["DeepseekV41Args", "LayerRole", "Mode"]
