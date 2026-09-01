"""GLM-5.2 (``glm_moe_dsa``) MLA + DSA hyperparameters.

GLM-5.2 is a DeepSeek-V3.2-class model: Multi-head Latent Attention (MLA) plus DSA
sparse attention (a Lightning/IndexShare indexer). This payload carries the MLA and
indexer dims the model module needs; it is stashed on ``ModelConfig.glm_dsa_args``
(opaque to the engine). The MoE/router knobs live on ``ModelConfig`` directly (shared
with the glm4_moe path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass(frozen=True)
class GlmMoeDsaArgs:
    hidden_size: int
    num_heads: int
    # MLA
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    norm_eps: float
    # RoPE
    rope_theta: float
    rope_interleave: bool
    indexer_rope_interleave: bool
    max_position: int
    # DSA indexer (IndexShare): "full" layers own an indexer, "shared" layers reuse
    # the most recent full layer's top-k selection; serving is faithful DSA
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    indexer_types: Tuple[str, ...]
    # DSA k-pool compression (GLM-5.3): `index_kpool` consecutive tokens are scored as ONE
    # candidate (softmax(gate+ape)-weighted key average); selected pools expand back to raw
    # token indices and the in-progress tail pool is always selected. 1 = per-token scoring
    # (GLM-5.2; every field below is inert then).
    index_kpool: int = 1
    index_kpool_always_select_tail: bool = True

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def mha_head_dim(self) -> int:
        # After the MLA up-projection Q/K carry qk_head_dim and V carries v_head_dim; the
        # checkpoint sizes them equal (256), so a single generic-MHA head_dim serves both.
        assert self.qk_head_dim == self.v_head_dim, (self.qk_head_dim, self.v_head_dim)
        return self.v_head_dim


def load_args(hf_config: Any) -> GlmMoeDsaArgs:
    rope = getattr(hf_config, "rope_parameters", None) or {}
    rope_theta = float(rope.get("rope_theta", getattr(hf_config, "rope_theta", 10000.0)))
    return GlmMoeDsaArgs(
        hidden_size=hf_config.hidden_size,
        num_heads=hf_config.num_attention_heads,
        q_lora_rank=hf_config.q_lora_rank,
        kv_lora_rank=hf_config.kv_lora_rank,
        qk_nope_head_dim=hf_config.qk_nope_head_dim,
        qk_rope_head_dim=hf_config.qk_rope_head_dim,
        v_head_dim=hf_config.v_head_dim,
        norm_eps=hf_config.rms_norm_eps,
        rope_theta=rope_theta,
        rope_interleave=bool(getattr(hf_config, "rope_interleave", True)),
        # sglang convention: is_neox_style = not indexer_rope_interleave. DSV3.2
        # defaults half-split (False); GLM-5.2 sets True (interleaved indexer rope --
        # transformers >= 5.13 is explicit that this DIFFERS from DeepSeek-V3.2).
        indexer_rope_interleave=bool(getattr(hf_config, "indexer_rope_interleave", False)),
        max_position=hf_config.max_position_embeddings,
        index_n_heads=int(getattr(hf_config, "index_n_heads", 0)),
        index_head_dim=int(getattr(hf_config, "index_head_dim", 0)),
        index_topk=int(getattr(hf_config, "index_topk", 0)),
        indexer_types=tuple(getattr(hf_config, "indexer_types", ()) or ()),
    )


__all__ = ["GlmMoeDsaArgs", "load_args"]
