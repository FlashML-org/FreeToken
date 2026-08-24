from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.utils import nvtx_annotate

from freetoken.models.blocks import BaseLLMModel

from .attention import Gemma4Attention
from .moe import Gemma4DenseMLP, Gemma4MLP
from .vision import Gemma4MultimodalEmbedder, Gemma4VisionModel

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# Upper bound on tokens in a single forward (prefill chunk / decode batch). The E-series
# KV-sharing stash is allocated once at this size so its address is stable across CUDA-graph
# capture and replay. Matches the default --max-prefill-length.
_STASH_MAX_TOKENS = 8192

# The E-series per-layer-embedding table's state-dict key (kept in host RAM).
PER_LAYER_EMBED_KEY = "model.embed_tokens_per_layer.weight"


class CpuPerLayerEmbedding(BaseOP):
    """Host-resident per-layer-embedding table (Gemma-4 E-series PLE). The table is huge
    (vocab x num_layers*ple_dim ~= 4.7 GB for E2B) but each step only gathers a handful of
    rows, so it stays in CPU RAM and the gather runs on the host -- freeing VRAM for the rest
    of the model. Only viable because the E-series runs with CUDA graphs disabled (the CPU
    gather + H2D copy cannot be captured in a graph)."""

    def __init__(self, num_embeddings: int, embedding_dim: int, embed_scale: float):
        # Built on the meta device; assign-load replaces this with the real CPU tensor
        # (engine keeps it on CPU via cpu_offloaded_weight_keys).
        self.weight = torch.empty(num_embeddings, embedding_dim)
        self._embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        idx = input_ids.to("cpu", dtype=torch.long)
        rows = F.embedding(idx, self.weight)  # [T, num_layers*ple_dim] on CPU
        return rows * self._embed_scale  # caller moves to the compute device


class Gemma4DecoderLayer(BaseOP):
    """Gemma 4 decoder block: attention sandwich + feed-forward sandwich, scaled by a
    per-layer ``layer_scalar``. The feed-forward is the dual (shared MLP || routed MoE)
    branch for MoE checkpoints, or a single dense MLP branch for dense checkpoints.

    Gemma-4 E-series (E2B/E4B) additionally: (1) KV-shared tail layers get a 2x-wide dense
    MLP (``use_double_wide_mlp``), and (2) every layer runs a Per-Layer-Embedding (PLE) step
    -- gate -> gelu -> multiply by the layer's per-layer input -> project -> norm -> residual
    -- inserted after the feed-forward sandwich and before the per-layer scale."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self.self_attn = Gemma4Attention(config, layer_id)

        self.has_ple = config.per_layer_hidden_size > 0
        if config.is_moe:
            self.feed_forward = Gemma4MLP(config, layer_id)
        else:
            # E-series: KV-shared layers use a double-wide MLP; PLE runs before layer_scalar,
            # so the dense MLP defers the scale to the decoder.
            double_wide = (
                config.use_double_wide_mlp and config.is_kv_shared_layer(layer_id)
            )
            inter = config.intermediate_size * (2 if double_wide else 1)
            self.feed_forward = Gemma4DenseMLP(
                config,
                intermediate_size=inter,
                apply_scalar=not self.has_ple,
            )

        eps = config.rms_norm_eps
        H = config.hidden_size
        self.input_layernorm = GemmaRMSNorm(H, eps=eps)
        self.post_attention_layernorm = GemmaRMSNorm(H, eps=eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(H, eps=eps)

        if self.has_ple:
            P = config.per_layer_hidden_size
            self.per_layer_input_gate = LinearReplicated(H, P, has_bias=False)
            self.per_layer_projection = LinearReplicated(P, H, has_bias=False)
            self.post_per_layer_input_norm = GemmaRMSNorm(H, eps=eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- attention sandwich ---
        residual = x
        h = self.input_layernorm.forward(x)
        h = self.self_attn.forward(h)
        h = self.post_attention_layernorm.forward(h)
        pre_ff, x = self.pre_feedforward_layernorm.forward_add_residual(h, residual)
        h = self.feed_forward.forward(pre_ff, x)
        if not self.has_ple:
            return h

        # --- Per-Layer-Embedding step (E-series), before the per-layer scale ---
        ple_in = get_global_ctx()._gemma4_ple[:, self._layer_id, :]
        g = self.per_layer_input_gate.forward(h)
        g = F.gelu(g, approximate="tanh")
        g = g * ple_in
        g = self.per_layer_projection.forward(g)
        g = self.post_per_layer_input_norm.forward(g)
        h = h + g
        return h * self.feed_forward.layer_scalar


class Gemma4Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            embed_scale=config.embedding_scale,
        )
        self.layers = OPList(
            [Gemma4DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._image_token_id = config.image_token_id

        # --- Gemma-4 E-series Per-Layer Embeddings (PLE) ---
        self._has_ple = config.per_layer_hidden_size > 0
        # PLE per-layer table (~4.7 GB bf16 for E2B). Default: an on-GPU embedding, which is
        # CUDA-graph-safe (the reason graphs are on by default). Opt in with
        # FREETOKEN_GEMMA4_PLE_CPU=1 to keep it in host RAM on a VRAM-tight box -- but the
        # per-forward host gather cannot be captured, so the engine then disables CUDA graphs
        # for this model (see Gemma4ForCausalLM.supports_cuda_graph). The GGUF path replaces
        # this with a compact Q6_K GGUFEmbedding on-GPU regardless (also graph-safe).
        self._ple_on_cpu = os.getenv("FREETOKEN_GEMMA4_PLE_CPU", "0").strip().lower() in (
            "1", "true", "yes", "on"
        )
        if self._has_ple:
            P = config.per_layer_hidden_size
            L = config.num_layers
            per_layer_embed_cls = (
                CpuPerLayerEmbedding if self._ple_on_cpu else VocabParallelEmbedding
            )
            self.embed_tokens_per_layer = per_layer_embed_cls(
                num_embeddings=config.per_layer_vocab_size,
                embedding_dim=L * P,
                embed_scale=float(P) ** 0.5,
            )
            self.per_layer_model_projection = LinearReplicated(
                config.hidden_size, L * P, has_bias=False
            )
            self.per_layer_projection_norm = GemmaRMSNorm(P, eps=config.rms_norm_eps)
            self._ple_proj_scale = float(config.hidden_size) ** -0.5
            self._ple_combine_scale = 2.0**-0.5
            self._ple_hidden = P
            self._num_layers = L

        # --- E-series KV-layer sharing: per-group stash buffers (lazy, fixed size) ---
        self._kv_shared = config.num_kv_shared_layers > 0
        self._kv_stash: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._stash_kv_dim: dict[str, int] = {}
        if self._kv_shared:
            first_shared = config.first_kv_shared_layer_idx()
            for layer_id in range(first_shared, config.num_layers):
                group = config.attention_group_for_layer(layer_id)
                self._stash_kv_dim[group.name] = group.num_kv_heads * group.head_dim
            # share the (initially empty) stash dict with every attention module
            for layer in self.layers.op_list:
                layer.self_attn._kv_stash = self._kv_stash

    def _ensure_stash(self, device: torch.device, dtype: torch.dtype) -> None:
        if not self._kv_shared or self._kv_stash:
            return
        for name, kv_dim in self._stash_kv_dim.items():
            k = torch.zeros(_STASH_MAX_TOKENS, kv_dim, device=device, dtype=dtype)
            v = torch.zeros(_STASH_MAX_TOKENS, kv_dim, device=device, dtype=dtype)
            self._kv_stash[name] = (k, v)

    def _merge_multimodal(self, input_ids: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Scatter precomputed image soft-token embeddings at image-token positions."""
        batch = get_global_ctx().batch
        mm_embeds = getattr(batch, "mm_embeds", None)
        if mm_embeds is None or self._image_token_id is None:
            return x
        mask = input_ids == self._image_token_id
        n_slots = int(mask.sum().item())
        assert n_slots == mm_embeds.shape[0], (
            f"image-token slots ({n_slots}) != vision features ({mm_embeds.shape[0]}); "
            "image tokens must not be split across prefill chunks"
        )
        return x.masked_scatter(mask.unsqueeze(-1), mm_embeds.to(x.dtype))

    def _compute_per_layer_inputs(
        self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        """PLE inputs [T, num_layers, ple_hidden]: token-identity embedding combined with a
        context projection of ``inputs_embeds``. Mirrors transformers' get_per_layer_inputs +
        project_per_layer_inputs."""
        T = input_ids.shape[0]
        # embed_tokens_per_layer takes input_ids and returns [T, L*P]: an on-GPU
        # VocabParallelEmbedding (default), a host-RAM CpuPerLayerEmbedding (opt-in), or a
        # Q6_K GGUFEmbedding after the GGUF swap. The .to() is a no-op for the on-GPU variants.
        tok = self.embed_tokens_per_layer.forward(input_ids).to(inputs_embeds.device).view(
            T, self._num_layers, self._ple_hidden
        )
        proj = self.per_layer_model_projection.forward(inputs_embeds) * self._ple_proj_scale
        proj = proj.view(T, self._num_layers, self._ple_hidden)
        proj = self.per_layer_projection_norm.forward(proj)
        return (proj + tok) * self._ple_combine_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        x = self._merge_multimodal(input_ids, x)
        if self._has_ple:
            self._ensure_stash(x.device, x.dtype)
            get_global_ctx()._gemma4_ple = self._compute_per_layer_inputs(input_ids, x)
        for layer in self.layers.op_list:
            x = layer.forward(x)
        return self.norm.forward(x)


class Gemma4ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Gemma4Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        self._final_logit_softcapping = config.final_logit_softcapping
        if config.is_multimodal:
            self.vision_tower = Gemma4VisionModel(config.vision_config)
            self.embed_vision = Gemma4MultimodalEmbedder(config.vision_config)
        super().__init__()

        # GGUF checkpoints carry native block-quantized weights: swap the dense
        # projections + embedding for GGUF-quant ops (experts stay on the offload cache).
        from .gguf import convert_gemma4_to_gguf, is_gguf_model

        if is_gguf_model(config):
            convert_gemma4_to_gguf(self, config)

    @property
    def supports_cuda_graph(self) -> bool:
        """False only when the PLE table is host-resident (FREETOKEN_GEMMA4_PLE_CPU=1): the
        per-forward host gather in CpuPerLayerEmbedding does a CPU<->GPU copy that CUDA graph
        capture forbids (and that graph replay could not re-run anyway). The default on-GPU /
        GGUF PLE embeddings are graph-safe, so graphs stay on by default."""
        return not getattr(self.model, "_ple_on_cpu", False)

    def cpu_offloaded_weight_keys(self) -> tuple[str, ...]:
        """Weights the engine should keep in host RAM instead of VRAM (E-series bf16 PLE
        table). Not needed for GGUF, where the table is a compact Q6_K GGUFEmbedding on GPU."""
        if self.model._has_ple and isinstance(
            self.model.embed_tokens_per_layer, CpuPerLayerEmbedding
        ):
            return (PER_LAYER_EMBED_KEY,)
        return ()

    @torch.inference_mode()
    def encode_images(
        self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor
    ) -> torch.Tensor:
        features = self.vision_tower.forward(pixel_values, image_position_ids)
        return self.embed_vision.forward(features)

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        if self._final_logit_softcapping is not None:
            cap = self._final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits


__all__ = ["Gemma4ForCausalLM"]
