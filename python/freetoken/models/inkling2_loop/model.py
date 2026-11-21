"""Inkling 2 Mini decoder: block attention residuals, the three-pass loop over layers 6-17 and the fp32 softcapped head.

Residual stream (as the reference): a layer returns the running sum of its block; at a block boundary (every
``block_size`` layers) the finished sum becomes a block state. Each sublayer input is a softmax over depth of
the RMS-normalized block states plus the running sum, with learned fp32 queries (three on the global layers,
for q, k and v). Pass 2 and 3 of the loop restart from the prelude's states; each loop layer first mixes in four
messages read from the previous pass by learned queries, through low-rank fp32 gates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F

from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearReplicated, OPList, ParallelLMHead, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Inkling2Attention, unweighted_rms_norm
from .config import CONV_STATES, Inkling2LoopArgs
from .conv import ConvMetadata, build_conv_metadata, short_conv
from .moe import Inkling2MoE

if TYPE_CHECKING:
    from freetoken.message import MMItem
    from freetoken.models.config import ModelConfig

# the reference's depth-attention norm uses its own default eps, not the config's
_DEPTH_EPS = 1e-6


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Weighted RMSNorm computed in fp32; the caller picks the output dtype."""
    x = x.float()
    return weight * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps))


def depth_attention(sources: List[torch.Tensor], queries: torch.Tensor) -> torch.Tensor:
    """``[T, Q, H]`` fp32: per token, softmax over the RMS-normalized ``sources`` [T, H] (fp32, already normalized)
    of ``query . source / sqrt(H)``, the weighted sum of the normalized sources."""
    hidden = sources[0].shape[-1]
    logits = torch.stack([s @ queries.t() for s in sources], dim=-1) * hidden**-0.5  # [T, Q, N]
    p = torch.softmax(logits, dim=-1)
    out = p[..., 0, None] * sources[0][:, None, :]
    for j in range(1, len(sources)):
        out = out + p[..., j, None] * sources[j][:, None, :]
    return out


def normalized(x: torch.Tensor) -> torch.Tensor:
    return unweighted_rms_norm(x, _DEPTH_EPS)


class MessageGate(BaseOP):
    """``(1 - tanh(f)) * h + tanh(g) * m`` with ``f``, ``g`` fp32 projections of a bf16 low-rank mix of norm(h), norm(m)."""

    def __init__(self, config: ModelConfig, *, prefix: str = ""):
        args: Inkling2LoopArgs = config.inkling2_args
        H, rank = args.hidden_size, args.gate_rank
        self.down_proj = LinearReplicated(2 * H, rank, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.down_proj")
        self.forget_proj = torch.empty(H, rank, dtype=torch.float32)
        self.message_proj = torch.empty(H, rank, dtype=torch.float32)
        self._eps = args.rms_norm_eps

    def forward(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        h, m = h.float(), m.float()
        mix = torch.cat([unweighted_rms_norm(h, self._eps), unweighted_rms_norm(m, self._eps)], dim=-1)
        mix = self.down_proj.forward(mix.to(self.down_proj.weight.dtype)).float()
        forget = torch.tanh(F.linear(mix, self.forget_proj))
        gain = torch.tanh(F.linear(mix, self.message_proj))
        return (1 - forget) * h + gain * m


class Inkling2DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        args: Inkling2LoopArgs = config.inkling2_args
        self._args = args
        self.layer_id = layer_id
        self.is_sliding = args.is_sliding(layer_id)
        self.is_block_boundary = layer_id % args.block_size == 0
        H, K = args.hidden_size, args.conv_kernel
        # layer 0 reads the embeddings directly
        self.attention_query = None if layer_id == 0 else torch.empty(1 if self.is_sliding else 3, H, dtype=torch.float32)
        self.mlp_query = torch.empty(1, H, dtype=torch.float32)
        if self.is_sliding:
            self.input_layernorm = torch.empty(H, dtype=torch.float32)
        else:
            self.q_input_layernorm = torch.empty(H, dtype=torch.float32)
            self.k_input_layernorm = torch.empty(H, dtype=torch.float32)
            self.v_input_layernorm = torch.empty(H, dtype=torch.float32)
        self.post_attention_layernorm = torch.empty(H, dtype=torch.float32)
        self.attn_conv = torch.empty(K, H, dtype=torch.float32)
        self.mlp_conv = torch.empty(K, H, dtype=torch.float32)
        self.self_attn = Inkling2Attention(config, layer_id, prefix=f"{prefix}.self_attn")
        self.mlp = Inkling2MoE(config, layer_id, prefix=f"{prefix}.mlp")
        if args.is_loop_layer(layer_id):
            self.q_message_gate = MessageGate(config, prefix=f"{prefix}.q_message_gate")
            self.k_message_gate = MessageGate(config, prefix=f"{prefix}.k_message_gate")
            self.v_message_gate = MessageGate(config, prefix=f"{prefix}.v_message_gate")
            self.mlp_message_gate = MessageGate(config, prefix=f"{prefix}.mlp_message_gate")

    def _attn_inputs(self, h: torch.Tensor, blocks: List[torch.Tensor], messages: torch.Tensor | None):
        eps, dt = self._args.rms_norm_eps, h.dtype
        if self.attention_query is None:
            heads = [h] * (1 if self.is_sliding else 3)
        else:
            mixed = depth_attention(blocks + [normalized(h)], self.attention_query)
            heads = list(mixed.unbind(1))
        if self.is_sliding:
            if messages is None:
                x = rms_norm(heads[0], self.input_layernorm, eps).to(dt)
                return x, x, x
            gates = (self.q_message_gate, self.k_message_gate, self.v_message_gate)
            return tuple(rms_norm(g.forward(heads[0], messages[:, i]), self.input_layernorm, eps).to(dt) for i, g in enumerate(gates))
        if messages is not None:
            gates = (self.q_message_gate, self.k_message_gate, self.v_message_gate)
            heads = [g.forward(x, messages[:, i]) for i, (g, x) in enumerate(zip(gates, heads))]
        norms = (self.q_input_layernorm, self.k_input_layernorm, self.v_input_layernorm)
        return tuple(rms_norm(x, w, eps).to(dt) for x, w in zip(heads, norms))

    @nvtx_annotate("Layer_{}", layer_id_field="layer_id")
    def forward(
        self, h: torch.Tensor, blocks: List[torch.Tensor], messages: torch.Tensor | None, cache_layer: int, meta: ConvMetadata
    ) -> torch.Tensor:
        """``h`` [T, H] the running block sum (bf16), ``blocks`` the normalized block states; returns the new running sum."""
        pool = get_global_ctx().linear_state_pool
        args, dt = self._args, h.dtype

        def stash(name: str):
            return pool.verify_handler(name).stash(cache_layer) if meta.verify else None

        xq, xk, xv = self._attn_inputs(h, blocks, messages)
        a = self.self_attn.forward(xq, xk, xv, cache_layer, meta)
        a = short_conv(a, self.attn_conv, pool.slot_state(CONV_STATES[2], cache_layer), meta, verify_stash=stash(CONV_STATES[2]))
        residual = a.float()
        if self.is_block_boundary:
            blocks = blocks + [normalized(h)]
        else:
            residual = h.float() + residual
        m = depth_attention(blocks + [normalized(residual)], self.mlp_query)[:, 0]
        if messages is not None:
            m = self.mlp_message_gate.forward(m, messages[:, 3])
        y = self.mlp.forward(rms_norm(m, self.post_attention_layernorm, args.rms_norm_eps).to(dt))
        y = short_conv(y, self.mlp_conv, pool.slot_state(CONV_STATES[3], cache_layer), meta, verify_stash=stash(CONV_STATES[3]))
        return (residual + y.float()).to(dt)


class Inkling2LoopModel(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        args: Inkling2LoopArgs = config.inkling2_args
        self._args = args
        H = args.hidden_size
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=H, quant_config=config.quant, prefix=f"{prefix}.embed_tokens"
        )
        self.embed_norm = torch.empty(H, dtype=torch.float32)
        self.layers = OPList([Inkling2DecoderLayer(config, i, prefix=f"{prefix}.layers.{i}") for i in range(args.num_layers)])
        # per loop layer: the query, key, value and MLP message queries
        self.message_queries = torch.empty(args.num_loop_layers, 4, H, dtype=torch.float32)
        self.final_query = torch.empty(1, H, dtype=torch.float32)
        self.norm = torch.empty(H, dtype=torch.float32)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch
        ids = input_ids if batch.mm_embeds is None else input_ids.clamp(max=self.embed_tokens.num_embeddings - 1)
        x = self.embed_tokens.forward(ids)
        x = rms_norm(x, self.embed_norm, self._args.rms_norm_eps).to(x.dtype)
        if batch.mm_embeds is not None:
            # the image features replace the normalized token embeddings, as in the reference
            x.index_copy_(0, batch.mm_rows, batch.mm_embeds[:, : x.shape[1]].to(x.dtype))
        return x

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        args = self._args
        batch = get_global_ctx().batch
        meta = build_conv_metadata(batch, args.conv_kernel, input_ids.device)
        features = batch.features
        wanted = features.wanted_layers if features is not None else frozenset()
        layers = self.layers.op_list

        def run(layer_id: int, pass_idx: int, h, blocks, messages=None):
            layer = layers[layer_id]
            out = layer.forward(h, blocks, messages, args.cache_layer(layer_id, pass_idx), meta)
            if layer.is_block_boundary:
                blocks = blocks + [normalized(h)]
            access = args.access(layer_id, pass_idx)
            if access in wanted:
                features.layer_output(access, out)
            return out, blocks

        h = self.embed(input_ids)
        blocks: List[torch.Tensor] = []
        for i in range(args.loop_start):
            h, blocks = run(i, 0, h, blocks)
        pre_h, pre_blocks = h, blocks
        for i in range(args.loop_start, args.loop_end):
            h, blocks = run(i, 0, h, blocks)
        for p in range(1, args.num_passes):
            # the previous pass's states: the blocks it closed after the first loop boundary, then its running sum
            previous = blocks[len(pre_blocks) + 1 :] + [normalized(h)]
            h, blocks = pre_h, pre_blocks
            for j, i in enumerate(range(args.loop_start, args.loop_end)):
                messages = depth_attention(previous, self.message_queries[j])
                h, blocks = run(i, p, h, blocks, messages)
        for i in range(args.loop_end, args.num_layers):
            h, blocks = run(i, 0, h, blocks)
        out = depth_attention(blocks + [normalized(h)], self.final_query)[:, 0]
        return rms_norm(out, self.norm, args.rms_norm_eps).to(h.dtype)


class Inkling2LoopForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        args: Inkling2LoopArgs = config.inkling2_args
        self.model = Inkling2LoopModel(config)
        # the checkpoint pads the unembedding past the tokenizer; those rows never sample
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size, embedding_dim=args.hidden_size, quant_config=config.quant, prefix="lm_head"
        )
        self._softcap = config.final_logit_softcapping
        self._unpadded = args.unpadded_vocab_size
        # the offload prefill streams experts in this order: each MoE layer is keyed by its weight layer
        self.moe_layer_order = tuple(layer for layer, _ in args.schedule())
        super().__init__()

    @nvtx_annotate("Inkling2Loop")
    def forward(self) -> torch.Tensor:
        h = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(h).float()
        logits = torch.tanh(logits / self._softcap) * self._softcap
        logits[:, self._unpadded :] = float("-inf")
        return logits


class Inkling2LoopForConditionalGeneration(Inkling2LoopForCausalLM):
    """Text decoder plus the vision tower (``visual``); the tower's features replace the image-token embeddings."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.is_multimodal:
            from .vision import Inkling2VisionModel

            # state-dict names visual.*, the checkpoint's model.visual.* (weight.iter_weights renames)
            self.visual = Inkling2VisionModel(config.vision_config)

    def place_encoder_weights(self, mode: str) -> None:
        self.visual.place_weights(mode)

    def encode(self, item: MMItem) -> torch.Tensor:
        """``[rows * cols, hidden]`` image features, one per image token (``item.grid_thw`` is ``[1, rows, cols]``)."""
        feature = item.feature.to(self.visual.proj_in.weight.device, non_blocking=True)
        return self.visual.forward(feature, item.grid_thw)


__all__ = [
    "Inkling2DecoderLayer",
    "Inkling2LoopForCausalLM",
    "Inkling2LoopForConditionalGeneration",
    "Inkling2LoopModel",
    "MessageGate",
    "depth_attention",
    "rms_norm",
]
