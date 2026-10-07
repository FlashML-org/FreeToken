"""DeepSeek-V4.1-Flash model (engine-native port of ``inference/model.py``).

    embed -> hc_mult residual streams -> [Engram at 1, 14] -> 40 blocks (single-pass mHC around
    DSV41 attention and MoE) -> collapse with the last block's pre gates -> norm -> head

Single-pass mHC (``layers/mhc.py``): each sublayer boundary applies the PREVIOUS sublayer's post /
comb to the streams, predicts this sublayer's (pre, post, comb) from the updated streams, and mixes
the sublayer input with the pre gates the previous sublayer predicted. The chain state
``(y, residual, post, comb, pre)`` threads through the layer loop; layer 0's attention starts it
with a one-hot pre on stream 0 and no pending post.

Causal Encoder-Decoder: the decoder's kv source (layer 20) compresses ITS INPUT -- the final encoder
hidden state -- into the global KV every decoder layer reads. Under Decoder SWA Bounded Replay the
decoder layers run on each request's last ``window_size`` prompt tokens only (their sliding window
truncated at the start of that window, see the backend; a prefix hit stops before it, so those tokens
are always prefilled); layer 20 still publishes global KV for every prompt token.
``--swa-decoder-replay exact`` runs the decoder on every token (the reference numerics).

KV addressing is the attention backend's; pool buffers are read off the live pool per access, so a
runtime cache rebuild needs no unbind. Decode is batched and CUDA-graph safe (the DSV4 precedent).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple

import torch
from freetoken.attention.dsv41_sparse import DSV41AttnMetadata, PrefillSegment
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNorm, VocabParallelEmbedding
from freetoken.layers.mhc import mhc_fused_post_pre_single_pass, mhc_mix_input, mhc_post
from freetoken.models.blocks import BaseLLMModel, embed_input_ids

from .args import DeepseekV41Args
from .attention import DSV41Attention, DecodeStepContext
from .engram import EngramLayer
from .moe import MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

POST_MULT = 2.0  # post gates are 2 * sigmoid (reference hc_split_sinkhorn)


class Streams(NamedTuple):
    """The single-pass mHC chain state between sublayers: the last sublayer output ``y`` with its
    not-yet-applied ``post`` / ``comb``, the residual streams ``[T, hc, dim]`` they apply to, and
    the ``pre`` gates ``[T, hc]`` the next sublayer mixes its input with."""

    y: torch.Tensor | None
    residual: torch.Tensor
    post: torch.Tensor | None
    comb: torch.Tensor | None
    pre: torch.Tensor

    def materialize(self) -> torch.Tensor:
        """The streams with the pending post applied."""
        if self.post is None:
            return self.residual
        return mhc_post(self.y, self.residual, self.post, self.comb)


class Block(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        args: DeepseekV41Args = config.dsv41_args
        self.layer_id = layer_id
        self.role = args.roles[layer_id]
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.sinkhorn = args.hc_sinkhorn_iters
        hc, dim = args.hc_mult, args.dim
        mix = (2 + hc) * hc
        self.attn = DSV41Attention(args, self.role, quant_config=config.quant, prefix=f"{prefix}.attn")
        self.ffn = MoE(config, layer_id, prefix=f"{prefix}.ffn")
        self.attn_norm = RMSNorm(dim, args.norm_eps)
        self.ffn_norm = RMSNorm(dim, args.norm_eps)
        self.engram = EngramLayer(args, layer_id, quant_config=config.quant, prefix=f"{prefix}.engram") if layer_id in args.engram_layer_ids else None
        self.hc_attn_fn = torch.empty(mix, hc * dim, dtype=torch.float32)
        self.hc_ffn_fn = torch.empty(mix, hc * dim, dtype=torch.float32)
        self.hc_attn_base = torch.empty(mix, dtype=torch.float32)
        self.hc_ffn_base = torch.empty(mix, dtype=torch.float32)
        self.hc_attn_scale = torch.empty(3, dtype=torch.float32)
        self.hc_ffn_scale = torch.empty(3, dtype=torch.float32)

    def _boundary(self, s: Streams, fn, scale, base):
        residual, post, comb, pre_next, x = mhc_fused_post_pre_single_pass(
            s.y if s.y is not None else s.residual[:, 0], s.residual, s.post, s.comb, s.pre, fn, scale, base,
            self.norm_eps, self.hc_eps, POST_MULT, self.sinkhorn,
        )
        return residual, post, comb, pre_next, x

    def attn_input(self, s: Streams, *, image_mask: torch.Tensor | None = None):
        """Apply the pending post, predict the attention gates, mix and norm the attention input.
        Returns ``(residual, post, comb, pre_next, x)``."""
        if self.engram is not None:
            s = Streams(None, self.engram.forward(s.materialize(), image_mask=image_mask), None, None, s.pre)
        residual, post, comb, pre_next, x = self._boundary(s, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        return residual, post, comb, pre_next, self.attn_norm.forward(x)

    def ffn_step(self, y: torch.Tensor, residual, post, comb, pre, *, image_mask: torch.Tensor | None = None) -> Streams:
        """The FFN sublayer after the attention output ``y``: boundary, norm, MoE; returns the chain state."""
        residual, post, comb, pre_next, x = self._boundary(Streams(y, residual, post, comb, pre), self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        return Streams(self.ffn.forward(self.ffn_norm.forward(x), image_mask=image_mask), residual, post, comb, pre_next)

    def forward_prefill(self, s: Streams, segments: List[PrefillSegment], positions: torch.Tensor, image_mask: torch.Tensor | None = None) -> Streams:
        residual, post, comb, pre_next, x = self.attn_input(s, image_mask=image_mask)
        if self.attn.compressor is not None:
            self.attn.publish_prefill(x, segments)
        y = self.attn.forward_prefill(x, segments, positions)
        return self.ffn_step(y, residual, post, comb, pre_next, image_mask=image_mask)

    def forward_prefill_narrowed(
        self, s: Streams, segments: List[PrefillSegment], rows: torch.Tensor, query_segments: List[PrefillSegment],
        query_positions: torch.Tensor, image_mask: torch.Tensor | None = None, query_image_mask: torch.Tensor | None = None,
    ) -> Streams | None:
        """``forward_prefill`` with a row cut after the global KV is published: the attention input and
        ``publish_prefill`` cover every token of ``segments``, the attention and the FFN only ``rows``
        (tiled by ``query_segments``) -- the CED decoder's kv source under bounded replay. None when no
        query segment is left (every chunk in the batch ends before its prompt's last window)."""
        residual, post, comb, pre_next, x = self.attn_input(s, image_mask=image_mask)
        if self.attn.compressor is not None:
            self.attn.publish_prefill(x, segments)
        if not query_segments:
            return None
        y = self.attn.forward_prefill(x[rows], query_segments, query_positions)
        return self.ffn_step(y, residual[rows], post[rows], comb[rows], pre_next[rows], image_mask=query_image_mask)

    def forward_decode(self, s: Streams, pos: torch.Tensor, rows: torch.Tensor, dctx, cmp_stage_cap: int) -> Streams:
        residual, post, comb, pre_next, x = self.attn_input(s)
        y = self.attn.forward_decode(x, pos, rows, dctx, cmp_stage_cap)
        return self.ffn_step(y, residual, post, comb, pre_next)


class Transformer(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        args: DeepseekV41Args = config.dsv41_args
        self.args = args
        self.embed = VocabParallelEmbedding(num_embeddings=args.vocab_size, embedding_dim=args.dim)
        self.layers = OPList([Block(config, i, prefix=f"{prefix}.layers.{i}") for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)

    def bind(self, device: torch.device) -> None:
        tables: dict = {}  # one RoPE table per distinct parameter set, shared by the layers using it
        for block in self.layers.op_list:
            block.attn.bind(device, tables)

    def _entry(self, input_ids: torch.Tensor) -> Streams:
        h = embed_input_ids(self.embed, input_ids, get_global_ctx().batch)
        residual = h.unsqueeze(1).expand(-1, self.args.hc_mult, -1).contiguous()
        pre = torch.zeros(h.shape[0], self.args.hc_mult, dtype=torch.float32, device=h.device)
        pre[:, 0] = 1.0
        return Streams(None, residual, None, None, pre)

    def _exit(self, s: Streams) -> torch.Tensor:
        return self.norm.forward(mhc_mix_input(s.materialize(), s.pre))

    def prefill(self, input_ids: torch.Tensor, positions: torch.Tensor, md: DSV41AttnMetadata) -> torch.Tensor:
        """Ragged prefill over the flat new-token stream; returns the hidden states the head reads (the
        decoder stream under bounded replay, where ``md.last_indices`` also point, plus the placeholder
        row ``md.decoder_pad`` asks for)."""
        s = self._entry(input_ids)
        dtype = s.residual.dtype
        image_mask = input_ids >= self.args.vocab_size if get_global_ctx().batch.mm_embeds is not None else None
        segments, dec_segments, rows = md.segments, md.decoder_segments, md.decoder_rows
        decoder_start = self.args.decoder_start_layer
        if rows is not None:
            dec_positions = positions[rows]
            dec_image_mask = None if image_mask is None else image_mask[rows]
        for block in self.layers.op_list:
            if rows is None or block.layer_id < decoder_start:
                s = block.forward_prefill(s, segments, positions, image_mask)
            elif block.layer_id == decoder_start:
                s = block.forward_prefill_narrowed(s, segments, rows, dec_segments, dec_positions, image_mask, dec_image_mask)
                if s is None:
                    break
            else:
                s = block.forward_prefill(s, dec_segments, dec_positions, dec_image_mask)
        pad = [torch.zeros(1, self.args.dim, dtype=dtype, device=input_ids.device)] if md.decoder_pad else []
        return torch.cat([self._exit(s)] + pad) if s is not None else pad[0]

    def decode(self, input_ids: torch.Tensor, pos: torch.Tensor, md: DSV41AttnMetadata, cmp_stage_cap: int) -> torch.Tensor:
        B = input_ids.shape[0]
        rows = torch.arange(B, device=input_ids.device)
        s = self._entry(input_ids)
        # the layer-invariant step context, resolved once (a capture records it once): ring slots and
        # window candidates of the shared pool -- and of the private rings when bounded replay keeps
        # the decoder's window KV per request -- plus each RoPE table at the step's positions
        dctx = DecodeStepContext.build([b.attn for b in self.layers.op_list], md, pos, rows, self.args.window_size, self.args.index_topk)
        for block in self.layers.op_list:
            s = block.forward_decode(s, pos, rows, dctx, cmp_stage_cap)
        return self._exit(s)


class DeepseekV41ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self._config = config
        self._args: DeepseekV41Args = config.dsv41_args
        self.model = Transformer(config)
        self.head = ParallelLMHead(num_embeddings=self._args.vocab_size, embedding_dim=self._args.dim, quant_config=config.quant, prefix="head")
        self._bound = False
        if config.is_multimodal:
            from .vision import DeepseekV41Vision

            self.visual = DeepseekV41Vision(config.vision_config, config.hidden_size,
                                           num_layers=self._args.n_layers, num_experts=self._args.n_routed_experts)

    def encode(self, item):
        return self.visual.forward(item)

    def place_encoder_weights(self, mode: str) -> None:
        self.visual.place_weights(mode)

    def _ensure_bound(self) -> None:
        if not self._bound:
            self.model.bind(get_global_ctx().kv_cache.device)
            if self._config.is_multimodal:
                for i, block in enumerate(self.model.layers.op_list):
                    block.ffn.gate._bias_vl = self.visual.routing_bias[i]
            self._bound = True

    def mark_for_rebind(self) -> None:
        """A runtime pool rebuild changes nothing the model caches (rope tables are pool-independent),
        but the engine asks; re-bind is cheap and keeps the contract with the DSV4 sibling."""
        self._bound = False

    def engram_layers(self) -> List[EngramLayer]:
        return [b.engram for b in self.model.layers.op_list if b.engram is not None]

    def load_host_tables(self, engine_config) -> int:
        """Attach the Engram tables (streamed from the checkpoint shards; zeros for dummy weights)
        and take over ``forward_host_ctx`` so every dispatch stages its rows. Returns the pinned
        host bytes the engine reserves from its pin budget."""
        import os

        layers = self.engram_layers()
        if not layers:
            return 0
        device = torch.device("cuda", torch.cuda.current_device())
        if getattr(engine_config, "use_dummy_weight", False):
            from .engram import ZeroEngramTable

            for layer in layers:
                layer.attach_table(ZeroEngramTable(layer.width, device))
            return 0
        from freetoken.utils import download_hf_weight
        from transformers import AutoTokenizer

        from .engram import EngramHash, build_compressed_token_map
        from .engram_table import EngramDiskTable, EngramHost, engram_row_source

        folder = download_hf_weight(engine_config.model_path)
        token_map, compressed_vocab = build_compressed_token_map(AutoTokenizer.from_pretrained(folder, trust_remote_code=True))
        hash = EngramHash(self._args, token_map, compressed_vocab)
        graph_rows = max(engine_config.max_running_req, engine_config.cuda_graph_max_bs or 0, 1)
        tables = [
            EngramDiskTable(
                engram_row_source(folder, layer.layer_id), hash.n_cols, device,
                max_graph_rows=graph_rows,
                max_extend_tokens=engine_config.max_extend_tokens,
                use_io_uring=os.getenv("FREETOKEN_ENGRAM_IO_URING", "1") != "0",
            )
            for layer in layers
        ]
        for layer, table in zip(layers, tables):
            layer.attach_table(table)
        self._engram_host = EngramHost(hash, tables, device, sync_mode=os.getenv("FREETOKEN_ENGRAM_SYNC", "auto"))
        self.forward_host_ctx = self._engram_host.forward_host_ctx
        return sum(t._graph_pinned.numel() + sum(b.numel() for b in t._eager_pinned) for t in tables)

    def forward(self) -> torch.Tensor:
        self._ensure_bound()
        batch = get_global_ctx().batch
        md = batch.attn_metadata
        assert isinstance(md, DSV41AttnMetadata)
        if batch.is_prefill:
            hidden = self.model.prefill(batch.input_ids, batch.positions.long(), md)
        else:
            input_ids = batch.input_ids
            B = batch.padded_size
            pos = batch.positions.long().view(-1)[:B]
            if torch.cuda.is_current_stream_capturing():
                cmp_stage_cap = md.stage_width - 1
            else:
                cmp_stage_cap = int(pos.max().item())
            hidden = self.model.decode(input_ids.view(B), pos, md, cmp_stage_cap)
        return self.head.forward(hidden)


__all__ = ["DeepseekV41ForCausalLM", "Transformer", "Block", "Streams"]
