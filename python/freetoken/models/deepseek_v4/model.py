"""DeepSeek-V4-Flash model (engine-native FreeToken port of inference/model.py).

A faithful single-stream port of the reference (MLA attention with a sliding window
+ stateful KV compressors / Lightning Indexer, manifold-constrained Hyper-Connections,
sqrtsoftplus / hash MoE), wired onto FreeToken's shared paged engine:

  - KV lives in DSV4-owned paged pools (:class:`~freetoken.kvcache.dsv4_paged_pool.DSV4PagedKVCache`):
    each layer's window-ring + compressed KV is a region of a shared global pool, addressed by
    per-layer slot maps. Sparse attention is a PAGED physical-slot gather:
    ``sparse_attn_paged`` reads KV directly from the two global pools (window / compressed) at
    the GLOBAL top-k slots inside the kernel, with no per-forward staging ``index_select``.
  - The model is a registered :class:`BaseLLMModel` built by ``create_model``; it is driven
    by ``ctx.batch`` (positions / input_ids), captured by the engine ``GraphRunner`` for
    decode, and freed by dropping page-table indices (no separate runner / python cursor).
  - Routed FP4 experts are served from :class:`OffloadMoeCache` (on-demand) -- the
    framework's core acceleration.

Heavy ops are FreeToken Triton kernels. Precision matches the reference (FP8/FP4
activation quant + Hadamard rotation re-introduced; see ``ops.py`` / the dsv4 kernels).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator
from freetoken.kernel.triton.dsv4.hc import hc_post_combine, hc_pre_combine
from freetoken.kernel.triton.dsv4.sinkhorn import hc_split_sinkhorn
from freetoken.models.blocks import BaseLLMModel

from .args import DeepseekV4Args
from .attention import Attention
from .moe import MoE

# Re-exports: keep every class/helper previously defined here importable from .model
# (external import stability; the moved definitions live in their own modules).
from .compress import Compressor, Indexer  # noqa: F401
from .layers import (  # noqa: F401
    FP8,
    Linear,
    RMSNorm,
    get_compress_topk_idxs,
    get_window_topk_idxs,
)
from .moe import Expert, Gate  # noqa: F401
from .parallel import div_tp, tp_info, validate_tp


class Block(nn.Module):
    """Decoder block with manifold-constrained Hyper-Connections (4 residual streams)."""

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.dim = args.dim
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False)
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False)
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32), requires_grad=False)
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32), requires_grad=False)
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32), requires_grad=False)
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32), requires_grad=False)

    def hc_pre(self, x, hc_fn, hc_scale, hc_base):
        shape, dtype = x.size(), x.dtype
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(xf, hc_fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(
            mixes.view(-1, mixes.size(-1)), hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )
        M = shape[0] * shape[1]
        y = hc_pre_combine(xf.view(M, self.hc_mult, self.dim), pre, dtype).view(*shape[:2], self.dim)
        return y, post.view(M, self.hc_mult), comb.view(M, self.hc_mult, self.hc_mult)

    def hc_post(self, x, residual, post, comb):
        shape = residual.size()
        M = shape[0] * shape[1]
        y = hc_post_combine(
            x.reshape(M, self.dim), residual.reshape(M, self.hc_mult, self.dim), post, comb
        )
        return y.view(shape)

    def prefill_batched(self, x, input_ids, segments, flat_positions):
        # Ragged batched prefill (cu_seqlens, no padding; bs >= 1, cold and radix-hit segments
        # mixed freely). ``x`` is [1, T, hc_mult, dim] -- the requests' token streams
        # concatenated. Per-token ops (HC / norm / MoE) run batched over ALL T tokens (the
        # MoE-offload amortization win); attention runs as ONE flat cu_seqlens launch over all
        # T queries (Attention.forward_ragged), with the stateful compressor/indexer looped per
        # request. ``segments`` = [(offset, extend_len, table_idx, start_pos)] off the attention
        # metadata; ``flat_positions`` [T] = per-token ABSOLUTE position (batch.positions).
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn.forward_ragged(x, segments, flat_positions)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x

    def decode_step(self, x, pos, rows, cmp_stage_cap, input_ids, wctx=None):
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn.decode_step(x, pos, rows, cmp_stage_cap, wctx)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


class Transformer(nn.Module):
    def __init__(self, args: DeepseekV4Args):
        super().__init__()
        # Check every tensor-parallel split before building a single layer, so a bad
        # --tensor-parallel-size fails with one clear message, not a reshape deep in a
        # forward. embed / head / norm / hyper-connections stay replicated.
        validate_tp(args)
        self.args = args
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.hc_mult = hc_mult = args.hc_mult
        # Vocabulary parallelism. The embedding table and the output head are the two
        # largest replicated tensors (3.0 GiB together), so each rank keeps one
        # contiguous vocabulary block: the embedding all-reduces its masked lookup, the
        # head all-gathers its logit slice.
        rank, tp = tp_info()
        self.tp_size = tp
        self.vocab_size = args.vocab_size
        vocab_local = div_tp(args.vocab_size, "vocab_size")
        self.vocab_start = rank * vocab_local
        self.vocab_local = vocab_local
        self._comm = DistributedCommunicator() if tp > 1 else None
        self.embed = nn.Embedding(vocab_local, args.dim)
        self.embed.weight.requires_grad_(False)
        self.layers = nn.ModuleList([Block(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, self.norm_eps)
        self.head = nn.Parameter(torch.empty(vocab_local, args.dim, dtype=torch.bfloat16), requires_grad=False)
        hc_dim = hc_mult * args.dim
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32), requires_grad=False)
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32), requires_grad=False)
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)

    def bind(self, pool, device: torch.device) -> None:
        for layer in self.layers:
            layer.attn.bind(pool, device)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Vocabulary-parallel lookup. Rows outside this rank's block contribute zero, so
        the all-reduce sums exactly one real row per token. Index arithmetic only -- no
        host sync, so a captured decode graph replays it."""
        if self._comm is None:
            return self.embed(input_ids)
        local = input_ids - self.vocab_start
        outside = (local < 0) | (local >= self.vocab_local)
        y = self.embed(local.masked_fill(outside, 0))
        y = y.masked_fill(outside.unsqueeze(-1), 0)
        return self._comm.all_reduce(y)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """Head projection over this rank's vocabulary block, then an all-gather back to
        the full [rows, vocab]. all_gather concatenates on dim 0, so the gathered tensor
        is rank-major: one row block per rank, which is the vocabulary order."""
        local = F.linear(h, self.head)  # [rows, vocab_local]
        if self._comm is None:
            return local
        rows = local.shape[0]
        gathered = self._comm.all_gather(local)  # [tp*rows, vocab_local]
        if rows == 1:
            return gathered.view(1, -1)[:, : self.vocab_size]
        gathered = gathered.view(self.tp_size, rows, self.vocab_local)
        gathered = gathered.permute(1, 0, 2).reshape(rows, self.tp_size * self.vocab_local)
        return gathered[:, : self.vocab_size]

    def hc_head(self, x):
        shape, dtype = x.size(), x.dtype
        dim = self.args.dim
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(xf, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        M = shape[0] * shape[1]
        return hc_pre_combine(xf.view(M, self.hc_mult, dim), pre.view(M, self.hc_mult), dtype).view(*shape[:2], dim)

    def prefill_batched(
        self, input_ids: torch.Tensor, segments, flat_positions: torch.Tensor,
        last_indices: torch.Tensor,
    ) -> torch.Tensor:
        # Ragged batched prefill (bs >= 1). ``input_ids`` is [1, T] -- the requests' NEW tokens
        # concatenated (cu_seqlens, no padding); each request starts at its own cached_len
        # (cold == 0, radix hit / chunk continuation > 0). Per-token ops run batched over all T
        # tokens; attention runs per request on its [offset, offset+n) segment (see Block). Each
        # request's window/cmp/idx slots + ring blocks were allocated disjointly by
        # allocate_paged, so the per-request attention never reads another request's KV/carry.
        # ``segments`` [(offset, extend_len, table_idx, start_pos)] comes off the attention
        # metadata; ``flat_positions`` [T] is the scheduler-staged batch.positions (per-token
        # ABSOLUTE position); ``last_indices`` [B] is the flattened index of each request's
        # final token -> its next-token logits row.
        h = self.embed_tokens(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        for layer in self.layers:
            h = layer.prefill_batched(h, input_ids, segments, flat_positions)
        h = self.hc_head(h)
        h = self.norm(h)
        return self.logits(h[0, last_indices])  # [B, vocab]

    def decode(
        self, input_ids: torch.Tensor, pos: torch.Tensor, cmp_stage_cap: int
    ) -> torch.Tensor:
        # input_ids [B,1], pos [B] (GPU int). cmp_stage_cap = max position any row reaches; each
        # layer derives its compressed staging width = (cmp_stage_cap+1)//ratio (max valid count
        # over rows in eager; a static capture width = max_seq-1 under graph).
        #
        # Overlap safety: the global page-table rows are NOT read inside the graph. The attention
        # metadata carries a snapshot of the active rows' whole-history full locs (staged before a
        # replay); the decode derives every window/cmp/idx read slot from that snapshot IN-GRAPH by
        # LOCAL row ``rows`` = arange(B). So the next batch's allocate_paged cannot corrupt this
        # in-flight replay (it mutates only the live map).
        B = input_ids.size(0)
        rows = torch.arange(B, device=input_ids.device)
        h = self.embed_tokens(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        # Hoist the layer-invariant per-step decode tensors (shared window-ring global slots):
        # resolved ONCE off the attention metadata (recomputed per call, so a capture records
        # the gathers -- never cache it on the metadata) and threaded
        # into every layer. They read only the shared snapshot / positions, so they are identical
        # across layers.
        wctx = get_global_ctx().batch.attn_metadata.window_ctx(pos, rows)
        for layer in self.layers:
            h = layer.decode_step(h, pos, rows, cmp_stage_cap, input_ids, wctx)
        h = self.hc_head(h)
        h = self.norm(h)
        return self.logits(h[:, -1])


class DeepseekV4ForCausalLM(BaseLLMModel):
    """Engine adapter: a registered :class:`BaseLLMModel` wrapping the DSV4 transformer.

    Built on the engine ``meta`` device then loaded via :meth:`load_state_dict` (which
    casts each incoming tensor to its parameter dtype and assigns it). KV pools, rope
    constants and compressor state are bound on the first forward.
    """

    def __init__(self, config):
        self._config = config
        self._args: DeepseekV4Args = config.dsv4_args
        self._transformer = Transformer(self._args)
        self._bound = False

    def _ensure_bound(self) -> None:
        if self._bound:
            return
        pool = get_global_ctx().kv_cache
        self._transformer.bind(pool, pool.device)
        self._bound = True

    def mark_for_rebind(self) -> None:
        """Force a re-bind on the next forward. The model holds NO pool reference -- buffers are read
        off ctx.kv_cache via @property -- so a runtime rebuild needs no unbind; the old pool frees
        when the engine drops ctx.kv_cache. But the per-bind scratch (the indexer's arange over the
        block count, freqs) depends on the new pool's geometry, so re-derive it via _ensure_bound."""
        self._bound = False

    def _iter_offload_moe_layers(self):
        # Hook for moe.offload_cache.iter_offload_moe_layers: DSV4's MoE block (block.ffn)
        # is a bespoke nn.Module (not OffloadMoELayer), so yield its inner routed-expert
        # layer instead, whose .offload_cache the generic attach sets.
        for block in self._transformer.layers:
            yield block.ffn.experts

    def state_dict(self, *, prefix: str = "", result=None):
        result = {} if result is None else result
        for name, p in self._transformer.named_parameters():
            result[f"{prefix}.{name}" if prefix else name] = p
        return result

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        casted = {}
        for name, p in self._transformer.named_parameters():
            key = f"{prefix}.{name}" if prefix else name
            if key not in state_dict:
                raise RuntimeError(f"Missing weight for DeepSeek-V4 parameter: {key}")
            casted[name] = state_dict.pop(key).to(p.dtype)
        if state_dict and not _internal:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
        self._transformer.load_state_dict(casted, assign=True, strict=False)

    def forward(self) -> torch.Tensor:
        self._ensure_bound()
        batch = get_global_ctx().batch
        input_ids = batch.input_ids.long()
        md = batch.attn_metadata
        if batch.is_prefill:
            # Ragged batched prefill (bs >= 1): each request starts from its own cached_len.
            # A cold segment (start_pos == 0) re-seeds the compressor carry register inside its
            # own attention segment; a radix hit / chunk continuation (start_pos > 0) resumes it
            # FROM THE RING. Per-token ops (embed / HC / norm / MoE) run batched over the
            # concatenated tokens; attention runs per segment so the carry / slot maps never
            # cross requests.
            return self._transformer.prefill_batched(
                input_ids.view(1, -1), md.segments, batch.positions.long(),
                md.last_indices.long(),
            )
        # DECODE (bs>=1): per-row position (GPU int tensor -> no host syncs / graph safe). The
        # compressed staging cap is the max position any row reaches (eager); a static max_seq-1
        # under graph capture (so the captured static-shape graph serves any real replay position).
        B = batch.padded_size
        pos = batch.positions.long().view(-1)[:B]
        if torch.cuda.is_current_stream_capturing():
            # Stage exactly as wide as the snapshot those columns are gathered FROM. The backend
            # sizes it to the live ceiling min(model max, KV token budget), which the scheduler
            # also admits against, so no replay can reach a column past it -- and the two stay in
            # lockstep by construction rather than by convention.
            cmp_stage_cap = md.stage_width - 1
        else:
            cmp_stage_cap = int(pos.max().item())
        return self._transformer.decode(input_ids.view(B, 1), pos, cmp_stage_cap)


__all__ = ["Transformer", "DeepseekV4ForCausalLM"]
