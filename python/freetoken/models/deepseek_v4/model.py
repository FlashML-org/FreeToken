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
  - Routed MXFP4 experts are served from :class:`OffloadMoeCache` (on-demand) -- the
    framework's core acceleration; their format and the dense fp8 block linears come from
    ``config.quant``.

Heavy ops are FreeToken Triton kernels. Precision matches the reference (FP8/FP4
activation quant + Hadamard rotation re-introduced; see ``ops.py`` / the dsv4 kernels).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv4.hc import hc_post_combine, hc_pre_combine
from freetoken.kernel.triton.dsv4.norm import inv_rms
from freetoken.kernel.triton.dsv4.sinkhorn import hc_split_sinkhorn
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNorm, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import init_logger

from .args import DeepseekV4Args
from .attention import Attention
from .moe import MoE

# Re-exports: keep every class/helper previously defined here importable from .model
# (external import stability; the moved definitions live in their own modules).
from .compress import Compressor, Indexer  # noqa: F401
from .layers import get_compress_topk_idxs, get_window_topk_idxs  # noqa: F401
from .moe import Expert, Gate  # noqa: F401
from .parallel import validate_tp

logger = init_logger(__name__)

# FREETOKEN_SPEC_DEBUG=1 also reports where the draft context KV lands.
_DRAFT_CTX_DEBUG = os.environ.get("FREETOKEN_SPEC_DEBUG", "0") == "1"


@dataclass(frozen=True)
class DSparkTargetFeatures:
    """The target outputs consumed by the next DSpark proposal.

    vLLM passes auxiliary hidden states, target positions, and context slot mappings
    together.  FreeToken stores request table rows instead of physical slots so the
    slots are resolved from the live page map when the feature bundle is consumed.
    Every tensor is produced by the same target forward.
    """

    hidden: torch.Tensor
    positions: torch.Tensor
    table_rows: torch.Tensor


class Block(BaseOP):
    """Decoder block with manifold-constrained Hyper-Connections (4 residual streams)."""

    def __init__(self, layer_id: int, args: DeepseekV4Args, compress_ratio: int | None = None, *, strategy: str = "offload", decode_target: str = "gpu", quant_config=None, prefix: str = ""):
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.dim = args.dim
        self.attn = Attention(layer_id, args, compress_ratio, quant_config=quant_config, prefix=f"{prefix}.attn")
        self.ffn = MoE(layer_id, args, strategy=strategy, decode_target=decode_target, quant_config=quant_config, prefix=f"{prefix}.ffn")
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_attn_fn = torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        self.hc_ffn_fn = torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        self.hc_attn_base = torch.empty(mix_hc, dtype=torch.float32)
        self.hc_ffn_base = torch.empty(mix_hc, dtype=torch.float32)
        self.hc_attn_scale = torch.empty(3, dtype=torch.float32)
        self.hc_ffn_scale = torch.empty(3, dtype=torch.float32)

    def hc_pre(self, x, hc_fn, hc_scale, hc_base):
        shape, dtype = x.size(), x.dtype
        x2d = x.flatten(2)
        xf = x2d.float()
        rsqrt = inv_rms(x2d, self.norm_eps)
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
        x = self.attn_norm.forward(x)
        x = self.attn.forward_ragged(x, segments, flat_positions)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm.forward(x)
        x = self.ffn.forward(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x

    def decode_step(self, x, pos, rows, cmp_stage_cap, input_ids, wctx=None):
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm.forward(x)
        x = self.attn.decode_step(x, pos, rows, cmp_stage_cap, wctx)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm.forward(x)
        x = self.ffn.forward(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x

    def verify_block(
        self, x, pos, rows, cmp_stage_cap, input_ids, num_reqs, span, wctx
    ):
        """Fixed-width DSpark target block; algebra matches ``decode_step``."""
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn_norm.forward(x)
        x = self.attn.verify_block(
            x, pos, rows, cmp_stage_cap, num_reqs, span, wctx
        )
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm.forward(x)
        x = self.ffn.forward(x, input_ids)
        return self.hc_post(x, residual, post, comb)


class Transformer(BaseOP):
    def __init__(self, args: DeepseekV4Args, quant_config=None, *, strategy: str = "offload", decode_target: str = "gpu", prefix: str = ""):
        # Check every tensor-parallel split before building a single layer, so a bad
        # --tensor-parallel-size fails with one clear message, not a reshape deep in a
        # forward. embed / head / norm / hyper-connections stay replicated or are split
        # by the shared TP-aware layer classes.
        validate_tp(args)
        self.args = args
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.hc_mult = hc_mult = args.hc_mult
        # The embedding table and the output head are the two largest tensors (~3.0 GiB
        # together), so they are vocabulary-parallel: VocabParallelEmbedding keeps one
        # contiguous block of rows per rank and all-reduces the masked lookup, and
        # ParallelLMHead all-gathers its logit slice and picks each request's final
        # token off the attention metadata. Both live in freetoken.layers.
        self.embed = VocabParallelEmbedding(args.vocab_size, args.dim)
        self.layers = OPList([Block(i, args, strategy=strategy, decode_target=decode_target, quant_config=quant_config, prefix=f"{prefix}.layers.{i}") for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, self.norm_eps)
        self.head = ParallelLMHead(args.vocab_size, args.dim, quant_config=quant_config, prefix=f"{prefix}.head")
        hc_dim = hc_mult * args.dim
        self.hc_head_fn = torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        self.hc_head_base = torch.empty(hc_mult, dtype=torch.float32)
        self.hc_head_scale = torch.empty(1, dtype=torch.float32)
        # dSpark drafter, only when the checkpoint ships it AND the run asked for it.
        # Its blocks continue this model's layer ids, so they share the expert banks,
        # the GPU slot cache and the KV pools with no separate index space.
        self.drafter = None
        # Layers whose output the drafter reads, as 0-based indices. vLLM converts the
        # checkpoint ids [40, 41, 42] to aux ids [41, 42, 43], then captures when
        # ``idx + 1`` is in that set: target outputs 40, 41 and 42 exactly.
        self._aux_layer_ids: frozenset[int] = frozenset()
        if args.n_draft_layers:
            from .dspark import DSparkDrafter

            self.drafter = DSparkDrafter(args, strategy=strategy, decode_target=decode_target, quant_config=quant_config, prefix=f"{prefix}.drafter")
            # The drafter shares this model's embedding table and output head; both are
            # vocabulary-parallel under TP, so it reaches them through these methods
            # rather than holding tensors that would only cover one rank's slice.
            self.drafter._embed_tokens = self.embed_tokens
            ids = tuple(args.dspark_target_layer_ids)
            self._aux_layer_ids = frozenset(ids)
            bad = [i for i in ids if not 0 <= i < args.n_layers]
            if bad:
                raise ValueError(
                    f"0-based dspark_target_layer_ids {ids} fall outside the "
                    f"{args.n_layers} target layers"
                )
            logger.info_rank0(
                f"dSpark: tapping target layer outputs {sorted(ids)}"
            )
        self._target_features: DSparkTargetFeatures | None = None

    def bind(self, pool, device: torch.device) -> None:
        for layer in self.layers.op_list:
            layer.attn.bind(pool, device)
        if self.drafter is not None:
            self.drafter.bind(pool, device)

    def target_features(self) -> DSparkTargetFeatures | None:
        return self._target_features

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Vocabulary-parallel lookup keeping ``input_ids``' shape, plus a trailing ``dim``."""
        return self.embed.forward(input_ids.reshape(-1)).view(*input_ids.shape, self.args.dim)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary logits for EVERY row of ``h`` [rows, dim].

        ``ParallelLMHead.forward`` keeps one row per request (it picks each prefill's last
        token off the batch); a speculative verify and the drafter need a row per drafted
        position, so they project and gather here. ``all_gather`` concatenates on dim 0, so
        the gathered tensor is rank-major: one row block per rank, in vocabulary order.
        """
        head = self.head
        local = head.quant_method.apply(head, h)  # [rows, vocab_local]
        if head.tp_size == 1:
            return local[:, : head.num_embeddings]
        rows = local.shape[0]
        gathered = head._comm.all_gather(local).view(head.tp_size, rows, -1)
        return gathered.permute(1, 0, 2).reshape(rows, -1)[:, : head.num_embeddings]

    def hc_head(self, x):
        shape, dtype = x.size(), x.dtype
        dim = self.args.dim
        x2d = x.flatten(2)
        xf = x2d.float()
        rsqrt = inv_rms(x2d, self.norm_eps)
        mixes = F.linear(xf, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        M = shape[0] * shape[1]
        return hc_pre_combine(xf.view(M, self.hc_mult, dim), pre.view(M, self.hc_mult), dtype).view(*shape[:2], dim)

    def prefill_batched(
        self, input_ids: torch.Tensor, segments, flat_positions: torch.Tensor,
        logit_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Ragged batched prefill (bs >= 1). ``input_ids`` is [1, T] -- the requests' NEW tokens
        # concatenated (cu_seqlens, no padding); each request starts at its own cached_len
        # (cold == 0, radix hit / chunk continuation > 0). Per-token ops run batched over all T
        # tokens; attention runs per request on its [offset, offset+n) segment (see Block). Each
        # request's window/cmp/idx slots + ring blocks were allocated disjointly by
        # allocate_paged, so the per-request attention never reads another request's KV/carry.
        # ``segments`` [(offset, extend_len, table_idx, start_pos)] comes off the attention
        # metadata; ``flat_positions`` [T] is the scheduler-staged batch.positions (per-token
        # ABSOLUTE position); the head picks each request's final token off the attention
        # metadata -> its next-token logits row.
        h = self.embed.forward(input_ids.view(-1)).view(1, -1, self.args.dim)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        aux: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers.op_list):
            h = layer.prefill_batched(h, input_ids, segments, flat_positions)
            if i in self._aux_layer_ids:
                # The drafter's whole view of the context: this block's output with the
                # hyper-connection copies averaged away, [1, T, dim].
                aux.append(h.mean(dim=2))
        aux_hidden = (
            torch.cat(aux, dim=-1).view(-1, self.args.dim * len(aux))
            if aux else None
        )
        if aux:
            rows = torch.empty_like(flat_positions)
            for off, n, ti, _start in segments:
                rows[off:off + n] = ti
            self._target_features = DSparkTargetFeatures(
                aux_hidden,
                flat_positions.clone(),
                rows.clone(),
            )
        else:
            self._target_features = None
        h = self.hc_head(h)
        h = self.norm.forward(h)
        # Normally only each request's final token needs logits, and the head picks those
        # rows itself. A speculative VERIFY pass is the same ragged prefill -- each request
        # resuming from its own start_pos -- but it needs the logits at EVERY drafted
        # position, to compare them against what the drafter proposed.
        if logit_indices is None:
            return self.head.forward(h[0])  # [B, vocab]
        return self.logits(h[0, logit_indices])  # [len(logit_indices), vocab]

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
        h = self.embed.forward(input_ids.view(-1)).view(B, 1, self.args.dim)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        # Hoist the layer-invariant per-step decode tensors (shared window-ring global slots):
        # resolved ONCE off the attention metadata (recomputed per call, so a capture records
        # the gathers -- never cache it on the metadata) and threaded
        # into every layer. They read only the shared snapshot / positions, so they are identical
        # across layers.
        wctx = get_global_ctx().batch.attn_metadata.window_ctx(pos, rows)
        aux: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers.op_list):
            h = layer.decode_step(h, pos, rows, cmp_stage_cap, input_ids, wctx)
            if i in self._aux_layer_ids:
                aux.append(h.mean(dim=2))  # [B, 1, dim]
        aux_hidden = (
            torch.cat(aux, dim=-1).view(-1, self.args.dim * len(aux))
            if aux else None
        )
        if aux:
            table_rows = get_global_ctx().batch.active_table_idx
            if table_rows is None:
                raise RuntimeError("DSpark decode features require request table rows")
            # clone() is captured: replay updates these output buffers without leaving
            # them aliased to the graph input buffers that the next batch overwrites.
            self._target_features = DSparkTargetFeatures(
                aux_hidden,
                pos.reshape(-1).clone(),
                table_rows[:B].reshape(-1).clone(),
            )
        else:
            self._target_features = None
        h = self.hc_head(h)
        h = self.norm.forward(h)
        return self.head.forward(h[:, -1])

    def verify_block(
        self, input_ids: torch.Tensor, pos: torch.Tensor, span: int
    ) -> torch.Tensor:
        """Run one fixed-shape DSpark target verification inside a CUDA graph.

        ``input_ids`` is request-major ``[1, R * span]`` and every request contributes
        exactly anchor + gamma rows. The attention backend supplies one persistent
        whole-history snapshot per request; ``rows`` maps every token to that request.
        """
        total = input_ids.numel()
        if span < 1 or total % span:
            raise ValueError(f"invalid DSpark verify shape: {total} rows, span={span}")
        num_reqs = total // span
        rows = torch.arange(num_reqs, device=input_ids.device).repeat_interleave(span)
        md = get_global_ctx().batch.attn_metadata
        cmp_stage_cap = md.stage_width - 1
        wctx = md.window_ctx(pos, rows)

        h = self.embed_tokens(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        aux: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers.op_list):
            h = layer.verify_block(
                h, pos, rows, cmp_stage_cap, input_ids,
                num_reqs, span, wctx,
            )
            if i in self._aux_layer_ids:
                aux.append(h.mean(dim=2))

        aux_hidden = (
            torch.cat(aux, dim=-1).view(-1, self.args.dim * len(aux))
            if aux else None
        )
        if aux:
            table_rows = get_global_ctx().batch.active_table_idx
            if table_rows is None or table_rows.numel() < num_reqs:
                raise RuntimeError("DSpark verify features require one table row per request")
            self._target_features = DSparkTargetFeatures(
                aux_hidden,
                pos.reshape(-1).clone(),
                table_rows[:num_reqs].repeat_interleave(span).clone(),
            )
        else:
            self._target_features = None
        h = self.norm.forward(self.hc_head(h))
        return self.logits(h.view(-1, self.args.dim))


class DeepseekV4ForCausalLM(BaseLLMModel):
    """Engine adapter: a registered :class:`BaseLLMModel` wrapping the DSV4 transformer.

    KV pools, rope constants and compressor state are bound on the first forward.
    """

    def __init__(self, config):
        self._config = config
        self._args: DeepseekV4Args = config.dsv4_args
        self.model = Transformer(self._args, config.quant, strategy=config.moe_strategy, decode_target=config.decode_target, prefix="model")
        self._bound = False
        self.speculative_verify_block_size = (
            self._args.dspark_block_size if self._args.n_draft_layers else 0
        )

    def _ensure_bound(self) -> None:
        if self._bound:
            return
        pool = get_global_ctx().kv_cache
        self.model.bind(pool, pool.device)
        self._bound = True

    def mark_for_rebind(self) -> None:
        """Force a re-bind on the next forward. The model holds NO pool reference -- buffers are read
        off ctx.kv_cache via @property -- so a runtime rebuild needs no unbind; the old pool frees
        when the engine drops ctx.kv_cache. But the per-bind scratch (the indexer's arange over the
        block count, freqs) depends on the new pool's geometry, so re-derive it via _ensure_bound."""
        self._bound = False

    def dspark_target_features(self) -> DSparkTargetFeatures | None:
        """Outputs of the most recent target forward, as one addressed bundle."""
        return self.model.target_features()

    def catch_up_draft_context(self, features: DSparkTargetFeatures) -> None:
        """Precompute draft-layer context KV from one target feature bundle."""
        drafter = self.model.drafter
        if drafter is None:
            return
        flat = features.hidden
        positions = features.positions
        rows = features.table_rows
        if flat.shape[0] != positions.numel() or rows.numel() != positions.numel():
            raise RuntimeError(
                "DSpark target features must have one hidden, position, and table row "
                f"per token; got {flat.shape[0]}, {positions.numel()}, {rows.numel()}"
            )
        backend = get_global_ctx().attn_backend
        slots = backend.window_slots_at(rows, positions)
        valid = slots >= 0
        flat = flat[valid]
        positions = positions[valid]
        slots = slots[valid]
        if positions.numel() == 0:
            return
        if _DRAFT_CTX_DEBUG:
            logger.debug_rank0(
                "draft ctx: %d rows, positions %s..%s, slots %s..%s",
                flat.shape[0], int(positions.min()), int(positions.max()),
                int(slots.min()), int(slots.max()),
            )
        drafter.catch_up_context(flat, positions, slots)

    def draft(
        self, sampling_params
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Propose the current batch's block with the dSpark drafter.

        Runs over the SAME prepared batch as the verify pass that follows -- the same
        positions, the same segments, the same paged slots. The draft layers own layer
        ids after the target's, so ``allocate_paged`` has already given them their own
        window slots at these positions and they write KV there without disturbing the
        target's.

        Returns ``(proposed [R*gamma], q [R*gamma,V], confidence [R*gamma])``.
        """
        if self.model.drafter is None:
            return None
        self._ensure_bound()
        batch = get_global_ctx().batch
        md = batch.attn_metadata
        return self.model.drafter.propose(
            batch.input_ids.long().view(1, -1),
            md.segments,
            batch.positions.long(),
            self.model.logits,
            sampling_params,
        )

    def forward(self) -> torch.Tensor:
        self._ensure_bound()
        batch = get_global_ctx().batch
        input_ids = batch.input_ids.long()
        md = batch.attn_metadata
        if getattr(batch, "spec_verify_decode", False):
            span = int(batch.spec_block) + 1
            return self.model.verify_block(
                input_ids.view(1, -1), batch.positions.long(), span
            )
        if batch.is_prefill:
            # Ragged batched prefill (bs >= 1): each request starts from its own cached_len.
            # A cold segment (start_pos == 0) re-seeds the compressor carry register inside its
            # own attention segment; a radix hit / chunk continuation (start_pos > 0) resumes it
            # FROM THE RING. Per-token ops (embed / HC / norm / MoE) run batched over the
            # concatenated tokens; attention runs per segment so the carry / slot maps never
            # cross requests.
            # A speculative verify needs logits at EVERY drafted position, not just each
            # request's last one: acceptance compares the target's own prediction at each
            # position against what the drafter proposed there.
            logit_indices = None
            if getattr(batch, "speculative", False):
                logit_indices = torch.arange(
                    input_ids.numel(), device=input_ids.device, dtype=torch.long
                )
            return self.model.prefill_batched(
                input_ids.view(1, -1), md.segments, batch.positions.long(),
                logit_indices=logit_indices,
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
        return self.model.decode(input_ids.view(B, 1), pos, cmp_stage_cap)


__all__ = ["DSparkTargetFeatures", "Transformer", "DeepseekV4ForCausalLM"]
