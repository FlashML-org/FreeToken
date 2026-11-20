"""DSparkProposer: drafts one block of k tokens per step with a DSpark draft over the target's tapped layer visits. A family selects it through its spec_drafts behind the mtp CLI algorithm; it is never a --spec-algo name.

The draft's context K/V live in a pool of its own on the target's full-attention slot ids, written from the target's features of every prefill row and every verify row; context K/V of a position depend on the tokens up to it only, so a prefix-cache hit finds them where the target's KV is and no lookahead key is needed.
A verify writes every row, rejected ones included: rows past the accepted ones sit at or after the next block's start, which reads only positions below it, and the next verify rewrites them.
The Markov head's context is the features of each request's last committed token, kept per table row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, List

import torch

from freetoken.spec.proposer import DraftContext, Proposer, SpecProposal, check_draft_path
from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.core import Batch, Req
    from freetoken.models.inkling2_loop.dspark import DSparkConfig, Inkling2DSparkDraft

logger = init_logger(__name__)

DRAFT_METHOD = "dspark_mtp"


def _write_taps(dst: torch.Tensor, cols: list[int], x: torch.Tensor, residual: torch.Tensor | None, hidden: int) -> None:
    rows = x.shape[0]
    src = x if residual is None else x + residual
    for col in cols:
        dst[:rows, col * hidden : (col + 1) * hidden].copy_(src)


class DSparkFeatureSink:
    """The tapped visits' outputs and the final-norm output of every row, side by side in ``buffer [rows, taps*H]``; the column of a visit is the tap whose input it is."""

    def __init__(self, config: DSparkConfig, buffer: torch.Tensor) -> None:
        self.cols = config.output_taps()
        self.wanted_layers = frozenset(self.cols)
        self.final_col, self.hidden, self.buffer = config.num_taps - 1, config.hidden_size, buffer
        self.got: set[int] = set()

    def final_hidden(self, x: torch.Tensor) -> None:
        self.got.add(-1)
        _write_taps(self.buffer, [self.final_col], x, None, self.hidden)

    def layer_output(self, layer_id: int, x: torch.Tensor, residual: torch.Tensor | None = None, num_streams: int = 1) -> None:
        assert num_streams == 1, "Inkling 2 has one residual stream"
        self.got.add(layer_id)
        _write_taps(self.buffer, self.cols[layer_id], x, residual, self.hidden)


def _draft_entry(config) -> str:
    drafts = dict(config.model_spec.spec_drafts)
    if DRAFT_METHOD not in drafts:
        raise ValueError(f"DSpark is not supported for {config.hf_config.architectures[0]}")
    return drafts[DRAFT_METHOD]


class DSparkProposer(Proposer):
    # context K/V of position i read tokens up to i only
    radix_lookahead = False
    # support of a sampled draft: the step's top logits, renormalized at the request's temperature
    SAMPLE_TOP_K: ClassVar[int] = 64
    # request-positions of a full layer's context gathered at once: about 12 KiB each with the fp32 copies
    CONTEXT_CHUNK_ROWS: ClassVar[int] = 16384

    @classmethod
    def _draft_config(cls, config) -> DSparkConfig:
        from freetoken.models.inkling2_loop.dspark import parse_dspark_config

        return parse_dspark_config(config.hf_config)

    @classmethod
    def num_steps(cls, config) -> tuple[int, int]:
        # one forward drafts the checkpoint's whole block [anchor, mask x (block-1)]; a smaller k keeps its first drafts
        k = cls._draft_config(config).block_size - 1
        return k, k

    @classmethod
    def adjust_config(cls, config, override) -> None:
        from freetoken.models.inkling2_loop.dspark import mtp_file

        super().adjust_config(config, override)
        _draft_entry(config)
        if config.speculative_token_map:
            raise ValueError("--speculative-token-map does not apply to DSpark drafts, whose Markov head biases the whole vocabulary")
        if config.use_dummy_weight:
            raise ValueError("DSpark reads its draft from the checkpoint, so it cannot run with --dummy-weight")
        path = config.speculative_draft_model_path or config.model_path
        if config.speculative_draft_model_path:
            check_draft_path(path)
        # checked here because a draft that cannot load otherwise fails only after the whole target has loaded
        mtp_file(path)
        draft = cls._draft_config(config)
        if config.max_seq_len > draft.max_position:
            raise ValueError(f"the DSpark draft runs positions below {draft.max_position}; lower --max-seq-len-override")

    @classmethod
    def budget(cls, config) -> tuple[int, int]:
        c = cls._draft_config(config)
        item, block = 2, config.speculative_num_steps + 1
        aux = config.max_running_req * block * c.num_taps * c.hidden_size * item
        f_rnn = (config.max_running_req + 1) * c.hidden_size * item
        return aux + f_rnn, 2 * c.num_layers * c.num_kv_heads * c.head_dim * item

    @classmethod
    def target_may_be_windowed(cls, config) -> bool:
        from freetoken.kvcache import resolve_pool_class
        from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache

        # the draft reads its own pool on the full-attention slot ids, which the window frees never release
        return resolve_pool_class(config.model_config) is HybridSWAKVCache

    def __init__(self, config, target_model) -> None:
        from freetoken.models.register import _load_attr

        super().__init__(config, target_model)
        self.device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        path = config.speculative_draft_model_path or config.model_path
        self.draft_config = c = self._draft_config(config)
        self.draft: Inkling2DSparkDraft = _load_attr(*_draft_entry(config).split(":")).load(c, target_model, path, self.device, config.dtype)
        # rows of a request in a verify batch
        self.width = self.k + 1
        self.k_cache = self.v_cache = self.aux_buf = self.f_rnn = None
        self._prefill: tuple[DSparkFeatureSink, list[tuple[int, int, int]]] | None = None
        logger.info_rank0(
            f"speculative decoding: DSpark draft {c.num_layers} layers (local {list(c.local_layer_ids)}, window {c.window}), "
            f"block {c.block_size}, {self.k} drafts per step, features of target visits {list(c.target_layer_ids)} (the last is the final norm)"
        )

    # ------------------------------------------------------------------ lifecycle

    def attach(self, ctx: DraftContext) -> None:
        super().attach(ctx)
        c, mr = self.draft_config, self.config.max_running_req
        # a rebuild that keeps the target KV pool keeps this one, which the prefix cache still indexes
        if self.k_cache is None:
            slots = (ctx.num_pages + 1) * self.config.page_size
            shape = (c.num_layers, slots, c.num_kv_heads, c.head_dim)
            self.k_cache = torch.zeros(shape, dtype=self.draft.dtype, device=ctx.device)
            self.v_cache = torch.zeros(shape, dtype=self.draft.dtype, device=ctx.device)
        self.aux_buf = torch.zeros(mr * self.width, c.num_taps * c.hidden_size, dtype=self.draft.dtype, device=ctx.device)
        self.f_rnn = torch.zeros(mr + 1, c.hidden_size, dtype=self.draft.dtype, device=ctx.device)

    def detach(self, *, keep_kv: bool) -> None:
        if not keep_kv:
            self.k_cache = self.v_cache = None
        self.aux_buf = self.f_rnn = None
        self._prefill = None
        super().detach(keep_kv=keep_kv)

    # ------------------------------------------------------------------ target features

    def feature_sink(self, batch: Batch) -> DSparkFeatureSink:
        if batch.phase == "verify":
            return DSparkFeatureSink(self.draft_config, self.aux_buf)
        c = self.draft_config
        spans = [(req.table_idx, req.cached_len, req.device_len) for req in batch.reqs]
        rows = sum(e - s for _, s, e in spans)
        sink = DSparkFeatureSink(c, torch.empty(rows, c.num_taps * c.hidden_size, dtype=self.draft.dtype, device=self.device))
        self._prefill = sink, spans
        return sink

    def _store_context(self, features: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor) -> None:
        # chunked: the per-layer mixtures of a long prefill chunk are fp32 [rows, taps, H]
        for lo in range(0, features.shape[0], 4096):
            hi = lo + 4096
            k, v = self.draft.context_kv(features[lo:hi], positions[lo:hi])
            self.k_cache[:, slots[lo:hi]] = k
            self.v_cache[:, slots[lo:hi]] = v

    def on_prefill(self, batch: Batch, next_tokens_gpu: torch.Tensor) -> None:
        """Context K/V of every chunk row, and each request's Markov context from its last row."""
        (sink, spans), self._prefill = self._prefill, None
        wanted = set(sink.wanted_layers) | {-1}
        if sink.got != wanted:
            raise RuntimeError(
                f"the target handed the DSpark draft visits {sorted(sink.got - {-1})} and final hidden {-1 in sink.got}, "
                f"not visits {sorted(sink.wanted_layers)} and the final hidden: its forward does not export every tap to the feature sink"
            )
        n = sink.buffer.shape[0]
        self._store_context(sink.buffer, batch.positions[:n], batch.out_loc[:n].long())
        ends = torch.tensor([e - s for _, s, e in spans]).cumsum(0) - 1
        table = torch.tensor([row for row, _, _ in spans])
        self.f_rnn[table.to(self.device)] = self.draft.f_rnn(sink.buffer[ends.to(self.device)])

    def post_verify(self, batch: Batch, out: torch.Tensor, emitted: torch.Tensor) -> None:
        """Context K/V of every verify row, and each request's Markov context from its last accepted row."""
        bs = batch.size
        n = bs * self.width
        features = self.aux_buf[:n]
        self._store_context(features, batch.positions[:n], batch.out_loc[:n].long())
        pick = torch.arange(bs, device=self.device) * self.width + emitted[:bs].to(self.device)
        table = torch.tensor([r.table_idx for r in batch.reqs], device=self.device)
        self.f_rnn[table] = self.draft.f_rnn(features[pick])

    # ------------------------------------------------------------------ draft step

    def _context(self, rows: torch.Tensor, starts: torch.Tensor):
        """``context(layer)`` for the draft's block forward, read from the slots each request's page-table row maps: a local layer's window right-aligned below each start, a full layer's positions in chunks."""
        c, bs = self.draft_config, rows.shape[0]
        longest = int(starts.max())
        table = self.ctx.page_table

        def gather(layer: int, pos: torch.Tensor, valid: torch.Tensor):
            slots = table[rows[:, None], pos.clamp(min=0)].long()
            return self.k_cache[layer][slots], self.v_cache[layer][slots], valid

        def context(layer: int):
            if layer in c.local_layer_ids:
                width = min(longest, c.window)
                pos = starts[:, None] - width + torch.arange(width, device=starts.device)
                yield gather(layer, pos, pos >= (starts[:, None] - c.window).clamp(min=0))
                return
            chunk = max(1, self.CONTEXT_CHUNK_ROWS // bs)
            for lo in range(0, longest, chunk):
                pos = (lo + torch.arange(min(chunk, longest - lo), device=starts.device)).expand(bs, -1)
                yield gather(layer, pos, pos < starts[:, None])

        return context

    def propose(self, reqs: List[Req], rows: torch.Tensor, pending: torch.Tensor) -> SpecProposal:
        greedy = [r.sampling_params.is_greedy for r in reqs]
        starts = torch.tensor([r.cached_len for r in reqs], dtype=torch.int64, device=self.device)
        anchor = pending.to(torch.int64)
        hidden = self.draft.forward_block(anchor, starts, self._context(rows.to(self.device), starts))
        f_rnn = self.f_rnn[rows]
        if all(greedy):
            return SpecProposal(self.draft.walk(hidden[:, 1 : self.width], anchor, f_rnn, lambda j, logits: logits.argmax(-1)))
        temps = torch.tensor([1.0 if g else max(r.sampling_params.temperature, 1e-5) for r, g in zip(reqs, greedy)], device=self.device)
        is_greedy = torch.tensor(greedy, device=self.device)
        support, probs = [], []

        def pick(j: int, logits: torch.Tensor) -> torch.Tensor:
            vals, ids = logits.float().topk(self.SAMPLE_TOP_K, dim=-1)
            q = torch.softmax(vals / temps[:, None], dim=-1)
            # greedy rows verify one-hot; their q is ignored, so give them the argmax, which topk puts first
            q = torch.where(is_greedy[:, None], torch.nn.functional.one_hot(torch.zeros_like(ids[:, 0]), q.shape[1]).float(), q)
            idx = (q.cumsum(-1) < torch.rand(q.shape[0], 1, device=q.device)).sum(-1, keepdim=True).clamp(max=q.shape[1] - 1)
            support.append(ids)
            probs.append(q)
            return ids.gather(1, idx).squeeze(1)

        tokens = self.draft.walk(hidden[:, 1 : self.width], anchor, f_rnn, pick)
        return SpecProposal(tokens, torch.stack(support, dim=1), torch.stack(probs, dim=1))
