"""DFlashProposer: drafts one block of k tokens per step with a DFlash 2 draft over the target's tapped layer outputs.

The draft's context rows live in a per-table-row K/V ring (positions ``p % R``) that the prefill and every verify extend from the target's features; on a hybrid target the same rows also go to a window on the GDN slots, so a prefix-cache hit restores them and the ring is rebuilt from the window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import torch

from freetoken.attention.linear import gdn_slot
from freetoken.engine.sample import make_device_tensor
from freetoken.kernel.triton.dflash import ring_attention_workspace
from freetoken.spec.dflash.window import DFlashWindow
from freetoken.spec.proposer import DraftContext, Proposer, SpecProposal, check_draft_path
from freetoken.utils import cached_load_hf_config, init_logger

if TYPE_CHECKING:
    from freetoken.core import Batch, Req
    from freetoken.models.config import SlotStateSpec
    from freetoken.models.dflash import DFlashDraftConfig

logger = init_logger(__name__)


def _write_tap(dst: torch.Tensor, col: int, x: torch.Tensor, residual: torch.Tensor | None, num_streams: int, hidden: int) -> None:
    """Column block ``col`` of ``dst`` takes the rows of ``x`` (plus ``residual`` when given), averaged over their ``num_streams`` residual streams."""
    assert x.shape[1] == num_streams * hidden, f"a tapped layer gave {x.shape[1]} columns, expected {num_streams} x {hidden}"
    out = dst[: x.shape[0], col * hidden : (col + 1) * hidden]
    if num_streams > 1:
        x = x if residual is None else x + residual
        out.copy_(x.view(x.shape[0], num_streams, hidden).float().mean(1))
    elif residual is None:
        out.copy_(x)
    else:
        torch.add(x, residual, out=out)


class DFlashPrefillSink:
    """The tapped layers' outputs of each request's last ``m`` chunk rows, side by side in one feature block; ``spans`` holds each request's (cached_len, device_len, m) when the batch was formed."""

    def __init__(self, taps: tuple[int, ...], rows_idx: torch.Tensor, aux: torch.Tensor, hidden: int, spans: List[tuple[int, int, int]]) -> None:
        self.wanted_layers = frozenset(taps)
        self.cols = {layer: j for j, layer in enumerate(taps)}
        self.rows_idx, self.aux, self.hidden, self.spans = rows_idx, aux, hidden, spans
        self.got: set[int] = set()

    def final_hidden(self, x: torch.Tensor) -> None:
        pass

    def layer_output(self, layer_id: int, x: torch.Tensor, residual: torch.Tensor | None = None, num_streams: int = 1) -> None:
        self.got.add(layer_id)
        kept = None if residual is None else residual.index_select(0, self.rows_idx)
        _write_tap(self.aux, self.cols[layer_id], x.index_select(0, self.rows_idx), kept, num_streams, self.hidden)


class DFlashVerifySink:
    """Every verify row's tapped outputs into the persistent feature buffer, so a captured verify graph writes it too."""

    def __init__(self, taps: tuple[int, ...], buffer: torch.Tensor, hidden: int) -> None:
        self.wanted_layers = frozenset(taps)
        self.cols = {layer: j for j, layer in enumerate(taps)}
        self.buffer, self.hidden = buffer, hidden

    def final_hidden(self, x: torch.Tensor) -> None:
        pass

    def layer_output(self, layer_id: int, x: torch.Tensor, residual: torch.Tensor | None = None, num_streams: int = 1) -> None:
        _write_tap(self.buffer, self.cols[layer_id], x, residual, num_streams, self.hidden)


def _draft_entry(config) -> tuple[str, str]:
    """(module, class name) of the target family's DFlash draft."""
    drafts = dict(config.model_spec.spec_drafts)
    if "dflash" not in drafts:
        raise ValueError(f"DFlash is not supported for {config.hf_config.architectures[0]} yet")
    module, name = drafts["dflash"].split(":")
    return module, name


class DFlashProposer(Proposer):
    @classmethod
    def num_steps(cls, config) -> tuple[int, int]:
        # one forward drafts [bonus, mask x k], and the draft is trained on blocks of block_size
        k = cls._draft_config(config).block_size - 1
        return k, k

    @classmethod
    def adjust_config(cls, config, override) -> None:
        super().adjust_config(config, override)
        if config.speculative_token_map:
            raise ValueError("--speculative-token-map only works with --speculative-algorithm mtp")
        module, name = _draft_entry(config)
        path = config.speculative_draft_model_path
        if not path:
            raise ValueError("--speculative-algorithm dflash needs --speculative-draft-model-path: a DFlash 2 draft checkpoint")
        check_draft_path(path)
        archs = list(getattr(cached_load_hf_config(path), "architectures", None) or [])
        if archs[:1] != [name]:
            raise ValueError(f"{path}: expected a {name} checkpoint, got architectures {archs}")
        draft, target, arch = cls._draft_config(config), config.model_config, config.hf_config.architectures[0]
        if draft.num_target_layers != target.num_layers or max(draft.target_layer_ids) >= target.num_layers:
            raise ValueError(
                f"{path} reads layers {list(draft.target_layer_ids)} of a {draft.num_target_layers}-layer target, but {arch} has {target.num_layers} layers"
            )
        if (draft.hidden_size, draft.vocab_size) != (target.hidden_size, target.vocab_size) or draft.mask_token_id >= target.vocab_size:
            raise ValueError(
                f"{path} (hidden {draft.hidden_size}, vocab {draft.vocab_size}, mask token {draft.mask_token_id}) does not fit "
                f"{arch} (hidden {target.hidden_size}, vocab {target.vocab_size})"
            )
        if target.linear_attention_group() is None and config.cache_type != "naive":
            raise ValueError(
                "DFlash keeps its context window on the target's GDN slots, which a prefix-cache hit restores; "
                f"{arch} has no linear-attention layers, so run it with --cache-type naive"
            )

    @classmethod
    def _draft_config(cls, config) -> DFlashDraftConfig:
        from freetoken.models.register import _load_attr

        module, _ = _draft_entry(config)
        return _load_attr(module, "parse_config")(cached_load_hf_config(config.speculative_draft_model_path))

    @classmethod
    def budget(cls, config) -> tuple[int, int]:
        c = cls._draft_config(config)
        block, rows, item = config.speculative_num_steps + 1, config.max_running_req + 1, config.dtype.itemsize
        ring_len = DFlashWindow.length(c.window, block)
        kv = 2 * c.num_layers * rows * ring_len * c.num_kv_heads * c.head_dim * item
        scratch = sum(t.nbytes for t in ring_attention_workspace(rows, block, c.num_heads, c.num_kv_heads, c.head_dim, ring_len, torch.device("meta")))
        aux = config.max_running_req * block * len(c.target_layer_ids) * c.hidden_size * item
        # the rope table is sized by the page table at attach; max_seq_len bounds its width
        rope = config.max_seq_len * c.head_dim * 4
        return kv + scratch + aux + rope, 0

    @classmethod
    def slot_states(cls, config) -> tuple[SlotStateSpec, ...]:
        # naive GDN keeps no snapshots, so nothing is ever restored
        if config.cache_type != "hybrid_radix":
            return ()
        c = cls._draft_config(config)
        return DFlashWindow.slot_specs(DFlashWindow.length(c.window, config.speculative_num_steps + 1), c.hidden_size)

    def __init__(self, config, target_model) -> None:
        from freetoken.models.register import _load_attr

        super().__init__(config, target_model)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = config.dtype
        module, name = _draft_entry(config)
        path = config.speculative_draft_model_path
        self.draft_config = c = self._draft_config(config)
        self.draft = _load_attr(module, name).load(c, target_model, path, self.device, self.dtype)
        self.taps, self.hidden, self.block = c.target_layer_ids, c.hidden_size, self.k + 1
        self.ring_len = DFlashWindow.length(c.window, self.block)
        self._seen: Dict[int, int] = {}  # table row -> uid of the request whose chunks its ring holds
        self._prefill_sink: DFlashPrefillSink | None = None
        self.dflash_ring = self.window = self.aux_buf = None
        self._graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self._inject_graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        params = sum(t.numel() for t in self.draft.state_dict().values())
        logger.info_rank0(
            f"speculative decoding: DFlash 2 draft {path} resident ({params / 1e6:.0f}M params, {c.num_layers} layers, "
            f"window {c.window}, block {self.block}, top-{c.selector_top_k} selector, features of target layers {list(self.taps)})"
        )

    # ------------------------------------------------------------------ lifecycle

    def attach(self, ctx: DraftContext) -> None:
        super().attach(ctx)
        mr = self.config.max_running_req
        self.dflash_ring = self.draft.make_dflash_ring(mr + 1, self.block, self.ring_len, ctx.page_table.shape[1], ctx.device)
        self.window = DFlashWindow(ctx.slot_states, self.ring_len, self.hidden) if DFlashWindow.NAME in ctx.slot_states else None
        self.aux_buf = torch.zeros(mr * self.block, len(self.taps) * self.hidden, dtype=self.dtype, device=ctx.device)

    def detach(self, *, keep_kv: bool) -> None:
        # the ring is keyed by table row and position, not by KV slot, and a rebuild finds no request running
        self._graphs, self._inject_graphs = {}, {}
        self.dflash_ring = self.window = self.aux_buf = None
        self._seen.clear()
        super().detach(keep_kv=keep_kv)

    def release(self, req: Req) -> None:
        self._seen.pop(req.table_idx, None)

    # ------------------------------------------------------------------ draft step

    def _draft(self, pending, starts, rows, temperature, greedy) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.draft.forward_block(pending, starts, rows, self.dflash_ring)
        # row 0 of each block is the pending token; rows 1..k predict the drafts
        pred = hidden.view(pending.shape[0], self.block, -1)[:, 1:]
        candidate_ids, unary_logits = self.draft.compute_candidates(pred)
        uniforms = torch.rand(pred.shape[:2], device=pred.device)
        tokens, q = self.draft.candidate_selector.sample_path(candidate_ids, unary_logits, pred, pending, temperature, greedy, uniforms)
        return tokens, candidate_ids, q

    def _inject(self, features: torch.Tensor, positions: torch.Tensor, rows: torch.Tensor, slots: torch.Tensor | None) -> None:
        """The context rows of ``features`` into the ring rows ``rows`` and the window slots ``slots`` at ``positions``."""
        ctx = self.draft.context_rows(features)
        self.draft.store_context(ctx, positions, rows, self.dflash_ring)
        if self.window is not None:
            self.window.write(ctx, positions, slots)

    def capture_graphs(self, bs_list: List[int], pool) -> None:
        """One graph per batch size for the draft step (block forward + selector) and one for the verify's context inject."""
        if not bs_list:
            return
        ctx, dev, k, top_k = self.ctx, self.ctx.device, self.k, self.draft_config.selector_top_k
        max_bs, dummy = max(bs_list), ctx.dummy_req.table_idx
        self._in = (
            torch.zeros(max_bs, dtype=torch.int64, device=dev),                 # bonus
            torch.zeros(max_bs, dtype=torch.int64, device=dev),                 # starts
            torch.full((max_bs,), dummy, dtype=torch.int64, device=dev),        # ring rows
            torch.ones(max_bs, dtype=torch.float32, device=dev),                # temperature
            torch.ones(max_bs, dtype=torch.bool, device=dev),                   # greedy
        )
        self._out = (
            torch.zeros(max_bs, k, dtype=torch.int64, device=dev),
            torch.zeros(max_bs, k, top_k, dtype=torch.int64, device=dev),
            torch.zeros(max_bs, k, top_k, dtype=torch.float32, device=dev),
        )
        n = max_bs * self.block
        self._inject_in = (
            torch.zeros(n, dtype=torch.int32, device=dev),                      # positions
            torch.full((n,), dummy, dtype=torch.int64, device=dev),             # ring rows
            torch.zeros(n, dtype=torch.int64, device=dev),                      # window slots
        )

        def draft(bs: int) -> None:
            for out, got in zip(self._out, self._draft(*(t[:bs] for t in self._in))):
                out[:bs].copy_(got)

        def inject(bs: int) -> None:
            n = bs * self.block
            self._inject(self.aux_buf[:n], *(t[:n] for t in self._inject_in))

        for bs in sorted(bs_list, reverse=True):
            for fn, graphs in ((draft, self._graphs), (inject, self._inject_graphs)):
                # outside the graph: triton compiles and the scratch is allocated here
                with torch.cuda.stream(ctx.stream):
                    fn(bs)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=ctx.stream):
                    fn(bs)
                if pool is None:
                    pool = graph.pool()
                graphs[bs] = graph
        logger.info_rank0(f"speculative decoding: DFlash draft graphs captured for bs {sorted(self._graphs)}")

    def propose(self, reqs: List[Req], rows: torch.Tensor, pending: torch.Tensor) -> SpecProposal:
        bs = len(reqs)
        greedy = [r.sampling_params.is_greedy for r in reqs]
        padded = next((b for b in sorted(self._graphs) if b >= bs), None)
        pad = (padded or bs) - bs
        pin = dict(pin_memory=True)
        starts = torch.tensor([r.cached_len for r in reqs] + [0] * pad, dtype=torch.int64, **pin)
        flags = torch.tensor(greedy + [True] * pad, dtype=torch.bool, **pin)
        temps = torch.tensor([1.0 if g else max(r.sampling_params.temperature, 1e-5) for r, g in zip(reqs, greedy)] + [1.0] * pad, dtype=torch.float32, **pin)
        if padded is None:
            to = lambda t: t.to(self.device, non_blocking=True)
            tokens, cand, q = self._draft(pending.to(torch.int64), to(starts), rows, to(temps), to(flags))
        else:
            bonus, g_starts, g_rows, g_temps, g_flags = self._in
            bonus[:bs].copy_(pending)
            g_rows[:bs].copy_(rows)
            if pad:
                # padded rows draft on the dummy ring row at position 0
                bonus[bs:padded].zero_()
                g_rows[bs:padded].fill_(self.ctx.dummy_req.table_idx)
            g_starts[:padded].copy_(starts, non_blocking=True)
            g_temps[:padded].copy_(temps, non_blocking=True)
            g_flags[:padded].copy_(flags, non_blocking=True)
            self._graphs[padded].replay()
            tokens, cand, q = (t[:bs] for t in self._out)
        return SpecProposal(tokens) if all(greedy) else SpecProposal(tokens, cand, q)

    # ------------------------------------------------------------------ target features

    def feature_sink(self, batch: Batch) -> DFlashPrefillSink | DFlashVerifySink:
        if batch.phase == "verify":
            return DFlashVerifySink(self.taps, self.aux_buf, self.hidden)
        # a later block reads at most the ring's length of rows behind it
        idx, spans, off = [], [], 0
        for req in batch.reqs:
            s, e = req.cached_len, req.device_len
            m = min(e - s, self.ring_len)
            idx.extend(range(off + e - s - m, off + e - s))
            spans.append((s, e, m))
            off += e - s
        aux = torch.empty(len(idx), len(self.taps) * self.hidden, dtype=self.dtype, device=self.device)
        self._prefill_sink = DFlashPrefillSink(self.taps, make_device_tensor(idx, torch.int64, self.device), aux, self.hidden, spans)
        return self._prefill_sink

    def post_verify(self, batch: Batch, out: torch.Tensor, emitted: torch.Tensor) -> None:
        """Every verify row's context into the ring and the window; the rows past the accepted ones are overwritten by the next block before anything reads them."""
        padded, block = batch.padded_size, self.block
        n = padded * block
        rows = batch.active_table_idx.repeat_interleave(block)
        slots = None
        if self.window is not None:
            slots = make_device_tensor([gdn_slot(r) for r in batch.padded_reqs], torch.int64, self.device).repeat_interleave(block)
        graph = self._inject_graphs.get(padded)
        if graph is None:
            self._inject(self.aux_buf[:n], batch.positions[:n], rows, slots)
            return
        g_pos, g_rows, g_slots = self._inject_in
        g_pos[:n].copy_(batch.positions[:n])
        g_rows[:n].copy_(rows)
        if slots is not None:
            g_slots[:n].copy_(slots)
        graph.replay()

    def on_prefill(self, batch: Batch, next_tokens_gpu: torch.Tensor) -> None:
        """The chunk's context rows into the ring and the window; a prefix hit's first chunk first rebuilds the ring from the restored window, and a tracked request's window is copied to its snapshot slot."""
        sink, self._prefill_sink = self._prefill_sink, None
        if sink.got != sink.wanted_layers:
            raise RuntimeError(
                f"the target handed the draft the outputs of layers {sorted(sink.got)}, not {sorted(sink.wanted_layers)}: "
                "its forward does not export every tapped layer to the feature sink"
            )
        ctx = self.draft.context_rows(sink.aux)
        window, W = self.window, self.draft_config.window
        pos, rows, slots, restore = [], [], [], []
        for req, (s, e, m) in zip(batch.reqs, sink.spans):
            row, first = req.table_idx, self._seen.get(req.table_idx) != req.uid
            self._seen[row] = req.uid
            # a first chunk past position 0 continues a restored snapshot; the prompt's own rows cover the first block's window only when at least W-1 of them follow
            if first and s > 0 and window is not None and req.prompt_len - s < W - 1:
                restore.append((gdn_slot(req), max(0, s - W + 1), s, row))
            pos.extend(range(e - m, e))
            rows.extend([row] * m)
            slots.extend([gdn_slot(req)] * m)
        if restore:
            # before this chunk's rows: with a long chunk they take the same ring slots
            self.draft.store_context(
                torch.cat([window.read(slot, lo, hi - lo) for slot, lo, hi, _ in restore]),
                make_device_tensor([p for _, lo, hi, _ in restore for p in range(lo, hi)], torch.int32, self.device),
                make_device_tensor([row for _, lo, hi, row in restore for _ in range(lo, hi)], torch.int64, self.device),
                self.dflash_ring,
            )
        positions = make_device_tensor(pos, torch.int32, self.device)
        self.draft.store_context(ctx, positions, make_device_tensor(rows, torch.int64, self.device), self.dflash_ring)
        if window is None:
            return
        window.write(ctx, positions, make_device_tensor(slots, torch.int64, self.device))
        fla = batch.fla_metadata
        if fla is not None and fla.track_dst is not None:
            # the snapshot at the tracked boundary b holds [b-W, b): inside the live window [e-R, e), since e - b <= R - W
            window.tensor[fla.track_dst] = window.tensor[fla.track_src]
