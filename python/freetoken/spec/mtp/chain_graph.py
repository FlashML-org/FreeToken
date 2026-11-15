"""MTP drafting as one CUDA graph per batch size: the chain's k-1 draft decode steps, and the read-only shared draft's k steps over the target's cache, with each step's metadata and KV slots built on the device."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import torch

from freetoken.attention.qsa_sparse import QSASparseAttnBackend, QSASparseMetadata
from freetoken.attention.triton import TritonAttentionBackend, TritonMetadata
from freetoken.core import Batch
from freetoken.kernel.triton.attention import gather_kv_indices

if TYPE_CHECKING:
    from freetoken.core import Req

    from .proposer import MTPProposer
    from .shared_proposer import SharedKVMTPProposer


class _TritonSteps:
    """Each step's flat KV slot list, gathered from the page table in the graph."""

    def __init__(self, ctx, steps: int, max_bs: int) -> None:
        self.page_table = ctx.page_table
        self.indptr = torch.zeros(steps, max_bs + 1, dtype=torch.int32, device=ctx.device)
        self.indices = torch.zeros(steps, max_bs * ctx.page_table.shape[1], dtype=torch.int32, device=ctx.device)
        self.arange = torch.arange(max_bs + 1, dtype=torch.int32, device=ctx.device)

    def metadata(self, bs: int, step: int, positions: torch.Tensor) -> TritonMetadata:
        return TritonMetadata(
            cu_seqlens_q_gpu=self.arange[: bs + 1],
            indptr=self.indptr[step, : bs + 1],
            indices=self.indices[step],
            q_to_req=self.arange[:bs],
            q_positions=positions,
            is_decode=True,
            prefix_lens=positions,
            max_q_len=1,
        )

    def stage(self, bs: int, rows: torch.Tensor, entry: torch.Tensor) -> None:
        for step in range(self.indptr.shape[0]):
            self.indptr[step, 1 : bs + 1] = torch.cumsum(entry + step + 1, 0)
        gather_kv_indices(self.page_table, rows, self.indptr[:, : bs + 1], self.indices)


class _QSASteps:
    """The block table and ring slots of the rows, shared by the steps, and one KV length row per step."""

    def __init__(self, backend: QSASparseAttnBackend, ctx, steps: int, bs_list: List[int]) -> None:
        # static scratch for the sparse kernels, so the captured graphs keep their addresses
        backend.init_capture_graph(ctx.page_table.shape[1], bs_list)
        max_bs = max(bs_list)
        self.page_table = ctx.page_table
        self.page_size = backend.page_size
        pages = -(-ctx.page_table.shape[1] // self.page_size)
        self.block_table = torch.zeros(max_bs, pages, dtype=torch.int32, device=ctx.device)
        self.slots = torch.zeros(max_bs, dtype=torch.int32, device=ctx.device)
        self.kv_len = torch.zeros(steps, max_bs, dtype=torch.int32, device=ctx.device)
        self.arange = torch.arange(max_bs + 1, dtype=torch.int32, device=ctx.device)
        self.cpu = torch.zeros(max_bs + 1, dtype=torch.int32)

    def metadata(self, bs: int, step: int, positions: torch.Tensor) -> QSASparseMetadata:
        return QSASparseMetadata(
            is_decode=True,
            last_indices=self.arange[:bs],
            qo_indptr_cpu=self.cpu[: bs + 1],
            kv_len_cpu=self.cpu[:bs],
            token_to_req=self.arange[:bs],
            cu_seqlens=self.arange[: bs + 1],
            seq_lens=self.kv_len[step, :bs],
            ring_slots=self.slots[:bs],
            block_table=self.block_table[:bs],
        )

    def stage(self, bs: int, rows: torch.Tensor, entry: torch.Tensor) -> None:
        self.block_table[:bs] = self.page_table[rows, :: self.page_size] // self.page_size
        self.slots[:bs] = rows
        for step in range(self.kv_len.shape[0]):
            self.kv_len[step, :bs] = entry + step + 1


class ChainGraphs:
    """Replays ``MTPProposer.propose``: the inputs are each request's table row, entry position and mrope delta, everything else derives from them in the graph."""

    @staticmethod
    def supports(backend) -> bool:
        return isinstance(backend, (TritonAttentionBackend, QSASparseAttnBackend))

    def __init__(self, proposer: MTPProposer, bs_list: List[int], pool) -> None:
        self.p = proposer
        ctx = proposer.ctx
        device, steps, max_bs = ctx.device, proposer.k - 1, max(bs_list)
        backend = proposer.chain_backend
        self.backend_steps = (
            _QSASteps(backend, ctx, steps, bs_list) if isinstance(backend, QSASparseAttnBackend) else _TritonSteps(ctx, steps, max_bs)
        )
        self.dummy_row = ctx.dummy_req.table_idx
        # rows: table row, entry position, mrope delta
        self.inputs = torch.zeros(3, max_bs, dtype=torch.int64, device=device)
        self.drafts = torch.zeros(max_bs, proposer.k, dtype=torch.int64, device=device)
        self.positions = torch.zeros(steps, max_bs, dtype=torch.int32, device=device)
        self.mrope = torch.zeros(steps, 3, max_bs, dtype=torch.int32, device=device) if proposer.mrope else None
        self.out_loc = torch.zeros(steps, max_bs, dtype=torch.int32, device=device)
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        # the metadata of these batches holds the decode scratch the graphs read
        self._batches: Dict[int, List[Batch]] = {}
        for bs in sorted(bs_list, reverse=True):
            self.inputs[0, :bs] = self.dummy_row
            batches = [self._batch(bs, s) for s in range(steps)]
            # outside the graph: triton compiles and the decode scratch is allocated here
            self._run(bs, batches)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=ctx.stream):
                self._run(bs, batches)
            if pool is None:
                pool = graph.pool()
            self.graphs[bs] = graph
            self._batches[bs] = batches

    def _batch(self, bs: int, step: int) -> Batch:
        batch = Batch(reqs=[None] * bs, phase="decode")
        batch.padded_reqs = batch.reqs
        batch.positions = self.positions[step, :bs]
        if self.mrope is not None:
            batch.mrope_positions = self.mrope[step, :, :bs]
        batch.out_loc = self.out_loc[step, :bs]
        batch.attn_metadata = self.backend_steps.metadata(bs, step, batch.positions)
        return batch

    def _run(self, bs: int, batches: List[Batch]) -> None:
        p = self.p
        page_table = p.ctx.page_table
        rows, entry, delta = self.inputs[:, :bs]
        for step in range(len(batches)):
            # step s drafts position c+s over keys [0, c+s]
            pos = entry + step
            self.positions[step, :bs] = pos
            if self.mrope is not None:
                self.mrope[step, :, :bs] = (pos + delta).unsqueeze(0)
            self.out_loc[step, :bs] = page_table[rows, pos]
        self.backend_steps.stage(bs, rows, entry)
        token, hidden = p.relay.d1[rows], p.relay.hidden[rows]
        self.drafts[:bs, 0] = token
        for step, batch in enumerate(batches):
            hidden = p._forward(batch, p.chain_backend, p.draft.embed_tokens.forward(token), hidden)
            token = p._draft_token(hidden)
            self.drafts[:bs, step + 1] = token

    def replay(self, reqs: List[Req], cs: List[int]) -> torch.Tensor | None:
        """The drafts [bs, k], or None when no captured batch size fits."""
        bs = len(reqs)
        padded = next((b for b in sorted(self.graphs) if b >= bs), None)
        if padded is None:
            return None
        pad = padded - bs
        # padded rows must point at the dummy page: a stale real row would get draft KV written into it
        host = torch.tensor(
            [[r.table_idx for r in reqs] + [self.dummy_row] * pad, cs + [0] * pad,
             [getattr(r, "mrope_delta", 0) for r in reqs] + [0] * pad],
            dtype=torch.int64, pin_memory=True,
        )
        self.inputs[:, :padded].copy_(host, non_blocking=True)
        self.graphs[padded].replay()
        return self.drafts[:bs].clone()


class SharedDraftGraphs:
    """Replays ``SharedKVMTPProposer.propose``: the draft stores no KV, so every one of the k steps
    reads the same [0, cached_len) window and one metadata set per batch size holds all k launches."""

    def __init__(self, proposer: "SharedKVMTPProposer", bs_list: List[int], pool) -> None:
        self.p = proposer
        ctx = proposer.ctx
        device, max_bs = ctx.device, max(bs_list)
        backend = proposer.chain_backend
        self.page_table = ctx.page_table
        self.dummy_row = ctx.dummy_req.table_idx
        # inputs: the table row and its entry position; the pending token rides its own buffer
        self.inputs = torch.zeros(2, max_bs, dtype=torch.int64, device=device)
        self.token = torch.zeros(max_bs, dtype=torch.int64, device=device)
        self.positions = torch.zeros(max_bs, dtype=torch.int32, device=device)
        # one window shared by every captured bs: its live entries are gathered per replay, in place
        self.steps = _TritonSteps(ctx, 1, max_bs)
        self.swa_map = proposer.kv.full_to_swa_map if getattr(proposer.kv, "swa_paged", False) else None
        self.swa_indices = (
            torch.zeros(1, max_bs * ctx.page_table.shape[1], dtype=torch.int32, device=device)
            if self.swa_map is not None
            else None
        )
        # one decode scratch pair for every captured bs, each graph using its own rows
        heads, dim = max(1, backend.num_q_heads), max(1, backend.max_head_dim)
        splits = max(backend._split_cap(bs) for bs in bs_list)
        self.attn_logits = torch.empty((max_bs, heads, splits, dim), dtype=torch.float32, device=device)
        self.attn_lse = torch.empty((max_bs, heads, splits), dtype=torch.float32, device=device)
        self.drafts = torch.zeros(max_bs, proposer.k, dtype=torch.int64, device=device)
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        # the metadata of these batches holds the decode scratch the graphs read
        self._batches: Dict[int, Batch] = {}
        for bs in sorted(bs_list, reverse=True):
            # warmup rows gather one dummy key: a zero-length window is a decode path nothing else runs
            self.inputs[0, :bs] = self.dummy_row
            self.inputs[1, :bs] = 1
            batch = self._batch(bs)
            # outside the graph: triton compiles, and the shared scratch must already fit every bs
            self._run(bs, batch)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=ctx.stream):
                self._run(bs, batch)
            if pool is None:
                pool = graph.pool()
            self.graphs[bs] = graph
            self._batches[bs] = batch

    def _batch(self, bs: int) -> Batch:
        batch = Batch(reqs=[None] * bs, phase="decode")
        batch.padded_reqs = batch.reqs
        batch.positions = self.positions[:bs]
        batch.attn_metadata = self.steps.metadata(bs, 0, batch.positions)
        batch.attn_metadata.swa_indices = self.swa_indices[0] if self.swa_indices is not None else None
        # the shared scratch's prefix: _ensure_decode_scratch sees it and allocates nothing in capture
        batch.attn_metadata.attn_logits = self.attn_logits[:bs]
        batch.attn_metadata.attn_lse = self.attn_lse[:bs]
        return batch

    def _run(self, bs: int, batch: Batch) -> None:
        p = self.p
        rows, entry = self.inputs[:, :bs]
        # the window [0, entry) serves every step: the draft writes no KV between them, so only the
        # carried token and hidden move
        self.positions[:bs] = entry
        self.steps.indptr[0, 1 : bs + 1] = torch.cumsum(entry, 0)
        # the sliding-window lookup rides the gather: only the live entries are translated, into the
        # same shared buffer every graph's SWA layers read
        gather_kv_indices(
            self.page_table, rows, self.steps.indptr[:, : bs + 1], self.steps.indices,
            self.swa_map, self.swa_indices,
        )
        token = self.token[:bs]
        hidden = p.seed_hidden[rows]
        for step in range(p.k):
            out = p._forward(batch, p.chain_backend, p.draft.embed_tokens.forward(token), hidden)
            token = p._draft_token(out)
            hidden = p._relayed(out)
            self.drafts[:bs, step] = token

    def replay(self, reqs: List[Req], cs: List[int], pending: torch.Tensor) -> torch.Tensor | None:
        """The drafts [bs, k], or None when no captured batch size fits."""
        bs = len(reqs)
        padded = next((b for b in sorted(self.graphs) if b >= bs), None)
        if padded is None:
            return None
        pad = padded - bs
        # padded rows draft over the dummy page's first key: their drafts are discarded
        host = torch.tensor(
            [[r.table_idx for r in reqs] + [self.dummy_row] * pad, cs + [1] * pad],
            dtype=torch.int64, pin_memory=True,
        )
        self.inputs[:, :padded].copy_(host, non_blocking=True)
        self.token[bs:padded].zero_()
        self.token[:bs].copy_(pending)
        self.graphs[padded].replay()
        return self.drafts[:bs].clone()
