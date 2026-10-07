"""Engram tables streamed directly from checkpoint shards.

Each Engram layer owns a ``[384M, 256]`` fp8 table (98 GB) plus a ``[384M, 8]`` ue8m0 scale
tensor (3 GB). The tables stay where the checkpoint put them: ``EngramDiskTable`` maps the shard
tensor as one ``RowStore`` extent and reads the 24 rows a token needs per fill (io_uring, O_DIRECT,
duplicates deduplicated), while the scales are host-resident and gathered next to the values, so
the staged row is exactly the pool row format ``fp8_e8m0_b32`` and the device dequantizes it with
the same kernel the KV pools use.

Protocol (the Qwen3.8 PLE disk table's, ``models/qwen4_exp/ple_disk.py``): the engine wraps every
dispatch in ``EngramHost.forward_host_ctx``. Eager: hash + stage + read before the launch. CUDA-graph
decode: the sampled token lives on the device under overlap scheduling, so the host reads it back,
fills the graph's pinned staging and releases a per-table flag the captured ``lookup`` WAITs on (or,
without stream memops, fills before the replay). The fill runs on a worker thread that is started
BEFORE the graph is launched: with a graph this large (40 layers) the driver's launch call itself
blocks once the GPU sits on the WAIT with the rest of the graph queued behind it, so a fill scheduled
after the launch returns would deadlock. The worker works from an immutable dispatch snapshot plus
the readback event (which the GPU completes on its own), and the dispatch context waits for it after
the launch so its failure surfaces before the step's output is consumed.
"""

from __future__ import annotations

import json
import os
import struct
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

import torch
from freetoken.core import Batch
from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.kvcache.dsv4.v41_row_format import FP8_E8M0_B32
from freetoken.utils import init_logger

from .engram import EngramHash

logger = init_logger(__name__)


@dataclass(frozen=True)
class _TensorExtent:
    """Where a safetensors tensor's bytes sit in its shard: a contiguous ``[offset, offset + nbytes)``."""

    path: str
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str  # the safetensors dtype string (e.g. "F8_E4M3")


def _tensor_extent(folder: str, name: str) -> _TensorExtent:
    """Locate ``name`` in the checkpoint directory without reading it (the table is streamed in place)."""
    from freetoken.models.loader import safetensors_weight_map

    path = os.path.join(folder, safetensors_weight_map(folder)[name])
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    meta = header[name]
    start, end = meta["data_offsets"]
    return _TensorExtent(path, 8 + n + start, end - start, tuple(meta["shape"]), meta["dtype"])


@dataclass(frozen=True)
class EngramRowSource:
    """The fp8 table tensor in its shard (one extent) and its host-resident scales."""

    path: str
    base: int
    num_rows: int
    head_dim: int
    scales: torch.Tensor  # [num_rows, head_dim // 32] uint8 (e8m0 codes), CPU


def engram_row_source(folder: str, layer_id: int) -> EngramRowSource:
    """Locate a layer's table in whatever safetensors files the checkpoint directory holds: the HF
    shards (through the index) or the shards an FTW conversion copied next to it (through their headers)."""
    from safetensors import safe_open

    weight = _tensor_extent(folder, f"layers.{layer_id}.engram.embed.weight")
    if weight.dtype != "F8_E4M3":
        raise ValueError(f"engram table of layer {layer_id} is {weight.dtype}, expected F8_E4M3")
    rows, head_dim = weight.shape
    scale = _tensor_extent(folder, f"layers.{layer_id}.engram.embed.scale")
    with safe_open(scale.path, framework="pt", device="cpu") as f:
        scales = f.get_tensor(f"layers.{layer_id}.engram.embed.scale").view(torch.uint8).contiguous()
    if tuple(scales.shape) != (rows, head_dim // 32):
        raise ValueError(f"engram scales of layer {layer_id} are {tuple(scales.shape)}, expected {(rows, head_dim // 32)}")
    return EngramRowSource(weight.path, weight.offset, rows, head_dim, scales)


class EngramDiskTable:
    """One layer's table: stages ``[T, n_cols]`` rows into pinned memory on the host, dequantizes on
    the device. Implements the model's ``EngramTable`` protocol (``lookup``)."""

    def __init__(
        self, source: EngramRowSource, n_cols: int, device: torch.device, *, max_graph_rows: int, max_extend_tokens: int,
        use_io_uring: bool = True,
    ) -> None:
        from freetoken.kernel.row_store import RowStore

        self.head_dim = source.head_dim
        self.n_cols = n_cols
        self.num_rows = source.num_rows
        self.row_bytes = FP8_E8M0_B32.row_bytes(source.head_dim)  # values | scales
        self.scales = source.scales
        self.device = device
        self.store = RowStore(
            paths=[source.path], extent_file=[0], extent_base=[source.base], rows_per_extent=source.num_rows,
            row_bytes=source.head_dim, row_stride=source.head_dim, use_io_uring=use_io_uring,
        )
        # the graph staging is allocated up front (pinned alloc inside capture is illegal) and
        # outlives any one capture so a re-capture keeps the same pointers; the eager staging
        # grows to the largest prefill chunk seen (never inside a capture)
        self.max_graph_rows = max_graph_rows
        self._graph_pinned = alloc_pinned_tensor(max_graph_rows * n_cols * self.row_bytes, dtype=torch.uint8)
        self._graph_pinned.zero_()
        self._graph_dev = torch.empty(max_graph_rows * n_cols * self.row_bytes, dtype=torch.uint8, device=device)
        # completes once the last graph launch that read ``_graph_pinned`` has run (recorded by the
        # host coordinator after the launch: a record inside the capture would not be observable)
        self.graph_consumed = torch.cuda.Event()
        # the overlap scheduler stages batch k+1 while batch k's lookup copy may still be queued: the
        # eager staging alternates two pinned buffers, and each is rewritten only once the H2D copy
        # that last read it has run (``_eager_read[i]``, recorded right after that copy)
        self._eager_pinned: list[torch.Tensor] = []
        self._eager_read = [torch.cuda.Event(), torch.cuda.Event()]
        self._eager_slot = 0
        self._eager_dev = None
        self._grow_eager(max_extend_tokens)
        self.flag = alloc_pinned_tensor(1, dtype=torch.int64)
        self.flag.zero_()
        self.wait_sync = False  # set by the host coordinator after its probe

    def _grow_eager(self, num_tokens: int) -> None:
        nbytes = num_tokens * self.n_cols * self.row_bytes
        if self._eager_pinned and self._eager_pinned[0].numel() >= nbytes:
            return
        for event in self._eager_read:
            event.synchronize()
        self._eager_pinned = [alloc_pinned_tensor(nbytes, dtype=torch.uint8) for _ in self._eager_read]
        for buf in self._eager_pinned:
            buf.zero_()  # a warmup prefill stages nothing and reads whatever sits here
        self._eager_dev = torch.empty(nbytes, dtype=torch.uint8, device=self.device)

    # ----- host -----
    def stage(self, row_ids: torch.Tensor, *, graph: bool) -> None:
        """Queue ``row_ids [T, n_cols]`` (int64, host): values through the store, scales from the resident tensor."""
        n = row_ids.numel()
        if graph:
            self.graph_consumed.synchronize()
            pinned = self._graph_pinned
            assert n <= self.max_graph_rows * self.n_cols, f"graph staging holds {self.max_graph_rows} rows, batch has {n // self.n_cols}"
        else:
            self._grow_eager(n // self.n_cols)
            self._eager_slot ^= 1
            self._eager_read[self._eager_slot].synchronize()
            pinned = self._eager_pinned[self._eager_slot]
        ids = row_ids.reshape(-1).to(torch.int64).contiguous()
        rows = pinned[: n * self.row_bytes].view(n, self.row_bytes)
        self.store.stage_rows(ids.data_ptr(), n, rows.data_ptr(), self.row_bytes)
        rows[:, self.head_dim :].copy_(self.scales.index_select(0, ids))

    def flush(self, *, signal: bool) -> None:
        self.store.flush(self.flag.data_ptr() if signal else 0)

    # ----- device (EngramTable protocol) -----
    def lookup(self, num_tokens: int) -> torch.Tensor:
        from freetoken.kernel.row_store import wait_reset
        from freetoken.kernel.triton.dsv41.pack import unpack_rows

        stream = torch.cuda.current_stream(self.device)
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and self.wait_sync:
            wait_reset(stream, self.flag)
        pinned, dev = (self._graph_pinned, self._graph_dev) if capturing else (self._eager_pinned[self._eager_slot], self._eager_dev)
        n = num_tokens * self.n_cols
        nbytes = n * self.row_bytes
        dev[:nbytes].copy_(pinned[:nbytes], non_blocking=True)
        if not capturing:
            self._eager_read[self._eager_slot].record(stream)
        values = unpack_rows(dev[:nbytes].view(n, self.row_bytes), FP8_E8M0_B32, self.head_dim)
        return values.view(num_tokens, self.n_cols * self.head_dim)


def _context(ids: torch.Tensor, position: int, width: int) -> torch.Tensor:
    """The up-to-``width`` raw ids before ``position`` (fewer at the sequence start)."""
    return ids[max(0, position - width) : position]


class EngramHost:
    """The per-model coordinator: hashes each batch's token runs and fills every layer's table.

    The token range a prefill hashes is the attention metadata's encoder segments -- the one
    execution plan the scheduler, the attention layers and the model already agreed on. A graph
    decode's fill runs on a worker thread from an immutable dispatch snapshot (position + the
    ``max_ngram - 1`` preceding ids per row) plus the readback of the sampled token; the request
    objects themselves advance on the main thread right after dispatch and are never read late.
    """

    def __init__(self, hash: EngramHash, tables: Sequence[EngramDiskTable], device: torch.device, *, sync_mode: str = "auto") -> None:
        from freetoken.kernel.row_store import probe_wait_sync

        self.hash = hash
        self.tables = list(tables)
        self.device = device
        self.wait_sync = probe_wait_sync(sync_mode, device)
        for t in self.tables:
            t.wait_sync = self.wait_sync
        self._token_readback = alloc_pinned_tensor(min(t.max_graph_rows for t in self.tables), dtype=torch.int32)
        self._readback_event = torch.cuda.Event()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engram-fill")
        self._pending: Future | None = None
        logger.info_rank0(f"Engram disk backend: {self.tables[0].store.io_backend()}, {'wait-sync' if self.wait_sync else 'launch-gating'}")

    def close(self) -> None:
        self._worker.shutdown(wait=True)

    def __del__(self) -> None:
        worker = getattr(self, "_worker", None)
        if worker is not None:
            worker.shutdown(wait=False)

    def fill(self, runs: Sequence[tuple[torch.Tensor, torch.Tensor]], *, graph: bool) -> None:
        """``runs``: per request ``(context ids, token ids)`` in batch order; stages every layer's rows."""
        ids = torch.cat([self.hash.row_ids(tokens, ctx) for ctx, tokens in runs]) if runs else torch.empty(0, len(self.tables), self.hash.n_cols, dtype=torch.int64)
        for l, table in enumerate(self.tables):
            table.stage(ids[:, l], graph=graph)
            table.flush(signal=graph and self.wait_sync)

    @torch.inference_mode()
    def _deferred_fill(self, bs: int, histories: list[tuple[int, ...]]) -> None:
        """Worker side of a graph decode: wait for the sampled tokens, hash them against the dispatch
        snapshot, stage, and release the tables' flags. Reads nothing that the main thread mutates.
        Inference mode is thread-local; staging buffers were created under it by the server."""
        try:
            self._readback_event.synchronize()
            tokens = self._token_readback[:bs].to(torch.int64)
            runs = [(torch.tensor(hist, dtype=torch.int64), tokens[i : i + 1]) for i, hist in enumerate(histories)]
            self.fill(runs, graph=True)
        except BaseException:
            from freetoken.kernel.row_store import signal

            for t in self.tables:  # unblock the stream before surfacing; the step's output is discarded
                signal(t.flag)
            raise

    def _await_pending(self) -> None:
        """Surface the deferred fill's outcome (it depends only on GPU work already enqueued, so this
        cannot wait on the host)."""
        if self._pending is not None:
            pending, self._pending = self._pending, None
            pending.result()

    def host_fill_batch(self, batch: Batch, use_graph: bool) -> None:
        """Stage this batch's rows: eagerly, or -- a graph decode under flag-sync -- on the worker
        thread once the sampled tokens have been read back."""
        self._await_pending()
        width = self.hash.max_ngram - 1
        if batch.is_decode:
            reqs = list(batch.padded_reqs)
            # the dispatch snapshot: each row's position context, taken before anything advances
            histories = [tuple(_context(r.input_ids, r.device_len - 1, width).tolist()) for r in reqs]
            if use_graph and self.wait_sync:
                bs = batch.padded_size
                self._token_readback[:bs].copy_(batch.input_ids, non_blocking=True)
                self._readback_event.record(torch.cuda.current_stream(self.device))
                self._pending = self._worker.submit(self._deferred_fill, bs, histories)
                return
            tokens = batch.input_ids.to("cpu").to(torch.int64)
            runs = [(torch.tensor(hist, dtype=torch.int64), tokens[i : i + 1]) for i, hist in enumerate(histories)]
            self.fill(runs, graph=use_graph)
            return
        # prefill: the encoder segments are the token range every consumer of this forward executes
        segments = batch.attn_metadata.segments
        assert segments is not None and len(segments) == len(batch.reqs)
        runs = [(_context(r.input_ids, seg.start_pos, width), r.input_ids[seg.start_pos : seg.end]) for r, seg in zip(batch.reqs, segments)]
        self.fill(runs, graph=False)

    @contextmanager
    def forward_host_ctx(self, batch: Batch, use_graph: bool):
        self.host_fill_batch(batch, use_graph)
        yield
        # the launch has been enqueued: wait for the deferred fill here so a failure surfaces before
        # this step's output is consumed (the fill only waits on the GPU readback, never on the host)
        self._await_pending()
        if use_graph:
            # only after this step's fill: the next fill waits on this launch, not the fill on it
            stream = torch.cuda.current_stream(self.device)
            for t in self.tables:
                t.graph_consumed.record(stream)


__all__ = ["EngramDiskTable", "EngramHost", "EngramRowSource", "engram_row_source"]
