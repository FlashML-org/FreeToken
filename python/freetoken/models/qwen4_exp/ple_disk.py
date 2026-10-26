"""Disk-backed PLE table (--ple-backend disk): the C++ store hashes n-gram windows and batch-reads rows from the checkpoint's fp8 or bf16 shard tensors into pinned staging; the captured ``lookup`` is a fixed-shape H2D copy + dequant.

Hash windows are pure functions of ``req.input_ids`` + ``device_len`` (prefix hits, restores and COW forks need no bookkeeping); the decode input token lives device-side under overlap scheduling and is read back here.
"""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

import safetensors
import torch

from freetoken.mm import MM_PAD_SHIFT_VALUE, restore_placeholder

from freetoken.core import Batch
from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.utils import init_logger

from .weight import (
    _PLE_SCALE_SUFFIX,
    _PLE_SHARD_RE,
    _PLE_ST_DTYPES,
    _ple_table_files,
    _safetensors_header,
)

_IO_URING_ENV = "FREETOKEN_PLE_IO_URING"
_SYNC_ENV = "FREETOKEN_PLE_SYNC"  # auto | wait | gate

logger = init_logger(__name__)


def _context(ids: torch.Tensor, position: int, eos: int) -> list[int]:
    """The two token ids before ``position``; eos pads past the start."""
    return [int(ids[position - 2]) if position >= 2 else eos,
            int(ids[position - 1]) if position >= 1 else eos]


@dataclass(frozen=True)
class PleRowSource:
    """On-disk row layout: equal extents, row i of an extent at ``base + i * row_stride`` (a repacked flat file is one extent with its own stride)."""

    paths: list[str]
    extent_file: list[int]
    extent_base: list[int]
    rows_per_extent: int
    row_bytes: int
    row_stride: int
    scale: float
    # packed ggml rows (a GGUF table) of ``row_values`` values each; None for float rows of ``row_dtype``
    ggml_type: int | None = None
    row_values: int | None = None
    row_dtype: torch.dtype = torch.float8_e4m3fn

    @property
    def total_rows(self) -> int:
        return len(self.extent_base) * self.rows_per_extent


def source_from_safetensors(folder: str) -> PleRowSource:
    """Map the checkpoint's ``ngram_embedding.shard_<i>`` tensors in place: one extent per shard, no copy."""
    rows = cols = 0
    dtype: torch.dtype | None = None
    scale: torch.Tensor | None = None
    paths: list[str] = []
    path_idx: dict[str, int] = {}
    shards: dict[int, tuple[int, int]] = {}
    for path in _ple_table_files(folder):
        header, base = _safetensors_header(path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key.endswith(_PLE_SCALE_SUFFIX):
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    scale = f.get_tensor(key).reshape(())
                continue
            match = _PLE_SHARD_RE.search(key)
            if match is None:
                continue
            shard_dtype = _PLE_ST_DTYPES.get(meta["dtype"])
            if shard_dtype is None or dtype not in (None, shard_dtype):
                raise ValueError(f"PLE shard {key} has dtype {meta['dtype']}, expected one of {sorted(_PLE_ST_DTYPES)} for every shard")
            dtype = shard_dtype
            if rows and tuple(meta["shape"]) != (rows, cols):
                raise ValueError(f"PLE shard {key} is {meta['shape']}, expected {[rows, cols]}")
            rows, cols = meta["shape"]
            if path not in path_idx:
                path_idx[path] = len(paths)
                paths.append(path)
            idx = int(match.group("shard"))
            if idx in shards:
                raise ValueError(f"duplicate PLE shard {idx} in {path}")
            shards[idx] = (path_idx[path], base + meta["data_offsets"][0])
    if sorted(shards) != list(range(len(shards))) or not shards:
        raise ValueError(f"PLE shard indices are not contiguous 0..N-1: {sorted(shards)[:8]}")
    if scale is None:
        if dtype is torch.float8_e4m3fn:
            raise ValueError("PLE table has no weight_scale")
        scale = torch.tensor(1.0)
    order = [shards[i] for i in range(len(shards))]
    row_bytes = cols * dtype.itemsize
    return PleRowSource(paths, [f for f, _ in order], [b for _, b in order], rows, row_bytes, row_bytes, float(scale),
                        row_values=cols, row_dtype=dtype)


def source_from_gguf(model_path: str) -> PleRowSource:
    """The GGUF ``per_layer_token_embd`` table in place: one extent of packed ggml rows."""
    from freetoken.layers.quantization.ggml_types import row_bytes

    from .gguf import ple_table_info

    info = ple_table_info(model_path)
    values, rows = info.ne
    nbytes = row_bytes(info.ggml_type, values)
    return PleRowSource([info.path], [0], [info.offset], rows, nbytes, nbytes, 1.0, info.ggml_type, values)


def resolve_row_source(model_path: str) -> PleRowSource:
    """Pick the row source for a checkpoint; the seam where a repacked format would plug in."""
    from freetoken.models.gguf.reader import gguf_config_source

    if (gguf_src := gguf_config_source(model_path)) is not None:
        return source_from_gguf(gguf_src)
    from freetoken.utils import download_hf_weight

    return source_from_safetensors(download_hf_weight(model_path))


class DiskRowTable:
    """``PLETableBackend`` whose rows are read from disk per fill (--ple-backend disk)."""

    def __init__(
        self,
        source: PleRowSource,
        hash_constants: dict,
        *,
        max_graph_rows: int = 256,
        max_extend_tokens: int = 8192,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        from freetoken.kernel.row_store import PleStore
        from freetoken.layers.quantization.ggml_types import float_dtype

        self.num_rows = source.total_rows
        self.head_dim = source.row_values or source.row_bytes
        self._ggml_type = source.ggml_type
        self._row_bytes = source.row_bytes
        # float rows (fp8, or F32 / F16 / BF16 in a GGUF) are only cast; quantized ggml rows go through ggml
        self._row_dtype = source.row_dtype if source.ggml_type is None else float_dtype(source.ggml_type)
        self.dtype = dtype
        self.heads = int(hash_constants["num_ngram_heads"])
        self.scale = source.scale
        self.eos_token_id = int(hash_constants["eos_token_id"])
        self.image_token_id = hash_constants.get("image_token_id")
        sizes = [int(x) for x in hash_constants["per_head_vocab_sizes"]]
        offsets = [int(x) for x in hash_constants["per_head_offsets"]]
        need = max(o + s for o, s in zip(offsets, sizes))
        if need > source.total_rows:
            raise ValueError(
                f"PLE row source holds {source.total_rows} rows but the hash addresses {need}; incomplete checkpoint?"
            )
        self._store = PleStore(
            paths=list(source.paths),
            extent_file=list(source.extent_file),
            extent_base=list(source.extent_base),
            rows_per_extent=source.rows_per_extent,
            row_bytes=source.row_bytes,
            row_stride=source.row_stride,
            multipliers=[int(x) for x in hash_constants["layer_multipliers"]],
            head_vocab_sizes=sizes,
            head_offsets=offsets,
            eos_token_id=self.eos_token_id,
            use_io_uring=os.getenv(_IO_URING_ENV, "1") != "0",
        )
        self._device = torch.device("cuda", torch.cuda.current_device())
        self._token_bytes = self.heads * self._row_bytes
        # allocated up front: pinned alloc inside stream capture is illegal; one replay consumes it at a time
        self._graph_pinned = alloc_pinned_tensor(max_graph_rows * self._token_bytes, dtype=torch.uint8)
        self._graph_pinned.zero_()  # padded decode lanes read whatever sits here
        # outlives any one graph: a cache rebuild recaptures against the same pointer
        self._graph_dev = torch.empty(
            max_graph_rows * self._token_bytes, dtype=torch.uint8, device=self._device
        )
        # completes once the last graph launch that read ``_graph_pinned`` has run (recorded by the
        # dispatch context after the step's fill: a record inside the capture would not be observable)
        self._graph_consumed = torch.cuda.Event()
        eager_bytes = max_extend_tokens * self._token_bytes
        # the overlap scheduler fills batch k+1 while batch k's lookup copy may still be queued: the
        # eager staging alternates two pinned buffers, and each is rewritten only once the H2D copy
        # that last read it has run (``_eager_read[i]``, recorded right after that copy)
        self._eager_pinned = [alloc_pinned_tensor(eager_bytes, dtype=torch.uint8) for _ in range(2)]
        for buf in self._eager_pinned:
            buf.zero_()  # the warmup prefill stages nothing and reads whatever sits here
        self._eager_read = [torch.cuda.Event(), torch.cuda.Event()]
        self._eager_slot = 0
        self._eager_dev = torch.empty(eager_bytes, dtype=torch.uint8, device=self._device)
        # probe picks flag-sync (graph WAITs at the consume, host fills then signals) or launch-gating
        from freetoken.kernel.row_store import probe_wait_sync

        self._wait_sync = probe_wait_sync(os.getenv(_SYNC_ENV, "auto"), self._device)
        # one flag for all graphs: the readback event orders a fill after the previous graph, so signals never overlap
        from freetoken.kernel.row_store import alloc_sync_flag

        self._flag = alloc_sync_flag() if self._wait_sync else None
        self._token_readback = alloc_pinned_tensor(max_graph_rows, dtype=torch.int32)
        self._readback_event = torch.cuda.Event()
        sync = "wait-sync" if self._wait_sync else "launch-gating"
        logger.info_rank0(f"PLE disk backend: {self._store.io_backend()}, {sync}")
        self._worker = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="ple-fill")
            if self._wait_sync and torch.version.hip is not None else None
        )
        self._pending: Future | None = None
        self._failed = False

    def _fill_decode(self, histories: Sequence[Sequence[int]], bs: int, steps: int) -> None:
        try:
            self._readback_event.synchronize()
            tokens = self._token_readback[: bs * steps].to(torch.int64).tolist()
            runs = [torch.tensor([*history, *tokens[i * steps : (i + 1) * steps]], dtype=torch.int64)
                    for i, history in enumerate(histories)]
            self.fill(runs, graph=True)
        except BaseException:
            from freetoken.kernel.row_store import signal

            # unblock the stream before surfacing; the step's output is discarded
            signal(self._flag)
            raise

    def _start_fill(self, histories: Sequence[Sequence[int]], bs: int, steps: int) -> None:
        @torch.inference_mode()
        def _complete() -> None:
            self._fill_decode(histories, bs, steps)

        # Like Engram, release HIP's WAIT independently of a queue-blocked graph launch.
        self._pending = self._worker.submit(_complete)

    def _await_pending(self) -> None:
        if self._pending is not None:
            pending, self._pending = self._pending, None
            try:
                pending.result()
            except BaseException:
                self._failed = True
                if self._worker is not None:
                    # An interrupted waiter must drain the fill before recording its consumed event.
                    self._worker.shutdown(wait=True)
                raise

    def __del__(self) -> None:
        worker = getattr(self, "_worker", None)
        if worker is not None:
            worker.shutdown(wait=False)

    # ---------------- host side (engine thread, before the forward launches) ----------------

    def fill(self, runs: Sequence[torch.Tensor], *, graph: bool) -> None:
        """Stage per-request token runs (two context ids, then the new tokens) in batch order."""
        if graph:
            self._graph_consumed.synchronize()
            pinned = self._graph_pinned
        else:
            self._eager_slot ^= 1
            self._eager_read[self._eager_slot].synchronize()
            pinned = self._eager_pinned[self._eager_slot]
        offset = 0
        for run in runs:
            self._store.stage(run.data_ptr(), run.numel() - 2, pinned.data_ptr() + offset * self._token_bytes)
            offset += run.numel() - 2
        self._store.flush(self._flag.data_ptr() if graph and self._wait_sync else 0)

    def _ple_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.image_token_id is None:
            return input_ids
        return restore_placeholder(input_ids, self.image_token_id)

    def _ple_context(self, ids: torch.Tensor, position: int) -> list[int]:
        if self.image_token_id is None:
            return _context(ids, position, self.eos_token_id)
        return [self.image_token_id if t >= MM_PAD_SHIFT_VALUE else t for t in _context(ids, position, self.eos_token_id)]

    def host_fill_batch(self, batch: Batch, use_graph: bool):
        """Stage this batch's rows; returns the deferred fill for wait-sync without a worker."""
        if torch.version.hip is not None:
            self._await_pending()
            if self._failed:
                raise RuntimeError("PLE disk table failed; recreate the engine before dispatch")
        eos = self.eos_token_id
        # decode and verify: rows_per_req tokens per request, the first at device_len - rows_per_req
        steps = getattr(batch, "rows_per_req", None) or 1
        if batch.is_decode or steps > 1:
            reqs = list(batch.reqs)
            histories = [self._ple_context(r.input_ids, r.device_len - steps) for r in reqs]
            if use_graph and self._wait_sync:
                bs = batch.padded_size
                self._token_readback[: bs * steps].copy_(batch.input_ids, non_blocking=True)
                self._readback_event.record(torch.cuda.current_stream(self._device))
                if self._worker is not None:
                    self._start_fill(histories, bs, steps)
                    return None

                def _complete() -> None:
                    self._fill_decode(histories, bs, steps)

                return _complete
            # launch-gating: this D2H is the step's readback and orders the fill after sampling
            tokens = batch.input_ids.to("cpu").to(torch.int64).tolist()
            runs = [torch.tensor([*history, *tokens[i * steps : (i + 1) * steps]], dtype=torch.int64)
                    for i, history in enumerate(histories)]
            self.fill(runs, graph=use_graph)
            return None
        runs = [
            torch.cat((
                torch.tensor(self._ple_context(req.input_ids, req.cached_len), dtype=torch.int64),
                self._ple_ids(req.input_ids[req.cached_len : req.device_len]).to(torch.int64),
            ))
            for req in batch.padded_reqs
        ]
        self.fill(runs, graph=False)
        return None

    @contextmanager
    def forward_host_ctx(self, batch: Batch, use_graph: bool):
        """Around one dispatch: stage on enter, finish the fill+signal on exit."""
        deferred = self.host_fill_batch(batch, use_graph)
        if self._worker is not None:
            try:
                yield
            except BaseException:
                self._failed = True
                raise
            finally:
                try:
                    self._await_pending()
                finally:
                    # The worker must finish before recording the event it waits on at the next fill.
                    if use_graph:
                        self._graph_consumed.record(torch.cuda.current_stream(self._device))
            return
        yield
        # no try/finally: a failed launch leaves no WAIT pending, so the fill must not run
        if deferred is not None:
            deferred()
        if use_graph:
            # only after this step's fill: the next fill waits on this launch, not the fill on it
            self._graph_consumed.record(torch.cuda.current_stream(self._device))

    # ---------------- device side (PLETableBackend protocol) ----------------

    def lookup(self, row_ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        rows = row_ids.shape[0]
        stream = torch.cuda.current_stream(self._device)
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and self._wait_sync:
            from freetoken.kernel.row_store import wait_reset

            wait_reset(stream, self._flag)
        pinned, dev = (
            (self._graph_pinned, self._graph_dev) if capturing else (self._eager_pinned[self._eager_slot], self._eager_dev)
        )
        nbytes = rows * self._token_bytes
        dev[:nbytes].copy_(pinned[:nbytes], non_blocking=True)
        if not capturing:
            self._eager_read[self._eager_slot].record(stream)
        if self._row_dtype is not None:
            # copy=True: rows already in self.dtype must not alias the staging buffer the next fill reuses
            values = dev[:nbytes].view(self._row_dtype).to(self.dtype, copy=True)
        else:
            from freetoken.kernel import ggml

            values = ggml.dequantize(dev[:nbytes].view(-1, self._row_bytes), self._ggml_type, self.head_dim, self.dtype)
        if self.scale != 1.0:
            values = values * self.scale
        values = values.view(*row_ids.shape[:-1], -1)
        if out is None:
            return values
        out.copy_(values)
        return out

    def prefetch(self, row_ids: torch.Tensor) -> None:
        return None
