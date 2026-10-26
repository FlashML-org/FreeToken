from __future__ import annotations

import ctypes
import functools
import gc
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal

import torch
from freetoken.core import Batch, Req, get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.utils import init_logger, mem_GB
from freetoken.utils.progress import emit_progress
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    # [3, bs * steps] t/h/w rope positions; allocated only for mrope models (else None).
    mrope_positions: torch.Tensor | None
    table_idx: torch.Tensor  # per-request slot id for GatedDeltaNet state gather/scatter
    # GDN query indptr = arange(bs+1) * steps; a constant per captured bs, filled once.
    fla_cu_seqlens: torch.Tensor
    steps: int = 1

    @classmethod
    def init(cls, bs: int, device: torch.device, mrope: bool = False, steps: int = 1) -> GraphCaptureBuffer:
        n = bs * steps
        return GraphCaptureBuffer(
            input_ids=torch.zeros(n, dtype=torch.int32, device=device),
            out_loc=torch.zeros(n, dtype=torch.int32, device=device),
            positions=torch.zeros(n, dtype=torch.int32, device=device),
            mrope_positions=(
                torch.zeros(3, n, dtype=torch.int32, device=device) if mrope else None
            ),
            table_idx=torch.zeros(bs, dtype=torch.int32, device=device),
            fla_cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device) * steps,
            steps=steps,
        )

    def set_batch(self, batch: Batch) -> None:
        from freetoken.attention.linear import FLAMetadata

        bs = batch.padded_size
        _rows = slice(bs * self.steps)
        batch.input_ids = self.input_ids[_rows]
        batch.out_loc = self.out_loc[_rows]
        batch.positions = self.positions[_rows]
        if self.mrope_positions is not None:
            batch.mrope_positions = self.mrope_positions[:, _rows]
        batch.linear_table_idx = self.table_idx[:bs]
        # Decode GDN metadata reads the persistent cu_seqlens (constant arange) and the
        # persistent table_idx slot map, so the captured kernels see stable addresses.
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.fla_cu_seqlens[: bs + 1], cache_indices=self.table_idx[:bs]
        )

    def copy_from(self, batch: Batch) -> None:
        _rows = slice(batch.padded_size * self.steps)
        self.input_ids[_rows] = batch.input_ids
        if batch.out_loc is not None:
            self.out_loc[_rows] = batch.out_loc
        self.positions[_rows] = batch.positions
        if self.mrope_positions is not None:
            self.mrope_positions[:, _rows] = batch.mrope_positions
        if batch.linear_table_idx is not None:
            self.table_idx[: batch.padded_size] = batch.linear_table_idx


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    candidates = [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
    return [bs for bs in candidates if bs <= cuda_graph_max_bs]


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


@functools.cache
def _cu_graph_upload():
    if sys.platform != "linux" or torch.version.hip is not None:
        return None
    fn = ctypes.CDLL("libcuda.so.1").cuGraphUpload
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


def _upload_graph(graph: torch.cuda.CUDAGraph, stream: torch.cuda.Stream) -> None:
    """Upload the graph at capture time.

    Otherwise the first replay pushes every node behind the graph's own stream WAIT, and a large graph then blocks cudaGraphLaunch until the WAIT is released. PLE wait-sync releases it only after the launch returns, so that deadlocks.
    """
    upload = _cu_graph_upload()
    if upload is None:
        return
    rc = upload(graph.raw_cuda_graph_exec(), stream.cuda_stream)
    if rc != 0:
        raise RuntimeError(f"cuGraphUpload failed with CUresult {rc}")
    stream.synchronize()


class GraphRunner:
    """CUDA graphs of ``forward`` over padded batches of ``steps`` rows per request, one graph per batch size.

    ``forward`` returns one tensor or a tuple of tensors; each is copied into a buffer sized by the largest
    captured batch, and ``replay`` returns the real requests' rows of each.
    """

    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        forward: Callable[[], Any],
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        dummy_req: Req,
        moe_offload_cache: OffloadMoeCache | None = None,
        mrope: bool = False,
        *,
        steps: int = 1,
        phase: Literal["decode", "verify"] = "decode",
        ctx_override: Dict[str, Any] | None = None,
        batch_hook: Callable[[Batch], None] | None = None,
        pool: Any = None,
        out_dtype: torch.dtype | None = None,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        # the padded rows replay the dummy request, so it must carry one row per step
        assert steps == 1 or dummy_req.extend_len == steps, (steps, dummy_req.extend_len)
        self.forward = forward
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.moe_offload_cache = moe_offload_cache
        self.mrope = mrope
        self.stream = stream
        self.device = device
        self.steps = steps
        self.phase = phase
        self.ctx_override = ctx_override or {}
        self.batch_hook = batch_hook
        self.graph_pool = pool
        self.out_dtype = out_dtype
        self._out: List[torch.Tensor] | None = None
        self._out_is_tuple = False
        self._out_rows: Dict[int, tuple[int, ...]] = {}
        self._capture_graphs(max_seq_len)

    def _reset_moe_offload_cache(self) -> None:
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.reset()

    def _store_output(self, bs: int, out: Any) -> None:
        self._out_is_tuple = isinstance(out, tuple)
        outs = out if self._out_is_tuple else (out,)
        if self._out is None:
            # the largest batch is captured first, so its outputs size the buffers
            assert self.out_dtype is None or len(outs) == 1, "out_dtype applies to a single output"
            self._out = [torch.empty_like(o, dtype=self.out_dtype or o.dtype) for o in outs]
        for buf, o in zip(self._out, outs, strict=True):
            buf[: o.shape[0]] = o
        self._out_rows[bs] = tuple(o.shape[0] for o in outs)

    def _capture_graphs(self, max_seq_len: int):
        # Mark the post-weights "warmup" phase for /health: this stretch (graph capture — or the
        # remaining readiness work when graphs are disabled) moves no bytes, so without this the
        # loader would sit at 100% (last byte bar) until the ready ack. total=0 ⇒ the desktop
        # reads it as an indeterminate phase and animates the bar. Must precede the
        # graphs-disabled early return so that config gets the phase too.
        emit_progress("Capturing CUDA graphs / warming up", 0, 0)
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list, steps=self.steps)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, self.device, mrope=self.mrope, steps=self.steps)
        self._reset_moe_offload_cache()

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = self.graph_pool
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase=self.phase, rows_per_req=self.steps)
            batch.padded_reqs = batch.reqs
            if self.batch_hook is not None:
                self.batch_hook(batch)
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            # capture on the dummy linear-state slot so GatedDeltaNet gather/scatter
            # touches scratch (real slot indices are written by copy_from on replay). Hybrid-
            # radix decouples the GDN slot from table_idx -> use the GDN padding slot.
            dummy_slot = (self.dummy_req.linear_slot_idx
                          if self.dummy_req.linear_slot_idx is not None
                          else self.dummy_req.table_idx)
            self.buffer.table_idx[:bs].fill_(dummy_slot)
            with get_global_ctx().forward_batch(batch, **self.ctx_override):
                self._store_output(bs, self.forward())
                # Keep the offload cache warmed for capture. Resetting here forces
                # CUDA graph capture to replay cold-cache expert copies.
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self._store_output(bs, self.forward())
                _upload_graph(graph, self.stream)
                self._reset_moe_offload_cache()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph
        self.graph_pool = pool

        self._reset_moe_offload_cache()
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return (
            batch.phase == self.phase
            and batch.rows_per_req == self.steps
            and batch.size <= self.max_graph_bs
        )

    def replay(self, batch: Batch) -> Any:
        assert self.can_use_cuda_graph(batch) and self._out is not None
        self.buffer.copy_from(batch)
        padded = batch.padded_size
        g = self.graph_map[padded]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        outs = tuple(
            buf[: rows // padded * batch.size] for buf, rows in zip(self._out, self._out_rows[padded], strict=True)
        )
        return outs if self._out_is_tuple else outs[0]

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        # Drop the CUDAGraph objects (and the shared mempool they hold) AND the static
        # GraphCaptureBuffer tensors (input/out_loc/positions/...) and output buffers ([max_bs, vocab] logits).
        # Dropping the references is the load-bearing step; without it a runtime rebuild's
        # free-before-alloc cannot reclaim this GPU memory. empty_cache() is left to the
        # caller / next capture (GraphRunner._capture_graphs already runs it).
        self.graph_map = {}
        self.buffer = None
        self._out = None
        gc.collect()
