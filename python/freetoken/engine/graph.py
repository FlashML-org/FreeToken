from __future__ import annotations

import gc
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

import torch
from freetoken.core import Batch, Req, get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.utils import init_logger, mem_GB
from freetoken.utils.progress import emit_progress
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.models import BaseLLMModel
    from freetoken.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor
    table_idx: torch.Tensor  # per-request slot id for GatedDeltaNet state gather/scatter
    request_table_idx: torch.Tensor  # per-request paged-KV table row
    # Decode GDN query indptr = arange(bs+1); a constant per captured bs, filled once.
    fla_cu_seqlens: torch.Tensor

    @classmethod
    def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
            table_idx=torch.zeros(bs, dtype=torch.int32, device=device),
            request_table_idx=torch.zeros(bs, dtype=torch.int64, device=device),
            fla_cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        from freetoken.attention.linear import FLAMetadata

        _slice = slice(batch.padded_size)
        bs = batch.padded_size
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]
        batch.linear_table_idx = self.table_idx[_slice]
        batch.active_table_idx = self.request_table_idx[_slice]
        # Decode GDN metadata reads the persistent cu_seqlens (constant arange) and the
        # persistent table_idx slot map, so the captured kernels see stable addresses.
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.fla_cu_seqlens[: bs + 1], cache_indices=self.table_idx[_slice]
        )

    def copy_from(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        if batch.out_loc is not None:
            self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions
        if batch.linear_table_idx is not None:
            self.table_idx[_slice] = batch.linear_table_idx
        if batch.active_table_idx is not None:
            self.request_table_idx[_slice] = batch.active_table_idx


@dataclass
class SpecGraphCaptureBuffer:
    """Persistent inputs/outputs for ``anchor + prefix`` target graphs.

    The allocation is sized for the checkpoint's maximum ``anchor + gamma`` span,
    while each captured graph takes a prefix of it.  Keeping one backing allocation
    gives every replay a stable address without paying one logits buffer per prefix.
    """

    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor
    request_table_idx: torch.Tensor
    span: int

    @classmethod
    def init(
        cls, max_reqs: int, span: int, vocab_size: int, device: torch.device
    ) -> "SpecGraphCaptureBuffer":
        max_tokens = max_reqs * span
        return cls(
            input_ids=torch.zeros(max_tokens, dtype=torch.int32, device=device),
            out_loc=torch.zeros(max_tokens, dtype=torch.int32, device=device),
            positions=torch.zeros(max_tokens, dtype=torch.int32, device=device),
            logits=torch.empty(max_tokens, vocab_size, dtype=torch.float32, device=device),
            request_table_idx=torch.zeros(max_reqs, dtype=torch.int64, device=device),
            span=span,
        )

    def set_batch(self, batch: Batch) -> None:
        reqs = batch.padded_size
        span = int(batch.spec_block) + 1
        if not 1 <= span <= self.span:
            raise RuntimeError(
                f"DSpark graph span {span} is outside captured range 1..{self.span}"
            )
        tokens = reqs * span
        batch.input_ids = self.input_ids[:tokens]
        batch.out_loc = self.out_loc[:tokens]
        batch.positions = self.positions[:tokens]
        batch.active_table_idx = self.request_table_idx[:reqs]

    def copy_from(self, batch: Batch) -> None:
        reqs = batch.padded_size
        span = int(batch.spec_block) + 1
        if not 1 <= span <= self.span:
            raise RuntimeError(
                f"DSpark graph span {span} is outside captured range 1..{self.span}"
            )
        tokens = reqs * span
        if batch.input_ids.numel() != tokens or batch.positions.numel() != tokens:
            raise RuntimeError(
                f"DSpark graph expected {tokens} target rows, got "
                f"{batch.input_ids.numel()}/{batch.positions.numel()}"
            )
        self.input_ids[:tokens].copy_(batch.input_ids)
        if batch.out_loc is not None:
            self.out_loc[:tokens].copy_(batch.out_loc)
        self.positions[:tokens].copy_(batch.positions)
        if batch.active_table_idx is None or batch.active_table_idx.numel() != reqs:
            raise RuntimeError("DSpark graph needs one active table row per request")
        self.request_table_idx[:reqs].copy_(batch.active_table_idx)


class SharedSpecCarryJournal:
    """Maximum-span carry outputs shared by every adaptive verify graph.

    The largest graph owns the actual tensors produced by each compressor step.
    Smaller graph captures copy their prefix states into those same addresses.  This
    is safe because verify graphs never replay concurrently, and it avoids retaining
    one very large compressor journal per CUDA-graph bucket.
    """

    def __init__(self, storage: dict) -> None:
        if not storage:
            raise RuntimeError("DSpark maximum verify graph produced no carry states")
        self.storage = storage
        self._active_rows = 0
        self._cursors: dict = {}

    def reset(self, active_rows: int) -> None:
        if active_rows < 1:
            raise ValueError(f"DSpark carry journal needs at least one row, got {active_rows}")
        self._active_rows = active_rows
        self._cursors = {key: 0 for key in self.storage}

    def record(self, key, state: torch.Tensor) -> None:
        pieces = self.storage.get(key)
        if pieces is None:
            raise RuntimeError(f"DSpark carry journal saw unknown compressor {key}")
        cursor = self._cursors[key]
        if cursor >= self._active_rows or cursor >= len(pieces):
            raise RuntimeError(
                f"DSpark carry journal overflow for {key}: row {cursor}, "
                f"active={self._active_rows}, capacity={len(pieces)}"
            )
        pieces[cursor].copy_(state)
        self._cursors[key] = cursor + 1

    def items(self):
        for key, pieces in self.storage.items():
            yield key, pieces[: self._active_rows]


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


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        moe_offload_cache: OffloadMoeCache | None = None,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.moe_offload_cache = moe_offload_cache
        self.stream = stream
        self.device = device
        # A captured model forward does not execute Python on replay, so a model-global
        # "last features" attribute remains pointed at whichever graph was captured
        # last.  Keep the actual output bundle owned by each graph size instead.
        self._dspark_feature_map: Dict[int, object] = {}
        self._spec_feature_map: Dict[Tuple[int, int], object] = {}
        self._spec_carry_map: Dict[Tuple[int, int], dict] = {}
        # Measured (span, milliseconds) curve for the paper's hardware-aware
        # verification scheduler.  It is populated only for the single-request
        # graphs that FreeToken's current DSV4 serving mode can compact safely.
        self.spec_verify_cost_curve: List[Tuple[int, float]] = []
        self.spec_block_size = int(
            getattr(model, "speculative_verify_block_size", 0) or 0
        )
        self.spec_span = self.spec_block_size + 1 if self.spec_block_size else 0
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _reset_moe_offload_cache(self) -> None:
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.reset()

    def _profile_graph_ms(self, graph: torch.cuda.CUDAGraph, replays: int = 5) -> float:
        """Median replay cost for DSpark's hardware-capacity curve.

        Begin the series cold, then let ordinary LRU fills and hits participate in
        the curve. This prices the cache reuse present during normal engine execution
        instead of forcing an artificial cold miss before every timed replay.
        """
        samples: list[float] = []
        self._reset_moe_offload_cache()
        for _ in range(replays):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(self.stream)
            graph.replay()
            end.record(self.stream)
            end.synchronize()
            samples.append(start.elapsed_time(end))
        return float(statistics.median(samples))

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        # Mark the post-weights "warmup" phase for /health: this stretch (graph capture — or the
        # remaining readiness work when graphs are disabled) moves no bytes, so without this the
        # loader would sit at 100% (last byte bar) until the ready ack. total=0 ⇒ the desktop
        # reads it as an indeterminate phase and animates the bar. Must precede the
        # graphs-disabled early return so that config gets the phase too.
        emit_progress("Capturing CUDA graphs / warming up", 0, 0)
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        self.spec_graph_map: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)
        self._reset_moe_offload_cache()

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            # capture on the dummy linear-state slot so GatedDeltaNet gather/scatter
            # touches scratch (real slot indices are written by copy_from on replay). Hybrid-
            # radix decouples the GDN slot from table_idx -> use the GDN padding slot.
            dummy_slot = (self.dummy_req.linear_slot_idx
                          if self.dummy_req.linear_slot_idx is not None
                          else self.dummy_req.table_idx)
            self.buffer.table_idx[:bs].fill_(dummy_slot)
            with get_global_ctx().forward_batch(batch):
                self.buffer.logits[:bs] = model.forward()
                # Keep the offload cache warmed for capture. Resetting here forces
                # CUDA graph capture to replay cold-cache expert copies.
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
                self._reset_moe_offload_cache()
            get_features = getattr(model, "dspark_target_features", None)
            if get_features is not None:
                features = get_features()
                if features is not None:
                    # These tensors are graph outputs: replay of this exact graph writes
                    # them in place.  Holding the bundle preserves the matching addresses.
                    self._dspark_feature_map[bs] = features
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph

        spec_capture = getattr(self.attn_backend, "prepare_for_spec_capture", None)
        if self.spec_span and spec_capture is not None:
            self.spec_buffer = SpecGraphCaptureBuffer.init(
                self.max_graph_bs, self.spec_span, vocab_size, self.device
            )
            # Section 5.2 chooses among the actual CUDA-graph cost cliffs.  DSV4
            # currently serves one request at a time, so capture every legal prefix
            # there.  Keep the prior fixed-width behavior for larger request batches;
            # those need the paper's marker-tensor varlen layout before independent
            # per-request widths are safe.
            adaptive_single_req = self.graph_bs_list == [1]
            spans = (
                list(range(1, self.spec_span + 1))
                if adaptive_single_req
                else [self.spec_span]
            )
            logger.info_rank0(
                f"Capturing DSpark target verify graphs: requests={self.graph_bs_list}, "
                f"rows/request={spans}"
            )
            shared_carry: SharedSpecCarryJournal | None = None
            for span in sorted(spans, reverse=True):
                for bs in sorted(self.graph_bs_list, reverse=True):
                    tokens = bs * span
                    graph = torch.cuda.CUDAGraph()
                    batch = Batch(reqs=[self.dummy_req] * bs, phase="prefill")
                    batch.padded_reqs = batch.reqs
                    batch.speculative = True
                    batch.spec_block = span - 1
                    batch.spec_verify_decode = True
                    batch.spec_carry_states = {}
                    self.spec_buffer.set_batch(batch)
                    # Valid device positions keep capture-time writes within the reserved
                    # dummy row; replay overwrites this same stable buffer.
                    self.spec_buffer.positions[:tokens].copy_(
                        torch.arange(span, device=self.device, dtype=torch.int32).repeat(bs)
                    )
                    self.spec_buffer.request_table_idx[:bs].fill_(self.dummy_req.table_idx)
                    spec_capture(batch)
                    active_rows = tokens
                    with get_global_ctx().forward_batch(batch):
                        if shared_carry is not None:
                            shared_carry.reset(active_rows)
                            batch.spec_carry_states = shared_carry
                        self.spec_buffer.logits[:tokens] = model.forward()
                        if shared_carry is None:
                            batch.spec_carry_states = {}
                        else:
                            shared_carry.reset(active_rows)
                            batch.spec_carry_states = shared_carry
                        with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                            self.spec_buffer.logits[:tokens] = model.forward()
                        self._reset_moe_offload_cache()
                    if shared_carry is None and adaptive_single_req:
                        # Spans are descending, so this is the maximum graph. Its
                        # captured outputs become the one persistent journal backing
                        # all remaining prefix graphs.
                        shared_carry = SharedSpecCarryJournal(batch.spec_carry_states)
                        shared_carry.reset(active_rows)
                    get_features = getattr(model, "dspark_target_features", None)
                    key = (bs, span)
                    if get_features is not None:
                        features = get_features()
                        if features is not None:
                            self._spec_feature_map[key] = features
                    self._spec_carry_map[key] = (
                        shared_carry if shared_carry is not None else batch.spec_carry_states
                    )
                    self.spec_graph_map[key] = graph
                    if adaptive_single_req and bs == 1:
                        cost_ms = self._profile_graph_ms(graph)
                        self.spec_verify_cost_curve.append((span, cost_ms))

            if self.spec_verify_cost_curve:
                # vLLM makes profiled costs nondecreasing before scheduling. Sampling
                # noise must not make a wider target batch appear artificially cheaper.
                high = 0.0
                monotonic = []
                for span, cost in sorted(self.spec_verify_cost_curve):
                    high = max(high, cost)
                    monotonic.append((span, high))
                self.spec_verify_cost_curve = monotonic
                logger.info_rank0(
                    "DSpark profiled target verify curve: %s",
                    ", ".join(f"{span} rows={cost:.2f}ms" for span, cost in monotonic),
                )

        self._reset_moe_offload_cache()
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.size <= self.max_graph_bs

    def can_use_spec_cuda_graph(self, batch: Batch) -> bool:
        key = (batch.padded_size, int(batch.spec_block) + 1)
        return bool(
            batch.speculative
            and key in self.spec_graph_map
            and batch.padded_size == batch.size
        )

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return self.buffer.logits[: batch.size]

    def replay_spec(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_spec_cuda_graph(batch)
        span = int(batch.spec_block) + 1
        tokens = batch.padded_size * span
        key = (batch.padded_size, span)
        self.spec_buffer.copy_from(batch)
        self.spec_buffer.set_batch(batch)
        batch.spec_verify_decode = True
        journal = self._spec_carry_map[key]
        reset_journal = getattr(journal, "reset", None)
        if reset_journal is not None:
            reset_journal(tokens)
        batch.spec_carry_states = journal
        prepare = getattr(self.attn_backend, "prepare_for_spec_replay")
        prepare(batch)
        self.spec_graph_map[key].replay()
        return self.spec_buffer.logits[:tokens]

    def dspark_target_features(self, batch: Batch):
        """The feature output owned by the graph replayed for ``batch``."""
        return self._dspark_feature_map.get(batch.padded_size)

    def dspark_spec_target_features(self, batch: Batch):
        return self._spec_feature_map.get(
            (batch.padded_size, int(batch.spec_block) + 1)
        )

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
        # GraphCaptureBuffer tensors ([max_bs, vocab] logits + input/out_loc/positions/...).
        # Dropping the references is the load-bearing step; without it a runtime rebuild's
        # free-before-alloc cannot reclaim this GPU memory. empty_cache() is left to the
        # caller / next capture (GraphRunner._capture_graphs already runs it).
        self.graph_map = {}
        self.spec_graph_map = {}
        self._dspark_feature_map = {}
        self._spec_feature_map = {}
        self._spec_carry_map = {}
        self.spec_verify_cost_curve = []
        self.buffer = None
        self.spec_buffer = None
        gc.collect()
