from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

import torch
from freetoken.core import Batch, Req, get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.utils import init_logger, mem_GB
from freetoken.utils.progress import emit_progress
from freetoken.utils.step_profiler import profiler_phase
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.models import BaseLLMModel
    from freetoken.moe.offload_cache import OffloadMoeCache
    from freetoken.engine.sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


def _has_weight_format(value, wanted: str, seen: set[int] | None = None) -> bool:
    """Walk FreeToken's BaseOP tree without assuming torch.nn.Module APIs."""
    if seen is None:
        seen = set()
    if value is None or id(value) in seen or isinstance(value, torch.Tensor):
        return False
    seen.add(id(value))
    if getattr(value, "weight_format", None) == wanted:
        return True
    if isinstance(value, (list, tuple)):
        return any(_has_weight_format(item, wanted, seen) for item in value)
    attrs = getattr(value, "__dict__", None)
    return bool(attrs) and any(_has_weight_format(item, wanted, seen) for item in attrs.values())


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor
    sampled_tokens: torch.Tensor
    sampled_indices: torch.Tensor
    table_idx: torch.Tensor  # per-request slot id for GatedDeltaNet state gather/scatter
    # Decode GDN query indptr = arange(bs+1); a constant per captured bs, filled once.
    fla_cu_seqlens: torch.Tensor

    @classmethod
    def init(
        cls,
        bs: int,
        vocab_size: int,
        device: torch.device,
        *,
        sampled_tokens: torch.Tensor | None = None,
        sampled_indices: torch.Tensor | None = None,
    ) -> GraphCaptureBuffer:
        if sampled_tokens is None:
            sampled_tokens = torch.empty(bs, dtype=torch.int32, device=device)
        if sampled_indices is None:
            sampled_indices = torch.empty(bs, dtype=torch.int64, device=device)
        if (
            sampled_tokens.ndim != 1
            or sampled_tokens.shape[0] < bs
            or sampled_tokens.dtype != torch.int32
            or sampled_tokens.device != device
        ):
            raise ValueError("sampled token chain must be a device int32 vector with graph capacity")
        if (
            sampled_indices.ndim != 1
            or sampled_indices.shape[0] < bs
            or sampled_indices.dtype != torch.int64
            or sampled_indices.device != device
        ):
            raise ValueError("sampled index scratch must be a device int64 vector with graph capacity")
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
            sampled_tokens=sampled_tokens[:bs],
            sampled_indices=sampled_indices[:bs],
            table_idx=torch.zeros(bs, dtype=torch.int32, device=device),
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
        # Decode GDN metadata reads the persistent cu_seqlens (constant arange) and the
        # persistent table_idx slot map, so the captured kernels see stable addresses.
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.fla_cu_seqlens[: bs + 1], cache_indices=self.table_idx[_slice]
        )

    def copy_from(self, batch: Batch) -> None:
        with profiler_phase("graph_input_copy"):
            _slice = slice(batch.padded_size)
            self.input_ids[_slice] = batch.input_ids
            if batch.out_loc is not None:
                self.out_loc[_slice] = batch.out_loc
            self.positions[_slice] = batch.positions
            if batch.linear_table_idx is not None:
                self.table_idx[_slice] = batch.linear_table_idx


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
        sampler: "Sampler | None" = None,
        token_chain: Any | None = None,
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
        self.sampler = sampler
        self.token_chain = token_chain
        self.resident_gguf = _has_weight_format(model, "gguf")
        self.graph_telemetry = {
            "expert_storage": "resident_gguf" if self.resident_gguf else "offload_or_dense",
            "expert_fetches": 0,
            "expert_remaps": 0,
        }
        self.capture_sampler = sampler is not None and os.environ.get(
            "FREETOKEN_GRAPH_SAMPLER", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.stream = stream
        self.device = device
        self._capture_graphs(max_seq_len, vocab_size, model)

    def runtime_telemetry(self) -> dict:
        """Return graph/storage facts without synchronizing or touching model state."""
        try:
            from freetoken.kernel.gguf import gguf_dispatch_report

            dispatch = gguf_dispatch_report()
        except Exception:  # noqa: BLE001 -- optional GGUF telemetry
            dispatch = []
        result = {
            **self.graph_telemetry,
            "resident_gguf": self.resident_gguf,
            "graph_batches": sorted(self.graph_map),
            "sampler_graph_batches": sorted(self.sampler_graph_map),
        }
        if dispatch and os.environ.get("FREETOKEN_GGUF_DISPATCH_TRACE", "").lower() in {
            "1", "true", "yes", "on"
        }:
            result["gguf_dispatch"] = dispatch
        return result

    def _reset_moe_offload_cache(self) -> None:
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.reset()

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        # Mark the post-weights "warmup" phase for /health: this stretch (graph capture — or the
        # remaining readiness work when graphs are disabled) moves no bytes, so without this the
        # loader would sit at 100% (last byte bar) until the ready ack. total=0 ⇒ the desktop
        # reads it as an indeterminate phase and animates the bar. Must precede the
        # graphs-disabled early return so that config gets the phase too.
        emit_progress("Capturing CUDA graphs / warming up", 0, 0)
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        self.sampler_graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        # ROCm parity: honour the graph-capture gate result. If capture is not
        # viable on this AMD card, skip graphs entirely so decode uses the kernel-launch
        # path (correct, just not graph-accelerated) rather than erroring mid-capture.
        from freetoken.utils.arch import is_rocm
        from freetoken.utils.graph_gate import graph_capture_status, rocm_blas_report, run_graph_gate

        if is_rocm():
            logger.info_rank0(f"graph capture BLAS policy: {rocm_blas_report()}")

        if is_rocm() and graph_capture_status() == "fail":
            # Variant detail matters: the all-variants record is what closes the thread.
            detail = run_graph_gate().get("detail", "")
            logger.info_rank0(
                "AMD ROCm build: HIP graph capture gate FAILED on this device (all "
                f"capture variants: {detail[:160]}); using the kernel-launch decode "
                "path (CUDA graphs disabled)."
            )
            return None
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        # A forced/selected GGUF candidate must compile, execute both measured
        # quant paths, and synchronize before any graph captures its addresses.
        # Auto mode falls back to legacy inside this gate; forced gfx1100 errors.
        from freetoken.kernel.gguf import ensure_gguf_moe_candidate_ready
        if ensure_gguf_moe_candidate_ready():
            logger.info_rank0("gfx1100 GGUF MoE candidate compile/self-test passed before graph capture")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        sampled_tokens = None
        sampled_indices = None
        if self.token_chain is not None:
            sampled_tokens = self.token_chain.device_tokens
            sampled_indices = self.token_chain.sampled_indices
        self.buffer = GraphCaptureBuffer.init(
            self.max_graph_bs,
            vocab_size,
            self.device,
            sampled_tokens=sampled_tokens,
            sampled_indices=sampled_indices,
        )
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
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph
            if self.capture_sampler:
                from freetoken.engine.sample import BatchSamplingArgs

                sampler_graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(sampler_graph, pool=pool, stream=self.stream):
                        self.sampler.sample_into_device(
                            self.buffer.logits[:bs],
                            BatchSamplingArgs(temperatures=None),
                            batch,
                            self.buffer.sampled_tokens[:bs],
                            self.buffer.sampled_indices[:bs],
                        )
                except Exception as exc:  # graph stage is optional; model graph remains valid
                    logger.warning_rank0(
                        f"greedy sampler graph disabled for bs={bs}; using fallback sampler "
                        f"({type(exc).__name__}: {str(exc)[:160]})"
                    )
                else:
                    self.sampler_graph_map[bs] = sampler_graph

        self._reset_moe_offload_cache()
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")
        logger.info_rank0(f"GGUF graph telemetry: {self.runtime_telemetry()}")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        # ``self.graph_map`` is empty when graphs were skipped (ROCm graph-gate fail or
        # disabled); decode must then fall back to the kernel-launch path.
        return bool(self.graph_map) and batch.is_decode and batch.size <= self.max_graph_bs

    def replay(
        self, batch: Batch, sample_args: "BatchSamplingArgs | None" = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        assert self.can_use_cuda_graph(batch)
        with profiler_phase("graph_replay"):
            self.buffer.copy_from(batch)
            g = self.graph_map[batch.padded_size]
            self.attn_backend.prepare_for_replay(batch)
            g.replay()
        logits = self.buffer.logits[: batch.size]
        if sample_args is None:
            return logits
        sampler_graph = self.sampler_graph_map.get(batch.padded_size)
        sampled = (
            self.buffer.sampled_tokens[: batch.size]
            if sampler_graph is not None and self.sampler.capture_safe(sample_args)
            else None
        )
        if sampled is not None:
            sampler_graph.replay()
        return logits, sampled

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
        self.sampler_graph_map = {}
        self.buffer = None
        gc.collect()
