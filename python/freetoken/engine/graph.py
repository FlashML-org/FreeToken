from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
import torch.nn.functional as F
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


def project_lm_head_all_positions(lm_head, hidden_states: torch.Tensor) -> torch.Tensor:
    """Project every hidden row through an LM head, bypassing prefill last-token slicing."""
    module = getattr(lm_head, "tied_embedding", None) or lm_head
    logits = F.linear(hidden_states, module.weight, getattr(lm_head, "bias", None))
    tp_size = getattr(lm_head, "tp_size", 1)
    if tp_size == 1:
        return logits

    output_tensor = lm_head._comm.all_gather(logits)
    input_shape = logits.shape
    output_tensor = output_tensor.view((tp_size,) + input_shape)
    output_tensor = output_tensor.permute(1, 0, 2).contiguous()
    output_tensor = output_tensor.reshape(input_shape[:1] + (tp_size * input_shape[1],))
    return output_tensor[:, : lm_head.num_embeddings]


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor
    hidden_states: list[torch.Tensor] | None
    table_idx: torch.Tensor  # per-request slot id for GatedDeltaNet state gather/scatter
    # Decode GDN query indptr = arange(bs+1); a constant per captured bs, filled once.
    fla_cu_seqlens: torch.Tensor
    fla_has_initial_state: torch.Tensor | None = None
    dflash_conv_states: torch.Tensor | None = None
    dflash_recurrent_states: torch.Tensor | None = None
    dflash_recurrent_state_indices: torch.Tensor | None = None

    @classmethod
    def init(
        cls,
        bs: int,
        vocab_size: int,
        device: torch.device,
        *,
        hidden_size: int | None = None,
        hidden_dtype: torch.dtype | None = None,
        num_hidden_layers: int = 0,
        linear_state_pool=None,
    ) -> GraphCaptureBuffer:
        hidden_states = None
        if num_hidden_layers > 0:
            assert hidden_size is not None
            assert hidden_dtype is not None
            hidden_states = [
                torch.empty(bs, hidden_size, dtype=hidden_dtype, device=device)
                for _ in range(num_hidden_layers)
            ]
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
            hidden_states=hidden_states,
            table_idx=torch.zeros(bs, dtype=torch.int32, device=device),
            fla_cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device),
        )

    @classmethod
    def init_dflash_verify(
        cls,
        verify_len: int,
        vocab_size: int,
        device: torch.device,
        *,
        hidden_size: int | None = None,
        hidden_dtype: torch.dtype | None = None,
        num_hidden_layers: int = 0,
        linear_state_pool=None,
    ) -> GraphCaptureBuffer:
        buffer = cls.init(
            verify_len,
            vocab_size,
            device,
            hidden_size=hidden_size,
            hidden_dtype=hidden_dtype,
            num_hidden_layers=num_hidden_layers,
        )
        buffer.table_idx = torch.zeros(1, dtype=torch.int32, device=device)
        buffer.fla_cu_seqlens = torch.tensor([0, verify_len], dtype=torch.int32, device=device)
        buffer.fla_has_initial_state = torch.ones(1, dtype=torch.bool, device=device)
        if linear_state_pool is not None:
            snapshot_len = max(verify_len - 1, 0)
            buffer.dflash_conv_states = torch.empty(
                (snapshot_len, *linear_state_pool.conv_states[:, 0].shape),
                dtype=linear_state_pool.conv_states.dtype,
                device=device,
            )
            buffer.dflash_recurrent_states = torch.empty(
                (snapshot_len, *linear_state_pool.recurrent_states[:, 0].shape),
                dtype=linear_state_pool.recurrent_states.dtype,
                device=device,
            )
        return buffer

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
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        if batch.out_loc is not None:
            self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions
        if batch.linear_table_idx is not None:
            self.table_idx[_slice] = batch.linear_table_idx

    def copy_dflash_verify_from(self, batch: Batch, verify_len: int) -> None:
        self.input_ids[:verify_len] = batch.input_ids[:verify_len]
        if batch.out_loc is not None:
            self.out_loc[:verify_len] = batch.out_loc[:verify_len]
        self.positions[:verify_len] = batch.positions[:verify_len]
        if batch.linear_table_idx is not None:
            self.table_idx[:1] = batch.linear_table_idx[:1]

    def set_dflash_target_verify_batch(
        self,
        batch: Batch,
        verify_len: int,
        *,
        return_linear_snapshots: bool = False,
    ) -> None:
        from freetoken.attention.linear import FLAMetadata

        batch.input_ids = self.input_ids[:verify_len]
        batch.out_loc = self.out_loc[:verify_len]
        batch.positions = self.positions[:verify_len]
        batch.linear_table_idx = self.table_idx[:1]
        dflash_conv_states = None
        dflash_recurrent_states = None
        dflash_recurrent_state_indices = None
        if return_linear_snapshots:
            self._ensure_dflash_target_verify_linear_buffers(verify_len)
            dflash_conv_states = self.dflash_conv_states
            dflash_recurrent_states = self.dflash_recurrent_states
            dflash_recurrent_state_indices = self.dflash_recurrent_state_indices
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.fla_cu_seqlens[:2],
            cache_indices=self.table_idx[:1],
            has_initial_state=self.fla_has_initial_state,
            dflash_disable_state_update=return_linear_snapshots,
            dflash_conv_states_buffer=dflash_conv_states,
            dflash_recurrent_states_buffer=dflash_recurrent_states,
            dflash_recurrent_state_indices=dflash_recurrent_state_indices,
        )

    def _ensure_dflash_target_verify_linear_buffers(self, verify_len: int) -> None:
        if self.dflash_conv_states is None or self.dflash_recurrent_states is None:
            raise RuntimeError("DFlash target verify graph requires linear snapshot buffers")
        if self.dflash_conv_states.shape[0] != verify_len:
            self.dflash_conv_states = torch.empty(
                (verify_len, *self.dflash_conv_states.shape[1:]),
                dtype=self.dflash_conv_states.dtype,
                device=self.dflash_conv_states.device,
            )
        num_layers = self.dflash_conv_states.shape[1]
        if (
            self.dflash_recurrent_states.shape[0] != num_layers
            or self.dflash_recurrent_states.shape[1] != verify_len
        ):
            self.dflash_recurrent_states = torch.empty(
                (
                    num_layers,
                    verify_len,
                    *self.dflash_recurrent_states.shape[2:],
                ),
                dtype=self.dflash_recurrent_states.dtype,
                device=self.dflash_recurrent_states.device,
            )
        if (
            self.dflash_recurrent_state_indices is None
            or self.dflash_recurrent_state_indices.numel() != self.dflash_recurrent_states.shape[0]
        ):
            self.dflash_recurrent_state_indices = torch.arange(
                self.dflash_recurrent_states.shape[0],
                dtype=torch.int32,
                device=self.dflash_recurrent_states.device,
            )



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


def _dflash_target_verify_lens_within_budget(
    lens: List[int],
    linear_state_pool,
    budget_bytes: int,
) -> List[int]:
    """Keep the largest prefix of verify lens whose per-len GDN snapshot buffers fit
    in ``budget_bytes``. Target-verify graphs own a per-len conv/recurrent snapshot
    buffer set sized ``layers * len * state_bytes``; without a budget, large block
    sizes (e.g. 16) OOM at capture. A dropped len simply falls back to decode-loop
    verify at runtime, so this is a pure performance/robustness gate."""
    if linear_state_pool is None:
        return list(lens)
    conv = linear_state_pool.conv_states        # [layers, slots, conv_dim, K-1]
    rec = linear_state_pool.recurrent_states    # [layers, slots, heads, K, V]
    layers = rec.shape[0]
    conv_per_token = conv.shape[0] * conv[0, 0].numel() * conv.element_size()
    rec_per_token = layers * rec[0, 0].numel() * rec.element_size()
    per_token_bytes = conv_per_token + rec_per_token
    kept: List[int] = []
    total = 0
    for verify_len in sorted(lens):
        total += verify_len * per_token_bytes
        if total > budget_bytes:
            break
        kept.append(verify_len)
    return kept


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
        hidden_layer_ids: set[int] | None = None,
        hidden_size: int | None = None,
        hidden_dtype: torch.dtype | None = None,
        dflash_target_verify_lens: list[int] | None = None,
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
        self.hidden_layer_ids = set(hidden_layer_ids or [])
        self.hidden_size = hidden_size
        self.hidden_dtype = hidden_dtype
        self.dflash_target_verify_lens = sorted(set(dflash_target_verify_lens or []))
        self._capture_graphs(max_seq_len, vocab_size, model)

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
        self.dflash_target_verify_graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        self.dflash_target_verify_buffers: Dict[int, GraphCaptureBuffer] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(
            self.max_graph_bs,
            vocab_size,
            self.device,
            hidden_size=self.hidden_size,
            hidden_dtype=self.hidden_dtype,
            num_hidden_layers=len(self.hidden_layer_ids),
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
                self._run_model_into_buffer(model, bs)
                # Keep the offload cache warmed for capture. Resetting here forces
                # CUDA graph capture to replay cold-cache expert copies.
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self._run_model_into_buffer(model, bs)
                self._reset_moe_offload_cache()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph

        self._reset_moe_offload_cache()
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")
        self._capture_dflash_target_verify_graphs(max_seq_len, vocab_size, model, pool)

    def _capture_dflash_target_verify_graphs(
        self,
        max_seq_len: int,
        vocab_size: int,
        model: BaseLLMModel,
        pool,
    ) -> None:
        if not self.dflash_target_verify_lens:
            return
        linear_state_pool = get_global_ctx().linear_state_pool
        if linear_state_pool is not None:
            budget = int(get_free_memory(self.device) * 0.30)
            kept = _dflash_target_verify_lens_within_budget(
                self.dflash_target_verify_lens, linear_state_pool, budget
            )
            if len(kept) < len(self.dflash_target_verify_lens):
                logger.warning_rank0(
                    "DFlash target verify graphs limited to lens "
                    f"{kept or '[]'} (snapshot memory budget); longer verifies "
                    "fall back to decode-loop verify."
                )
            self.dflash_target_verify_lens = kept
            if not kept:
                return
        init_verify = getattr(self.attn_backend, "init_dflash_target_verify_capture_graph", None)
        prepare_capture = getattr(self.attn_backend, "prepare_for_dflash_target_verify_capture", None)
        if init_verify is None or prepare_capture is None:
            logger.warning_rank0("DFlash target verify CUDA graph is disabled for this attention backend.")
            return

        init_verify(max_seq_len=max_seq_len, verify_lens=self.dflash_target_verify_lens)
        for verify_len in self.dflash_target_verify_lens:
            graph = torch.cuda.CUDAGraph()
            buffer = GraphCaptureBuffer.init_dflash_verify(
                verify_len,
                vocab_size,
                self.device,
                hidden_size=self.hidden_size,
                hidden_dtype=self.hidden_dtype,
                num_hidden_layers=len(self.hidden_layer_ids),
                linear_state_pool=linear_state_pool,
            )
            buffer.fla_cu_seqlens = torch.tensor(
                [0, verify_len], dtype=torch.int64, device=self.device
            )
            batch = Batch(reqs=[self.dummy_req], phase="decode")
            batch.padded_reqs = batch.reqs
            prepare_capture(batch, verify_len)
            buffer.set_dflash_target_verify_batch(
                batch,
                verify_len,
                return_linear_snapshots=linear_state_pool is not None,
            )
            buffer.input_ids.fill_(0)
            buffer.positions.copy_(torch.arange(verify_len, dtype=torch.int32, device=self.device))
            buffer.out_loc.fill_(0)
            dummy_slot = (self.dummy_req.linear_slot_idx
                          if self.dummy_req.linear_slot_idx is not None
                          else self.dummy_req.table_idx)
            buffer.table_idx.fill_(dummy_slot)
            with get_global_ctx().forward_batch(batch):
                self._run_dflash_target_verify_into_buffer(model, verify_len, buffer)
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self._run_dflash_target_verify_into_buffer(model, verify_len, buffer)
                self._reset_moe_offload_cache()
            self.dflash_target_verify_graph_map[verify_len] = graph
            self.dflash_target_verify_buffers[verify_len] = buffer

    def _run_model_into_buffer(
        self,
        model: BaseLLMModel,
        bs: int,
        buffer: GraphCaptureBuffer | None = None,
        offset: int = 0,
    ) -> None:
        if buffer is None:
            buffer = self.buffer
        if self.hidden_layer_ids:
            logits, hidden_states = model.forward(return_hidden_layers=self.hidden_layer_ids)
            assert buffer.hidden_states is not None
            assert len(hidden_states) == len(buffer.hidden_states)
            buffer.logits[offset : offset + bs].copy_(logits[:bs])
            for dst, src in zip(buffer.hidden_states, hidden_states):
                dst[offset : offset + bs].copy_(src[:bs])
            return
        buffer.logits[offset : offset + bs].copy_(model.forward()[:bs])

    def _run_dflash_target_verify_into_buffer(
        self,
        model: BaseLLMModel,
        bs: int,
        buffer: GraphCaptureBuffer,
        offset: int = 0,
    ) -> None:
        hidden, hidden_states = model.model.forward(
            get_global_ctx().batch.input_ids,
            return_hidden_layers=self.hidden_layer_ids,
        )
        logits = project_lm_head_all_positions(model.lm_head, hidden)
        buffer.logits[offset : offset + bs].copy_(logits[:bs])
        if buffer.hidden_states is None:
            return
        assert len(hidden_states) == len(buffer.hidden_states)
        for dst, src in zip(buffer.hidden_states, hidden_states):
            dst[offset : offset + bs].copy_(src[:bs])

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.size <= self.max_graph_bs

    def can_return_hidden_layers(self, hidden_layer_ids: set[int] | None) -> bool:
        return set(hidden_layer_ids or []) == self.hidden_layer_ids

    def can_use_dflash_target_verify_graph(self, batch: Batch, verify_len: int) -> bool:
        return (
            batch.size == 1
            and batch.padded_size == 1
            and batch.input_ids.numel() == verify_len
            and verify_len in self.dflash_target_verify_graph_map
        )

    def replay_dflash_target_verify(
        self,
        batch: Batch,
        verify_len: int,
        *,
        return_hidden_layers: set[int] | None = None,
        return_linear_snapshots: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        assert self.can_use_dflash_target_verify_graph(batch, verify_len)
        wants_hidden = bool(return_hidden_layers)
        assert not wants_hidden or self.can_return_hidden_layers(return_hidden_layers)
        buffer = self.dflash_target_verify_buffers[verify_len]
        assert not wants_hidden or buffer.hidden_states is not None
        assert not return_linear_snapshots or buffer.dflash_conv_states is not None
        assert not return_linear_snapshots or buffer.dflash_recurrent_states is not None
        original_phase = batch.phase
        batch.phase = "decode"
        try:
            buffer.copy_dflash_verify_from(batch, verify_len)
            buffer.set_dflash_target_verify_batch(
                batch,
                verify_len,
                return_linear_snapshots=return_linear_snapshots,
            )
            prepare_replay = getattr(self.attn_backend, "prepare_for_dflash_target_verify_replay")
            prepare_replay(batch, verify_len)
            self.dflash_target_verify_graph_map[verify_len].replay()
        finally:
            batch.phase = original_phase
        logits = buffer.logits[:verify_len]
        linear_snapshots = None
        if return_linear_snapshots:
            assert buffer.dflash_conv_states is not None
            assert buffer.dflash_recurrent_states is not None
            linear_snapshots = (
                buffer.dflash_conv_states[:verify_len],
                buffer.dflash_recurrent_states[:, :verify_len].transpose(0, 1).contiguous(),
            )
        if wants_hidden:
            assert buffer.hidden_states is not None
            if return_linear_snapshots:
                return logits, [h[:verify_len] for h in buffer.hidden_states], linear_snapshots
            return logits, [h[:verify_len] for h in buffer.hidden_states]
        if return_linear_snapshots:
            return logits, linear_snapshots
        return logits

    def replay(
        self,
        batch: Batch,
        *,
        return_hidden_layers: set[int] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        assert self.can_use_cuda_graph(batch)
        wants_hidden = bool(return_hidden_layers)
        assert not wants_hidden or self.can_return_hidden_layers(return_hidden_layers)
        assert not wants_hidden or self.buffer.hidden_states is not None
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        logits = self.buffer.logits[: batch.size]
        if wants_hidden:
            assert self.buffer.hidden_states is not None
            return logits, [h[: batch.size] for h in self.buffer.hidden_states]
        return logits

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
        self.dflash_target_verify_graph_map = {}
        self.dflash_target_verify_buffers = {}
        self.buffer = None
        gc.collect()
