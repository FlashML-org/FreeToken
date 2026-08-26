from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from freetoken.core import Batch, get_global_ctx

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


@dataclass
class TritonCaptureData(BaseCaptureData):
    q_to_req: torch.Tensor
    attn_logits: torch.Tensor
    attn_lse: torch.Tensor
    num_kv_splits: torch.Tensor
    swa_page_table: torch.Tensor | None = None

    @classmethod
    def create(
        cls,
        max_bs: int,
        max_seq_len: int,
        device: torch.device,
        *,
        num_q_heads: int,
        max_head_dim: int,
        max_kv_splits: int,
        **kwargs,
    ):
        return cls(
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),
            positions=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device),
            q_to_req=torch.arange(max_bs, dtype=torch.int32, device=device),
            attn_logits=torch.empty(
                (max_bs, num_q_heads, max_kv_splits, max_head_dim),
                dtype=torch.float32,
                device=device,
            ),
            attn_lse=torch.empty(
                (max_bs, num_q_heads, max_kv_splits),
                dtype=torch.float32,
                device=device,
            ),
            num_kv_splits=torch.full(
                (max_bs,),
                max_kv_splits,
                dtype=torch.int32,
                device=device,
            ),
            **kwargs,
        )


@dataclass
class TritonMetadata(BaseAttnMetadata):
    cu_seqlens_q_gpu: torch.Tensor
    indptr: torch.Tensor
    indices: torch.Tensor
    q_to_req: torch.Tensor
    q_positions: torch.Tensor
    is_decode: bool
    prefix_lens: torch.Tensor
    max_q_len: int
    attn_logits: torch.Tensor | None = None
    attn_lse: torch.Tensor | None = None
    num_kv_splits: torch.Tensor | None = None
    swa_indices: torch.Tensor | None = None

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


class TritonAttentionBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        self.capture: TritonCaptureData | None = None
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.prefill_tile_min_q = 128
        self.num_q_heads = int(getattr(config, "num_qo_heads", 1))
        kv_groups = getattr(config, "kv_cache_group_specs", lambda: ())()
        self.max_head_dim = max(
            (group.head_dim for group in kv_groups),
            default=int(getattr(config, "head_dim", 1)),
        )
        from freetoken.kernel.triton.attention import decode_launch_config

        quant = getattr(self.kvcache, "quant", None)
        quant_name = getattr(quant, "name", None) if getattr(quant, "enabled", False) else None
        capability = (
            torch.cuda.get_device_capability(self.device)
            if self.device.type == "cuda"
            else None
        )
        self.max_kv_splits = decode_launch_config(
            quant_name=quant_name,
            head_dim=self.max_head_dim,
            num_q_heads=self.num_q_heads,
            num_kv_heads=int(getattr(config, "num_kv_heads", 1)),
            compute_capability=capability,
        )[0]

    def _ensure_decode_scratch(
        self,
        metadata: TritonMetadata,
        bs: int,
        num_q_heads: int,
        head_dim: int,
    ) -> None:
        if (
            metadata.attn_logits is not None
            and metadata.attn_lse is not None
            and metadata.num_kv_splits is not None
            and metadata.attn_logits.shape[0] >= bs
            and metadata.attn_logits.shape[1] >= num_q_heads
            and metadata.attn_logits.shape[3] >= head_dim
        ):
            return
        scratch_head_dim = max(self.max_head_dim, head_dim)
        metadata.attn_logits = torch.empty(
            (bs, num_q_heads, self.max_kv_splits, scratch_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        metadata.attn_lse = torch.empty(
            (bs, num_q_heads, self.max_kv_splits),
            dtype=torch.float32,
            device=self.device,
        )
        metadata.num_kv_splits = torch.full(
            (bs,),
            self.max_kv_splits,
            dtype=torch.int32,
            device=self.device,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        from freetoken.kernel.triton.attention import (
            decode_paged_attention,
            extend_paged_attention,
            paged_attention,
        )
        from freetoken.kvcache.quant import BLOCK as QBLOCK

        metadata = batch.attn_metadata
        assert isinstance(metadata, TritonMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

        k_raw = self.kvcache.k_cache(layer_id)
        v_raw = self.kvcache.v_cache(layer_id)
        kv_heads = k_raw.shape[-2]
        quant = getattr(self.kvcache, "quant", None)
        epb = 1 if quant is None or not quant.enabled else quant.elements_per_byte
        # The slab's last dim is the STORAGE head dim -- it halves for packed int4. The
        # kernels get logical element space (head_dim, offs_d, scales) plus the EPB
        # constexpr, and only the loads divide by it.
        head_dim = q.shape[-1]
        assert k_raw.shape[-1] == head_dim // epb and v_raw.shape[-1] == head_dim // epb, (
            f"packed KV slab {k_raw.shape[-1]} != logical head_dim {head_dim} / epb {epb}"
        )
        k_cache = k_raw.view(-1, kv_heads, head_dim // epb)
        v_cache = v_raw.view(-1, kv_heads, head_dim // epb)
        # A quantized pool hands its per-block scales alongside the slabs; an unquantized
        # one has none, and the kernels compile their bf16 path unchanged.
        k_scale = v_scale = None
        if quant is not None and quant.enabled:
            k_scale = self.kvcache.k_scale(layer_id).view(-1, kv_heads, head_dim // QBLOCK)
            v_scale = self.kvcache.v_scale(layer_id).view(-1, kv_heads, head_dim // QBLOCK)

        spec = attn_spec or AttentionSpec()
        indices = metadata.indices
        if spec.sliding_window is not None and metadata.swa_indices is not None:
            indices = metadata.swa_indices
        scale = spec.sm_scale if spec.sm_scale is not None else q.shape[-1] ** -0.5
        if metadata.is_decode and q.dtype in (torch.float16, torch.bfloat16):
            bs = metadata.indptr.numel() - 1
            self._ensure_decode_scratch(metadata, bs, q.shape[1], q.shape[-1])
            assert metadata.attn_logits is not None
            assert metadata.attn_lse is not None
            assert metadata.num_kv_splits is not None
            return decode_paged_attention(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                indptr=metadata.indptr,
                indices=indices,
                q_positions=metadata.q_positions,
                attn_logits=metadata.attn_logits[:bs],
                attn_lse=metadata.attn_lse[:bs],
                num_kv_splits=metadata.num_kv_splits[:bs],
                max_kv_splits=self.max_kv_splits,
                sm_scale=scale,
                sliding_window=spec.sliding_window,
                sinks=spec.sinks,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        if (
            (not metadata.is_decode)
            and q.dtype in (torch.float16, torch.bfloat16)
            and (q.shape[-1] <= 256 or metadata.max_q_len >= self.prefill_tile_min_q)
        ):
            return extend_paged_attention(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                qo_indptr=metadata.cu_seqlens_q_gpu,
                kv_indptr=metadata.indptr,
                kv_indices=indices,
                prefix_lens=metadata.prefix_lens,
                max_q_len=metadata.max_q_len,
                sm_scale=scale,
                sliding_window=spec.sliding_window,
                sinks=spec.sinks,
                k_extend=k.view(q.shape[0], kv_heads, head_dim),
                v_extend=v.view(q.shape[0], kv_heads, head_dim),
                k_scale=k_scale,
                v_scale=v_scale,
            )
        return paged_attention(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            indptr=metadata.indptr,
            indices=indices,
            q_to_req=metadata.q_to_req,
            q_positions=metadata.q_positions,
            sm_scale=scale,
            sliding_window=spec.sliding_window,
            sinks=spec.sinks,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        device = self.device
        ctx = get_global_ctx()
        page_table = ctx.page_table
        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        num_query_tokens = sum(seqlens_q)
        is_decode = max(seqlens_q) == 1
        prefix_lens = torch.tensor(cached_lens, dtype=torch.int32, device=device)

        indptr = torch.tensor([0] + seqlens_k, dtype=torch.int32, device=device).cumsum_(0)
        if is_decode:
            cu_seqlens_q_gpu = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):
            cu_seqlens_q_gpu = indptr
        else:
            cu_seqlens_q_gpu = torch.tensor(
                [0] + seqlens_q, dtype=torch.int32, device=device
            ).cumsum_(0)
        indices = torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs])
        swa_indices = None
        if getattr(self.kvcache, "swa_paged", False):
            # Global-paged SWA (naive + radix): the swa-layer gather reads swa-pool slots = full->swa
            # map of the full page-table slots (live for in-window tokens; out-of-window -> sentinel 0,
            # masked by the sliding window). Recomputed each step (incl. graph replay via
            # _point_to_capture) since the page table grows during decode.
            swa_indices = self.kvcache.translate_loc_from_full_to_swa(indices)

        q_to_req = torch.empty(num_query_tokens, dtype=torch.int32, device=device)
        offset = 0
        for req_idx, q_len in enumerate(seqlens_q):
            q_to_req[offset : offset + q_len].fill_(req_idx)
            offset += q_len

        q_positions = getattr(batch, "positions", None)
        if q_positions is None:
            q_positions = torch.zeros(num_query_tokens, dtype=torch.int64, device=device)

        batch.attn_metadata = TritonMetadata(
            cu_seqlens_q_gpu=cu_seqlens_q_gpu,
            indptr=indptr,
            indices=indices,
            q_to_req=q_to_req,
            q_positions=q_positions,
            is_decode=is_decode,
            prefix_lens=prefix_lens,
            max_q_len=max(seqlens_q),
            swa_indices=swa_indices,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        self.capture = TritonCaptureData.create(
            max_bs,
            max_seq_len,
            self.device,
            num_q_heads=max(1, self.num_q_heads),
            max_head_dim=max(1, self.max_head_dim),
            max_kv_splits=self.max_kv_splits,
        )
        if self._swa_capture_enabled():
            self.capture.swa_page_table = torch.zeros(
                (max_bs, max_seq_len),
                dtype=torch.int32,
                device=self.device,
            )
        self.capture_bs = sorted(bs_list)
        self.max_graph_bs = max_bs

    def _swa_capture_enabled(self) -> bool:
        # A persistent swa-index capture buffer is needed for the global-paged SWA mode
        # (full->swa mapping translate, recomputed each replay).
        return getattr(self.kvcache, "swa_paged", False)

    def _capture_swa_indices(self) -> torch.Tensor | None:
        assert self.capture is not None
        if not self._swa_capture_enabled():
            return None
        assert self.capture.swa_page_table is not None
        return self.capture.swa_page_table.view(-1)

    def _point_to_capture(self, metadata: TritonMetadata, bs: int) -> None:
        assert self.capture is not None
        indices = self.capture.page_table.view(-1)
        self.capture.cu_seqlens_q[: bs + 1].copy_(metadata.cu_seqlens_q_gpu)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.indptr)
        total = metadata.indices.numel()
        indices[:total].copy_(metadata.indices)
        if metadata.swa_indices is not None:
            swa_indices = self._capture_swa_indices()
            assert swa_indices is not None
            swa_indices[:total].copy_(metadata.swa_indices)
            metadata.swa_indices = swa_indices
        else:
            metadata.swa_indices = None
        q_tokens = metadata.q_positions.numel()
        self.capture.positions[:q_tokens].copy_(metadata.q_positions)
        metadata.cu_seqlens_q_gpu = self.capture.cu_seqlens_q[: bs + 1]
        metadata.indptr = self.capture.cu_seqlens_k[: bs + 1]
        metadata.indices = indices
        metadata.q_to_req = self.capture.q_to_req[: metadata.q_to_req.numel()]
        metadata.q_positions = self.capture.positions[:q_tokens]
        metadata.attn_logits = self.capture.attn_logits[:bs]
        metadata.attn_lse = self.capture.attn_lse[:bs]
        metadata.num_kv_splits = self.capture.num_kv_splits[:bs]

    def prepare_for_capture(self, batch: Batch) -> None:
        bs = batch.size
        assert bs in self.capture_bs and self.capture is not None
        capture = self.capture
        batch.attn_metadata = TritonMetadata(
            cu_seqlens_q_gpu=capture.cu_seqlens_q[: bs + 1],
            indptr=capture.cu_seqlens_k[: bs + 1],
            indices=capture.page_table.view(-1),
            q_to_req=capture.q_to_req[:bs],
            q_positions=capture.positions[:bs],
            is_decode=True,
            prefix_lens=capture.positions[:bs],
            max_q_len=1,
            attn_logits=capture.attn_logits[:bs],
            attn_lse=capture.attn_lse[:bs],
            num_kv_splits=capture.num_kv_splits[:bs],
            swa_indices=self._capture_swa_indices(),
        )

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, TritonMetadata)
        assert self.capture is not None and bs in self.capture_bs
        self._point_to_capture(metadata, bs)
