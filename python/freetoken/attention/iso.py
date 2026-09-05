"""ISO attention backend: full attention over ISOKVCache (ISO3/ISO4 packed KV).

Decode: quantize-on-write (store_kv packs the new token), then the packed
paged decode kernel. Extend (prefill): attention reads the packed prefix plus
the bf16 extend rows, and the new tokens are packed into the pool AFTER the
attention pass (deferred quantization — prefill never consumes its own
quantized K/V).

No sliding window / attention sinks (plain FULL attention only).
"""

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
class IsoCaptureData(BaseCaptureData):
    @classmethod
    def create(cls, max_bs: int, max_seq_len: int, device: torch.device, **kwargs):
        return cls(
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),
            positions=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device),
            **kwargs,
        )


@dataclass
class IsoMetadata(BaseAttnMetadata):
    cu_seqlens_q_gpu: torch.Tensor
    indptr: torch.Tensor
    indices: torch.Tensor
    prefix_indptr: torch.Tensor
    prefix_indices: torch.Tensor
    is_decode: bool
    max_q_len: int

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


class IsoAttentionBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        self.capture: IsoCaptureData | None = None
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.num_q_heads = int(getattr(config, "num_qo_heads", 1))
        kv_groups = getattr(config, "kv_cache_group_specs", lambda: ())()
        self.max_head_dim = max(
            (group.head_dim for group in kv_groups),
            default=int(getattr(config, "head_dim", 1)),
        )
        self.iso_fmt = getattr(self.kvcache, "iso_fmt", "iso3")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        from freetoken.kernel.iso import iso_attention_decode, iso_attention_extend

        metadata = batch.attn_metadata
        assert isinstance(metadata, IsoMetadata)
        spec = attn_spec or AttentionSpec()
        if spec.sliding_window is not None or spec.sinks is not None:
            raise NotImplementedError(
                "iso attention backend does not support sliding windows or sinks"
            )
        scale = spec.sm_scale if spec.sm_scale is not None else q.shape[-1] ** -0.5

        k_raw = self.kvcache.k_cache(layer_id)  # [pages, ps, heads, row_bytes] uint8
        v_raw = self.kvcache.v_cache(layer_id)
        kv_heads = k_raw.shape[-2]
        head_dim = q.shape[-1]
        k_flat = k_raw.view(-1, k_raw.shape[-2] * k_raw.shape[-1])
        v_flat = v_raw.view(-1, v_raw.shape[-2] * v_raw.shape[-1])

        n = q.shape[0]
        nq = q.shape[1]
        out = torch.empty_like(q)
        if metadata.is_decode:
            # quantize-on-write, then attend the whole packed sequence
            self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
            iso_attention_decode(
                q.reshape(n, -1), out.reshape(n, -1), k_flat, v_flat,
                metadata.indptr, metadata.indices,
                nq, kv_heads, head_dim, scale, self.iso_fmt,
            )
        else:
            # attend packed prefix + bf16 extend rows, THEN pack the new tokens
            iso_attention_extend(
                q.reshape(n, -1), out.reshape(n, -1), k_flat, v_flat,
                k.reshape(n, -1), v.reshape(n, -1),
                metadata.cu_seqlens_q_gpu, metadata.prefix_indptr,
                metadata.prefix_indices,
                nq, kv_heads, head_dim, scale, metadata.max_q_len, self.iso_fmt,
            )
            self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        return out

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        device = self.device
        page_table = get_global_ctx().page_table
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        is_decode = max(seqlens_q) == 1

        indptr = torch.tensor([0] + seqlens_k, dtype=torch.int32, device=device).cumsum_(0)
        if is_decode:
            cu_seqlens_q_gpu = torch.arange(0, len(reqs) + 1, device=device, dtype=torch.int32)
        else:
            cu_seqlens_q_gpu = torch.tensor(
                [0] + seqlens_q, dtype=torch.int32, device=device
            ).cumsum_(0)
        indices = torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs])
        if is_decode:
            # decode attends the packed pool only; prefix split stays unused
            prefix_indptr = cu_seqlens_q_gpu
            prefix_indices = indices[:0]
        else:
            prefix_indptr = torch.tensor(
                [0] + cached_lens, dtype=torch.int32, device=device
            ).cumsum_(0)
            prefix_indices = torch.cat(
                [page_table[req.table_idx, : req.cached_len] for req in reqs]
            )

        batch.attn_metadata = IsoMetadata(
            cu_seqlens_q_gpu=cu_seqlens_q_gpu,
            indptr=indptr,
            indices=indices,
            prefix_indptr=prefix_indptr,
            prefix_indices=prefix_indices,
            is_decode=is_decode,
            max_q_len=max(seqlens_q),
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        self.capture = IsoCaptureData.create(max_bs, max_seq_len, self.device)
        self.capture_bs = sorted(bs_list)
        self.max_graph_bs = max_bs

    def prepare_for_capture(self, batch: Batch) -> None:
        bs = batch.size
        assert bs in self.capture_bs and self.capture is not None
        capture = self.capture
        batch.attn_metadata = IsoMetadata(
            cu_seqlens_q_gpu=capture.cu_seqlens_q[: bs + 1],
            indptr=capture.cu_seqlens_k[: bs + 1],
            indices=capture.page_table.view(-1),
            prefix_indptr=capture.cu_seqlens_k[: bs + 1],
            prefix_indices=capture.page_table.view(-1),
            is_decode=True,
            max_q_len=1,
        )

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, IsoMetadata)
        assert self.capture is not None and bs in self.capture_bs
        capture = self.capture
        capture.cu_seqlens_q[: bs + 1].copy_(metadata.cu_seqlens_q_gpu)
        capture.cu_seqlens_k[: bs + 1].copy_(metadata.indptr)
        indices = capture.page_table.view(-1)
        total = metadata.indices.numel()
        indices[:total].copy_(metadata.indices)
        metadata.cu_seqlens_q_gpu = capture.cu_seqlens_q[: bs + 1]
        metadata.indptr = capture.cu_seqlens_k[: bs + 1]
        metadata.indices = indices
