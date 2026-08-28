from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import torch
from freetoken.core import Batch, get_global_ctx
from freetoken.utils import is_arch_supported, is_sm100_supported

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


@dataclass
class FACaptureData(BaseCaptureData):
    pass


@dataclass
class FAMetadata(BaseAttnMetadata):
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    max_seqlen_k: int
    max_seqlen_q: int

    page_table: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class FlashAttentionBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig):
        if is_arch_supported(12, 0):
            raise RuntimeError(
                "The 'fa' attention backend does not support compute capability 12.x GPUs: "
                "sgl-kernel ships no FA3/FA4 kernels for sm_120/121. "
                "Use --attention-backend fi (or triton) instead."
            )
        ctx = get_global_ctx()
        self.config = config
        self.kvcache = ctx.kv_cache
        self.page_size = ctx.page_size
        self.capture: FACaptureData | None = None
        self.dflash_target_verify_capture: dict[int, FACaptureData] = {}
        self.max_graph_bs = 0
        self.capture_bs: List[int] = []
        self.scale = config.head_dim**-0.5
        self.version = 4 if is_sm100_supported() else 3

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        if attn_spec is not None:
            # This backend has no window/sinks/sm_scale plumbing; dropping the spec
            # silently would attend with the wrong scale or an unbounded window.
            raise ValueError("The fa backend does not support per-call AttentionSpec.")
        metadata = batch.attn_metadata
        assert isinstance(metadata, FAMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        return _fa_sgl_impl(
            q=q,
            k_cache=self.kvcache.k_cache(layer_id),
            v_cache=self.kvcache.v_cache(layer_id),
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            softmax_scale=self.scale,
            version=self.version,
        )

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs

        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_k = max(seqlens_k)
        max_seqlen_q = max(seqlens_q)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.kvcache.device
        cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS)
        cache_seqlens = cache_seqlens.to(device, non_blocking=True)
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

        if max_seqlen_q == 1:
            cu_seqlens_q = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            cu_seqlens_q = cu_seqlens_k
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)
            cu_seqlens_q = cu_seqlens_q.to(self.kvcache.device, non_blocking=True)

        page_table = get_global_ctx().page_table
        new_page_table = torch.stack(  # NOTE: global page table treat page_size = 1, we need slice
            [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
        )
        if self.page_size > 1:
            new_page_table.div_(self.page_size, rounding_mode="floor")
        batch.attn_metadata = FAMetadata(
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FACaptureData.create(max_bs, max_seq_len // self.page_size, self.kvcache.device)
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture
        metadata = FAMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            max_seqlen_q=1,  # decode only
            page_table=capture.page_table[:bs, :],
        )
        batch.attn_metadata = metadata

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FAMetadata)
        assert self.capture is not None and bs in self.capture_bs
        # cu_seqlens_q is always [0, 1, 2, ..., bs] for decode (i.e. no-op)
        table_len = metadata.page_table.size(1)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)

    def reset_capture(self) -> None:
        super().reset_capture()
        self.dflash_target_verify_capture = {}

    def init_dflash_target_verify_capture_graph(
        self,
        max_seq_len: int,
        verify_lens: List[int],
    ) -> None:
        assert not self.dflash_target_verify_capture, "DFlash target verify capture already initialized."
        self.dflash_target_verify_capture = {
            verify_len: FACaptureData.create(
                1,
                max_seq_len // self.page_size,
                self.kvcache.device,
            )
            for verify_len in sorted(verify_lens)
        }
        for verify_len, capture in self.dflash_target_verify_capture.items():
            capture.cu_seqlens_q.copy_(torch.tensor([0, verify_len], dtype=torch.int32, device=self.kvcache.device))
            capture.cu_seqlens_k.copy_(torch.tensor([0, verify_len], dtype=torch.int32, device=self.kvcache.device))
            capture.seq_lens[:1].fill_(verify_len)

    def prepare_for_dflash_target_verify_capture(self, batch: Batch, verify_len: int) -> None:
        assert batch.size == 1 and verify_len in self.dflash_target_verify_capture
        capture = self.dflash_target_verify_capture[verify_len]
        batch.attn_metadata = FAMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[:2],
            cu_seqlens_q=capture.cu_seqlens_q[:2],
            cache_seqlens=capture.seq_lens[:1],
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            max_seqlen_q=verify_len,
            page_table=capture.page_table[:1, :],
        )

    def prepare_for_dflash_target_verify_replay(self, batch: Batch, verify_len: int) -> None:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FAMetadata)
        assert verify_len in self.dflash_target_verify_capture
        capture = self.dflash_target_verify_capture[verify_len]
        table_len = metadata.page_table.size(1)
        capture.seq_lens[:1].copy_(metadata.cache_seqlens[:1])
        capture.cu_seqlens_k[:2].copy_(metadata.cu_seqlens_k[:2])
        capture.page_table[:1, :table_len].copy_(metadata.page_table[:1, :table_len])
        batch.attn_metadata = FAMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[:2],
            cu_seqlens_q=capture.cu_seqlens_q[:2],
            cache_seqlens=capture.seq_lens[:1],
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            max_seqlen_q=verify_len,
            page_table=capture.page_table[:1, :],
        )


def _fa_sgl_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    version: int,
    sm_margin: int = 0,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    softcap: float = 0.0,  # 0.0 means deactivated
    num_splits: int = 0,  # Can be tuned for speed
    pack_gqa: bool | None = None,  # Can be tuned for speed
    causal: bool = True,
) -> torch.Tensor:
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
    except ImportError as e:
        raise ImportError(
            "sgl_kernel.flash_attn is not found. Please install it with `pip install 'freetoken[sgl]'`.\n"
            "If you're sure it's correctly installed, try `apt update && apt install libnuma1`."
        ) from e

    return flash_attn_with_kvcache(  # type: ignore
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=softmax_scale,
        sm_margin=sm_margin,
        window_size=window_size,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        causal=causal,
        ver=version,  # TODO: support FA4 on blackwell
    )
