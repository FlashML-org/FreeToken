"""Paged KV storage for Qwen compressed sparse attention (QSA).

The ordinary K/V cache keeps one row per token.  The QSA index cache keeps
one row per ``compress_ratio`` tokens.  A full KV page therefore maps to one
smaller, page-aligned QSA page.  This keeps the translation exact and cheap:
the compressed row for a complete token group is ``full_row // ratio``.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .mha_pool import MHAKVCache


class QSAKVCache(MHAKVCache):
    """MHA/GQA K/V plus compressed QSA index keys and a small pending ring."""

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        index_num_kv_heads: int,
        index_head_dim: int,
        compress_ratio: int,
        layer_ids: Sequence[int],
    ) -> None:
        if compress_ratio < 2 or page_size % compress_ratio:
            raise ValueError(
                "QSA needs a compression ratio >= 2 that divides the KV page size"
            )
        if dtype.itemsize != 2:
            raise ValueError(f"QSA index keys require a 2-byte compute dtype, got {dtype}")
        self._page_size = int(page_size)
        self._compress_ratio = int(compress_ratio)
        self._compressed_page_size = self._page_size // self._compress_ratio
        self._index_num_kv_heads = int(index_num_kv_heads)
        self._index_head_dim = int(index_head_dim)
        self._num_index_layers = len(layer_ids)
        self._pending_k: torch.Tensor | None = None
        self._pending_pos: torch.Tensor | None = None
        self._pending_rope: torch.Tensor | None = None
        super().__init__(
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            num_pages=num_pages,
            page_size=page_size,
            dtype=dtype,
            device=device,
            layer_ids=layer_ids,
        )
        self._alloc_compressed(num_pages)

    def _alloc_compressed(self, num_pages: int) -> None:
        self._compressed_k = torch.empty(
            self._num_index_layers,
            num_pages,
            self._compressed_page_size,
            self._index_num_kv_heads,
            self._index_head_dim,
            dtype=self.dtype,
            device=self.device,
        )

    def rebuild(self, num_pages: int) -> None:
        self._compressed_k = None
        super().rebuild(num_pages)
        self._alloc_compressed(num_pages)

    def unit_bytes(self) -> tuple[int, int]:
        kv, swa = super().unit_bytes()
        full_tokens = int(self._kv_buffer.shape[2]) * self._page_size
        index_bytes = int(self._compressed_k.numel() * self._compressed_k.element_size())
        return kv + index_bytes // full_tokens, swa

    @property
    def compress_ratio(self) -> int:
        return self._compress_ratio

    @property
    def compressed_page_size(self) -> int:
        return self._compressed_page_size

    def compressed_k_cache(self, layer_id: int) -> torch.Tensor:
        """Return row-flat compressed keys ``[rows, kv_heads, index_dim]``."""
        return self._compressed_k[self._dense(layer_id)].view(
            -1, self._index_num_kv_heads, self._index_head_dim
        )

    def store_compressed_k(
        self, keys: torch.Tensor, compressed_rows: torch.Tensor, layer_id: int
    ) -> None:
        self.compressed_k_cache(layer_id)[compressed_rows.long()] = keys

    def ensure_pending_capacity(self, request_rows: int) -> None:
        """Allocate or grow the per-request incomplete-group ring.

        The ring is tiny compared with the paged cache.  It stores at most
        ``ratio - 1`` useful raw keys per active request and layer.
        """
        current = 0 if self._pending_k is None else int(self._pending_k.shape[1])
        if current >= request_rows:
            return
        new_rows = max(request_rows, max(16, current * 2))
        shape = (
            self._num_index_layers,
            new_rows,
            self._compress_ratio,
            self._index_num_kv_heads,
            self._index_head_dim,
        )
        pending = torch.empty(shape, dtype=self.dtype, device=self.device)
        positions = torch.full(
            shape[:3], -1, dtype=torch.int64, device=self.device
        )
        rope = torch.full(
            (*shape[:3], 3), -1, dtype=torch.int64, device=self.device
        )
        if self._pending_k is not None:
            pending[:, :current].copy_(self._pending_k)
            positions[:, :current].copy_(self._pending_pos)
            rope[:, :current].copy_(self._pending_rope)
        self._pending_k = pending
        self._pending_pos = positions
        self._pending_rope = rope

    def clear_pending(self, layer_id: int, request_row: int) -> None:
        self._pending_pos[self._dense(layer_id), request_row].fill_(-1)

    def pending_group(
        self, layer_id: int, request_row: int, positions: torch.Tensor
    ) -> torch.Tensor:
        dense = self._dense(layer_id)
        slots = torch.remainder(positions, self._compress_ratio).long()
        actual = self._pending_pos[dense, request_row].index_select(0, slots)
        expected = positions.to(device=actual.device, dtype=actual.dtype)
        if not torch.equal(actual, expected):
            raise RuntimeError(
                "QSA pending-key state is missing; use the naive cache and do not "
                "resume a prefix without its QSA state"
            )
        return self._pending_k[dense, request_row].index_select(0, slots)

    def pending_rope_group(
        self, layer_id: int, request_row: int, positions: torch.Tensor
    ) -> torch.Tensor:
        """Return stored [tokens, 3] rotary coordinates after state validation."""
        self.pending_group(layer_id, request_row, positions)
        dense = self._dense(layer_id)
        slots = torch.remainder(positions, self._compress_ratio).long()
        return self._pending_rope[dense, request_row].index_select(0, slots)

    def store_pending(
        self,
        layer_id: int,
        request_row: int,
        positions: torch.Tensor,
        keys: torch.Tensor,
        rope_positions: torch.Tensor | None = None,
    ) -> None:
        dense = self._dense(layer_id)
        slots = torch.remainder(positions, self._compress_ratio).long()
        self._pending_k[dense, request_row].index_copy_(0, slots, keys)
        self._pending_pos[dense, request_row].index_copy_(
            0, slots, positions.to(device=self.device, dtype=torch.int64)
        )
        if rope_positions is None:
            rope_positions = positions.to(device=self.device, dtype=torch.int64).view(-1, 1).expand(-1, 3)
        if rope_positions.shape != (positions.numel(), 3):
            raise ValueError(
                "QSA pending RoPE positions must have shape [tokens, 3], got "
                f"{tuple(rope_positions.shape)}"
            )
        self._pending_rope[dense, request_row].index_copy_(
            0, slots, rope_positions.to(device=self.device, dtype=torch.int64)
        )


__all__ = ["QSAKVCache"]
