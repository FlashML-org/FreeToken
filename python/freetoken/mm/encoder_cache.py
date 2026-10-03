"""Encoder embedding cache keyed by content hash; an entry lives while any request still has rows of it to gather."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)


@dataclass
class _Entry:
    tensor: torch.Tensor | None = None
    # uid -> embedding rows the request still has to gather, summed over its occurrences of the image
    remaining: Dict[int, int] = field(default_factory=dict)


class EncoderCache:
    def __init__(self, storage: str = "cpu") -> None:
        assert storage in ("cpu", "cuda")
        self._storage = storage
        self._entries: Dict[int, _Entry] = {}

    def register(self, item_hash: int, uid: int, rows: int) -> None:
        """Claim rows of the image for uid before any of its chunks run; a repeated image adds up."""
        if rows <= 0:
            return
        entry = self._entries.setdefault(item_hash, _Entry())
        entry.remaining[uid] = entry.remaining.get(uid, 0) + rows

    def has(self, item_hash: int) -> bool:
        entry = self._entries.get(item_hash)
        return entry is not None and entry.tensor is not None

    def put(self, item_hash: int, embedding: torch.Tensor) -> None:
        entry = self._entries.get(item_hash)
        assert entry is not None, f"image {item_hash} encoded before any request registered it"
        if entry.tensor is not None:
            return
        if self._storage == "cpu":
            stored = torch.empty(
                embedding.shape, dtype=embedding.dtype, device="cpu",
                pin_memory=torch.cuda.is_available(),
            )
            stored.copy_(embedding, non_blocking=True)
        else:
            stored = embedding
        entry.tensor = stored

    def get_slice(self, item_hash: int, lo: int, hi: int, device: torch.device) -> torch.Tensor:
        view = self._entries[item_hash].tensor[lo:hi]
        if view.device == device:
            return view
        return view.to(device, non_blocking=True)

    def consume(self, item_hash: int, uid: int, rows: int) -> None:
        """Account rows uid gathered; the last row of the last holder frees the entry."""
        entry = self._entries[item_hash]
        left = entry.remaining[uid] - rows
        assert left >= 0, f"request {uid} gathered more rows of image {item_hash} than it registered"
        if left:
            entry.remaining[uid] = left
            return
        del entry.remaining[uid]
        if not entry.remaining:
            del self._entries[item_hash]

    def release(self, uid: int, hashes: list[int]) -> None:
        """Drop uid's claims whatever it gathered (abort); entries nobody else holds are freed."""
        for h in hashes:
            entry = self._entries.get(h)
            if entry is None:
                continue
            entry.remaining.pop(uid, None)
            if not entry.remaining:
                del self._entries[h]

    def stats(self) -> tuple[int, int]:
        stored = [e.tensor for e in self._entries.values() if e.tensor is not None]
        return len(stored), sum(t.numel() * t.element_size() for t in stored)


__all__ = ["EncoderCache"]
