"""Block weights streamed from pinned host banks through two device staging buffers: block i computes from buffer i % 2 while the copy stream fills the other with block i + 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, List, Sequence

import torch

from freetoken.layers import BaseOP

_ALIGN = 256


@dataclass(frozen=True)
class _Slot:
    owner: Any
    attr: str
    offset: int
    nbytes: int
    dtype: torch.dtype
    shape: torch.Size

    def view(self, row: torch.Tensor) -> torch.Tensor:
        return row[self.offset : self.offset + self.nbytes].view(self.dtype).view(self.shape)


def _slots(block: BaseOP) -> tuple[List[_Slot], int]:
    """Every tensor attribute of the block (recursively), laid out back to back with 256-byte alignment."""
    slots: List[_Slot] = []
    offset = 0

    def walk(op: BaseOP) -> None:
        nonlocal offset
        for name, value in op.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(value, torch.Tensor):
                nbytes = value.numel() * value.element_size()
                slots.append(_Slot(op, name, offset, nbytes, value.dtype, value.shape))
                offset += -(-nbytes // _ALIGN) * _ALIGN
            elif isinstance(value, BaseOP):
                walk(value)

    walk(block)
    return slots, offset


class BlockWeightStreamer:
    """Owns the host banks and the two staging buffers of a block stack; drives one forward with `blocks()`."""

    def __init__(self, blocks: Sequence[BaseOP], device: torch.device):
        self.device = device
        self._layouts = [_slots(b) for b in blocks]
        row_bytes = self._layouts[0][1]
        assert all(nbytes == row_bytes for _, nbytes in self._layouts), "streamed blocks must share one layout"
        self.bank = torch.empty((len(blocks), row_bytes), dtype=torch.uint8, pin_memory=True)
        self.staging = torch.empty((2, row_bytes), dtype=torch.uint8, device=device)
        for b, (slots, _) in enumerate(self._layouts):
            for s in slots:
                s.view(self.bank[b]).copy_(getattr(s.owner, s.attr))
        torch.cuda.synchronize(device)
        self._bind_all_to_host()
        self.copy_stream = torch.cuda.Stream(device=device)
        self.begin_event = torch.cuda.Event()
        self.ready_events = [torch.cuda.Event() for _ in range(2)]
        self.release_events = [torch.cuda.Event() for _ in range(2)]
        self._holder: List[int | None] = [None, None]
        self._has_release: List[bool] = [False, False]

    @property
    def device_bytes(self) -> int:
        return self.staging.numel()

    def _bind_all_to_host(self) -> None:
        # between forwards the attributes point at the pinned bank, so state_dict() stays complete and correct
        for b, (slots, _) in enumerate(self._layouts):
            for s in slots:
                setattr(s.owner, s.attr, s.view(self.bank[b]))

    def unstream(self) -> None:
        """Put every block's tensors back on the device as plain resident copies."""
        for b, (slots, _) in enumerate(self._layouts):
            for s in slots:
                setattr(s.owner, s.attr, s.view(self.bank[b]).to(self.device))
        torch.cuda.synchronize(self.device)

    def _prefetch(self, i: int) -> None:
        if i >= len(self._layouts):
            return
        buf = i % 2
        if self._holder[buf] == i:
            return
        with torch.cuda.stream(self.copy_stream):
            if self._has_release[buf]:
                self.copy_stream.wait_event(self.release_events[buf])
            self.staging[buf].copy_(self.bank[i], non_blocking=True)
            self.ready_events[buf].record(self.copy_stream)
        self._holder[buf] = i

    def blocks(self, ops: Sequence[BaseOP]) -> Iterator[tuple[int, BaseOP]]:
        """Yield (i, block) with block i's tensors bound to a filled staging buffer while block i + 1 streams in."""
        compute = torch.cuda.current_stream(self.device)
        # the copy stream must not overwrite staging that a still-running previous forward reads
        self.begin_event.record(compute)
        self.copy_stream.wait_event(self.begin_event)
        for i, op in enumerate(ops):
            buf = i % 2
            self._prefetch(i)
            self._prefetch(i + 1)
            compute.wait_event(self.ready_events[buf])
            slots, _ = self._layouts[i]
            for s in slots:
                setattr(s.owner, s.attr, s.view(self.staging[buf]))
            yield i, op
            self.release_events[buf].record(compute)
            self._has_release[buf] = True
            for s in slots:
                setattr(s.owner, s.attr, s.view(self.bank[i]))


__all__ = ["BlockWeightStreamer"]
