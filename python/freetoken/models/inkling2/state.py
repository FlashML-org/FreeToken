from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx

if TYPE_CHECKING:
    from freetoken.core import Batch

    from .config import Inkling2Args


class ConvState:
    """The short-convolution left context of every visit, one row per request.

    ``rows[v]`` is ``[table_rows, K - 1, width]``: the last ``K - 1`` inputs of visit ``v``'s
    four convolutions (key, value, attention output, MoE output) side by side, keyed by
    ``Req.table_idx`` like the page table. The inputs are bf16 projections, so bf16 storage
    is exact. A request's first prefill chunk (no prefix) starts from zeros, so rows are
    never cleared. ``slots`` maps the batch's requests to their rows; ``stage`` fills it
    before every forward, eager or replayed, so captured graphs read the current batch.
    """

    def __init__(self, args: Inkling2Args, dtype: torch.dtype):
        kv = args.num_kv_heads * args.head_dim
        self.widths = {"k": kv, "v": kv, "attn": args.hidden_size, "mlp": args.hidden_size}
        self.offsets: dict[str, int] = {}
        offset = 0
        for name, width in self.widths.items():
            self.offsets[name] = offset
            offset += width
        self.width = offset
        self.num_visits = args.num_visits
        self.taps = args.conv_kernel - 1
        self.dtype = dtype
        self.state: torch.Tensor | None = None
        self.slots: torch.Tensor | None = None

    def _ensure(self) -> None:
        # built at the first forward: the model itself is constructed on the meta device
        if self.state is not None:
            return
        page_table = get_global_ctx().page_table
        rows, device = page_table.shape[0], page_table.device
        self.state = torch.zeros((self.num_visits, rows, self.taps, self.width), dtype=self.dtype, device=device)
        self.slots = torch.zeros(rows, dtype=torch.int32, device=device)

    def stage(self, batch: Batch) -> None:
        self._ensure()
        assert self.slots is not None
        rows = [req.table_idx for req in batch.padded_reqs]
        host = torch.tensor(rows, dtype=torch.int32, pin_memory=self.slots.is_cuda)
        self.slots[: len(rows)].copy_(host, non_blocking=True)

    def batch_slots(self, num_reqs: int) -> torch.Tensor:
        self._ensure()
        assert self.slots is not None
        return self.slots[:num_reqs]

    def view(self, visit: int, name: str) -> torch.Tensor:
        """``[table_rows, K - 1, width]`` view of one convolution's rows in one visit."""
        self._ensure()
        assert self.state is not None
        lo = self.offsets[name]
        return self.state[visit, :, :, lo : lo + self.widths[name]]

    def bytes_per_request(self) -> int:
        return self.num_visits * self.taps * self.width * self.dtype.itemsize


__all__ = ["ConvState"]
