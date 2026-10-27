"""Per-request draft state carried from one step to the next, by page-table row."""

from __future__ import annotations

import torch


class HiddenRelay:
    """For each table row: the draft hidden the next proposal starts from, the first draft token it predicts, and the target hidden that closes a prefill chunk."""

    def __init__(self, rows: int, hidden_size: int, dtype: torch.dtype, device: torch.device) -> None:
        self.hidden = torch.zeros(rows, hidden_size, dtype=dtype, device=device)
        self.d1 = torch.zeros(rows, dtype=torch.int64, device=device)
        self.carry = torch.zeros(rows, hidden_size, dtype=dtype, device=device)
