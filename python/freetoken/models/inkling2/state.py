from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx

from .config import CONV_STATE, conv_state_widths

if TYPE_CHECKING:
    from .config import Inkling2Args


class ConvState:
    """Views of the short-convolution left context, the LinearStatePool slot state ``CONV_STATE``.

    Pool layer ``v`` is visit ``v``: ``[slots, K - 1, width]``, the last ``K - 1`` inputs of the
    visit's four convolutions (key, value, attention output, MoE output) side by side. Requests
    index it by their state slot (``fla_metadata.cache_indices``); a prefill writes the snapshots
    its cache manager planned (``fla_metadata.track_dst``) from the inputs before each boundary.
    """

    def __init__(self, args: Inkling2Args):
        self.widths = conv_state_widths(args)
        self.offsets: dict[str, int] = {}
        offset = 0
        for name, width in self.widths.items():
            self.offsets[name] = offset
            offset += width
        self._tensor: torch.Tensor | None = None
        self._views: list[dict[str, torch.Tensor]] = []

    def view(self, visit: int, name: str) -> torch.Tensor:
        pool = get_global_ctx().linear_state_pool
        tensor = pool.slot_states[CONV_STATE]
        if tensor is not self._tensor:  # a pool rebuild replaces the tensor
            self._tensor = tensor
            self._views = [
                {n: tensor[v, :, :, lo : lo + self.widths[n]] for n, lo in self.offsets.items()}
                for v in range(tensor.shape[0])
            ]
        return self._views[visit][name]

    @staticmethod
    def slots() -> torch.Tensor:
        return get_global_ctx().batch.fla_metadata.cache_indices

    @staticmethod
    def track(view: torch.Tensor, x: torch.Tensor) -> None:
        """Snapshot the inputs before each planned boundary into its slot (prefill only)."""
        fla = get_global_ctx().batch.fla_metadata
        if fla.track_dst is not None:
            view[fla.track_dst] = x[fla.track_conv_src]


__all__ = ["ConvState"]
