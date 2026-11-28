from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx

from .config import CONV_STATE, conv_state_widths

if TYPE_CHECKING:
    from .config import Inkling2Args


class ConvState:
    """Views of the short-convolution inputs kept per token, the KV pool's window state ``CONV_STATE``.

    Pool layer ``v`` is visit ``v``: ``[rows, width]``, a token's inputs to the visit's four
    convolutions (key, value, attention output, MoE output) side by side. A token's row is
    reached from its full KV slot through ``row_map``; the cache gives rows only to the tokens a
    later chunk, decode step or resume reads.
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

    def rows(self, visit: int, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """The rows of convolution ``name`` at ``visit``, and the full slot -> row mapping."""
        tensor, row_map = get_global_ctx().kv_cache.state_rows(CONV_STATE)
        if tensor is not self._tensor:  # a pool rebuild replaces the tensor
            self._tensor = tensor
            self._views = [
                {n: tensor[v, :, lo : lo + self.widths[n]] for n, lo in self.offsets.items()}
                for v in range(tensor.shape[0])
            ]
        return self._views[visit][name], row_map


__all__ = ["ConvState"]
