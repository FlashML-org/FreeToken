"""DFlashWindow: the draft's context rows of each request's last positions, riding the target's GDN slots so a prefix-cache hit restores them with the GDN state and the ring is rebuilt from them."""

from __future__ import annotations

import torch

from freetoken.models.config import SlotStateSpec


class DFlashWindow:
    """Context rows of positions ``p`` at row ``p % R`` of each GDN slot; ``fmt="unquantized"`` keeps them in the activation dtype, one ``[R, H]`` tensor per slot."""

    NAME = "dflash_window"
    FORMATS = ("unquantized",)

    @classmethod
    def length(cls, window: int, block: int) -> int:
        """Rows per slot: the window plus room for a verify's unaccepted block rows and for a prefill's rows past the GDN boundary it tracks."""
        from freetoken.kernel.fla.chunk import CHUNK_SIZE

        return window + max(block, CHUNK_SIZE)

    @classmethod
    def slot_specs(cls, R: int, H: int, fmt: str = "unquantized") -> tuple[SlotStateSpec, ...]:
        cls._check(fmt)
        return (SlotStateSpec(name=cls.NAME, shape=(R, H)),)

    @classmethod
    def _check(cls, fmt: str) -> None:
        if fmt not in cls.FORMATS:
            raise ValueError(f"unknown DFlash window format {fmt!r}, known: {cls.FORMATS}")

    def __init__(self, tensors: dict[str, torch.Tensor], R: int, H: int, fmt: str = "unquantized") -> None:
        self._check(fmt)
        self.R, self.H, self.fmt = R, H, fmt
        self.tensor = tensors[self.NAME]
        assert tuple(self.tensor.shape[1:]) == (R, H), f"window tensor {tuple(self.tensor.shape)} is not [slots, {R}, {H}]"

    def write(self, rows: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor) -> None:
        """``rows [n, H]`` of ``positions`` into ``slots``; positions distinct modulo R within one call."""
        self.tensor[slots, positions.long() % self.R] = rows

    def read(self, slot: int, start: int, n: int) -> torch.Tensor:
        """``[n, H]`` rows of positions ``start .. start+n-1`` held by ``slot``."""
        return self.tensor[slot, torch.arange(start, start + n, device=self.tensor.device) % self.R]
