"""Slicing routed experts along their intermediate dimension.

Tensor parallelism, CPU expert pools and partitioned fetches each hold a contiguous
interval of every expert's intermediate dim ``I``. Which axis of a tensor carries ``I``,
and how densely it is packed there, is a property of the expert format, so the kernel
declares it once (``MoEKernel.piece_axes`` for checkpoint pieces, ``bank_axes`` for its
bank layout) and the rest is derived from that declaration: the loader's piece slicing,
the bytes a narrow bank occupies inside a wider one, and the shard alignment.

Shards of one cover compute partial expert outputs that a reduction sums, so a term that
must enter the sum once (the down-projection bias) is kept by the shard starting at 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal, Mapping

import torch

from .base import BankSpec, MoEConfig, MoEKernel


@dataclass(frozen=True, order=True)
class Interval:
    """The half-open range ``[start, stop)`` of the intermediate dim."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if not 0 <= self.start <= self.stop:
            raise ValueError(f"invalid interval [{self.start}, {self.stop})")

    @property
    def width(self) -> int:
        return self.stop - self.start

    def contains(self, other: Interval) -> bool:
        return self.start <= other.start and other.stop <= self.stop

    @classmethod
    def split(cls, total: int, parts: int, *, align: int = 1) -> tuple[Interval, ...]:
        """``parts`` equal contiguous intervals covering ``[0, total)``, each ``align``-aligned."""
        if parts < 1 or align < 1 or total < 1 or total % (parts * align):
            raise ValueError(f"cannot split {total} into {parts} intervals aligned to {align}")
        width = total // parts
        return tuple(cls(r * width, (r + 1) * width) for r in range(parts))


@dataclass(frozen=True)
class IntermediateAxis:
    """How one per-expert tensor (expert dim excluded) carries the intermediate dim.

    ``split``: ``axis`` holds ``halves`` blocks laid end to end, each spanning all of ``I``
    (``[gate; up]`` rows have two). ``broadcast`` lets a split tensor be a per-expert value
    instead (no such axis, or size 1), which every shard keeps. ``replicate``: no ``I`` dim.
    ``once``: no ``I`` dim, but summed by the shard reduction, so only the shard at 0 keeps it.
    """

    mode: Literal["split", "replicate", "once"]
    axis: int = 0
    halves: int = 1
    broadcast: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("split", "replicate", "once") or self.axis < 0 or self.halves < 1:
            raise ValueError(f"invalid intermediate axis {self}")

    def is_split(self, shape: tuple[int, ...]) -> bool:
        if self.mode != "split":
            return False
        return not (self.broadcast and (len(shape) <= self.axis or shape[self.axis] == 1))


REPLICATE = IntermediateAxis("replicate")
ONCE = IntermediateAxis("once")
ROWS = IntermediateAxis("split", 0)
COLS = IntermediateAxis("split", 1)
GATE_UP_ROWS = IntermediateAxis("split", 0, halves=2)


def standard_piece_axes(interleaved: bool) -> dict[str, IntermediateAxis]:
    """The checkpoint piece roles of ``expert_pieces``: ``gate`` / ``up`` / ``down`` or a fused
    ``gate_up``, with their ``_scale`` / ``_global`` / ``_input`` / ``_bias`` companions."""
    fused = IntermediateAxis("split", 0, halves=1 if interleaved else 2)
    axes: dict[str, IntermediateAxis] = {}
    for proj, rows in (("gate", ROWS), ("up", ROWS), ("gate_up", fused)):
        axes[proj] = axes[f"{proj}_scale"] = axes[f"{proj}_bias"] = rows
        axes[f"{proj}_global"] = IntermediateAxis("split", 0, halves=rows.halves, broadcast=True)
        axes[f"{proj}_input"] = REPLICATE
    axes["down"] = axes["down_scale"] = COLS
    axes["down_global"] = axes["down_input"] = REPLICATE
    axes["down_bias"] = ONCE
    return axes


def _span(rule: IntermediateAxis, size: int, intermediate: int, view: Interval, origin: int = 0) -> tuple[int, int]:
    """Element offset and length of ``view`` within one half of a ``size``-long axis whose
    halves each span ``intermediate`` units starting at unit ``origin``."""
    if size % rule.halves:
        raise ValueError(f"axis of {size} does not hold {rule.halves} equal halves")
    per_unit = Fraction(size // rule.halves, intermediate)
    lo, n = (view.start - origin) * per_unit, view.width * per_unit
    if lo.denominator != 1 or n.denominator != 1:
        raise ValueError(f"interval [{view.start}, {view.stop}) splits a packed or quantized group ({per_unit} elements per unit)")
    return int(lo), int(n)


def granularity(rule: IntermediateAxis, shape: tuple[int, ...], intermediate: int) -> int:
    """Smallest interval step that keeps a tensor of per-expert ``shape`` whole-element aligned."""
    if not rule.is_split(shape):
        return 1
    size = shape[rule.axis]
    if size % rule.halves:
        raise ValueError(f"axis of {size} does not hold {rule.halves} equal halves")
    return Fraction(size // rule.halves, intermediate).denominator


def slice_tensor(tensor: torch.Tensor, rule: IntermediateAxis, view: Interval, intermediate: int) -> torch.Tensor:
    """The ``view`` shard of a stacked ``[E, ...]`` tensor of ``I = intermediate`` experts (a view where possible)."""
    shape = tuple(tensor.shape[1:])
    if rule.mode == "once":
        return tensor if view.start == 0 else torch.zeros_like(tensor)
    if not rule.is_split(shape):
        return tensor
    if len(shape) <= rule.axis:
        raise ValueError(f"tensor of per-expert shape {shape} has no intermediate axis {rule.axis}")
    dim = rule.axis + 1
    half = shape[rule.axis] // rule.halves
    lo, n = _span(rule, shape[rule.axis], intermediate, view)
    parts = [tensor.narrow(dim, h * half + lo, n) for h in range(rule.halves)]
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim)


@dataclass(frozen=True)
class ByteRuns:
    """Where a narrow bank row lives inside a wider one: ``count`` runs of ``length`` bytes,
    ``stride`` bytes apart, the first ``offset`` bytes into the wide expert row. ``zero``: the
    narrow row is all zeros instead (a ``once`` term another shard owns)."""

    offset: int
    length: int
    stride: int
    count: int
    zero: bool = False

    @property
    def nbytes(self) -> int:
        return self.length * self.count

    def gather(self, row: torch.Tensor) -> torch.Tensor:
        """Reference copy: the narrow bytes of one wide expert row (a 1-D uint8 tensor)."""
        if self.zero:
            return torch.zeros(self.nbytes, dtype=torch.uint8, device=row.device)
        return row.as_strided((self.count, self.length), (self.stride, 1), row.storage_offset() + self.offset).reshape(-1)


def _nbytes(spec: BankSpec) -> int:
    return math.prod(spec.shape) * spec.dtype.itemsize


def bank_runs(spec: BankSpec, rule: IntermediateAxis, stored: Interval, view: Interval) -> ByteRuns:
    """Byte runs of ``view`` within one expert row of a bank ``spec`` laid out for ``stored``."""
    if not stored.contains(view):
        raise ValueError(f"view [{view.start}, {view.stop}) is outside the stored [{stored.start}, {stored.stop})")
    total = _nbytes(spec)
    if rule.mode == "once":
        return ByteRuns(0, total, total, 1, zero=view.start != 0)
    if not rule.is_split(spec.shape):
        return ByteRuns(0, total, total, 1)
    size = spec.shape[rule.axis]
    lo, n = _span(rule, size, stored.width, view, stored.start)
    inner = math.prod(spec.shape[rule.axis + 1:]) * spec.dtype.itemsize
    count = math.prod(spec.shape[: rule.axis]) * rule.halves
    stride = size // rule.halves * inner
    if n * inner == stride:
        return ByteRuns(0, total, total, 1)
    return ByteRuns(lo * inner, n * inner, stride, count)


@dataclass(frozen=True)
class ExpertSlicing:
    """One expert layer's slicing rules at its full width; see ``MoEMethod.expert_slicing``."""

    kernel: MoEKernel
    cfg: MoEConfig

    @property
    def intermediate(self) -> int:
        return self.cfg.intermediate

    @property
    def piece_axes(self) -> dict[str, IntermediateAxis] | None:
        return self.kernel.piece_axes(self.cfg)

    @property
    def bank_axes(self) -> dict[str, IntermediateAxis] | None:
        return self.kernel.bank_axes(self.cfg)

    def config_at(self, width: int) -> MoEConfig:
        """The layer config of a shard ``width`` wide, which the kernel's layout and pack take."""
        return replace(self.cfg, intermediate=width, tp_rank=0, tp_size=1, shard=None)

    def layout_at(self, width: int) -> dict[str, BankSpec]:
        return self.kernel.layout(self.config_at(width))

    @property
    def alignment(self) -> int:
        """Step every shard boundary must be a multiple of: the kernel's own need and the bank layout's packing."""
        step = self.kernel.shard_alignment(self.cfg)
        axes = self.bank_axes
        if axes is not None:
            for role, spec in self.layout_at(self.intermediate).items():
                step = math.lcm(step, granularity(axes[role], spec.shape, self.intermediate))
        return step

    def check(self, view: Interval) -> None:
        if self.piece_axes is None:
            raise ValueError("this expert format cannot be sliced along its intermediate dim")
        if view.width == 0 or view.stop > self.intermediate:
            raise ValueError(f"shard [{view.start}, {view.stop}) is not a nonempty part of [0, {self.intermediate})")
        step = self.alignment
        if any(edge % step for edge in (view.start, view.stop) if edge not in (0, self.intermediate)):
            raise ValueError(f"shard [{view.start}, {view.stop}) is not aligned to {step}")

    def slice_pieces(self, pieces: Mapping[str, torch.Tensor], view: Interval) -> dict[str, torch.Tensor]:
        """The ``view`` shard of one checkpoint piece (``{role: [E, ...]}``), ready for the kernel's pack."""
        self.check(view)
        if view.width == self.intermediate:
            return dict(pieces)
        axes = self.piece_axes
        unknown = set(pieces) - set(axes)
        if unknown:
            raise ValueError(f"no intermediate-axis rule for expert piece roles {sorted(unknown)}")
        return {role: slice_tensor(t, axes[role], view, self.intermediate) for role, t in pieces.items()}

    def bank_runs(self, stored: Interval, view: Interval) -> dict[str, ByteRuns]:
        """Per bank role, where the ``view`` layout's expert row lives inside a ``stored`` one."""
        axes = self.bank_axes
        if axes is None:
            raise ValueError("this expert kernel's banks cannot be sliced; repack from pieces instead")
        self.check(stored)
        self.check(view)
        wide, narrow = self.layout_at(stored.width), self.layout_at(view.width)
        runs = {role: bank_runs(spec, axes[role], stored, view) for role, spec in wide.items()}
        for role, spec in narrow.items():
            if runs[role].nbytes != _nbytes(spec):
                raise AssertionError(f"bank {role}: {runs[role].nbytes} sliced bytes, layout wants {_nbytes(spec)}")
        return runs


__all__ = [
    "COLS", "GATE_UP_ROWS", "ONCE", "REPLICATE", "ROWS",
    "ByteRuns", "ExpertSlicing", "IntermediateAxis", "Interval",
    "bank_runs", "granularity", "slice_tensor", "standard_piece_axes",
]
