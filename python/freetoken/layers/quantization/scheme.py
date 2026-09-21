"""QuantScheme: the storage description and role set of one quantized weight, plus the per-kind builders."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import torch

# -1 selects the whole dimension: (-1, -1) one scale per tensor, (1, -1) one per output row, (1, 16) one per 16 input elements, (128, 128) a 2-D block
GroupShape = tuple[int, int]

class QuantKind(Enum):
    """The quantization kinds; one Method per (kind, layer kind)."""

    NONE = "none"
    FP8_TENSOR = "fp8_tensor"
    FP8_BLOCK = "fp8_block"
    MXFP8 = "mxfp8"
    NVFP4 = "nvfp4"
    MXFP4 = "mxfp4"

    def __str__(self) -> str:
        return self.value


FP8_BLOCK = 128
NVFP4_GROUP = 16
MX_GROUP = 32


@dataclass(frozen=True)
class WeightDesc:
    elem: str
    group: GroupShape | None
    scale: str | None




@dataclass(frozen=True, eq=False)
class QuantScheme:
    """``roles`` are the canonical tensor names the kind's Method declares; which checkpoint tensors feed them is the dialect Config's business."""

    kind: QuantKind
    weight: WeightDesc
    roles: frozenset[str]

    def __init__(self, kind: QuantKind, weight: WeightDesc, roles: Iterable[str]):
        if not isinstance(kind, QuantKind):
            raise TypeError(f"kind must be a QuantKind, got {kind!r}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "roles", frozenset(roles))

    def _key(self):
        return (self.kind, self.weight, self.roles)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, QuantScheme) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def has(self, role: str) -> bool:
        return role in self.roles

    def __repr__(self) -> str:
        return f"QuantScheme({self.kind}, {self.weight}, roles={sorted(self.roles)})"


# ---------------------------------------------------------------------------
# Builders: one per kind; the dialect configs pick the variant their checkpoints carry
# ---------------------------------------------------------------------------


def fp8_tensor_scheme(scale: str, *, per_row: bool = False, input_scale: bool = False) -> QuantScheme:
    roles = {"weight", "weight_scale"} | ({"input_scale"} if input_scale else set())
    return QuantScheme(QuantKind.FP8_TENSOR, WeightDesc("e4m3", (1, -1) if per_row else (-1, -1), scale), roles)


def fp8_block_scheme(scale: str) -> QuantScheme:
    return QuantScheme(QuantKind.FP8_BLOCK, WeightDesc("e4m3", (FP8_BLOCK, FP8_BLOCK), scale), {"weight", "weight_scale_inv"})


def mxfp8_scheme() -> QuantScheme:
    return QuantScheme(QuantKind.MXFP8, WeightDesc("e4m3", (1, MX_GROUP), "e8m0"), {"weight", "weight_scale_inv"})


def nvfp4_scheme(*, input_scale: bool) -> QuantScheme:
    roles = {"weight", "weight_scale", "weight_global"} | ({"input_scale"} if input_scale else set())
    return QuantScheme(QuantKind.NVFP4, WeightDesc("e2m1", (1, NVFP4_GROUP), "e4m3"), roles)


def mxfp4_scheme() -> QuantScheme:
    return QuantScheme(QuantKind.MXFP4, WeightDesc("e2m1", (1, MX_GROUP), "e8m0"), {"weight", "weight_scale"})
