"""QuantMethod: one (kind, layer kind) pair bound to one kernel object; the selector and the post-load hook."""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, Sequence

from freetoken.utils import init_logger

from .registry import LayerKind
from .scheme import QuantKind

logger = init_logger(__name__)


class KernelSelectionError(RuntimeError):
    pass


def select_kernel(candidates: Sequence[type], requested: str, cfg: Any):
    """Pick one kernel from an ordered table: the requested name if usable, else the first usable one that is worth it."""
    if not candidates:
        raise KernelSelectionError("empty kernel table")
    names = [c.name for c in candidates]
    if requested != "auto" and requested in names:
        kernel = candidates[names.index(requested)]()
        reason = kernel.unusable_reason(cfg)
        if reason:
            raise KernelSelectionError(f"kernel {requested!r} was requested but cannot run here: {reason}")
        return kernel
    skipped: list[str] = []
    fallback = None
    for cls in candidates:
        kernel = cls()
        reason = kernel.unusable_reason(cfg)
        if reason:
            skipped.append(f"{kernel.name}: {reason}")
            continue
        if kernel.worth_it(cfg):
            if skipped:
                logger.info("kernel %s selected; skipped %s", kernel.name, "; ".join(skipped))
            return kernel
        if fallback is None:
            fallback = kernel
    if fallback is not None:
        return fallback
    raise KernelSelectionError("no usable kernel in table; " + "; ".join(skipped))


class QuantMethod(ABC):
    kind: ClassVar[QuantKind]
    layer_kind: ClassVar[LayerKind]
    candidates: ClassVar[tuple[type, ...]] = ()

    def __init__(self, cfg: Any, requested: str = "auto"):
        self.cfg = cfg
        self.requested = requested
        self.kernel = select_kernel(self.candidates, requested, cfg)

    @property
    def scheme(self):
        return self.cfg.scheme

def finalize_quant(root: Any) -> int:
    """Call ``quant_method.finalize(layer)`` on every layer under ``root``; returns the count."""
    from freetoken.layers.base import BaseOP

    seen: set[int] = set()
    count = 0

    def walk(op: Any) -> None:
        nonlocal count
        if id(op) in seen:
            return
        seen.add(id(op))
        method = getattr(op, "quant_method", None)
        if method is not None:
            method.finalize(op)
            count += 1
        for value in vars(op).values():
            if isinstance(value, BaseOP):
                walk(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, BaseOP):
                        walk(item)

    walk(root)
    return count
