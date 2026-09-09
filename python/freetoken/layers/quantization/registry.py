"""Two registries: checkpoint dialects, and (kind, layer kind) -> Method."""

from __future__ import annotations

from enum import Enum
from typing import Any

from .scheme import QuantKind


class LayerKind(Enum):
    LINEAR = "linear"
    MOE = "moe"

    def __str__(self) -> str:
        return self.value


_DIALECTS: dict[str, type] = {}
_METHODS: dict[tuple[QuantKind, LayerKind], type] = {}


def register_dialect(cls: type) -> type:
    name = cls.dialect
    if name in _DIALECTS:
        raise ValueError(f"quant dialect {name!r} already registered: {_DIALECTS[name].__name__}")
    _DIALECTS[name] = cls
    return cls


def dialects() -> tuple[type, ...]:
    return tuple(_DIALECTS.values())


def register_method(kind: QuantKind, layer_kind: LayerKind):
    if not isinstance(kind, QuantKind) or not isinstance(layer_kind, LayerKind):
        raise TypeError(f"cannot register quant method for {layer_kind!r}.{kind!r}")

    def deco(cls: type) -> type:
        key = (kind, layer_kind)
        if key in _METHODS:
            raise ValueError(f"quant method for {layer_kind}.{kind} already registered: {_METHODS[key].__name__}")
        cls.kind = kind
        cls.layer_kind = layer_kind
        _METHODS[key] = cls
        return cls

    return deco


def method_class(kind: QuantKind, layer_kind: LayerKind) -> type[Any]:
    try:
        return _METHODS[(kind, layer_kind)]
    except KeyError:
        raise NotImplementedError(f"no quant method for {layer_kind}.{kind}") from None


def methods_for(layer_kind: LayerKind) -> dict[QuantKind, type]:
    return {kind: cls for (kind, lk), cls in _METHODS.items() if lk == layer_kind}
