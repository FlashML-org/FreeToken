"""GGUF: every tensor carries its own ggml type, read from the file's tensor table."""

from __future__ import annotations

from typing import Any, Callable, ClassVar

from ..ggml_types import is_quantized, type_name
from ..names import NameMap
from ..quant_backend import get_quant_backend
from ..registry import LayerKind, method_class, register_dialect
from ..scheme import QuantKind, QuantScheme, gguf_scheme
from .base import QuantConfig, Stored


@register_dialect
class GGUFConfig(QuantConfig):
    """``tensor_types`` is the file's name -> ggml type table; ``to_gguf(name, tensor_types)`` (the family's adapter) maps a checkpoint module name to the GGUF tensors that store it."""

    dialect = "gguf"
    STORAGE: ClassVar[dict[QuantKind, dict[str, str | Stored]]] = {QuantKind.GGUF: {"weight": "weight"}}

    def __init__(
        self,
        tensor_types: dict[str, int],
        to_gguf: Callable[[str, dict[str, int]], tuple[str, ...]],
        name_map: NameMap | None = None,
        unquantized: tuple[str, ...] = (),
    ):
        super().__init__(name_map, unquantized)
        self.tensor_types = tensor_types
        self.to_gguf = to_gguf

    @classmethod
    def claims(cls, q: dict[str, Any]) -> bool:
        return False

    def _part_types(self, name: str) -> tuple[int, ...]:
        types = []
        for tensor in self.to_gguf(name, self.tensor_types):
            key = f"{tensor}.weight" if f"{tensor}.weight" in self.tensor_types else tensor
            if key not in self.tensor_types:
                raise KeyError(f"GGUF has no tensor {key!r} for {name!r}")
            types.append(self.tensor_types[key])
        return tuple(types)

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        types = self._part_types(name)
        if not any(is_quantized(t) for t in types):
            return None
        if len(types) != 1:
            raise NotImplementedError(f"{name} is stored as {len(types)} quantized GGUF tensors")
        return gguf_scheme(types)

    def scheme_for(self, prefix: str) -> QuantScheme | None:
        if prefix in self._schemes:
            return self._schemes[prefix]
        names = self.name_map.to_checkpoint(prefix)
        parts = [() if self.unquantized(n) else self._part_types(n) for n in names]
        quantized = [any(is_quantized(t) for t in p) for p in parts]
        if not any(quantized):
            scheme = None
        elif any(len(p) != 1 for p in parts):
            found = {n: [type_name(t) for t in p] for n, p in zip(names, parts)}
            raise NotImplementedError(f"fused module {prefix!r} needs one GGUF tensor per part, got {found}")
        else:
            scheme = gguf_scheme(p[0] for p in parts)
        self._schemes[prefix] = scheme
        return scheme

    def get_quant_method(self, layer: Any, prefix: str):
        if layer.quant_layer_kind is not LayerKind.MOE:
            return super().get_quant_method(layer, prefix)
        # every expert layer loads its GGUF banks, float experts included; the GGUF MoE kernel picks how to multiply them
        parts = [self._part_types(n) for n in self.name_map.to_checkpoint(prefix)]
        if any(len(p) != 1 for p in parts):
            raise NotImplementedError(f"experts {prefix!r} are stored as {parts} GGUF tensors per projection")
        scheme = gguf_scheme(p[0] for p in parts)
        cfg = self.layer_config(layer, LayerKind.MOE, scheme)
        return method_class(QuantKind.GGUF, LayerKind.MOE)(cfg, get_quant_backend().select(LayerKind.MOE, QuantKind.GGUF))


__all__ = ["GGUFConfig"]
