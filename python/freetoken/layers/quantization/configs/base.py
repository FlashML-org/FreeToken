"""QuantConfig: reads one quantization_config dialect and answers what scheme each module has."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ..linear import LinearConfig
from ..moe import MoEConfig
from ..names import Matcher, NameMap, name_set
from ..quant_backend import get_quant_backend
from ..registry import LayerKind, dialects, method_class, register_dialect
from ..scheme import QuantKind, QuantScheme

LAYER_CONFIGS = {LayerKind.LINEAR: LinearConfig, LayerKind.MOE: MoEConfig}


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def quantization_config_of(hf_config: Any) -> dict[str, Any] | None:
    q = cfg_get(hf_config, "quantization_config")
    if q is None:
        q = cfg_get(cfg_get(hf_config, "text_config"), "quantization_config")
    if q is None:
        return None
    if isinstance(q, dict):
        return q
    if hasattr(q, "to_dict"):
        return dict(q.to_dict())
    return dict(vars(q))


class QuantConfig(ABC):
    dialect: ClassVar[str]

    def __init__(self, name_map: NameMap | None = None, unquantized: tuple[str, ...] = ()):
        self.name_map = name_map or NameMap()
        # modules the family serves in bf16 regardless of the dialect (glob patterns on checkpoint names)
        self.unquantized: Matcher = name_set(tuple(unquantized))
        self._schemes: dict[str, QuantScheme | None] = {}

    @classmethod
    def claims(cls, q: dict[str, Any]) -> bool:
        """Whether this dialect reads the given ``quantization_config``."""
        return str(q.get("quant_method") or "").lower() == cls.dialect

    @abstractmethod
    def scheme_for_name(self, name: str) -> QuantScheme | None:
        """Scheme of one module by its checkpoint name; None means unquantized."""

    def scheme_for(self, prefix: str) -> QuantScheme | None:
        if prefix in self._schemes:
            return self._schemes[prefix]
        names = self.name_map.to_checkpoint(prefix)
        schemes = {None if self.unquantized(n) else self.scheme_for_name(n) for n in names}
        if len(schemes) != 1:
            raise ValueError(f"fused module {prefix!r} mixes quantization schemes across {names}: {schemes}")
        scheme = schemes.pop()
        self._schemes[prefix] = scheme
        return scheme

    def get_quant_method(self, layer: Any, prefix: str):
        scheme = self.scheme_for(prefix)
        layer_kind = layer.quant_layer_kind
        kind = scheme.kind if scheme else QuantKind.NONE
        cls = method_class(kind, layer_kind)
        cfg = LAYER_CONFIGS[layer_kind].from_layer(layer, scheme)
        return cls(cfg, get_quant_backend().select(layer_kind, kind))

    @staticmethod
    def from_hf(
        hf_config: Any,
        *,
        name_map: NameMap | None = None,
        unquantized: tuple[str, ...] = (),
        hf_quant_config: dict[str, Any] | None = None,
    ) -> "QuantConfig":
        """``hf_quant_config`` is the parsed ``hf_quant_config.json`` of ModelOpt exports that keep
        no ``quantization_config`` in config.json (modelopt < 0.41); ``unquantized`` lists the modules
        the family keeps bf16 when the checkpoint's config does not (DeepSeek-V4's compressors)."""
        q = quantization_config_of(hf_config)
        if q is None and hf_quant_config and isinstance(hf_quant_config.get("quantization"), dict):
            q = dict(hf_quant_config["quantization"])
            q.setdefault("quant_method", "modelopt")
        if q is None:
            return NoQuantConfig(name_map, unquantized)
        for cls in dialects():
            if cls.claims(q):
                return cls(q, hf_config, name_map=name_map, unquantized=unquantized)
        raise NotImplementedError(f"quantization method {q.get('quant_method')!r} is not supported")


@register_dialect
class NoQuantConfig(QuantConfig):
    """No ``quantization_config``: every module is bf16."""

    dialect = "none"

    @classmethod
    def claims(cls, q: dict[str, Any]) -> bool:
        return False

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        return None
