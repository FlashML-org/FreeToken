"""QuantConfig: reads one quantization_config dialect and answers what scheme each module has."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from freetoken.utils import init_logger

from ..embedding import EmbeddingConfig
from ..linear import LinearConfig
from ..method import KernelSelectionError
from ..moe import MoEConfig
from ..names import Matcher, NameMap, name_set
from ..quant_backend import get_quant_backend
from ..registry import LayerKind, dialects, method_class, methods_for, online_kinds, register_dialect
from ..scheme import QuantKind, QuantScheme

logger = init_logger(__name__)

LAYER_CONFIGS = {LayerKind.LINEAR: LinearConfig, LayerKind.MOE: MoEConfig, LayerKind.EMBEDDING: EmbeddingConfig}


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


@dataclass(frozen=True)
class Stored:
    """One checkpoint tensor behind a role: its suffix, other suffixes exports of the same dialect use for it, and whether it holds the quant-side scale whose reciprocal the layer wants."""

    name: str
    reciprocal: bool = False
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


class QuantConfig(ABC):
    dialect: ClassVar[str]
    # per kind the dialect exports, role -> the checkpoint tensor suffix (or Stored) that feeds it
    STORAGE: ClassVar[dict[QuantKind, dict[str, str | Stored]]]
    # --online-quant: the kind served for the layers this checkpoint stores bf16; --online-quant-modules narrows them (glob patterns on checkpoint names)
    online: QuantKind | None = None
    online_modules: Matcher | None = None

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

    def stored_tensors(self, kind: QuantKind) -> dict[str, Stored]:
        """role -> checkpoint tensor for every role the dialect stores for ``kind``."""
        return {role: entry if isinstance(entry, Stored) else Stored(entry) for role, entry in self.STORAGE[kind].items()}

    def storage(self, scheme: QuantScheme) -> dict[str, Stored]:
        """role -> checkpoint tensor for one scheme's tensors."""
        return {role: entry for role, entry in self.stored_tensors(scheme.kind).items() if scheme.has(role)}

    def get_quant_method(self, layer: Any, prefix: str):
        scheme = self.scheme_for(prefix)
        layer_kind = layer.quant_layer_kind
        kind = scheme.kind if scheme else QuantKind.NONE
        if layer_kind is LayerKind.EMBEDDING and kind not in methods_for(layer_kind):
            # a checkpoint that quantizes its linears stores the vocab table dense
            scheme, kind = None, QuantKind.NONE
        cfg = self.layer_config(layer, layer_kind, scheme)
        if kind is QuantKind.NONE and (method := self._online_method(layer, layer_kind, prefix, cfg)) is not None:
            return method
        cls = method_class(kind, layer_kind)
        return cls(cfg, get_quant_backend().select(layer_kind, kind))

    def with_online(self, kind: QuantKind | None, modules: tuple[str, ...] = ()) -> "QuantConfig":
        """A copy that serves the layers this checkpoint stores bf16 as ``kind`` (only those ``modules`` names, when given)."""
        if kind is not None and kind not in online_kinds():
            raise ValueError(f"{kind} has no online method; known: {sorted(str(k) for k in online_kinds())}")
        quant = copy.copy(self)
        quant.online = kind
        quant.online_modules = name_set(tuple(modules)) if kind is not None and modules else None
        return quant

    def _online_method(self, layer: Any, layer_kind: LayerKind, prefix: str, cfg: Any):
        if self.online is None or self.online not in methods_for(layer_kind):
            return None
        if self.online_modules is not None:
            names = self.name_map.to_checkpoint(prefix)
            hits = {self.online_modules(n) for n in names}
            if len(hits) > 1:
                raise ValueError(f"--online-quant-modules covers only part of the fused module {prefix!r}: {names}")
            if not hits.pop():
                return None
        cls = method_class(self.online, layer_kind)
        reason = cls.refuses(layer)
        if reason is None:
            try:
                return cls(cfg, get_quant_backend().select(layer_kind, self.online))
            except KernelSelectionError as exc:
                reason = str(exc)
        logger.info("--online-quant %s leaves %s as stored: %s", self.online, prefix, reason)
        return None

    def layer_config(self, layer: Any, layer_kind: LayerKind, scheme: QuantScheme | None) -> LinearConfig | MoEConfig:
        """The layer's Linear / MoE config; a dialect overrides it to add what only its checkpoint knows."""
        return LAYER_CONFIGS[layer_kind].from_layer(layer, scheme)

    @staticmethod
    def from_hf(
        hf_config: Any,
        *,
        name_map: NameMap | None = None,
        unquantized: tuple[str, ...] = (),
    ) -> "QuantConfig":
        """``unquantized`` lists the modules the family keeps bf16 when the checkpoint's config does not (DeepSeek-V4's compressors).

        ``hf_config`` comes from ``cached_load_hf_config``, which folds an old ModelOpt ``hf_quant_config.json`` into ``quantization_config``."""
        q = quantization_config_of(hf_config)
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
    STORAGE: ClassVar[dict[QuantKind, dict[str, str | Stored]]] = {}

    @classmethod
    def claims(cls, q: dict[str, Any]) -> bool:
        return False

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        return None
