"""The --quant-backend value: the kernel name wanted for each table, ``layer=name`` for every kind of a layer type, ``layer.kind=name`` for one."""

from __future__ import annotations

from dataclasses import dataclass

from .registry import LayerKind, method_class, methods_for
from .scheme import QuantKind

Table = tuple[LayerKind, "QuantKind | None"]


def _kernel_names(layer_kind: LayerKind, kind: QuantKind | None) -> set[str]:
    if kind is not None:
        try:
            return {c.name for c in method_class(kind, layer_kind).candidates}
        except NotImplementedError:
            return set()
    return {c.name for cls in methods_for(layer_kind).values() for c in cls.candidates}


def _parse_table(key: str) -> Table | None:
    layer, _, kind = key.partition(".")
    try:
        return LayerKind(layer.strip()), QuantKind(kind.strip()) if kind else None
    except ValueError:
        return None


def _table_text(table: Table) -> str:
    layer, kind = table
    return f"{layer}.{kind}" if kind is not None else str(layer)


@dataclass(frozen=True)
class QuantBackend:
    items: tuple[tuple[Table, str], ...] = ()

    @classmethod
    def parse(cls, text: str | None) -> "QuantBackend":
        items: dict[Table, str] = {}
        for raw in (text or "").split(","):
            item = raw.strip()
            if not item:
                continue
            key, sep, name = item.partition("=")
            table = _parse_table(key)
            name = name.strip().lower()
            if not sep or not name or table is None:
                layers = [l.value for l in LayerKind]
                raise ValueError(f"bad --quant-backend item {item!r}; expected layer[.kind]=name with layer in {layers}")
            known = _kernel_names(*table)
            if name not in known:
                raise ValueError(f"--quant-backend {key}: no kernel {name!r}; known: {sorted(known)}")
            items[table] = name
        return cls(tuple(sorted(items.items(), key=lambda kv: _table_text(kv[0]))))

    def select(self, layer_kind: LayerKind, kind: QuantKind) -> str:
        """The kernel name for one table: its own entry, else the layer-wide entry when the table lists that kernel, else ``"auto"``."""
        entries = dict(self.items)
        exact = entries.get((layer_kind, kind))
        if exact:
            return exact
        broad = entries.get((layer_kind, None))
        if broad and broad in _kernel_names(layer_kind, kind):
            return broad
        return "auto"


_QUANT_BACKEND = QuantBackend()


def set_quant_backend(backend: QuantBackend) -> None:
    """Install the kernel requests every layer built afterwards consults; the engine sets them once before create_model."""
    global _QUANT_BACKEND
    _QUANT_BACKEND = backend


def get_quant_backend() -> QuantBackend:
    return _QUANT_BACKEND
