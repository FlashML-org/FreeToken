from __future__ import annotations

from typing import Any, ClassVar

from ..names import name_set
from ..registry import register_dialect
from ..scheme import QuantScheme
from ..scheme import mxfp4_scheme
from .base import QuantConfig


@register_dialect
class Mxfp4Config(QuantConfig):
    """gpt-oss: the stacked ``experts`` containers are MXFP4 with biases, everything else bf16."""

    dialect = "mxfp4"

    SCHEME: ClassVar[QuantScheme] = mxfp4_scheme()

    def __init__(self, q: dict[str, Any], hf_config: Any = None, *, name_map=None, unquantized=()):
        super().__init__(name_map, unquantized)
        self.not_convert = name_set(tuple(q.get("modules_to_not_convert") or ()))

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        if not name.endswith(".experts") or self.not_convert(name):
            return None
        return self.SCHEME
