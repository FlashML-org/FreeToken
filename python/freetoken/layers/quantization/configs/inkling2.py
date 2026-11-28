from __future__ import annotations

from typing import Any, ClassVar

from ..registry import register_dialect
from ..scheme import QuantKind, QuantScheme, mxfp8_scheme, nvfp4_scheme
from .base import QuantConfig, Stored

@register_dialect
class Inkling2QuantConfig(QuantConfig):
    """Thinking Machines' Inkling 2 exports: ``hf_quant_config.json`` names every stored tensor
    under a parameter group with its ``compute_dtype`` (config.json only carries a ModelOpt stub).

    Stacked routed experts are ``nvfp4`` (packed e2m1, e4m3 scales per 16, a per-expert global
    ``amax / outer_scale_denominator``) or ``mxfp8`` (e4m3, ue8m0 scales per 32); everything
    else is bf16 or fp32. The activation quantization the checkpoint records is not served:
    experts run W4A16 / W8A16, as the reference implementation does.
    """

    dialect = "inkling2"

    STORAGE: ClassVar[dict[QuantKind, dict[str, str | Stored]]] = {
        QuantKind.NVFP4: {"weight": "weight", "weight_scale": "weight_scale", "weight_global": "weight_amax"},
        QuantKind.MXFP8: {"weight": "weight", "weight_scale_inv": "weight_scale"},
    }

    def __init__(self, q: dict[str, Any], hf_config: Any = None, *, name_map=None, unquantized=()):
        super().__init__(name_map, unquantized)
        self.compute: dict[str, str] = {}
        for group in (q.get("model") or {}).get("parameter_groups") or ():
            for tensor in group.get("tensors") or ():
                self.compute[tensor] = str(group.get("compute_dtype"))

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        dtype = self.compute.get(f"{name}.w13_weight") or self.compute.get(f"{name}.weight")
        if dtype == "nvfp4":
            return nvfp4_scheme(input_scale=False)
        if dtype == "mxfp8":
            return mxfp8_scheme()
        return None
