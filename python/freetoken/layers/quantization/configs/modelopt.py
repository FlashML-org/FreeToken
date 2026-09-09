from __future__ import annotations

from typing import Any, ClassVar

from ..names import ancestors, name_set
from ..registry import register_dialect
from ..scheme import QuantScheme
from ..scheme import fp8_block_scheme, fp8_tensor_scheme, mxfp8_scheme, nvfp4_scheme
from .base import QuantConfig


@register_dialect
class ModelOptConfig(QuantConfig):
    """NVIDIA ModelOpt exports: one ``quant_algo`` for every Linear minus ``ignore``, or
    ``MIXED_PRECISION`` with a per-module ``quantized_layers`` allow-list."""

    dialect = "modelopt"

    SCHEMES: ClassVar[dict[str, QuantScheme]] = {
        "NVFP4": nvfp4_scheme(input_scale=True),
        "NVFP4_NO_INPUT": nvfp4_scheme(input_scale=False),
        "FP8": fp8_tensor_scheme("fp32", input_scale=True),
        "FP8_PER_CHANNEL_PER_TOKEN": fp8_tensor_scheme("fp32", per_row=True),
        "FP8_PB_WO": fp8_block_scheme("fp32"),
        "MXFP8": mxfp8_scheme(),
    }

    @classmethod
    def claims(cls, q: dict[str, Any]) -> bool:
        method = str(q.get("quant_method") or "").lower()
        return method == "modelopt" or (not method and bool(q.get("quant_algo")))

    def __init__(self, q: dict[str, Any], hf_config: Any = None, *, name_map=None, unquantized=()):
        super().__init__(name_map, unquantized)
        self.algo = str(q.get("quant_algo") or "").upper()
        self.ignore = name_set(tuple(q.get("ignore") or q.get("exclude_modules") or ()))
        layers = q.get("quantized_layers") or {}
        self.quantized_layers = {k: str((v or {}).get("quant_algo") or "").upper() for k, v in layers.items()} if isinstance(layers, dict) else {}
        self.with_input_scale = bool(q.get("with_input_scale", True))
        if self.algo == "MIXED_PRECISION" and not self.quantized_layers:
            raise NotImplementedError("ModelOpt MIXED_PRECISION without quantized_layers in quantization_config")
        if self.algo != "MIXED_PRECISION":
            self._scheme_of(self.algo)

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        if self.ignore(name):
            return None
        algo = self._module_algo(name)
        return None if algo is None else self._scheme_of(algo)

    def _module_algo(self, name: str) -> str | None:
        if self.algo != "MIXED_PRECISION":
            return self.algo
        for a in ancestors(name):
            algo = self.quantized_layers.get(a)
            if algo is not None:
                return algo
        return None

    def _scheme_of(self, algo: str) -> QuantScheme:
        if algo in ("NVFP4", "W4A16_NVFP4"):
            return self.SCHEMES["NVFP4" if self.with_input_scale else "NVFP4_NO_INPUT"]
        try:
            return self.SCHEMES[algo]
        except KeyError:
            raise NotImplementedError(f"ModelOpt quant_algo {algo!r} is not supported") from None
