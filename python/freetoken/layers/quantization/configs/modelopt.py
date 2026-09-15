from __future__ import annotations

from typing import Any, ClassVar

from ..names import ancestors, name_set
from ..registry import register_dialect
from ..scheme import QuantKind, QuantScheme
from ..scheme import fp8_block_scheme, fp8_tensor_scheme, mxfp8_scheme, nvfp4_scheme
from .base import QuantConfig, Stored


@register_dialect
class ModelOptConfig(QuantConfig):
    """NVIDIA ModelOpt exports: one ``quant_algo`` for every Linear minus ``ignore``, or
    ``MIXED_PRECISION`` with a per-module ``quantized_layers`` allow-list."""

    dialect = "modelopt"

    SCHEMES: ClassVar[dict[str, QuantScheme]] = {
        "NVFP4": nvfp4_scheme(input_scale=True),
        "W4A16_NVFP4": nvfp4_scheme(input_scale=False),
        "FP8": fp8_tensor_scheme("fp32", input_scale=True),
        "FP8_PER_CHANNEL_PER_TOKEN": fp8_tensor_scheme("fp32", per_row=True),
        "FP8_PB_WO": fp8_block_scheme("fp32"),
        "MXFP8": mxfp8_scheme(),
    }
    STORAGE: ClassVar[dict[QuantKind, dict[str, str | Stored]]] = {
        QuantKind.FP8_TENSOR: {"weight": "weight", "weight_scale": "weight_scale", "input_scale": "input_scale"},
        QuantKind.FP8_BLOCK: {"weight": "weight", "weight_scale_inv": "weight_scale_inv"},
        QuantKind.MXFP8: {"weight": "weight", "weight_scale_inv": "weight_scale_inv"},
        QuantKind.NVFP4: {"weight": "weight", "weight_scale": "weight_scale", "weight_global": "weight_scale_2", "input_scale": "input_scale"},
    }

    @classmethod
    def claims(cls, q: dict[str, Any]) -> bool:
        method = str(q.get("quant_method") or "").lower()
        return method == "modelopt" or (not method and bool(q.get("quant_algo")))

    def __init__(self, q: dict[str, Any], hf_config: Any = None, *, name_map=None, unquantized=()):
        super().__init__(name_map, unquantized)
        self.algo = str(q.get("quant_algo") or "").upper()
        # Inferact exports carry this key; false says the FP8 and NVFP4 layers store no input_scale, and an explicit value wins over the config_groups rule below
        self.with_input_scale = q.get("with_input_scale")
        groups = q.get("config_groups")
        # an export with no activation quantizer can still say NVFP4; every config group then has input_activations null (vLLM applies the same rule)
        if self.with_input_scale is None and self.algo == "NVFP4" and isinstance(groups, dict) and groups and all(isinstance(g, dict) and g.get("input_activations") is None for g in groups.values()):
            self.algo = "W4A16_NVFP4"
        self.ignore = name_set(tuple(q.get("ignore") or q.get("exclude_modules") or ()))
        layers = q.get("quantized_layers") or {}
        self.quantized_layers = {k: str((v or {}).get("quant_algo") or "").upper() for k, v in layers.items()} if isinstance(layers, dict) else {}
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
        if self.with_input_scale is False:
            if algo == "FP8":
                return fp8_tensor_scheme("fp32")  # ModelOpt has no algo name for fp8 without an activation scale
            if algo == "NVFP4":
                algo = "W4A16_NVFP4"
        try:
            return self.SCHEMES[algo]
        except KeyError:
            raise NotImplementedError(f"ModelOpt quant_algo {algo!r} is not supported") from None
