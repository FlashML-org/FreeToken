from __future__ import annotations

from typing import Any, ClassVar

from ..names import is_routed_expert, name_set, substr_set
from ..registry import register_dialect
from ..scheme import QuantScheme
from ..scheme import FP8_BLOCK, fp8_block_scheme, fp8_tensor_scheme, mxfp4_scheme
from .base import QuantConfig, cfg_get


@register_dialect
class Fp8BlockConfig(QuantConfig):
    """HF ``quant_method: fp8`` (DeepSeek-V3 style 128x128 block scales) plus the DeepSeek-V4 e8m0 / fp4-expert variant."""

    dialect = "fp8"

    SCHEMES: ClassVar[dict[str, QuantScheme]] = {
        "BLOCK": fp8_block_scheme("float"),
        "BLOCK_E8M0": fp8_block_scheme("e8m0"),
        # HF ``modules_to_convert``: a table (Qwen3.8-Flash-Next PLE) stored e4m3 with one scalar scale
        "TABLE": fp8_tensor_scheme("float"),
        "EXPERT_MXFP4": mxfp4_scheme(),
    }

    def __init__(self, q: dict[str, Any], hf_config: Any = None, *, name_map=None, unquantized=()):
        super().__init__(name_map, unquantized)
        block = tuple(int(x) for x in (q.get("weight_block_size") or ()))
        if q.get("weight_per_tensor") or block != (FP8_BLOCK, FP8_BLOCK):
            raise NotImplementedError(f"fp8 checkpoint with weight_block_size={block} per_tensor={q.get('weight_per_tensor')} is not supported; only 128x128 blocks are")
        # transformers skips lm_head when the checkpoint gives no list
        not_convert = tuple(q.get("modules_to_not_convert") or ("lm_head",))
        self.not_convert = name_set(not_convert)
        self.not_convert_substr = substr_set(not_convert)
        self.convert_tables = name_set(tuple(q.get("modules_to_convert") or ()))
        self.e8m0 = str(q.get("scale_fmt") or "").lower() == "ue8m0"
        self.expert_fp4 = str(cfg_get(hf_config, "expert_dtype") or "").lower() == "fp4"

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        if self.convert_tables(name):
            return self.SCHEMES["TABLE"]
        if self.not_convert(name) or self.not_convert_substr(name):
            return None
        if self.expert_fp4 and is_routed_expert(name):
            return self.SCHEMES["EXPERT_MXFP4"]
        return self.SCHEMES["BLOCK_E8M0" if self.e8m0 else "BLOCK"]
