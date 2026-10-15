from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

from ..linear import LinearConfig
from ..moe import MoEConfig
from ..names import is_routed_expert, name_set, substr_set
from ..registry import LayerKind, register_dialect
from ..scheme import QuantKind, QuantScheme
from ..scheme import FP8_BLOCK_SIZES, fp8_block_scheme, fp8_tensor_scheme, mxfp4_scheme
from .base import QuantConfig, Stored, cfg_get


@register_dialect
class Fp8BlockConfig(QuantConfig):
    """HF ``quant_method: fp8`` (DeepSeek-V3 style 128x128 block scales) plus the DeepSeek-V4 e8m0 / fp4-expert
    variant; DeepSeek-V4.1 keeps that dialect with 32x32 blocks (``weight_block_size: [32, 32]``)."""

    dialect = "fp8"

    # the 128-block schemes every dialect table names; ``block_scheme`` carries the checkpoint's actual block edge
    SCHEMES: ClassVar[dict[str, QuantScheme]] = {
        "BLOCK": fp8_block_scheme("float"),
        "BLOCK_E8M0": fp8_block_scheme("e8m0"),
        # HF ``modules_to_convert``: a table (Qwen3.8-Flash-Next PLE) stored e4m3 with one scalar scale
        "TABLE": fp8_tensor_scheme("float"),
        "EXPERT_MXFP4": mxfp4_scheme(),
    }
    # transformers' fp8 names; DeepSeek-V4's e8m0 export calls every scale ``scale`` (see storage)
    STORAGE: ClassVar[dict[QuantKind, dict[str, str | Stored]]] = {
        QuantKind.FP8_BLOCK: {"weight": "weight", "weight_scale_inv": "weight_scale_inv"},
        QuantKind.FP8_TENSOR: {"weight": "weight", "weight_scale": "weight_scale"},
        QuantKind.MXFP4: {"weight": "weight", "weight_scale": "scale"},
    }

    def __init__(self, q: dict[str, Any], hf_config: Any = None, *, name_map=None, unquantized=()):
        super().__init__(name_map, unquantized)
        block = tuple(int(x) for x in (q.get("weight_block_size") or ()))
        square = len(block) == 2 and block[0] == block[1] and block[0] in FP8_BLOCK_SIZES
        if q.get("weight_per_tensor") or not square:
            raise NotImplementedError(
                f"fp8 checkpoint with weight_block_size={block} per_tensor={q.get('weight_per_tensor')} is not supported; "
                f"only square blocks of {FP8_BLOCK_SIZES} are"
            )
        self.block = block[0]
        # transformers skips lm_head when the checkpoint gives no list
        not_convert = tuple(q.get("modules_to_not_convert") or ("lm_head",))
        self.not_convert = name_set(not_convert)
        self.not_convert_substr = substr_set(not_convert)
        self.convert_tables = name_set(tuple(q.get("modules_to_convert") or ()))
        self.e8m0 = str(q.get("scale_fmt") or "").lower() == "ue8m0"
        # ``expert_dtype`` sits at the config top level (V4) or inside quantization_config (V4.1)
        expert_dtype = q.get("expert_dtype") or cfg_get(hf_config, "expert_dtype")
        self.expert_fp4 = str(expert_dtype or "").lower() == "fp4"
        self.block_scheme = fp8_block_scheme("e8m0" if self.e8m0 else "float", self.block)

    def storage(self, scheme: QuantScheme) -> dict[str, Stored]:
        names = super().storage(scheme)
        if self.e8m0 and scheme.kind is QuantKind.FP8_BLOCK:
            names["weight_scale_inv"] = Stored("scale")
        return names

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        if self.convert_tables(name):
            return self.SCHEMES["TABLE"]
        if self.not_convert(name) or self.not_convert_substr(name):
            return None
        if self.expert_fp4 and is_routed_expert(name):
            return self.SCHEMES["EXPERT_MXFP4"]
        return self.block_scheme

    def layer_config(self, layer: Any, layer_kind: LayerKind, scheme: QuantScheme | None) -> LinearConfig | MoEConfig:
        cfg = super().layer_config(layer, layer_kind, scheme)
        if layer_kind is LayerKind.MOE and scheme is not None and scheme.kind is QuantKind.MXFP4:
            # the fp4 experts quantize their activations at the same block as the dense linears
            cfg = dataclasses.replace(cfg, act_block=self.block)
        return cfg
