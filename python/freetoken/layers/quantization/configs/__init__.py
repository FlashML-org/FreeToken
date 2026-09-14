"""One QuantConfig per checkpoint dialect; importing the package registers them all."""

from .base import NoQuantConfig, QuantConfig, quantization_config_of
from . import compressed_tensors, fp8, modelopt, mxfp4
from .compressed_tensors import CompressedTensorsConfig
from .fp8 import Fp8BlockConfig
from .modelopt import ModelOptConfig
from .mxfp4 import Mxfp4Config

_NO_QUANT = NoQuantConfig()


def quant_method_for(quant_config, layer, prefix: str):
    """The layer's Method from its config; a layer built without one is plain bf16."""
    return (quant_config if quant_config is not None else _NO_QUANT).get_quant_method(layer, prefix)


_UNSET = object()
_QUANT_CONFIG = _UNSET


def set_quant_config(quant: QuantConfig | None) -> None:
    """Install the checkpoint's QuantConfig for the weight readers, which only get the model path; EngineConfig sets it once."""
    global _QUANT_CONFIG
    _QUANT_CONFIG = quant


def get_quant_config() -> QuantConfig | None:
    if _QUANT_CONFIG is _UNSET:
        raise RuntimeError("no QuantConfig installed: EngineConfig.model_config sets it before the weights load")
    return _QUANT_CONFIG


__all__ = [
    "QuantConfig", "quantization_config_of", "quant_method_for", "set_quant_config", "get_quant_config",
    "NoQuantConfig", "ModelOptConfig", "CompressedTensorsConfig", "Fp8BlockConfig", "Mxfp4Config",
    "compressed_tensors", "fp8", "modelopt", "mxfp4",
]
