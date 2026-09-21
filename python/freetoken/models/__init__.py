from .blocks import BaseLLMModel
from .config import (
    AttentionGroupConfig,
    BaseAttentionGroupConfig,
    DSV4AttentionGroupConfig,
    FullAttentionGroupConfig,
    KVCacheGroupSpec,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
)
from .register import get_model_class
from .weight import load_weight


def create_model(model_config: ModelConfig) -> BaseLLMModel:
    return get_model_class(model_config.architectures[0], model_config)


__all__ = [
    "BaseLLMModel",
    "create_model",
    "load_weight",
    "AttentionGroupConfig",
    "BaseAttentionGroupConfig",
    "DSV4AttentionGroupConfig",
    "FullAttentionGroupConfig",
    "LinearGatedDeltaGroupConfig",
    "ModelConfig",
    "RotaryConfig",
    "SWAAttentionGroupConfig",
    "KVCacheGroupSpec",
]
