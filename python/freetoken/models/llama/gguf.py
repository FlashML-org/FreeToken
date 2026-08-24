""" Llama GGUF adapter: build the FreeToken ``ModelConfig`` from GGUF metadata.
"""
from torch import Tensor
from typing import Iterator
from freetoken.models.config import ModelConfig

def parse_gguf_config(shim:GGUFConfigShim) -> ModelConfig:
    """Parse a GGUF config shim into a FreeToken ``ModelConfig``."""

    # The GGUF shim is a minimal HF-config-like dict with the fields we need.
    return ModelConfig(
        model_type=shim.model_type,
        architectures=shim.architectures,
        vocab_size=shim.vocab_size,
        tie_word_embeddings=shim.tie_word_embeddings,
        torch_dtype="bfloat16",  # GGUF weights dequantize to bf16
    )

def iter_gguf_weights(
        model_path: str,
        device,
        *,
        include_moe_experts: bool,
        include_non_moe: bool
        ) -> Iterator[tuple[str, Tensor]]:
    """Iterate over GGUF weights, yielding (name, tensor) pairs for the model's parameters."""
    pass
    
    

def convert_llama_to_gguf(model, config) -> None:
    """Convert a FreeToken Llama model to GGUF format in-place.

    This is a no-op for non-GGUF models, and raises an error if the model is not a Llama.
    """
    pass

# ...existing code...
def is_gguf_model(config) -> bool:
    """Check if the model config is for a GGUF model."""
    return getattr(config, "moe_weight_format", None) == "q4_0"
# ...existing code...