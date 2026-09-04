"""GGUF config shim: the object the model registry sees for a ``.gguf`` model.

``cached_load_hf_config`` returns one of these for GGUF paths instead of a HF
``PretrainedConfig``. It carries the architecture key (so the registry can dispatch),
the raw GGUF metadata dict, and a few derived facts that need the tensor table
(``vocab_size``, ``tie_word_embeddings``). The per-arch ``parse_gguf_config`` reads
``metadata`` to build the FreeToken ``ModelConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .reader import gguf_architecture, load_gguf_metadata, gguf_tensor_names

# GGUF ``general.architecture`` -> FreeToken registry key (a GGUF-specific spec that
# reuses the model classes but a GGUF parse_config / iter_weights). The key is the
# value of the GGUF general.architecture metadata key.
GGUF_ARCH_TO_REGISTRY: dict[str, str] = {
    "gemma4": "Gemma4GGUFForCausalLM",
    "qwen35moe": "Qwen35MoeGGUFForCausalLM",
    # Dense sibling (Qwen3.8-27B): same hybrid GDN/full-attention decoder, a plain SwiGLU
    # MLP instead of routed experts. Same model classes and the same GGUF adapter; the
    # config's expert_count is absent so moe_enabled comes out False.
    "qwen35": "Qwen35GGUFForCausalLM",
    "qwen3moe": "Qwen3MoeGGUFForCausalLM",
    "deepseek4": "DeepseekV4GGUFForCausalLM",
}


@dataclass(frozen=True)
class GgufConfigShim:
    architectures: list[str]
    model_path: str
    model_type: str
    metadata: dict[str, Any]
    vocab_size: int
    tie_word_embeddings: bool

    def to_dict(self) -> dict[str, Any]:
        """Minimal HF-config-like dict for trunk code that introspects the config
        (e.g. server arg parsing reads ``torch_dtype`` to resolve ``--dtype auto``).
        GGUF weights dequantize to a bf16 compute path."""
        return {
            "architectures": list(self.architectures),
            "model_type": self.model_type,
            "torch_dtype": "bfloat16",
            "vocab_size": self.vocab_size,
            "tie_word_embeddings": self.tie_word_embeddings,
        }


def _vocab_size(model_path: str) -> int:
    from .reader import _reader

    for t in _reader(model_path).tensors:
        if t.name == "token_embd.weight":
            return int(t.shape[-1])  # ggml [hidden, vocab] -> vocab is last
    # A metadata-only GGUF (an FTW dir's source_metadata.gguf) strips the tensor table, so
    # fall back to the tokenizer vocab. llama.cpp sizes token_embd's rows to n_vocab =
    # len(tokenizer.ggml.tokens), so this equals the tensor-derived value exactly.
    toks = load_gguf_metadata(model_path).get("tokenizer.ggml.tokens")
    if toks is not None:
        return len(toks)
    raise ValueError(f"GGUF {model_path}: no token_embd.weight to size the vocab")


def build_gguf_shim(model_path: str) -> GgufConfigShim:
    arch = gguf_architecture(model_path)
    registry_key = GGUF_ARCH_TO_REGISTRY.get(arch)
    if registry_key is None:
        raise ValueError(
            f"GGUF architecture {arch!r} is not supported "
            f"(known: {sorted(GGUF_ARCH_TO_REGISTRY)})"
        )
    names = gguf_tensor_names(model_path)
    metadata = load_gguf_metadata(model_path)
    if names:
        # No separate output projection -> embeddings are tied.
        tie_word_embeddings = "output.weight" not in names
    else:
        # Metadata-only GGUF (an FTW dir's source_metadata.gguf): the tensor table is
        # stripped, so the fact travels as a KV written at convert time.
        from .reader import OUTPUT_WEIGHT_PRESENT_KV

        present = metadata.get(OUTPUT_WEIGHT_PRESENT_KV)
        if present is None:
            raise ValueError(
                f"{model_path}: metadata-only GGUF lacks {OUTPUT_WEIGHT_PRESENT_KV!r}; "
                "reconvert the checkpoint with the current freetoken.checkpoint.convert"
            )
        tie_word_embeddings = not present
    return GgufConfigShim(
        architectures=[registry_key],
        model_path=model_path,
        model_type=arch,
        metadata=metadata,
        vocab_size=_vocab_size(model_path),
        tie_word_embeddings=tie_word_embeddings,
    )


__all__ = ["GgufConfigShim", "GGUF_ARCH_TO_REGISTRY", "build_gguf_shim"]
