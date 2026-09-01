from __future__ import annotations

from typing import Iterator
import torch
from freetoken.models.loader import iter_weight_files
from freetoken.utils import nvtx_annotate


def iter_dflash_weights(
    model_path: str,
    device: torch.device,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (name, tensor) for DFlash draft model weights.

    The DFlash checkpoint is a single safetensors file with keys like:
    - fc.weight
    - hidden_norm.weight
    - layers.{0..5}.input_layernorm.weight
    - layers.{0..5}.self_attn.{q,k,v,o}_proj.weight
    - layers.{0..5}.self_attn.{q,k}_norm.weight
    - layers.{0..5}.mlp.{gate,up,down}_proj.weight
    - layers.{0..5}.post_attention_layernorm.weight
    - norm.weight
    """
    import safetensors

    files = iter_weight_files(model_path)
    for file in files:
        f = safetensors.safe_open(file, framework="pt", device=str(device))
        for key in f.keys():
            yield key, f.get_tensor(key)


__all__ = ["iter_dflash_weights"]
