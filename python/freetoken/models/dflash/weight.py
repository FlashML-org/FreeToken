"""DFlash 2 draft checkpoint reader: safetensors with HF names, q/k/v and gate/up fused as the model names them."""

from __future__ import annotations

from typing import Iterator

import safetensors
import torch
from freetoken.models.loader import MergeRule, iter_merged_tensors, iter_weight_files

_MERGE_RULES = {
    ".q_proj": MergeRule(".qkv_proj", "q", ("q", "k", "v")),
    ".k_proj": MergeRule(".qkv_proj", "k", ("q", "k", "v")),
    ".v_proj": MergeRule(".qkv_proj", "v", ("q", "k", "v")),
    ".gate_proj": MergeRule(".gate_up_proj", "gate", ("gate", "up")),
    ".up_proj": MergeRule(".gate_up_proj", "up", ("gate", "up")),
}


def iter_weights(model_path: str, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    def tensors() -> Iterator[tuple[str, torch.Tensor]]:
        for file in iter_weight_files(model_path):
            with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)

    yield from iter_merged_tensors(tensors(), _MERGE_RULES, model_name="dflash")
