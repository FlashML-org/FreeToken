"""Weight loading for GLM-5.2 (``glm_moe_dsa``).

Resident (non routed-expert) weights stream through verbatim in the checkpoint's precision (bf16). The
router selection bias is remapped ``mlp.gate.e_score_correction_bias ->
mlp.e_score_correction_bias``; the DSA indexer tensors load bf16 on "full" indexer
layers (serving runs faithful DSA top-k sparse attention; see attention.py); only the
trailing MTP layer is skipped. Routed experts are NVFP4
and go to the offload cache via the shared glm4_moe loader (identical key layout).
"""

from __future__ import annotations

import json
import os
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.glm4_moe.weight import (
    nvfp4_expert_spec,
)
from freetoken.models.loader import drop_page_cache
from freetoken.utils import cached_load_hf_config, download_hf_weight
from tqdm import tqdm

from .config import parse_config

# fp8-e4m3 dynamic range for the per-row W8A16 quantization of the big MLA projections.
class _ShardReader:
    def __init__(self, folder: str, weight_map: dict, device: torch.device):
        self._folder = folder
        self._weight_map = weight_map
        self._device = device
        self._handles: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self._weight_map

    def get(self, name: str) -> torch.Tensor:
        shard = self._weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safetensors.safe_open(
                os.path.join(self._folder, shard), framework="pt", device=str(self._device)
            ).__enter__()
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        for shard, handle in self._handles.items():
            try:
                handle.__exit__(None, None, None)
            except Exception:  # pragma: no cover - best effort
                pass
            drop_page_cache(os.path.join(self._folder, shard))
        self._handles.clear()


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    assert not include_moe_experts, (
        "GLM-5.2 stores routed experts as NVFP4 and only supports the offload backend; "
        "experts are loaded into the offload cache from their NVFP4 pieces."
    )
    assert include_non_moe
    config = parse_config(cached_load_hf_config(model_path))
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    reader = _ShardReader(folder, weight_map, device)
    primary = get_tp_info().is_primary()
    dense = config.first_k_dense_replace
    try:
        for layer in tqdm(
            range(config.num_layers),
            desc="Loading GLM-5.2 dense weights",
            disable=not primary,
        ):
            a = f"model.layers.{layer}.self_attn"
            for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
                yield f"{a}.{proj}.weight", reader.get(f"{a}.{proj}.weight")
            for norm in ("q_a_layernorm", "kv_a_layernorm"):
                yield f"{a}.{norm}.weight", reader.get(f"{a}.{norm}.weight")
            # DSA lightning indexer ("full" layers only; "shared" layers reuse their
            # group leader's selection and ship no indexer tensors). Always bf16.
            idx_types = config.glm_dsa_args.indexer_types
            if idx_types and idx_types[layer] == "full":
                for proj in ("wq_b", "wk", "weights_proj"):
                    yield f"{a}.indexer.{proj}.weight", reader.get(f"{a}.indexer.{proj}.weight")
                yield f"{a}.indexer.k_norm.weight", reader.get(f"{a}.indexer.k_norm.weight")
                yield f"{a}.indexer.k_norm.bias", reader.get(f"{a}.indexer.k_norm.bias")
            for norm in ("input_layernorm", "post_attention_layernorm"):
                yield (
                    f"model.layers.{layer}.{norm}.weight",
                    reader.get(f"model.layers.{layer}.{norm}.weight"),
                )

            m = f"model.layers.{layer}.mlp"

            if layer < dense:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    yield f"{m}.{proj}.weight", reader.get(f"{m}.{proj}.weight")
            else:
                yield f"{m}.gate.weight", reader.get(f"{m}.gate.weight")
                yield (
                    f"{m}.e_score_correction_bias",
                    reader.get(f"{m}.gate.e_score_correction_bias").to(torch.bfloat16),
                )
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    yield f"{m}.shared_experts.{proj}.weight", reader.get(f"{m}.shared_experts.{proj}.weight")

        yield "model.embed_tokens.weight", reader.get("model.embed_tokens.weight")
        yield "model.norm.weight", reader.get("model.norm.weight")
        yield "lm_head.weight", reader.get("lm_head.weight")
    finally:
        reader.close()


__all__ = ["iter_weights", "nvfp4_expert_spec"]
