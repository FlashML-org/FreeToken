from __future__ import annotations

from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import iter_weight_files
from tqdm import tqdm

from freetoken.models.qwen3_5_moe.weight import (
    iter_weights_parallel,
    load_nvfp4_expert_sources,
    load_nvfp4_expert_sources_parallel,
    setup_offload_expert_banks,
)


_FUSIONS = {
    ".self_attn.qkv_proj.weight": (
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
    ),
    ".linear_attn.in_proj.weight": (
        ".linear_attn.in_proj_qkv.weight",
        ".linear_attn.in_proj_z.weight",
        ".linear_attn.in_proj_b.weight",
        ".linear_attn.in_proj_a.weight",
    ),
    ".mlp.shared_expert.gate_up_proj.weight": (
        ".mlp.shared_expert.gate_proj.weight",
        ".mlp.shared_expert.up_proj.weight",
    ),
}


def _rename(raw_name: str) -> str | None:
    if raw_name.startswith("mtp."):
        return None
    if raw_name.startswith("model.visual."):
        return "visual." + raw_name[len("model.visual.") :]
    if raw_name.startswith("visual."):
        return raw_name
    if ".ple.ple_embedding.ngram_embedding." in raw_name:
        return None
    if raw_name.startswith("model.language_model."):
        return "model." + raw_name[len("model.language_model.") :]
    if raw_name.startswith("language_model."):
        return "model." + raw_name[len("language_model.") :]
    return raw_name


def _try_fuse(name: str, tensor: torch.Tensor, buffers: dict):
    for fused_suffix, parts in _FUSIONS.items():
        for index, part in enumerate(parts):
            if name.endswith(part):
                fused_name = name[: -len(part)] + fused_suffix
                slots = buffers.setdefault(fused_name, {})
                slots[index] = tensor
                if len(slots) == len(parts):
                    del buffers[fused_name]
                    return fused_name, torch.cat([slots[i] for i in range(len(parts))], dim=0)
                return ()
    return None


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    if get_tp_info().size > 1:
        raise NotImplementedError("Qwen4-Exp currently supports TP=1 only")
    if include_moe_experts:
        raise ValueError("Qwen4-Exp requires --moe-backend offload, cpu, or hybrid")
    if not include_non_moe:
        return

    buffers = {}
    for filename in tqdm(
        iter_weight_files(model_path),
        desc="Loading Qwen4-Exp resident weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(filename, framework="pt", device=str(device)) as handle:
            for raw_name in handle.keys():
                name = _rename(raw_name)
                if (
                    name is None
                    or ".mlp.experts." in name
                    or raw_name.endswith(".weight_scale_inv")
                ):
                    continue
                tensor = handle.get_tensor(raw_name)
                fused = _try_fuse(name, tensor, buffers)
                if fused is not None:
                    if fused:
                        yield fused
                    continue
                yield name, tensor
    if buffers:
        raise RuntimeError(f"Incomplete Qwen4-Exp projection fusions: {sorted(buffers)}")


__all__ = [
    "iter_weights",
    "iter_weights_parallel",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
    "setup_offload_expert_banks",
]
