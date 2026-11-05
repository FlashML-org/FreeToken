from __future__ import annotations

from typing import Iterator

import safetensors
import torch
from tqdm import tqdm

from freetoken.distributed import get_tp_info
from freetoken.layers.quantization import get_quant_config
from freetoken.models.loader import iter_weight_files
from freetoken.models.qwen3_5_moe.weight import _DenseReader
from freetoken.models.register import get_model_spec
from freetoken.utils import cached_load_hf_config


def iter_weights(
    model_path: str, device: torch.device, *, include_moe_experts: bool, include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    if not include_non_moe:
        return
    reader = _DenseReader(get_quant_config(), get_model_spec(cached_load_hf_config(model_path).architectures[0]))
    for file in tqdm(iter_weight_files(model_path), desc="Loading weights", disable=not get_tp_info().is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if name.endswith(".value_res_lambda"):
                    yield name.replace(".value_res_lambda", ".value_res_blend"), torch.sigmoid(tensor.float()).to(tensor.dtype)
                    continue
                emitted = reader.add(name, tensor)
                if emitted is None:
                    yield name, tensor
                else:
                    yield from emitted
    if reader.pending:
        raise ValueError(f"{model_path}: incomplete projections: {reader.missing()}")
