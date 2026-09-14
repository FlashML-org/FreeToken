"""Native V4.1 resident weights and NVFP4 W4A16 expert banks.

The converted experts use E4M3 scales per 16 values plus a global scale.
W4A16 intentionally does not use the checkpoint's W4A4 input_scale metadata;
it is not bit-equivalent to the original activation-quantized reference.
"""

from __future__ import annotations

import json
import os
import re
import struct

import safetensors
import torch

from freetoken.layers.quantization import QuantKind
from freetoken.models.loader import drop_page_cache
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec, iter_nvfp4_expert_pieces,
)
from freetoken.utils import download_hf_weight

from .args import load_args

_EXPERT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w1|w2|w3)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_EXPERT_PREFIX_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.")
_ENGRAM_TABLE_RE = re.compile(r"^layers\.\d+\.engram\.embed\.(weight|scale)$")


def _weight_map(folder):
    with open(os.path.join(folder, "model.safetensors.index.json"), encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def read_checkpoint_headers(folder, weight_map=None):
    """Read only safetensors JSON headers, before any expert bank is allocated."""
    weight_map = _weight_map(folder) if weight_map is None else weight_map
    result = {}
    for shard in sorted(set(weight_map.values())):
        with open(os.path.join(folder, shard), "rb") as f:
            length_bytes = f.read(8)
            if len(length_bytes) != 8:
                raise ValueError(f"Truncated safetensors header: {shard}")
            length = struct.unpack("<Q", length_bytes)[0]
            if length <= 0 or length > 100_000_000:
                raise ValueError(f"Invalid safetensors header length: {shard}")
            header = json.loads(f.read(length))
        for name, item in header.items():
            if name == "__metadata__":
                continue
            if name in result:
                raise ValueError(f"Duplicate checkpoint tensor: {name}")
            if weight_map.get(name) != shard:
                raise ValueError(f"Checkpoint index/shard mismatch for {name}")
            result[name] = item
    missing = weight_map.keys() - result.keys()
    if missing:
        raise ValueError(f"Checkpoint index names missing tensor {min(missing)}")
    return result


def validate_expert_headers(headers, config):
    """Validate every expert component and geometry without materializing a tensor."""
    count = 0
    layers, experts = config.num_layers, config.num_experts
    hidden, inter = config.hidden_size, config.moe_intermediate_size
    if hidden % 16 or inter % 16:
        raise ValueError("NVFP4 expert dimensions must be divisible by 16")
    for name, item in headers.items():
        prefix = _EXPERT_PREFIX_RE.match(name)
        if prefix is None:
            continue
        match = _EXPERT_RE.fullmatch(name)
        if match is None:
            if name.endswith(".input_scale"):
                continue
            raise ValueError(f"Unrecognized V4.1 expert tensor: {name}")
        layer, expert = int(match["layer"]), int(match["expert"])
        if not (0 <= layer < layers and 0 <= expert < experts):
            raise ValueError(f"V4.1 expert outside configured backbone: {name}")
        out_dim, in_dim = (hidden, inter) if match["proj"] == "w2" else (inter, hidden)
        kind = match["kind"]
        shape = tuple(item["shape"])
        expected_shape = {"weight": (out_dim, in_dim // 2),
                          "weight_scale": (out_dim, in_dim // 16)}.get(kind)
        expected_dtype = {"weight": "U8", "weight_scale": "F8_E4M3", "weight_scale_2": "F32"}[kind]
        if item["dtype"] != expected_dtype or (
                shape != expected_shape if expected_shape is not None else shape not in ((), (1,))):
            raise ValueError(f"Malformed NVFP4 tensor {name}: {item['dtype']} {shape}; "
                             f"expected {expected_dtype} {expected_shape or 'scalar'}")
        count += 1
    expected = layers * experts * 9
    if count != expected:
        for layer in range(layers):
            for expert in range(experts):
                for proj in ("w1", "w2", "w3"):
                    for kind in ("weight", "weight_scale", "weight_scale_2"):
                        name = f"layers.{layer}.ffn.experts.{expert}.{proj}.{kind}"
                        if name not in headers:
                            raise ValueError(f"Missing NVFP4 tensor: {name}")
        raise ValueError(f"Expected {expected} NVFP4 tensors, found {count}")


def _dequant_fp8_block(weight, scale, block=32):
    rows, cols = weight.shape
    if scale.shape != (rows // block, cols // block):
        raise ValueError(f"FP8 scale shape {tuple(scale.shape)} incompatible with weight {tuple(weight.shape)}")
    values = torch.exp2(scale.view(torch.uint8).float() - 127)
    values = values.repeat_interleave(block, 0).repeat_interleave(block, 1)
    return (weight.float() * values).to(torch.bfloat16)


class _ShardReader:
    def __init__(self, folder, weight_map, device):
        self.folder, self.weight_map, self.device = folder, weight_map, str(device)
        self.handles = {}

    def get(self, name):
        if name not in self.weight_map:
            raise ValueError(f"Missing DeepSeek-V4.1 tensor: {name}")
        shard = self.weight_map[name]
        if shard not in self.handles:
            self.handles[shard] = safetensors.safe_open(os.path.join(self.folder, shard),
                                                       framework="pt", device=self.device).__enter__()
        return self.handles[shard].get_tensor(name)

    def close(self):
        for shard, handle in self.handles.items():
            handle.__exit__(None, None, None)
            drop_page_cache(os.path.join(self.folder, shard))
        self.handles.clear()


def iter_weights(model_path, device, *, include_moe_experts=True, include_non_moe=True, include_vision=True):
    if include_moe_experts:
        raise ValueError("DeepSeek-V4.1 NVFP4 experts require --moe-strategy offload")
    if not include_non_moe:
        return
    folder = download_hf_weight(model_path)
    args = load_args(folder)
    weight_map = _weight_map(folder)
    reader = _ShardReader(folder, weight_map, device)
    try:
        for name in sorted(weight_map, key=lambda name: (weight_map[name], name)):
            if name.startswith("mtp.") or _EXPERT_PREFIX_RE.match(name) or _ENGRAM_TABLE_RE.match(name):
                continue
            if (not include_vision or not args.vision_enabled) and name.startswith(("vision.", "aligner.", "image_")):
                continue
            if name.endswith(".attn.wo_a.scale"):
                continue
            if name.endswith(".attn.wo_a.weight"):
                prefix = name.removesuffix(".weight")
                yield prefix, _dequant_fp8_block(reader.get(name), reader.get(prefix + ".scale"))
            else:
                yield name, reader.get(name)
    finally:
        reader.close()


def is_expert_tensor(name):
    return _EXPERT_RE.fullmatch(name) is not None


_NVFP4_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_EXPERT_RE, proj_to_role={"w1": "gate", "w3": "up", "w2": "down"},
    layer_to_bank=lambda layer, config: layer if layer < config.num_layers else None,
    desc="DeepSeek-V4.1 NVFP4 experts",
)


def iter_expert_pieces(model_path, config, kind, *, parallel=False, workers=8, chunk=8 << 20):
    if kind is not QuantKind.NVFP4:
        return None
    folder = download_hf_weight(model_path)
    validate_expert_headers(read_checkpoint_headers(folder), config)
    return iter_nvfp4_expert_pieces(folder, config, _NVFP4_SPEC, parallel=parallel,
                                    workers=workers, chunk=chunk, drop_page_cache=drop_page_cache)


__all__ = ["iter_weights", "is_expert_tensor", "iter_expert_pieces",
           "read_checkpoint_headers", "validate_expert_headers"]
