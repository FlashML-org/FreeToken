"""Windows-safe on-demand rows for native NVFP4 expert banks."""

from __future__ import annotations

import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DiskTierSpec:
    ram_experts: int


def _header_offsets(path: str) -> dict[str, tuple[int, int]]:
    with open(path, "rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    base = 8 + header_size
    return {
        name: (base + meta["data_offsets"][0], base + meta["data_offsets"][1])
        for name, meta in header.items() if name != "__metadata__"
    }


class Nvfp4DiskIndex:
    """Map (bank, layer, expert) to raw safetensors byte ranges."""

    _CANONICAL_BANKS = (
        (("gate_proj", "weight"), ("up_proj", "weight")),
        (("gate_proj", "weight_scale"), ("up_proj", "weight_scale")),
        (("gate_proj", "weight_scale_2"), ("up_proj", "weight_scale_2")),
        (("down_proj", "weight"),),
        (("down_proj", "weight_scale"),),
        (("down_proj", "weight_scale_2"),),
    )

    def __init__(self, model_dir: str, config, spec) -> None:
        from freetoken.models.loader import safetensors_weight_map
        from freetoken.utils import download_hf_weight

        folder = download_hf_weight(model_dir)
        weight_map = safetensors_weight_map(folder)
        wanted = {}
        self.global_reciprocal = bool(getattr(spec, "global_reciprocal", False))
        kind_map = getattr(spec, "kind_map", None) or {}
        banks = tuple(
            tuple((proj, next((raw for raw, canonical in kind_map.items() if canonical == kind), kind)) for proj, kind in segments)
            for segments in self._CANONICAL_BANKS
        )
        for name, shard in weight_map.items():
            match = spec.key_pattern.match(name)
            if match is None:
                continue
            layer = spec.layer_to_bank(int(match.group("layer")), config)
            if layer is not None:
                wanted[(layer, int(match.group("expert")), match.group("proj"),
                       match.group("kind"))] = (name, shard)
        shard_names = sorted({shard for _, shard in wanted.values()})
        self.paths = [os.path.join(folder, shard) for shard in shard_names]
        offsets = {shard: _header_offsets(os.path.join(folder, shard)) for shard in shard_names}
        shard_ids = {shard: index for index, shard in enumerate(shard_names)}
        self.rows = {}
        for layer in range(int(config.num_moe_layers)):
            for expert in range(int(config.num_experts)):
                for bank, segments in enumerate(banks):
                    values = []
                    for proj, kind in segments:
                        name, shard = wanted[(layer, expert, proj, kind)]
                        start, end = offsets[shard][name]
                        values.append((shard_ids[shard], start, end - start))
                    self.rows[(bank, layer, expert)] = tuple(values)

    def row_segments(self, bank: int, layer: int, expert: int):
        return self.rows[(bank, layer, expert)]


class DiskTier:
    """Synchronously fetch tail rows into their already-assigned GPU slots."""

    def __init__(self, index: Nvfp4DiskIndex, cache, ram_experts: int, workers: int = 8):
        self.index = index
        self.cache = cache
        self.ram_experts = ram_experts
        self.workers = workers

    def _read(self, shard: int, offset: int, size: int) -> bytes:
        with open(self.index.paths[shard], "rb") as handle:
            handle.seek(offset)
            data = handle.read(size)
        if len(data) != size:
            raise OSError(f"short NVFP4 disk-tier read: {len(data)} != {size}")
        return data

    def fetch(self, layer: int, expert: int, slot: int) -> None:
        for bank, (_sources, gpu_cache) in enumerate(self.cache.banks):
            segments = self.index.row_segments(bank, layer, expert)
            with ThreadPoolExecutor(max_workers=min(self.workers, len(segments))) as pool:
                pieces = list(pool.map(lambda item: self._read(*item), segments))
            row = gpu_cache[slot]
            if bank in (2, 5):
                value = struct.unpack("<f", pieces[0][:4])[0]
                if self.index.global_reciprocal:
                    value = 1.0 / value
                row.fill_(value)
                continue
            split = row.shape[0] // 2 if len(pieces) == 2 else 0
            for segment, data in enumerate(pieces):
                target = row[:split] if segment == 0 and split else row[split:]
                source = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                target.view(torch.uint8).copy_(source.view_as(target.view(torch.uint8)))