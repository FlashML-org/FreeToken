"""V4.1 conditional memory, with demand-paged FP8/FP4 tables and request-local hashes."""

from __future__ import annotations

import json
import math
import os
import struct
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def compressed_token_map(tokenizer) -> np.ndarray:
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalize = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(),
        normalizers.Lowercase(), normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel), normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    keys = {}
    result = np.empty(len(tokenizer), dtype=np.int64)
    backend = tokenizer.backend_tokenizer
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        key = backend.id_to_token(token_id) if "\ufffd" in text else normalize.normalize_str(text) or text
        if key not in keys:
            keys[key] = len(keys)
        result[token_id] = keys[key]
    return result


@dataclass(frozen=True)
class HashLayout:
    layer_ids: tuple[int, ...]
    max_ngram_size: int
    heads: int
    primes: np.ndarray
    offsets: np.ndarray
    multipliers: np.ndarray

    @classmethod
    def from_args(cls, args):
        from sympy import nextprime

        layers = tuple(args.engram_layer_ids)
        max_ngram = args.engram_max_ngram_size
        heads = args.engram_n_heads
        if not layers or max_ngram < 2 or heads < 1 or args.engram_compressed_vocab_size < 1:
            raise ValueError("invalid Engram hash geometry")
        primes, used = [], set()
        for _ in layers:
            per_layer = []
            for _ in range(max_ngram - 1):
                current = args.engram_vocab_size - 1
                for _ in range(heads):
                    current = int(nextprime(current))
                    while current in used:
                        current = int(nextprime(current))
                    used.add(current)
                    per_layer.append(current)
            primes.append(per_layer)
        primes = np.asarray(primes, dtype=np.int64)
        if tuple(primes.sum(1)) != tuple(args.engram_num_embeddings):
            raise ValueError("Engram hash buckets do not match the checkpoint table sizes")
        offsets = np.cumsum(np.concatenate((np.zeros((len(layers), 1), dtype=np.int64), primes[:, :-1]), 1), 1)
        bound = max(1, (np.iinfo(np.int64).max // args.engram_compressed_vocab_size) // 2)
        multipliers = np.stack([
            np.random.default_rng(10007 * layer).integers(0, bound, size=max_ngram, dtype=np.int64) * 2 + 1
            for layer in layers
        ])
        return cls(layers, max_ngram, heads, primes, offsets, multipliers)


def hash_token_run(ids: np.ndarray, image_mask: np.ndarray, token_map: np.ndarray,
                   layout: HashLayout, pad_id: int, prefix_length: int = 0) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    image_mask = np.asarray(image_mask, dtype=bool)
    if ids.ndim != 1 or image_mask.shape != ids.shape or not 0 <= prefix_length <= len(ids):
        raise ValueError("invalid Engram token run")
    ids = np.where(image_mask, pad_id, ids)
    if np.any(ids < 0) or np.any(ids >= len(token_map)):
        raise ValueError("Engram input token is outside the tokenizer vocabulary")
    compressed = token_map[ids]
    positions = np.arange(prefix_length, len(ids))
    blocked = np.zeros(len(positions), dtype=bool)
    lookbacks = []
    for shift in range(layout.max_ngram_size):
        source = positions - shift
        safe = source.clip(0)
        blocked |= (source < 0) | image_mask[safe]
        lookbacks.append(np.where(blocked, token_map[pad_id], compressed[safe]))
    tokens = np.stack(lookbacks, -1)
    products = tokens[:, None, :] * layout.multipliers[None, :, :]
    rolling = products[..., 0]
    hashes = []
    for shift in range(1, layout.max_ngram_size):
        rolling = np.bitwise_xor(rolling, products[..., shift])
        start = (shift - 1) * layout.heads
        hashes.append(rolling[..., None] % layout.primes[None, :, start:start + layout.heads])
    return np.concatenate(hashes, -1) + layout.offsets[None, :, :]


def _header(path: str):
    with open(path, "rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors header: {path}")
        size = struct.unpack("<Q", prefix)[0]
        if size > 64 << 20:
            raise ValueError(f"oversized safetensors header: {path}")
        raw = stream.read(size)
    if len(raw) != size:
        raise ValueError(f"truncated safetensors header: {path}")
    return json.loads(raw), 8 + size


class DiskEngramTable:
    def __init__(self, weights: np.ndarray, scales: np.ndarray, *, dtype="fp8",
                 block_size=32, scale_fmt="ue8m0"):
        if dtype not in ("fp8", "fp4") or block_size != 32 or scale_fmt != "ue8m0":
            raise ValueError("Engram requires FP8 or packed FP4 with block-32 UE8M0 scales")
        if weights.dtype != np.uint8 or scales.dtype != np.uint8:
            raise ValueError("Engram mapped weights and scales must use raw uint8 storage")
        head_dim = weights.shape[1] * (2 if dtype == "fp4" else 1) if weights.ndim == 2 else 0
        if (weights.ndim != 2 or head_dim == 0 or head_dim % 32
                or scales.shape != (weights.shape[0], head_dim // 32)):
            raise ValueError("Engram table requires one E8M0 scale per 32 weight channels")
        self.weights = weights
        self.scales = scales
        self.num_rows, self.head_dim = weights.shape[0], head_dim
        self.dtype, self.block_size, self.scale_fmt = dtype, block_size, scale_fmt

    @classmethod
    def from_checkpoint(cls, folder: str, layer_id: int, *, args=None):
        manifest_path = os.path.join(folder, "engram_tables.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as stream:
                manifest = json.load(stream)
            if manifest.get("version") not in (1, 2):
                raise ValueError("Unsupported Engram table manifest version")
            info = manifest["layers"][str(layer_id)]
            format_info = ({"dtype": "fp8", "block_size": 32, "scale_fmt": "ue8m0"}
                           if manifest["version"] == 1 else
                           {name: info[name] for name in ("dtype", "block_size", "scale_fmt")})
            arrays = []
            for kind in ("weight", "scale"):
                entry = info[kind]
                path = os.path.abspath(os.path.join(folder, entry["file"]))
                if os.path.commonpath((os.path.abspath(folder), path)) != os.path.abspath(folder):
                    raise ValueError("Engram table path escapes the checkpoint")
                arrays.append(cls._map(path, entry["shape"], entry["offset"]))
            table = cls(*arrays, **format_info)
            table._validate_config(args)
            return table
        with open(os.path.join(folder, "model.safetensors.index.json"), encoding="utf-8") as stream:
            index = json.load(stream)["weight_map"]
        arrays = []
        dtype = None
        for kind in ("weight", "scale"):
            key = f"layers.{layer_id}.engram.embed.{kind}"
            if key not in index:
                raise ValueError(f"missing Engram tensor {key}")
            path = os.path.abspath(os.path.join(folder, index[key]))
            if os.path.commonpath((os.path.abspath(folder), path)) != os.path.abspath(folder):
                raise ValueError("Engram tensor path escapes the checkpoint")
            header, base = _header(path)
            entry = header[key]
            if kind == "weight":
                dtype = {"F8_E4M3": "fp8", "U8": "fp4"}.get(entry["dtype"])
                if dtype is None:
                    raise ValueError(f"{key} must have dtype F8_E4M3 or U8")
            elif entry["dtype"] != "F8_E8M0":
                raise ValueError(f"{key} must have dtype F8_E8M0")
            begin, end = entry["data_offsets"]
            if end - begin != math.prod(entry["shape"]):
                raise ValueError(f"invalid byte extent for {key}")
            arrays.append(cls._map(path, entry["shape"], base + begin))
        table = cls(*arrays, dtype=dtype)
        table._validate_config(args)
        return table

    def _validate_config(self, args):
        if args is not None and (self.dtype, self.block_size, self.scale_fmt) != (
                getattr(args, "engram_dtype", "fp8"), getattr(args, "engram_block_size", 32),
                getattr(args, "engram_scale_fmt", "ue8m0")):
            raise ValueError("Engram table format disagrees with model configuration")

    @staticmethod
    def _map(path, shape, offset):
        if len(shape) != 2 or min(shape) < 1 or offset < 0 or offset + math.prod(shape) > os.path.getsize(path):
            raise ValueError(f"invalid Engram table extent: {path}")
        return np.memmap(path, mode="r", dtype=np.uint8, offset=offset, shape=tuple(shape))

    def lookup(self, row_ids: np.ndarray) -> torch.Tensor:
        ids = np.asarray(row_ids, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= self.num_rows):
            raise ValueError("Engram row index outside table")
        unique, inverse = np.unique(ids, return_inverse=True)
        weights = torch.from_numpy(np.array(self.weights[unique], copy=True))
        if self.dtype == "fp4":
            codes = torch.stack((weights & 15, weights >> 4), -1).flatten(-2).long()
            values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                                   -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                                  dtype=torch.float32, device="cpu")
            weights = values[codes]
        else:
            weights = weights.view(torch.float8_e4m3fn).float()
        scales = torch.from_numpy(np.array(self.scales[unique], copy=True)).view(torch.float8_e8m0fnu).float()
        rows = (weights.reshape(-1, self.head_dim // 32, 32) * scales[..., None]).flatten(-2).to(torch.bfloat16)
        return rows[torch.from_numpy(inverse)].reshape(*ids.shape, self.head_dim)


class Engram(nn.Module):
    def __init__(self, args, layer_id: int):
        super().__init__()
        from .layers import Linear

        self.args = args
        self.layer_id = layer_id
        self.layer_hash_index = tuple(args.engram_layer_ids).index(layer_id)
        self.dim, self.hc_mult, self.eps = args.dim, args.hc_mult, args.norm_eps
        self.hash_cols = (args.engram_max_ngram_size - 1) * args.engram_n_heads
        self.wkv = Linear(self.hash_cols * args.engram_head_dim, args.dim * (args.hc_mult + 1))
        self.q_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim, dtype=torch.bfloat16), requires_grad=False)
        self.k_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim, dtype=torch.bfloat16), requires_grad=False)
        self._values = None
        self._mask = None

    def forward(self, x: torch.Tensor, hash_ids=None, token_mask=None):
        if self._values is None:
            raise RuntimeError("Engram tables were not attached before inference")
        count = x.shape[0] * x.shape[1]
        values = self._values[:count].reshape(*x.shape[:2], -1)
        kv = self.wkv(values)
        key, value = kv.split([self.hc_mult * self.dim, self.dim], -1)
        key = key.float().reshape(*x.shape[:2], self.hc_mult, self.dim)
        h = x.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * self.q_weight.float() * self.k_weight.float() * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        mask = token_mask if token_mask is not None else self._mask[:count].reshape(*x.shape[:2])
        gate = gate.masked_fill(~mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


def _image_flags(req, start: int, length: int) -> np.ndarray:
    flags = np.zeros(length, dtype=bool)
    spans = [span for item in getattr(req, "mm_items", None) or () for span in item.offsets]
    for media in getattr(req, "media", None) or ():
        spans.append((int(media["start"]), int(media["start"]) + len(media["types"])))
    for lo, hi in spans:
        left = max(start, lo)
        right = min(start + length, hi)
        if left < right:
            flags[left-start:right-start] = True
    return flags


class EngramRuntime:
    def __init__(self, args, modules, tables, token_map, max_tokens, device, *, dummy=False):
        self.args, self.modules, self.tables = args, modules, tables
        self.token_map = token_map
        self.layout = None if dummy else HashLayout.from_args(args)
        self.dummy = dummy
        self.device = torch.device(device)
        self.capacity = max_tokens
        self.host = []
        self.host_mask = torch.ones(max_tokens, dtype=torch.bool, pin_memory=self.device.type == "cuda")
        self.device_mask = self.host_mask.to(self.device)
        self._copy_done = torch.cuda.Event() if self.device.type == "cuda" else None
        self._has_copy = False
        for module in modules:
            width = module.hash_cols * args.engram_head_dim
            host = torch.zeros((max_tokens, width), dtype=torch.bfloat16, pin_memory=self.device.type == "cuda")
            self.host.append(host)
            module._values = host.to(self.device)
            module._mask = self.device_mask

    @property
    def pinned_bytes(self):
        return sum(x.numel() * x.element_size() for x in self.host) + self.host_mask.numel()

    @contextmanager
    def forward_host_ctx(self, batch, use_graph):
        if self._has_copy:
            self._copy_done.synchronize()
        runs, masks = [], []
        current = batch.input_ids.detach().to("cpu", dtype=torch.int64).numpy().reshape(-1)
        offset = 0
        for req in batch.padded_reqs:
            count = 1 if batch.is_decode else req.extend_len
            start = req.device_len - 1 if batch.is_decode else req.cached_len
            prefix_start = max(0, start - (self.args.engram_max_ngram_size - 1))
            prefix = req.input_ids[prefix_start:start].to(dtype=torch.int64).numpy()
            if len(prefix) != start - prefix_start:
                raise RuntimeError("Engram history is not available for the active request")
            ids = np.concatenate((prefix, current[offset:offset+count]))
            image_flags = _image_flags(req, prefix_start, len(ids))
            if not self.dummy:
                pad_id = getattr(self.args, "engram_pad_id", 2)
                runs.append(hash_token_run(ids, image_flags, self.token_map, self.layout, pad_id, len(prefix)))
            masks.append(~image_flags[len(prefix):])
            offset += count
        if offset != len(current) or offset > self.capacity:
            raise ValueError(f"Engram staging requires {offset} rows; capacity is {self.capacity}")
        live = np.concatenate(masks) if masks else np.empty(0, dtype=bool)
        self.host_mask[:offset].copy_(torch.from_numpy(live))
        all_hashes = np.concatenate(runs) if runs else None
        for index, (module, host) in enumerate(zip(self.modules, self.host)):
            host[:offset].zero_()
            if all_hashes is not None and live.any():
                rows = self.tables[index].lookup(all_hashes[live, index])
                host[:offset][torch.from_numpy(live)] = rows.flatten(-2)
            module._values[:offset].copy_(host[:offset], non_blocking=True)
        self.device_mask[:offset].copy_(self.host_mask[:offset], non_blocking=True)
        if self._copy_done is not None:
            self._copy_done.record()
            self._has_copy = True
        yield


def prepare_engram(model, engine_config) -> int:
    args = model._args
    modules = [layer.engram for layer in model._transformer.layers if layer.engram is not None]
    if not modules:
        return 0
    dummy = getattr(engine_config, "use_dummy_weight", False)
    tables, mapping = [], None
    if not dummy:
        from freetoken.utils import download_hf_weight, load_tokenizer

        folder = download_hf_weight(engine_config.model_path)
        mapping = compressed_token_map(load_tokenizer(folder))
        if int(mapping.max()) + 1 != args.engram_compressed_vocab_size:
            raise ValueError("Engram tokenizer normalization does not match the checkpoint compressed vocabulary")
        tables = [DiskEngramTable.from_checkpoint(folder, module.layer_id, args=args) for module in modules]
        for index, table in enumerate(tables):
            if table.num_rows != args.engram_num_embeddings[index] or table.head_dim != args.engram_head_dim:
                raise ValueError("Engram table shape disagrees with model configuration")
    prefill_capacity = getattr(engine_config, "max_extend_tokens", None)
    if prefill_capacity is None:
        prefill_capacity = min(engine_config.max_forward_len, 8192)
    runtime = EngramRuntime(
        args, modules, tables, mapping,
        max(prefill_capacity, engine_config.max_running_req, engine_config.cuda_graph_max_bs or 0, 1),
        engine_config.device if hasattr(engine_config, "device") else next(model._transformer.parameters()).device,
        dummy=dummy,
    )
    model._engram_runtime = runtime
    return runtime.pinned_bytes


def export_engram_tables(folder: str, out_dir: str, args) -> list[str]:
    tables_dir = os.path.join(out_dir, "engram")
    os.makedirs(tables_dir, exist_ok=True)
    manifest = {"version": 2, "layers": {}}
    copied = []
    for layer in args.engram_layer_ids:
        table = DiskEngramTable.from_checkpoint(folder, layer, args=args)
        entries = {"dtype": table.dtype, "block_size": table.block_size, "scale_fmt": table.scale_fmt}
        for kind, source in (("weight", table.weights), ("scale", table.scales)):
            relative = f"engram/layer-{layer}-{kind}.bin"
            destination = os.path.join(out_dir, relative)
            if os.path.realpath(source.filename) == os.path.realpath(destination):
                raise ValueError("Engram source and destination must differ")
            temporary = destination + ".partial"
            remaining = source.nbytes
            with open(source.filename, "rb") as reader, open(temporary, "wb") as writer:
                reader.seek(source.offset)
                while remaining:
                    chunk = reader.read(min(16 << 20, remaining))
                    if not chunk:
                        raise OSError("Engram source was truncated during FTW conversion")
                    writer.write(chunk)
                    remaining -= len(chunk)
            os.replace(temporary, destination)
            entries[kind] = {"file": relative, "offset": 0, "shape": list(source.shape)}
            copied.append(relative)
        manifest["layers"][str(layer)] = entries
    path = os.path.join(out_dir, "engram_tables.json")
    with open(path + ".partial", "w", encoding="utf-8") as writer:
        json.dump(manifest, writer, indent=2)
    os.replace(path + ".partial", path)
    return copied + ["engram_tables.json"]
