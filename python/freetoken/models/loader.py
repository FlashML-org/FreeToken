from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, Iterator

import torch
from freetoken.utils import div_ceil, download_hf_weight

SPLIT_DIM_0 = (".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj")
SPLIT_DIM_1 = (".o_proj", ".down_proj")


@dataclass(frozen=True)
class MergeRule:
    fused_suffix: str
    slot: str
    slots: tuple[str, ...]


def shard_tensor(
    key: str,
    value: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    num_kv_heads: int,
) -> torch.Tensor:
    if any(key.count(sub) for sub in SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < world_size:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = rank * num_kv_heads // world_size
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(world_size, dim=0)[rank].clone()
    if any(key.count(sub) for sub in SPLIT_DIM_1):
        return value.chunk(world_size, dim=1)[rank].clone()
    if key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, world_size)
        vocab_start_idx = rank * num_embeddings_per_partition
        vocab_end_idx = min((rank + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    return value


def iter_weight_files(model_path: str) -> list[str]:
    model_folder = download_hf_weight(model_path)
    files = glob.glob(f"{model_folder}/*.safetensors")
    return [f for f in files if not f.endswith("consolidated.safetensors")] or files


def drop_page_cache(path: str) -> None:
    """drop a file's page cache: banks + full checkpoint cache don't both fit in host RAM (OOM)."""
    posix_fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if posix_fadvise is None or dontneed is None:
        # Windows has no POSIX page-cache advisory API.  The hint is optional;
        # skipping it is correct and lets the serial expert loader run there.
        return
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            posix_fadvise(fd, 0, 0, dontneed)
        finally:
            os.close(fd)
    except OSError:
        pass


def iter_root_safetensor_files_from_index(
    model_path: str,
    *,
    index_file: str = "model.safetensors.index.json",
) -> list[str]:
    model_folder = download_hf_weight(model_path)
    if os.path.basename(os.path.normpath(model_folder)) in {"metal", "original"}:
        raise ValueError("GPT-OSS loading requires the root GPT-OSS model directory")

    root_files = sorted(glob.glob(os.path.join(model_folder, "*.safetensors")))
    index_path = os.path.join(model_folder, index_file)
    if not os.path.isfile(index_path):
        files = [f for f in root_files if not f.endswith("consolidated.safetensors")]
        if not files:
            raise ValueError("No root GPT-OSS safetensors shards found")
        return files

    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    indexed_files = []
    for filename in dict.fromkeys(weight_map.values()):
        if os.path.dirname(filename):
            continue
        path = os.path.join(model_folder, filename)
        if path in root_files:
            indexed_files.append(path)

    if not indexed_files:
        raise ValueError(
            "No root GPT-OSS safetensors shards found from model.safetensors.index.json"
        )
    return sorted(indexed_files)


def _merge_info(key: str, rules: dict[str, MergeRule]) -> tuple[str, MergeRule] | None:
    for suffix, rule in rules.items():
        if key.endswith(suffix + ".weight") or key.endswith(suffix) or key.count(suffix):
            return key.replace(suffix, rule.fused_suffix), rule
    return None


def iter_merged_tensors(
    tensors: Iterable[tuple[str, torch.Tensor]],
    rules: dict[str, MergeRule],
    *,
    model_name: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    merge_buf: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in tensors:
        info = _merge_info(name, rules)
        if info is None:
            yield name, tensor
            continue
        merged_key, rule = info
        slots = merge_buf.setdefault(merged_key, {})
        slots[rule.slot] = tensor
        if not all(slot in slots for slot in rule.slots):
            continue
        parts = [slots[slot] for slot in rule.slots]
        del merge_buf[merged_key]
        yield merged_key, torch.cat(parts, dim=0)

    assert not merge_buf, (
        f"{model_name}: Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    )


# ---------------------------------------------------------------------------------
# compressed-tensors NVFP4 (llm-compressor) dense-weight helpers, shared by the
# models that serve such checkpoints natively (qwen3_5_moe, muse_glimmer). Storage:
# ``weight_packed`` (uint8 [O, IN//2]) + ``weight_scale`` (fp8-e4m3 block [O, IN//16])
# + a scalar ``weight_global_scale``. The stored global is the *quant-side* scale, so
# the dequant/native global is its reciprocal (vLLM inverts it identically).
# ---------------------------------------------------------------------------------

# Quant scales consumed with their ``weight_packed`` (or unused: the input scales are
# for W4A4 activation quant, which FreeToken does not run).
CT_SCALE_SUFFIXES = (
    ".weight_scale", ".weight_global_scale", ".input_global_scale", ".input_scale",
)


class ShardReader:
    """Serves tensors by name across safetensors shards (handles opened lazily).

    The quant scales of a ``weight_packed`` can land in a DIFFERENT shard than the
    packed weight itself (Muse-Glimmer-30B-NVFP4 splits layer 49's down_proj across
    the shard boundary), so sibling lookups must go through the index's weight_map
    rather than the shard the packed weight came from."""

    def __init__(self, model_path: str, device: torch.device):
        folder = download_hf_weight(model_path)
        index = os.path.join(folder, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index, encoding="utf-8") as f:
                weight_map = json.load(f)["weight_map"]
            self._map = {
                name: os.path.join(folder, shard) for name, shard in weight_map.items()
            }
        else:  # single-file checkpoint
            import safetensors

            self._map = {}
            for file in iter_weight_files(model_path):
                with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                    for name in f.keys():
                        self._map[name] = file
        self._device = str(device)
        self._handles: dict[str, object] = {}

    def files(self) -> list[str]:
        return sorted(set(self._map.values()))

    def names_in(self, file: str) -> list[str]:
        return [name for name, shard in self._map.items() if shard == file]

    def get_tensor(self, name: str) -> torch.Tensor:
        import safetensors

        file = self._map[name]
        h = self._handles.get(file)
        if h is None:
            h = safetensors.safe_open(file, framework="pt", device=self._device).__enter__()
            self._handles[file] = h
        return h.get_tensor(name)

    def close(self) -> None:
        for h in self._handles.values():
            try:
                h.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 -- best-effort handle cleanup
                pass
        self._handles.clear()


def nvfp4_parts_ct(f, raw_base: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """compressed-tensors NVFP4 -> ``(packed uint8 [O, IN//2], block scale fp8 [O, IN//16],
    per-output-row global fp16 [O])`` for the W4A16 kernels."""
    w = f.get_tensor(raw_base + ".weight_packed")
    s = f.get_tensor(raw_base + ".weight_scale")
    wg = f.get_tensor(raw_base + ".weight_global_scale").reshape(1).to(torch.float32)
    g = (1.0 / wg).to(torch.float16).expand(w.shape[0]).contiguous()
    return w, s, g


def ct_nvfp4_fuse(base: str, parts_tuple: tuple, buf: dict, groups: dict[str, tuple[str, ...]]):
    """Buffer a native NVFP4 fusion part ``(w, s, g)``; emit the concatenated native parts
    (``.weight``/``.weight_scale``/``.weight_global``, output-dim concat with each part
    keeping its own scales, so the fused FP4 weight is exact) once complete, ``[]`` while
    incomplete, ``None`` if ``base`` is not a fusion part of any group in ``groups``."""
    for fused_suffix, parts in groups.items():
        for idx, part in enumerate(parts):
            if base.endswith(part):
                key = base[: -len(part)] + fused_suffix
                slots = buf.setdefault(key, {})
                slots[idx] = parts_tuple
                if len(slots) < len(parts):
                    return []
                del buf[key]
                ws = [slots[i][0] for i in range(len(parts))]
                ss = [slots[i][1] for i in range(len(parts))]
                gs = [slots[i][2] for i in range(len(parts))]
                return [
                    (key + ".weight", torch.cat(ws, dim=0)),
                    (key + ".weight_scale", torch.cat(ss, dim=0)),
                    (key + ".weight_global", torch.cat(gs, dim=0)),
                ]
    return None


def ct_bf16_fuse(base: str, tensor: torch.Tensor, buf: dict, groups: dict[str, tuple[str, ...]]):
    """Buffer a bf16 fusion part; emit the concatenated ``.weight`` once complete, ``[]``
    while incomplete, ``None`` if ``base`` is not a part of any group in ``groups``."""
    for fused_suffix, parts in groups.items():
        for idx, part in enumerate(parts):
            if base.endswith(part):
                key = base[: -len(part)] + fused_suffix
                slots = buf.setdefault(key, {})
                slots[idx] = tensor
                if len(slots) < len(parts):
                    return []
                del buf[key]
                return [(key + ".weight", torch.cat([slots[i] for i in range(len(parts))], dim=0))]
    return None


def _expert_stack_info(key: str, expert_pattern: re.Pattern[str]) -> tuple[str, int] | None:
    match = expert_pattern.match(key)
    if match is None:
        return None
    packed_name = match.group("name")
    if packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


def iter_stacked_experts(
    tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    num_experts: int,
    model_name: str,
    expert_pattern: re.Pattern[str],
) -> Iterator[tuple[str, torch.Tensor]]:
    expert_buf: dict[str, dict[int, torch.Tensor]] = {}
    for name, tensor in tensors:
        expert_info = _expert_stack_info(name, expert_pattern)
        if expert_info is None:
            yield name, tensor
            continue
        packed_key, expert_idx = expert_info
        slots = expert_buf.setdefault(packed_key, {})
        slots[expert_idx] = tensor
        if len(slots) != num_experts:
            continue
        experts = [slots[idx] for idx in range(num_experts)]
        del expert_buf[packed_key]
        yield packed_key, torch.stack(experts, dim=0)

    assert not expert_buf, (
        f"{model_name}: Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"
    )


def _packed_expert_source_info(key: str) -> tuple[int, str] | None:
    parts = key.split(".")
    if len(parts) < 5 or parts[0] != "model" or parts[1] != "layers":
        return None
    if parts[-2] != "experts" or parts[-1] not in {"gate_up_proj", "down_proj"}:
        return None
    try:
        return int(parts[2]), parts[-1]
    except ValueError:
        return None


class _PlainBank:
    """CPU-only fallback bank: a plain unpinned tensor with a no-op pin (no CUDA)."""

    __slots__ = ("tensor",)

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def pin(self) -> None:
        pass


def _alloc_expert_bank(shape: tuple[int, ...], *, dtype: torch.dtype):
    """Allocate an UNPINNED bank (lazy host mmap), to be filled then pinned at the end --
    pin-after-fill. Registering already-resident pages skips cudaHostAlloc's slow commit
    (~2.8 GiB/s) / zero-fill. Returns a bank object exposing ``.tensor`` and ``.pin()``."""
    if torch.cuda.is_available():
        from freetoken.moe.host_banks import HostBank

        return HostBank(tuple(shape), dtype)
    return _PlainBank(torch.empty(shape, dtype=dtype))


def _copy_expert_layer_into_bank(
    banks: dict[str, list],
    row_shape: dict[str, tuple[int, ...]],
    seen_layers: dict[str, set[int]],
    *,
    bank_name: str,
    tensor: torch.Tensor,
    layer: int,
    config,
    dtype: torch.dtype,
) -> None:
    if layer < 0 or layer >= config.num_layers:
        raise ValueError(
            f"Unexpected MoE expert layer {layer}; expected [0, {config.num_layers})"
        )
    if tensor.size(0) != config.num_experts:
        raise ValueError(
            f"Unexpected {bank_name} expert count {tensor.size(0)}; "
            f"expected {config.num_experts}"
        )
    expected_shape = row_shape.setdefault(bank_name, tuple(tensor.shape[1:]))
    if tuple(tensor.shape[1:]) != expected_shape:
        raise ValueError(
            f"Inconsistent {bank_name} expert shape {tuple(tensor.shape[1:])}; "
            f"expected {expected_shape}"
        )

    bank = banks[bank_name][layer]
    if bank is None:
        banks[bank_name][layer] = bank = _alloc_expert_bank(
            (config.num_experts, *tensor.shape[1:]), dtype=dtype
        )
    bank.tensor.copy_(tensor)  # whole-layer arrival; pinned later, after fully resident
    seen_layers[bank_name].add(layer)


def stream_moe_expert_sources(
    tensors: Iterable[tuple[str, torch.Tensor]],
    config,
    *,
    dtype: torch.dtype,
    layer_sink=None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Stream packed per-layer BF16 expert tensors into final offload banks.

    Model adapters normalize ordinary MoE expert weights to
    ``...experts.gate_up_proj`` and ``...experts.down_proj`` with shape
    ``[num_experts, ...]``. Each arrives whole-layer, so it's written directly into
    its own ``[num_experts, ...]`` per-layer bank (independent allocation).

    ``layer_sink=None`` (serving): pin each layer's banks as its writes complete,
    via an internally-owned :class:`PinPipeline`. ``layer_sink`` given (converter):
    the tracker fires into it instead -- nothing is pinned, and the sink may release
    banks it has written out, so the returned tensors are only valid until then (the
    caller owns that tradeoff).
    """
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline

    banks: dict[str, list] = {  # name -> per-layer [bank obj (HostBank/_PlainBank) or None]
        "gate_up": [None] * config.num_layers,
        "down": [None] * config.num_layers,
    }
    row_shape: dict[str, tuple[int, ...]] = {}
    seen_layers: dict[str, set[int]] = {"gate_up": set(), "down": set()}

    def _load(sink) -> None:
        tracker = LayerCompletionTracker(2, banks, sink)  # gate_up + down per layer
        for name, tensor in tensors:
            expert_info = _packed_expert_source_info(name)
            if expert_info is None:
                raise ValueError(f"Unexpected expert weight key: {name}")
            layer, packed_name = expert_info
            bank_name = "gate_up" if packed_name == "gate_up_proj" else "down"
            _copy_expert_layer_into_bank(
                banks,
                row_shape,
                seen_layers,
                bank_name=bank_name,
                tensor=tensor,
                layer=layer,
                config=config,
                dtype=dtype,
            )
            tracker.note(layer)

        expected_layers = set(range(config.num_layers))
        missing = {
            name: sorted(expected_layers - seen)
            for name, seen in seen_layers.items()
            if seen != expected_layers
        }
        if missing:
            raise ValueError(f"Missing MoE expert source layers: {missing}")

    if layer_sink is not None:
        _load(layer_sink)
    else:
        with PinPipeline() as pins:
            _load(pins)
    return (
        [bank.tensor for bank in banks["gate_up"]],
        [bank.tensor for bank in banks["down"]],
    )


__all__ = [
    "MergeRule",
    "iter_root_safetensor_files_from_index",
    "iter_weight_files",
    "iter_merged_tensors",
    "iter_stacked_experts",
    "shard_tensor",
    "stream_moe_expert_sources",
]
