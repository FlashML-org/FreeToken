"""Replace Gemma 4 embedding and router tensors with original BF16 data in a GGUF."""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

import gguf
import safetensors
import torch

HEADER = struct.Struct("<4sIQQ")
ALIGNMENT = 32
GGML_BF16 = 30


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def tensor_info_end(handle, count: int) -> tuple[int, list[tuple[bytes, int, int, int]]]:
    records = []
    for _ in range(count):
        record_start = handle.tell()
        name_len = struct.unpack("<Q", handle.read(8))[0]
        name = handle.read(name_len)
        n_dims = struct.unpack("<I", handle.read(4))[0]
        dims = handle.read(8 * n_dims)
        type_pos = handle.tell()
        tensor_type = struct.unpack("<I", handle.read(4))[0]
        offset_pos = handle.tell()
        offset = struct.unpack("<Q", handle.read(8))[0]
        records.append((name, record_start, type_pos, offset_pos))
    return handle.tell(), records


def load_replacements(source_dir: Path, names: list[str]) -> dict[str, bytes]:
    index = json.loads((source_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    replacements: dict[str, bytes] = {}
    for name in names:
        source_name = name
        if source_name not in index:
            raise KeyError(f"source checkpoint lacks {source_name}")
        with safetensors.safe_open(source_dir / index[source_name], framework="pt", device="cpu") as shard:
            tensor = shard.get_tensor(source_name)
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{source_name} is {tensor.dtype}, expected BF16")
        replacements[name] = tensor.contiguous().view(torch.uint16).numpy().tobytes()
    return replacements


def rewrite(source: Path, hf_source: Path, output: Path) -> None:
    reader = gguf.GGUFReader(str(source))
    index = json.loads(
        (hf_source / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    source_names = set(index)
    replacements_names: set[str] = set()
    for tensor in reader.tensors:
        name = tensor.name
        source_name = name
        if name.startswith("model.language_model."):
            source_name = name
        elif name.startswith("model."):
            source_name = "model.language_model." + name[len("model."):]
        if source_name in source_names:
            with safetensors.safe_open(
                hf_source / index[source_name], framework="pt", device="cpu"
            ) as shard:
                if shard.get_tensor(source_name).dtype == torch.bfloat16:
                    replacements_names.add(name)
    replacements = load_replacements(hf_source, sorted(
        name if name.startswith("model.language_model.") else "model.language_model." + name[len("model."):]
        for name in replacements_names
    ))
    replacements = {
        name: replacements[
            name if name.startswith("model.language_model.")
            else "model.language_model." + name[len("model."):]
        ]
        for name in replacements_names
    }

    with source.open("rb") as handle:
        header = handle.read(24)
        magic, version, tensor_count, kv_count = HEADER.unpack(header)
        if magic != b"GGUF" or version != 3:
            raise ValueError("source must be GGUF v3")
        info_start = int(reader.tensors[0].field.offset)
        handle.seek(info_start)
        info_end, records = tensor_info_end(handle, tensor_count)
        old_data_start = align(info_end)
        handle.seek(0, os.SEEK_END)
        source_size = handle.tell()
        handle.seek(0)
        source_bytes = handle.read()

    info_blob = bytearray(source_bytes[info_start:info_end])
    kv_blob = source_bytes[24:info_start]
    new_records = []
    data_blob = bytearray()
    cursor = 0
    record_cursor = 0
    for tensor, (name_bytes, _start, type_pos, offset_pos) in zip(reader.tensors, records):
        name = name_bytes.decode("utf-8")
        old_data = source_bytes[tensor.data_offset : tensor.data_offset + tensor.data.nbytes]
        if name in replacements:
            data = replacements[name]
            new_type = GGML_BF16
        else:
            data = old_data
            new_type = int(tensor.tensor_type)
        cursor = align(cursor)
        data_blob += b"\0" * (cursor - len(data_blob))
        relative_offset = cursor
        data_blob += data
        record_len = 8 + len(name_bytes) + 4 + 8 * len(tensor.shape) + 4 + 8
        local_type = type_pos - info_start
        local_offset = offset_pos - info_start
        struct.pack_into("<I", info_blob, local_type, new_type)
        struct.pack_into("<Q", info_blob, local_offset, relative_offset)
        cursor += len(data)

    new_info_end = 24 + len(kv_blob) + len(info_blob)
    new_data_start = align(new_info_end)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(HEADER.pack(magic, version, tensor_count, kv_count))
        handle.write(kv_blob)
        handle.write(info_blob)
        handle.write(b"\0" * (new_data_start - handle.tell()))
        handle.write(data_blob)
    os.replace(temporary, output)

    check = gguf.GGUFReader(str(output))
    if len(check.tensors) != len(reader.tensors):
        raise AssertionError("tensor count changed")
    for tensor in check.tensors:
        expected = GGML_BF16 if tensor.name in replacements else int(next(x.tensor_type for x in reader.tensors if x.name == tensor.name))
        if int(tensor.tensor_type) != expected:
            raise AssertionError(f"unexpected type for {tensor.name}: {tensor.tensor_type}")
    print(f"wrote {output}")
    print(f"replaced {len(replacements)} tensors with BF16; preserved {len(reader.tensors) - len(replacements)} tensors")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("hf_source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rewrite(args.source, args.hf_source, args.output)


if __name__ == "__main__":
    main()
