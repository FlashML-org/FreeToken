"""Add generation sampling defaults to an existing GGUF without touching tensors."""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

import gguf

HEADER = struct.Struct("<4sIQQ")
ALIGNMENT = 32


def encode_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def encode_kv(key: str, value, value_type: int) -> bytes:
    raw_key = key.encode("utf-8")
    if value_type == int(gguf.GGUFValueType.FLOAT32):
        payload = struct.pack("<f", float(value))
    elif value_type == int(gguf.GGUFValueType.UINT32):
        payload = struct.pack("<I", int(value))
    else:
        raise ValueError(f"unsupported value type {value_type}")
    return struct.pack("<Q", len(raw_key)) + raw_key + struct.pack("<I", value_type) + payload


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def tensor_info_end(source_bytes: bytes, start: int, count: int) -> int:
    cursor = start
    for _ in range(count):
        name_len = struct.unpack_from("<Q", source_bytes, cursor)[0]
        cursor += 8 + name_len
        n_dims = struct.unpack_from("<I", source_bytes, cursor)[0]
        cursor += 4 + 8 * n_dims + 4 + 8
    return cursor


def patch(source: Path, output: Path) -> None:
    reader = gguf.GGUFReader(str(source))
    if not reader.tensors:
        raise ValueError("source GGUF has no tensors")
    kv_end = int(reader.tensors[0].field.offset)
    with source.open("rb") as handle:
        source_bytes = handle.read()
    magic, version, tensor_count, kv_count = HEADER.unpack(source_bytes[:24])
    extra = b"".join(
        (
            encode_kv("general.sampling.temp", 1.0, int(gguf.GGUFValueType.FLOAT32)),
            encode_kv("general.sampling.top_k", 64, int(gguf.GGUFValueType.UINT32)),
            encode_kv("general.sampling.top_p", 0.95, int(gguf.GGUFValueType.FLOAT32)),
        )
    )
    old_data_start = reader.tensors[0].data_offset
    old_info_end = tensor_info_end(source_bytes, kv_end, tensor_count)
    old_data = source_bytes[old_data_start:]
    kv = source_bytes[24:kv_end] + extra
    new_info_start = 24 + len(kv)
    info_size = old_info_end - kv_end
    tensor_infos = source_bytes[kv_end:old_info_end]
    new_data_start = align(new_info_start + info_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(HEADER.pack(magic, version, tensor_count, kv_count + 3))
        handle.write(kv)
        handle.write(tensor_infos)
        handle.write(b"\0" * (new_data_start - handle.tell()))
        handle.write(old_data)
    os.replace(tmp, output)
    check = gguf.GGUFReader(str(output))
    assert len(check.tensors) == len(reader.tensors)
    print(f"wrote {output}")
    print("general.sampling.temp=1.0 top_k=64 top_p=0.95")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    patch(args.source, args.output)


if __name__ == "__main__":
    main()
