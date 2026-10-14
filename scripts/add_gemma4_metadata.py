"""Add Gemma 4/tokenizer metadata to a quantizer-produced GGUF."""

from __future__ import annotations

import argparse
import importlib.util
import os
import struct
from pathlib import Path

import gguf

HEADER = struct.Struct("<4sIQQ")
ALIGNMENT = 32


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def encode_kv(key: str, value) -> bytes:
    raw_key = key.encode("utf-8")
    if isinstance(value, str):
        raw = value.encode("utf-8")
        value_type = int(gguf.GGUFValueType.STRING)
        payload = struct.pack("<Q", len(raw)) + raw
    elif isinstance(value, bool):
        value_type = int(gguf.GGUFValueType.BOOL)
        payload = bytes([int(value)])
    elif isinstance(value, int):
        value_type = int(gguf.GGUFValueType.UINT32)
        payload = struct.pack("<I", value)
    elif isinstance(value, float):
        value_type = int(gguf.GGUFValueType.FLOAT32)
        payload = struct.pack("<f", value)
    elif isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            item_type = int(gguf.GGUFValueType.STRING)
            payload = b"".join(struct.pack("<Q", len(item.encode())) + item.encode() for item in value)
        elif all(isinstance(item, bool) for item in value):
            item_type = int(gguf.GGUFValueType.BOOL)
            payload = bytes(int(item) for item in value)
        elif all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            item_type = int(gguf.GGUFValueType.INT32)
            payload = b"".join(struct.pack("<i", item) for item in value)
        elif all(isinstance(item, (int, float)) for item in value):
            item_type = int(gguf.GGUFValueType.FLOAT32)
            payload = b"".join(struct.pack("<f", float(item)) for item in value)
        else:
            raise TypeError(f"unsupported array type for {key}")
        value_type = int(gguf.GGUFValueType.ARRAY)
        payload = struct.pack("<IQ", item_type, len(value)) + payload
    else:
        raise TypeError(f"unsupported GGUF metadata type for {key}: {type(value)}")
    return struct.pack("<Q", len(raw_key)) + raw_key + struct.pack("<I", value_type) + payload


def tensor_info_end(data: bytes, start: int, count: int) -> int:
    cursor = start
    for _ in range(count):
        name_len = struct.unpack_from("<Q", data, cursor)[0]
        cursor += 8 + name_len
        dims = struct.unpack_from("<I", data, cursor)[0]
        cursor += 4 + dims * 8 + 4 + 8
    return cursor


def load_source_metadata() -> dict[str, object]:
    path = Path(__file__).with_name("repair_gemma4_gguf.py")
    spec = importlib.util.spec_from_file_location("repair_gemma4_gguf", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.source_metadata


def patch(source: Path, config: Path, tokenizer: Path, chat_template: Path, output: Path) -> None:
    reader = gguf.GGUFReader(str(source))
    if not reader.tensors:
        raise ValueError("source GGUF has no tensors")
    source_bytes = source.read_bytes()
    magic, version, tensor_count, kv_count = HEADER.unpack(source_bytes[:24])
    if magic != b"GGUF" or version != 3:
        raise ValueError("source must be GGUF v3")
    kv_end = int(reader.tensors[0].field.offset)
    info_end = tensor_info_end(source_bytes, kv_end, tensor_count)
    old_data_start = reader.tensors[0].data_offset
    info_blob = source_bytes[kv_end:info_end]
    data_blob = source_bytes[old_data_start:]
    metadata = load_source_metadata()(config, tokenizer, chat_template)
    encoded = b"".join(encode_kv(key, value) for key, value in metadata.items())
    kv_blob = source_bytes[24:kv_end] + encoded
    new_data_start = align(24 + len(kv_blob) + len(info_blob))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(HEADER.pack(magic, version, tensor_count, kv_count + len(metadata)))
        handle.write(kv_blob)
        handle.write(info_blob)
        handle.write(b"\0" * (new_data_start - handle.tell()))
        handle.write(data_blob)
    os.replace(temporary, output)
    check = gguf.GGUFReader(str(output))
    assert len(check.tensors) == tensor_count
    assert check.fields["general.architecture"].contents() == "gemma4"
    print(f"wrote {output}")
    print(f"metadata keys added: {len(metadata)}; tensors preserved: {tensor_count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("chat_template", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    patch(args.source, args.config, args.tokenizer, args.chat_template, args.output)


if __name__ == "__main__":
    main()
