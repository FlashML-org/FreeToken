"""Add Gemma 4 model and tokenizer metadata to a tensor-only GGUF file."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path
from typing import BinaryIO, Iterable

HEADER = struct.Struct("<4sIQQ")
ALIGNMENT = 32
KV_STRING = 8
KV_FLOAT32 = 6
KV_BOOL = 7
KV_ARRAY = 9
KV_UINT32 = 4
KV_INT32 = 5
TOKEN_NORMAL = 1
TOKEN_CONTROL = 3


def align(value: int, boundary: int = ALIGNMENT) -> int:
    return (value + boundary - 1) // boundary * boundary


def encode_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def encode_value(value) -> tuple[int, bytes]:
    if isinstance(value, str):
        return KV_STRING, encode_string(value)
    if isinstance(value, bool):
        return KV_BOOL, bytes([int(value)])
    if isinstance(value, int):
        return KV_UINT32, struct.pack("<I", value)
    if isinstance(value, float):
        return KV_FLOAT32, struct.pack("<f", value)
    if isinstance(value, list):
        if not value:
            raise ValueError("empty GGUF arrays need an explicit element type")
        if all(isinstance(item, str) for item in value):
            item_type = KV_STRING
            payload = b"".join(encode_string(item) for item in value)
        elif all(isinstance(item, bool) for item in value):
            item_type = KV_BOOL
            payload = bytes(int(item) for item in value)
        elif all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            item_type = KV_INT32
            payload = b"".join(struct.pack("<i", item) for item in value)
        elif all(isinstance(item, (int, float)) for item in value):
            item_type = KV_FLOAT32
            payload = b"".join(struct.pack("<f", float(item)) for item in value)
        else:
            raise TypeError(f"unsupported GGUF array values: {type(value[0])}")
        return KV_ARRAY, struct.pack("<IQ", item_type, len(value)) + payload
    raise TypeError(f"unsupported GGUF value: {type(value)}")


def encode_kv(metadata: dict[str, object]) -> bytes:
    encoded = bytearray()
    for key, value in metadata.items():
        key_bytes = key.encode("utf-8")
        value_type, value_bytes = encode_value(value)
        encoded += struct.pack("<Q", len(key_bytes)) + key_bytes
        encoded += struct.pack("<I", value_type) + value_bytes
    return bytes(encoded)


def tensor_info_end(handle: BinaryIO, tensor_count: int) -> int:
    handle.seek(24)
    for _ in range(tensor_count):
        name_length = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(name_length, os.SEEK_CUR)
        dimensions = struct.unpack("<I", handle.read(4))[0]
        handle.seek(8 * dimensions + 4 + 8, os.SEEK_CUR)
    return handle.tell()


def source_metadata(config_path: Path, tokenizer_path: Path, chat_template_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text = config["text_config"]
    layer_types = text["layer_types"]
    is_swa = [layer_type == "sliding_attention" for layer_type in layer_types]

    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    vocab = tokenizer["model"]["vocab"]
    tokens = [None] * len(vocab)
    for token, token_id in vocab.items():
        tokens[token_id] = token
    if any(token is None for token in tokens):
        raise ValueError("tokenizer vocabulary IDs are not contiguous")
    special_ids = {item["id"] for item in tokenizer.get("added_tokens", []) if item.get("special")}
    merges = tokenizer["model"].get("merges", [])
    merges = [" ".join(item) if isinstance(item, list) else item for item in merges]
    chat_template = chat_template_path.read_text(encoding="utf-8")

    metadata = {
        "general.architecture": "gemma4",
        "general.alignment": ALIGNMENT,
        "general.name": config.get("_name_or_path", "Gemma 4"),
        "freetoken.gguf_format": "hf_nvfp4",
        "gemma4.block_count": int(text["num_hidden_layers"]),
        "gemma4.embedding_length": int(text["hidden_size"]),
        "gemma4.attention.head_count": int(text["num_attention_heads"]),
        "gemma4.attention.head_count_kv": [int(text["num_key_value_heads"] if is_swa_item else text["num_global_key_value_heads"]) for is_swa_item in is_swa],
        "gemma4.attention.sliding_window_pattern": is_swa,
        "gemma4.attention.key_length_swa": int(text["head_dim"]),
        "gemma4.attention.key_length": int(text["global_head_dim"]),
        "gemma4.context_length": int(text["max_position_embeddings"]),
        "gemma4.feed_forward_length": int(text["intermediate_size"]),
        "gemma4.expert_count": int(text["num_experts"]),
        "gemma4.expert_used_count": int(text["top_k_experts"]),
        "gemma4.expert_feed_forward_length": int(text["moe_intermediate_size"]),
        "gemma4.attention.layer_norm_rms_epsilon": float(text["rms_norm_eps"]),
        "gemma4.rope.dimension_count_swa": int(text["head_dim"]),
        "gemma4.rope.dimension_count": int(text["global_head_dim"]),
        "gemma4.rope.freq_base": float(text["rope_parameters"]["full_attention"]["rope_theta"]),
        "gemma4.rope.freq_base_swa": float(text["rope_parameters"]["sliding_attention"]["rope_theta"]),
        "gemma4.attention.sliding_window": int(text["sliding_window"]),
        "gemma4.final_logit_softcapping": float(text["final_logit_softcapping"]),
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.scores": [0.0] * len(tokens),
        "tokenizer.ggml.token_type": [TOKEN_CONTROL if index in special_ids else TOKEN_NORMAL for index in range(len(tokens))],
        "tokenizer.ggml.merges": merges,
        "tokenizer.ggml.bos_token_id": 2,
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.unknown_token_id": 3,
        "tokenizer.ggml.padding_token_id": 0,
        "tokenizer.chat_template": chat_template,
    }
    generation_path = config_path.with_name("generation_config.json")
    if generation_path.exists():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if generation.get("temperature") is not None:
            metadata["general.sampling.temp"] = float(generation["temperature"])
        if generation.get("top_k") is not None:
            metadata["general.sampling.top_k"] = int(generation["top_k"])
        if generation.get("top_p") is not None:
            metadata["general.sampling.top_p"] = float(generation["top_p"])
    return metadata


def repair(source: Path, output: Path, metadata: dict[str, object]) -> None:
    encoded_metadata = encode_kv(metadata)
    with source.open("rb") as source_handle:
        magic, version, tensor_count, kv_count = HEADER.unpack(source_handle.read(24))
        if magic != b"GGUF" or version != 3:
            raise ValueError(f"expected GGUF v3, got magic={magic!r}, version={version}")
        if kv_count:
            raise ValueError("refusing to overwrite existing GGUF metadata")
        info_end = tensor_info_end(source_handle, tensor_count)
        old_data_start = align(info_end)
        source_handle.seek(0, os.SEEK_END)
        source_size = source_handle.tell()
        source_handle.seek(24)
        tensor_infos = source_handle.read(info_end - 24)
        new_info_end = 24 + len(encoded_metadata) + len(tensor_infos)
        new_data_start = align(new_info_end)

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("wb") as output_handle:
            output_handle.write(HEADER.pack(b"GGUF", version, tensor_count, len(metadata)))
            output_handle.write(encoded_metadata)
            output_handle.write(tensor_infos)
            output_handle.write(b"\0" * (new_data_start - output_handle.tell()))
            source_handle.seek(old_data_start)
            remaining = source_size - old_data_start
            while remaining:
                chunk = source_handle.read(min(16 * 1024 * 1024, remaining))
                if not chunk:
                    raise IOError("source GGUF ended while copying tensor data")
                output_handle.write(chunk)
                remaining -= len(chunk)
        os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("chat_template", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    metadata = source_metadata(args.config, args.tokenizer, args.chat_template)
    repair(args.source, args.output, metadata)
    print(f"wrote {args.output} with {len(metadata)} metadata keys")


if __name__ == "__main__":
    main()
