"""Shared GGUF access helpers: detection, metadata, and tensor enumeration.

Thin layer over ``gguf.GGUFReader`` (gguf-py). Metadata is read into a plain dict
keyed by the GGUF field name (``general.architecture``, ``gemma4.block_count`` ...);
tensors are indexed as ``GgufTensorInfo`` records (ggml dims, ggml type, and where the
packed bytes sit in the memory-mapped file).
"""

from __future__ import annotations

import functools
import os
import re
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np


def is_gguf_path(model_path: str) -> bool:
    """A single ``.gguf`` file (the only GGUF layout FreeToken loads directly)."""
    return isinstance(model_path, str) and os.path.isfile(model_path) and model_path.endswith(
        ".gguf"
    )


# Canonical name of the metadata-only GGUF that ``convert_checkpoint`` drops into an FTW
# dir built from a bare ``.gguf`` source. A GGUF carries its config AND tokenizer in the
# file's KV section, not sibling files, so a converted checkpoint has nowhere else to read
# them from -- this file is the header + KV bytes verbatim (tensor_count patched to 0, no
# tensor infos, no weight data), letting the FTW dir resolve config/tokenizer the exact
# same way the original ``.gguf`` file does.
FTW_METADATA_GGUF = "source_metadata.gguf"
# Records whether the source carried an untied "output.weight" head (the tensor table
# is stripped from metadata-only gguf files, so the fact travels as a KV).
OUTPUT_WEIGHT_PRESENT_KV = "freetoken.output_weight_present"


def gguf_config_source(model_path: str) -> str | None:
    """The ``.gguf`` file to source config/tokenizer/metadata from, or ``None``.

    A bare ``.gguf`` file resolves to itself; an FTW dir carrying a
    :data:`FTW_METADATA_GGUF` resolves to that embedded metadata file. This is the single
    seam config/tokenizer dispatch uses to decide "this checkpoint is GGUF-config-sourced"
    -- a real file and a converted-FTW dir both land on a genuine ``.gguf`` path the reader
    can parse, so no downstream code learns about the FTW wrapper.
    """
    if is_gguf_path(model_path):
        return model_path
    if isinstance(model_path, str) and os.path.isdir(model_path):
        cand = os.path.join(model_path, FTW_METADATA_GGUF)
        if os.path.isfile(cand):
            return cand
    return None


def write_metadata_gguf(source_gguf: str, dest_path: str) -> None:
    """Write a metadata-only GGUF: the source's header + KV section byte-for-byte, with
    ``tensor_count`` patched to 0 (no tensor infos, no weight data). Reading only the
    header+KV is cheap; the multi-GB tensor data is never touched.

    Validates by re-parsing: the copy must list zero tensors and expose the identical KV
    key set (the KV *bytes* are copied verbatim, so identical keys imply identical values).
    """
    from freetoken.third_party import gguf

    reader = gguf.GGUFReader(source_gguf)
    assert reader.tensors, f"{source_gguf}: no tensors to bound the KV section"
    # The first tensor-info record starts exactly where the KV section ends (GGUF places no
    # padding between KV and tensor infos; padding is only before the tensor *data*).
    kv_end = int(reader.tensors[0].field.offset)
    buf = bytearray(reader.data[:kv_end].tobytes())  # header + all KV, verbatim
    buf[8:16] = b"\x00" * 8  # tensor_count is a u64 at byte 8; 0 is byte-order agnostic
    # The tensor table is dropped, but config derivation needs one fact from it (an
    # untied output head shows up only as an "output.weight" tensor). Append it as an
    # extra KV and bump kv_count (u64 at byte 16). Little-endian only -- the re-parse
    # below fails loudly on a big-endian source.
    key = OUTPUT_WEIGHT_PRESENT_KV.encode()
    present = any(t.name == "output.weight" for t in reader.tensors)
    buf += struct.pack("<Q", len(key)) + key
    buf += struct.pack("<I", int(gguf.GGUFValueType.BOOL)) + bytes([1 if present else 0])
    struct.pack_into("<Q", buf, 16, struct.unpack_from("<Q", buf, 16)[0] + 1)
    tmp = dest_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(buf)
    os.replace(tmp, dest_path)

    check = gguf.GGUFReader(dest_path)
    assert not check.tensors, "metadata gguf still lists tensors after patch"
    src_keys = {k for k in reader.fields if not k.startswith("GGUF.")}
    dst_keys = {k for k in check.fields if not k.startswith("GGUF.")}
    assert dst_keys == src_keys | {OUTPUT_WEIGHT_PRESENT_KV}, (
        f"metadata gguf KV keys differ from source: "
        f"missing {sorted(src_keys - dst_keys)}, extra {sorted(dst_keys - src_keys - {OUTPUT_WEIGHT_PRESENT_KV})}"
    )


def _field_value(reader, name: str) -> Any:
    field = reader.fields.get(name)
    if field is None:
        return None
    return field.contents()


@functools.cache
def _reader(model_path: str):
    from freetoken.third_party import gguf

    return gguf.GGUFReader(model_path)


@functools.cache
def load_gguf_metadata(model_path: str) -> dict[str, Any]:
    """All GGUF KV metadata as ``{field_name: python_value}`` (arrays -> lists)."""
    reader = _reader(model_path)
    return {name: field.contents() for name, field in reader.fields.items()}


_SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


@functools.cache
def gguf_split_paths(model_path: str) -> tuple[str, ...]:
    """Every file of a split GGUF (``name-00001-of-00002.gguf``), in order; a single file is its own list."""
    count = int(_field_value(_reader(model_path), "split.count") or 1)
    if count == 1:
        return (model_path,)
    match = _SPLIT_RE.search(model_path)
    if match is None:
        raise ValueError(f"{model_path} is one of {count} GGUF splits but is not named like name-00001-of-{count:05d}.gguf")
    stem = model_path[: match.start()]
    paths = tuple(f"{stem}-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1))
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"GGUF split files missing: {missing}")
    return paths


@dataclass(frozen=True)
class GgufTensorInfo:
    """Where one tensor's packed bytes sit: ``ne`` is in ggml order (fastest dim first)."""

    name: str
    path: str
    offset: int
    nbytes: int
    ggml_type: int
    ne: tuple[int, ...]

    def data(self) -> np.ndarray:
        """Zero-copy uint8 view of the bytes (memory-mapped)."""
        reader = _reader(self.path)
        return reader.data[self.offset : self.offset + self.nbytes]


@functools.cache
def gguf_tensor_index(model_path: str) -> dict[str, GgufTensorInfo]:
    """Every tensor of every split, by name."""
    index: dict[str, GgufTensorInfo] = {}
    for path in gguf_split_paths(model_path):
        for t in _reader(path).tensors:
            index[t.name] = GgufTensorInfo(
                t.name, path, int(t.data_offset), int(t.n_bytes), int(t.tensor_type), tuple(int(x) for x in t.shape)
            )
    return index


def gguf_architecture(model_path: str) -> str:
    arch = _field_value(_reader(model_path), "general.architecture")
    if arch is None:
        raise ValueError(f"GGUF file {model_path} has no general.architecture")
    return str(arch)


def gguf_tensor_names(model_path: str) -> set[str]:
    return {t.name for path in gguf_split_paths(model_path) for t in _reader(path).tensors}


__all__ = [
    "is_gguf_path",
    "FTW_METADATA_GGUF",
    "OUTPUT_WEIGHT_PRESENT_KV",
    "gguf_config_source",
    "write_metadata_gguf",
    "GgufTensorInfo",
    "gguf_split_paths",
    "gguf_tensor_index",
    "load_gguf_metadata",
    "gguf_architecture",
    "gguf_tensor_names",
]
