"""Shared GGUF access helpers: detection, metadata, and tensor enumeration.

Thin layer over ``gguf.GGUFReader`` (gguf-py). Metadata is read into a plain dict
keyed by the GGUF field name (``general.architecture``, ``gemma4.block_count`` ...);
tensors are exposed as ``GgufTensor`` records carrying the *torch* shape (ggml dims
reversed), the ggml quant type, and a zero-copy ``uint8`` view of the packed block
bytes laid out as ``[rows, row_bytes]`` (rows = product of all but the fastest ggml
dim; row_bytes spans whole quant blocks of the fastest dim).

Split GGUFs (``split.count > 1``) are read transparently: any one shard path resolves
the whole set from its ``-NNNNN-of-MMMMM.gguf`` name, metadata comes from shard 0, and
tensor enumeration walks the shards in ``split.no`` order.
"""

from __future__ import annotations

import functools
import os
import re
import struct
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch

# llama.cpp split-GGUF naming: `<...>-NNNNN-of-MMMMM.gguf`, 1-based shard number.
_SPLIT_NAME_RE = re.compile(r"^(?P<prefix>.+)-(?P<no>\d+)-of-(?P<count>\d+)\.gguf$")


def is_gguf_path(model_path: str) -> bool:
    """A single ``.gguf`` file, including one shard of a split GGUF."""
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


# llama.cpp split GGUFs are a few shards; anything far larger is a malformed/untrusted file.
_MAX_SHARDS = 1024


@functools.cache
def _shard_paths(model_path: str, count: int) -> tuple[str, ...]:
    """Shard files of a split GGUF whose KV already declared ``count`` (name math + isfile)."""
    match = _SPLIT_NAME_RE.match(os.path.basename(model_path))
    if match is None:
        raise ValueError(
            f"{model_path}: split.count={count} but the name is not '<name>-NNNNN-of-MMMMM.gguf'"
        )
    if int(match.group("count")) != count:
        raise ValueError(
            f"{model_path}: split.count={count} but the name declares {match.group('count')}"
        )
    stem, width = match.group("prefix"), len(match.group("no"))
    total = len(match.group("count"))
    folder = os.path.dirname(model_path)
    paths = tuple(
        os.path.join(folder, f"{stem}-{str(i).zfill(width)}-of-{str(count).zfill(total)}.gguf")
        for i in range(1, count + 1)
    )
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"split GGUF {model_path}: missing shards {missing}")
    return paths


@functools.cache
def _shard_readers(model_path: str) -> tuple[Any, ...]:
    """One ``gguf.GGUFReader`` per shard, split.no order; the passed path is opened once."""
    import gguf

    head = gguf.GGUFReader(model_path)
    count = _field_value(head, "split.count")
    count = int(count) if count else 1
    if count > _MAX_SHARDS:
        raise ValueError(f"{model_path}: split.count={count} exceeds the {_MAX_SHARDS} shard limit")
    if count <= 1:
        readers = (head,)
    else:
        here = os.path.realpath(model_path)
        readers = tuple(
            head if os.path.realpath(path) == here else gguf.GGUFReader(path)
            for path in _shard_paths(model_path, count)
        )
    total = _field_value(head, "split.tensors.count")
    if len(readers) > 1 and total is not None:
        seen = sum(len(r.tensors) for r in readers)
        if seen != int(total):
            raise ValueError(
                f"{model_path}: {seen} tensors across shards but split.tensors.count={total}"
            )
    return readers


def _metadata_reader(model_path: str):
    """The shard carrying the KV section (split.no == 0)."""
    return _shard_readers(model_path)[0]


def split_shard_count(model_path: str) -> int:
    """Number of shard files (1 for a plain GGUF)."""
    return len(_shard_readers(model_path))


def write_metadata_gguf(source_gguf: str, dest_path: str) -> None:
    """Write a metadata-only GGUF: the source's header + KV section byte-for-byte, with
    ``tensor_count`` patched to 0 (no tensor infos, no weight data). Reading only the
    header+KV is cheap; the multi-GB tensor data is never touched. For a split source the
    KV lives in shard 0, so that shard is copied.

    Validates by re-parsing: the copy must list zero tensors and expose the identical KV
    key set (the KV *bytes* are copied verbatim, so identical keys imply identical values).
    """
    import gguf

    reader = _metadata_reader(source_gguf)
    assert reader.tensors, f"{source_gguf}: no tensors to bound the KV section"
    # The first tensor-info record starts exactly where the KV section ends (GGUF places no
    # padding between KV and tensor infos; padding is only before the tensor *data*).
    kv_end = int(reader.tensors[0].field.offset)
    buf = bytearray(reader.data[:kv_end].tobytes())  # header + all KV, verbatim
    buf[8:16] = b"\x00" * 8  # tensor_count is a u64 at byte 8; 0 is byte-order agnostic
    # A split source carries split.count; the metadata copy has no shards, so pin it to 1.
    # Otherwise the reader would try to resolve siblings of the rewritten file name.
    if "split.count" in reader.fields:
        field = reader.fields["split.count"]
        # llama.cpp writes split.count as uint16; accept both widths and pack the same one
        # (a 4-byte write into a 2-byte field would corrupt the following KV entry).
        assert len(field.types) == 1 and field.types[0] in (
            gguf.GGUFValueType.UINT16, gguf.GGUFValueType.UINT32
        ), f"unexpected split.count type {field.types}"
        width = 2 if field.types[0] == gguf.GGUFValueType.UINT16 else 4
        struct.pack_into("<H" if width == 2 else "<I",
                         buf, int(field.offset) + 8 + len("split.count") + 4, 1)
    # The tensor table is dropped, but config derivation needs one fact from it (an
    # untied output head shows up only as an "output.weight" tensor). Append it as an
    # extra KV and bump kv_count (u64 at byte 16). Little-endian only -- the re-parse
    # below fails loudly on a big-endian source.
    key = OUTPUT_WEIGHT_PRESENT_KV.encode()
    present = any(t.name == "output.weight" for r in _shard_readers(source_gguf) for t in r.tensors)
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


@dataclass(frozen=True)
class GgufTensor:
    name: str
    shape: tuple[int, ...]  # torch order (ggml dims reversed)
    ggml_type: int
    rows: int  # product of shape[:-1] over the *ggml* layout = blocks-major rows
    row_bytes: int  # packed bytes per row (whole quant blocks of the fastest dim)
    _raw: np.ndarray  # uint8 view, shape [rows, row_bytes]

    def packed(self) -> torch.Tensor:
        """Zero-copy ``[rows, row_bytes]`` uint8 tensor of the native block bytes."""
        return torch.from_numpy(self._raw)


def _field_value(reader, name: str) -> Any:
    field = reader.fields.get(name)
    if field is None:
        return None
    return field.contents()


@functools.cache
def _reader(model_path: str):
    return _metadata_reader(model_path)


@functools.cache
def load_gguf_metadata(model_path: str) -> dict[str, Any]:
    """All GGUF KV metadata as ``{field_name: python_value}`` (arrays -> lists)."""
    reader = _reader(model_path)
    return {name: field.contents() for name, field in reader.fields.items()}


def gguf_architecture(model_path: str) -> str:
    arch = _field_value(_reader(model_path), "general.architecture")
    if arch is None:
        raise ValueError(f"GGUF file {model_path} has no general.architecture")
    return str(arch)


def iter_gguf_tensors(model_path: str) -> Iterator[GgufTensor]:
    """Yield every tensor from every shard (split.no order) with its packed block bytes."""
    import gguf

    for reader in _shard_readers(model_path):
        for t in reader.tensors:
            ne = [int(s) for s in t.shape]  # ggml order, fastest dim first
            torch_shape = tuple(reversed(ne))
            block, type_size = gguf.GGML_QUANT_SIZES[t.tensor_type]
            n_fast = ne[0]
            if n_fast % block != 0:
                raise ValueError(
                    f"{t.name}: fastest dim {n_fast} not a multiple of block {block} "
                    f"for {t.tensor_type.name}"
                )
            row_bytes = n_fast // block * type_size
            rows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
            # gguf-py returns quantized tensors as raw uint8 but F32/F16 as typed arrays;
            # normalize everything to a flat byte view before shaping into [rows, row_bytes].
            flat = np.ascontiguousarray(t.data).reshape(-1).view(np.uint8)
            raw = flat.reshape(rows, row_bytes)
            yield GgufTensor(
                name=t.name,
                shape=torch_shape,
                ggml_type=int(t.tensor_type),
                rows=rows,
                row_bytes=row_bytes,
                _raw=raw,
            )


def gguf_tensor_names(model_path: str) -> set[str]:
    return {t.name for reader in _shard_readers(model_path) for t in reader.tensors}


__all__ = [
    "is_gguf_path",
    "FTW_METADATA_GGUF",
    "OUTPUT_WEIGHT_PRESENT_KV",
    "gguf_config_source",
    "split_shard_count",
    "write_metadata_gguf",
    "GgufTensor",
    "load_gguf_metadata",
    "gguf_architecture",
    "iter_gguf_tensors",
    "gguf_tensor_names",
]
