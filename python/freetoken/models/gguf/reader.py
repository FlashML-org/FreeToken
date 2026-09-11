"""Shared GGUF access helpers: detection, metadata, and tensor enumeration.

Thin layer over ``gguf.GGUFReader`` (gguf-py). Metadata is read into a plain dict
keyed by the GGUF field name (``general.architecture``, ``gemma4.block_count`` ...);
tensors are exposed as ``GgufTensor`` records carrying the *torch* shape (ggml dims
reversed), the ggml quant type, and a zero-copy ``uint8`` view of the packed block
bytes laid out as ``[rows, row_bytes]`` (rows = product of all but the fastest ggml
dim; row_bytes spans whole quant blocks of the fastest dim).

Multi-shard GGUF support (llama.cpp split convention, per ground truth in spec):

Filenames follow ``<base>-%05d-of-%05d.gguf``, both numbers 1-based. The split layout is:

  - Shard 1 (``-00001-of-000NN``) holds the FULL KV metadata: ``general.architecture``,
    all ``<arch>.*`` config keys, all ``tokenizer.*`` keys. It also carries tensor count
    and a ``split.no = 0`` marker (0-based, even though filenames are 1-based).
  - Shards 2..N (``-00002-of-000NN`` to ``-000NN-of-000NN``) carry exactly 3 metadata keys:
    ``split.no`` (1..N-1), ``split.count`` (always NN), and ``split.tensors.count`` (the
    TOTAL tensor count across all shards, not per-shard). They list no architecture keys.
  - Tensor distribution: e.g. Hy3 IQ1_M (1298 total) splits as shard 1 with 694 tensors,
    shard 2 with 604 tensors. ``split.tensors.count`` is always 1298.

Examples:

  - A bare ``.gguf`` file (no shard marker) -> no change, single-file path throughout.
  - ``model-00001-of-00002.gguf`` passed to any function -> caller gets shard 1 metadata
    and tensors from both shards 1..2 in order.
  - A directory containing ``model-00001-of-00002.gguf`` -> caller passes the dir,
    is_gguf_path resolves it, downstream gets the first-shard path.

Validation on open: if shard 1 declares ``split.count = N``, all N shards must exist
(1..N), no gaps. Also assert summed tensor count across all shards equals
``split.tensors.count``. Both keys are read from shard 1 only.

Shard readers are cached per path (one per shard file), so opening ``-00002-of-00002``
after ``-00001-of-00002`` will reuse the first-shard reader (no double-load of shard 1).
"""

from __future__ import annotations

import functools
import glob
import os
import re
import struct
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch


def gguf_shards(path: str) -> list[str]:
    r"""Return the ordered list of shard paths given any shard's path (or a plain .gguf).

    Matches the llama.cpp pattern ``(?P<base>.+)-(\d{5})-of-(\d{5})\.gguf$`` on the
    basename. A non-shard path returns ``[path]``. For a shard path, globs sibling shards,
    sorts by index, and validates the set is complete 1..N with none missing.

    Raises a clear error naming the missing indices if any are absent (truncated downloads
    are the common failure case and must not load silently).
    """
    # A directory: find the first shard inside it and continue from there. Users routinely
    # pass the folder a split model was downloaded into rather than a specific shard.
    if os.path.isdir(path):
        first = sorted(glob.glob(os.path.join(path, "*-00001-of-?????.gguf")))
        if len(first) > 1:
            raise ValueError(
                f"{path}: contains {len(first)} different split models "
                f"({[os.path.basename(f) for f in first]}); point at one shard instead"
            )
        if not first:
            return [path]
        path = first[0]

    basename = os.path.basename(path)
    match = re.match(r"(?P<base>.+)-(\d{5})-of-(\d{5})\.gguf$", basename)
    if not match:
        # Not a shard file; return as single-file path.
        return [path]

    base, shard_idx_str, total_shards_str = match.group("base"), match.group(2), match.group(3)
    total_shards = int(total_shards_str)
    shard_dir = os.path.dirname(path)

    # Glob all sibling shards
    pattern = os.path.join(shard_dir, f"{base}-?????-of-{total_shards_str}.gguf")
    found_shards = sorted(glob.glob(pattern))

    # Parse indices and validate completeness
    shard_indices = set()
    shard_map = {}  # index -> path
    for shard_path in found_shards:
        shard_basename = os.path.basename(shard_path)
        shard_match = re.match(rf"{re.escape(base)}-(\d{{5}})-of-{total_shards_str}\.gguf$", shard_basename)
        if shard_match:
            idx = int(shard_match.group(1))
            shard_indices.add(idx)
            shard_map[idx] = shard_path

    # Verify complete range 1..N
    expected = set(range(1, total_shards + 1))
    if shard_indices != expected:
        missing = sorted(expected - shard_indices)
        raise ValueError(
            f"Incomplete shard set for {base}: expected shards 1..{total_shards}, "
            f"missing {missing}. (Truncated download?)"
        )

    # Return in order 1..N
    return [shard_map[i] for i in range(1, total_shards + 1)]


def resolve_gguf_path(model_path: str) -> str | None:
    """Resolve a path to the first shard (shard 1) of a GGUF file.

    Accepts:
      - A single ``.gguf`` file -> returns it as-is.
      - A shard file (e.g., ``-00002-of-00002.gguf``) -> returns shard 1 path.
      - A directory containing exactly one shard 1 file -> returns that file path.

    Returns ``None`` if the path is none of the above.
    """
    if not isinstance(model_path, str):
        return None

    # Case 1: A single .gguf file (not a shard)
    if os.path.isfile(model_path) and model_path.endswith(".gguf"):
        basename = os.path.basename(model_path)
        if not re.match(r".+-\d{5}-of-\d{5}\.gguf$", basename):
            # Plain .gguf, not a shard
            return model_path

    # Case 2: A shard file or a directory
    if os.path.isfile(model_path) and model_path.endswith(".gguf"):
        # It's a shard file; get shard 1
        shards = gguf_shards(model_path)
        return shards[0] if shards else None

    if os.path.isdir(model_path):
        # Look for exactly one shard-1 file in the directory
        pattern = os.path.join(model_path, "*-00001-of-?????.gguf")
        candidates = glob.glob(pattern)
        if len(candidates) == 1:
            return candidates[0]

    return None


def is_gguf_path(model_path: str) -> bool:
    """A ``.gguf`` file or directory, supporting single files and multi-shard layouts.

    Accepts:
      - A single ``.gguf`` file.
      - Any shard of a multi-shard ``.gguf`` (e.g., shard 2 of 5).
      - A directory containing exactly one shard 1 file.

    Returns ``True`` only if one of these conditions holds.
    """
    if not isinstance(model_path, str):
        return False

    # Case 1: A .gguf file (single or shard)
    if os.path.isfile(model_path) and model_path.endswith(".gguf"):
        return True

    # Case 2: A directory with a shard 1 file
    if os.path.isdir(model_path):
        pattern = os.path.join(model_path, "*-00001-of-?????.gguf")
        candidates = glob.glob(pattern)
        return len(candidates) == 1

    return False


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

    A bare ``.gguf`` file or any shard resolves to the first shard; an FTW dir carrying
    a :data:`FTW_METADATA_GGUF` resolves to that embedded metadata file. A directory
    containing shards resolves to shard 1. This is the single seam config/tokenizer
    dispatch uses to decide "this checkpoint is GGUF-config-sourced" -- a real file, a
    shard file, a shard directory, and a converted-FTW dir all land on a genuine ``.gguf``
    path the reader can parse, so no downstream code learns about the layout.
    """
    # Case 1: Check for FTW metadata file first (highest priority)
    if isinstance(model_path, str) and os.path.isdir(model_path):
        cand = os.path.join(model_path, FTW_METADATA_GGUF)
        if os.path.isfile(cand):
            return cand

    # Case 2: Try to resolve to a GGUF (single, shard, or directory)
    resolved = resolve_gguf_path(model_path)
    if resolved is not None:
        return resolved

    return None


def write_metadata_gguf(source_gguf: str, dest_path: str) -> None:
    """Write a metadata-only GGUF: the source's header + KV section byte-for-byte, with
    ``tensor_count`` patched to 0 (no tensor infos, no weight data). Reading only the
    header+KV is cheap; the multi-GB tensor data is never touched.

    Validates by re-parsing: the copy must list zero tensors and expose the identical KV
    key set (the KV *bytes* are copied verbatim, so identical keys imply identical values).
    """
    import gguf

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
    """Get or create a GGUFReader for the given path, with shard validation.

    For single-shard files, this is a pass-through. For shard 1 of a multi-shard set,
    this validates that:
      1. All shards 1..N are present and complete (no missing indices).
      2. The summed tensor count across all shards matches split.tensors.count (if present).
    """
    import gguf

    reader = gguf.GGUFReader(model_path)

    # Check if this is shard 1 of a multi-shard set
    split_count = _field_value(reader, "split.count")
    split_no = _field_value(reader, "split.no")

    if split_count is not None and split_no == 0:
        # This is shard 1 of a multi-shard set; validate completeness
        try:
            shards = gguf_shards(model_path)
            if len(shards) != split_count:
                raise ValueError(
                    f"GGUF shard validation: {model_path} declares split.count={split_count}, "
                    f"but found {len(shards)} shards"
                )

            # Validate tensor count sum if split.tensors.count is declared
            split_tensors_count = _field_value(reader, "split.tensors.count")
            if split_tensors_count is not None:
                total_tensor_count = 0
                for shard_path in shards:
                    shard_reader = gguf.GGUFReader(shard_path)
                    total_tensor_count += len(shard_reader.tensors)

                if total_tensor_count != split_tensors_count:
                    raise ValueError(
                        f"GGUF shard validation: {model_path} declares "
                        f"split.tensors.count={split_tensors_count}, but summed tensor count "
                        f"across all shards is {total_tensor_count}"
                    )
        except ValueError:
            raise

    return reader


@functools.cache
def load_gguf_metadata(model_path: str) -> dict[str, Any]:
    """All GGUF KV metadata as ``{field_name: python_value}`` (arrays -> lists).

    Metadata is read from shard 1 only. If the caller passes any other shard, this
    function resolves to shard 1 first.
    """
    shard1_path = resolve_gguf_path(model_path)
    if shard1_path is None:
        raise ValueError(f"Cannot resolve GGUF path: {model_path}")

    reader = _reader(shard1_path)
    return {name: field.contents() for name, field in reader.fields.items()}


def gguf_architecture(model_path: str) -> str:
    """The model architecture string (e.g., "qwen3moe", "qwen35moe").

    Architecture is read from shard 1 only. If the caller passes any other shard,
    this function resolves to shard 1 first.
    """
    shard1_path = resolve_gguf_path(model_path)
    if shard1_path is None:
        raise ValueError(f"Cannot resolve GGUF path: {model_path}")

    arch = _field_value(_reader(shard1_path), "general.architecture")
    if arch is None:
        raise ValueError(f"GGUF file {shard1_path} has no general.architecture")
    return str(arch)


def iter_gguf_tensors(model_path: str) -> Iterator[GgufTensor]:
    """Yield every tensor with its torch shape, ggml type, and packed block bytes.

    For multi-shard files, yields tensors from shard 1, then shard 2, ..., in order.
    Single-shard files take exactly the same code path (gguf_shards returns [path]).
    """
    import gguf

    shard1_path = resolve_gguf_path(model_path)
    if shard1_path is None:
        raise ValueError(f"Cannot resolve GGUF path: {model_path}")

    # Get all shard paths in order
    shards = gguf_shards(shard1_path)

    # Iterate over each shard and yield tensors
    for shard_path in shards:
        reader = _reader(shard_path)
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
    """The union of tensor names across all shards.

    For multi-shard files, returns the union of tensor names from shard 1, shard 2, etc.
    Single-shard files take exactly the same code path.
    """
    shard1_path = resolve_gguf_path(model_path)
    if shard1_path is None:
        raise ValueError(f"Cannot resolve GGUF path: {model_path}")

    shards = gguf_shards(shard1_path)
    names = set()
    for shard_path in shards:
        reader = _reader(shard_path)
        names.update(t.name for t in reader.tensors)
    return names


__all__ = [
    "gguf_shards",
    "resolve_gguf_path",
    "is_gguf_path",
    "FTW_METADATA_GGUF",
    "OUTPUT_WEIGHT_PRESENT_KV",
    "gguf_config_source",
    "write_metadata_gguf",
    "GgufTensor",
    "load_gguf_metadata",
    "gguf_architecture",
    "iter_gguf_tensors",
    "gguf_tensor_names",
]
