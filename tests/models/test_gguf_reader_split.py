"""GGUF reader: split (multi-shard) resolution.

llama.cpp writes large models as ``<name>-NNNNN-of-MMMMM.gguf`` shards, each with its own
tensor table; the KV section lives in shard 0. The reader must resolve the whole set from
any one shard, take metadata from shard 0, and enumerate tensors across shards in
``split.no`` order. Shards are synthesized with ``gguf.GGUFWriter`` so the test needs no
real checkpoint.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from freetoken.models.gguf.reader import (
    gguf_architecture,
    gguf_tensor_names,
    is_gguf_path,
    iter_gguf_tensors,
    load_gguf_metadata,
    split_shard_count,
    write_metadata_gguf,
)

ARCH = "testarch"


def _write_shard(folder: str, no: int, count: int, total: int, tensors) -> str:
    import gguf

    path = os.path.join(folder, f"m-{no + 1:05d}-of-{count:05d}.gguf")
    writer = gguf.GGUFWriter(path, ARCH)
    writer.add_uint16("split.count", count)  # llama.cpp writes these as uint16
    writer.add_uint16("split.no", no)
    writer.add_uint64("split.tensors.count", total)
    for name, array in tensors:
        writer.add_tensor(name, array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _split_model(folder: str) -> tuple[str, str]:
    first = _write_shard(folder, 0, 2, 3, [
        ("a", np.arange(4, dtype=np.float32)),
        ("output.weight", np.arange(8, dtype=np.float32).reshape(2, 4)),
    ])
    second = _write_shard(folder, 1, 2, 3, [("b", np.arange(6, dtype=np.float32).reshape(2, 3))])
    return first, second


def test_split_resolves_from_any_shard(tmp_path) -> None:
    first, second = _split_model(str(tmp_path))
    for path in (first, second):
        assert is_gguf_path(path)
        assert split_shard_count(path) == 2
        assert gguf_architecture(path) == ARCH
        assert gguf_tensor_names(path) == {"a", "output.weight", "b"}
        metadata = load_gguf_metadata(path)
        assert metadata["split.count"] == 2 and metadata["general.architecture"] == ARCH


def test_split_tensors_span_shards_in_order(tmp_path) -> None:
    first, _ = _split_model(str(tmp_path))
    tensors = list(iter_gguf_tensors(first))
    assert [t.name for t in tensors] == ["a", "output.weight", "b"]
    by_name = {t.name: t for t in tensors}
    assert by_name["a"].shape == (4,) and by_name["a"].rows == 1 and by_name["a"].row_bytes == 16
    assert by_name["b"].shape == (2, 3) and by_name["b"].rows == 2 and by_name["b"].row_bytes == 12
    # zero-copy bytes come from the shard that owns the tensor
    assert np.array_equal(by_name["b"].packed().numpy().reshape(-1), np.arange(6, dtype=np.float32).view(np.uint8))


def test_plain_single_file_is_one_shard(tmp_path) -> None:
    import gguf

    path = os.path.join(tmp_path, "single.gguf")
    writer = gguf.GGUFWriter(path, ARCH)
    writer.add_tensor("x", np.arange(3, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    assert split_shard_count(path) == 1
    assert [t.name for t in iter_gguf_tensors(path)] == ["x"]


def test_missing_shard_fails_loudly(tmp_path) -> None:
    first, second = _split_model(str(tmp_path))
    os.remove(second)
    with pytest.raises(FileNotFoundError, match="missing shards"):
        split_shard_count(first)


def test_metadata_gguf_copies_shard_zero_and_records_output(tmp_path) -> None:
    from freetoken.models.gguf.reader import OUTPUT_WEIGHT_PRESENT_KV

    first, _ = _split_model(str(tmp_path))
    dest = os.path.join(str(tmp_path), "meta.gguf")
    write_metadata_gguf(first, dest)
    metadata = load_gguf_metadata(dest)
    assert metadata["general.architecture"] == ARCH
    # output.weight lives in shard 0 here; the KV records it either way.
    assert metadata[OUTPUT_WEIGHT_PRESENT_KV] is True
    # the rewritten metadata file is a single GGUF (split.count pinned to 1)
    assert split_shard_count(dest) == 1
    assert metadata["split.count"] == 1


def test_absurd_split_count_is_rejected(tmp_path) -> None:
    import gguf

    path = os.path.join(str(tmp_path), "m-00001-of-00002.gguf")
    writer = gguf.GGUFWriter(path, ARCH)
    writer.add_uint16("split.count", 5000)
    writer.add_uint16("split.no", 0)
    writer.add_uint64("split.tensors.count", 1)
    writer.add_tensor("a", np.arange(4, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    with pytest.raises(ValueError, match="exceeds"):
        split_shard_count(path)
