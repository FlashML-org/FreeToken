"""Multi-shard GGUF reading: discovery, shard-1 metadata, tensor aggregation.

Large GGUF checkpoints ship split (``-00001-of-000NN``), and llama.cpp's convention has
three properties that are easy to get wrong and nearly invisible when you do:

* ``split.no`` is 0-BASED while the filenames are 1-BASED. Shard ``-00002-of-00003``
  carries ``split.no = 1``.
* ``split.tensors.count`` is the TOTAL across every shard, not this shard's count.
* Only shard 1 carries the real metadata. Later shards hold exactly three ``split.*`` keys
  and no ``general.architecture`` at all, so anything that reads arch or tokenizer from an
  arbitrary shard gets nothing.

The fixtures below write GGUF bytes directly rather than going through ``gguf.GGUFWriter``:
the format is small, and hand-writing it keeps these tests independent of that writer's
API (which takes a required ``arch`` positional and has moved around between releases).
Layout per the spec: magic, uint32 version, uint64 tensor_count, uint64 kv_count, the KV
pairs, the tensor infos, then padding to ``general.alignment`` and the tensor data.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from freetoken.models.gguf.reader import (
    gguf_architecture,
    gguf_shards,
    gguf_tensor_names,
    is_gguf_path,
    iter_gguf_tensors,
    load_gguf_metadata,
)

# GGUF value type tags
_UINT32, _UINT64, _STRING = 4, 10, 8
_F32_TENSOR_TYPE = 0
_ALIGN = 32


def _u32(v: int) -> bytes:
    return struct.pack("<I", v)


def _u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def _string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return _u64(len(raw)) + raw


def _kv(key: str, tag: int, value) -> bytes:
    out = _string(key) + _u32(tag)
    if tag == _STRING:
        return out + _string(value)
    if tag == _UINT32:
        return out + _u32(value)
    if tag == _UINT64:
        return out + _u64(value)
    raise AssertionError(f"unhandled tag {tag}")


def _write_gguf(path: Path, kvs: list[bytes], tensors: list[tuple[str, int]]) -> None:
    """Write a GGUF with ``tensors`` as [(name, n_elements)], each F32.

    Tensor data is written contiguously after the aligned header; the values themselves are
    irrelevant here since these tests only exercise discovery, metadata and the tensor
    table.
    """
    head = b"GGUF" + _u32(3) + _u64(len(tensors)) + _u64(len(kvs))
    head += b"".join(kvs)
    offset = 0
    infos = b""
    for name, n in tensors:
        infos += _string(name) + _u32(1) + _u64(n) + _u32(_F32_TENSOR_TYPE) + _u64(offset)
        nbytes = n * 4
        offset += (nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
    body = head + infos
    pad = (-len(body)) % _ALIGN
    body += b"\0" * pad
    body += b"\0" * offset
    path.write_bytes(body)


def _full_kvs(arch: str = "qwen3moe", *, extra: list[bytes] | None = None) -> list[bytes]:
    """Shard 1's KV block: the real metadata."""
    kvs = [
        _kv("general.architecture", _STRING, arch),
        _kv("general.alignment", _UINT32, _ALIGN),
        _kv(f"{arch}.block_count", _UINT32, 4),
        _kv(f"{arch}.embedding_length", _UINT32, 128),
    ]
    return kvs + (extra or [])


def _split_kvs(no: int, count: int, total_tensors: int) -> list[bytes]:
    """A non-first shard's KV block: exactly the three split keys, no architecture."""
    return [
        _kv("split.no", _UINT32, no),
        _kv("split.count", _UINT32, count),
        _kv("split.tensors.count", _UINT32, total_tensors),
    ]


def _make_split(tmp_path: Path, base: str, per_shard: list[list[str]], *,
                declared_count: int | None = None) -> list[Path]:
    """Write a split set; returns the shard paths in order."""
    n = len(per_shard)
    declared = declared_count if declared_count is not None else n
    total = sum(len(names) for names in per_shard)
    paths = []
    for i, names in enumerate(per_shard):
        p = tmp_path / f"{base}-{i + 1:05d}-of-{n:05d}.gguf"
        kvs = (_full_kvs() + _split_kvs(0, declared, total)) if i == 0 \
            else _split_kvs(i, declared, total)
        _write_gguf(p, kvs, [(nm, 8) for nm in names])
        paths.append(p)
    return paths


class TestSingleFileUnchanged:
    def test_single_file_unchanged(self, tmp_path: Path):
        """A plain one-file GGUF must behave exactly as before the shard work."""
        p = tmp_path / "single.gguf"
        _write_gguf(p, _full_kvs(), [("token_embd.weight", 8), ("output.weight", 8)])
        assert is_gguf_path(str(p))
        assert gguf_shards(str(p)) == [str(p)]
        assert gguf_architecture(str(p)) == "qwen3moe"
        assert load_gguf_metadata(str(p))["qwen3moe.block_count"] == 4
        assert gguf_tensor_names(str(p)) == {"token_embd.weight", "output.weight"}
        assert len(list(iter_gguf_tensors(str(p)))) == 2


class TestShardDiscovery:
    def test_discovery_from_any_shard_or_directory(self, tmp_path: Path):
        paths = _make_split(tmp_path, "m", [["a.weight"], ["b.weight"], ["c.weight"]])
        want = [str(p) for p in paths]
        for handed in (paths[0], paths[1], paths[2]):
            assert gguf_shards(str(handed)) == want, f"from {handed.name}"
        # a user pointing at the folder must work too
        assert gguf_shards(str(tmp_path)) == want

    def test_every_shard_is_a_gguf_path(self, tmp_path: Path):
        paths = _make_split(tmp_path, "m", [["a.weight"], ["b.weight"]])
        for p in paths:
            assert is_gguf_path(str(p))


class TestMissingShard:
    def test_missing_shard_raises_naming_the_index(self, tmp_path: Path):
        """A truncated download must fail loudly, never load as a partial model."""
        paths = _make_split(tmp_path, "m", [["a.weight"], ["b.weight"], ["c.weight"]])
        paths[1].unlink()  # drop the middle shard
        with pytest.raises(Exception) as e:
            gguf_shards(str(paths[0]))
        assert "2" in str(e.value), f"error should name the missing index: {e.value}"


class TestMetadataFromShardOne:
    def test_metadata_resolves_to_shard_one(self, tmp_path: Path):
        """Later shards carry no architecture, so reads must resolve back to shard 1."""
        paths = _make_split(tmp_path, "m", [["a.weight"], ["b.weight"], ["c.weight"]])
        for p in paths:
            assert gguf_architecture(str(p)) == "qwen3moe", f"from {p.name}"
            assert load_gguf_metadata(str(p))["qwen3moe.block_count"] == 4, f"from {p.name}"

    def test_later_shards_really_lack_arch(self, tmp_path: Path):
        """Guards the fixture itself: if shard 2 carried arch, the test above proves nothing."""
        paths = _make_split(tmp_path, "m", [["a.weight"], ["b.weight"]])
        import gguf as gguf_pkg

        r = gguf_pkg.GGUFReader(str(paths[1]))
        assert "general.architecture" not in r.fields
        assert "split.no" in r.fields


class TestTensorAggregation:
    def test_tensors_aggregate_across_shards(self, tmp_path: Path):
        per = [["a.weight", "b.weight"], ["c.weight"], ["d.weight", "e.weight"]]
        paths = _make_split(tmp_path, "m", per)
        expected = {n for names in per for n in names}
        for p in paths:
            assert gguf_tensor_names(str(p)) == expected, f"from {p.name}"
        names = [t.name for t in iter_gguf_tensors(str(paths[0]))]
        assert names == [n for names_ in per for n in names_], "shard order must be preserved"
        assert len(names) == load_gguf_metadata(str(paths[0]))["split.tensors.count"]


class TestDeclaredCountMismatch:
    def test_declared_count_mismatch_raises(self, tmp_path: Path):
        """shard 1 says 3 shards but only 2 exist on disk."""
        tmp = tmp_path / "sub"
        tmp.mkdir()
        _make_split(tmp, "m", [["a.weight"], ["b.weight"]], declared_count=3)
        first = tmp / "m-00001-of-00002.gguf"
        with pytest.raises(Exception):
            list(iter_gguf_tensors(str(first)))
