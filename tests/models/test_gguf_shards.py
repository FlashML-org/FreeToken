from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


class _Field:
    def __init__(self, value):
        self._value = value

    def contents(self):
        return self._value


def _fake_tensor(name, ggml_type, shape, data):
    import gguf

    enum_type = next(key for key in gguf.GGML_QUANT_SIZES if int(key) == ggml_type)
    return SimpleNamespace(name=name, tensor_type=enum_type, shape=shape, data=data)


def _fake_reader(*tensors, split_count=1):
    fields = {"split.count": _Field(split_count)} if split_count > 1 else {}
    return SimpleNamespace(fields=fields, tensors=list(tensors))


def test_split_gguf_files_are_read_in_order(monkeypatch, tmp_path):
    from freetoken.models.gguf import reader

    first = tmp_path / "model-00001-of-00002.gguf"
    second = tmp_path / "model-00002-of-00002.gguf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    q0 = _fake_tensor("first", 2, [32], np.zeros(18, dtype=np.uint8))
    q1 = _fake_tensor("second", 2, [32], np.zeros(18, dtype=np.uint8))
    readers = {
        str(first): _fake_reader(q0, split_count=2),
        str(second): _fake_reader(q1, split_count=2),
    }
    monkeypatch.setattr(reader, "_reader", readers.__getitem__)

    assert reader.gguf_shard_paths(str(first)) == (str(first), str(second))
    tensors = list(reader.iter_gguf_tensors(str(first)))
    assert [tensor.name for tensor in tensors] == ["first", "second"]
    assert reader.gguf_tensor_names(str(first)) == {"first", "second"}


def test_missing_split_gguf_file_fails_before_loading(monkeypatch, tmp_path):
    from freetoken.models.gguf import reader

    first = tmp_path / "model-00001-of-00002.gguf"
    first.write_bytes(b"first")
    monkeypatch.setattr(
        reader,
        "_reader",
        lambda path: _fake_reader(split_count=2),
    )

    with pytest.raises(FileNotFoundError, match="missing GGUF shard"):
        reader.gguf_shard_paths(str(first))


def test_duplicate_tensor_names_across_shards_fail(monkeypatch, tmp_path):
    from freetoken.models.gguf import reader

    first = tmp_path / "model-00001-of-00002.gguf"
    second = tmp_path / "model-00002-of-00002.gguf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    duplicate0 = _fake_tensor("same", 2, [32], np.zeros(18, dtype=np.uint8))
    duplicate1 = _fake_tensor("same", 2, [32], np.zeros(18, dtype=np.uint8))
    readers = {
        str(first): _fake_reader(duplicate0, split_count=2),
        str(second): _fake_reader(duplicate1, split_count=2),
    }
    monkeypatch.setattr(reader, "_reader", readers.__getitem__)

    with pytest.raises(ValueError, match="duplicate GGUF tensor 'same'"):
        list(reader.iter_gguf_tensors(str(first)))


def test_bad_packed_row_size_fails_loudly(monkeypatch, tmp_path):
    from freetoken.models.gguf import reader

    path = tmp_path / "model.gguf"
    path.write_bytes(b"model")
    bad = _fake_tensor("bad", 2, [32], np.zeros(17, dtype=np.uint8))
    monkeypatch.setattr(reader, "_reader", lambda _: _fake_reader(bad))

    with pytest.raises(ValueError, match="expected 18"):
        list(reader.iter_gguf_tensors(str(path)))
