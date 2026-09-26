from __future__ import annotations

import copy
import json
import mmap
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch

from freetoken.checkpoint.ftw import (
    ALIGN,
    FORMAT_TAG,
    FORMAT_VERSION,
    INDEX_NAME,
    FTWFormatError,
    FTWReader,
    FTWWriter,
    iter_ftw_weights,
)

MAX_INDEX_INT = (1 << 63) - 1


def _write_index(path, index) -> None:
    with open(path / INDEX_NAME, "w") as f:
        json.dump(index, f)


def _write_raw_index(path, data: bytes) -> None:
    with open(path / INDEX_NAME, "wb") as f:
        f.write(data)


def _two_tensor_checkpoint(tmp_path):
    path = tmp_path / "checkpoint"
    writer = FTWWriter(str(path), shard_limit=ALIGN)
    writer.add_tensor("first", torch.arange(1000, dtype=torch.float32))  # 4000 + padding
    writer.add_tensor("second", torch.arange(4, dtype=torch.int16))  # 8 + padding
    return path, writer.finalize({"model_type": "test"})


def _three_tensor_checkpoint(tmp_path):
    path = tmp_path / "checkpoint"
    writer = FTWWriter(str(path), shard_limit=ALIGN)
    for name in ("first", "middle", "last"):
        writer.add_tensor(name, torch.arange(8, dtype=torch.uint8))
    return path, writer.finalize({})


def _expect_invalid(path, index, match: str) -> None:
    _write_index(path, index)
    with pytest.raises(FTWFormatError, match=match):
        FTWReader(str(path))


def test_valid_writer_output_reads_tensor_spanning_shards(tmp_path):
    path = tmp_path / "checkpoint"
    expected = (torch.arange(5000, dtype=torch.int64) % 251).to(torch.uint8)
    writer = FTWWriter(str(path), shard_limit=ALIGN)
    writer.add_tensor("spanning", expected)
    index = writer.finalize({})

    assert len(index["shards"]) == 2
    reader = FTWReader(str(path))
    reader._direct = 0  # make this portable across tmpfs/overlayfs and CI kernels
    entry = reader.tensors["spanning"]
    pieces = list(reader._pieces(entry["global_off"], entry["nbytes"]))
    assert [piece[3] for piece in pieces] == [ALIGN, 5000 - ALIGN]

    buf = mmap.mmap(-1, 2 * ALIGN)
    dest = memoryview(buf)
    try:
        reader.read_into(dest, entry, workers=1, chunk=ALIGN)
        actual = torch.frombuffer(buf, dtype=torch.uint8, count=5000).clone()
        assert torch.equal(actual, expected)
    finally:
        dest.release()
        buf.close()
        reader.close()


def test_valid_empty_and_zero_tensor_writer_outputs(tmp_path):
    empty_path = tmp_path / "empty"
    empty_index = FTWWriter(str(empty_path), shard_limit=ALIGN).finalize({})
    assert empty_index["shards"] == []
    empty_reader = FTWReader(str(empty_path))
    assert empty_reader.shards == []
    empty_reader.close()

    zero_path = tmp_path / "zero"
    zero_writer = FTWWriter(str(zero_path), shard_limit=ALIGN)
    zero_writer.add_tensor("zero", torch.empty(0, dtype=torch.float32))
    zero_index = zero_writer.finalize({})
    assert zero_index["shards"][0]["nbytes"] == 0
    zero_reader = FTWReader(str(zero_path))
    assert zero_reader.tensors["zero"]["nbytes"] == 0
    zero_reader.close()


def test_valid_zero_tensors_at_writer_allocation_boundaries(tmp_path):
    path = tmp_path / "mixed-zero"
    writer = FTWWriter(str(path), shard_limit=ALIGN)
    writer.add_tensor("zero-before", torch.empty(0, dtype=torch.float32))
    writer.add_tensor("first", torch.arange(1000, dtype=torch.float32))
    writer.add_tensor("zero-between", torch.empty((2, 0, 3), dtype=torch.int16))
    writer.add_tensor("second", torch.arange(4, dtype=torch.int16))
    writer.add_tensor("zero-after", torch.empty(0, dtype=torch.uint8))
    index = writer.finalize({})

    reader = FTWReader(str(path))
    assert [reader.tensors[name]["global_off"] for name in (
        "zero-before", "zero-between", "zero-after",
    )] == [0, ALIGN, 2 * ALIGN]
    assert index["total_bytes"] == 2 * ALIGN
    reader.close()


def test_zero_tensor_can_be_iterated(tmp_path, monkeypatch):
    path = tmp_path / "zero-iteration"
    writer = FTWWriter(str(path), shard_limit=ALIGN)
    writer.add_tensor("zero", torch.empty((2, 0, 3), dtype=torch.float32))
    writer.finalize({})

    class _Progress:
        def update(self, _nbytes):
            pass

        def close(self):
            pass

    progress = types.ModuleType("freetoken.utils.progress")
    progress.byte_bar = lambda *_args, **_kwargs: _Progress()
    monkeypatch.setitem(sys.modules, "freetoken.utils.progress", progress)

    loaded = list(iter_ftw_weights(str(path), workers=1, chunk=ALIGN, prefetch=1))
    assert len(loaded) == 1
    assert loaded[0][0] == "zero"
    assert loaded[0][1].shape == (2, 0, 3)


def test_scalar_tensor_round_trips_with_scalar_shape(tmp_path, monkeypatch):
    path = tmp_path / "scalar-iteration"
    writer = FTWWriter(str(path), shard_limit=ALIGN)
    writer.add_tensor("scalar", torch.tensor(17, dtype=torch.int32))
    writer.finalize({})

    class _Progress:
        def update(self, _nbytes):
            pass

        def close(self):
            pass

    progress = types.ModuleType("freetoken.utils.progress")
    progress.byte_bar = lambda *_args, **_kwargs: _Progress()
    monkeypatch.setitem(sys.modules, "freetoken.utils.progress", progress)

    loaded = list(iter_ftw_weights(str(path), workers=1, chunk=ALIGN, prefetch=1))
    assert loaded[0][0] == "scalar"
    assert loaded[0][1].shape == ()
    assert loaded[0][1].item() == 17


def test_shards_may_be_declared_out_of_order(tmp_path):
    path, index = _two_tensor_checkpoint(tmp_path)
    index["shards"].reverse()
    _write_index(path, index)

    reader = FTWReader(str(path))
    assert [shard["global_off"] for shard in reader.shards] == [0, ALIGN]
    reader.close()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("format", "other", "not a freetoken_weight"),
        ("version", FORMAT_VERSION + 1, "unsupported FTW version"),
        ("version", True, "version must be an integer"),
        ("align", ALIGN * 2, "unsupported FTW alignment"),
        ("shard_limit", ALIGN + 1, "shard_limit must be a multiple"),
        ("shard_limit", 0, "shard_limit must be an integer"),
        ("total_bytes", ALIGN + 1, "total_bytes must be 4096-aligned"),
        ("total_bytes", -1, "total_bytes must be an integer"),
        ("total_bytes", MAX_INDEX_INT + 1, "total_bytes must be an integer"),
    ],
)
def test_rejects_invalid_header_fields(tmp_path, field, value, match):
    path, index = _two_tensor_checkpoint(tmp_path)
    index[field] = value
    _expect_invalid(path, index, match)


def test_rejects_non_object_root_and_missing_fields(tmp_path):
    path, index = _two_tensor_checkpoint(tmp_path)
    _expect_invalid(path, [], "index root must be an object")

    index.pop("tensors")
    _expect_invalid(path, index, "missing required field 'tensors'")


@pytest.mark.parametrize("raw", [b"{", b"\xff"])
def test_rejects_malformed_json_and_encoding(tmp_path, raw):
    path, _index = _two_tensor_checkpoint(tmp_path)
    _write_raw_index(path, raw)
    with pytest.raises(FTWFormatError, match="malformed FTW index"):
        FTWReader(str(path))


def test_rejects_duplicate_json_object_keys(tmp_path):
    path, _index = _two_tensor_checkpoint(tmp_path)
    _write_raw_index(
        path,
        b'{"format":"freetoken_weight","version":1,"version":1}',
    )
    with pytest.raises(FTWFormatError, match="duplicate JSON key 'version'"):
        FTWReader(str(path))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_rejects_non_standard_json_constants_even_in_metadata(tmp_path, constant):
    path, index = _two_tensor_checkpoint(tmp_path)
    raw = json.dumps(index).removesuffix("}") + f', "metadata_value": {constant}}}'
    _write_raw_index(path, raw.encode())
    with pytest.raises(FTWFormatError, match="non-standard JSON constant"):
        FTWReader(str(path))


def test_rejects_json_float_that_overflows_python_float(tmp_path):
    path, index = _two_tensor_checkpoint(tmp_path)
    raw = json.dumps(index).removesuffix("}") + ', "metadata_value": 1e999}'
    _write_raw_index(path, raw.encode())
    with pytest.raises(FTWFormatError, match="floating-point value is out of range"):
        FTWReader(str(path))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("shards", {}, "shards must be an array"),
        ("tensors", {}, "tensors must be an array"),
    ],
)
def test_rejects_non_array_index_collections(tmp_path, field, value, match):
    path, index = _two_tensor_checkpoint(tmp_path)
    index[field] = value
    _expect_invalid(path, index, match)


@pytest.mark.parametrize(
    "file",
    ["../outside.ftw", "/tmp/outside.ftw", "sub/file.ftw", "sub\\file.ftw", "..", "bad\x00.ftw"],
)
def test_rejects_unsafe_shard_names_before_filesystem_access(tmp_path, file):
    path, index = _two_tensor_checkpoint(tmp_path)
    index["shards"][0]["file"] = file
    _expect_invalid(path, index, "safe shard basename")


def test_rejects_duplicate_shard_names(tmp_path):
    path, index = _two_tensor_checkpoint(tmp_path)
    index["shards"][1]["file"] = index["shards"][0]["file"]
    _expect_invalid(path, index, "duplicate shard file")


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        (1, r"shards\[0\] must be an object"),
        ({"global_off": 0, "nbytes": ALIGN}, "missing required field 'file'"),
        ({"file": 1, "global_off": 0, "nbytes": ALIGN}, "safe shard basename"),
    ],
)
def test_rejects_malformed_shard_entries(tmp_path, entry, match):
    path, index = _two_tensor_checkpoint(tmp_path)
    index["shards"][0] = entry
    _expect_invalid(path, index, match)


def test_rejects_missing_truncated_extended_and_non_regular_shards_eagerly(tmp_path):
    path, index = _two_tensor_checkpoint(tmp_path)
    first = path / index["shards"][0]["file"]
    first.unlink()
    _expect_invalid(path, index, "cannot stat FTW shard")

    path, index = _two_tensor_checkpoint(tmp_path / "truncated")
    first = path / index["shards"][0]["file"]
    with open(first, "r+b") as f:
        f.truncate(ALIGN - 1)
    _expect_invalid(path, index, "size mismatch")

    path, index = _two_tensor_checkpoint(tmp_path / "extended")
    first = path / index["shards"][0]["file"]
    with open(first, "ab") as f:
        f.write(b"\x00")
    _expect_invalid(path, index, "size mismatch")

    path, index = _two_tensor_checkpoint(tmp_path / "directory")
    first = path / index["shards"][0]["file"]
    first.unlink()
    first.mkdir()
    _expect_invalid(path, index, "not a regular file")


def test_accepts_symlinked_shard_whose_target_matches(tmp_path):
    # Hugging Face hub snapshots are symlink farms (every file links into blobs/), so a
    # shard entry that resolves through a symlink to a matching regular file must load.
    path, index = _two_tensor_checkpoint(tmp_path)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    first = path / index["shards"][0]["file"]
    target = blobs / "first-blob"
    target.write_bytes(first.read_bytes())
    first.unlink()
    first.symlink_to(target)

    reader = FTWReader(str(path))
    reader._direct = 0
    buf = mmap.mmap(-1, ALIGN)
    dest = memoryview(buf)
    try:
        reader.read_into(dest, reader.tensors["first"], workers=1, chunk=ALIGN)
        actual = torch.frombuffer(buf, dtype=torch.float32, count=1000).clone()
        assert torch.equal(actual, torch.arange(1000, dtype=torch.float32))
    finally:
        dest.release()
        buf.close()
        reader.close()


def test_symlinked_shard_is_checked_against_its_target(tmp_path):
    path, index = _two_tensor_checkpoint(tmp_path)
    external = tmp_path / "outside.ftw"
    external.write_bytes(b"\x00" * (ALIGN + 1))
    first = path / index["shards"][0]["file"]
    first.unlink()
    first.symlink_to(external)

    _expect_invalid(path, index, "size mismatch")


def test_rejects_shard_name_that_is_not_a_filesystem_name(tmp_path):
    # A JSON-escaped lone surrogate decodes to a valid str that os.stat cannot encode;
    # it must surface as FTWFormatError, not UnicodeEncodeError.
    path, index = _two_tensor_checkpoint(tmp_path)
    index["shards"][0]["file"] = "\ud800.ftw"

    _expect_invalid(path, index, "cannot stat FTW shard")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda index: index["shards"][1].update(global_off=3 * ALIGN), "coverage has a gap"),
        (lambda index: index["shards"][1].update(global_off=0), "coverage has an overlap"),
        (lambda index: index.update(total_bytes=3 * ALIGN), "shards cover .* total_bytes"),
        (lambda index: index["shards"][0].update(global_off=1), "global_off must be 4096-aligned"),
        (lambda index: index["shards"][0].update(global_off=MAX_INDEX_INT + 1), "global_off must be an integer"),
        (lambda index: index["shards"][0].update(
            global_off=(MAX_INDEX_INT // ALIGN) * ALIGN,
        ), "byte range exceeds signed 64-bit"),
        (lambda index: index["shards"][0].update(nbytes=2 * ALIGN), "exceeds shard_limit"),
        (lambda index: index["shards"][0].update(nbytes=ALIGN - 1), "nbytes must be 4096-aligned"),
        (lambda index: index["shards"][0].update(nbytes=True), "nbytes must be an integer"),
    ],
)
def test_rejects_invalid_shard_geometry(tmp_path, mutation, match):
    path, index = _two_tensor_checkpoint(tmp_path)
    mutation(index)
    _expect_invalid(path, index, match)


def test_rejects_zero_length_shard_in_nonempty_stream(tmp_path):
    path, index = _two_tensor_checkpoint(tmp_path)
    empty_name = "empty.ftw"
    (path / empty_name).touch()
    index["shards"].insert(0, {"file": empty_name, "global_off": 0, "nbytes": 0})
    _expect_invalid(path, index, "zero-length shards")


def test_rejects_non_writer_zero_byte_shard_forms(tmp_path):
    empty_path = tmp_path / "empty"
    empty_index = FTWWriter(str(empty_path), shard_limit=ALIGN).finalize({})
    (empty_path / "empty.ftw").touch()
    empty_index["shards"] = [{"file": "empty.ftw", "global_off": 0, "nbytes": 0}]
    _expect_invalid(empty_path, empty_index, "empty checkpoint must not contain shard")

    zero_path = tmp_path / "zero"
    zero_writer = FTWWriter(str(zero_path), shard_limit=ALIGN)
    zero_writer.add_tensor("zero", torch.empty(0, dtype=torch.float32))
    zero_index = zero_writer.finalize({})
    zero_index["shards"] = []
    _expect_invalid(zero_path, zero_index, "zero-sized tensor checkpoint must contain")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda index: index["tensors"][1].update(name="first"), "duplicate tensor name"),
        (lambda index: index["tensors"][0].update(name=1), "name must be a string"),
        (lambda index: index["tensors"][0].update(kind=1), "kind must be a string"),
        (lambda index: index["tensors"][0].update(dtype="not_a_dtype"), "not a known torch dtype"),
        (lambda index: index["tensors"][0].update(dtype="strided"), "not a known torch dtype"),
        (lambda index: index["tensors"][0].update(dtype=1), "dtype must be a string"),
        (lambda index: index["tensors"][0].update(shape="1000"), "shape must be an array"),
        (lambda index: index["tensors"][0].update(shape=[True]), r"shape\[0\] must be an integer"),
        (lambda index: index["tensors"][0].update(shape=[1.0]), r"shape\[0\] must be an integer"),
        (lambda index: index["tensors"][0].update(shape=[-1]), r"shape\[0\] must be an integer"),
        (lambda index: index["tensors"][0].update(shape=[(1 << 63) - 1, 2]), "element count exceeds"),
        (lambda index: index["tensors"][0].update(
            shape=[0, MAX_INDEX_INT, MAX_INDEX_INT], nbytes=0,
        ), "shape cannot be reconstructed by torch"),
        (lambda index: index["tensors"][0].update(shape=[MAX_INDEX_INT], dtype="float32"), "byte size exceeds"),
        (lambda index: index["tensors"][0].update(nbytes=4), "shape .* require 4000"),
        (lambda index: index["tensors"][0].update(global_off=1), "global_off must be 4096-aligned"),
        (lambda index: index["tensors"][0].update(
            dtype="uint8", shape=[ALIGN], nbytes=ALIGN,
            global_off=(MAX_INDEX_INT // ALIGN) * ALIGN,
        ), "byte range exceeds signed 64-bit"),
        (lambda index: index["tensors"][1].update(global_off=2 * ALIGN), "exceeds total_bytes"),
        (lambda index: index["tensors"][0].update(nbytes=True), "nbytes must be an integer"),
    ],
)
def test_rejects_invalid_tensor_entries(tmp_path, mutation, match):
    path, index = _two_tensor_checkpoint(tmp_path)
    mutation(index)
    _expect_invalid(path, index, match)


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        (1, r"tensors\[0\] must be an object"),
        ({"kind": "weight", "dtype": "float32", "shape": [1],
          "global_off": 0, "nbytes": 4}, "missing required field 'name'"),
    ],
)
def test_rejects_malformed_tensor_entries(tmp_path, entry, match):
    path, index = _two_tensor_checkpoint(tmp_path)
    index["tensors"][0] = entry
    _expect_invalid(path, index, match)


def test_rejects_dtype_that_torch_cannot_reconstruct(tmp_path, monkeypatch):
    path, index = _two_tensor_checkpoint(tmp_path)
    original = torch.frombuffer

    def reject_float32(buffer, *, dtype, count):
        if dtype == torch.float32:
            raise RuntimeError("unsupported storage dtype")
        return original(buffer, dtype=dtype, count=count)

    monkeypatch.setattr(torch, "frombuffer", reject_float32)
    _expect_invalid(path, index, "dtype is not readable FTW storage")


def test_rejects_tensor_overlap_including_padded_allocation(tmp_path):
    path, index = _two_tensor_checkpoint(tmp_path)
    index["tensors"][1]["global_off"] = 0
    _expect_invalid(path, index, "overlaps padded allocation")


@pytest.mark.parametrize(
    ("remove", "match"),
    [
        (0, "tensor coverage has a gap"),
        (1, "tensor coverage has a gap"),
        (2, "tensors cover .* total_bytes"),
    ],
)
def test_rejects_leading_internal_and_trailing_tensor_gaps(tmp_path, remove, match):
    path, index = _three_tensor_checkpoint(tmp_path)
    index["tensors"].pop(remove)
    _expect_invalid(path, index, match)


def test_rejects_zero_sized_tensor_inside_nonzero_allocation(tmp_path):
    path = tmp_path / "checkpoint"
    writer = FTWWriter(str(path), shard_limit=ALIGN)
    writer.add_tensor("spanning", torch.arange(5000, dtype=torch.uint8))
    index = writer.finalize({})
    index["tensors"].append({
        "name": "zero-inside",
        "kind": "weight",
        "dtype": "float32",
        "shape": [0],
        "global_off": ALIGN,
        "nbytes": 0,
    })
    _expect_invalid(path, index, "zero-sized tensor .* is not on a tensor boundary")


def test_index_validation_does_not_mutate_tensor_order(tmp_path):
    path, index = _two_tensor_checkpoint(tmp_path)
    index["tensors"].reverse()
    expected = copy.deepcopy(index["tensors"])
    _write_index(path, index)

    reader = FTWReader(str(path))
    assert reader.index["tensors"] == expected
    assert [entry["name"] for entry in reader.entries()] == ["second", "first"]
    reader.close()


def test_writer_emits_declared_v1_header(tmp_path):
    path, index = _two_tensor_checkpoint(tmp_path)
    assert index["format"] == FORMAT_TAG
    assert index["version"] == FORMAT_VERSION
    assert index["align"] == ALIGN
    assert sum(os.path.getsize(path / shard["file"]) for shard in index["shards"]) == index["total_bytes"]


@pytest.mark.parametrize("shard_limit", [0, ALIGN - 1, ALIGN + 1, True, MAX_INDEX_INT + 1])
def test_writer_rejects_invalid_shard_limit_without_assertions(tmp_path, shard_limit):
    with pytest.raises(ValueError, match="shard_limit must be an integer multiple"):
        FTWWriter(str(tmp_path / "checkpoint"), shard_limit=shard_limit)


def test_writer_does_not_emit_indexes_its_reader_would_reject(tmp_path):
    path = tmp_path / "checkpoint"
    writer = FTWWriter(str(path), shard_limit=ALIGN)
    writer.add_tensor("tensor", torch.arange(4, dtype=torch.float32))

    with pytest.raises(ValueError, match="duplicate FTW tensor name"):
        writer.add_tensor("tensor", torch.arange(4, dtype=torch.float32))
    with pytest.raises(ValueError, match="cannot override index fields"):
        writer.finalize({"version": FORMAT_VERSION})
    with pytest.raises(ValueError, match="must not contain NaN or infinity"):
        writer.finalize({"metadata_value": float("nan")})
    with pytest.raises(ValueError, match="keys must be strings"):
        writer.finalize({"nested": {1: "ambiguous with a JSON string key"}})
    with pytest.raises(ValueError, match="JSON-compatible values"):
        writer.finalize({"metadata_value": object()})

    writer.finalize({"metadata_value": 1.5})
    FTWReader(str(path)).close()


def test_validation_remains_active_under_optimized_python(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = r'''
import json
import logging
from pathlib import Path
import sys
import types

import torch

utils = types.ModuleType("freetoken.utils")
utils.init_logger = logging.getLogger
sys.modules["freetoken.utils"] = utils

from freetoken.checkpoint.ftw import FTWFormatError, FTWReader, FTWWriter, INDEX_NAME, load_ftw_banks

moe = types.ModuleType("freetoken.moe")
moe.__path__ = []
host_banks = types.ModuleType("freetoken.moe.host_banks")
host_banks.HostBank = object
host_banks.HostResidency = types.SimpleNamespace(PINNED=types.SimpleNamespace(value="pinned"))
host_banks.PinPipeline = object
host_banks.alloc_banks = lambda _specs: {}
host_banks.born_pinned_default = lambda: False
progress = types.ModuleType("freetoken.utils.progress")
progress.byte_bar = lambda *_args, **_kwargs: None
sys.modules["freetoken.moe"] = moe
sys.modules["freetoken.moe.host_banks"] = host_banks
sys.modules["freetoken.utils.progress"] = progress

path = Path(sys.argv[1]) / "optimized"
path.mkdir()
(path / INDEX_NAME).write_text(json.dumps({"format": "wrong"}))

try:
    FTWReader(str(path))
except FTWFormatError:
    pass
else:
    raise SystemExit("malformed index was accepted under -O")

for action in (
    lambda: FTWWriter(str(path / "writer"), shard_limit=0),
    lambda: load_ftw_banks(str(path), num_layers=0),
    lambda: load_ftw_banks(str(path), num_layers=1, layer_residency=[]),
):
    try:
        action()
    except ValueError:
        pass
    else:
        raise SystemExit("invalid writer/reader argument was accepted under -O")

schema_cases = []

mixed = path / "mixed"
writer = FTWWriter(str(mixed), shard_limit=4096)
writer.add_tensor("bank", torch.ones(1), kind="experts_bank")
writer.add_tensor("bank#L00000", torch.ones(1), kind="experts_bank")
writer.finalize({})
schema_cases.append((mixed, 1))

nondivisible = path / "nondivisible"
writer = FTWWriter(str(nondivisible), shard_limit=4096)
writer.add_tensor("bank", torch.ones(3), kind="experts_bank")
writer.finalize({})
schema_cases.append((nondivisible, 2))

missing_layer = path / "missing-layer"
writer = FTWWriter(str(missing_layer), shard_limit=4096)
writer.add_tensor("bank#L00000", torch.ones(1), kind="experts_bank")
writer.finalize({})
schema_cases.append((missing_layer, 2))

for checkpoint, layers in schema_cases:
    try:
        load_ftw_banks(str(checkpoint), num_layers=layers)
    except FTWFormatError:
        pass
    else:
        raise SystemExit("malformed expert-bank schema was accepted under -O")
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [
        str(repo_root / "python"), env.get("PYTHONPATH", ""),
    ]))
    result = subprocess.run(
        [sys.executable, "-O", "-c", script, str(tmp_path)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
