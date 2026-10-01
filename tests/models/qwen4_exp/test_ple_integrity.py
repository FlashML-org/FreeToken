"""PLE n-gram table integrity: the aggregate shard schema and the config-derived geometry are checked, by name, before either backend touches a row.

The fixture models the shipping checkpoint: ``shard_<i>`` F8_E4M3 blocks of equal shape spread over several ``model-*.safetensors`` files, one BF16 ``weight_scale`` in one of them, and a ``model.safetensors.index.json`` naming them. The geometry is the toy config's own (``ngram_vocab_size_base`` 1000 -> primes 1009+1013+1019+1021 = 4062, padded to 8 -> 4064 rows over 4 parts of 1016), so the exact-row-count gate is exercised for real, not with a table sized to whatever the loader accepts.
"""

from __future__ import annotations

import json
import os
import random
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from freetoken.models.qwen4_exp import weight
from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.ple_disk import source_from_safetensors
from freetoken.models.qwen4_exp.weight import (
    check_ple_geometry,
    expected_ple_rows,
    load_ple_table,
    ple_table_layout,
)

from .common import toy_hf_config

LAYER = 1  # toy config ple_layer_ids [2] is 1-based -> the checkpoint keys say layers.1
PREFIX = f"model.language_model.layers.{LAYER}.ple.ple_embedding.ngram_embedding"
ROWS, COLS, PARTS = 1016, 16, 4  # toy geometry: 4064 rows = 4 x 1016, ple_embed_dim 64 / 4 heads


def _args():
    return parse_config(toy_hf_config()).qwen4_args


def _st_bytes(entries: list[tuple[str, str, list[int], bytes]]) -> bytes:
    """A safetensors file from ``(name, dtype, shape, payload)`` entries; the header says whatever it is told."""
    header: dict = {}
    payload, offset = [], 0
    for name, dtype, shape, raw in entries:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        payload.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + b"".join(payload)


def _bf16(value: float) -> bytes:
    return struct.pack("<H", torch.tensor(value, dtype=torch.bfloat16).view(torch.int16).item() & 0xFFFF)


def _write_table(
    folder: Path, *, rows: int = ROWS, cols: int = COLS, parts: int = PARTS, files: int = 2,
    scale: float = 0.125, layer: int = LAYER, index: bool = True,
) -> dict[str, bytes]:
    """Write the toy table; returns ``{tensor name: payload bytes}``."""
    prefix = f"model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding"
    raw = {f"{prefix}.shard_{i}.weight": random.Random(i).randbytes(rows * cols) for i in range(parts)}
    per_file: list[list[tuple[str, str, list[int], bytes]]] = [[] for _ in range(files)]
    for i, (name, data) in enumerate(raw.items()):
        per_file[i % files].append((name, "F8_E4M3", [rows, cols], data))
    per_file[-1].append((f"{prefix}.weight_scale", "BF16", [], _bf16(scale)))
    weight_map = {}
    for n, entries in enumerate(per_file):
        filename = f"model-{n:05d}-of-{files:05d}.safetensors"
        (folder / filename).write_bytes(_st_bytes(entries))
        weight_map.update({name: filename for name, *_ in entries})
    if index:
        (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return raw


def _file_of(folder: Path, name: str) -> Path:
    weight_map = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    return folder / weight_map[name]


def _rewrite_header(path: Path, edit) -> None:
    """Re-serialise ``path``'s header after ``edit(header_dict)``; the payload bytes stay where they are."""
    data = path.read_bytes()
    n = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + n])
    edit(header)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data[8 + n :])


@pytest.fixture
def no_bank(monkeypatch):
    """Any refusal below must happen before a HostBank exists."""
    def refuse(*_args, **_kwargs):
        raise AssertionError("HostBank allocated before the table was admitted")

    monkeypatch.setattr(weight, "HostBank", refuse)


# ======================================================================================
# 1. aggregate schema + exact geometry
# ======================================================================================


def test_layout_reproduces_the_config_geometry(tmp_path):
    args = _args()
    raw = _write_table(tmp_path)
    layout = ple_table_layout(str(tmp_path))
    assert (len(layout.parts), layout.rows_per_part, layout.cols, layout.layer) == (PARTS, ROWS, COLS, LAYER)
    assert [p.index for p in layout.parts] == list(range(PARTS))
    assert layout.total_rows == expected_ple_rows(args) == 4064
    assert layout.total_bytes == 4064 * COLS
    assert layout.scale.dtype is torch.bfloat16 and float(layout.scale) == 0.125
    check_ple_geometry(layout, args)

    source = source_from_safetensors(str(tmp_path), args)
    assert (source.total_rows, source.row_bytes, source.rows_per_extent, source.scale) == (4064, COLS, ROWS, 0.125)
    assert len(source.paths) == 2 and len(source.extent_base) == PARTS

    table = load_ple_table(str(tmp_path), args, pin=False)
    assert table.tensor.shape == (4064, COLS) and table.tensor.dtype is torch.float8_e4m3fn
    for part in layout.parts:
        got = table.tensor[part.index * ROWS : (part.index + 1) * ROWS].view(torch.uint8).numpy().tobytes()
        assert got == raw[part.name]


def test_expected_rows_reproduces_the_production_table():
    """Qwen3.8-Flash-Next: 16 primes after 19,999,999 padded to 128 = 320,001,536 rows = 128 x 2,500,012."""
    args = SimpleNamespace(
        ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=20_000_000,
        make_ngram_vocab_size_divisible_by=128, split_ngram_parts=128,
    )
    assert expected_ple_rows(args) == 320_001_536
    assert expected_ple_rows(args) % 128 == 0 and expected_ple_rows(args) // 128 == 2_500_012


@pytest.mark.parametrize(
    "case, kwargs, message",
    [
        ("one row too many per part", dict(rows=ROWS + 1), r"PLE table has 4068 rows \(4 x 1017\); config geometry requires 4064"),
        ("one row too few per part", dict(rows=ROWS - 1), r"PLE table has 4060 rows \(4 x 1015\); config geometry requires 4064"),
        ("one part short", dict(parts=PARTS - 1), r"PLE table needs shards 0\.\.3, found 3"),
        ("narrow rows", dict(cols=COLS // 2), r"PLE table row is 8 wide, config says 16"),
        ("another layer", dict(layer=LAYER + 1), r"PLE table tensors are for layer 2, config ple_layer_ids is \[1\]"),
    ],
)
def test_geometry_must_equal_the_config_before_allocation(tmp_path, no_bank, case, kwargs, message):
    args = _args()
    _write_table(tmp_path, **kwargs)
    layout = ple_table_layout(str(tmp_path))  # the shards agree with each other ...
    with pytest.raises(ValueError, match=message):
        check_ple_geometry(layout, args)  # ... but not with the config
    with pytest.raises(ValueError, match=message):
        source_from_safetensors(str(tmp_path), args)
    with pytest.raises(ValueError, match=message):
        load_ple_table(str(tmp_path), args, pin=False)


def _set(name: str, field: str, value):
    def edit(header):
        header[name][field] = value

    return edit


def _rename(old: str, new: str):
    def edit(header):
        header[new] = header.pop(old)

    return edit


@pytest.mark.parametrize(
    "case, target, edit, message",
    [
        ("part dtype", 1, _set(f"{PREFIX}.shard_1.weight", "dtype", "F8_E5M2"),
         r"PLE shard .*shard_1\.weight in model-00001-of-00002\.safetensors has dtype F8_E5M2, expected F8_E4M3"),
        ("part shape disagrees", 1, _set(f"{PREFIX}.shard_1.weight", "shape", [ROWS + 1, COLS]),
         r"shard_1\.weight in model-00001-of-00002\.safetensors is \[1017, 16\], expected \[1016, 16\] like .*shard_0\.weight"),
        ("first part shape vs bytes", 0, _set(f"{PREFIX}.shard_0.weight", "shape", [ROWS + 1, COLS]),
         r"shard_0\.weight in model-00000-of-00002\.safetensors spans 16256 bytes; shape \[1017, 16\] x 1 byte needs 16272"),
        ("part shape 1-D", 0, _set(f"{PREFIX}.shard_0.weight", "shape", [ROWS * COLS]),
         r"shard_0\.weight in model-00000-of-00002\.safetensors has shape \[16256\], expected a positive 2-D \[rows, cols\]"),
        ("part offsets short", 0, _set(f"{PREFIX}.shard_0.weight", "data_offsets", [0, ROWS * COLS - 1]),
         r"shard_0\.weight in model-00000-of-00002\.safetensors spans 16255 bytes; shape \[1016, 16\] x 1 byte needs 16256"),
        ("part offsets past the payload", 1, _set(f"{PREFIX}.shard_1.weight", "data_offsets", [ROWS * COLS + 3, 2 * ROWS * COLS + 3]),
         r"shard_1\.weight in model-00001-of-00002\.safetensors data_offsets \[16259, 32515\] run past the 32514-byte payload"),
        ("scale dtype", 1, _set(f"{PREFIX}.weight_scale", "dtype", "F16"),
         r"PLE weight_scale .*weight_scale in model-00001-of-00002\.safetensors must be one BF16 scalar, got dtype F16, shape \[\], 2 bytes"),
        ("scale shape", 1, _set(f"{PREFIX}.weight_scale", "shape", [2]),
         r"must be one BF16 scalar, got dtype BF16, shape \[2\], 2 bytes"),
        ("part on another layer", 1, _rename(f"{PREFIX}.shard_1.weight", f"{PREFIX.replace(f'layers.{LAYER}', 'layers.2')}.shard_1.weight"),
         r"PLE tensor .*layers\.2\.ple.*shard_1\.weight in model-00001-of-00002\.safetensors is for layer 2; .*layers\.1\.ple.*shard_0\.weight in model-00000-of-00002\.safetensors is for layer 1"),
        ("duplicate part", 1, _rename(f"{PREFIX}.shard_1.weight", f"{PREFIX}.shard_0.weight"),
         r"duplicate PLE shard 0: .*shard_0\.weight in model-00000-of-00002\.safetensors and .*shard_0\.weight in model-00001-of-00002\.safetensors"),
        ("part index gap", 1, _rename(f"{PREFIX}.shard_3.weight", f"{PREFIX}.shard_7.weight"),
         r"PLE shard indices are not contiguous 0\.\.N-1: \[0, 1, 2, 7\]"),
        ("second scale", 0, lambda h: h.__setitem__(f"{PREFIX}.weight_scale", {"dtype": "BF16", "shape": [], "data_offsets": [0, 2]}),
         r"PLE table has two weight_scale tensors: .*weight_scale in model-00000-of-00002\.safetensors and .*weight_scale in model-00001-of-00002\.safetensors"),
    ],
)
def test_parts_must_agree_on_dtype_shape_offsets_scale_and_layer(tmp_path, no_bank, case, target, edit, message):
    _write_table(tmp_path)
    _rewrite_header(_file_of(tmp_path, f"{PREFIX}.shard_{target}.weight"), edit)
    with pytest.raises(ValueError, match=message):
        ple_table_layout(str(tmp_path))
    with pytest.raises(ValueError, match=message):
        source_from_safetensors(str(tmp_path))  # the disk backend's entry point, no config
    with pytest.raises(ValueError, match=message):
        load_ple_table(str(tmp_path), _args(), pin=False)


@pytest.mark.parametrize("value, message", [(0.0, "got 0.0"), (float("nan"), "got nan"), (-0.5, "got -0.5")])
def test_scale_must_be_finite_and_positive(tmp_path, no_bank, value, message):
    _write_table(tmp_path, scale=value)
    with pytest.raises(ValueError, match=rf"PLE weight_scale .*weight_scale in model-00001-of-00002\.safetensors must be finite and positive, {message}"):
        ple_table_layout(str(tmp_path))


def test_indexless_folder_is_discovered_from_the_shard_headers(tmp_path):
    raw = _write_table(tmp_path, index=False)
    layout = ple_table_layout(str(tmp_path))
    assert layout.total_rows == 4064
    assert {os.path.basename(p.path) for p in layout.parts} == {"model-00000-of-00002.safetensors", "model-00001-of-00002.safetensors"}
    assert len(raw) == PARTS


# ======================================================================================
# 2. discovery: nofollow shard opens, bounded header reads
# ======================================================================================


def test_hf_cache_symlink_layout_resolves_once_and_then_opens_nofollow(tmp_path):
    """The HF cache is symlinks into blobs/: discovery canonicalises each shard path exactly once; every later open is O_NOFOLLOW on the target."""
    snapshot, blobs = tmp_path / "snapshot", tmp_path / "blobs"
    snapshot.mkdir()
    blobs.mkdir()
    raw = _write_table(snapshot)
    for shard in sorted(snapshot.glob("*.safetensors")):
        target = blobs / f"blob-{shard.name}"
        shard.rename(target)
        shard.symlink_to(target)
    layout = ple_table_layout(str(snapshot))
    assert layout.total_rows == 4064
    assert {p.path for p in layout.parts} == {str(blobs / f"blob-model-{n:05d}-of-00002.safetensors") for n in range(2)}
    table = load_ple_table(str(snapshot), _args(), pin=False)
    assert table.tensor[:ROWS].view(torch.uint8).numpy().tobytes() == raw[f"{PREFIX}.shard_0.weight"]


def test_symlink_and_non_regular_shard_paths_are_refused_by_name(tmp_path):
    _write_table(tmp_path)
    shard = tmp_path / "model-00000-of-00002.safetensors"
    link = tmp_path / "link.safetensors"
    link.symlink_to(shard)
    with pytest.raises(ValueError, match=rf"PLE shard {link} is a symlink; refusing to follow it"):
        weight._safetensors_header(str(link))
    fifo = tmp_path / "fifo.safetensors"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match=rf"PLE shard {fifo} is not a regular file"):
        weight._safetensors_header(str(fifo))
    with pytest.raises(ValueError, match=rf"PLE shard {tmp_path} is not a regular file"):
        weight._safetensors_header(str(tmp_path))
    dangling = tmp_path / "model-00001-of-00002.safetensors"
    dangling.unlink()
    dangling.symlink_to(tmp_path / "gone")
    with pytest.raises(ValueError, match=rf"cannot resolve PLE shard {dangling}: No such file or directory"):
        ple_table_layout(str(tmp_path))


@pytest.mark.parametrize(
    "case, header_len, message",
    [
        ("one byte over budget", (64 << 20) + 1,
         r"declares a 67108865-byte safetensors header; the budget is 67108864 bytes"),
        ("u64 max", (1 << 64) - 1,
         r"declares a 18446744073709551615-byte safetensors header; the budget is 67108864 bytes"),
        ("past the end of the file", None,
         r"declares a {n}-byte safetensors header that runs past the end of the {size}-byte file"),
    ],
)
def test_header_length_is_bounded_before_it_is_read(tmp_path, no_bank, case, header_len, message):
    _write_table(tmp_path)
    shard = tmp_path / "model-00000-of-00002.safetensors"
    size = shard.stat().st_size
    if header_len is None:
        header_len = size - 7  # in budget, one byte past the file
    message = message.format(n=header_len, size=size)
    with shard.open("r+b") as fh:
        fh.write(struct.pack("<Q", header_len))
    with pytest.raises(ValueError, match=rf"PLE shard {shard} {message}"):
        ple_table_layout(str(tmp_path))
    with pytest.raises(ValueError, match=message):
        load_ple_table(str(tmp_path), _args(), pin=False)
    assert shard.stat().st_size == size  # nothing was read past the declared length, nothing written


@pytest.mark.parametrize(
    "case, edit, message",
    [
        ("length +1 runs into the payload", "plus_one", r"safetensors header is not UTF-8 JSON: 'utf-8' codec can't decode"),
        ("not JSON", b"{not json", r"safetensors header is not UTF-8 JSON: Expecting property name"),
        ("JSON array", b"[]", r"safetensors header is a JSON list, expected an object"),
        ("six-byte file", b"", r"is 6 bytes; not a safetensors file"),
    ],
)
def test_header_must_be_a_utf8_json_object(tmp_path, no_bank, case, edit, message):
    _write_table(tmp_path)
    shard = tmp_path / "model-00000-of-00002.safetensors"
    data = shard.read_bytes()
    if edit == "plus_one":
        n = struct.unpack("<Q", data[:8])[0]
        shard.write_bytes(struct.pack("<Q", n + 1) + data[8:])
    elif edit == b"":
        shard.write_bytes(b"\x00" * 6)
    else:
        padded = edit + b" " * (-len(edit) % 8)
        shard.write_bytes(struct.pack("<Q", len(padded)) + padded + data[8 + struct.unpack("<Q", data[:8])[0] :])
    with pytest.raises(ValueError, match=rf"PLE shard {shard} {message}"):
        ple_table_layout(str(tmp_path))


# ======================================================================================
# 3. mutation evidence: stat gate + header SHA-256 between preflight and the payload reads
# ======================================================================================


def _flip_dtype_in_place(shard: Path) -> None:
    """Same inode, same size, mtime restored: only ctime and the header bytes betray the edit."""
    before = shard.stat()
    data = shard.read_bytes()
    n = struct.unpack("<Q", data[:8])[0]
    at = data.index(b"F8_E4M3", 8)
    assert at < 8 + n
    with shard.open("r+b") as fh:
        fh.seek(at)
        fh.write(b"F8_E5M2")
    os.utime(shard, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = shard.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (before.st_ino, before.st_size, before.st_mtime_ns)


def test_layout_records_each_shard_identity_and_header_digest(tmp_path):
    import hashlib

    _write_table(tmp_path)
    layout = ple_table_layout(str(tmp_path))
    assert [os.path.basename(f.path) for f in layout.files] == ["model-00000-of-00002.safetensors", "model-00001-of-00002.safetensors"]
    for identity in layout.files:
        st = os.stat(identity.path)
        assert (identity.st_dev, identity.st_ino, identity.st_size, identity.st_mtime_ns, identity.st_ctime_ns) == (
            st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        data = Path(identity.path).read_bytes()
        assert identity.header_base == 8 + struct.unpack("<Q", data[:8])[0]
        assert identity.header_sha256 == hashlib.sha256(data[8 : identity.header_base]).hexdigest()
        weight.revalidate_ple_shard(identity)  # untouched: passes
    source = source_from_safetensors(str(tmp_path))
    assert source.files == layout.files


@pytest.mark.parametrize(
    "case, mutate, message",
    [
        ("same-inode header edit, mtime restored", _flip_dtype_in_place, r"changed after preflight: st_ctime_ns \d+ -> \d+"),
        ("one byte appended", lambda p: p.open("ab").write(b"\0"), r"changed after preflight: st_size (\d+) -> \d+"),
        ("replaced by a copy", lambda p: (p.with_suffix(".new").write_bytes(p.read_bytes()), os.replace(p.with_suffix(".new"), p)),
         r"changed after preflight: st_ino \d+ -> \d+"),
        ("swapped for a symlink", lambda p: (p.rename(p.with_suffix(".blob")), p.symlink_to(p.with_suffix(".blob"))),
         r"is a symlink; refusing to follow it"),
    ],
)
def test_shard_changed_between_preflight_and_read_is_refused_with_the_reason(tmp_path, case, mutate, message):
    _write_table(tmp_path)
    layout = ple_table_layout(str(tmp_path))
    shard = Path(layout.files[1].path)
    mutate(shard)
    with pytest.raises(ValueError, match=rf"PLE shard {shard} {message}"):
        weight.revalidate_ple_shard(layout.files[1])
    weight.revalidate_ple_shard(layout.files[0])  # the other shard is still what preflight saw


def test_header_digest_catches_an_edit_the_stat_gate_cannot_see(tmp_path):
    """A writer that also restores ctime (privileged, or a coarse filesystem) still changes the bytes preflight hashed."""
    import dataclasses

    _write_table(tmp_path)
    layout = ple_table_layout(str(tmp_path))
    identity = layout.files[0]
    _flip_dtype_in_place(Path(identity.path))
    forged = dataclasses.replace(identity, st_ctime_ns=os.stat(identity.path).st_ctime_ns)
    with pytest.raises(ValueError, match=rf"PLE shard {identity.path} header changed after preflight: sha256 {identity.header_sha256[:16]}\S* -> [0-9a-f]{{16}}"):
        weight.revalidate_ple_shard(forged)


def test_pinned_load_revalidates_each_shard_before_its_first_read(tmp_path, monkeypatch):
    _write_table(tmp_path)
    layout = ple_table_layout(str(tmp_path))
    other = Path(layout.files[1].path)  # parts 1 and 3
    reads = []
    real = weight.read_range_into

    def read_then_mutate(buf, path, **kwargs):
        reads.append(os.path.basename(path))
        if len(reads) == 1:
            _flip_dtype_in_place(other)  # while part 0 is being read, model-00001 changes under us
        return real(buf, path, **kwargs)

    monkeypatch.setattr(weight, "read_range_into", read_then_mutate)
    with pytest.raises(ValueError, match=rf"PLE shard {other} changed after preflight: st_ctime_ns"):
        load_ple_table(str(tmp_path), _args(), pin=False)
    assert reads == ["model-00000-of-00002.safetensors"]  # part 1's read never happened


def test_pinned_load_refuses_a_shard_that_changed_while_it_was_being_read(tmp_path, monkeypatch):
    _write_table(tmp_path)
    layout = ple_table_layout(str(tmp_path))
    first = Path(layout.files[0].path)
    reads = []
    real = weight.read_range_into

    def read_then_mutate(buf, path, **kwargs):
        reads.append(os.path.basename(path))
        out = real(buf, path, **kwargs)
        if len(reads) == 4:
            first.open("ab").write(b"\0")  # after the last read, model-00000 grows
        return out

    monkeypatch.setattr(weight, "read_range_into", read_then_mutate)
    with pytest.raises(ValueError, match=rf"PLE shard {first} changed while reading: st_size \d+ -> \d+"):
        load_ple_table(str(tmp_path), _args(), pin=False)
    assert len(reads) == 4


def test_disk_source_is_revalidated_before_the_store_opens_its_descriptors(tmp_path):
    from freetoken.models.qwen4_exp.ple_disk import DiskRowTable

    from .common import EOS, hash_constants

    _write_table(tmp_path)
    args = _args()
    source = source_from_safetensors(str(tmp_path), args)
    multipliers, sizes, offsets = hash_constants(args)
    constants = {
        "num_ngram_heads": args.num_ngram_heads, "layer_multipliers": multipliers.tolist(),
        "per_head_vocab_sizes": sizes.tolist(), "per_head_offsets": offsets.tolist(), "eos_token_id": EOS,
    }
    shard = Path(source.files[0].path)
    _flip_dtype_in_place(shard)
    with pytest.raises(ValueError, match=rf"PLE shard {shard} changed after preflight: st_ctime_ns"):
        DiskRowTable(source, constants)


# ======================================================================================
# 4. pinned backend: effective-memory admission before the bank exists; rollback after
# ======================================================================================


def _live_mmaps():
    from freetoken.moe import host_banks

    return [m for m in host_banks._LIVE_BUFFERS if not m.closed]


def test_effective_memory_is_the_tighter_of_meminfo_and_the_cgroup(tmp_path):
    meminfo = tmp_path / "meminfo"
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    meminfo.write_text("MemTotal:       65536000 kB\nMemFree:        1000 kB\nMemAvailable:   60000000 kB\n")
    (cgroup / "memory.max").write_text("max\n")
    (cgroup / "memory.current").write_text("1000\n")
    assert weight.effective_memory_available(meminfo=str(meminfo), cgroup=str(cgroup)) == 60_000_000 * 1024
    (cgroup / "memory.max").write_text("50000000000\n")
    (cgroup / "memory.current").write_text("10000000000\n")
    assert weight.effective_memory_available(meminfo=str(meminfo), cgroup=str(cgroup)) == 40_000_000_000
    (cgroup / "memory.current").write_text("60000000000\n")  # over its limit: a known zero, not "unknown"
    assert weight.effective_memory_available(meminfo=str(meminfo), cgroup=str(cgroup)) == 0
    meminfo.write_text("MemTotal: 1 kB\n")  # no MemAvailable, cgroup only
    (cgroup / "memory.current").write_text("10000000000\n")
    assert weight.effective_memory_available(meminfo=str(meminfo), cgroup=str(cgroup)) == 40_000_000_000
    assert weight.effective_memory_available(meminfo=str(tmp_path / "absent"), cgroup=str(tmp_path / "absent")) is None
    assert weight.effective_memory_available() == weight.effective_memory_available("/proc/meminfo", "/sys/fs/cgroup")


@pytest.mark.parametrize(
    "available, admitted",
    [
        (65_024 * 100 // 88, True),      # exactly the 12% headroom: 65,024 <= 73,890 * 88 // 100 = 65,023 -> refused
        (65_024 * 100 // 88 + 2, True),  # one byte of headroom to spare
        (65_024, False),                 # the table fits in RAM but not with headroom
        (0, False),                      # a known zero is a hard refusal
        (None, True),                    # unknown: best effort, admit
    ],
)
def test_pinned_bank_is_admitted_by_effective_memory_before_it_is_allocated(tmp_path, monkeypatch, available, admitted):
    _write_table(tmp_path)
    monkeypatch.setattr(weight, "effective_memory_available", lambda: available)
    if available == 65_024 * 100 // 88:
        admitted = False  # 73,890 * 88 // 100 = 65,023 < 65,024
    if admitted:
        table = load_ple_table(str(tmp_path), _args(), pin=False)
        assert table.bank.nbytes == 65_024
        return
    ceiling = available * 88 // 100
    allocations = []
    real_bank = weight.HostBank

    class Bank(real_bank):
        def __init__(self, *a, **k):
            allocations.append(a)
            super().__init__(*a, **k)

    monkeypatch.setattr(weight, "HostBank", Bank)
    with pytest.raises(ValueError, match=rf"PLE table needs 65024 bytes of host memory for the pinned bank; effective memory available is {available} bytes, {ceiling} after the 12% headroom"):
        load_ple_table(str(tmp_path), _args(), pin=False)
    assert allocations == []


def test_host_bank_discard_frees_the_mapping_and_forgets_it():
    from freetoken.moe.host_banks import HostBank

    before = len(_live_mmaps())
    bank = HostBank((1024, 16), torch.float8_e4m3fn)
    assert len(_live_mmaps()) == before + 1
    view = bank.memoryview()
    view.release()
    bank.discard()
    assert len(_live_mmaps()) == before
    assert bank.tensor is None and bank.nbytes == 0
    bank.discard()  # idempotent


@pytest.mark.parametrize(
    "case, failure, message",
    [
        ("a read fails", "read", r"disk went away"),
        ("a shard changed under the read", "mutate", r"changed after preflight: st_ctime_ns"),
        ("the pin fails", "pin", r"cudaHostRegister failed for 0\.0 GiB"),
    ],
)
def test_failure_after_allocation_discards_the_bank_and_publishes_nothing(tmp_path, monkeypatch, case, failure, message):
    from freetoken.moe.host_banks import HostBank, PinFailed

    _write_table(tmp_path)
    layout = ple_table_layout(str(tmp_path))
    banks = []
    real_bank = weight.HostBank
    real_read = weight.read_range_into

    class Bank(real_bank):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            banks.append(self)

    def read(buf, path, **kwargs):
        if failure == "read":
            raise OSError("disk went away")
        if failure == "mutate" and not getattr(read, "done", False):
            read.done = True
            _flip_dtype_in_place(Path(layout.files[1].path))
        return real_read(buf, path, **kwargs)

    monkeypatch.setattr(weight, "HostBank", Bank)
    monkeypatch.setattr(weight, "read_range_into", read)
    if failure == "pin":
        monkeypatch.setattr(weight.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(HostBank, "pin", lambda self: (_ for _ in ()).throw(PinFailed("cudaHostRegister failed for 0.0 GiB")))
    live = len(_live_mmaps())
    with pytest.raises((ValueError, OSError, PinFailed), match=message):
        load_ple_table(str(tmp_path), _args(), pin=True)
    assert len(banks) == 1
    assert banks[0].tensor is None and banks[0].nbytes == 0  # discarded ...
    assert len(_live_mmaps()) == live  # ... and its mapping is closed and forgotten
