import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from freetoken.models.deepseek_v41.engram import (
    DiskEngramTable, Engram, EngramRuntime, HashLayout, compressed_token_map,
    export_engram_tables, hash_token_run, prepare_engram,
)


def _args():
    return SimpleNamespace(
        engram_layer_ids=(1, 14), engram_max_ngram_size=3, engram_n_heads=2,
        engram_vocab_size=10, engram_num_embeddings=(60, 120),
        engram_compressed_vocab_size=7, engram_head_dim=32, engram_pad_id=2,
        engram_dtype="fp8", engram_block_size=32, engram_scale_fmt="ue8m0",
    )


def _scalar_hash(ids, images, mapping, layout, pad):
    output = []
    for position in range(len(ids)):
        row = []
        for layer in range(len(layout.layer_ids)):
            values, blocked = [], False
            for shift in range(layout.max_ngram_size):
                source = position - shift
                blocked |= source < 0 or (source >= 0 and images[source])
                values.append(int(mapping[pad if blocked else ids[source]]))
            running = values[0] * int(layout.multipliers[layer, 0])
            per_layer = []
            for shift in range(1, layout.max_ngram_size):
                running ^= values[shift] * int(layout.multipliers[layer, shift])
                for head in range(layout.heads):
                    column = (shift - 1) * layout.heads + head
                    per_layer.append(running % int(layout.primes[layer, column]) + int(layout.offsets[layer, column]))
            row.append(per_layer)
        output.append(row)
    return np.asarray(output)


def test_hash_matches_scalar_and_chunk_boundary():
    args = _args()
    layout = HashLayout.from_args(args)
    mapping = np.arange(7, dtype=np.int64)
    ids = np.array([1, 3, 5, 6, 4, 1, 3, 6])
    images = np.array([False, False, True, True, False, False, False, False])
    expected = _scalar_hash(ids, images, mapping, layout, args.engram_pad_id)
    np.testing.assert_array_equal(hash_token_run(ids, images, mapping, layout, args.engram_pad_id), expected)
    for boundary in range(1, len(ids)):
        start = max(0, boundary - 2)
        actual = hash_token_run(ids[start:], images[start:], mapping, layout, args.engram_pad_id, boundary-start)
        np.testing.assert_array_equal(actual, expected[boundary:])
    alternate = ids.copy()
    alternate[:2] = [6, 4]
    changed = hash_token_run(alternate, images, mapping, layout, args.engram_pad_id)
    np.testing.assert_array_equal(changed[4:], expected[4:])


def test_hash_rejects_wrong_bucket_geometry():
    args = _args()
    args.engram_num_embeddings = (61, 120)
    with pytest.raises(ValueError, match="table sizes"):
        HashLayout.from_args(args)


def test_content_pad_ids_use_image_boundaries_before_vocabulary_lookup():
    from freetoken.models.deepseek_v41.engram import _image_flags

    args, mapping = _args(), np.arange(7, dtype=np.int64)
    layout = HashLayout.from_args(args)
    ids = np.array([1, 3, 5, 6, 4, 1, 3, 6])
    req = SimpleNamespace(mm_items=[SimpleNamespace(offsets=[[2, 4]])])
    flags = _image_flags(req, 0, len(ids))
    expected = hash_token_run(ids, flags, mapping, layout, args.engram_pad_id)
    ids[2:4] = [1_000_100, 1_000_100]
    np.testing.assert_array_equal(hash_token_run(ids, flags, mapping, layout, args.engram_pad_id), expected)
    np.testing.assert_array_equal(_image_flags(req, 3, 4), [True, False, False, False])
    with pytest.raises(ValueError, match="outside the tokenizer"):
        hash_token_run(ids, np.zeros_like(flags), mapping, layout, args.engram_pad_id)


def test_compressed_tokens_match_training_normalization():
    texts = [" The", "the", "THE", "\u00e9", "e", "\uff25", " ", "\t", "",
             "a\r\n\tb", "a b", "\ufffd", "\ufffd"]

    class Tokenizer:
        def __len__(self):
            return len(texts)

        @property
        def backend_tokenizer(self):
            return self

        def decode(self, ids, *, skip_special_tokens):
            assert skip_special_tokens is False
            return texts[ids[0]]

        def id_to_token(self, token_id):
            return f"<byte-{token_id}>"

    np.testing.assert_array_equal(compressed_token_map(Tokenizer()),
                                  [0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 4, 5, 6])


def test_disk_table_decodes_all_e8m0_codes():
    weights = torch.ones(256, 32).to(torch.float8_e4m3fn)
    codes = torch.arange(256, dtype=torch.uint8).view(256, 1)
    table = DiskEngramTable(weights.view(torch.uint8).numpy(), codes.numpy())
    expected = (weights.float() * codes.view(torch.float8_e8m0fnu).float()).to(torch.bfloat16)
    torch.testing.assert_close(table.lookup(np.arange(256)), expected, rtol=0, atol=0, equal_nan=True)


def test_disk_table_rejects_partial_scale_group():
    with pytest.raises(ValueError, match="per 32"):
        DiskEngramTable(np.zeros((4, 33), dtype=np.uint8), np.zeros((4, 1), dtype=np.uint8))


def _checkpoint(folder, dtype="fp8"):
    args = _args()
    args.engram_dtype = dtype
    tensors = {}
    for layer, rows in zip(args.engram_layer_ids, args.engram_num_embeddings):
        w = (torch.arange(rows * 32).reshape(rows, 32) % 9 - 4).float().to(torch.float8_e4m3fn)
        if dtype == "fp4":
            w = (torch.arange(rows * 16).reshape(rows, 16) % 256).to(torch.uint8)
        s = torch.full((rows, 1), 128, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        tensors[f"layers.{layer}.engram.embed.weight"] = w
        tensors[f"layers.{layer}.engram.embed.scale"] = s
    save_file(tensors, str(folder / "tables.safetensors"))
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "tables.safetensors" for k in tensors}}))
    return args, tensors


def _reference_rows(tensors, layer, ids):
    weight = tensors[f"layers.{layer}.engram.embed.weight"]
    scale = tensors[f"layers.{layer}.engram.embed.scale"]
    if weight.dtype != torch.uint8:
        return (weight.float().reshape(len(weight), -1, 32) * scale.float()[..., None]).flatten(-2)[ids]
    result = []
    for row in ids.reshape(-1).tolist():
        values = []
        for column in range(weight.shape[1] * 2):
            code = int(weight[row, column // 2]) >> (4 * (column % 2)) & 15
            exponent, fraction = (code & 7) >> 1, code & 1
            magnitude = fraction * .5 if exponent == 0 else math.ldexp(1 + fraction * .5, exponent - 1)
            exponent_scale = int(scale.view(torch.uint8)[row, column // 32]) - 127
            value = math.ldexp(magnitude, exponent_scale) * (-1 if code & 8 else 1)
            values.append(value)
        result.append(values)
    return torch.tensor(result).reshape(*ids.shape, weight.shape[1] * 2)


@pytest.mark.parametrize("dtype", ["fp8", "fp4"])
def test_disk_table_and_standalone_export(tmp_path, dtype):
    from freetoken.checkpoint.convert import _copy_metadata

    source, dest = tmp_path / "source", tmp_path / "ftw"
    source.mkdir()
    dest.mkdir()
    args, tensors = _checkpoint(source, dtype)
    metadata = {"config.json": '{"model_type": "deepseek_v41"}',
                "inference/config.json": '{"dim": 5120}',
                "encoding/encoding.py": "IS_DSV41 = True\n",
                "tokenizer.json": "{}"}
    for name, text in metadata.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    ids = np.array([[3, 1, 3, 7], [5, 2, 0, 1]])
    for layer in args.engram_layer_ids:
        table = DiskEngramTable.from_checkpoint(str(source), layer, args=args)
        assert isinstance(table.weights, np.memmap) and not table.weights.flags.writeable
        assert isinstance(table.scales, np.memmap) and not table.scales.flags.writeable
        expected = _reference_rows(tensors, layer, torch.from_numpy(ids))
        torch.testing.assert_close(table.lookup(ids).float(), expected, rtol=0, atol=0)
    copied = _copy_metadata(str(source), str(dest))
    assert set(copied) == set(metadata)
    export_engram_tables(str(source), str(dest), args)
    manifest = json.loads((dest / "engram_tables.json").read_text())
    assert manifest["version"] == 2
    for layer in args.engram_layer_ids:
        info = manifest["layers"][str(layer)]
        assert info["dtype"] == dtype and info["block_size"] == 32 and info["scale_fmt"] == "ue8m0"
        for kind in ("weight", "scale"):
            original = tensors[f"layers.{layer}.engram.embed.{kind}"].view(torch.uint8).numpy().tobytes()
            assert (dest / info[kind]["file"]).read_bytes() == original
    source.rename(tmp_path / "unavailable")
    for name, text in metadata.items():
        assert (dest / name).read_text() == text
    for layer in args.engram_layer_ids:
        restored = DiskEngramTable.from_checkpoint(str(dest), layer, args=args)
        expected = _reference_rows(tensors, layer, torch.from_numpy(ids))
        torch.testing.assert_close(restored.lookup(ids).float(), expected, rtol=0, atol=0)


def test_legacy_fp8_manifest_remains_loadable(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "ftw"
    source.mkdir()
    args, tensors = _checkpoint(source)
    export_engram_tables(str(source), str(dest), args)
    manifest_path = dest / "engram_tables.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["version"] = 1
    for info in manifest["layers"].values():
        for field in ("dtype", "block_size", "scale_fmt"):
            info.pop(field)
    manifest_path.write_text(json.dumps(manifest))
    ids = torch.tensor([1, 7, 3])
    actual = DiskEngramTable.from_checkpoint(str(dest), 1, args=args).lookup(ids.numpy())
    torch.testing.assert_close(actual.float(), _reference_rows(tensors, 1, ids), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["fp8", "fp4"])
def test_table_rejects_config_format_mismatch(tmp_path, dtype):
    source, dest = tmp_path / "source", tmp_path / "ftw"
    source.mkdir()
    args, _ = _checkpoint(source, dtype)
    export_engram_tables(str(source), str(dest), args)
    args.engram_dtype = "fp4" if dtype == "fp8" else "fp8"
    for folder in (source, dest):
        with pytest.raises(ValueError, match="format disagrees"):
            DiskEngramTable.from_checkpoint(str(folder), 1, args=args)


def test_fp4_table_uses_per_block_scale_and_low_nibble_first():
    codes = torch.arange(256, dtype=torch.uint8).repeat(2, 1)
    scale_codes = torch.arange(111, 127, dtype=torch.uint8).repeat(2, 1)
    tensors = {"layers.1.engram.embed.weight": codes,
               "layers.1.engram.embed.scale": scale_codes.view(torch.float8_e8m0fnu)}
    ids = torch.tensor([[1, 0, 1]])
    table = DiskEngramTable(codes.numpy(), scale_codes.numpy(), dtype="fp4")
    expected = _reference_rows(tensors, 1, ids).to(torch.bfloat16)
    torch.testing.assert_close(table.lookup(ids.numpy()), expected, rtol=0, atol=0)
    with torch.device("meta"):
        actual = table.lookup(ids.numpy())
    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert table.lookup(np.empty((0, 2), dtype=np.int64)).shape == (0, 2, 512)


def test_fp4_lookup_reads_only_unique_requested_rows():
    class RequestedRowsOnly:
        dtype = np.dtype("uint8")
        ndim = 2

        def __init__(self, width, fill):
            self.shape = (384006168, width)
            self.fill = fill
            self.reads = []

        def __getitem__(self, ids):
            np.testing.assert_array_equal(ids, [1, 7, 1000])
            self.reads.append(ids.copy())
            return np.full((len(ids), self.shape[1]), self.fill, dtype=np.uint8)

        def __array__(self, *args, **kwargs):
            raise AssertionError("full Engram table materialization")

    weight, scale = RequestedRowsOnly(128, 0x32), RequestedRowsOnly(8, 128)
    table = DiskEngramTable(weight, scale, dtype="fp4")
    actual = table.lookup(np.array([[1000, 1, 7, 1]]))
    expected = torch.tensor([2., 3.], dtype=torch.bfloat16).repeat(128).expand(1, 4, 256)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(weight.reads) == len(scale.reads) == 1


@pytest.mark.parametrize("dtype,block,fmt", [("nvfp4", 32, "ue8m0"),
                                           ("fp4", 16, "ue8m0"), ("fp4", 32, "e4m3")])
def test_disk_table_rejects_unsupported_format(dtype, block, fmt):
    with pytest.raises(ValueError, match="block-32 UE8M0"):
        DiskEngramTable(np.zeros((4, 128), dtype=np.uint8), np.zeros((4, 8), dtype=np.uint8),
                        dtype=dtype, block_size=block, scale_fmt=fmt)


@pytest.mark.parametrize("shape", [(4, 4), (4, 16), (3, 8)])
def test_fp4_table_rejects_scale_shape_for_wrong_logical_width(shape):
    with pytest.raises(ValueError, match="per 32 weight channels"):
        DiskEngramTable(np.zeros((4, 128), dtype=np.uint8), np.zeros(shape, dtype=np.uint8), dtype="fp4")


def test_prepare_engram_loads_fp4_tables_without_expanding_storage(tmp_path, monkeypatch):
    import freetoken.models.deepseek_v41.engram as engram_module
    import freetoken.utils

    args, tensors = _checkpoint(tmp_path, "fp4")
    layers = [SimpleNamespace(engram=SimpleNamespace(layer_id=i, hash_cols=4)) for i in args.engram_layer_ids]
    model = SimpleNamespace(_args=args, _transformer=SimpleNamespace(layers=layers))
    config = SimpleNamespace(model_path=str(tmp_path), device="cpu", use_dummy_weight=False,
                             max_extend_tokens=3, max_running_req=1, cuda_graph_max_bs=0)
    monkeypatch.setattr(freetoken.utils, "download_hf_weight", lambda path: path)
    monkeypatch.setattr(freetoken.utils, "load_tokenizer", lambda path: object())
    monkeypatch.setattr(engram_module, "compressed_token_map", lambda tokenizer: np.arange(7))
    assert prepare_engram(model, config) == 3 * (2 * 4 * 32 * 2 + 1)
    for layer_id, table in zip(args.engram_layer_ids, model._engram_runtime.tables):
        assert table.dtype == "fp4" and table.head_dim == 32
        assert isinstance(table.weights, np.memmap) and isinstance(table.scales, np.memmap)
        assert table.weights.nbytes + table.scales.nbytes == table.num_rows * 17
        ids = torch.tensor([1, 3, 5])
        torch.testing.assert_close(table.lookup(ids.numpy()).float(), _reference_rows(tensors, layer_id, ids),
                                   rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["fp8", "fp4"])
def test_runtime_isolates_requests_and_images(tmp_path, dtype):
    args, _ = _checkpoint(tmp_path, dtype)
    tables = [DiskEngramTable.from_checkpoint(str(tmp_path), layer) for layer in args.engram_layer_ids]
    modules = [SimpleNamespace(hash_cols=4) for _ in tables]
    runtime = EngramRuntime(args, modules, tables, np.arange(7), 12, "cpu")
    first = SimpleNamespace(input_ids=torch.tensor([1, 3, 5, 4]), cached_len=2, extend_len=2,
                            device_len=4, media=[{"start": 2, "types": [0]}])
    second = SimpleNamespace(input_ids=torch.tensor([6, 1]), cached_len=0, extend_len=2, device_len=2, media=None)
    batch = SimpleNamespace(input_ids=torch.tensor([5, 4, 6, 1]), padded_reqs=[first, second], is_decode=False)
    with runtime.forward_host_ctx(batch, False):
        assert not modules[0]._mask[0]
        assert modules[0]._mask[1:4].all()
        assert not modules[0]._values[0].any()
        expected_ids = hash_token_run(np.array([6, 1]), np.zeros(2, bool), np.arange(7), runtime.layout, 2)
        torch.testing.assert_close(modules[0]._values[2:4], tables[0].lookup(expected_ids[:, 0]).flatten(-2))


def test_runtime_decode_uses_current_gpu_token_and_keeps_image_boundary(tmp_path):
    args, _ = _checkpoint(tmp_path)
    tables = [DiskEngramTable.from_checkpoint(str(tmp_path), layer) for layer in args.engram_layer_ids]
    modules = [SimpleNamespace(hash_cols=4) for _ in tables]
    runtime = EngramRuntime(args, modules, tables, np.arange(7), 3, "cpu")
    # Under overlap, the current token is in batch.input_ids before append_host runs.
    req = SimpleNamespace(input_ids=torch.tensor([1, 3, 5, 4]), device_len=5,
                          media=[{"start": 1, "types": [0, 1, 3]}])
    dummy = SimpleNamespace(input_ids=torch.tensor([0]), device_len=1, media=None)
    batch = SimpleNamespace(input_ids=torch.tensor([6, 0]), padded_reqs=[req, dummy], is_decode=True)
    with runtime.forward_host_ctx(batch, True):
        expected = hash_token_run(np.array([1, 3, 5, 4, 6]), np.array([0, 1, 1, 1, 0]),
                                  np.arange(7), runtime.layout, 2)[-1:]
        for index, table in enumerate(tables):
            torch.testing.assert_close(modules[index]._values[:1], table.lookup(expected[:, index]).flatten(-2))
        assert modules[0]._mask[:2].all()


@pytest.mark.parametrize("prefill,decode,graph,expected", [(7, 3, 16, 16), (9, 16, 3, 16), (None, 4, 4, 8192)])
def test_engram_staging_capacity_follows_batch_budget(prefill, decode, graph, expected):
    args = _args()
    layers = [SimpleNamespace(engram=SimpleNamespace(hash_cols=4)) for _ in args.engram_layer_ids]
    model = SimpleNamespace(_args=args, _transformer=SimpleNamespace(layers=layers))
    config = SimpleNamespace(use_dummy_weight=True, device="cpu", max_forward_len=1 << 20,
                             max_running_req=decode, cuda_graph_max_bs=graph)
    if prefill is not None:
        config.max_extend_tokens = prefill
    pinned_bytes = prepare_engram(model, config)
    assert model._engram_runtime.capacity == expected
    assert pinned_bytes == expected * (len(layers) * 4 * 32 * 2 + 1)


def test_engram_staging_rejects_larger_batch_before_writing():
    args = _args()
    module = SimpleNamespace(hash_cols=4)
    runtime = EngramRuntime(args, [module], [], None, 2, "cpu", dummy=True)
    req = SimpleNamespace(input_ids=torch.tensor([1, 3, 5]), cached_len=0, extend_len=3, media=None)
    batch = SimpleNamespace(input_ids=req.input_ids, padded_reqs=[req], is_decode=False)
    with pytest.raises(ValueError, match="requires 3 rows; capacity is 2"):
        with runtime.forward_host_ctx(batch, False):
            pass
    assert not module._values.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", ["fp8", "fp4"])
def test_engram_staging_updates_captured_graph(tmp_path, dtype):
    args, _ = _checkpoint(tmp_path, dtype)
    args.dim, args.hc_mult, args.norm_eps = 32, 2, 1e-6
    module = Engram(args, 1).cuda()
    torch.manual_seed(7)
    with torch.no_grad():
        module.wkv.weight.copy_(torch.randn(module.wkv.weight.shape, device="cuda") * .1)
        module.wkv.scale.view(torch.uint8).fill_(127)
    table = DiskEngramTable.from_checkpoint(str(tmp_path), 1)
    runtime = EngramRuntime(args, [module], [table], np.arange(7), 2, "cuda")
    hidden = torch.randn(1, 1, 2, 32, device="cuda", dtype=torch.bfloat16)
    req = SimpleNamespace(input_ids=torch.tensor([1, 3]), device_len=3, media=None)
    batch = SimpleNamespace(input_ids=torch.tensor([6], device="cuda"), padded_reqs=[req], is_decode=True)
    with torch.inference_mode():
        with runtime.forward_host_ctx(batch, False):
            expected = module(hidden).clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                module(hidden)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = module(hidden)
        with runtime.forward_host_ctx(batch, True):
            graph.replay()
        torch.testing.assert_close(output, expected)
        req.media = [{"start": 2, "types": [0]}]
        with runtime.forward_host_ctx(batch, True):
            graph.replay()
        torch.testing.assert_close(output, hidden, rtol=0, atol=0)
