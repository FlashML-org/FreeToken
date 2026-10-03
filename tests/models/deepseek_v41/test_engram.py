"""Engram: the hash against a transcription of the reference ``NgramHashState``, the prime buckets,
the disk table against a torch oracle over the synthetic checkpoint (eager, decode with context,
CUDA-graph flag sync), and the layer's gate math against the reference ``Engram.forward``."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.models.deepseek_v41.engram import EngramHash, bucket_primes, hash_multipliers

from .common import VOCAB, requires_cuda, tiny_hf_config, write_tiny_checkpoint


def _args(**over) -> DeepseekV41Args:
    return DeepseekV41Args.from_hf(tiny_hf_config(**over))


def _reference_hashes(args: DeepseekV41Args, ids: torch.Tensor, start_pos: int, cache: torch.Tensor) -> torch.Tensor:
    """``NgramHashState.forward`` for one sequence with an identity token map; ``cache`` is the
    sequence's compressed-id history buffer (mutated like the reference's)."""
    layout_primes = bucket_primes(args)
    flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout_primes]
    offsets = torch.tensor(np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat]))
    primes = torch.tensor(layout_primes)
    multipliers = hash_multipliers(args.engram_layer_ids, args.engram_max_ngram_size, args.engram_compressed_vocab_size)
    pad_id = args.engram_pad_id
    seqlen = ids.numel()
    cache[start_pos : start_pos + seqlen] = ids
    positions = torch.arange(start_pos, start_pos + seqlen)
    tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
    for shift in range(args.engram_max_ngram_size):
        source = cache[(positions - shift).clamp_min(0)]
        blocked = blocked | (positions < shift)
        tokens.append(torch.where(blocked, pad_id, source))
    tokens = torch.stack(tokens, dim=-1)
    products = tokens.unsqueeze(1) * multipliers
    rolling, hashes = products[..., 0], []
    for i in range(1, args.engram_max_ngram_size):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])
    return torch.cat(hashes, dim=-1) + offsets


def test_primes_are_distinct_and_ordered():
    args = _args()
    primes = bucket_primes(args)
    flat = [p for layer in primes for per in layer for p in per]
    assert len(flat) == len(set(flat)) and flat == sorted(flat) and min(flat) > args.engram_vocab_size - 1
    from sympy import isprime

    assert all(isprime(p) for p in flat)


def test_hash_matches_the_reference_across_a_prefill_decode_split():
    args = _args()
    hash = EngramHash(args, list(range(VOCAB)), VOCAB)
    ids = torch.randint(3, VOCAB, (37,), generator=torch.Generator().manual_seed(0))
    cache = torch.empty(64, dtype=torch.int64)
    want = _reference_hashes(args, ids[:30], 0, cache)
    got = hash.row_ids(ids[:30])
    assert torch.equal(got, want)
    # decode continues with the three preceding tokens as context
    for step in range(30, 37):
        want = _reference_hashes(args, ids[step : step + 1], step, cache)
        got = hash.row_ids(ids[step : step + 1], context=ids[:step])
        assert torch.equal(got, want), step
    # a run resumed mid-sequence (chunked prefill) hashes like the uninterrupted one
    whole = hash.row_ids(ids)
    assert torch.equal(torch.cat([hash.row_ids(ids[:20]), hash.row_ids(ids[20:], context=ids[:20])]), whole)
    assert whole.shape == (37, 1, hash.n_cols) and int(whole.max()) < hash.rows_per_layer[0]
    with pytest.raises(ValueError, match="compressed vocab"):
        EngramHash(args, list(range(VOCAB)), VOCAB + 1)


@requires_cuda
def test_disk_table_matches_the_dequantized_shard(tmp_path):
    from freetoken.models.deepseek_v41.engram_table import EngramDiskTable, engram_row_source

    tensors = write_tiny_checkpoint(str(tmp_path))
    args = _args()
    hash = EngramHash(args, list(range(VOCAB)), VOCAB)
    src = engram_row_source(str(tmp_path), 1)
    assert src.num_rows == tensors["layers.1.engram.embed.weight"].shape[0] and src.head_dim == 32
    table = EngramDiskTable(src, hash.n_cols, torch.device("cuda"), max_graph_rows=4, max_extend_tokens=16)
    fp8 = tensors["layers.1.engram.embed.weight"]
    scale = torch.exp2(tensors["layers.1.engram.embed.scale"].view(torch.uint8).float() - 127.0)

    def oracle(rows: torch.Tensor) -> torch.Tensor:
        vals = fp8[rows.flatten()].float().view(-1, 32 // 32, 32) * scale[rows.flatten()].unsqueeze(-1)
        return vals.view(rows.shape[0], -1).to(torch.bfloat16).cuda()

    ids = torch.randint(3, VOCAB, (40,), generator=torch.Generator().manual_seed(1))
    rows = hash.row_ids(ids)[:, 0]
    table.stage(rows, graph=False)  # larger than the initial eager staging: grows
    table.flush(signal=False)
    assert torch.equal(table.lookup(40), oracle(rows))
    # the graph staging + flag protocol: capture a lookup, fill after the launch, replay
    from freetoken.kernel.row_store import probe_wait_sync

    table.wait_sync = probe_wait_sync("auto", torch.device("cuda"))
    dec_rows = hash.row_ids(ids[40 - 1 :], context=ids[: 40 - 1])[:, 0]
    out = torch.empty(1, hash.n_cols * 32, dtype=torch.bfloat16, device="cuda")
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        if not table.wait_sync:
            table.stage(dec_rows, graph=True)
            table.flush(signal=False)
        with torch.cuda.graph(graph, stream=stream):
            out.copy_(table.lookup(1))
    torch.cuda.synchronize()
    graph.replay()  # under wait-sync the replay blocks until the host signals
    table.stage(dec_rows, graph=True)
    table.flush(signal=table.wait_sync)
    torch.cuda.synchronize()
    assert torch.equal(out, oracle(dec_rows))


@requires_cuda
def test_eager_staging_does_not_overwrite_an_in_flight_lookup(tmp_path):
    """The overlap scheduler stages batch k+1 while batch k's lookup copy is still queued behind
    the layers before the Engram layer (a sleep kernel here): batch k must read its own rows."""
    from freetoken.models.deepseek_v41.engram_table import EngramDiskTable, engram_row_source

    tensors = write_tiny_checkpoint(str(tmp_path))
    hash = EngramHash(_args(), list(range(VOCAB)), VOCAB)
    table = EngramDiskTable(engram_row_source(str(tmp_path), 1), hash.n_cols, torch.device("cuda"), max_graph_rows=4, max_extend_tokens=64)
    fp8 = tensors["layers.1.engram.embed.weight"]
    scale = torch.exp2(tensors["layers.1.engram.embed.scale"].view(torch.uint8).float() - 127.0)

    def oracle(rows):
        vals = fp8[rows.flatten()].float().view(-1, 1, 32) * scale[rows.flatten()].unsqueeze(-1)
        return vals.view(rows.shape[0], -1).to(torch.bfloat16).cuda()

    gen = torch.Generator().manual_seed(0)
    batches = [hash.row_ids(torch.randint(3, VOCAB, (40,), generator=gen))[:, 0] for _ in range(4)]
    got = []
    for rows in batches:
        table.stage(rows, graph=False)
        table.flush(signal=False)
        torch.cuda._sleep(50_000_000)
        got.append(table.lookup(rows.shape[0]))
    torch.cuda.synchronize()
    for i, (rows, out) in enumerate(zip(batches, got)):
        assert torch.equal(out, oracle(rows)), f"batch {i} read another batch's rows"


@requires_cuda
@torch.inference_mode()
def test_host_fills_a_graph_decode_step_from_a_worker_thread(tmp_path):
    """The deferred graph-decode fill must not wait for the launch call to return: a 40-layer graph
    blocks cuGraphLaunch on the host while the GPU sits on the WAIT, so the host coordinator reads the
    tokens back and signals from a worker thread. The worker hashes against the DISPATCH snapshot of
    the request state: advancing the request right after dispatch (what the engine does) must not
    change which rows are fetched."""
    from types import SimpleNamespace

    from freetoken.models.deepseek_v41.engram_table import EngramDiskTable, EngramHost, engram_row_source

    tensors = write_tiny_checkpoint(str(tmp_path))
    args = _args()
    hash = EngramHash(args, list(range(VOCAB)), VOCAB)
    table = EngramDiskTable(engram_row_source(str(tmp_path), 1), hash.n_cols, torch.device("cuda"), max_graph_rows=4, max_extend_tokens=16)
    host = EngramHost(hash, [table], torch.device("cuda"))
    if not host.wait_sync:
        pytest.skip("stream memops unavailable: launch-gating mode needs no worker")
    ids = torch.randint(3, VOCAB, (20,), generator=torch.Generator().manual_seed(7))
    req = SimpleNamespace(input_ids=ids.clone(), device_len=20, cached_len=19)
    batch = SimpleNamespace(is_decode=True, is_prefill=False, padded_reqs=[req], padded_size=1, input_ids=ids[19:20].to(torch.int32).cuda())
    out = torch.empty(1, hash.n_cols * 32, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            out.copy_(table.lookup(1))
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        # slow the readback so the request advances (complete_one + append) before the worker runs
        torch.cuda._sleep(200_000_000)
        with host.forward_host_ctx(batch, use_graph=True):
            graph.replay()
            req.device_len += 1
            req.input_ids = torch.cat([req.input_ids, torch.tensor([5], dtype=ids.dtype)])
    torch.cuda.synchronize()
    fp8 = tensors["layers.1.engram.embed.weight"]
    rows = hash.row_ids(ids[19:20], context=ids[:19])[:, 0].flatten()
    want = fp8[rows].float().view(1, -1).to(torch.bfloat16).cuda()  # scales are all 1.0 in the synthetic table
    assert torch.equal(out, want)
    host.close()


@requires_cuda
def test_host_surfaces_a_failed_deferred_fill_at_dispatch_exit(tmp_path):
    from types import SimpleNamespace

    from freetoken.models.deepseek_v41.engram_table import EngramDiskTable, EngramHost, engram_row_source

    write_tiny_checkpoint(str(tmp_path))
    args = _args()
    hash = EngramHash(args, list(range(VOCAB)), VOCAB)
    table = EngramDiskTable(engram_row_source(str(tmp_path), 1), hash.n_cols, torch.device("cuda"), max_graph_rows=4, max_extend_tokens=16)
    host = EngramHost(hash, [table], torch.device("cuda"))
    if not host.wait_sync:
        pytest.skip("stream memops unavailable")
    ids = torch.randint(3, VOCAB, (8,), generator=torch.Generator().manual_seed(8))
    req = SimpleNamespace(input_ids=ids, device_len=8, cached_len=7)
    batch = SimpleNamespace(is_decode=True, is_prefill=False, padded_reqs=[req], padded_size=1, input_ids=ids[7:8].to(torch.int32).cuda())
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    out = torch.empty(1, hash.n_cols * 32, dtype=torch.bfloat16, device="cuda")
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            out.copy_(table.lookup(1))
    torch.cuda.synchronize()
    table.stage = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk gone"))  # the fill fails
    with pytest.raises(RuntimeError, match="disk gone"):
        with torch.cuda.stream(stream):
            with host.forward_host_ctx(batch, use_graph=True):
                graph.replay()
    torch.cuda.synchronize()  # the failing worker released the flag: the stream is not stuck
    host.close()


def test_ftw_side_files_carry_the_tables_and_reopen(tmp_path):
    """An FTW conversion copies the shards holding the Engram tables next to it (the converter's
    ``ftw_side_files`` hook, which skips the HF index); the row source resolves the copies through
    their headers exactly like the HF shards, byte for byte."""
    from freetoken.kernel.row_store import RowStore
    from freetoken.models.deepseek_v41.engram_table import engram_row_source
    from freetoken.models.deepseek_v41.weight import ftw_side_files

    src, out = tmp_path / "hf", tmp_path / "ftw"
    out.mkdir()
    tensors = write_tiny_checkpoint(str(src))
    (out / "config.json").write_bytes((src / "config.json").read_bytes())  # the converter copies the metadata
    shard = "model-00001-of-00001.safetensors"  # the tiny checkpoint's single shard holds the table
    assert ftw_side_files(str(src), str(out)) == [shard]
    assert (out / shard).read_bytes() == (src / shard).read_bytes() and not (out / "model.safetensors.index.json").exists()
    source = engram_row_source(str(out), 1)
    assert source.path == str(out / shard) and source.num_rows == 4096 and source.head_dim == 32
    assert torch.equal(source.scales, tensors["layers.1.engram.embed.scale"].view(torch.uint8))
    store = RowStore(paths=[source.path], extent_file=[0], extent_base=[source.base], rows_per_extent=source.num_rows,
                     row_bytes=source.head_dim, row_stride=source.head_dim, use_io_uring=False)
    ids = torch.tensor([0, 7, 4095], dtype=torch.int64)
    buf = torch.zeros(3 * 32, dtype=torch.uint8)
    store.stage_rows(ids.data_ptr(), 3, buf.data_ptr(), 0)
    store.flush(0)
    assert torch.equal(buf.view(3, 32), tensors["layers.1.engram.embed.weight"][ids].view(torch.uint8))


def test_layer_gate_matches_the_reference():
    """``EngramLayer.forward`` against the reference ``Engram.forward`` (bf16 wkv, fp32 gate math)."""
    from freetoken.distributed.info import set_tp_info, try_get_tp_info
    from freetoken.models.deepseek_v41.engram import EngramLayer

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    args = _args()
    torch.manual_seed(0)
    layer = EngramLayer(args, 1)
    T, hc, dim = 5, args.hc_mult, args.dim
    wkv = torch.randn(dim * (hc + 1), layer.width, dtype=torch.bfloat16) * 0.05
    layer.wkv.weight = wkv
    layer.q_weight = torch.rand(hc, dim) + 0.5
    layer.k_weight = torch.rand(hc, dim) + 0.5
    x = torch.randn(T, hc, dim, dtype=torch.bfloat16)
    rows = torch.randn(T, layer.width, dtype=torch.bfloat16)

    kv = torch.nn.functional.linear(rows, wkv)
    key, value = kv.split([hc * dim, dim], dim=-1)
    key = key.float().view(T, hc, dim)
    weight = layer.q_weight * layer.k_weight
    h = x.float()
    rstd = torch.rsqrt(h.square().mean(-1) + args.norm_eps) * torch.rsqrt(key.square().mean(-1) + args.norm_eps)
    dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    want = (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
    assert torch.equal(layer.forward(x, rows), want)
