from types import SimpleNamespace

import pytest
import torch

from freetoken.attention.dsv41_sparse import DSV41SparseAttnBackend
from freetoken.kernel.triton.dsv41.sparse_attn import sparse_attn_paged


def packed_pools(dim, device):
    generator = torch.Generator().manual_seed(514)
    window_codes = torch.randn(43, dim, generator=generator).to(torch.float8_e4m3fn).view(torch.uint8)
    window_scales = torch.randint(123, 130, (43, dim // 32), generator=generator, dtype=torch.uint8)
    window = torch.cat((window_codes, window_scales), -1).to(device)
    restored_window = (window_codes.view(torch.float8_e4m3fn).float()
                       * torch.exp2(window_scales.float() - 127).repeat_interleave(32, -1)).bfloat16().to(device)
    cmp_codes = torch.randint(0, 256, (71, dim // 2), generator=generator, dtype=torch.uint8)
    scale_grid = torch.tensor([.09375, .125, .28125, .3125, .625, 1.25, 3.5])
    cmp_scales = scale_grid[torch.randint(0, len(scale_grid), (71, dim // 16), generator=generator)]
    compressed = torch.cat((cmp_codes, cmp_scales.to(torch.float8_e4m3fn).view(torch.uint8)), -1).to(device)
    code = torch.stack((cmp_codes & 15, cmp_codes >> 4), -1).flatten(-2).long()
    values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    restored_compressed = (values[code & 7] * torch.where(code & 8 > 0, -1, 1)
                           * cmp_scales.repeat_interleave(16, -1)).bfloat16().to(device)
    return window, compressed, restored_window, restored_compressed


def inputs(dim, queries, device):
    torch.manual_seed(103)
    q = (torch.randn(2, queries, 5, dim, device=device) * .15).bfloat16()
    sink = torch.linspace(-1, 1, 5, device=device)
    window_ids = torch.randint(0, 43, (2, queries, 33), device=device)
    cmp_ids = torch.randint(0, 71, (2, queries, 148), device=device)
    ids = torch.cat((window_ids, cmp_ids), -1).int()
    ids[..., ::7] = -1
    ids[0, 0] = -1
    counts = torch.full((2, queries), 117, dtype=torch.int32, device=device)
    counts[0, 0] = 0
    return q, sink, ids, counts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dim", [32, 512])
@pytest.mark.parametrize("queries,splits", [(1, 0), (1, 4), (3, 0), (3, 4)])
def test_packed_sparse_matches_bf16_kernel_for_mixed_tiles_and_counts(monkeypatch, dim, queries, splits):
    from freetoken.kernel.triton.dsv4 import sparse_attn as reference

    win, cmp, restored_win, restored_cmp = packed_pools(dim, "cuda")
    q, sink, ids, counts = inputs(dim, queries, "cuda")
    monkeypatch.setattr(reference, "split_count", lambda *args: splits)
    expected = reference.sparse_attn_paged(q, restored_win, restored_cmp, sink, ids, 33, dim ** -.5, counts)
    actual = sparse_attn_paged(q, win, cmp, sink, ids, 33, dim ** -.5, counts, force_splits=splits)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.count_nonzero(actual[0, 0]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("splits", [0, 4])
def test_packed_window_only_alias_and_no_count_limit(monkeypatch, splits):
    from freetoken.kernel.triton.dsv4 import sparse_attn as reference

    win, _, restored, _ = packed_pools(512, "cuda")
    q, sink, ids, _ = inputs(512, 1, "cuda")
    ids = ids[..., :33]
    monkeypatch.setattr(reference, "split_count", lambda *args: splits)
    expected = reference.sparse_attn_paged(q, restored, restored, sink, ids, 33, 512 ** -.5)
    actual = sparse_attn_paged(q, win, win, sink, ids, 33, 512 ** -.5, force_splits=splits)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("source", [None, 0])
def test_cpu_backend_restores_selected_rows_and_honors_live_counts(monkeypatch, source):
    from freetoken.kernel.triton.dsv41 import quant

    win, cmp, restored_win, restored_cmp = packed_pools(32, "cpu")
    pool = SimpleNamespace(kv_sources=[source], window_pool=[win], cmp_pool=[cmp], kv_quant="fp8-fp4")
    backend = DSV41SparseAttnBackend.__new__(DSV41SparseAttnBackend)
    monkeypatch.setattr(DSV41SparseAttnBackend, "pool", property(lambda self: pool))
    q, sink, ids, counts = inputs(32, 3, "cpu")
    if source is None:
        ids, counts = ids[..., :33], None
    original_unpack = quant.unpack_fp8
    seen = []

    def unpack_selected(packed, **kwargs):
        seen.append(tuple(packed.shape))
        assert packed.shape == (6, 33, 33)
        return original_unpack(packed, **kwargs)

    monkeypatch.setattr(quant, "unpack_fp8", unpack_selected)
    actual = backend.attend(q, 0, ids, 33, sink, 32 ** -.5, counts, source is not None)
    pool.kv_quant = "none"
    pool.window_pool = [restored_win]
    pool.cmp_pool = [restored_cmp]
    expected = backend.attend(q, 0, ids, 33, sink, 32 ** -.5, counts, source is not None)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(seen) == 1


def test_backend_routes_compressed_stores_without_casting(monkeypatch):
    calls = []
    pool = SimpleNamespace(store_compressed=lambda *args: calls.append(("attn", args)),
                           store_indexer=lambda *args: calls.append(("idx", args)))
    monkeypatch.setattr(DSV41SparseAttnBackend, "pool", property(lambda self: pool))
    backend = DSV41SparseAttnBackend.__new__(DSV41SparseAttnBackend)
    rows, packed = torch.tensor([1, 5]), torch.tensor([[255, 129], [3, 192]], dtype=torch.uint8)
    for tier in ("attn", "idx"):
        backend.scatter_compressed(2, tier, rows, packed)
    assert [call[0] for call in calls] == ["attn", "idx"]
    for _, args in calls:
        assert args[0] is packed and args[1] == 2 and args[2] is rows
    with pytest.raises(ValueError, match="tier"):
        backend.scatter_compressed(2, "unknown", rows, packed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_sparse_cuda_graph_reads_new_indices_counts_and_codes():
    win, cmp, _, _ = packed_pools(512, "cuda")
    q, sink, ids, counts = inputs(512, 1, "cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            sparse_attn_paged(q, win, cmp, sink, ids, 33, 512 ** -.5, counts, force_splits=4)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = sparse_attn_paged(q, win, cmp, sink, ids, 33, 512 ** -.5, counts, force_splits=4)
    ids[1, 0, :33].fill_(3)
    ids[1, 0, 33:].fill_(7)
    counts[1, 0] = 71
    win[3, :512].zero_()
    cmp[7, :256].fill_(0x77)
    graph.replay()
    expected = sparse_attn_paged(q, win, cmp, sink, ids, 33, 512 ** -.5, counts, force_splits=4)
    torch.testing.assert_close(captured, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seed", [514, 515, 516])
@pytest.mark.parametrize("queries", [1, 32])
def test_native_head_count_preserves_attention_with_bf16_rounding_tolerance(seed, queries):
    from freetoken.kernel.triton.dsv4 import sparse_attn as reference
    from freetoken.kernel.triton.dsv41.quant import pack_fp4, pack_fp8, unpack_fp4, unpack_fp8

    torch.manual_seed(seed)
    win = pack_fp8(torch.randn(2048, 512, device="cuda", dtype=torch.bfloat16) * .2)
    cmp = pack_fp4(torch.randn(8192, 512, device="cuda", dtype=torch.bfloat16) * .2)
    q = torch.randn(1, queries, 64, 512, device="cuda", dtype=torch.bfloat16) * .1
    sink = torch.linspace(-1, 1, 64, device="cuda")
    ids = torch.cat((torch.randint(0, 2048, (1, queries, 128), device="cuda"),
                     torch.randint(0, 8192, (1, queries, 512), device="cuda")), -1).int()
    expected = reference.sparse_attn_paged(q, unpack_fp8(win), unpack_fp4(cmp), sink, ids, 128, 512 ** -.5)
    actual = sparse_attn_paged(q, win, cmp, sink, ids, 128, 512 ** -.5)
    # FP32 dot layouts can differ at a final BF16 rounding boundary; one BF16 ULP is sufficient.
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1 / 128)
