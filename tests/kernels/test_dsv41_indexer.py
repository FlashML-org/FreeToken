import pytest
import torch

from freetoken.kernel.triton.dsv41.indexer import index_scores, select_indices


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))])
def test_paged_scores_match_reference_with_noncontiguous_pages_and_per_query_limits(device):
    torch.manual_seed(180)
    q = torch.randn(3, 4, 32, device=device, dtype=torch.bfloat16)
    weights = torch.randn(3, 4, device=device, dtype=torch.bfloat16)
    pool = torch.randn(256, 32, device=device, dtype=torch.bfloat16)
    table = torch.stack([torch.cat((torch.arange(base, base + 64), torch.arange(0, 64)))
                         for base in [128, 192, 256]]).to(device)
    ids = torch.arange(64, device=device).expand(3, -1)
    valid = torch.tensor([64, 31, 0], device=device)
    got = index_scores(q, weights, pool, table, ids, 2, valid)
    rows = table[:, ::2] // 2
    keys = pool[rows]
    dot = torch.einsum("qhd,qkd->qhk", q.float(), keys.float()).to(q.dtype)
    ref = (dot.relu() * weights[..., None]).sum(1)
    ref.masked_fill_(ids >= valid[:, None], -torch.inf)
    torch.testing.assert_close(got, ref, atol=0, rtol=0)


def reference_selection(q, weights, keys, valid, topk, candidate_count, block_size, mask=None):
    dot = torch.einsum("qhd,kd->qhk", q.float(), keys.float()).to(q.dtype)
    scores = (dot.relu() * weights[..., None]).sum(1)
    scores.masked_fill_(torch.arange(keys.shape[0])[None] >= valid[:, None], -torch.inf)
    if mask is not None:
        scores.masked_fill_(~mask, -torch.inf)
    picks = scores.argsort(dim=-1, descending=True, stable=True)[..., :topk].sort(-1).values
    picks = torch.where(picks < valid[:, None], picks, -1)
    if not candidate_count:
        return picks, None
    padded = torch.nn.functional.pad(scores, (0, -keys.shape[0] % block_size), value=-torch.inf)
    block_scores = padded.unflatten(-1, (-1, block_size)).amax(-1)
    newest = (valid - 1) // block_size
    block_scores.masked_fill_(torch.arange(block_scores.shape[1])[None] == newest[:, None], torch.inf)
    best = block_scores.argsort(dim=-1, descending=True, stable=True)[..., :candidate_count]
    keep = torch.zeros_like(block_scores, dtype=torch.bool).scatter_(-1, best, block_scores.gather(-1, best) > -torch.inf)
    return picks, keep.repeat_interleave(block_size, -1)[:, :keys.shape[0]]


def test_streaming_topk_candidate_blocks_and_second_stage_match_dense_reference():
    torch.manual_seed(32)
    q = torch.randn(5, 3, 32)
    weights = torch.rand(5, 3)
    keys = torch.randn(77, 32)
    table = torch.arange(77).expand(5, -1)
    valid = torch.tensor([1, 14, 45, 76, 77])
    got, blocks = select_indices(q, weights, keys, table, valid, 77, 1, 5,
                                 candidate_topk=3, block_size=8, query_tile=2, key_tile=16)
    ref, mask = reference_selection(q, weights, keys, valid, 5, 3, 8)
    torch.testing.assert_close(got, ref)
    expanded = torch.zeros_like(mask)
    for row in range(5):
        for block in blocks[row].tolist():
            if block >= 0:
                expanded[row, block * 8:(block + 1) * 8] = True
    assert torch.equal(expanded, mask)
    q2 = torch.randn_like(q)
    got2, _ = select_indices(q2, weights, keys, table, valid, 77, 1, 5,
                             candidates=blocks, block_size=8, query_tile=2, key_tile=16)
    ref2, _ = reference_selection(q2, weights, keys, valid, 5, 0, 8, mask)
    torch.testing.assert_close(got2, ref2)


def test_empty_index_history_has_no_candidates_or_read():
    q = torch.randn(2, 3, 32)
    got, _ = select_indices(q, torch.ones(2, 3), torch.zeros(1, 32), torch.zeros(2, 1, dtype=torch.long),
                             torch.zeros(2, dtype=torch.long), 0, 2, 8)
    assert got.shape == (2, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("packed", [False, True])
def test_million_positions_keep_score_working_memory_bounded(packed):
    width = 1 << 20
    q = torch.zeros(33, 4, 32, device="cuda", dtype=torch.bfloat16)
    weights = torch.ones(33, 4, device="cuda", dtype=torch.bfloat16)
    pool = torch.zeros(width, 17 if packed else 32, device="cuda",
                       dtype=torch.uint8 if packed else torch.bfloat16)
    table = torch.arange(width, device="cuda").expand(33, -1)
    valid = torch.full((33,), width, device="cuda")
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    selected, candidates = select_indices(q, weights, pool, table, valid, width, 1, 512,
                                          candidate_topk=2048, block_size=8)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - baseline < 16 << 20
    torch.testing.assert_close(selected, torch.arange(512, device="cuda").expand(33, -1))
    assert ((candidates == (width - 1) // 8).any(-1)).all()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("heads", [4, 32])
def test_packed_index_scores_and_candidate_selection_match_restored_keys(device, ratio, heads):
    torch.manual_seed(917)
    dim, total, width = 128, 257, 77
    raw = torch.randint(0, 256, (total, dim // 2), dtype=torch.uint8)
    scales = torch.randint(121, 128, (total, dim // 32), dtype=torch.uint8)
    packed = torch.cat((raw, scales), -1).to(device)
    codes = torch.stack((raw & 15, raw >> 4), -1).flatten(-2).long()
    magnitudes = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    restored = (magnitudes[codes & 7] * torch.where(codes & 8 > 0, -1, 1)
                * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)).bfloat16().to(device)
    q = torch.randn(3, heads, dim, device=device, dtype=torch.bfloat16)
    weights = torch.rand(3, heads, device=device, dtype=torch.bfloat16)
    rows = torch.randperm(total, device=device)[:width].expand(3, -1)
    table = (rows * ratio)[:, :, None] + torch.arange(ratio, device=device)
    table = table.flatten(1)
    table[1, 8:12] = -1
    valid = torch.tensor([77, 45, 0], device=device)
    ids = torch.arange(width, device=device).expand(3, -1).clone()
    ids[:, ::13] = -1
    expected = index_scores(q, weights, restored, table, ids, ratio, valid)
    actual = index_scores(q, weights, packed, table, ids, ratio, valid)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    options = dict(candidate_topk=3, block_size=8, query_tile=2, key_tile=16)
    expected_ids, expected_blocks = select_indices(q, weights, restored, table, valid, width, ratio, 7, **options)
    actual_ids, actual_blocks = select_indices(q, weights, packed, table, valid, width, ratio, 7, **options)
    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_blocks, expected_blocks)
    options.pop("candidate_topk")
    next_q = torch.randn_like(q)
    expected, _ = select_indices(next_q, weights, restored, table, valid, width, ratio, 7,
                                 candidates=expected_blocks, **options)
    actual, _ = select_indices(next_q, weights, packed, table, valid, width, ratio, 7,
                               candidates=actual_blocks, **options)
    torch.testing.assert_close(actual, expected)


def test_packed_index_rejects_incomplete_scale_row():
    q = torch.zeros(1, 4, 128)
    with pytest.raises(ValueError, match="Packed index"):
        index_scores(q, torch.ones(1, 4), torch.zeros(3, 65, dtype=torch.uint8),
                     torch.arange(3)[None], torch.arange(3)[None], 1, torch.tensor([3]))
