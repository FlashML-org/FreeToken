from __future__ import annotations

import pytest
import torch

from freetoken.attention.qsa import _compact_expanded_selection, select_qsa_logical_rows
from freetoken.kernel.triton.qsa import qsa_sparse_gqa


def test_qsa_selection_is_dense_before_budget_and_keeps_tail():
    torch.manual_seed(1)
    # Four complete groups plus a two-token tail at query position 17.
    q = torch.randn(1, 4, 8)
    keys = torch.randn(4, 1, 8)
    selected, counts = select_qsa_logical_rows(
        q,
        keys,
        torch.tensor([17]),
        compress_ratio=4,
        token_budget=2048,
    )
    assert counts.tolist() == [18]
    assert set(selected[0, :18].tolist()) == set(range(18))
    assert torch.all(selected[0, 18:] == -1)


def test_qsa_selection_obeys_query_causality_for_prefill_rows():
    torch.manual_seed(2)
    q = torch.randn(4, 4, 8)
    keys = torch.randn(2, 1, 8)
    positions = torch.tensor([0, 3, 4, 7])
    selected, counts = select_qsa_logical_rows(
        q, keys, positions, compress_ratio=4, token_budget=8
    )
    assert counts.tolist() == [1, 4, 5, 8]
    for row, position in enumerate(positions.tolist()):
        assert set(selected[row, : counts[row]].tolist()) == set(range(position + 1))


def test_qsa_sparse_gqa_matches_explicit_attention_cpu():
    torch.manual_seed(3)
    q = torch.randn(3, 4, 8, dtype=torch.bfloat16)
    k = torch.randn(12, 2, 8, dtype=torch.bfloat16)
    v = torch.randn(12, 2, 8, dtype=torch.bfloat16)
    rows = torch.tensor(
        [[0, 2, 4, -1], [1, 3, 5, 7], [8, 9, -1, -1]], dtype=torch.int32
    )
    counts = torch.tensor([3, 4, 2], dtype=torch.int32)
    actual = qsa_sparse_gqa(q, k, v, rows, counts, 8**-0.5)

    expected = torch.zeros_like(q)
    for row in range(3):
        chosen = rows[row, : counts[row]].long()
        for kv_head in range(2):
            heads = slice(kv_head * 2, (kv_head + 1) * 2)
            score = torch.einsum(
                "hd,td->ht", q[row, heads].float(), k[chosen, kv_head].float()
            ) * 8**-0.5
            expected[row, heads] = torch.einsum(
                "ht,td->hd",
                torch.softmax(score, dim=-1).to(v.dtype),
                v[chosen, kv_head],
            ).to(expected.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qsa_sparse_gqa_matches_reference_on_cuda():
    torch.manual_seed(17)
    q = torch.randn(3, 24, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(4096, 2, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    rows = torch.randint(0, 4096, (3, 131), device="cuda", dtype=torch.int32)
    counts = torch.tensor([131, 97, 41], device="cuda", dtype=torch.int32)
    actual = qsa_sparse_gqa(q, k, v, rows, counts, 256**-0.5)
    reference = qsa_sparse_gqa(
        q.cpu(), k.cpu(), v.cpu(), rows.cpu(), counts.cpu(), 256**-0.5
    )
    torch.testing.assert_close(actual.cpu(), reference, rtol=0.03, atol=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qsa_cuda_compaction_matches_cpu_with_short_tail():
    blocks = torch.tensor([[-1, -1], [1, 0], [0, -1]], dtype=torch.int32)
    positions = torch.tensor([0, 11, 6], dtype=torch.int64)
    expected_rows, expected_counts = _compact_expanded_selection(
        blocks,
        positions,
        compress_ratio=4,
        token_budget=8,
    )
    actual_rows, actual_counts = _compact_expanded_selection(
        blocks.cuda(),
        positions.cuda(),
        compress_ratio=4,
        token_budget=8,
    )
    assert torch.equal(actual_rows.cpu(), expected_rows)
    assert torch.equal(actual_counts.cpu(), expected_counts)
