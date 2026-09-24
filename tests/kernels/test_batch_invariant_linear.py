"""The fixed-walk linear (``kernel/triton/batch_invariant_linear``): correct against F.linear / einsum,
and -- the property it exists for -- every output row depends on its input row alone (identical
across M, batch order and grouping), so cached KV never depends on the batch that produced it."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_batch_invariant_linear_matches_torch_and_is_row_independent(dtype):
    from freetoken.kernel.triton.batch_invariant_linear import batch_invariant_linear

    torch.manual_seed(0)
    x = torch.randn(300, 1280, device="cuda", dtype=dtype)
    w = torch.randn(200, 1280, device="cuda", dtype=dtype) * 0.05
    b = torch.randn(200, device="cuda", dtype=dtype)
    got = batch_invariant_linear(x, w, b)
    want = F.linear(x, w, b)
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(got.float(), want.float(), atol=tol * want.float().abs().max().item(), rtol=tol)
    for m in (1, 2, 33, 128):  # a prefix of the batch, a lone row, a permuted batch: identical rows
        assert torch.equal(batch_invariant_linear(x[:m], w, b), got[:m])
    perm = torch.randperm(300, device="cuda")
    assert torch.equal(batch_invariant_linear(x[perm], w, b)[perm.argsort()], got)
    assert torch.equal(batch_invariant_linear(x[7:8], w, b)[0], got[7])
    # an fp32 operand promotes to IEEE fp32 dots (no TF32): the bf16 x against an fp32 weight
    if dtype == torch.bfloat16:
        promoted = batch_invariant_linear(x, w.float())
        assert promoted.dtype == torch.float32
        torch.testing.assert_close(promoted, F.linear(x.float(), w.float()), atol=1e-3, rtol=1e-4)


def test_batch_invariant_grouped_linear_matches_einsum_and_is_row_independent():
    from freetoken.kernel.triton.batch_invariant_linear import batch_invariant_grouped_linear

    torch.manual_seed(1)
    T, G, d, r = 130, 4, 512, 256
    x = torch.randn(T, G, d, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(G * r, d, device="cuda", dtype=torch.bfloat16) * 0.05
    got = batch_invariant_grouped_linear(x, w)
    want = torch.einsum("tgd,grd->tgr", x, w.view(G, r, d)).flatten(1)
    torch.testing.assert_close(got.float(), want.float(), atol=2e-2 * want.float().abs().max().item(), rtol=2e-2)
    assert torch.equal(batch_invariant_grouped_linear(x[:1], w), got[:1]) and torch.equal(batch_invariant_grouped_linear(x[5:9], w), got[5:9])


def test_batch_invariant_linear_rejects_fp16_operands():
    """The contract is bf16 / fp32: fp16 would be computed as bf16 and silently lose mantissa bits."""
    from freetoken.kernel.triton.batch_invariant_linear import batch_invariant_grouped_linear, batch_invariant_linear

    x = torch.ones(2, 64, device="cuda", dtype=torch.float16) * 1.0009765625
    w = torch.eye(64, device="cuda", dtype=torch.float16)
    with pytest.raises(TypeError, match="bf16 or fp32"):
        batch_invariant_linear(x, w)
    with pytest.raises(TypeError, match="bf16 or fp32"):
        batch_invariant_grouped_linear(x.view(2, 1, 64), w)
    with pytest.raises(TypeError, match="out_dtype"):
        batch_invariant_linear(x.to(torch.bfloat16), w.to(torch.bfloat16), out_dtype=torch.float16)
