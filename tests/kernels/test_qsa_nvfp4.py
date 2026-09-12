"""QSA sparse attention over the packed NVFP4 KV tier."""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.triton.kv_nvfp4 import quantize_nvfp4_to_cache
from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton attention needs CUDA"
)

PAGE = 64
HEAD_DIM = 64


def _layout(rows: int, topk: int):
    block_table = torch.tensor(
        [[0, 1, 2], [2, 1, 0]], dtype=torch.int32, device="cuda"
    ).contiguous()
    indices = (
        torch.arange(topk, dtype=torch.int32, device="cuda")[None, :] + 32
    ).repeat(rows, 1).contiguous()
    token_to_req = torch.arange(rows, dtype=torch.int32, device="cuda") % 2
    return indices, block_table, token_to_req.contiguous()


def _decode(codes: torch.Tensor, row_scale: torch.Tensor, block_scale: torch.Tensor):
    """Decode with torch, independently of the Triton read path under test."""
    lo = (codes & 15).to(torch.long)
    hi = (codes >> 4).to(torch.long)
    code = torch.stack((lo, hi), dim=-1).reshape(*codes.shape[:-1], HEAD_DIM)
    magnitude = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=codes.device
    )[code & 7]
    value = torch.where(code < 8, magnitude, -magnitude)
    block = block_scale.view(torch.float8_e4m3fn).to(torch.float32)
    return (value * block.repeat_interleave(16, dim=-1) * row_scale.unsqueeze(-1)).to(
        torch.bfloat16
    )


@pytest.mark.parametrize(("rows", "kv_heads"), [(1, 1), (16, 2)])
def test_qsa_nvfp4_matches_its_bf16_decode(rows, kv_heads):
    """Page-table indirection must address the same packed slot and both scale tiers."""
    torch.manual_seed(19)
    pages, query_heads, slots = 3, 2 * kv_heads, 3 * PAGE
    k = torch.randn(slots, kv_heads * HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k) * 0.25
    out_loc = torch.arange(slots, dtype=torch.int32, device="cuda")
    code_shape = (slots, kv_heads, HEAD_DIM // 2)
    block_shape = (slots, kv_heads, HEAD_DIM // 16)
    k_codes = torch.zeros(code_shape, dtype=torch.uint8, device="cuda")
    v_codes = torch.zeros_like(k_codes)
    k_scale = torch.zeros((slots, kv_heads), dtype=torch.float32, device="cuda")
    v_scale = torch.zeros_like(k_scale)
    k_block = torch.zeros(block_shape, dtype=torch.uint8, device="cuda")
    v_block = torch.zeros_like(k_block)
    quantize_nvfp4_to_cache(
        k, v, out_loc, k_codes, v_codes, k_scale, v_scale, k_block, v_block
    )
    torch.cuda.synchronize()

    q = torch.randn(rows, query_heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    indices, block_table, token_to_req = _layout(rows, topk=64)
    packed_shape = (pages, PAGE, kv_heads, HEAD_DIM // 2)
    dense_shape = (pages, PAGE, kv_heads, HEAD_DIM)
    got = qsa_sparse_paged_attention(
        q,
        k_codes.view(packed_shape),
        v_codes.view(packed_shape),
        indices,
        block_table,
        token_to_req,
        k_scale=k_scale,
        v_scale=v_scale,
        kv_quant="nvfp4",
        k_block_scale=k_block,
        v_block_scale=v_block,
    )
    want = qsa_sparse_paged_attention(
        q,
        _decode(k_codes, k_scale, k_block).view(dense_shape),
        _decode(v_codes, v_scale, v_block).view(dense_shape),
        indices,
        block_table,
        token_to_req,
    )
    assert torch.equal(got, want), (
        "NVFP4 QSA attend diverged from its bf16 decode (max diff "
        f"{(got.float() - want.float()).abs().max().item():.3e})"
    )


def test_qsa_nvfp4_requires_both_block_scale_tensors():
    q = torch.zeros(1, 2, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    codes = torch.zeros(2, PAGE, 1, HEAD_DIM // 2, device="cuda", dtype=torch.uint8)
    scales = torch.ones(2 * PAGE, 1, device="cuda", dtype=torch.float32)
    indices = torch.zeros(1, 8, device="cuda", dtype=torch.int32)
    block_table = torch.zeros(1, 2, device="cuda", dtype=torch.int32)
    token_to_req = torch.zeros(1, device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="NVFP4"):
        qsa_sparse_paged_attention(
            q,
            codes,
            codes,
            indices,
            block_table,
            token_to_req,
            k_scale=scales,
            v_scale=scales,
            kv_quant="nvfp4",
        )


def test_qsa_nvfp4_splitk_reads_value_rows():
    """A zero query makes the split-K result the selected V-row average."""
    torch.manual_seed(23)
    slots = 3 * PAGE
    v = torch.randn(slots, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    codes = torch.zeros((slots, 1, HEAD_DIM // 2), dtype=torch.uint8, device="cuda")
    row = torch.zeros((slots, 1), dtype=torch.float32, device="cuda")
    block = torch.zeros((slots, 1, HEAD_DIM // 16), dtype=torch.uint8, device="cuda")
    quantize_nvfp4_to_cache(
        v, v, torch.arange(slots, dtype=torch.int32, device="cuda"), codes, codes,
        row, row, block, block,
    )
    q = torch.zeros(1, 2, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    indices, block_table, token_to_req = _layout(1, topk=64)
    args = (indices, block_table, token_to_req)
    got = qsa_sparse_paged_attention(
        q, codes.view(3, PAGE, 1, HEAD_DIM // 2), codes.view(3, PAGE, 1, HEAD_DIM // 2),
        *args, k_scale=row, v_scale=row, kv_quant="nvfp4", k_block_scale=block,
        v_block_scale=block,
    )
    ref = _decode(codes, row, block).view(3, PAGE, 1, HEAD_DIM)
    want = qsa_sparse_paged_attention(q, ref, ref, *args)
    assert torch.equal(got, want)
