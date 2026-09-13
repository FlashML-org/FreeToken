"""QSA sparse attention reading an fp8 KV pool (kernel/triton/qsa/attend.py).

The oracle is the SAME data in a bf16 cache, not a tolerance. The kernel widens the e4m3
codes to fp32, multiplies by the row scale, and casts back to the query dtype before
``tl.dot`` -- so a cache holding ``c * s`` and a cache holding ``c`` with scale ``s`` feed
the matmul bit-identical operands, and the two runs must agree bit for bit. Anything wrong
in scale indexing (the page/offset -> slot arithmetic), in the K-vs-V broadcast direction,
or in the masked-fill values shows up as a mismatch instead of slop inside an epsilon.

Two quantizer variants are covered:
  * hand-made codes with power-of-two scales, which also keeps the real values on the e4m3
    grid, so the fp8 buffer, the bf16 buffer and the test agree on the numbers;
  * the fused store kernel the pools actually call (``kv_quant.quantize_kv_to_cache``,
    amax/448 scales) over ordinary gaussian rows -- whose quantization error against the
    ORIGINAL rows is bounded separately, because that part is quality, not exactness.

Each parametrization also covers a different launcher profile: a small
``rows * kv_heads`` pushes it into split-K (partials + merge kernel), a large one takes
the direct-write path.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.triton.kv_quant import (
    FP8,
    alloc_codes,
    codes_to_f32,
    kv_codes_dtype,
    quantize_kv_to_cache,
)
from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton attention needs CUDA"
)

PAGE = 64  # the page size the qsa_sparse backend registers
HEAD_DIM = 64


def _on_grid(shape, device, generator) -> torch.Tensor:
    """Values of the form +-m * 2^e with m in [8, 16): four significant bits, i.e. every
    one of them is representable in e4m3 AND in bf16, so encoding is lossless and both
    caches hold the very same real numbers."""
    mantissa = torch.randint(
        8, 16, shape, device=device, generator=generator, dtype=torch.int32
    )
    exponent = torch.randint(
        -6, 5, shape, device=device, generator=generator, dtype=torch.int32
    )
    sign = torch.where(
        torch.randint(0, 2, shape, device=device, generator=generator).bool(), 1.0, -1.0
    )
    return sign * mantissa.to(torch.float32) * torch.pow(2.0, exponent.to(torch.float32))


def _code_buffer(values_f32: torch.Tensor) -> torch.Tensor:
    """Encode e4m3-exact fp32 values into a buffer of the pool's code dtype."""
    codes = values_f32.to(FP8)
    if kv_codes_dtype() is torch.uint8:
        codes = codes.view(torch.uint8)
    assert codes.dtype == kv_codes_dtype(), codes.dtype
    return codes.contiguous()


def _layout(rows: int, topk: int, num_req: int):
    """(indices, block_table, token_to_req) over three 64-token pages.

    Tokens 32..31+topk straddle pages 0 and 1, and the two requests map their logical
    pages to physical ones in OPPOSITE order -- a page-table or page/offset slip cannot
    hide behind a symmetric fixture.
    """
    device = "cuda"
    block_table = torch.tensor(
        [[0, 1, 2], [2, 1, 0]], dtype=torch.int32, device=device
    )[:num_req].contiguous()
    indices = (
        torch.arange(topk, dtype=torch.int32, device=device)[None, :] + 32
    ).repeat(rows, 1).contiguous()
    token_to_req = (
        torch.arange(rows, dtype=torch.int32, device=device) % num_req
    ).contiguous()
    return indices, block_table, token_to_req


def _run(codes, scales, q, indices, block_table, token_to_req):
    return qsa_sparse_paged_attention(
        q,
        codes[0],
        codes[1],
        indices,
        block_table,
        token_to_req,
        k_scale=scales[0],
        v_scale=scales[1],
    )



# rows 1 x 1 kv head -> base_programs 1  -> BLOCK_N 16, 4 tiles -> NUM_SPLITS 4 (split-K).
# rows 16 x 2        -> base_programs 32 -> BLOCK_N 64, 1 tile  -> NUM_SPLITS 1 (direct).
@pytest.mark.parametrize(("rows", "kv_heads", "topk"), [(1, 1, 64), (16, 2, 64)])
def test_fp8_codes_match_the_bf16_cache_bit_for_bit(rows, kv_heads, topk):
    torch.manual_seed(3)
    device = torch.device("cuda")
    num_pages, num_query_heads = 3, 2 * kv_heads
    slots = num_pages * PAGE

    k_values = _on_grid((num_pages, PAGE, kv_heads, HEAD_DIM), device, None)
    v_values = _on_grid((num_pages, PAGE, kv_heads, HEAD_DIM), device, None)
    # Alternate the scale exponent by slot -- a scale broadcast along the wrong axis then
    # changes the answer instead of cancelling out -- and give K and V opposite parities.
    parity = torch.arange(slots, device=device, dtype=torch.float32) % 2
    k_scale = (
        torch.pow(2.0, parity * 5 - 3).unsqueeze(-1).expand(slots, kv_heads).contiguous()
    )
    v_scale = (
        torch.pow(2.0, (1 - parity) * 4 - 2).unsqueeze(-1).expand(slots, kv_heads).contiguous()
    )
    k_ref = (k_values * k_scale.view(num_pages, PAGE, kv_heads, 1)).to(torch.bfloat16)
    v_ref = (v_values * v_scale.view(num_pages, PAGE, kv_heads, 1)).to(torch.bfloat16)

    q = torch.randn(rows, num_query_heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    indices, block_table, token_to_req = _layout(rows, topk, num_req=2)

    got = _run(
        (_code_buffer(k_values), _code_buffer(v_values)),
        (k_scale, v_scale),
        q,
        indices,
        block_table,
        token_to_req,
    )
    want = qsa_sparse_paged_attention(
        q, k_ref, v_ref, indices, block_table, token_to_req
    )
    assert torch.equal(got, want), (
        "fp8 QSA attend diverged from the same data in a bf16 cache (max diff "
        f"{(got.float() - want.float()).abs().max().item():.3e})"
    )


@pytest.mark.parametrize(("rows", "kv_heads", "topk"), [(1, 1, 64), (16, 2, 64)])
def test_pool_writer_codes_match_the_bf16_cache_bit_for_bit(rows, kv_heads, topk):
    """Ordinary rows through the fused quantize+scatter writer store_kv calls: the
    codes/scales it produces, decoded on the host with torch's own e4m3 cast (never the
    decoder under test), must feed the dot identical operands."""
    torch.manual_seed(5)
    device = torch.device("cuda")
    num_pages, num_query_heads = 3, 2 * kv_heads
    slots = num_pages * PAGE

    k_rows = torch.randn(slots, kv_heads * HEAD_DIM, device=device, dtype=torch.float32)
    v_rows = torch.randn(slots, kv_heads * HEAD_DIM, device=device, dtype=torch.float32)
    # Wide amplitude spread: every 8th row carries a 64x outlier. That is what a per-row
    # amax scale exists to absorb, and what a scale read from the wrong slot blows up on.
    outlier = (torch.arange(slots, device=device) % 8 == 0).unsqueeze(-1)
    k_rows = k_rows * torch.where(outlier, 64.0, 1.0)
    v_rows = v_rows * torch.where(outlier.flip(0), 32.0, 0.5)

    k_flat = alloc_codes((slots, kv_heads, HEAD_DIM), device)
    v_flat = alloc_codes((slots, kv_heads, HEAD_DIM), device)
    k_scale = torch.zeros((slots, kv_heads), dtype=torch.float32, device=device)
    v_scale = torch.zeros_like(k_scale)
    quantize_kv_to_cache(
        k=k_rows,
        v=v_rows,
        out_loc=torch.arange(slots, dtype=torch.int32, device=device),
        k_cache=k_flat,
        v_cache=v_flat,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.cuda.synchronize()

    k_ref = (codes_to_f32(k_flat) * k_scale.unsqueeze(-1)).to(torch.bfloat16)
    v_ref = (codes_to_f32(v_flat) * v_scale.unsqueeze(-1)).to(torch.bfloat16)
    q = torch.randn(rows, num_query_heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    indices, block_table, token_to_req = _layout(rows, topk, num_req=2)

    shape = (num_pages, PAGE, kv_heads, HEAD_DIM)
    got = _run(
        (k_flat.view(shape), v_flat.view(shape)),
        (k_scale, v_scale),
        q,
        indices,
        block_table,
        token_to_req,
    )
    want = qsa_sparse_paged_attention(
        q, k_ref.view(shape), v_ref.view(shape), indices, block_table, token_to_req
    )
    assert torch.equal(got, want), (
        "fused-writer codes diverged (max diff "
        f"{(got.float() - want.float()).abs().max().item():.3e})"
    )

    # Quantization QUALITY, bounded per row: e4m3's half-ulp is 2^-4 of the binade a
    # value lands in, and the writer scales each row so its amax sits at 448.
    for source, codes, scales in ((k_rows, k_flat, k_scale), (v_rows, v_flat, v_scale)):
        assert torch.isfinite(scales).all() and (scales > 0).all()
        decoded = codes_to_f32(codes) * scales.unsqueeze(-1)
        original = source.view(slots, kv_heads, HEAD_DIM)
        amax = original.abs().amax(dim=-1, keepdim=True)
        rel = ((decoded - original).abs() / amax.clamp_min(1e-9)).max()
        assert rel.item() <= 0.07, f"e4m3 grid error {rel.item():.4f} above its half-ulp"


@pytest.mark.parametrize("which", ["k_only", "v_only", "bf16_cache", "wrong_shape"])
def test_scale_arguments_are_validated(which):
    """A half-supplied or mismatched scale pair must fail loudly: attending over raw
    codes while believing they are bf16 is the exact failure mode this feature cannot be
    allowed to have, and it produces plausible garbage rather than an error."""
    device = "cuda"
    kv_heads, rows = 1, 1
    k = torch.zeros(2, PAGE, kv_heads, HEAD_DIM, device=device, dtype=kv_codes_dtype())
    v = torch.zeros_like(k)
    scale = torch.ones(2 * PAGE, kv_heads, dtype=torch.float32, device=device)
    q = torch.zeros(rows, 2 * kv_heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    indices = torch.zeros(rows, 8, dtype=torch.int32, device=device)
    block_table = torch.zeros(1, 2, dtype=torch.int32, device=device)
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=device)

    kwargs = {"k_scale": scale, "v_scale": scale}
    if which == "k_only":
        kwargs.pop("v_scale")
    elif which == "v_only":
        kwargs.pop("k_scale")
    elif which == "bf16_cache":
        k, v = k.to(torch.bfloat16), v.to(torch.bfloat16)
    else:
        kwargs["k_scale"] = scale[:-1]

    with pytest.raises(ValueError, match="QSA"):
        qsa_sparse_paged_attention(
            q, k, v, indices, block_table, token_to_req, **kwargs
        )


def test_qsa_scoring_refuses_fp8_operands():
    """The indexer's scoring dot is 16-bit only, and the wrapper has to say so.

    In the field an e4m3 ``q_index`` -- produced by a backend that took its scratch dtype
    from a pool reporting its STORE dtype -- surfaced as ``CompilationError: Unsupported
    rhs dtype fp8e4nv`` inside CUDA-graph capture: a dead scheduler and a stopped API
    server, forty seconds after the pool allocated fine. Same mistake, now stopped at
    the call with a name on it.
    """
    from freetoken.kernel.triton.qsa import qsa_mqa_paged

    device = "cuda"
    rows, heads, dim, pages, cmp_page = 2, 2, 64, 2, 16
    good = {
        "q": torch.zeros(rows, heads, dim, device=device, dtype=torch.bfloat16),
        "k_cache": torch.zeros(
            pages, cmp_page, 1, dim, device=device, dtype=torch.bfloat16
        ),
        "page_table": torch.zeros(1, pages, dtype=torch.int32, device=device),
        "token_to_req": torch.zeros(rows, dtype=torch.int32, device=device),
        "query_positions": torch.arange(rows, dtype=torch.int32, device=device),
        "sequence_lengths": torch.zeros(1, dtype=torch.int32, device=device),
        "compress_ratio": cmp_page,
        # Zero columns -> the wrapper returns before launching. This control proves the
        # new check still lets the dtype it exists to allow through.
        "logits": torch.zeros(rows, 0, dtype=torch.float32, device=device),
        "visible_blocks": torch.zeros(rows, dtype=torch.int32, device=device),
    }
    qsa_mqa_paged(**good)

    for name in ("q", "k_cache"):
        bad = dict(good)
        bad[name] = alloc_codes(tuple(good[name].shape), device)
        with pytest.raises(ValueError, match="16-bit only"):
            qsa_mqa_paged(**bad)

