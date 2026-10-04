"""Packed KV row formats (``kvcache/dsv4/v41_row_format.py`` + ``kernel/triton/dsv41/pack.py``) against a torch
transcription of the reference quantizers in DeepSeek-V4.1's ``inference/kernel.py``: the round-tripped
value the pool hands back must equal what the reference bakes into its bf16 cache, bit for bit."""

from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.dsv4.v41_row_format import BF16, FP4_E4M3_B16, FP4_E8M0_B32, FP8_E8M0_B32, RowFormat

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _round_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even onto the signed e2m1 grid (the ``float4_e2m1fn`` cast)."""
    a = x.abs()
    grid = FP4_GRID.to(x.device)
    # nearest grid point; ties go to the even-mantissa neighbour (0, 1, 2, 4 are even)
    dist = (a[..., None] - grid).abs()
    lo = torch.searchsorted(grid, a.reshape(-1).contiguous(), right=True).reshape(a.shape).clamp(1, 7)
    below, above = grid[lo - 1], grid[lo]
    mid = (below + above) / 2
    even_above = torch.isin(above, torch.tensor([0.0, 1.0, 2.0, 4.0], device=x.device))
    pick_above = (a > mid) | ((a == mid) & even_above)
    r = torch.where(pick_above, above, below)
    r = torch.where(a >= 6.0, torch.full_like(r, 6.0), r)
    del dist
    return torch.copysign(r, x)


def reference_roundtrip(x: torch.Tensor, fmt: RowFormat) -> torch.Tensor:
    """``act_quant`` / ``fp4_act_quant`` with ``inplace=True``: quant + dequant back to bf16."""
    m, d = x.shape
    if fmt is BF16:
        return x.to(torch.bfloat16)
    g = x.float().view(m, d // fmt.group, fmt.group)
    amax = g.abs().amax(dim=-1)
    if fmt is FP8_E8M0_B32:
        s = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-4) / 448.0)))
        q = (g / s[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    elif fmt is FP4_E8M0_B32:
        s = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(6.0 * 2.0**-126) / 6.0)))
        q = _round_fp4((g / s[..., None]).clamp(-6.0, 6.0))
    elif fmt is FP4_E4M3_B16:
        s = (amax.clamp_min(6.0 * 2.0**-9) / 6.0).to(torch.float8_e4m3fn).float()
        q = _round_fp4((g / s[..., None]).clamp(-6.0, 6.0))
    else:
        raise AssertionError(fmt)
    return (q * s[..., None]).view(m, d).to(torch.bfloat16)


@pytest.mark.parametrize("fmt", [BF16, FP8_E8M0_B32, FP4_E4M3_B16, FP4_E8M0_B32], ids=lambda f: f.name)
@pytest.mark.parametrize("dim", [128, 512])
def test_roundtrip_matches_the_reference_quantizer(fmt: RowFormat, dim: int):
    from freetoken.kernel.triton.dsv41.pack import pack_rows, unpack_rows

    torch.manual_seed(0)
    m = 300
    # mixed magnitudes, exact zeros, a tiny group and a huge one
    x = torch.randn(m, dim, device="cuda") * torch.exp2(torch.randint(-8, 6, (m, 1), device="cuda").float())
    x[7] = 0
    x[11, : fmt.group or 32] *= 1e-6
    x[13, -32:] = 400.0
    x = x.to(torch.bfloat16)

    packed = pack_rows(x, fmt)
    assert packed.shape == (m, fmt.row_bytes(dim)) and packed.dtype == torch.uint8
    got = unpack_rows(packed, fmt, dim)
    want = reference_roundtrip(x, fmt)
    assert torch.equal(got, want), f"{fmt.name}: {(got.float() - want.float()).abs().max().item()}"


def test_scatter_into_pool_rows_and_gather_back():
    from freetoken.kernel.triton.dsv41.pack import pack_rows, unpack_rows

    torch.manual_seed(1)
    fmt, dim, rows = FP4_E4M3_B16, 512, 64
    pool = torch.zeros((rows, fmt.row_bytes(dim)), dtype=torch.uint8, device="cuda")
    x = torch.randn(5, dim, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([3, 60, 0, 17, 42], device="cuda", dtype=torch.int64)
    assert pack_rows(x, fmt, pool=pool, row_ids=ids) is pool
    got = unpack_rows(pool, fmt, dim, row_ids=ids.to(torch.int32))
    assert torch.equal(got, reference_roundtrip(x, fmt))
    untouched = torch.ones(rows, dtype=torch.bool, device="cuda")
    untouched[ids] = False
    assert int(pool[untouched].sum()) == 0


def test_e4m3_encode_helper_matches_torch_bits():
    """``e4m3_f32_to_u8`` (the pre-sm_89 scale encoder) on every finite e4m3 code."""
    import triton
    import triton.language as tl

    from freetoken.kernel.triton.e4m3_compat import e4m3_f32_to_u8

    @triton.jit
    def _enc(x_ptr, o_ptr, N: tl.constexpr):
        offs = tl.arange(0, N)
        tl.store(o_ptr + offs, e4m3_f32_to_u8(tl.load(x_ptr + offs)))

    codes = torch.arange(256, dtype=torch.uint8, device="cuda")
    finite = (codes & 0x7F) != 0x7F  # skip the NaN codes
    vals = codes.view(torch.float8_e4m3fn).float()
    vals = torch.where(finite, vals, torch.zeros_like(vals))
    out = torch.empty(256, dtype=torch.uint8, device="cuda")
    _enc[(1,)](vals, out, N=256)
    want = torch.where(finite, codes, torch.zeros_like(codes))
    # -0.0 encodes to 0x80 in torch and 0x80 here; +0.0 -> 0x00
    assert torch.equal(out, want)
