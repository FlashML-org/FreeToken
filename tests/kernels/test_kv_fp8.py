"""FP8 (e4m3) KV quantization: the store kernel against an independent oracle.

Expectations come from a brute-force nearest-code search over an e4m3 table decoded
from first principles (sign / exponent / mantissa), NOT from the kernels' own
rounding helpers -- so a drift in ``round_e4m3`` or in the new ``e4m3_f32_to_u8``
encoder fails here instead of being blessed by itself.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kernel.triton.kv_quant import (
    KV_SCALE_DTYPE,
    alloc_codes,
    codes_to_f32,
    kv_codes_dtype,
    quantize_kv_to_cache,
)

DEV = torch.device("cuda")
FP8_MAX = 448.0
CODE_448 = 0x7E


def _init_tp() -> None:
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


@pytest.fixture(autouse=True)
def _tp():
    _init_tp()


def _e4m3_table() -> dict[int, float]:
    """Every finite e4m3fn value, decoded by hand from its bit fields.

    code = S EEEE MMM, exponent bias 7: normal (E>0) -> (-1)^S * 2^(E-7) * (1+M/8);
    subnormal (E==0) -> (-1)^S * 2^-6 * (M/8). E==15 & M==7 (0x7F/0xFF) is the NaN
    pattern and is dropped, which caps the format at +-448.
    """
    values: dict[int, float] = {}
    for code in range(256):
        sign = -1.0 if (code >> 7) & 1 else 1.0
        exp, mant = (code >> 3) & 0x0F, code & 0x07
        if exp == 15 and mant == 7:
            continue
        values[code] = sign * (
            (2.0**-6) * (mant / 8.0) if exp == 0 else (2.0 ** (exp - 7)) * (1.0 + mant / 8.0)
        )
    return values


E4M3_VALUES = _e4m3_table()
CODES = torch.tensor(sorted(E4M3_VALUES), dtype=torch.int32)
GRID = torch.tensor([E4M3_VALUES[c] for c in CODES.tolist()], dtype=torch.float64)
NEGATIVE_ZERO = 0x80


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    """A code buffer as raw bytes (the fp8 view on sm_89+, uint8 below it)."""
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def _canonical_zero(codes: torch.Tensor) -> torch.Tensor:
    """Fold -0.0 (0x80) onto +0.0 (0x00).

    The two store paths legitimately disagree on zero's sign bit: sm_89+ converts with
    ``x.to(fp8e4nv)`` (keeps it), the emulated path rounds first and
    ``round_e4m3(-0.0)`` is documented to return +0.0. They decode to the same number,
    so a test that compares bytes must not care which one it got.
    """
    return torch.where(codes == NEGATIVE_ZERO, torch.zeros_like(codes), codes)


def _ref_codes(x: torch.Tensor) -> torch.Tensor:
    """Independent encoder: nearest grid value by brute force, ties resolved to the
    EVEN code (RNE). ``x`` is any shape; returns int32 codes."""
    grid = GRID.to(x.device)
    codes = CODES.to(x.device)
    dist = (x.to(torch.float64).reshape(-1, 1) - grid.unsqueeze(0)).abs()
    near = dist == dist.min(dim=-1, keepdim=True).values
    big = 1 << 30
    codes = codes.unsqueeze(0).expand_as(dist)
    any_code = torch.where(near, codes, torch.full_like(codes, big))
    even = near & (codes % 2 == 0)
    even_code = torch.where(even, codes, torch.full_like(codes, big))
    has_even = (even_code < big).any(dim=-1)
    return torch.where(has_even, even_code.min(dim=-1).values, any_code.min(dim=-1).values)


def _store(rows_k: torch.Tensor, rows_v: torch.Tensor):
    """Quantize ``[T, heads, dim]`` rows into a fresh code buffer, returning
    ``(k_codes, v_codes, k_scales, v_scales)``."""
    _init_tp()
    tokens, heads, dim = rows_k.shape
    k_cache = alloc_codes((tokens, heads, dim), DEV)
    v_cache = alloc_codes((tokens, heads, dim), DEV)
    k_scale = torch.zeros((tokens, heads), dtype=KV_SCALE_DTYPE, device=DEV)
    v_scale = torch.zeros((tokens, heads), dtype=KV_SCALE_DTYPE, device=DEV)
    quantize_kv_to_cache(
        k=rows_k.reshape(tokens, -1),
        v=rows_v.reshape(tokens, -1),
        out_loc=torch.arange(tokens, dtype=torch.int32, device=DEV),
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.cuda.synchronize()
    return k_cache, v_cache, k_scale, v_scale


def test_scale_is_amax_over_e4m3_max():
    torch.manual_seed(0)
    k = (torch.randn(4, 2, 64, device=DEV, dtype=torch.bfloat16) * 3.0).to(torch.bfloat16)
    v = torch.randn_like(k)
    _, _, k_scale, v_scale = _store(k, v)
    # The kernel widens to fp32 before the amax/divide, so the reference must too:
    # a bf16 intermediate would round the expected scale and hide a precision bug.
    torch.testing.assert_close(
        k_scale, k.abs().to(torch.float32).amax(dim=-1) / FP8_MAX, rtol=1e-6, atol=0
    )
    torch.testing.assert_close(
        v_scale, v.abs().to(torch.float32).amax(dim=-1) / FP8_MAX, rtol=1e-6, atol=0
    )


def test_codes_match_the_reference_quantizer_and_reconstruction_is_close():
    torch.manual_seed(1)
    tokens, heads, dim = 8, 3, 128
    # Feed the rows as a qkv slice, the way the attention backends really do: the
    # row pitch is then wider than the row, which the store kernel must honour.
    qkv = torch.randn(tokens, heads * dim * 3, device=DEV, dtype=torch.bfloat16)
    qkv[:, heads * dim : 2 * heads * dim] *= 5.0  # K: large magnitude
    qkv[:, 2 * heads * dim :] *= 0.01  # V: subnormal end of the e4m3 grid
    _, k_rows, v_rows = qkv.split(heads * dim, dim=-1)
    k = k_rows.view(tokens, heads, dim)
    v = v_rows.view(tokens, heads, dim).clamp(-FP8_MAX, FP8_MAX)
    k_cache, v_cache, k_scale, v_scale = _store(k, v)

    for rows, cache, scale in ((k, k_cache, k_scale), (v, v_cache, v_scale)):
        f32 = rows.to(torch.float32)
        ref_scale = f32.abs().amax(dim=-1, keepdim=True) / FP8_MAX
        expected = _ref_codes((f32 / ref_scale).clamp(-FP8_MAX, FP8_MAX))
        got = _canonical_zero(_as_bytes(cache).reshape(-1).to(torch.int32))
        expected = _canonical_zero(expected)
        assert torch.equal(got, expected), (
            f"{int((got != expected).sum())} code mismatches of {got.numel()}"
        )
        deq = codes_to_f32(cache) * scale.unsqueeze(-1)
        err = (deq - f32).abs().max(dim=-1).values
        assert torch.all(err <= 0.08 * f32.abs().amax(dim=-1)), float(err.max())


def test_encoder_inverts_the_grid_through_the_scale_one_path():
    """Pack every e4m3 grid value into a row that also holds 448.0: the row scale is
    then exactly 1.0, so each stored byte IS the encoder's answer for that value."""
    dim, per_row = 256, 255
    pairs = sorted(E4M3_VALUES.items())
    tokens = -(-len(pairs) // per_row)
    rows = torch.zeros(tokens, 1, dim, dtype=torch.float32)
    expected = torch.zeros(tokens, dim, dtype=torch.uint8)
    for t in range(tokens):
        rows[t, 0, 0] = FP8_MAX  # the amax anchor
        expected[t, 0] = CODE_448
        for j in range(per_row):
            i = t * per_row + j
            if i >= len(pairs):
                break
            code, value = pairs[i]
            rows[t, 0, j + 1] = value
            expected[t, j + 1] = code

    k_cache, _, k_scale, _ = _store(
        rows.to(DEV, dtype=torch.bfloat16), torch.zeros(tokens, 1, dim, dtype=torch.bfloat16)
    )
    assert torch.equal(k_scale[:, 0], torch.ones_like(k_scale[:, 0]))
    got = _canonical_zero(_as_bytes(k_cache)[:, 0, :])
    want = _canonical_zero(expected.to(DEV))
    bad = got != want
    assert not bad.any().item(), (
        f"{int(bad.sum())} of {want.numel()} grid values round-tripped wrong; "
        f"first at {bad.nonzero()[0].tolist()}: expected "
        f"{want[bad][0].item():#x} got {got[bad][0].item():#x}"
    )


def test_zero_row_stays_finite_and_exact():
    k = torch.zeros(2, 2, 32, device=DEV, dtype=torch.bfloat16)
    k_cache, _, k_scale, _ = _store(k, k.clone())
    assert torch.isfinite(k_scale).all()
    assert (k_scale > 0).all(), "an all-zero row must still store a usable scale"
    assert (codes_to_f32(k_cache) == 0).all()


def test_codes_are_plain_bytes_and_the_kernel_decode_matches_torch():
    """The KV codec never puts an fp8 type in front of Triton.

    Codes live in a uint8 buffer on EVERY architecture and the kernel widens them with
    the software decoder, while the expectation below is torch's OWN e4m3 cast of those
    very bytes. That pins the one claim the design rests on: byte for byte, the
    software decode reads what a native fp8 unit would -- which is what lets the
    quantized cache behave identically on GPUs where the fp8 type is illegal.
    """
    import triton
    import triton.language as tl

    from freetoken.kernel.triton.e4m3_compat import kv_load_e4m3_tile_f32

    rows = torch.randn(5, 2, 64, device=DEV, dtype=torch.bfloat16) * 3.0
    k_cache, _, _, _ = _store(rows, rows.clone())
    assert kv_codes_dtype() is torch.uint8, "keep the fp8 type out of kernel signatures"
    assert k_cache.dtype is torch.uint8 and k_cache.element_size() == 1
    want = codes_to_f32(k_cache)  # torch reinterprets these bytes as e4m3 and casts

    @triton.jit
    def read_out(codes_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        got = kv_load_e4m3_tile_f32(codes_ptr + offs, offs < n)
        tl.store(out_ptr + offs, got, mask=offs < n)

    n = k_cache.numel()
    out = torch.zeros(n, dtype=torch.float32, device=DEV)
    read_out[(triton.cdiv(n, 256),)](k_cache, out, n, BLOCK=256)
    flat = want.reshape(-1)
    assert torch.equal(out, flat), (
        f"{int((out != flat).sum())} of {n} codes decode differently from torch's cast"
    )


def test_kv_codec_has_no_arch_or_dtype_branch():
    """The two rejected designs had one thing in common: they chose an arm.

    The compile-time fp8-native probe answers a question the allocator already
    answered -- and on one box answered wrongly -- while a test against the pointer's
    element type is NOT pruned by triton, so the dead arm still gets type-checked.
    That is how an int mask fill ended up in front of an fp8 pointer, twice. Both
    codecs are straight-line now; pin that, plus the identifiers of the two rejected
    designs, so neither creeps back in as a "fast path".
    """
    import inspect

    from freetoken.kernel.triton import kv_quant
    from freetoken.kernel.triton.e4m3_compat import kv_load_e4m3_tile_f32

    for obj in (kv_load_e4m3_tile_f32, kv_quant._kv_quant_scatter_kernel):
        src = inspect.getsource(getattr(obj, "fn", obj))
        for banned in ("e4m3_native", "dtype.element_ty"):
            assert banned not in src, f"{banned} is back in {obj.__name__}"
        body = src.split('"""')[-1].splitlines()
        arms = [
            line.strip() for line in body
            if line.strip().startswith(("if ", "elif ", "else"))
        ]
        assert not arms, f"{obj.__name__} must not branch: {arms}"
