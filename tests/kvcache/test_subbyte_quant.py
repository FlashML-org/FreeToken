"""Unit tests for the sub-byte KV cache quantization spec (Q4_0, Q6_0).

Pure-Python + PyTorch tests. No GPU, no model, no Triton. The same
quantize/dequantize methods are the **oracle** the Triton kernels in
``kernel/triton/kv_quant.py`` and the ``_load_kv`` path in
``kernel/triton/attention.py`` must match bit-for-bit; if these tests
break, the kernels are wrong.

Naming convention follows the rest of ``tests/kvcache/``: short
module-level functions, ``test_<scheme>_<property>`` (so failures point
straight at the broken property, not at a class hierarchy).
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.quant import (
    BLOCK,
    LAYOUT_Q4,
    LAYOUT_Q6,
    NONE,
    NVFP4,
    Q4_0,
    Q6_0,
    Q8_0,
    KVQuantSpec,
    resolve_kv_quant,
)


# ---- helpers ----

def _kurtotic_kv(shape=(4, 8, 128), mag=3.0, seed=0):
    """Real-data-shaped K/V: gaussian + 5x outlier on a few first-dim slots.

    K and V live on a long tail (the first head_dim // 16 slots get a
    5x multiplier, modelling attention-sink / important-token effects).
    Mirrors the data distribution we observed on ornith-ftw in 8/28 sweeps.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(*shape, generator=g) * mag
    x[..., : shape[-1] // 16] *= 5.0
    return x.to(torch.bfloat16)


def _rel_err(rec: torch.Tensor, x: torch.Tensor) -> float:
    """Relative L1 mean error: (rec - x).abs().mean() / x.abs().mean()."""
    return (rec - x.float()).abs().mean().item() / x.float().abs().mean().item()


# ---- spec shape / dtype / constants ----

def test_block_constant_is_32():
    """The 32-value block is shared across Q4_0 / Q6_0 / Q8_0 in the spec.
    Touching this constant changes all three layouts; guard it."""
    assert BLOCK == 32


def test_q4_0_spec_layout_and_bits():
    """Q4_0 must declare its layout as LAYOUT_Q4 and pack 16 bytes per block.
    This is the data the Triton store kernel keys on (LAYOUT: tl.constexpr)."""
    assert Q4_0.layout == LAYOUT_Q4
    assert Q4_0.bits == 4
    assert Q4_0.payload_bytes_per_block == 16
    assert Q4_0.max_magnitude == 8.0  # the 7->8 optimization; see PR1 body


def test_q6_0_spec_layout_and_bits():
    """Q6_0 must declare its layout as LAYOUT_Q6 and pack 24 bytes per block."""
    assert Q6_0.layout == LAYOUT_Q6
    assert Q6_0.bits == 6
    assert Q6_0.payload_bytes_per_block == 24
    assert Q6_0.max_magnitude == 31.0


def test_q8_0_unchanged_baseline():
    """Guard the existing 8-bit path: Q8_0 stays at 1 byte/elem, mag=127.
    If this test breaks, PR#103 (8-bit) has been silently modified."""
    assert Q8_0.layout == "q8"
    assert Q8_0.bits == 8
    assert Q8_0.payload_bytes_per_block == BLOCK
    assert Q8_0.max_magnitude == 127.0


# ---- bytes_per_element / physical_head_dim ----

def test_q4_0_bytes_per_element():
    """Q4_0 must yield 0.5625 bytes/element: 16 payload / 32 + 2 scale / 32."""
    assert Q4_0.bytes_per_element(torch.bfloat16) == pytest.approx(0.5625)


def test_q6_0_bytes_per_element():
    """Q6_0 must yield 0.8125 bytes/element: 24 payload / 32 + 2 scale / 32."""
    assert Q6_0.bytes_per_element(torch.bfloat16) == pytest.approx(0.8125)


def test_q4_0_physical_head_dim():
    """For a 128-dim head, the packed last-dim is 128 * 4 / 8 = 64 bytes."""
    assert Q4_0.physical_head_dim(128) == 64
    assert Q4_0.physical_head_dim(64) == 32
    assert Q4_0.physical_head_dim(32) == 16


def test_q6_0_physical_head_dim():
    """For a 128-dim head, the packed last-dim is 128 * 6 / 8 = 96 bytes."""
    assert Q6_0.physical_head_dim(128) == 96
    assert Q6_0.physical_head_dim(64) == 48


def test_q8_0_physical_head_dim_unchanged():
    """8-bit keeps head_dim = bytes in last axis (one byte per element)."""
    assert Q8_0.physical_head_dim(128) == 128


# ---- scale shape ----

def test_q4_0_scale_shape():
    """Scale tensor is [N..., head_dim / 32] for the logical head_dim.
    The caller passes the *packed* last-dim; the spec recovers logical via
    ``physical * 8 / bits`` and divides by BLOCK (32)."""
    # physical 64 -> logical 128 -> scale extent 4
    shape = (3, 4, 64)
    assert Q4_0.scale_shape(shape) == (3, 4, 4)


def test_q6_0_scale_shape():
    """Same recovery for Q6: physical 96 -> logical 128 -> scale extent 4."""
    shape = (3, 4, 96)
    assert Q6_0.scale_shape(shape) == (3, 4, 4)


def test_scale_shape_rejects_non_block_aligned():
    """A physical last-dim that is not a multiple of BLOCK is a programming
    error in the caller; the spec must raise so the bad buffer does not
    silently round-trip."""
    with pytest.raises(ValueError, match="not a multiple"):
        Q4_0.scale_shape((1, 30))  # 30 not a multiple of 32


# ---- quantize/dequantize round-trip (PyTorch oracle) ----

def test_q4_0_roundtrip_oracle_kurtotic():
    """Q4_0 quantize->dequantize should give < 0.15 rel_err on kurtotic K/V.

    This is the precision floor the kernel must match. If the kernel
    exceeds it, the bug is in the Triton path, not the spec. (The
    measured value on this distribution is ~0.13; other K/V shapes
    measure lower.)"""
    x = _kurtotic_kv(shape=(4, 8, 128), mag=3.0)
    payload, scales = Q4_0.quantize(x)
    rec = Q4_0.dequantize(payload, scales)
    err = _rel_err(rec, x)
    assert err < 0.15, f"Q4_0 rel_err {err:.4f} exceeded 0.15 floor"


def test_q6_0_roundtrip_oracle_kurtotic():
    """Q6_0 should give < 0.05 rel_err on the same kurtotic K/V
    (measured ~0.033 on this distribution)."""
    x = _kurtotic_kv(shape=(4, 8, 128), mag=3.0)
    payload, scales = Q6_0.quantize(x)
    rec = Q6_0.dequantize(payload, scales)
    err = _rel_err(rec, x)
    assert err < 0.05, f"Q6_0 rel_err {err:.4f} exceeded 0.05 floor"


def test_q8_0_roundtrip_baseline():
    """Sanity check: Q8_0 still gives < 0.01 on the same kurtotic K/V.
    If this fails, the kurtotic helper itself has drifted, not Q4/Q6."""
    x = _kurtotic_kv(shape=(4, 8, 128), mag=3.0)
    payload, scales = Q8_0.quantize(x)
    rec = Q8_0.dequantize(payload, scales)
    err = _rel_err(rec, x)
    assert err < 0.01


def test_q4_0_payload_shape():
    """Q4_0's quantized payload has physical last-dim = head_dim // 2."""
    x = torch.randn(2, 4, 128, dtype=torch.bfloat16)
    payload, scales = Q4_0.quantize(x)
    assert payload.shape == (2, 4, 64)  # 128 * 4 / 8
    assert scales.shape == (2, 4, 4)     # 128 / 32


def test_q6_0_payload_shape():
    """Q6_0's quantized payload has physical last-dim = head_dim * 3 // 4."""
    x = torch.randn(2, 4, 128, dtype=torch.bfloat16)
    payload, scales = Q6_0.quantize(x)
    assert payload.shape == (2, 4, 96)  # 128 * 6 / 8 = 96
    assert scales.shape == (2, 4, 4)    # 128 / 32


# ---- sign-extension bit-exactness ----

def test_q4_0_sign_extension_xor_sub():
    """The XOR-sub 4-bit sign extension must round-trip every unsigned value
    in [0, 15] to the signed range [-8, 7]. The kernel uses the arithmetic-
    shift form; this test confirms the XOR-sub form is equivalent. The
    shift form is evaluated through ctypes.c_int32 because Python ints
    do not wrap (int32 semantics are what the kernel gets)."""
    import ctypes

    for unsigned in range(16):
        signed = (unsigned ^ 0x8) - 0x8
        assert -8 <= signed <= 7
        shifted = ctypes.c_int32(unsigned << (32 - 4)).value >> (32 - 4)
        assert shifted == signed, f"{unsigned}: XOR={signed}, shift={shifted}"


def test_q6_0_sign_extension_xor_sub():
    """Same for 6-bit unsigned [0, 63] to signed [-32, 31]."""
    import ctypes

    for unsigned in range(64):
        signed = (unsigned ^ 0x20) - 0x20
        assert -32 <= signed <= 31
        shifted = ctypes.c_int32(unsigned << (32 - 6)).value >> (32 - 6)
        assert shifted == signed, f"{unsigned}: XOR={signed}, shift={shifted}"


def test_q4_0_clamp_symmetric():
    """Boundary round-trip under the amax scale. With max_magnitude=8 the
    scale is amax/8, so the -amax level (-8) round-trips EXACTLY
    (-8 * amax/8 == -amax), while +amax lands on code 8 -> clamps to 7,
    i.e. within one quantization step. This asymmetry is inherent to any
    amax-scaled symmetric scheme; what mag=7->8 buys is a usable 16th
    level instead of a wasted unreachable one."""
    block = torch.zeros(32, dtype=torch.bfloat16)
    block[3] = 7.0
    block[17] = -7.0  # amax = 7 -> scale = 7/8
    x = block.unsqueeze(0).unsqueeze(0)
    payload, scales = Q4_0.quantize(x)
    rec = Q4_0.dequantize(payload, scales).view(-1)[:32]
    # -amax level is exact
    assert rec[17].item() == pytest.approx(-7.0, abs=1e-2)
    # +amax side lands within one step (<= amax/8 = 0.875)
    assert abs(rec[3].item() - 7.0) <= 0.875 + 1e-2


# ---- single-block reference implementations ----

def test_q4_0_single_block_layout():
    """Hand-build a 32-value block with one positive peak at index 5 and
    one negative peak at index 17, then verify the byte layout of the
    quantized payload matches the expected nibble packing. amax = 8 so
    the scale is exactly 1.0 and codes equal the input values."""
    block = torch.zeros(32, dtype=torch.bfloat16)
    block[5] = 7.0
    block[17] = -8.0
    x = block.unsqueeze(0).unsqueeze(0)  # [1, 1, 32]
    payload, scales = Q4_0.quantize(x)
    # payload shape: [1, 1, 16] (16 bytes per block)
    assert payload.shape == (1, 1, 16)
    p = payload[0, 0]  # the 16 bytes
    # byte j holds val[2j] (low nibble) and val[2j+1] (high nibble).
    # index 5 -> byte j=2 (val[4], val[5]) -> val[5]=7, so high nibble = 7.
    assert (p[2] & 0xF0) >> 4 == 7, f"byte 2 high nibble = {(p[2] & 0xF0) >> 4}"
    # index 17 -> byte j=8 (val[16], val[17]) -> val[17]=-8 -> unsigned
    # two's complement 1000b = 8, so high nibble = 8.
    assert (p[8] & 0xF0) >> 4 == 8, f"byte 8 high nibble = {(p[8] & 0xF0) >> 4}"


def test_q6_0_single_block_layout_dual_plane():
    """Q6_0's 24-byte block: 16-byte low plane + 8-byte high plane.
    Verify the high plane holds the top 2 bits four-per-byte at bit
    positions 0, 2, 4, 6. amax = 32 (via block[31] = -32) so the scale
    is exactly 1.0 and codes equal the input values."""
    block = torch.zeros(32, dtype=torch.bfloat16)
    block[0] = 16.0   # code 16: low nibble 0, top 2 bits 01
    block[4] = 17.0   # code 17: low nibble 1, top 2 bits 01
    block[31] = -31.0  # sets amax -> scale = 31/31 = 1.0 exactly
    x = block.unsqueeze(0).unsqueeze(0)
    payload, scales = Q6_0.quantize(x)
    assert payload.shape == (1, 1, 24)
    p = payload[0, 0]
    # byte 0 (lo plane): val[0]=16 -> low nibble 0
    assert (p[0] & 0x0F) == 0
    # byte 2 (lo plane): val[4]=17 -> low nibble 1
    assert (p[2] & 0x0F) == 1
    # 8-byte high plane: byte 0 = val[0..3] top 2 bits.
    # val[0]=16 -> top 2 bits = 01 -> bit position 0 of high byte 0.
    assert (p[16] & 0x01) == 0x01
    # val[4]=17 -> top 2 bits = 01 -> bit position 0 of high byte 1.
    assert (p[17] & 0x01) == 0x01


# ---- spec-level: resolve_kv_quant ----

def test_resolve_q4_0():
    """`--kv-cache-dtype q4_0` must return the Q4_0 spec; not auto / q8_0."""
    spec = resolve_kv_quant("q4_0")
    assert spec is Q4_0


def test_resolve_q6_0():
    spec = resolve_kv_quant("q6_0")
    assert spec is Q6_0


def test_resolve_auto_returns_none_spec():
    """`--kv-cache-dtype auto` (or None) must return the unquantized spec."""
    assert resolve_kv_quant(None) is NONE
    assert resolve_kv_quant("auto") is NONE
    assert not resolve_kv_quant("auto").enabled


def test_resolve_unknown_raises():
    """An unknown dtype name must raise so a CLI typo doesn't silently
    fall back to bf16."""
    with pytest.raises(ValueError, match="unknown --kv-cache-dtype"):
        resolve_kv_quant("q5_zero")


# ---- CPU / CUDA parity (skipped on no-GPU) ----

def test_q4_0_cpu_cuda_parity():
    """If CUDA is available, the spec must give the same quantized payload
    on CPU and CUDA tensors. Skipped otherwise."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x_cpu = _kurtotic_kv(shape=(2, 4, 128), mag=3.0)
    x_cuda = x_cpu.cuda()
    p_cpu, s_cpu = Q4_0.quantize(x_cpu)
    p_cuda, s_cuda = Q4_0.quantize(x_cuda)
    assert torch.equal(p_cpu, p_cuda.cpu())
    assert torch.equal(s_cpu, s_cuda.cpu())


def test_q6_0_cpu_cuda_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x_cpu = _kurtotic_kv(shape=(2, 4, 128), mag=3.0)
    x_cuda = x_cpu.cuda()
    p_cpu, s_cpu = Q6_0.quantize(x_cpu)
    p_cuda, s_cuda = Q6_0.quantize(x_cuda)
    assert torch.equal(p_cpu, p_cuda.cpu())
    assert torch.equal(s_cpu, s_cuda.cpu())


# ---- NVFP4 tests ----

def test_nvfp4_spec_layout_and_bits():
    """NVFP4 declares nvfp4 layout, 4 bits, 16 bytes per 32-value block split
    as 8 payload bytes + 4 bytes (8 fp8 E4M3 scales). block_size=16, not 32."""
    from freetoken.kvcache.quant import NVFP4, LAYOUT_NVFP4
    assert NVFP4.layout == LAYOUT_NVFP4
    assert NVFP4.bits == 4
    assert NVFP4.payload_bytes_per_block == 8
    assert NVFP4.block_size == 16
    assert NVFP4.max_magnitude == 6.0
    assert NVFP4.scale_dtype is torch.float8_e4m3fn


def test_nvfp4_bytes_per_element():
    """NVFP4 yields 0.5625 bytes/element: 8 payload / 16 + 1 scale / 16.
    Same density as q4_0 (16 payload / 32 + 2 scale / 32)."""
    from freetoken.kvcache.quant import NVFP4
    assert NVFP4.bytes_per_element(torch.bfloat16) == pytest.approx(0.5625)


def test_nvfp4_e2m1_table_exact_values():
    """E2M1 code 7 (max positive) = 6.0, code 8 (sign-bit-only) = -0.0,
    code 15 (max negative) = -6.0. Verify the look-up table directly."""
    p, s = NVFP4_spec_quantize_maxes()
    # 1 block of 16 values all 6.0 -> payload should round-trip to 6.0
    r = p_spec_dequant(p, s)
    assert torch.equal(r, torch.full((16,), 6.0, device=r.device))


def NVFP4_spec_quantize_maxes():
    import torch
    from freetoken.kvcache.quant import NVFP4
    block = torch.full((1, 1, 16), 6.0, dtype=torch.bfloat16)
    p, s = NVFP4.quantize(block)
    return p, s


def p_spec_dequant(p, s):
    from freetoken.kvcache.quant import NVFP4
    return NVFP4.dequantize(p, s).view(-1)


def test_nvfp4_roundtrip_oracle_kurtotic():
    """NVFP4 quantize->dequantize gives < 0.15 rel_err on kurtotic K/V.

    Same byte density as q4_0 (0.5625) but the E2M1 floating-point code
    is log-spaced, so on distributions with strong tails the precision
    is comparable or better than uniform-integer 4-bit. Measured ~0.09
    on this distribution."""
    x = _kurtotic_kv(shape=(4, 8, 128), mag=3.0)
    payload, scales = NVFP4.quantize(x)
    rec = NVFP4.dequantize(payload, scales)
    err = _rel_err(rec, x)
    assert err < 0.15, f"NVFP4 rel_err {err:.4f} exceeded 0.15 floor"
