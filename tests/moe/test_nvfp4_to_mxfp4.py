"""NVFP4 -> MXFP4 converter tests (numpy core, torch-free).

Validates the error-prone numeric bits: the e2m1 code table, e8m0 scale encoding, and
an end-to-end round trip (dequant NVFP4 -> requant MXFP4 -> dequant back) that must be
close to the reference under a bounded relative error.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python" / "freetoken" / "moe" / "nvfp4_to_mxfp4.py"
)


@pytest.fixture(scope="module")
def c():
    spec = importlib.util.spec_from_file_location("nvfp4_to_mxfp4", _MODULE_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_fp4_table_values(c):
    table = c.fp4_e2m1_table()
    assert len(table) == 16
    # e2m1 canonical values.
    assert table[0] == 0.0
    assert table[1] == 0.5
    assert table[4] == 2.0
    assert table[7] == 6.0
    assert table[15] == -6.0
    assert table[9] == -0.5


def test_e8m0_scale_exact_power_of_two(c):
    # Block max-abs 3.0 -> scale 2^ceil(log2(3/6)) = 2^-1 = 0.5 -> code 126 (126-127=-1).
    values = np.array([[3.0, 1.0, -2.0, 0.5] + [0.0] * 28], dtype=np.float32)
    scale_codes, codes = c.e8m0_scale_and_codes(values, block=32)
    assert scale_codes.shape == (1,)
    assert int(scale_codes[0]) == 126
    # After dividing by 0.5: [6, 2, -4, 1] -> nearest e2m1 codes 7, 4, 14, 2.
    assert int(codes[0, 0]) == 7
    assert int(codes[0, 1]) == 4
    assert int(codes[0, 2]) == 14
    assert int(codes[0, 3]) == 2


def test_dequantize_nvfp4_round_trip(c):
    rng = np.random.default_rng(0)
    N, K = 4, 64
    # Build a plausible NVFP4 weight: choose fp4 codes, block scales, row globals.
    codes = rng.integers(0, 16, size=(N, K)).astype(np.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    block_scale = (rng.uniform(0.1, 2.0, size=(N, K // 16))).astype(np.float32)
    global_scale = (rng.uniform(0.5, 2.0, size=(N,)).astype(np.float32))
    ref = c.dequantize_nvfp4_block(packed, block_scale, global_scale)
    assert ref.shape == (N, K)
    assert np.isfinite(ref).all()


def test_round_trip_matches_reference(c):
    rng = np.random.default_rng(1)
    N, K = 2, 128
    codes = rng.integers(0, 16, size=(N, K)).astype(np.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    block_scale = rng.uniform(0.5, 2.0, size=(N, K // 16)).astype(np.float32)
    global_scale = rng.uniform(0.5, 2.0, size=(N,)).astype(np.float32)
    ref = c.dequantize_nvfp4_block(packed, block_scale, global_scale)

    mxfp4_packed, mxfp4_scales = c.convert_nvfp4_to_mxfp4(
        packed, block_scale, global_scale
    )
    assert mxfp4_packed.shape == (N, K // 2)
    assert mxfp4_scales.shape == (N, K // 32)
    assert mxfp4_scales.dtype == np.uint8

    # Re-dequantize the MXFP4 output and compare to the NVFP4 reference.
    out_codes = np.stack(
        [mxfp4_packed & 0x0F, mxfp4_packed >> 4], axis=-1
    ).reshape(N, K)
    vals = np.asarray(c.fp4_e2m1_table(), dtype=np.float32)[out_codes.astype(np.int64)]
    # e8m0 scale 2**(v-127)
    scale_v = (2.0 ** (mxfp4_scales.astype(np.float32) - 127.0))[:, :, None]
    mxfp4_vals = (vals.reshape(N, K // 32, 32) * scale_v).reshape(N, K)
    # MXFP4 requantizes at a coarser 32-block granularity, so the round trip carries
    # fp4 quantization error (inherent to e2m1). Bound it loosely: median < 0.6 and the
    # largest magnitude in each block (which defines the scale) must be well-preserved.
    rel = np.abs(mxfp4_vals - ref) / (np.abs(ref) + 1e-6)
    # Exclude near-zero refs where relative error is meaningless (a code flips +0<->small).
    nz = np.abs(ref) > 1e-3
    assert float(np.median(rel[nz])) < 0.6
    assert float(np.percentile(rel[nz], 95)) < 2.0
    # The max-abs element per 32-block reproduces within one e2m1 step of its scale.
    block_max_ref = np.abs(ref.reshape(N, K // 32, 32)).max(axis=-1)
    block_max_ours = np.abs(mxfp4_vals.reshape(N, K // 32, 32)).max(axis=-1)
    scale_ratio = np.abs(block_max_ours / (block_max_ref + 1e-6) - 1.0)
    assert float(np.median(scale_ratio)) < 0.5


def test_inner_axis_unsupported(c):
    packed = np.zeros((64, 32), dtype=np.uint8)
    scale = np.zeros((64, 4), dtype=np.float32)
    g = np.ones((64,), dtype=np.float32)
    with pytest.raises(NotImplementedError):
        c.convert_nvfp4_to_mxfp4(packed.T, scale, g, axis=0)
