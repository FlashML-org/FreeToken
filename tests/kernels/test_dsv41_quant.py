import pytest
import torch

from freetoken.kernel.triton.dsv41.quant import (
    block_fp8_linear, fp4_roundtrip, fp8_roundtrip,
    pack_fp4, pack_fp8, unpack_fp4, unpack_fp8,
)


def _fp4_reference(x, block, fmt):
    groups = x.float().reshape(-1, block)
    amax = groups.abs().amax(-1, keepdim=True)
    if fmt == "e4m3":
        scales = (amax / 6).clamp(1 / 512, 448).to(torch.float8_e4m3fn).float()
    else:
        scales = 2.0 ** torch.ceil(torch.log2(amax.clamp_min(6 * 2.0**-126) / 6))
    y = (groups / scales).clamp(-6, 6)
    thresholds = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
    codes = torch.bucketize(y.abs().contiguous(), thresholds)
    for index in (1, 3, 5):
        codes[y.abs() == thresholds[index]] = index + 1
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
    return (grid[codes] * y.sign() * scales).reshape_as(x).to(x.dtype)


@pytest.mark.parametrize("fmt,block", [("e4m3", 16), ("e8m0", 32)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fp4_roundtrip_formats(fmt, block, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(42)
    x = torch.randn(35, 128, dtype=torch.bfloat16)
    x[0].zero_()
    x[1] *= 0.001
    x[2] *= 10
    x[3, :16] = torch.tensor([0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, -6])
    expected = _fp4_reference(x, block, fmt)
    actual = fp4_roundtrip(x.to(device), block, fmt).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fp8_roundtrip_block32(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(19)
    x = torch.randn(3, 128, dtype=torch.bfloat16)
    x[:, 32:64] *= 128
    x[:, 96:].zero_()
    groups = x.float().reshape(3, 4, 32)
    scale = 2.0 ** torch.ceil(torch.log2(groups.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448))
    expected = ((groups / scale).to(torch.float8_e4m3fn).float() * scale).reshape_as(x).to(x.dtype)
    torch.testing.assert_close(fp8_roundtrip(x.to(device)).cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fp4_e4m3_scale_saturates_to_representable_range(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.full((2, 16), 6144, dtype=torch.bfloat16, device=device)
    x[1].neg_()
    expected = torch.full_like(x, 2688)
    expected[1].neg_()
    torch.testing.assert_close(fp4_roundtrip(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 2, 35])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_block32_linear_independent_scales(rows, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(31)
    x = torch.randn(rows, 160, dtype=torch.bfloat16)
    w = torch.randn(96, 160).to(torch.float8_e4m3fn)
    scale = torch.tensor([[125, 126, 127, 128, 129], [129, 125, 126, 128, 127], [127, 129, 125, 126, 128]], dtype=torch.uint8)
    a = x.float().reshape(rows, 5, 32)
    sa = 2.0 ** torch.ceil(torch.log2(a.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448))
    aq = (a / sa).to(torch.float8_e4m3fn).float()
    expected = torch.zeros(rows, 96)
    for kb in range(5):
        block_w = w.float()[:, kb*32:(kb+1)*32]
        sb = (2.0 ** (scale[:, kb].float() - 127)).repeat_interleave(32)
        expected += (aq[:, kb] @ block_w.T) * sa[:, kb] * sb
    actual = block_fp8_linear(x.to(device), w.to(device), scale.to(device)).cpu()
    torch.testing.assert_close(actual.float(), expected.to(torch.bfloat16).float(), rtol=0.008, atol=0.02)


def test_invalid_scale_geometry():
    with pytest.raises(ValueError, match="scale matrix"):
        block_fp8_linear(torch.zeros(1, 64), torch.zeros(64, 64).to(torch.float8_e4m3fn), torch.zeros(1, 1, dtype=torch.uint8))


@pytest.mark.parametrize("fmt,block,width", [("fp8", 32, 512), ("e4m3", 16, 512), ("e8m0", 32, 128)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_packed_native_rows_reconstruct_existing_roundtrip(fmt, block, width, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(122)
    x = torch.randn(2, 35, width * 2, dtype=torch.bfloat16)[..., ::2]
    x[:, 0].zero_()
    x[:, 1] *= .001
    x[:, 2] *= 10000
    x[:, 3] *= 2.0**-127
    x[:, 4, :16] = torch.tensor([0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, -6])
    source = x.to(device)
    if fmt == "fp8":
        packed = pack_fp8(source, block)
        actual = unpack_fp8(packed, block)
        expected = fp8_roundtrip(source, block)
        groups = x.float().reshape(2, 35, -1, block)
        scales = 2.0 ** torch.ceil(torch.log2(groups.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448))
        reference = ((groups / scales).to(torch.float8_e4m3fn).float() * scales).reshape_as(x).bfloat16()
        cpu_packed = pack_fp8(x, block)
        row_bytes = width + width // block
    else:
        packed = pack_fp4(source, block, fmt)
        actual = unpack_fp4(packed, block, fmt)
        expected = fp4_roundtrip(source, block, fmt)
        reference = _fp4_reference(x, block, fmt)
        cpu_packed = pack_fp4(x, block, fmt)
        row_bytes = width // 2 + width // block
    assert packed.shape == (2, 35, row_bytes) and packed.dtype == torch.uint8
    assert packed.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
    torch.testing.assert_close(packed.cpu(), cpu_packed, rtol=0, atol=0)


@pytest.mark.parametrize("fmt,block,scale_byte", [("e4m3", 16, 56), ("e8m0", 32, 127)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fp4_pack_nibble_order_and_inline_scale_bytes(fmt, block, scale_byte, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    row = [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6]
    x = torch.tensor(row * (block // 16), dtype=torch.bfloat16, device=device)
    expected = torch.tensor([0x10, 0x32, 0x54, 0x76, 0x90, 0xBA, 0xDC, 0xFE] * (block // 16)
                            + [scale_byte], dtype=torch.uint8, device=device)
    torch.testing.assert_close(pack_fp4(x, block, fmt), expected, rtol=0, atol=0)
    torch.testing.assert_close(unpack_fp4(expected, block, fmt), x, rtol=0, atol=0)


@pytest.mark.parametrize("fmt,block", [("fp8", 32), ("e4m3", 16), ("e8m0", 32)])
@pytest.mark.parametrize("shape", [(0, 128), (2, 0, 128), (128,)])
def test_packed_codecs_keep_empty_and_leading_shapes(fmt, block, shape):
    x = torch.zeros(shape, dtype=torch.bfloat16)
    if fmt == "fp8":
        packed = pack_fp8(x, block)
        restored = unpack_fp8(packed, block)
    else:
        packed = pack_fp4(x, block, fmt)
        restored = unpack_fp4(packed, block, fmt)
    assert restored.shape == x.shape
    torch.testing.assert_close(restored, x, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_packed_index_decode_handles_all_e8m0_scale_codes(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    packed = torch.full((256, 17), 0x22, dtype=torch.uint8)
    packed[:, -1] = torch.arange(256, dtype=torch.uint8)
    expected = packed[:, -1:].contiguous().view(torch.float8_e8m0fnu).float().expand(-1, 32).bfloat16()
    actual = unpack_fp4(packed.to(device), 32, "e8m0").cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("tier", ["window", "compressed"])
def test_packed_decode_handles_all_e4m3_codes_and_scales(device, tier):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    codes = torch.arange(256, dtype=torch.uint8)
    if tier == "window":
        packed = torch.cat((codes[:, None].expand(-1, 32), torch.full((256, 1), 127, dtype=torch.uint8)), 1)
        actual = unpack_fp8(packed.to(device)).cpu()
        width = 32
    else:
        packed = torch.cat((torch.full((256, 8), 0x22, dtype=torch.uint8), codes[:, None]), 1)
        actual = unpack_fp4(packed.to(device), 16, "e4m3").cpu()
        width = 16
    expected = codes.view(torch.float8_e4m3fn).float()[:, None].expand(-1, width).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.isnan(actual[[127, 255]]).all()


@pytest.mark.parametrize("fn,args", [(pack_fp8, (torch.zeros(2, 33),)),
                                    (pack_fp4, (torch.zeros(2, 17), 16, "e4m3")),
                                    (unpack_fp8, (torch.zeros(2, 34, dtype=torch.uint8),)),
                                    (unpack_fp4, (torch.zeros(2, 18), 16, "e4m3")),
                                    (unpack_fp4, (torch.zeros(2, 18, dtype=torch.uint8), 16, "unknown"))])
def test_packed_codecs_reject_bad_layout(fn, args):
    with pytest.raises(ValueError):
        fn(*args)
