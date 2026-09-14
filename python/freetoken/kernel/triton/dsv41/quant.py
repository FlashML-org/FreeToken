"""DeepSeek V4.1 block-32 FP8 linears and the two CSA2 FP4 scale formats."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.dsv4.fp8_linear import (
    _log2_ceil,
    _round_fp4,
    act_quant_fp8,
    act_quant_fp8_roundtrip,
)
from freetoken.kernel.triton.e4m3_compat import (
    e4m3_f32_to_u8,
    e4m3_kernel_view,
    e4m3_native_cx,
    e4m3_u8_to_f32,
    round_e4m3,
)
from freetoken.kernel.triton.kv_nvfp4 import _decode_e2m1, _encode_e2m1


def _check_block(x: torch.Tensor, block_size: int) -> None:
    if block_size not in (16, 32, 64, 128) or x.ndim < 1 or x.shape[-1] % block_size:
        raise ValueError(f"shape {tuple(x.shape)} is incompatible with block size {block_size}")


def fp8_roundtrip(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    _check_block(x, block_size)
    if x.numel() == 0:
        return x.clone()
    if x.is_cuda:
        return act_quant_fp8_roundtrip(x, block_size)
    groups = x.float().reshape(*x.shape[:-1], -1, block_size)
    scale = torch.exp2(torch.ceil(torch.log2(groups.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)))
    values = (groups / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (values * scale).reshape_as(x).to(x.dtype)


@triton.jit
def _fp4_roundtrip_kernel(X, Y, M: tl.constexpr, K: tl.constexpr,
                          GROUP: tl.constexpr, E4M3: tl.constexpr, ROWS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.program_id(1) * GROUP + tl.arange(0, GROUP)
    offsets = rows[:, None] * K + cols[None, :]
    x = tl.load(X + offsets, rows[:, None] < M, other=0).to(tl.float32)
    amax = tl.max(tl.abs(x), 1)
    if E4M3:
        scale = round_e4m3(tl.clamp(amax / 6.0, 2.0 ** -9, 448.0))
    else:
        exponent = _log2_ceil(tl.maximum(amax, 6.0 * (2.0 ** -126)) * (1.0 / 6.0))
        scale = tl.exp2(exponent.to(tl.float32))
    value = _round_fp4(tl.clamp(tl.div_rn(x, scale[:, None]), -6, 6)) * scale[:, None]
    tl.store(Y + offsets, value, rows[:, None] < M)


def fp4_roundtrip(x: torch.Tensor, block_size: int = 16, scale_format: str = "e4m3") -> torch.Tensor:
    _check_block(x, block_size)
    if scale_format not in ("e4m3", "e8m0"):
        raise ValueError(f"unsupported FP4 scale format: {scale_format}")
    if x.numel() == 0:
        return x.clone()
    if x.is_cuda:
        source = x.contiguous()
        out = torch.empty_like(source)
        k = x.shape[-1]
        m = x.numel() // k
        _fp4_roundtrip_kernel[(triton.cdiv(m, 32), k // block_size)](
            source, out, m, k, block_size, scale_format == "e4m3", 32,
            num_warps=4, enable_fp_fusion=False,
        )
        return out
    groups = x.float().reshape(*x.shape[:-1], -1, block_size)
    amax = groups.abs().amax(-1, keepdim=True)
    if scale_format == "e4m3":
        scale = (amax / 6).clamp(2.0**-9, 448).to(torch.float8_e4m3fn).float()
    else:
        scale = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(6 * 2.0**-126) / 6)))
    values = (groups / scale).clamp(-6, 6)
    magnitudes = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    distance = (values.abs().unsqueeze(-1) - magnitudes).abs()
    # Ties go to an even FP4 code, rather than always to the smaller magnitude.
    even = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=x.device)
    codes = even[distance[..., even].argmin(-1)]
    result = magnitudes[codes] * values.sign() * scale
    return result.reshape_as(x).to(x.dtype)


@triton.jit
def _e4m3_to_f32(code):
    value = e4m3_u8_to_f32(code)
    return tl.where((code.to(tl.int32) & 127) == 127, float("nan"), value)


@triton.jit
def _ue8m0_to_f32(code):
    bits = code.to(tl.uint32) << 23
    bits = tl.where(code == 0, 0x00400000, bits)
    bits = tl.where(code == 255, 0x7FC00000, bits).to(tl.uint32)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _pack_kernel(X, Y, M: tl.constexpr, K: tl.constexpr, WIDTH: tl.constexpr,
                 GROUP: tl.constexpr, FP4: tl.constexpr, E4M3: tl.constexpr,
                 ROWS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    group = tl.program_id(1)
    cols = group * GROUP + tl.arange(0, GROUP)
    x = tl.load(X + rows[:, None] * K + cols[None, :], rows[:, None] < M, other=0).to(tl.float32)
    amax = tl.max(tl.abs(x), 1)
    if FP4:
        if E4M3:
            scale = round_e4m3(tl.clamp(amax / 6.0, 2.0 ** -9, 448.0))
            scale_code = e4m3_f32_to_u8(scale)
        else:
            exponent = _log2_ceil(tl.maximum(amax, 6.0 * (2.0 ** -126)) * (1.0 / 6.0))
            scale = tl.exp2(exponent.to(tl.float32))
            scale_code = (exponent + 127).to(tl.uint8)
        normalized = tl.clamp(tl.div_rn(x, scale[:, None]), -6, 6)
        codes = _encode_e2m1(normalized).reshape(ROWS, GROUP // 2, 2)
        low, high = tl.split(codes)
        packed = low | (high << 4)
        out_cols = group * (GROUP // 2) + tl.arange(0, GROUP // 2)
        tl.store(Y + rows[:, None] * WIDTH + out_cols[None, :], packed, rows[:, None] < M)
        scale_offset = K // 2
    else:
        exponent = _log2_ceil(tl.maximum(amax, 1e-4) * (1.0 / 448.0))
        scale = tl.exp2(exponent.to(tl.float32))
        normalized = tl.clamp(x / scale[:, None], -448.0, 448.0)
        if e4m3_native_cx():
            codes = normalized.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        else:
            codes = e4m3_f32_to_u8(round_e4m3(normalized))
        tl.store(Y + rows[:, None] * WIDTH + cols[None, :], codes, rows[:, None] < M)
        scale_code = (exponent + 127).to(tl.uint8)
        scale_offset = K
    tl.store(Y + rows * WIDTH + scale_offset + group, scale_code, rows < M)


def _pack(x, block_size, *, fp4, scale_format):
    _check_block(x, block_size)
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("packed V4.1 KV input must be BF16, FP16, or FP32")
    if scale_format not in ("e4m3", "e8m0"):
        raise ValueError(f"unsupported FP4 scale format: {scale_format}")
    k = x.shape[-1]
    code_width = k // 2 if fp4 else k
    out = torch.empty((*x.shape[:-1], code_width + k // block_size), dtype=torch.uint8, device=x.device)
    if not x.numel():
        return out
    if x.is_cuda:
        source = x.contiguous()
        _pack_kernel[(triton.cdiv(x.numel() // k, 32), k // block_size)](
            source, out, x.numel() // k, k, out.shape[-1], block_size,
            fp4, scale_format == "e4m3", 32, num_warps=4, enable_fp_fusion=False,
        )
        return out
    groups = x.float().reshape(*x.shape[:-1], -1, block_size)
    amax = groups.abs().amax(-1, keepdim=True)
    if fp4 and scale_format == "e4m3":
        scales = (amax / 6).clamp(2.0**-9, 448).to(torch.float8_e4m3fn)
        scale = scales.float()
        scale_codes = scales.view(torch.uint8)
    else:
        maximum, floor = (6, 6 * 2.0**-126) if fp4 else (448, 1e-4)
        exponent = torch.ceil(torch.log2(amax.clamp_min(floor) / maximum))
        scale = torch.exp2(exponent)
        scale_codes = (exponent + 127).to(torch.uint8)
    if fp4:
        values = (groups / scale).clamp(-6, 6)
        grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=x.device)
        even = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=x.device)
        distance = (values.abs().unsqueeze(-1) - grid).abs()
        codes = even[distance[..., even].argmin(-1)].to(torch.uint8) | ((values < 0).to(torch.uint8) << 3)
        pairs = codes.reshape(*x.shape[:-1], k // 2, 2)
        out[..., :code_width] = pairs[..., 0] | (pairs[..., 1] << 4)
    else:
        codes = (groups / scale).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        out[..., :code_width] = codes.reshape_as(x)
    out[..., code_width:] = scale_codes.squeeze(-1)
    return out


def pack_fp8(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Keep E4M3 codes followed by one UE8M0 byte per block in each packed row."""
    return _pack(x, block_size, fp4=False, scale_format="e8m0")


def pack_fp4(x: torch.Tensor, block_size: int = 16, scale_format: str = "e4m3") -> torch.Tensor:
    """Keep low-first E2M1 nibbles followed by the original block-scale bytes."""
    return _pack(x, block_size, fp4=True, scale_format=scale_format)


@triton.jit
def _unpack_kernel(X, Y, M: tl.constexpr, K: tl.constexpr, WIDTH: tl.constexpr,
                   GROUP: tl.constexpr, FP4: tl.constexpr, E4M3: tl.constexpr,
                   ROWS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    group = tl.program_id(1)
    cols = group * GROUP + tl.arange(0, GROUP)
    if FP4:
        packed = tl.load(X + rows[:, None] * WIDTH + cols[None, :] // 2,
                         rows[:, None] < M, other=0).to(tl.int32)
        value = _decode_e2m1(tl.where((cols[None, :] & 1) == 0, packed & 15, packed >> 4))
        scale_offset = K // 2
    else:
        code = tl.load(X + rows[:, None] * WIDTH + cols[None, :], rows[:, None] < M, other=0)
        value = _e4m3_to_f32(code)
        scale_offset = K
    scale_code = tl.load(X + rows * WIDTH + scale_offset + group, rows < M, other=0)
    if E4M3:
        scale = _e4m3_to_f32(scale_code)
    else:
        scale = _ue8m0_to_f32(scale_code)
    tl.store(Y + rows[:, None] * K + cols[None, :], value * scale[:, None], rows[:, None] < M)


def _unpack(packed, block_size, dtype, *, fp4, scale_format):
    if packed.ndim < 1 or packed.dtype != torch.uint8 or block_size not in (16, 32, 64, 128):
        raise ValueError("packed V4.1 KV requires uint8 rows and a supported block size")
    unit = (block_size // 2 if fp4 else block_size) + 1
    if packed.shape[-1] % unit or scale_format not in ("e4m3", "e8m0"):
        raise ValueError("packed V4.1 KV row width or scale format is invalid")
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("unpacked V4.1 KV dtype must be BF16, FP16, or FP32")
    k = packed.shape[-1] // unit * block_size
    out = torch.empty((*packed.shape[:-1], k), device=packed.device, dtype=dtype)
    if not out.numel():
        return out
    if packed.is_cuda:
        source = packed.contiguous()
        _unpack_kernel[(triton.cdiv(out.numel() // k, 32), k // block_size)](
            source, out, out.numel() // k, k, packed.shape[-1], block_size,
            fp4, scale_format == "e4m3", 32, num_warps=4, enable_fp_fusion=False,
        )
        return out
    code_width = k // 2 if fp4 else k
    raw = packed[..., :code_width].contiguous()
    if fp4:
        codes = torch.stack((raw & 15, raw >> 4), -1).flatten(-2).long()
        grid = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                            dtype=torch.float32, device=packed.device)
        values = grid[codes]
    else:
        values = raw.view(torch.float8_e4m3fn).float()
    scale_dtype = torch.float8_e4m3fn if scale_format == "e4m3" else torch.float8_e8m0fnu
    scale = packed[..., code_width:].contiguous().view(scale_dtype).float()
    out.copy_((values.reshape(*packed.shape[:-1], -1, block_size) * scale[..., None]).flatten(-2))
    return out


def unpack_fp8(packed: torch.Tensor, block_size: int = 32, dtype=torch.bfloat16) -> torch.Tensor:
    return _unpack(packed, block_size, dtype, fp4=False, scale_format="e8m0")


def unpack_fp4(packed: torch.Tensor, block_size: int = 16, scale_format: str = "e4m3",
               dtype=torch.bfloat16) -> torch.Tensor:
    return _unpack(packed, block_size, dtype, fp4=True, scale_format=scale_format)


@triton.jit
def _gemm(A, W, SA, SW, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          GROUP: tl.constexpr, BM: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * GROUP + tl.arange(0, GROUP)
    ki = tl.arange(0, GROUP)
    acc = tl.zeros((BM, GROUP), tl.float32)
    for kb in range(K // GROUP):
        a = tl.load(A + rows[:, None] * K + kb * GROUP + ki[None, :], rows[:, None] < M, other=0.0)
        w = tl.load(W + cols[:, None] * K + kb * GROUP + ki[None, :], cols[:, None] < N, other=0.0)
        if e4m3_native_cx():
            product = tl.dot(a, tl.trans(w), out_dtype=tl.float32)
        else:
            product = tl.dot(a, tl.trans(e4m3_u8_to_f32(w).to(tl.bfloat16)), out_dtype=tl.float32)
        sa = tl.load(SA + rows * (K // GROUP) + kb, rows < M, other=127)
        sw = tl.load(SW + tl.program_id(1) * (K // GROUP) + kb)
        scale = tl.exp2(sa.to(tl.float32) - 127) * tl.exp2(sw.to(tl.float32) - 127)
        acc += product * scale[:, None]
    tl.store(Y + rows[:, None] * N + cols[None, :], acc, (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _gemv(A, W, SA, SW, PART, N: tl.constexpr, K: tl.constexpr,
          GROUP: tl.constexpr, BN: tl.constexpr, SPLITS: tl.constexpr):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    ki = tl.arange(0, GROUP)
    split = tl.program_id(1)
    acc = tl.zeros((BN,), tl.float32)
    for kb in range(split, K // GROUP, SPLITS):
        a = tl.load(A + kb * GROUP + ki).to(tl.float32)
        raw_w = tl.load(W + rows[:, None] * K + kb * GROUP + ki[None, :], rows[:, None] < N, other=0.0)
        if e4m3_native_cx():
            w = raw_w.to(tl.float32)
        else:
            w = e4m3_u8_to_f32(raw_w)
        sw = tl.load(SW + (rows // GROUP) * (K // GROUP) + kb, rows < N, other=127)
        sa = tl.load(SA + kb)
        acc += tl.sum(w * a[None, :], 1) * tl.exp2(sw.to(tl.float32) - 127) * tl.exp2(sa.to(tl.float32) - 127)
    tl.store(PART + split * N + rows, acc, rows < N)


def block_fp8_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
                     bias: torch.Tensor | None = None, block_size: int = 32) -> torch.Tensor:
    _check_block(x, block_size)
    if weight.ndim != 2 or weight.shape[1] != x.shape[-1] or weight.shape[0] % block_size:
        raise ValueError("FP8 linear weight dimensions do not match the input/block size")
    n, k = weight.shape
    if weight.dtype != torch.float8_e4m3fn or tuple(scale.shape) != (n // block_size, k // block_size):
        raise ValueError("expected E4M3 weights and a per-block E8M0 scale matrix")
    if scale.dtype not in (torch.float8_e8m0fnu, torch.uint8):
        raise ValueError("FP8 linear scales must be E8M0 values or unsigned E8M0 codes")
    if x.device != weight.device or x.device != scale.device:
        raise ValueError("FP8 linear operands must be on the same device")
    shape = (*x.shape[:-1], n)
    if x.numel() == 0:
        return x.new_empty(shape)
    if x.is_cuda:
        activation, act_scale = act_quant_fp8(x, block_size)
        w = e4m3_kernel_view(weight.contiguous())
        sw = scale.view(torch.uint8).contiguous()
        m = activation.shape[0]
        if m == 1:
            splits = min(16, k // block_size)
            partial = torch.empty((splits, n), device=x.device, dtype=torch.float32)
            _gemv[(triton.cdiv(n, 16), splits)](
                activation, w, act_scale, sw, partial, n, k, block_size, 16, splits,
                num_warps=4, enable_fp_fusion=False,
            )
            out = partial.sum(0).to(x.dtype).reshape(shape)
        else:
            out = x.new_empty((m, n))
            _gemm[(triton.cdiv(m, 32), n // block_size)](
                activation, w, act_scale, sw, out, m, n, k, block_size, 32,
                num_warps=4, enable_fp_fusion=False,
            )
            out = out.reshape(shape)
    else:
        activation = fp8_roundtrip(x, block_size).float().reshape(-1, k)
        codes = scale.view(torch.uint8).float()
        factors = torch.exp2(codes - 127).repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
        out = (activation @ (weight.float() * factors).T).to(x.dtype).reshape(shape)
    return out if bias is None else out + bias.to(out.dtype)
