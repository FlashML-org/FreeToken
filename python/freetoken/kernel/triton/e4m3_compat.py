"""e4m3 (fp8e4nv) emulation for pre-sm_89 CUDA GPUs.

Triton rejects the fp8e4nv type anywhere in a kernel compiled for sm < 89 -- the
check sits in ``dtype.to_ir``, so even an fp8 *pointer argument* is illegal.
Affected kernels branch on :func:`e4m3_native_cx` (a compile-time constexpr): the
native branch stays byte-identical on sm_89+, the emulated branch is dead-code
eliminated there. When the emulated branch is active, wrappers must pass e4m3
tensors as ``.view(torch.uint8)`` and allocate act-quant outputs as bf16 -- use
the host-side twin :func:`e4m3_native` for those decisions.

``FREETOKEN_FORCE_E4M3_EMU=1`` (or true/yes/on) forces the emulated path on any
GPU (for A/B validation against the native fp8 unit). The flag is read ONCE at
import and is deliberately NOT part of triton's compilation cache key, so:
flipping it later in the same process raises (see :func:`e4m3_native`), and when
it is set without an explicit ``TRITON_CACHE_DIR`` the default cache dir is
salted so a warm native cache can never serve the other branch's binary for the
signature-invariant kernels (``_act_quant_inplace_kernel``).

The primitives are validated bit-exact against the native fp8 unit on H100 (all
254 non-NaN codes; 2.1M adversarial rounding samples covering every grid
midpoint, the subnormal range and the 2^-6 boundary). Known deviations: the NaN
codes 0x7F/0xFF decode to +-480 (checkpoints never store NaN weights), and
``round_e4m3(-0.0)`` returns +0.0.
"""

from __future__ import annotations

import os

import torch
import triton.language as tl
from triton import jit
from triton.language import target_info
from triton.runtime.jit import constexpr_function

def _env_force() -> bool:
    return os.environ.get("FREETOKEN_FORCE_E4M3_EMU", "").lower() in ("1", "true", "yes", "on")


FORCE_EMU = _env_force()

if FORCE_EMU and "TRITON_CACHE_DIR" not in os.environ:
    os.environ["TRITON_CACHE_DIR"] = os.path.join(
        os.path.expanduser("~/.triton"), "cache-e4m3emu")

_native: bool | None = None


def e4m3_native() -> bool:
    """Host-side twin of :func:`e4m3_native_cx`: True when kernels take fp8e4nv
    tensors directly. False: pass ``.view(torch.uint8)`` and bf16 act buffers."""
    global _native
    if _env_force() != FORCE_EMU:
        raise RuntimeError(
            "FREETOKEN_FORCE_E4M3_EMU changed after import: the flag is read once at "
            "import and is not part of triton's compile cache key -- set it before "
            "the process starts (with its own TRITON_CACHE_DIR)"
        )
    if _native is None:
        if FORCE_EMU:
            _native = False
        else:
            from freetoken.gpu_select import assigned_visible_gpu

            # one process runs on one GPU, so its convention is that GPU's; None (-> the current device) only before the process binds
            _native = torch.cuda.get_device_capability(assigned_visible_gpu()) >= (8, 9)
    return _native


def e4m3_kernel_view(t: torch.Tensor) -> torch.Tensor:
    """An e4m3 tensor as the branched kernels expect it: unchanged when native,
    the uint8 view otherwise (the fp8 pointer type is illegal pre-sm_89)."""
    return t if e4m3_native() else t.view(torch.uint8)


def e4m3_act_dtype() -> torch.dtype:
    """Buffer dtype for quantized activations: fp8 when native, else bf16 (every
    e4m3 grid value is exactly representable)."""
    return torch.float8_e4m3fn if e4m3_native() else torch.bfloat16


@constexpr_function
def e4m3_native_cx():
    """Compile-time: does the compilation target have native fp8e4nv (sm_89+)?
    Delegates to ``target_info`` (reads the active driver's target, so
    cross-compilation tests that patch ``driver.active.get_current_target``
    resolve consistently)."""
    return not FORCE_EMU and target_info.cuda_capability_geq(8, 9)


@jit
def e4m3_u8_to_f32(v):
    """Decode e4m3 bits (uint8) to fp32: place exp+mantissa in the fp16 field
    (exact, incl. e4m3 subnormals) and rescale by 2^(15-7). NaN codes -> +-480."""
    h = ((v & 0x80).to(tl.uint16) << 8) | ((v & 0x7F).to(tl.uint16) << 7)
    return h.to(tl.float16, bitcast=True).to(tl.float32) * 256.0


@jit
def e4m3_u8_to_f16_x128(v):
    """Decode e4m3 bits (uint8) to fp16 pre-scaled by 128 (the nvfp4 GEMM form:
    native is ``.to(tl.float16) * 128``). (val/256) * 2^15 stays in fp16 range
    (max 448*128 = 57344) and is exact (power-of-two scaling)."""
    h = ((v & 0x80).to(tl.uint16) << 8) | ((v & 0x7F).to(tl.uint16) << 7)
    return h.to(tl.float16, bitcast=True) * 32768.0


@jit
def round_e4m3(x):
    """Round fp32 onto the e4m3 value grid (RNE), fp32 -> fp32, in a SINGLE
    rounding step -- an fp32 -> fp16 -> 3-bit chain double-rounds when the fp16
    result lands exactly on an e4m3 tie. Caller clamps to +-448 first.

    Normal range: RNE-truncate the fp32 mantissa 23 -> 3 bits with the integer
    round-half-to-even trick (carry into the exponent rounds up correctly; it
    cannot reach the sign bit for |x| <= 448). Subnormal range (|x| < 2^-6, grid
    fixed at 2^-9): quantize via the add-magic trick -- at magnitude 2^14 the
    fp32 ulp is exactly 2^-9, so the add rounds RNE onto the grid and the
    subtract is exact."""
    b = x.to(tl.uint32, bitcast=True)
    lsb = (b >> 20) & 1
    y_norm = ((b + 524287 + lsb) & 0xFFF00000).to(tl.float32, bitcast=True)
    y_sub = (x + 24576.0) - 24576.0
    return tl.where(tl.abs(x) >= 0.015625, y_norm, y_sub)
