"""e4m3 (fp8e4nv) emulation for pre-sm_89 CUDA GPUs.

Triton rejects the fp8e4nv type anywhere in a kernel compiled for sm < 89 -- the
check sits in ``dtype.to_ir``, so even an fp8 *pointer argument* is illegal.
Affected kernels branch on :func:`e4m3_native_cx` (a compile-time constexpr): the
native branch stays byte-identical on sm_89+, the emulated branch is dead-code
eliminated there. When the emulated branch is active, wrappers must pass e4m3
tensors as ``.view(torch.uint8)`` and allocate act-quant outputs as bf16.

TWO independent probes answer "is fp8e4nv native here": :func:`e4m3_native` (host,
torch's device capability -- decides what a buffer is ALLOCATED as) and
:func:`e4m3_native_cx` (compile-time, triton's target -- decides which arm a kernel
compiles). They cannot be merged, because a constexpr function that referenced the
host one would not survive triton's cache-key AST walk, so on a box where the probes
disagree the host holds real fp8 tensors while kernels take the emulated arm.
:func:`warn_if_probes_disagree` says so once at startup. Code handed a buffer whose
dtype the host already settled -- the KV pool -- must not re-ask at all and picks its
decode from the pointer: :func:`kv_load_e4m3_tile_f32`. Trusting the probe over the
tensor is what produced "cannot cast int32 to fp8e4nv" at CUDA graph capture.

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
    """Host-side probe: does THIS device take fp8e4nv tensors directly? True: kernels
    get fp8 tensors, False: pass ``.view(torch.uint8)`` and bf16 act buffers. NOT
    necessarily the answer :func:`e4m3_native_cx` gives -- see this module's header."""
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
        warn_if_probes_disagree()
    return _native


_warned_disagree = False


def warn_if_probes_disagree() -> None:
    """Log once when the two native-fp8e4nv probes answer differently.

    :func:`e4m3_native` decides what buffers the host ALLOCATES while
    :func:`e4m3_native_cx` decides which arm a kernel compiles, and the two cannot be
    unified (that function's docstring explains triton's cache-key walk). A box where
    triton's probe under-reports therefore runs every e4m3 kernel through the software
    decode -- bit-exact per this module's header, but slower -- and any kernel that
    trusts the probe over the tensor it was handed stops compiling outright. Reads the
    latched ``_native`` rather than calling back into :func:`e4m3_native`."""
    global _warned_disagree
    if _warned_disagree:
        return
    _warned_disagree = True
    if FORCE_EMU:  # emulating by request is not a disagreement
        return
    try:
        triton_native = target_info.cuda_capability_geq(8, 9)
        if triton_native == bool(_native):
            return
        major, minor = torch.cuda.get_device_capability()
    except Exception:  # noqa: BLE001 -- no driver/no target yet: nothing to compare
        return
    from freetoken.utils import init_logger

    init_logger(__name__).warning(
        "native fp8e4nv disagreement: torch reports sm_%d%d for this device but "
        "triton's target probe says %s, so e4m3 kernels compile the software-decode "
        "branch (bit-exact, slower). The fp8 KV cache is unaffected -- it follows the "
        "buffer it was given.",
        major, minor, "supported" if triton_native else "unsupported",
    )


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
    resolve consistently).

    It CANNOT defer to :func:`e4m3_native`, however much one verdict per process is
    what we want: triton hashes a constexpr function by walking its AST
    (runtime/jit.py: cache_key -> record_reference), and a bare reference to a plain
    python function raises "Unsupported function referenced: <function e4m3_native>"
    -- trying that once disabled every e4m3 kernel at once, PLE gather included.
    Module attributes (``target_info.whatever``) survive the walk, plain functions do
    not. The probes therefore stay separate, :func:`warn_if_probes_disagree` reports
    when they disagree, and code handed a buffer the host already typed -- the KV
    pool -- ignores this function and follows the pointer: kv_load_e4m3_tile_f32."""
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


@jit
def e4m3_f32_to_u8(x):
    """Encode an fp32 value that ALREADY lies on the e4m3 grid -- the output of
    :func:`round_e4m3`, clamped to +-448 -- into its e4m3 byte code. This is the
    encoder the pre-sm_89 emulated path needs to STORE fp8-sized data (the fp8
    type itself is unavailable there, so the bytes live in a uint8 buffer that
    :func:`e4m3_u8_to_f32` decodes back).

    Normal range: read the (unbiased) exponent and the now-zero-padded fp32
    mantissa back out of the fp32 header. Subnormal range (|x| < 2^-6, grid step
    2^-9): the value is an exact multiple of 2^-9, so ``|x| * 512`` IS the mantissa
    field -- the sign bit has to be carried in by hand, since that branch never
    looks at the header. The same 0.015625 boundary as :func:`round_e4m3` keeps the
    two consistent: ``e4m3_u8_to_f32(e4m3_f32_to_u8(round_e4m3(x)))`` is x's
    single-rounded value for every input, and no code it emits is a NaN pattern (the
    caller's +-448 clamp caps the code at 0x7E). ``-0.0`` encodes as 0x00 after
    round_e4m3 (which documents returning +0.0 for it).
    """
    u = x.to(tl.uint32, bitcast=True)
    sign = ((u >> 31) & 1).to(tl.int32)
    exp = ((u >> 23) & 0xFF).to(tl.int32) - 127
    mant = ((u >> 20) & 7).to(tl.int32)
    normal = (sign << 7) | ((exp + 7) << 3) | mant
    sub = (sign << 7) | (tl.abs(x) * 512.0).to(tl.int32)
    return tl.where(tl.abs(x) >= 0.015625, normal, sub).to(tl.uint8)


@jit
def kv_load_e4m3_tile_f32(ptrs, mask):
    """Load a tile of KV e4m3 codes and widen it to fp32.

    Straight-line on purpose: no probe, no dtype test, so there is no arm left to
    prune. The pools keep their codes in a plain byte buffer on EVERY architecture
    (kv_quant.kv_codes_dtype), so the fp8e4nv type never reaches Triton through here.
    Both ways of choosing an arm were tried on real hardware and each broke the run:
    the compile-time fp8-native answer is a second, independent verdict that can
    disagree with the host that allocated the buffer, and a comparison against the
    pointer's element type is NOT statically pruned -- Triton type-checks the arm that
    should have been dead, and an int mask fill against an fp8 pointer is rejected
    ("cannot cast int32 to fp8e4nv", raised at CUDA graph capture on sm_100).

    What remains is the decode that already runs wherever the fp8 type is unavailable,
    bit-exact per this module's header: the same load-with-int-fill and software
    widening as kernel/triton/ple.py and nvfp4_linear.py. Callers use this only in
    their quantized branch -- the 16-bit path keeps its own tl.load, so bf16 attention
    is untouched instruction for instruction -- and the dense paged kernels and the
    QSA sparse one read the same pool, hence this helper lives here.
    """
    return e4m3_u8_to_f32(tl.load(ptrs, mask=mask, other=0))
