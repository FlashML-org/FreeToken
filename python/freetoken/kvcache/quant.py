"""KV-cache quantization schemes: 8-bit storage with a per-block scale.

The KV pool normally stores K/V in the model's compute dtype (bf16). A quantized pool
stores them in an 8-bit dtype plus a parallel scale tensor holding one fp16 scale per
:data:`BLOCK` elements along ``head_dim`` -- the same geometry GGUF's Q8_0 uses, and the
reason the block is small: KV outliers (mostly in the keys) concentrate in a few
channels, and a block of 32 keeps an outlier from stretching the scale of the whole
head.

Both schemes share this layout, the store kernel and the dequant path in the attention
kernels; they differ only in the storage dtype and the divisor that maps a block's
max-abs onto the dtype's range. That is deliberate -- picking between them is a flag,
not a second port.

The scale varies along ``head_dim``, which is the reduction dimension of ``q @ k``, so
the attention kernels cannot fold it in after the dot: they dequantize to bf16 before
the dot. Storage bandwidth is what this buys, not tensor-core throughput.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Elements per scale, along head_dim. Matches GGUF Q8_0's block.
BLOCK = 32
# One fp16 scale per block.
SCALE_DTYPE = torch.float16


@dataclass(frozen=True)
class KVQuantSpec:
    """How a KV pool stores its K/V elements.

    ``name`` is the ``--kv-cache-dtype`` value. ``storage_dtype`` is None for the
    unquantized pool, in which case the pool allocates in the compute dtype and no
    scale tensor exists.
    """

    name: str
    storage_dtype: torch.dtype | None
    # Max-abs of a block maps to this magnitude in the storage dtype.
    max_magnitude: float

    @property
    def enabled(self) -> bool:
        return self.storage_dtype is not None

    @property
    def is_integer(self) -> bool:
        """Integer schemes round; float ones just divide."""
        return self.storage_dtype == torch.int8

    def bytes_per_element(self, compute_dtype: torch.dtype) -> float:
        """Storage bytes per K/V element, scales amortized over the block.

        Unquantized: the compute dtype's itemsize. Quantized: 1 byte + 2/32 for the
        fp16 scale = 1.0625 -- 6% over a bare 8 bits, versus 16 bits stored.
        """
        if not self.enabled:
            return float(compute_dtype.itemsize)
        return 1.0 + SCALE_DTYPE.itemsize / BLOCK

    def scale_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Scale-tensor shape for a KV buffer shape: last dim divided by the block."""
        if shape[-1] % BLOCK:
            raise ValueError(
                f"head_dim {shape[-1]} is not a multiple of the KV quant block {BLOCK}"
            )
        return (*shape[:-1], shape[-1] // BLOCK)

    # ---- reference implementations (correctness oracle for the Triton kernels) ----

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x[..., D]`` (float) -> ``(quantized[..., D], scales[..., D // BLOCK])``."""
        assert self.enabled, "quantize() on an unquantized spec"
        blocks = x.float().unflatten(-1, (x.shape[-1] // BLOCK, BLOCK))
        amax = blocks.abs().amax(dim=-1)
        # An all-zero block would divide by zero; its quantized values are zero either
        # way, so any positive scale works.
        scales = torch.where(amax > 0, amax / self.max_magnitude, torch.ones_like(amax))
        # Round the scale to its stored precision BEFORE dividing, so quantize and
        # dequantize use the identical value. Dividing by the fp32 scale and storing the
        # fp16 one leaves a residual error the round-trip cannot cancel.
        scales = scales.to(SCALE_DTYPE)
        q = blocks / scales.float().unsqueeze(-1)
        if self.is_integer:
            # Half away from zero, matching GGUF's Q8_0 and the store kernel.
            # ``Tensor.round`` is half-to-even and would disagree on ties.
            q = torch.where(q >= 0, (q + 0.5).floor(), (q - 0.5).ceil())
            q = q.clamp_(-self.max_magnitude, self.max_magnitude)
        return (q.flatten(-2).to(self.storage_dtype), scales)

    def dequantize(self, q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`quantize`, in float32."""
        assert self.enabled, "dequantize() on an unquantized spec"
        blocks = q.float().unflatten(-1, (q.shape[-1] // BLOCK, BLOCK))
        return (blocks * scales.float().unsqueeze(-1)).flatten(-2)


# int8 symmetric: a block's max-abs maps to 127.
Q8_0 = KVQuantSpec(name="q8_0", storage_dtype=torch.int8, max_magnitude=127.0)
# e4m3: 4-bit exponent, 3-bit mantissa, max finite magnitude 448.
FP8_E4M3 = KVQuantSpec(name="fp8_e4m3", storage_dtype=torch.float8_e4m3fn, max_magnitude=448.0)
NONE = KVQuantSpec(name="auto", storage_dtype=None, max_magnitude=0.0)

_BY_NAME = {spec.name: spec for spec in (NONE, Q8_0, FP8_E4M3)}
KV_CACHE_DTYPES = tuple(_BY_NAME)


def resolve_kv_quant(name: str | None) -> KVQuantSpec:
    """``--kv-cache-dtype`` value -> spec. ``None``/``"auto"`` means unquantized."""
    if name is None:
        return NONE
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"unknown --kv-cache-dtype {name!r}; choose from {', '.join(KV_CACHE_DTYPES)}"
        ) from None


__all__ = [
    "BLOCK",
    "SCALE_DTYPE",
    "KVQuantSpec",
    "KV_CACHE_DTYPES",
    "Q8_0",
    "FP8_E4M3",
    "NONE",
    "resolve_kv_quant",
]
