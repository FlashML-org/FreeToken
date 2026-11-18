"""Bank and format names on disk (FTW), in the CPU executor and in the GGUF provider; the kernels use the canonical roles."""

from __future__ import annotations

from freetoken.layers.quantization import QuantKind

CANONICAL_ROLE = {
    "gate_up_packed": "gate_up",
    "gate_up_blocks": "gate_up",
    "gate_up_scales": "gate_up_scale",
    "down_packed": "down",
    "down_blocks": "down",
    "down_scales": "down_scale",
}


def canonical_role(bank_name: str) -> str:
    return CANONICAL_ROLE.get(bank_name, bank_name)


def legacy_bank_names(quant_format: str) -> dict[str, str]:
    """Canonical role -> the bank name FTW files store for ``quant_format``."""
    from freetoken.moe.offload_cache import _BANK_SCHEMAS

    return {canonical_role(name): name for name in _BANK_SCHEMAS[quant_format]}


# (kind, kernel) <-> the quant_format tag
LEGACY_FORMAT = {
    (QuantKind.NONE, "fused"): "bf16",
    (QuantKind.FP8_BLOCK, "triton"): "fp8_block",
    (QuantKind.NVFP4, "triton"): "nvfp4",
    # same banks as triton, plus resident gate_up_alpha / down_alpha entries
    (QuantKind.NVFP4, "w4a4"): "nvfp4",
    (QuantKind.NVFP4, "marlin"): "nvfp4_marlin",
    (QuantKind.NVFP4, "b12x"): "nvfp4_b12x",
    (QuantKind.MXFP4, "triton_gptoss"): "mxfp4_triton",
    (QuantKind.MXFP4, "triton"): "ds_fp4",
}
_KIND_KERNEL: dict[str, tuple[QuantKind, str]] = {}
_KERNELS_FOR_FORMAT: dict[str, tuple[str, ...]] = {}
for _kk, _fmt in LEGACY_FORMAT.items():
    _KIND_KERNEL.setdefault(_fmt, _kk)
    _KERNELS_FOR_FORMAT[_fmt] = _KERNELS_FOR_FORMAT.get(_fmt, ()) + (_kk[1],)


def legacy_format_for(kind: QuantKind, kernel: str) -> str:
    return LEGACY_FORMAT[(kind, kernel)]


def kind_kernel_for(legacy_format: str) -> tuple[QuantKind, str]:
    return _KIND_KERNEL[legacy_format]


def kernels_for(legacy_format: str) -> tuple[str, ...]:
    return _KERNELS_FOR_FORMAT[legacy_format]
