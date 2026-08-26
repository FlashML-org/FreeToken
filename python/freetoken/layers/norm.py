from typing import Tuple

import torch

from .base import BaseOP


def _triton_importable() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except Exception:
        return False


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from freetoken.kernel.backend import is_flashinfer_installed

        if is_flashinfer_installed():
            from flashinfer import rmsnorm
        elif _triton_importable():
            from freetoken.kernel.triton.norm import rmsnorm
        else:
            from freetoken.kernel.torch_fallback import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)


class GemmaRMSNorm(BaseOP):
    """Gemma4-style RMSNorm backed directly by sgl_kernel.

    Gemma4 scales by the raw checkpoint weight. ``with_scale=False`` uses a
    runtime ones vector that is intentionally not part of ``state_dict``.
    """

    def __init__(self, size: int, eps: float, with_scale: bool = True) -> None:
        from freetoken.kernel.backend import is_sgl_kernel_installed

        if is_sgl_kernel_installed():
            from sgl_kernel import fused_add_rmsnorm, rmsnorm
        else:
            from freetoken.kernel.triton.norm import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.size = size
        self.with_scale = with_scale
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm
        if with_scale:
            self.weight = torch.empty(size)
        else:
            self._ones_weight: torch.Tensor | None = None

    def _kernel_weight(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_scale:
            return self.weight
        if self._ones_weight is None:
            self._ones_weight = torch.ones(self.size, device=x.device, dtype=x.dtype)
        return self._ones_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self.rmsnorm(x, self._kernel_weight(x), self.eps)
        original_shape = x.shape
        out = self.rmsnorm(
            x.contiguous().reshape(-1, original_shape[-1]),
            self._kernel_weight(x),
            self.eps,
        )
        return out.reshape(original_shape)

    def forward_add_residual(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.fused_add_rmsnorm(x, residual, self._kernel_weight(x), self.eps)
        return x, residual


class GemmaPlusOneRMSNorm(BaseOP):
    """(1 + w)-scaled RMSNorm (Gemma semantics: the checkpoint stores ``scale - 1``
    and the effective multiplier is ``1 + weight``, added in fp32 at runtime --
    never folded into the bf16 weight, which would round away the precision the
    format exists to keep). Used by MiniMax-M3 (``use_gemma_norm``) for the decoder
    layernorms, the per-head q/k norms, and the indexer q/k norms.

    Per-head 3D inputs are collapsed to 2D before the kernel call: flashinfer's
    ``gemma_rmsnorm`` CUDA binding is CHECK_DIM(2) (it rejects 3D outright on
    wheels without the CuTe path), and the per-head weight makes the 2D view
    exactly equivalent. The M3 call sites pass contiguous buffers, so the views
    are free; a non-contiguous 3D input is rejected rather than silently copied
    (an in-place norm on a copy would be dropped).
    """

    def __init__(self, size: int, eps: float) -> None:
        from freetoken.kernel.backend import is_flashinfer_installed

        if is_flashinfer_installed():
            from flashinfer.norm import gemma_rmsnorm
        else:
            from freetoken.kernel.triton.norm import gemma_rmsnorm

        self.eps = eps
        self.size = size
        self.weight = torch.empty(size)
        self.gemma_rmsnorm = gemma_rmsnorm

    def _flat(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x
        assert x.is_contiguous(), "per-head gemma norm needs a contiguous buffer"
        return x.view(-1, self.size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gemma_rmsnorm(self._flat(x), self.weight, self.eps).view(x.shape)

    def forward_inplace(self, x: torch.Tensor) -> None:
        flat = self._flat(x)
        self.gemma_rmsnorm(flat, self.weight, self.eps, out=flat)


class GemmaPlusOneRMSNormFused(BaseOP):
    """(1 + w)-scaled RMSNorm with the fused-add-residual API of ``RMSNormFused``
    (Gemma semantics, see :class:`GemmaPlusOneRMSNorm`). Drop-in for the decoder
    layernorm seam: ``forward(x, residual)`` returns ``(normed, residual)``."""

    def __init__(self, size: int, eps: float) -> None:
        from freetoken.kernel.backend import is_flashinfer_installed

        if is_flashinfer_installed():
            from flashinfer.norm import gemma_fused_add_rmsnorm, gemma_rmsnorm
        else:
            from freetoken.kernel.triton.norm import (
                gemma_fused_add_rmsnorm,
                gemma_rmsnorm,
            )

        self.eps = eps
        self.weight = torch.empty(size)
        self.gemma_rmsnorm = gemma_rmsnorm
        self.gemma_fused_add_rmsnorm = gemma_fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.gemma_rmsnorm(x, self.weight, self.eps), x
        self.gemma_fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from freetoken.kernel.backend import is_flashinfer_installed

        if is_flashinfer_installed():
            from flashinfer import fused_add_rmsnorm, rmsnorm
        elif _triton_importable():
            from freetoken.kernel.triton.norm import fused_add_rmsnorm, rmsnorm
        else:
            from freetoken.kernel.torch_fallback import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
