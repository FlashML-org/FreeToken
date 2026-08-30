"""Hyper-connection (gated residual) blocks for Qwen3.8-Flash-Next.

Every layer reads and writes ``hc_count`` residual streams packed as ``R [T, hc_count*hidden]``
(stream outer, hidden inner -- the checkpoint layout). On CUDA the mix/combine bodies are the
vendored vLLM Triton kernels (``kernel/triton/hc.py``: grouped_gemma_rmsnorm / hc_silu /
hc_gate_mix / hc_combine) around two ``F.linear`` GEMMs; the pure-torch chain stays as the CPU
path and as the reference the kernels are diffed against. Both keep fp32 intermediates and cast
back at the store, so they agree to fp32 rounding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from freetoken.kernel.triton.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_gate_mix,
    hc_silu,
)
from freetoken.layers import BaseOP, LinearReplicated

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


def grouped_plus_one_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int
) -> torch.Tensor:
    """RMSNorm each of ``num_groups`` equal slices of the last dim on its own fp32 statistic, then scale by (1+w)."""
    xf = x.float().unflatten(-1, (num_groups, -1))
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf.flatten(-2) * (1.0 + weight.float())).to(x.dtype)


class GroupedPlusOneRMSNorm(BaseOP):
    """Per-stream RMSNorm of an ``[..., num_groups*group]`` tensor with one weight element per feature.

    HF ``Qwen4ExpTextRMSNorm(dim, group_size)``. The checkpoint weight is zero-centered and is
    loaded RAW: (1+w) is applied at runtime in fp32, never folded into the bf16 weight (the
    vendored Triton kernel does the same). ``ple.py`` reuses this class for norm_key /
    norm_query / norm_conv, so keep it exported.
    """

    def __init__(self, size: int, eps: float, num_groups: int) -> None:
        self.weight = torch.empty(size)
        self.eps = eps
        self.num_groups = num_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # the kernel is 2D-only, higher-rank callers keep the torch chain
        if x.is_cuda and x.dim() == 2:
            return grouped_gemma_rmsnorm(x, self.weight, self.eps, self.num_groups)
        return grouped_plus_one_rms_norm(x, self.weight, self.eps, self.num_groups)


class GatedResidual(BaseOP):
    """One hyper-connection block: ``mix`` reads the residual streams, ``combine`` writes a block output back.

    Frozen API (HF ``Qwen4ExpTextGatedResidual``, formulas at modeling_qwen4_exp.py:959-969)::

        x, s = hc.mix(R)          # R [T, hc_count*hidden] -> x [T, hidden], s [T, hc_count] or None
        y    = block(x)           # attention / GDN / MoE, plain [T, hidden] -> [T, hidden]
        R    = hc.combine(R, y, s)

        Rn      = groupRMSNorm(R) * (1 + hc_norm.weight)        # per hidden-size stream, fp32 stats
        lora, s = input_mix_weight_down_block_inject(Rn)        # merged GEMM: [lowrank | hc_count | pad]
        gate    = input_mix_weight_up(silu(lora / hc_count))
        x       = mean_i(sigmoid(gate_i) * Rn_i)
        R'_i    = R_i + 2*sigmoid(s_i / hc_count) * y

    ``s`` is the RAW inject logit slice of the merged GEMM (pre 2*sigmoid), which is what the
    vendored ``hc_combine`` kernel expects; ``combine`` applies the activation. The merged weight
    is ``[lowrank + hc_count + pad, hc_count*hidden]`` (Qwen3.8: 320 + 4 + 12 = 336 rows), the pad
    rows are zero and their GEMM output is dropped. ``use_combine=False`` is the top-level mixer:
    it owns the unmerged ``input_mix_weight_down``, returns ``s = None`` and has no ``combine``.

    Weight keys (checkpoint names, prefix stripped): ``hc_norm.weight``,
    ``input_mix_weight_down_block_inject.weight`` (loader: concat of
    ``input_mix_weight_down`` [lowrank, hc*hidden], ``block_inject_weight`` [hc_count, hc*hidden]
    and ``pad`` zero rows), ``input_mix_weight_up.weight``.

    Launch budget on CUDA: ``mix`` is 3 kernels around 2 GEMMs, ``combine`` is 1.
    """

    def __init__(self, config: ModelConfig, use_combine: bool = True) -> None:
        args = config.qwen4_args
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        self.lowrank = args.hc_lowrank
        self.use_combine = use_combine
        width = args.ple_state_width
        self.hc_norm = GroupedPlusOneRMSNorm(width, config.rms_norm_eps, self.hc_count)
        if use_combine:
            # 16-row alignment for the merged skinny GEMM (vLLM hyperconnection.py:98)
            self.pad_size = (-(self.lowrank + self.hc_count)) % 16
            self.input_mix_weight_down_block_inject = LinearReplicated(
                width, self.lowrank + self.hc_count + self.pad_size, has_bias=False
            )
        else:
            self.pad_size = 0
            self.input_mix_weight_down = LinearReplicated(width, self.lowrank, has_bias=False)
        self.input_mix_weight_up = LinearReplicated(self.lowrank, width, has_bias=False)

    def _down(self, rn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Run the down GEMM and split off the raw inject logits; the pad columns are dropped."""
        if not self.use_combine:
            return self.input_mix_weight_down.forward(rn), None
        down = self.input_mix_weight_down_block_inject.forward(rn)
        # both slices keep unit inner stride, so the kernels read them without a copy
        return down[:, : self.lowrank], down[:, self.lowrank : self.lowrank + self.hc_count]

    def _mix_kernel(self, R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        rn = grouped_gemma_rmsnorm(R, self.hc_norm.weight, self.hc_norm.eps, self.hc_count)
        lora, s = self._down(rn)
        gate = self.input_mix_weight_up.forward(hc_silu(lora, self.hc_count))
        return hc_gate_mix(rn, gate, self.hc_count), s

    def _mix_torch(self, R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        rn = grouped_plus_one_rms_norm(R, self.hc_norm.weight, self.hc_norm.eps, self.hc_count)
        lora, s = self._down(rn)
        lora = F.silu(lora.float() / self.hc_count)
        gate = self.input_mix_weight_up.forward(lora.to(R.dtype))
        mixed = torch.sigmoid(gate.float()).unflatten(-1, (self.hc_count, self.hidden_size))
        mixed = mixed * rn.float().unflatten(-1, (self.hc_count, self.hidden_size))
        return mixed.mean(-2).to(R.dtype), s

    def _combine_torch(self, R: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        inject = 2.0 * torch.sigmoid(s.float() / self.hc_count)
        out = R.float().unflatten(-1, (self.hc_count, self.hidden_size))
        out = out + y.float().unsqueeze(-2) * inject.unsqueeze(-1)
        return out.flatten(-2).to(R.dtype)

    def mix(self, R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Return the block input ``x [T, hidden]`` and the inject logits ``s [T, hc_count]`` (None if no combine)."""
        return self._mix_kernel(R) if R.is_cuda else self._mix_torch(R)

    def combine(self, R: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Inject the block output ``y [T, hidden]`` back into every stream of ``R``."""
        if R.is_cuda:
            return hc_combine(R, y, s, self.hc_count)
        return self._combine_torch(R, y, s)


__all__ = ["GatedResidual", "GroupedPlusOneRMSNorm", "grouped_plus_one_rms_norm"]
