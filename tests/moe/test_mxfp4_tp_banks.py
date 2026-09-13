"""The ds_fp4 expert kernel must size its banks per rank under tensor parallelism.

DeepSeek-V4's routed experts are MXFP4 (e2m1 pairs + e8m0 per-32 scales, no global).
Under TP the intermediate dim splits: each rank caches rows [i_lo, i_lo+I/tp) of
``gate``/``up`` and the matching columns of ``down``, and the MoE layer's all-reduce
sums the partial ``down`` products back into the full output. That is the memory win
that makes the 143 GB of expert banks fit alongside the KV cache -- but only if the
kernel's ``layout()`` agrees with what the reader yields, since the loader binds
state-dict tensors by EXACT shape.

Two things are pinned here:

* the kernel must accept ``tp_size > 1`` and lay out RANK-LOCAL banks (the ``I`` axis
  divides, the ``H`` axis never does);
* summing the per-rank expert outputs must reproduce the single-rank result, because
  ``down`` is exactly linear along the axis being split.

The CUDA test runs on one device: it builds the full banks, then slices them into
per-rank banks and calls the same kernel per slice. No process group is needed to
prove the numerical identity.
"""

import pytest
import torch

from freetoken.layers.quantization import moe as moe_kernels  # noqa: F401  (registers kernels)
from freetoken.layers.quantization.moe.base import MoEConfig
from freetoken.layers.quantization.moe.mxfp4 import TritonMxfp4MoEKernel

# I is a multiple of 128 (the FP8 block / 32-wide e8m0 grid and TP=4 both divide it),
# H is a multiple of 32 (the scale grid along H). Same shape family as the checkpoint.
H, I, E, TOPK = 512, 256, 4, 2
TP = 2


def _cfg(tp_size: int, tp_rank: int = 0) -> MoEConfig:
    return MoEConfig(
        num_experts=E, hidden=H, intermediate=I, top_k=TOPK,
        tp_rank=tp_rank, tp_size=tp_size, strategy="offload", decode_target="gpu",
    )


def test_kernel_accepts_tensor_parallelism():
    """``tp_ok`` must be set: a rank-local kernel is what removes DSV4's TP=1 guard."""
    reason = TritonMxfp4MoEKernel().unusable_reason(_cfg(TP))
    assert reason is None, f"ds_fp4 kernel rejected TP={TP}: {reason}"


def test_layout_sizes_banks_for_one_rank():
    kernel = TritonMxfp4MoEKernel()
    i = I // TP
    layout = kernel.layout(_cfg(TP, tp_rank=0))

    assert layout["gate_up"].shape == (2 * i, H // 2), "gate|up rows must be the rank's I-block"
    assert layout["gate_up_scale"].shape == (2 * i, H // 32)
    assert layout["down"].shape == (H, i // 2), "down packs two e2m1 codes per byte along I"
    assert layout["down_scale"].shape == (H, i // 32), "one e8m0 scale per 32 elements along I"
    # The hidden axis is replicated: every rank reads the whole of H.
    assert layout["down"].shape[0] == H and layout["gate_up"].shape[1] == H // 2


def test_layout_at_tp1_is_unchanged():
    """No regression for the single-GPU path: TP=1 must keep the full-width banks."""
    full = TritonMxfp4MoEKernel().layout(_cfg(1))
    assert full["gate_up"].shape == (2 * I, H // 2)
    assert full["down"].shape == (H, I // 2)


def _banks(i: int, seed: int = 0):
    """Random-but-valid ds_fp4 banks for an intermediate width of ``i``."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    gate_up = torch.randint(0, 256, (E, 2 * i, H // 2), generator=g, dtype=torch.int64).to(torch.uint8)
    gate_up_scale = torch.full((E, 2 * i, H // 32), 121, dtype=torch.uint8)  # e8m0 2^-6
    down = torch.randint(0, 256, (E, H, i // 2), generator=g, dtype=torch.int64).to(torch.uint8)
    down_scale = torch.full((E, H, i // 32), 121, dtype=torch.uint8)
    return [b.cuda() for b in (gate_up, gate_up_scale, down, down_scale)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rank_partials_sum_to_the_single_rank_output():
    """Splitting I and summing the per-rank ``down`` products is exact.

    ``down`` is a linear map whose input axis is the intermediate dim, so the full
    expert output is the sum of the per-rank outputs over disjoint I-blocks. If the
    kernel reads a width that disagrees with its banks -- e.g. it still uses
    ``layer.intermediate_size`` while the banks are ``I/tp`` -- this is where it shows.
    """
    from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

    torch.manual_seed(0)
    dev = torch.device("cuda", 0)
    x = torch.randn(2, H, dtype=torch.bfloat16, device=dev) * 0.3
    ids = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.int32, device=dev)
    wts = torch.rand(2, TOPK, device=dev)
    limit = 10.0

    y_full = routed_experts_fp4(x, ids.clone(), wts, *_banks(I), limit)

    # Build the per-rank banks by slicing the SAME full banks, so rank 0 + rank 1
    # together are exactly the full expert.
    i = I // TP
    full = _banks(I)
    partial = None
    for r in range(TP):
        rows = slice(r * 2 * i, (r + 1) * 2 * i)          # gate|up rows carry I (x2, fused)
        cols = slice(r * i // 2, (r + 1) * i // 2)        # down packs 2 codes per byte
        scale_cols = slice(r * i // 32, (r + 1) * i // 32)  # one e8m0 per 32
        banks = [
            full[0][:, rows].contiguous(),
            full[1][:, rows].contiguous(),
            full[2][:, :, cols].contiguous(),
            full[3][:, :, scale_cols].contiguous(),
        ]
        y_rank = routed_experts_fp4(x, ids.clone(), wts, *banks, limit)
        partial = y_rank if partial is None else partial + y_rank

    # Same decode function, same bytes, partitioned input axis: only the fp32
    # accumulation order of the K reduction differs.
    assert torch.allclose(partial.float(), y_full.float(), atol=1e-1, rtol=5e-2), (
        f"TP={TP} partial sum diverges from the single-rank expert output: "
        f"max abs {(partial.float() - y_full.float()).abs().max():.4g}, "
        f"output scale {y_full.float().abs().mean():.4g}"
    )
