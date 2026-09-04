"""Fused GGUF MoE past CUDA's gridDim.z ceiling.

``moe_vec.cuh`` carries the routed-pair index (token * top_k + slot) in ``gridDim.z``,
which CUDA caps at 65535 on every architecture. ``fused_experts_gguf`` issues two launches
-- gate_up as ``(top_k, N)`` and down as ``(1, N * top_k)`` -- and both reach
``z = N * top_k``. So a prefill wider than ``65535 / top_k`` tokens used to fail the launch
with ``cudaErrorInvalidConfiguration``, which torch surfaces as the rather unhelpful
"CUDA error: invalid argument". At top_k 8 that ceiling is 8191 tokens, which any long
prompt crosses; it is not architecture- or OS-specific. Reported against PR #131 with a
20k-token prompt on an RTX 3070.

The launcher now issues token-aligned chunks. The equality assertion below is the part
that matters: chunking is only correct if each chunk's ``vy`` / ``dst`` / ``topk_ids``
offsets line up with the kernel's own ``token = blockIdx.z / topk`` arithmetic. A fix that
launches but mis-offsets would pass a smoke test, look entirely healthy, and emit fluent
nonsense in service.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

TOP_K = 8
SLOTS, H, I = 16, 512, 256
CEIL = 65535 // TOP_K  # 8191 tokens


def _pack_q4_0(S: int, OUT: int, K: int, dev) -> torch.Tensor:
    """Valid Q4_0 rows: per 32-element block, a finite fp16 scale then 16 nibble bytes.

    Random bytes are not a usable bank here -- the 2-byte scale would sometimes decode to
    Inf/NaN, and the resulting NaNs would fail the equality check for reasons that have
    nothing to do with the grid geometry under test.
    """
    nb = K // 32
    nib = torch.randint(0, 256, (S, OUT, nb, 16), dtype=torch.uint8)
    scale = (0.02 + 0.03 * torch.rand(S, OUT, nb)).to(torch.float16)
    sb = scale.view(torch.uint8).reshape(S, OUT, nb, 2)
    return torch.cat([sb, nib], dim=-1).reshape(S, OUT, nb * 18).contiguous().to(dev)


@pytest.fixture(scope="module")
def banks():
    dev = torch.device("cuda")
    torch.manual_seed(0)
    max_t = 2 * CEIL + 16
    return {
        "gate_up": _pack_q4_0(SLOTS, 2 * I, H, dev),
        "down": _pack_q4_0(SLOTS, H, I, dev),
        "ids": torch.randint(0, SLOTS, (max_t, TOP_K), dtype=torch.int32, device=dev),
        "w": torch.rand(max_t, TOP_K, device=dev, dtype=torch.float32),
        "x": (torch.randn(max_t, H, device=dev, dtype=torch.bfloat16) * 0.5).contiguous(),
    }


def _run(b, n: int) -> torch.Tensor:
    from freetoken.models.gguf.dequant import GGML_Q4_0
    from freetoken.moe.fused_q4_0 import fused_experts_gguf

    return fused_experts_gguf(
        b["x"][:n].contiguous(), b["gate_up"], b["down"],
        b["w"][:n].contiguous(), b["ids"][:n].contiguous(),
        "silu", GGML_Q4_0,
    )


@pytest.mark.parametrize(
    "n",
    [
        CEIL - 1,        # 65528: last z that fits
        CEIL,            # 65528 + 8: first launch that used to fail
        CEIL + 1,
        2 * CEIL + 7,    # several chunks, deliberately not a chunk multiple
    ],
)
def test_moe_vec_launches_past_grid_z_ceiling(banks, n):
    """A launch whose z exceeds 65535 must succeed rather than raise 'invalid argument'."""
    out = _run(banks, n)
    torch.cuda.synchronize()
    assert out.shape == (n, H)
    assert torch.isfinite(out).all(), "output contains non-finite values"


@pytest.mark.parametrize("n", [CEIL - 1, CEIL, CEIL + 1, 2 * CEIL + 7])
def test_chunking_does_not_disturb_rows(banks, n):
    """Rows below the ceiling must be bit-identical however many chunks were launched.

    This is what catches a wrong per-chunk pointer offset, which a crash test cannot.
    """
    ref = _run(banks, 64)
    got = _run(banks, n)
    torch.cuda.synchronize()
    assert torch.equal(got[:64], ref), (
        f"first 64 rows differ at n={n}: chunk offsets are wrong "
        f"(max abs diff {(got[:64].float() - ref.float()).abs().max().item()})"
    )
