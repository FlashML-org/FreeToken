#!/usr/bin/env python3
"""Compare Qwen's production MoE router with FreeToken's HIP Triton candidate.

This GMKtec EVO-X2-only diagnostic does not load a model or modify a server. It uses
Qwen3.6's 256-expert, top-8 router shape, checks every candidate result against
the current PyTorch reference, and reports synchronized GPU timings as JSON.
"""

from __future__ import annotations

import json
import time
from typing import Callable

import torch

from freetoken.kernel.triton.moe_router import fused_topk_softmax
from freetoken.moe.fused import _torch_fused_topk


Router = Callable[[torch.Tensor, int, bool, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]]


def elapsed_ms(operation: Callable[[], object], iterations: int) -> float:
    """Return the synchronized mean operation duration without timing queued GPU work."""

    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        operation()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / iterations


def run_shape(tokens: int, iterations: int) -> dict[str, float | int]:
    """Validate and time one token-batch shape used by Qwen prefill or decode."""

    generator = torch.Generator(device="cuda").manual_seed(tokens * 1009 + 8)
    logits = torch.randn((tokens, 256), device="cuda", dtype=torch.bfloat16, generator=generator)
    reference_weights, reference_ids = _torch_fused_topk(logits, 8, True, None)
    candidate_weights, candidate_ids = fused_topk_softmax(logits, 8, True, None)
    torch.testing.assert_close(candidate_ids, reference_ids)
    torch.testing.assert_close(candidate_weights, reference_weights, rtol=1e-5, atol=1e-6)
    for _ in range(50):
        _torch_fused_topk(logits, 8, True, None)
        fused_topk_softmax(logits, 8, True, None)
    reference_ms = elapsed_ms(lambda: _torch_fused_topk(logits, 8, True, None), iterations)
    candidate_ms = elapsed_ms(lambda: fused_topk_softmax(logits, 8, True, None), iterations)
    return {
        "tokens": tokens,
        "experts": 256,
        "topk": 8,
        "iterations": iterations,
        "torch_ms": reference_ms,
        "triton_ms": candidate_ms,
        "speedup": reference_ms / candidate_ms,
    }


def main() -> None:
    """Emit machine-readable parity and timing evidence for decode and small batches."""

    if not torch.cuda.is_available():
        raise RuntimeError("this diagnostic requires GMKtec EVO-X2's native ROCm device")
    result = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(),
        "hip": torch.version.hip,
        "results": [run_shape(tokens=1, iterations=1000), run_shape(tokens=4, iterations=1000)],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
