"""CPU MoE executor -- native GGUF IQ4_NL / IQ4_XS / Q5_K experts.

The CPU W4A8 GEMV reads the *same* packed banks the GPU offload path streams and
dequantizes inside the K-loop (``iq4_nl_dot`` / ``iq4_xs_dot`` / ``q5_k_dot`` in
csrc/cpu_moe/cpu_moe_ext.cpp). We check it against the GPU ggml MoE kernel
(``ggml_moe_a8_vec``, also W4A8) on byte-identical banks; both quantize activations to
int8, so the only spread is the activation-quant grid + reduction order.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from freetoken.models.gguf.dequant import (
    BLOCK_SHAPE,
    GGML_IQ4_NL,
    GGML_IQ4_XS,
    GGML_Q5_K,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_CASES = [("iq4_nl", GGML_IQ4_NL), ("iq4_xs", GGML_IQ4_XS), ("q5_K", GGML_Q5_K)]


def _random_bank(quant_type: int, rows: int, k: int, seed: int) -> torch.Tensor:
    """[rows, row_bytes] random packed blocks, fp16 scale fields forced finite/positive."""
    block, size = BLOCK_SHAPE[quant_type]
    rng = np.random.default_rng(seed)
    n = rows * (k // block)
    blocks = rng.integers(0, 256, size=(n, size), dtype=np.uint8)
    scales = (0.001 + 0.01 * rng.random(n)).astype(np.float16)
    raw = np.frombuffer(scales.tobytes(), np.uint8).reshape(n, 2)
    blocks[:, 0:2] = raw
    if quant_type in (GGML_Q5_K, GGML_IQ4_XS):
        blocks[:, 2:4] = raw  # dmin / scales_h
    return torch.from_numpy(blocks.reshape(rows, k // block * size)).contiguous()


def _make_cache(fmt: str, quant_type: int, L: int, E: int, H: int, I: int, seed: int):
    from freetoken.kernel.pinned import alloc_pinned_tensor

    def bank_layers(out_f: int, k: int, off: int) -> list[torch.Tensor]:
        # per-layer banks [E, out_f, row_bytes(k)], matching load_gguf_expert_sources
        packed = _random_bank(quant_type, L * E * out_f, k, seed + off).reshape(L, E, out_f, -1)
        layers = []
        for i in range(L):
            pinned = alloc_pinned_tensor(*packed[i].shape, dtype=torch.uint8)
            pinned.copy_(packed[i])
            layers.append(pinned)
        return layers

    return SimpleNamespace(
        quant_format=fmt,
        bank_sources={"gate_up": bank_layers(2 * I, H, 1), "down": bank_layers(H, I, 2)},
        num_layers=L,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
    )


@pytest.mark.parametrize("fmt,quant_type", _CASES)
@pytest.mark.parametrize("bs", [1, 3])
def test_cpu_decode_gguf_quant_matches_gpu(fmt: str, quant_type: int, bs: int):
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.gguf_experts import fused_experts_gguf

    torch.manual_seed(400 + bs)
    L, E, H, I, top_k = 2, 8, 512, 256, 2  # H, I multiples of the 256-elem K-quant block
    layer = 1
    dev = torch.device("cuda")
    cache = _make_cache(fmt, quant_type, L, E, H, I, seed=quant_type)

    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=0,
        max_tokens=bs,
        device=dev,
    )

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    banks = cache.bank_sources
    gpu_out = fused_experts_gguf(
        hidden, banks["gate_up"][layer].to(dev), banks["down"][layer].to(dev),
        w, ids.clone(), "silu", quant_type, quant_type,
    ).float()

    rel = (cpu_out - gpu_out).abs().max() / (gpu_out.abs().max() + 1e-6)
    assert rel < 6e-2, f"{fmt} bs={bs} cpu-vs-gpu rel err {rel.item()}"


if __name__ == "__main__":
    for fmt, quant_type in _CASES:
        for bs in (1, 3):
            test_cpu_decode_gguf_quant_matches_gpu(fmt, quant_type, bs)
            print(f"{fmt} bs={bs} OK")
