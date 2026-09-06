"""CPU MoE executor -- native GGUF Q4_0 experts (``--moe-backend cpu``/``hybrid`` for
the gemma4 GGUF checkpoint).

The CPU W4A16 GEMV (``q4_0_dot`` in csrc/cpu_moe/cpu_moe_ext.cpp) reads the *same*
packed Q4_0 banks the GPU offload path streams and dequantizes weights inside the
K-loop. We check it against the reference dequant (models/gguf/dequant.py) + the
production bf16 GPU decode kernel on byte-identical banks: both are W4A16, so the
only spread is weight bf16-rounding + reduction order -> tight relative tolerance.

Part 2 covers CUDA-graph capture/replay (the cudaLaunchHostFunc submit/sync nodes
must recompute from the freshly written pinned routing on each replay).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _skip_if_unsafe_rocm_graph() -> None:
    """Direct executor graph test requires native HIP capture/replay support."""
    if getattr(torch.version, "hip", None) is None:
        return
    from freetoken.kernel import _cpu_moe

    if not _cpu_moe.memops_use_shared_signal():
        pytest.skip("native ROCm graph handshake unavailable")


def _pack_q4_0(nibbles: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Pack [S, OUT, K] uint8 nibble codes (0..15) + [S, OUT, K//32] fp16 scales into
    the native Q4_0 row layout [S, OUT, K//32*18]: each 32-elem block is a 2-byte fp16
    scale followed by 16 bytes whose byte j holds elem j (low nibble) and elem j+16
    (high nibble) -- the exact layout ``dequant_q4_0`` reads back."""
    S, OUT, K = nibbles.shape
    nb = K // 32
    blk = nibbles.reshape(S, OUT, nb, 32)
    lo = blk[..., :16]      # elems 0..15 -> low nibbles of bytes 0..15
    hi = blk[..., 16:]      # elems 16..31 -> high nibbles of bytes 0..15
    packed = (lo | (hi << 4)).to(torch.uint8)                       # [S, OUT, nb, 16]
    d_bytes = scale.to(torch.float16).view(torch.uint8).reshape(S, OUT, nb, 2)
    row = torch.cat([d_bytes, packed], dim=-1)                      # [S, OUT, nb, 18]
    return row.reshape(S, OUT, nb * 18).contiguous()


def _dequant_bank(packed: torch.Tensor, K: int, dev) -> torch.Tensor:
    """[S, OUT, K//32*18] packed Q4_0 -> [S, OUT, K] bf16 (storage order == elem order)."""
    from freetoken.models.gguf.dequant import GGML_Q4_0, dequantize

    S, OUT, _ = packed.shape
    flat = dequantize(packed.reshape(-1), GGML_Q4_0, torch.bfloat16)
    return flat.reshape(S, OUT, K).to(dev)


def _make_q4_0_cache(L, E, H, I, seed=0):
    """Random but valid native Q4_0 banks for the cpu backend (pinned host tensors)."""
    from freetoken.kernel.pinned import alloc_pinned_tensor

    torch.manual_seed(seed)
    S = L * E

    def rows(OUT, K):
        nib = torch.randint(0, 16, (S, OUT, K), dtype=torch.uint8)
        scale = 0.02 + 0.03 * torch.rand(S, OUT, K // 32)
        packed = _pack_q4_0(nib, scale)                            # [S, OUT, K//32*18]
        pinned = alloc_pinned_tensor(*packed.shape, dtype=torch.uint8)
        pinned.copy_(packed)
        return pinned

    return SimpleNamespace(
        quant_format="q4_0",
        bank_sources={"gate_up": list(rows(2 * I, H).split(E)), "down": list(rows(H, I).split(E))},
        num_layers=L,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
    )


@pytest.mark.parametrize("bs", [1, 3, 8])
def test_cpu_decode_q4_0_matches_dequant_then_gpu(bs):
    """CPU inline-dequant Q4_0 GEMV vs. canonical dequant_q4_0 + bf16 GPU decode."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused import fused_experts_decode_impl

    torch.manual_seed(400 + bs)
    L, E, H, I, top_k = 3, 16, 2816, 704, 8   # gemma-4-26B-A4B geometry (H,I % 32 == 0)
    layer = 1
    dev = torch.device("cuda")
    cache = _make_q4_0_cache(L, E, H, I)

    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="gelu_tanh",
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

    b = cache.bank_sources
    gate_up_layer = _dequant_bank(b["gate_up"][layer], H, dev)  # [E, 2I, H]
    down_layer = _dequant_bank(b["down"][layer], I, dev)        # [E, H, I]
    gpu_out = fused_experts_decode_impl(
        hidden, gate_up_layer, down_layer, w, ids.clone(), "gelu_tanh", False
    ).float()

    rel = (cpu_out - gpu_out).abs().max() / (gpu_out.abs().max() + 1e-6)
    assert rel < 2e-2, f"q4_0 bs={bs} rel err {rel.item()}"


@pytest.mark.slow
def test_cpu_decode_q4_0_matches_ggml_mmvq():
    """Sanity: the CPU W4A16 GEMV lands close to the GPU ggml MMVQ (W4A8) kernel the
    offload path uses -- a looser tol since MMVQ quantizes activations to int8."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused_q4_0 import fused_experts_gguf_q4_0

    torch.manual_seed(77)
    L, E, H, I, top_k = 2, 16, 2816, 704, 8
    layer, bs = 1, 4
    dev = torch.device("cuda")
    cache = _make_q4_0_cache(L, E, H, I)

    ex = CpuMoeExecutor(
        cache, top_k=top_k, activation="gelu_tanh",
        apply_router_weight_on_input=False, num_threads=0, max_tokens=bs, device=dev,
    )

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    b = cache.bank_sources
    gate_up_layer = b["gate_up"][layer].to(dev)
    down_layer = b["down"][layer].to(dev)
    gpu_out = fused_experts_gguf_q4_0(
        hidden, gate_up_layer, down_layer, w, ids.clone(), "gelu_tanh"
    ).float()

    rel = (cpu_out - gpu_out).abs().max() / (gpu_out.abs().max() + 1e-6)
    assert rel < 6e-2, f"q4_0 vs ggml-mmvq rel err {rel.item()}"


def test_cpu_moe_decode_q4_0_cuda_graph_replay():
    """Q4_0 CPU path under capture/replay: the host nodes must recompute the GEMV from
    the freshly written pinned routing on each replay (dep flows through pinned buffers)."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused import fused_experts_decode_impl

    _skip_if_unsafe_rocm_graph()

    torch.manual_seed(9)
    L, E, H, I, top_k = 2, 8, 2816, 704, 8
    layer, bs = 1, 4
    cache = _make_q4_0_cache(L, E, H, I)

    dev = torch.device("cuda")
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    ex = CpuMoeExecutor(
        cache, top_k=top_k, activation="gelu_tanh",
        apply_router_weight_on_input=False, num_threads=8, max_tokens=bs, device=dev,
    )
    b = cache.bank_sources
    gate_up_layer = _dequant_bank(b["gate_up"][layer], H, dev)
    down_layer = _dequant_bank(b["down"][layer], I, dev)

    def reference(hidden, ids, w):
        return fused_experts_decode_impl(
            hidden, gate_up_layer, down_layer, w, ids.clone(), "gelu_tanh", False
        ).float()

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.randint(0, E, (bs, top_k), device=dev, dtype=torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    ex.decode(layer, hidden, w, ids)  # eager warmup: materialize buffers + task
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        out_cap = ex.decode(layer, hidden, w, ids)
    torch.cuda.synchronize()

    for it in range(3):
        torch.manual_seed(600 + it)
        hidden.copy_(torch.randn(bs, H, dtype=torch.bfloat16) * 0.5)
        ids.copy_(torch.randint(0, E, (bs, top_k), dtype=torch.int32))
        w.copy_(torch.rand(bs, top_k, dtype=torch.float32))
        g.replay()
        torch.cuda.synchronize()
        ref = reference(hidden, ids, w)
        rel = (out_cap.float() - ref).abs().max() / (ref.abs().max() + 1e-6)
        assert rel < 2e-2, f"q4_0 replay {it} rel err {rel.item()}"

    print("cpu moe cuda graph replay (q4_0) OK")


if __name__ == "__main__":
    for bs in (1, 3, 8):
        test_cpu_decode_q4_0_matches_dequant_then_gpu(bs)
        print(f"q4_0 bs={bs} OK")
    test_cpu_decode_q4_0_matches_ggml_mmvq()
    print("q4_0 vs ggml-mmvq OK")
    test_cpu_moe_decode_q4_0_cuda_graph_replay()
