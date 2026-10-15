"""ds_fp4 (DeepSeek W4A8) experts against an INDEPENDENT torch transcription over
``fp4_gemm(act_quant(x, block), W)`` at both fp8 activation blocks (128 on V4, 32 on V4.1):

    x_q  = fp8_roundtrip(x, block)                       # act_quant(..., inplace)
    gate = bf16(x_q @ deq(W1)^T), up = bf16(x_q @ deq(W3)^T)   # fp4_gemm outputs bf16
    h    = bf16(silu(min(gate, L)) * clamp(up, -L, L))
    y_r  = bf16(w_r * (fp8_roundtrip(h, block) @ deq(W2)^T))   # routing weight in the down epilogue
    y    = sum_r y_r

The routing weight scales the down output (the placement main serves V4 with); the reference
``Expert.forward`` scales the intermediate before its fp8 quant instead. The GEMV (decode), the
grouped GEMM (prefill) and the CPU executor all follow this transcription.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

E, H, I, TOP_K, LIMIT = 16, 512, 256, 4, 7.0
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _banks(device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def u8(*shape, low=0, high=256):
        return torch.randint(low, high, shape, dtype=torch.uint8, generator=g).to(device)

    return (u8(E, 2 * I, H // 2), u8(E, 2 * I, H // 32, low=120, high=130), u8(E, H, I // 2), u8(E, H, I // 32, low=120, high=130))


def _dequant(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """[E, N, K//2] e2m1 pairs (even channel low nibble) + [E, N, K//32] e8m0 -> [E, N, K] fp32."""
    lut = E2M1.to(packed.device)
    lo, hi = lut[(packed & 0xF).long()], lut[(packed >> 4).long()]
    vals = torch.stack([lo, hi], dim=-1).flatten(-2)  # [E, N, K]
    s = torch.exp2(scale.view(torch.uint8).float() - 127.0).repeat_interleave(32, dim=-1)
    return vals * s


def _fp8_roundtrip(x: torch.Tensor, block: int) -> torch.Tensor:
    g = x.float().unflatten(-1, (-1, block))
    s = torch.exp2(torch.ceil(torch.log2(g.abs().amax(-1).clamp_min(1e-4) / 448.0)))
    q = (g / s.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (q * s.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


def reference(x, slots, weights, banks, block):
    """The MoE over dequantized banks: per-route bf16 expert outputs accumulated into an fp32 ``y``,
    returned in fp32 (the kernels round that sum to bf16)."""
    gu_p, gu_s, dn_p, dn_s = banks
    W13 = _dequant(gu_p, gu_s)  # [E, 2I, H]
    W2 = _dequant(dn_p, dn_s)  # [E, H, I]
    xq = _fp8_roundtrip(x, block).float()
    T = x.shape[0]
    y = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    for t in range(T):
        for r in range(TOP_K):
            e = int(slots[t, r])
            gu = (xq[t] @ W13[e].T).to(torch.bfloat16).float()
            gate, up = gu[:I].clamp(max=LIMIT), gu[I:].clamp(-LIMIT, LIMIT)
            h = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
            hq = _fp8_roundtrip(h.view(1, -1), block).float().view(-1)
            y[t] += (weights[t, r].float() * (hq @ W2[e].T)).to(torch.bfloat16).float()
    return y


def _inputs(T, device, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(T, H, dtype=torch.bfloat16, generator=g) * 0.5).to(device)
    slots = torch.stack([torch.randperm(E, generator=g)[:TOP_K] for _ in range(T)]).to(device=device, dtype=torch.int32).contiguous()
    w = torch.rand(T, TOP_K, generator=g)
    w = (1.5 * w / w.sum(-1, keepdim=True)).float().to(device).contiguous()  # route_scale-style weights > 1 too
    return x, slots, w


@pytest.mark.parametrize("block", [32, 128])
def test_gemv_path_matches_the_transcription(block):
    from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

    banks = _banks("cuda")
    x, slots, w = _inputs(6, "cuda")
    got = routed_experts_fp4(x, slots, w, *banks, LIMIT, act_block=block).float()
    want = reference(x, slots, w, banks, block)
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("block", [32, 128])
def test_cpu_executor_matches_the_transcription(block):
    """The ``ds_fp4`` CPU executor (cpu / hybrid decode) implements the same contract as the GPU path."""
    from types import SimpleNamespace

    from freetoken.kernel.pinned import alloc_pinned_tensor
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    gu_p, gu_s, dn_p, dn_s = _banks("cuda")
    pinned = {}
    for name, t in (("gate_up_packed", gu_p), ("gate_up_scale", gu_s), ("down_packed", dn_p), ("down_scale", dn_s)):
        buf = alloc_pinned_tensor(*t.shape, dtype=torch.uint8)
        buf.copy_(t.cpu())
        pinned[name] = buf
    cache = SimpleNamespace(quant_format="ds_fp4", bank_sources={k: [v] for k, v in pinned.items()}, num_layers=1, num_experts=E)
    dev = torch.device("cuda", 0)
    x, slots, w = _inputs(2, dev)
    ex = CpuMoeExecutor(cache, top_k=TOP_K, activation="silu", apply_router_weight_on_input=False, num_threads=4, max_tokens=2, device=dev, swiglu_limit=LIMIT, act_block=block)
    got = ex.decode(0, x, w, slots.clone()).clone().float()
    torch.cuda.synchronize()
    del ex
    want = reference(x, slots, w, (gu_p, gu_s, dn_p, dn_s), block)
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)


def test_grouped_prefill_path_matches_the_transcription():
    from freetoken.moe import fused_ds_fp4

    banks = _banks("cuda")
    x, slots, w = _inputs(64, "cuda")
    got = fused_ds_fp4.routed_experts_fp4_prefill(x, slots, w, *banks, LIMIT, E, act_block=32).float()
    want = reference(x, slots, w, banks, 32)
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("block", [32, 128])
def test_inactive_routes_contribute_zero_without_reading_their_slots(block):
    """Hybrid decode hands the GPU kernel slot -1 (weight 0) for the routes the CPU computes. The kernel
    must produce exactly zero for them without touching any slot: every slot the active routes do not
    use is filled with 0xFF (an e8m0 scale of NaN), and the result still equals the reference over the
    active routes and the CPU executor's output for the same split."""
    from types import SimpleNamespace

    from freetoken.kernel.pinned import alloc_pinned_tensor
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

    gu_p, gu_s, dn_p, dn_s = (t.clone() for t in _banks("cuda"))
    x, slots, w = _inputs(2, "cuda")
    # routes: token 0 keeps routes 0..1 on the GPU, token 1 keeps route 3; the rest go to the CPU
    on_gpu = torch.zeros_like(slots, dtype=torch.bool)
    on_gpu[0, :2] = True
    on_gpu[1, 3] = True
    gpu_slots = torch.where(on_gpu, slots, -1)
    gpu_w = torch.where(on_gpu, w, 0.0)
    used = set(slots[on_gpu].tolist())
    poison = [e for e in range(E) if e not in used]
    for t in (gu_p, gu_s, dn_p, dn_s):
        t[poison] = 0xFF
    got = routed_experts_fp4(x, gpu_slots, gpu_w, gu_p, gu_s, dn_p, dn_s, LIMIT, act_block=block)
    assert torch.isfinite(got).all()
    clean = _banks("cuda")
    # the poisoned-bank result equals the clean-bank result with the same holes (neither reads them),
    # and each share matches the reference over its own routes (a zero weight drops a route)
    got_clean = routed_experts_fp4(x, gpu_slots, gpu_w, *clean, LIMIT, act_block=block)
    assert torch.equal(got, got_clean)
    torch.testing.assert_close(got.float(), reference(x, slots, gpu_w, clean, block), atol=2e-2, rtol=2e-2)
    # the CPU executor computes the complementary split from the same raw ids (ids < 0 skipped)
    pinned = {}
    for name, t in (("gate_up_packed", clean[0]), ("gate_up_scale", clean[1]), ("down_packed", clean[2]), ("down_scale", clean[3])):
        buf = alloc_pinned_tensor(*t.shape, dtype=torch.uint8)
        buf.copy_(t.cpu())
        pinned[name] = buf
    cache = SimpleNamespace(quant_format="ds_fp4", bank_sources={k: [v] for k, v in pinned.items()}, num_layers=1, num_experts=E)
    ex = CpuMoeExecutor(cache, top_k=TOP_K, activation="silu", apply_router_weight_on_input=False, num_threads=4, max_tokens=2, device=torch.device("cuda", 0), swiglu_limit=LIMIT, act_block=block)
    cpu_ids = torch.where(on_gpu, -1, slots)
    cpu_out = ex.decode(0, x, w, cpu_ids.clone()).clone()
    torch.cuda.synchronize()
    del ex
    torch.testing.assert_close(cpu_out.float(), reference(x, slots, torch.where(on_gpu, 0.0, w), clean, block), atol=2e-2, rtol=2e-2)
