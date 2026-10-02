"""CPU MoE executor (``CpuMoeExecutor``) tests.

Part 1 -- numerical alignment: the CPU SwiGLU MoE GEMV vs FreeToken's production
GPU decode kernels (bf16/nvfp4/mxfp4/ds_fp4) on identical banks and routing across
batch sizes (fp32-accumulate, so the only spread is reduction order -> tight tol).

Part 2 -- CUDA-graph capture/replay: the cudaLaunchHostFunc submit/sync host nodes
end to end (eager decode, capture, replay with *new* data -> result tracks the new
routing, i.e. the dependency flows through the pinned buffers, not baked-in).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as Fn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _make_cache(L, E, H, I, scale=0.1):
    from freetoken.kernel.pinned import alloc_pinned_tensor

    gate_up = alloc_pinned_tensor(L * E, 2 * I, H, dtype=torch.bfloat16)
    down = alloc_pinned_tensor(L * E, H, I, dtype=torch.bfloat16)
    gate_up.copy_(torch.randn(L * E, 2 * I, H) * scale)
    down.copy_(torch.randn(L * E, H, I) * scale)
    return SimpleNamespace(
        quant_format="bf16",
        bank_sources={"gate_up": list(gate_up.split(E)), "down": list(down.split(E))},
        num_layers=L,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
    )


@pytest.mark.parametrize("bs", [1, 2, 5, 16])
def test_cpu_decode_matches_gpu_decode_kernel(bs):
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused import fused_experts_decode_impl

    torch.manual_seed(bs)
    L, E, H, I, top_k = 4, 16, 1024, 512, 4
    layer = 2
    dev = torch.device("cuda")
    cache = _make_cache(L, E, H, I)

    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=0,
        max_tokens=bs,
        device=dev,
    )

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    gate_up_layer = cache.bank_sources["gate_up"][layer].to(dev)
    down_layer = cache.bank_sources["down"][layer].to(dev)
    gpu_out = fused_experts_decode_impl(
        hidden, gate_up_layer, down_layer, w, ids.clone(), "silu", False
    ).float()

    rel = (cpu_out - gpu_out).abs().max() / (gpu_out.abs().max() + 1e-6)
    assert rel < 2e-2, f"bs={bs} rel err {rel.item()}"


def _pack_nvfp4(codes: torch.Tensor) -> torch.Tensor:
    """Pack an [..., IN] uint8 tensor of 4-bit codes to [..., IN//2], low nibble first."""
    lo = codes[..., 0::2]
    hi = codes[..., 1::2]
    return (lo | (hi << 4)).contiguous().to(torch.uint8)


def _make_nvfp4_cache(L, E, H, I, seed=0):
    """Random but valid NVFP4 banks (native ModelOpt schema) for the cpu backend."""
    torch.manual_seed(seed)
    S = L * E

    def rows(OUT, IN):
        codes = torch.randint(0, 16, (S, OUT, IN), dtype=torch.uint8)
        packed = _pack_nvfp4(codes)  # [S, OUT, IN//2]
        # block scales in fp8-e4m3 (1 byte), ~[0.5, 1.5]; row globals in fp16, small.
        scale = (0.5 + torch.rand(S, OUT, IN // 16)).to(torch.float8_e4m3fn)
        glob = (0.02 + 0.01 * torch.rand(S, OUT)).to(torch.float16)
        return packed, scale.contiguous(), glob.contiguous()

    gup, gus, gug = rows(2 * I, H)
    dnp, dns, dng = rows(H, I)
    return SimpleNamespace(
        quant_format="nvfp4",
        bank_sources={
            "gate_up_packed": list(gup.split(E)), "gate_up_scale": list(gus.split(E)),
            "gate_up_global": list(gug.split(E)), "down_packed": list(dnp.split(E)),
            "down_scale": list(dns.split(E)), "down_global": list(dng.split(E)),
        },
        num_layers=L,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
    )


@pytest.mark.parametrize("bs", [1, 3, 8])
def test_cpu_decode_nvfp4_matches_dequant_then_gpu(bs):
    """CPU inline-dequant NVFP4 GEMV vs. canonical dequant_nvfp4 + bf16 GPU decode.

    Both paths read byte-identical quantized banks; the CPU dequantizes inside the
    K-loop (fp32), the reference materializes bf16 weights first, so the only spread
    is weight bf16-rounding + reduction order -> tight relative tolerance.
    """
    from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused import fused_experts_decode_impl

    torch.manual_seed(100 + bs)
    L, E, H, I, top_k = 3, 16, 1024, 512, 4
    layer = 1
    dev = torch.device("cuda")
    cache = _make_nvfp4_cache(L, E, H, I)

    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=0,
        max_tokens=bs,
        device=dev,
    )

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    b = cache.bank_sources
    slots = torch.arange(E, device=dev, dtype=torch.int32)  # layer-local expert rows
    gate_up_layer = dequant_nvfp4(
        b["gate_up_packed"][layer].to(dev), b["gate_up_scale"][layer].to(dev),
        b["gate_up_global"][layer].to(dev), slots, dtype=torch.bfloat16,
    )  # [E, 2I, H]
    down_layer = dequant_nvfp4(
        b["down_packed"][layer].to(dev), b["down_scale"][layer].to(dev),
        b["down_global"][layer].to(dev), slots, dtype=torch.bfloat16,
    )  # [E, H, I]
    gpu_out = fused_experts_decode_impl(
        hidden, gate_up_layer, down_layer, w, ids.clone(), "silu", False
    ).float()

    rel = (cpu_out - gpu_out).abs().max() / (gpu_out.abs().max() + 1e-6)
    assert rel < 3e-2, f"nvfp4 bs={bs} rel err {rel.item()}"


@pytest.mark.parametrize("bs", [1, 4])
def test_cpu_decode_nvfp4_swigluoai_matches_dequant_reference(bs):
    """MiniMax-M3's swigluoai routed experts on the CPU executor (ActKind 3 through
    the generic NVFP4 GEMV epilogue, with the layer's alpha/limit scalars) vs. a
    pure-torch dequant reference."""
    from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    torch.manual_seed(300 + bs)
    L, E, H, I, top_k = 2, 16, 1024, 512, 4
    layer = 1
    alpha, limit = 1.702, 7.0
    dev = torch.device("cuda")
    cache = _make_nvfp4_cache(L, E, H, I, seed=17)

    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="swigluoai",
        apply_router_weight_on_input=False,
        num_threads=0,
        max_tokens=bs,
        device=dev,
        swiglu_alpha=alpha,
        swiglu_limit=limit,
    )

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    b = cache.bank_sources
    slots = torch.arange(E, device=dev, dtype=torch.int32)
    gate_up_layer = dequant_nvfp4(
        b["gate_up_packed"][layer].to(dev), b["gate_up_scale"][layer].to(dev),
        b["gate_up_global"][layer].to(dev), slots, dtype=torch.bfloat16,
    ).float()  # [E, 2I, H]
    down_layer = dequant_nvfp4(
        b["down_packed"][layer].to(dev), b["down_scale"][layer].to(dev),
        b["down_global"][layer].to(dev), slots, dtype=torch.bfloat16,
    ).float()  # [E, H, I]

    ref = torch.zeros(bs, H, device=dev, dtype=torch.float32)
    x = hidden.float()
    for t in range(bs):
        for k in range(top_k):
            e = int(ids[t, k])
            h = gate_up_layer[e] @ x[t]
            gate = h[:I].clamp(max=limit)
            up = h[I:].clamp(-limit, limit)
            act = gate * torch.sigmoid(gate * alpha) * (up + 1.0)
            ref[t] += float(w[t, k]) * (down_layer[e] @ act)

    rel = (cpu_out - ref).abs().max() / (ref.abs().max() + 1e-6)
    assert rel < 3e-2, f"nvfp4 swigluoai bs={bs} rel err {rel.item()}"


def test_stale_extension_rejected_for_swigluoai(monkeypatch):
    """A prebuilt _cpu_moe.so from before ACT_SWIGLUOAI accepts act id 3 without
    error and silently computes the wrong activation in the generic epilogue.
    The executor probes the extension's `max_generic_act_id` marker -- absent on
    stale builds -- and must fail loudly with the rebuild instruction."""
    from freetoken.kernel import _cpu_moe
    from freetoken.moe.cpu_executor import CpuMoeExecutor, compiled_extension_supports

    # raising=False: on a GENUINELY stale extension the attribute is already
    # absent, and this test must still run (it is the test for that case).
    monkeypatch.delattr(_cpu_moe, "max_generic_act_id", raising=False)
    assert not compiled_extension_supports("swigluoai")
    assert compiled_extension_supports("silu")  # silu family predates the marker
    cache = _make_nvfp4_cache(1, 4, 256, 128)
    with pytest.raises(RuntimeError, match="rebuild"):
        CpuMoeExecutor(
            cache, top_k=2, activation="swigluoai",
            apply_router_weight_on_input=False, num_threads=0, max_tokens=1,
            device=torch.device("cuda"), swiglu_alpha=1.702, swiglu_limit=7.0,
        )
    # silu predates the marker and must keep working on a stale build.
    CpuMoeExecutor(
        cache, top_k=2, activation="silu",
        apply_router_weight_on_input=False, num_threads=0, max_tokens=1,
        device=torch.device("cuda"),
    )


def _pack_mxfp4_transposed(codes: torch.Tensor) -> torch.Tensor:
    """Pack [S, K, N] codes -> transposed blocks [S, K//2, N] (low nibble = even K)."""
    lo = codes[:, 0::2, :]
    hi = codes[:, 1::2, :]
    return (lo | (hi << 4)).contiguous().to(torch.uint8)


def _make_mxfp4_cache(L, E, H, I, seed=0):
    """Random gpt-oss ``mxfp4_triton`` banks (transposed split-K layout)."""
    torch.manual_seed(seed)
    S = L * E

    def gu_rows():  # gate_up: K=H, N=2I
        codes = torch.randint(0, 16, (S, H, 2 * I), dtype=torch.uint8)
        blocks = _pack_mxfp4_transposed(codes)  # [S, H//2, 2I]
        scales = torch.randint(122, 128, (S, H // 32, 2 * I), dtype=torch.uint8)
        bias = (0.01 * torch.randn(S, 2 * I)).to(torch.bfloat16)
        return blocks, scales, bias

    def dn_rows():  # down: K=I, N=H
        codes = torch.randint(0, 16, (S, I, H), dtype=torch.uint8)
        blocks = _pack_mxfp4_transposed(codes)  # [S, I//2, H]
        scales = torch.randint(122, 128, (S, I // 32, H), dtype=torch.uint8)
        bias = (0.01 * torch.randn(S, H)).to(torch.bfloat16)
        return blocks, scales, bias

    gub, gus, gob = gu_rows()
    dnb, dns, dob = dn_rows()
    return SimpleNamespace(
        quant_format="mxfp4_triton",
        bank_sources={
            "gate_up_blocks": list(gub.split(E)), "gate_up_scales": list(gus.split(E)),
            "gate_up_bias": list(gob.split(E)), "down_blocks": list(dnb.split(E)),
            "down_scales": list(dns.split(E)), "down_bias": list(dob.split(E)),
        },
        num_layers=L,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
    )


@pytest.mark.parametrize("bs", [1, 4, 8])
def test_cpu_decode_mxfp4_matches_gpu_splitk(bs):
    """CPU mxfp4 N-accumulator GEMV vs. gpt-oss production split-K decode, on
    byte-identical transposed banks (same dequant, clamped swiglu, bias, router wt)."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused_mxfp4 import run_mxfp4_splitk_decode_experts as _run_mxfp4_splitk_decode_experts

    torch.manual_seed(200 + bs)
    L, E, H, I, top_k = 2, 8, 256, 256, 2
    layer = 1
    alpha, limit = 1.702, 7.0
    dev = torch.device("cuda")
    cache = _make_mxfp4_cache(L, E, H, I)

    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="gpt_oss_swiglu",
        apply_router_weight_on_input=False,
        num_threads=0,
        max_tokens=bs,
        device=dev,
        swiglu_alpha=alpha,
        swiglu_limit=limit,
    )

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    b = cache.bank_sources
    gpu_out = _run_mxfp4_splitk_decode_experts(
        hidden, w, ids.clone(),
        b["gate_up_blocks"][layer].to(dev), b["gate_up_scales"][layer].to(dev), b["gate_up_bias"][layer].to(dev),
        b["down_blocks"][layer].to(dev), b["down_scales"][layer].to(dev), b["down_bias"][layer].to(dev),
        top_k=top_k, hidden_act_alpha=alpha, swiglu_limit=limit,
    ).float()

    rel = (cpu_out - gpu_out).abs().max() / (gpu_out.abs().max() + 1e-6)
    assert rel < 4e-2, f"mxfp4 bs={bs} rel err {rel.item()}"


def _make_dsfp4_cache(L, E, H, I, seed=0):
    """Random DeepSeek-V4 ``ds_fp4`` banks (row-major e2m1 + e8m0/32, no global)."""
    torch.manual_seed(seed)
    S = L * E

    def rows(OUT, IN):
        codes = torch.randint(0, 16, (S, OUT, IN), dtype=torch.uint8)
        packed = _pack_nvfp4(codes)  # [S, OUT, IN//2], low nibble = even K
        # e8m0 codes ~ 2^-5..2^0; stored as float8_e8m0fnu (raw byte == code).
        scale = torch.randint(122, 128, (S, OUT, IN // 32), dtype=torch.uint8)
        scale = scale.view(torch.float8_e8m0fnu).contiguous()
        return packed, scale

    gup, gus = rows(2 * I, H)
    dnp, dns = rows(H, I)
    return SimpleNamespace(
        quant_format="ds_fp4",
        bank_sources={
            "gate_up_packed": list(gup.split(E)), "gate_up_scale": list(gus.split(E)),
            "down_packed": list(dnp.split(E)), "down_scale": list(dns.split(E)),
        },
        num_layers=L,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
    )


@pytest.mark.parametrize("bs", [1, 3, 8])
def test_cpu_decode_dsfp4_matches_gpu(bs):
    """CPU ds_fp4 W4A8 GEMV vs. DeepSeek-V4 production ``routed_experts_fp4`` on
    byte-identical banks: same e2m1/e8m0 dequant, FP8-e4m3 activation round-trips
    (block 128), silu-swiglu with clamp, router weight on the down output."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

    torch.manual_seed(300 + bs)
    L, E, H, I, top_k = 2, 8, 256, 256, 4
    layer = 1
    swiglu_limit = 7.0
    dev = torch.device("cuda")
    cache = _make_dsfp4_cache(L, E, H, I)

    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=0,
        max_tokens=bs,
        device=dev,
        swiglu_limit=swiglu_limit,
    )

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    b = cache.bank_sources
    gpu_out = routed_experts_fp4(
        hidden, ids.clone(), w.contiguous(),
        b["gate_up_packed"][layer].to(dev), b["gate_up_scale"][layer].to(dev),
        b["down_packed"][layer].to(dev), b["down_scale"][layer].to(dev),
        swiglu_limit,
    ).float()

    rel = (cpu_out - gpu_out).abs().max() / (gpu_out.abs().max() + 1e-6)
    assert rel < 4e-2, f"ds_fp4 bs={bs} rel err {rel.item()}"


# --------------------------------------------------------------------------------
# Part 2: CUDA-graph capture/replay (cudaLaunchHostFunc submit/sync host nodes).
# --------------------------------------------------------------------------------


def _reference(cache, layer, hidden, ids, w, act="silu", apply_in=False):
    gu = cache.bank_sources["gate_up"][layer].float()
    dn = cache.bank_sources["down"][layer].float()
    I = gu.shape[1] // 2
    H = gu.shape[2]
    bs, top_k = ids.shape
    xf = hidden.float().cpu()
    out = torch.zeros(bs, H, dtype=torch.float32)
    for t in range(bs):
        for k in range(top_k):
            e = int(ids[t, k])
            if e < 0:
                continue
            wi = float(w[t, k]) if apply_in else 1.0
            wo = 1.0 if apply_in else float(w[t, k])
            gate = (gu[e, :I, :] @ xf[t]) * wi
            up = (gu[e, I : 2 * I, :] @ xf[t]) * wi
            g = (Fn.silu(gate) * up).bfloat16().float()
            out[t] += wo * (dn[e] @ g)
    return out


def test_cpu_moe_decode_cuda_graph_replay():
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    torch.manual_seed(0)
    L, E, H, I, top_k = 3, 8, 512, 256, 2
    layer = 1
    bs = 4
    cache = _make_cache(L, E, H, I)

    dev = torch.device("cuda")
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=8,
        max_tokens=bs,
        device=dev,
    )

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
    ids = torch.randint(0, E, (bs, top_k), device=dev, dtype=torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    # Eager forward: also materializes the per-bs pinned buffers + per-(layer,bs) task.
    out_eager = ex.decode(layer, hidden, w, ids)
    torch.cuda.synchronize()
    ref_eager = _reference(cache, layer, hidden, ids.cpu(), w.cpu())
    rel = (out_eager.float().cpu() - ref_eager).abs().max() / ref_eager.abs().max()
    assert rel < 2e-2, f"eager rel err {rel}"

    # Capture.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        out_cap = ex.decode(layer, hidden, w, ids)
    torch.cuda.synchronize()

    # Replay several times with NEW data; the captured host nodes must read the
    # freshly written pinned buffers, not the capture-time values.
    for it in range(3):
        torch.manual_seed(100 + it)
        new_hidden = torch.randn(bs, H, dtype=torch.bfloat16)
        new_ids = torch.randint(0, E, (bs, top_k), dtype=torch.int32)
        new_w = torch.rand(bs, top_k, dtype=torch.float32)
        hidden.copy_(new_hidden)
        ids.copy_(new_ids)
        w.copy_(new_w)
        g.replay()
        torch.cuda.synchronize()
        ref = _reference(cache, layer, new_hidden, new_ids, new_w)
        rel = (out_cap.float().cpu() - ref).abs().max() / ref.abs().max()
        assert rel < 2e-2, f"replay {it} rel err {rel}"

    # When the flag handshake is available (stream memops probed OK), the decodes above
    # must actually have exercised it -- guard against a silent fallback to the
    # host-func path making this test vacuous for the capture-embedded memops.
    if ex._flag_sync:
        assert (layer, bs) in ex._flag_slots, "flag slot expected for the decode task"
        slot = ex._flag_slots[(layer, bs)]
        assert ex._ext.flag_served_count(slot) >= 4, "1 eager + 3 replay dispatches expected"
        assert int(ex._done[slot]) == 1 and int(ex._ready[slot]) == 0, "handshake at rest"
        assert int(ex._err.sum()) == 0, "watchdog must not fire in normal operation"
        ex.raise_if_unhealthy()

    print("cpu moe cuda graph replay OK")


def test_cpu_moe_decode_cuda_graph_replay_mxfp4():
    """gpt-oss mxfp4 path under capture/replay: the host nodes must recompute the
    clamped-swiglu+bias GEMV from the freshly written pinned routing on each replay."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused_mxfp4 import run_mxfp4_splitk_decode_experts as _run_mxfp4_splitk_decode_experts

    torch.manual_seed(7)
    L, E, H, I, top_k = 2, 8, 256, 256, 2
    layer, bs, alpha, limit = 1, 4, 1.702, 7.0
    cache = _make_mxfp4_cache(L, E, H, I)

    dev = torch.device("cuda")
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    ex = CpuMoeExecutor(
        cache, top_k=top_k, activation="gpt_oss_swiglu",
        apply_router_weight_on_input=False, num_threads=8, max_tokens=bs, device=dev,
        swiglu_alpha=alpha, swiglu_limit=limit,
    )
    b = cache.bank_sources
    layer_banks = [b[n][layer].to(dev) for n in (
        "gate_up_blocks", "gate_up_scales", "gate_up_bias",
        "down_blocks", "down_scales", "down_bias")]

    def reference(hidden, ids, w):
        return _run_mxfp4_splitk_decode_experts(
            hidden, w, ids.clone(), *layer_banks,
            top_k=top_k, hidden_act_alpha=alpha, swiglu_limit=limit,
        ).float()

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
    ids = torch.randint(0, E, (bs, top_k), device=dev, dtype=torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    ex.decode(layer, hidden, w, ids)  # eager warmup: materialize buffers + task
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        out_cap = ex.decode(layer, hidden, w, ids)
    torch.cuda.synchronize()

    for it in range(3):
        torch.manual_seed(300 + it)
        hidden.copy_(torch.randn(bs, H, dtype=torch.bfloat16))
        ids.copy_(torch.randint(0, E, (bs, top_k), dtype=torch.int32))
        w.copy_(torch.rand(bs, top_k, dtype=torch.float32))
        g.replay()
        torch.cuda.synchronize()
        ref = reference(hidden, ids, w)
        rel = (out_cap.float() - ref).abs().max() / (ref.abs().max() + 1e-6)
        assert rel < 4e-2, f"mxfp4 replay {it} rel err {rel.item()}"

    print("cpu moe cuda graph replay (mxfp4) OK")


def test_cpu_moe_decode_cuda_graph_replay_dsfp4():
    """DeepSeek-V4 ds_fp4 path under capture/replay: the 4-phase pipeline (input
    FP8 round-trip -> gate_up -> intermediate FP8 round-trip -> down) must recompute
    from the freshly written pinned routing on each replay."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

    torch.manual_seed(11)
    L, E, H, I, top_k = 2, 8, 256, 256, 4
    layer, bs, limit = 1, 4, 7.0
    cache = _make_dsfp4_cache(L, E, H, I)

    dev = torch.device("cuda")
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    ex = CpuMoeExecutor(
        cache, top_k=top_k, activation="silu",
        apply_router_weight_on_input=False, num_threads=8, max_tokens=bs, device=dev,
        swiglu_limit=limit,
    )
    b = cache.bank_sources
    layer_banks = [b[n][layer].to(dev) for n in (
        "gate_up_packed", "gate_up_scale", "down_packed", "down_scale")]

    def reference(hidden, ids, w):
        return routed_experts_fp4(hidden, ids.clone(), w.contiguous(), *layer_banks, limit).float()

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16)
    ids = torch.randint(0, E, (bs, top_k), device=dev, dtype=torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    ex.decode(layer, hidden, w, ids)  # eager warmup: materialize buffers + task
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        out_cap = ex.decode(layer, hidden, w, ids)
    torch.cuda.synchronize()

    for it in range(3):
        torch.manual_seed(500 + it)
        hidden.copy_(torch.randn(bs, H, dtype=torch.bfloat16))
        ids.copy_(torch.randint(0, E, (bs, top_k), dtype=torch.int32))
        w.copy_(torch.rand(bs, top_k, dtype=torch.float32))
        g.replay()
        torch.cuda.synchronize()
        ref = reference(hidden, ids, w)
        rel = (out_cap.float() - ref).abs().max() / (ref.abs().max() + 1e-6)
        assert rel < 4e-2, f"ds_fp4 replay {it} rel err {rel.item()}"

    print("cpu moe cuda graph replay (ds_fp4) OK")


if __name__ == "__main__":
    for bs in (1, 2, 5, 16):
        test_cpu_decode_matches_gpu_decode_kernel(bs)
        print(f"bf16 bs={bs} OK")
    for bs in (1, 3, 8):
        test_cpu_decode_nvfp4_matches_dequant_then_gpu(bs)
        print(f"nvfp4 bs={bs} OK")
    for bs in (1, 4, 8):
        test_cpu_decode_mxfp4_matches_gpu_splitk(bs)
        print(f"mxfp4 bs={bs} OK")
    for bs in (1, 3, 8):
        test_cpu_decode_dsfp4_matches_gpu(bs)
        print(f"ds_fp4 bs={bs} OK")
    test_cpu_moe_decode_cuda_graph_replay()
    test_cpu_moe_decode_cuda_graph_replay_mxfp4()
    test_cpu_moe_decode_cuda_graph_replay_dsfp4()


def test_cpu_moe_executor_is_collectable():
    """The flag watchdog must not pin the executor against GC (it holds a weakref and
    re-derefs per tick): dropping the last reference collects the executor -- with its
    C++ worker pool and pinned banks -- and the watchdog daemon exits by itself."""
    import gc
    import time
    import weakref

    from freetoken.moe.cpu_executor import CpuMoeExecutor

    cache = _make_cache(2, 4, 64, 32)
    ex = CpuMoeExecutor(
        cache, top_k=2, activation="silu", apply_router_weight_on_input=False,
        num_threads=2, max_tokens=1, device=torch.device("cuda"),
    )
    watchdog = ex._watchdog if ex._flag_sync else None
    ref = weakref.ref(ex)
    del ex
    gc.collect()
    assert ref() is None, "executor must be collectable once the last reference drops"
    if watchdog is not None:
        watchdog.join(timeout=5.0)  # exits on the first tick after the weakref dies
        assert not watchdog.is_alive(), "watchdog thread must exit after executor GC"
