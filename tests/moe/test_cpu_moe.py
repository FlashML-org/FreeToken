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
        # block scales in fp8-e4m3 (1 byte), ~[0.5, 1.5]; row globals in fp32, small.
        scale = (0.5 + torch.rand(S, OUT, IN // 16)).to(torch.float8_e4m3fn)
        glob = 0.02 + 0.01 * torch.rand(S, OUT)
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
        layer_formats=[("ds_fp4", {"act_block": 128})] * cache.num_layers,
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


@pytest.fixture(scope="module")
def hip_probe_fault_library(tmp_path_factory):
    import importlib.util
    import os
    from pathlib import Path
    import shutil
    import subprocess

    if not torch.version.hip:
        pytest.skip("requires the HIP sync backend")
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler for HIP fault injection")
    candidates = [Path(os.getenv("ROCM_HOME", "/opt/rocm")), Path("/opt/rocm")]
    spec = importlib.util.find_spec("_rocm_sdk_core")
    if spec and spec.submodule_search_locations:
        candidates.append(Path(next(iter(spec.submodule_search_locations))))
    include = next((root / "include" for root in candidates if (root / "include/hip/hip_runtime.h").exists()), None)
    if include is None:
        pytest.skip("requires HIP SDK headers for fault injection")
    library = tmp_path_factory.mktemp("hip_probe") / "faults.so"
    command = [
        compiler, "-std=c++20", "-shared", "-fPIC", "-D__HIP_PLATFORM_AMD__=1",
        f"-I{include}", str(Path(__file__).with_name("hip_probe_faults.cpp")),
        "-ldl", "-o", str(library),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return library


@pytest.mark.parametrize("fault", ["dependency", "end-capture", "pending"])
def test_cpu_moe_hip_probe_error_recovery(hip_probe_fault_library, fault):
    """Rejected probes leave eager usable and preserve errors predating the probe."""
    import os
    from pathlib import Path
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import ctypes
        import os
        import runpy
        import sys

        import pytest
        import torch
        from freetoken.kernel import _cpu_moe
        from freetoken.moe.cpu_executor import CpuMoeExecutor

        shim = ctypes.CDLL(os.environ["LD_PRELOAD"])
        stream_ptr = torch.cuda.current_stream().cuda_stream
        fault = os.environ["FREETOKEN_TEST_HIP_PROBE_FAULT"]
        if fault == "pending":
            assert shim.probe_test_make_error() != 0
            pending = shim.probe_test_peek_error()
            with pytest.raises(RuntimeError, match="pending error before probe"):
                _cpu_moe.memops_probe(stream_ptr)
            assert shim.probe_test_peek_error() == pending
            assert shim.probe_test_begins() == 0
            sys.exit(0)

        helpers = runpy.run_path(sys.argv[1])
        torch.manual_seed(17)
        cache = helpers["_make_cache"](2, 8, 256, 128, scale=0.03)
        ex = CpuMoeExecutor(
            cache, top_k=4, activation="silu", apply_router_weight_on_input=False,
            num_threads=4, max_tokens=3, device=torch.device("cuda"),
        )
        assert not ex._flag_sync and not ex._graph_capture_safe
        assert shim.probe_test_injected() > 0
        assert shim.probe_test_begins() == shim.probe_test_ends()
        assert shim.probe_test_peek_error() == 0
        if fault == "dependency":
            assert shim.probe_test_removed_nodes() == 1
            assert shim.probe_test_last_end_rc() == 0
        else:
            assert shim.probe_test_last_end_rc() != 0

        for index in range(2):
            torch.manual_seed(23 + index)
            hidden = (torch.randn(3, 256) * 0.3).bfloat16().cuda()
            ids = torch.randint(0, 8, (3, 4), dtype=torch.int32).cuda()
            weights = torch.rand(3, 4).cuda()
            actual = ex.decode(1, hidden, weights, ids).float().cpu()
            expected = helpers["_reference"](cache, 1, hidden, ids.cpu(), weights.cpu())
            relative = (actual - expected).abs().max() / (expected.abs().max() + 1e-6)
            assert relative < 2e-2, relative

        stream = torch.cuda.Stream()
        with pytest.raises(RuntimeError, match="--cuda-graph-max-bs 0"):
            with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
                ex.decode(1, hidden, weights, ids)
        assert torch.isfinite(ex.decode(1, hidden, weights, ids).cpu()).all()
    """)
    env = dict(os.environ, LD_PRELOAD=str(hip_probe_fault_library),
               FREETOKEN_TEST_HIP_PROBE_FAULT=fault, FREETOKEN_CPU_MOE_FLAG_SYNC="1")
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve())],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(torch.version.hip is None, reason="requires ROCm")
@pytest.mark.parametrize("fallback", ["handshake-probe", "graph-data"])
def test_cpu_moe_hip_unsafe_graph_keeps_eager_usable(monkeypatch, fallback):
    from freetoken.kernel import _cpu_moe
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    if fallback == "handshake-probe":
        monkeypatch.setattr(_cpu_moe, "memops_probe", lambda stream: False)
    elif fallback == "graph-data":
        original = CpuMoeExecutor.decode_sync

        def corrupt_capture(self, *args, **kwargs):
            out = original(self, *args, **kwargs)
            return out + 1 if torch.cuda.is_current_stream_capturing() else out

        monkeypatch.setattr(CpuMoeExecutor, "decode_sync", corrupt_capture)

    cache = _make_cache(3, 8, 256, 128, scale=0.03)
    ex = CpuMoeExecutor(
        cache, top_k=4, activation="silu", apply_router_weight_on_input=False,
        num_threads=4, max_tokens=1, device=torch.device("cuda"),
    )
    dtype = torch.bfloat16
    assert not ex.graph_capture_safe(dtype)
    if fallback == "graph-data":
        assert ex._flag_sync and not ex._kernel_io
    hidden = torch.randn(1, 256, device="cuda", dtype=torch.bfloat16).to(dtype)
    ids = torch.tensor([[0, 2, -1, 5]], device="cuda", dtype=torch.int32)
    weights = torch.tensor([[0.2, 0.3, 0.1, 0.4]], device="cuda")
    actual = ex.decode(1, hidden, weights, ids)
    expected = _reference(cache, 1, hidden, ids.cpu(), weights.cpu())
    assert actual.dtype == dtype
    relative = (actual.float().cpu() - expected).abs().max() / (expected.abs().max() + 1e-6)
    assert relative < 2e-2, relative
    with pytest.raises(RuntimeError, match="--cuda-graph-max-bs 0"):
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=torch.cuda.Stream()):
            ex.decode(1, hidden, weights, ids)
    assert torch.isfinite(ex.decode(1, hidden, weights, ids).cpu()).all()


@pytest.mark.skipif(torch.version.hip is None, reason="requires ROCm")
def test_cpu_moe_hip_event_fallback_when_kernel_io_unavailable(monkeypatch):
    from freetoken.kernel import _cpu_moe
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    monkeypatch.setattr(_cpu_moe, "kernel_io_probe", lambda stream: "mapped IO unavailable")
    cache = _make_cache(3, 8, 256, 128, scale=0.03)
    ex = CpuMoeExecutor(
        cache, top_k=4, activation="silu", apply_router_weight_on_input=False,
        num_threads=4, max_tokens=1, device=torch.device("cuda"),
    )
    assert ex._flag_sync and not ex._kernel_io
    assert ex.graph_capture_safe(torch.bfloat16)


@pytest.mark.skipif(torch.version.hip is None, reason="requires ROCm")
@pytest.mark.parametrize("graph_bs,max_bs", [(None, 0), ([], 4), (None, 1), ([1], 0), (None, None), (None, -1)])
def test_cpu_moe_hip_kernel_graph_mismatch_uses_event(monkeypatch, caplog, graph_bs, max_bs):
    from freetoken.engine.engine import Engine, logger
    from freetoken.engine.graph import _determine_cuda_graph_bs
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    torch.manual_seed(0)
    graph_enabled = bool(_determine_cuda_graph_bs(graph_bs, max_bs, free_memory=1 << 30))
    monkeypatch.setattr(logger, "propagate", True)
    original = CpuMoeExecutor.decode_sync

    def corrupt_kernel_capture(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        if self._kernel_io and torch.cuda.is_current_stream_capturing():
            return out + 1
        return out

    monkeypatch.setattr(CpuMoeExecutor, "decode_sync", corrupt_kernel_capture)
    cache = _make_cache(3, 8, 256, 128, scale=0.03)
    cache.set_cpu_executor = lambda executor: setattr(cache, "cpu_executor", executor)
    config = SimpleNamespace(
        max_running_req=1, decode_rows_per_req=1, moe_cpu_threads=4,
        cuda_graph_bs=graph_bs, cuda_graph_max_bs=max_bs, tp_info=SimpleNamespace(size=1),
    )
    layers = [SimpleNamespace(
        layer_id=i, top_k=4, activation="silu", apply_router_weight_on_input=False,
        alpha=1.702, limit=None, quant_method=SimpleNamespace(cpu_spec=lambda: ("bf16", {})),
    ) for i in range(3)]
    engine = SimpleNamespace(device=torch.device("cuda"), dtype=torch.bfloat16, tp_cpu_group=None)
    Engine._init_cpu_moe_executor(engine, config, cache, layers)
    ex = engine.cpu_moe_executor
    assert "not graph-safe" not in caplog.text
    assert bool(_determine_cuda_graph_bs(config.cuda_graph_bs, config.cuda_graph_max_bs, free_memory=1 << 30)) == graph_enabled
    assert ex._flag_sync and ex._kernel_io == (not graph_enabled)
    assert ex.graph_capture_safe(torch.bfloat16) == graph_enabled
    hidden = torch.randn(1, 256, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[0, 2, -1, 5]], device="cuda", dtype=torch.int32)
    weights = torch.tensor([[0.2, 0.3, 0.1, 0.4]], device="cuda")
    if not graph_enabled:
        actual = ex.decode(1, hidden, weights, ids)
        expected = _reference(cache, 1, hidden, ids.cpu(), weights.cpu())
        relative = (actual.float().cpu() - expected).abs().max() / (expected.abs().max() + 1e-6)
        assert relative < 2e-2, relative
        return
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = ex.decode(1, hidden, weights, ids)
    for step in range(3):
        hidden.add_(0.125)
        graph.replay()
        torch.cuda.synchronize()
        expected = _reference(cache, 1, hidden, ids.cpu(), weights.cpu())
        relative = (actual.float().cpu() - expected).abs().max() / (expected.abs().max() + 1e-6)
        assert relative < 2e-2, relative


@pytest.mark.skipif(torch.version.hip is None, reason="requires ROCm")
@pytest.mark.parametrize("kernel_only", [False, True])
@pytest.mark.parametrize("graph_enabled", [False, True])
def test_cpu_moe_hip_probe_rejects_common_eager_graph_error(monkeypatch, kernel_only, graph_enabled):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    original = CpuMoeExecutor.decode_sync

    def corrupt(self, pending, add_to=None):
        out = original(self, pending, add_to=add_to)
        return out + 1 if not kernel_only or pending[2] else out

    monkeypatch.setattr(CpuMoeExecutor, "decode_sync", corrupt)
    cache = _make_cache(3, 8, 256, 128, scale=0.03)

    def make():
        return CpuMoeExecutor(cache, top_k=4, activation="silu", apply_router_weight_on_input=False,
                              num_threads=4, max_tokens=2, device=torch.device("cuda"), graph_enabled=graph_enabled)

    if kernel_only:
        ex = make()
        assert not ex._kernel_io
        assert ex.graph_capture_safe(torch.bfloat16) == graph_enabled
    else:
        with pytest.raises(RuntimeError, match="eager IO probe failed"):
            make()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("accumulate", [False, True])
def test_cpu_moe_graph_dtype_and_hybrid_accumulation(dtype, accumulate):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    cache = _make_cache(3, 8, 256, 128, scale=0.03)
    ex = CpuMoeExecutor(cache, top_k=4, activation="silu", apply_router_weight_on_input=False,
                       num_threads=4, max_tokens=2, device=torch.device("cuda"), activation_dtype=dtype)
    assert ex.graph_capture_safe(dtype)
    hidden = torch.randn(2, 256, device="cuda", dtype=dtype)
    ids = torch.tensor([[0, 2, -1, 5], [-1, -1, -1, -1]], device="cuda", dtype=torch.int32)
    weights = torch.rand(2, 4, device="cuda")
    add = torch.randn_like(hidden)

    def forward():
        pending = ex.decode_submit(1, hidden, weights, ids)
        assert pending[2] == (ex._kernel_io and dtype == torch.bfloat16)
        return ex.decode_sync(pending, add_to=add.clone() if accumulate else None)

    forward()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = forward()
    for step in range(6):
        hidden.add_(0.125)
        ids.fill_(-1) if step % 2 else ids[0].copy_(torch.tensor([1, 3, -1, 7], device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        ex._io["x"][:2].copy_(hidden.cpu())
        ex._io["ids"][:2].copy_(ids.cpu())
        ex._io["w"][:2].copy_(weights.cpu())
        ex._ext.run(1, 2)
        expected = ex._io["y"][:2].to(dtype).clone()
        if accumulate:
            expected += add.cpu()
        assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("kernel_io", [False, True])
def test_cpu_moe_hip_event_record_failure_is_loud(hip_probe_fault_library, kernel_io):
    import os
    from pathlib import Path
    import subprocess
    import sys

    script = r"""
import ctypes, os, runpy, sys
import pytest
import torch
from freetoken.kernel import _cpu_moe
from freetoken.moe.cpu_executor import CpuMoeExecutor
helpers = runpy.run_path(sys.argv[1])
shim = ctypes.CDLL(sys.argv[2])
kernel_io = sys.argv[3] == 'True'
if not kernel_io:
    _cpu_moe.kernel_io_probe = lambda stream: 'forced memcpy fallback'
cache = helpers['_make_cache'](3, 8, 256, 128, scale=0.03)
def make_executor():
    return CpuMoeExecutor(cache, top_k=4, activation='silu', apply_router_weight_on_input=False,
        num_threads=4, max_tokens=1, device=torch.device('cuda'))
if kernel_io:
    ex = make_executor()
    assert ex._kernel_io and ex.graph_capture_safe(torch.bfloat16)
    hidden = torch.randn(1, 256, device='cuda', dtype=torch.bfloat16)
    ids = torch.tensor([[0, 2, -1, 5]], device='cuda', dtype=torch.int32)
    weights = torch.tensor([[0.2, 0.3, 0.1, 0.4]], device='cuda')
    ex.decode(1, hidden, weights, ids)
    torch.cuda.synchronize()
    assert shim.probe_test_injected() == 0
    with pytest.raises(RuntimeError, match='publish input copies'):
        ex.decode(1, hidden.half(), weights, ids)
else:
    with pytest.raises(RuntimeError, match='publish input copies'):
        make_executor()
torch.cuda.synchronize()
assert shim.probe_test_injected() > 0
"""
    env = dict(os.environ, LD_PRELOAD=str(hip_probe_fault_library),
               FREETOKEN_TEST_HIP_PROBE_FAULT="publication-event", FREETOKEN_CPU_MOE_FLAG_SYNC="1")
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve()), str(hip_probe_fault_library), str(kernel_io)],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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

    # Eager forward.
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
        assert ex._ext.flag_served() >= 4, "1 eager + 3 replay dispatches expected"
        assert not ex._ext.flag_pending(), "handshake at rest"
        assert not ex._unhealthy, "watchdog must not fire in normal operation"
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
        swiglu_limit=limit, layer_formats=[("ds_fp4", {"act_block": 128})] * cache.num_layers,
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
