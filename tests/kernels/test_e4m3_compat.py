"""e4m3 emulation (pre-sm_89) vs the native fp8 unit.

Three layers of guarantee:

1. The emulation primitives are bit-exact against the hardware fp8 unit (all 254
   non-NaN codes; adversarial rounding inputs covering every grid midpoint, the
   subnormal range and the 2^-6 boundary).
2. Every affected wrapper produces bit-identical output under
   ``FREETOKEN_FORCE_E4M3_EMU=1`` (run in a subprocess with a fresh
   ``TRITON_CACHE_DIR``: the flag is read once at import and is deliberately NOT
   part of triton's cache key, so flipping it requires a fresh process + cache).
   Sole exception: the fp8-MMA -> bf16-MMA GEMMs may differ by fp32 reduction
   order inside the MMA (same e4m3 grid values), compared at the repo's 5e-2
   kernel tolerance.
3. A cross-arch compile gate: with the driver target AND the host-visible device
   capability patched to sm_80/86/89/120 and every triton launch forced into
   warmup (compile-only), the full wrapper->kernel paths -- uint8 views, PDL
   gates, e4m3 branches -- must compile for the foreign arch.
4. A cache-key guard: the @constexpr_function probes must not reference plain
   python functions. triton's AST walk rejects that, and it disables every kernel
   that branches on e4m3 native-ness at once, so the failure is not local.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

FP8 = torch.float8_e4m3fn
# fp8-MMA vs bf16-MMA GEMMs: identical grid values, but the MMA-internal fp32
# reduction order may differ (dsv4_gemm / moe_prefill_fp8 happen to be bit-exact
# on H100 -- that is lowering luck, not a guarantee).
_MMA_TOL_KEYS = {"blk_gemm", "dsv4_gemm", "moe_prefill_fp8"}
_MMA_TOL = dict(rtol=5e-2, atol=0.5)


def _native_cc() -> bool:
    return torch.cuda.get_device_capability() >= (8, 9)


def test_constexpr_probe_never_references_a_host_function():
    """triton hashes an @constexpr_function by walking its AST (runtime/jit.py:
    cache_key -> record_reference), and a bare reference to a plain python function
    raises "Unsupported function referenced: <function e4m3_native ...>". Making
    e4m3_native_cx defer to the host probe did exactly that, and it took out EVERY
    kernel that branches on e4m3 native-ness at once (here: the PLE gather, inside
    CUDA graph capture). Modules (``target_info.cuda_capability_geq``) survive the
    walk, functions do not -- so unifying the two probes has to go the other way: the
    code that owns a buffer follows the buffer (tests/kernels/test_kv_fp8.py)."""
    import inspect

    from freetoken.kernel.triton import e4m3_compat

    lines = inspect.getsource(e4m3_compat).splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("def e4m3_native_cx("))
    body = []
    for line in lines[start + 1:]:
        if line and not line.startswith((" ", "\t")):
            break
        body.append(line)
    assert "e4m3_native(" not in "\n".join(body), (
        "a constexpr_function may not call a host function: triton's cache-key walk "
        "rejects it and every e4m3 kernel stops compiling"
    )


# ======================================================================================
# 1. Primitives vs the native fp8 unit (needs sm_89+ hardware for the reference).
# ======================================================================================
@pytest.mark.skipif(not torch.cuda.is_available() or not _native_cc(),
                    reason="needs native fp8 (sm_89+) as reference")
class TestPrimitives:
    def test_decode_f32_bitexact(self):
        import triton
        import triton.language as tl
        from freetoken.kernel.triton.e4m3_compat import e4m3_u8_to_f32

        @triton.jit
        def k(v_ptr, o_ptr, BLOCK: tl.constexpr):
            offs = tl.arange(0, BLOCK)
            tl.store(o_ptr + offs, e4m3_u8_to_f32(tl.load(v_ptr + offs)))

        v = torch.arange(256, dtype=torch.uint8, device="cuda")
        out = torch.empty(256, dtype=torch.float32, device="cuda")
        k[(1,)](v, out, BLOCK=256)
        ref = v.view(FP8).to(torch.float32)
        ok = ~ref.isnan()
        assert torch.equal(out[ok], ref[ok])
        # NaN codes 0x7F/0xFF decode to +-480 by design (weights never store NaN)
        assert out[~ok].abs().eq(480.0).all()

    def test_decode_f16_x128_bitexact(self):
        import triton
        import triton.language as tl
        from freetoken.kernel.triton.e4m3_compat import e4m3_u8_to_f16_x128

        @triton.jit
        def k(v_ptr, o_ptr, BLOCK: tl.constexpr):
            offs = tl.arange(0, BLOCK)
            tl.store(o_ptr + offs, e4m3_u8_to_f16_x128(tl.load(v_ptr + offs)))

        v = torch.arange(256, dtype=torch.uint8, device="cuda")
        out = torch.empty(256, dtype=torch.float16, device="cuda")
        k[(1,)](v, out, BLOCK=256)
        ref = v.view(FP8).to(torch.float16) * 128.0
        ok = ~ref.isnan()
        assert torch.equal(out[ok], ref[ok])

    def test_round_to_grid_bitexact(self):
        import triton
        import triton.language as tl
        from freetoken.kernel.triton.e4m3_compat import round_e4m3

        @triton.jit
        def k(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            m = offs < N
            tl.store(y_ptr + offs, round_e4m3(tl.load(x_ptr + offs, mask=m, other=0.0)), mask=m)

        torch.manual_seed(0)
        grid = torch.arange(256, dtype=torch.uint8, device="cuda").view(FP8).to(torch.float32)
        grid = torch.sort(grid[~grid.isnan()]).values
        parts = [
            torch.empty(1 << 20, device="cuda").uniform_(-448, 448),
            (torch.empty(1 << 20, device="cuda").uniform_(-14, 9).exp2()
             * torch.where(torch.rand(1 << 20, device="cuda") < 0.5, -1.0, 1.0)),
            grid,                          # every representable value
            (grid[1:] + grid[:-1]) * 0.5,  # every exact tie midpoint
            torch.tensor([0.0, -0.0, 448.0, -448.0, 2**-6, 2**-6 - 2**-10, 2**-6 + 2**-10,
                          2**-9, 2**-10, 3 * 2**-10, 5 * 2**-10], device="cuda"),
        ]
        x = torch.cat(parts).clamp(-448, 448).contiguous()
        y = torch.empty_like(x)
        k[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), BLOCK=1024)
        ref = x.to(FP8).to(torch.float32)
        assert int(((y != ref) & ~(y.isnan() & ref.isnan())).sum()) == 0


# ======================================================================================
# Shared emit path: every affected wrapper, deterministic inputs.
# ======================================================================================
def _emit_all(path: str) -> None:
    from freetoken.kernel.triton.e4m3_compat import e4m3_native
    from freetoken.kernel.triton.dsv4.fp8_linear import (
        act_quant_fp8, act_quant_fp8_inplace, act_quant_fp8_roundtrip,
        block_fp8_linear as dsv4_block_fp8_linear, fp4_act_quant_inplace,
    )
    from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear
    from freetoken.kernel.triton.fp8_block_linear import (
        block_fp8_linear, per_token_group_quant_fp8,
    )
    from freetoken.kernel.triton.fp8_blockscale_moe import (
        fused_experts_decode_fp8_blockscale, fused_experts_fp8_blockscale,
    )
    from freetoken.kernel.triton.nvfp4_linear import (
        nvfp4_dense_linear, nvfp4_dense_linear_t, nvfp4_transpose_resident,
    )
    from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4
    from freetoken.moe.fused_nvfp4 import (
        fused_experts_decode_nvfp4_marlin, fused_experts_decode_nvfp4_serial,
        fused_experts_nvfp4,
    )

    torch.manual_seed(7)
    dev = "cuda"
    f32 = lambda t: t.to(torch.float32).cpu()  # noqa: E731
    out = {"native": e4m3_native()}

    # dsv4: act quant + roundtrips + gemv/gemm
    x = torch.randn(37, 4096, dtype=torch.bfloat16, device=dev) * 3.0
    y, s = act_quant_fp8(x, 128)
    out["dsv4_aq_y"], out["dsv4_aq_s"] = f32(y), s.cpu()
    x2 = torch.randn(33, 1024, dtype=torch.bfloat16, device=dev) * 2.5
    out["dsv4_rt"] = act_quant_fp8_roundtrip(x2.clone(), 128).cpu()
    xi = torch.randn(16, 768, dtype=torch.bfloat16, device=dev)
    out["dsv4_inplace"] = act_quant_fp8_inplace(xi.clone()[:, :512], 64).cpu()
    out["dsv4_fp4"] = fp4_act_quant_inplace(
        torch.randn(32, 128, dtype=torch.bfloat16, device=dev), 32).cpu()
    wq = torch.randn(1024, 4096, device=dev).to(FP8)
    sq = torch.full((8, 32), 129, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
    out["dsv4_gemv"] = f32(dsv4_block_fp8_linear(
        torch.randn(1, 4096, dtype=torch.bfloat16, device=dev), wq, sq))
    out["dsv4_gemm"] = f32(dsv4_block_fp8_linear(
        torch.randn(64, 4096, dtype=torch.bfloat16, device=dev), wq, sq))

    # per-tensor fp8 dense
    w = torch.randn(256, 512, device=dev).to(FP8)
    ws = torch.rand(256, device=dev) + 0.5
    x1 = torch.randn(1, 512, dtype=torch.bfloat16, device=dev)
    xm = torch.randn(64, 512, dtype=torch.bfloat16, device=dev)
    out["pt_gemv"] = f32(fp8_pertensor_linear(x1, w, ws))
    out["pt_gemm"] = f32(fp8_pertensor_linear(xm, w, ws))

    # generic block fp8 dense
    bs = torch.rand(2, 4, dtype=torch.bfloat16, device=dev) + 0.5
    aq_y, aq_s = per_token_group_quant_fp8(
        torch.randn(37, 512, dtype=torch.bfloat16, device=dev) * 2)
    out["blk_aq_y"], out["blk_aq_s"] = f32(aq_y), aq_s.cpu()
    out["blk_gemv"] = f32(block_fp8_linear(x1, w, bs))
    out["blk_gemm"] = f32(block_fp8_linear(xm, w, bs))

    # block-fp8 offload experts
    E, H, inter = 4, 256, 128
    gu = torch.randn(E, 2 * inter, H, device=dev).to(FP8)
    gus = torch.rand(E, (2 * inter) // 128, H // 128, dtype=torch.bfloat16, device=dev) + 0.5
    dn = torch.randn(E, H, inter, device=dev).to(FP8)
    dns = torch.rand(E, H // 128, 1, dtype=torch.bfloat16, device=dev) + 0.5
    hidden = torch.randn(2, H, dtype=torch.bfloat16, device=dev)
    tids = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32, device=dev)
    tw = torch.rand(2, 2, dtype=torch.float32, device=dev)
    out["moe_decode_fp8"] = f32(
        fused_experts_decode_fp8_blockscale(hidden, gu, gus, dn, dns, tw, tids))
    out["moe_prefill_fp8"] = f32(
        fused_experts_fp8_blockscale(hidden, gu, gus, dn, dns, tw, tids, E))

    # nvfp4 dense: row-major + transposed resident, gemv / in-kernel gemm / scratch gemm
    N, K = 128, 256
    pk = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    sc = (torch.rand(N, K // 16, device=dev) * 2 + 0.1).to(FP8)
    g = (torch.rand(N, device=dev) * 0.05 + 0.01).to(torch.float16)
    xa = torch.randn(1, K, dtype=torch.bfloat16, device=dev)
    xb = torch.randn(8, K, dtype=torch.bfloat16, device=dev)
    xc = torch.randn(200, K, dtype=torch.bfloat16, device=dev)
    out["nv_gemv"] = f32(nvfp4_dense_linear(xa, pk, sc, g))
    out["nv_gemm_inkernel"] = f32(nvfp4_dense_linear(xb, pk, sc, g))
    out["nv_gemm_scratch"] = f32(nvfp4_dense_linear(xc, pk, sc, g))
    wt, st = nvfp4_transpose_resident(pk, sc)
    out["nv_t_gemv"] = f32(nvfp4_dense_linear_t(xa, wt, st, g))
    out["nv_t_gemm"] = f32(nvfp4_dense_linear_t(xb, wt, st, g))

    # nvfp4 bank dequant
    pkc = torch.randint(0, 256, (E, 64, 128), dtype=torch.uint8, device=dev)
    scc = (torch.rand(E, 64, 16, device=dev) + 0.1).to(FP8)
    gc = (torch.rand(E, 64, device=dev) * 0.05 + 0.01).to(torch.float16)
    out["nv_dequant"] = f32(dequant_nvfp4(
        pkc, scc, gc, torch.tensor([1, 3], dtype=torch.int32, device=dev)))

    # nvfp4 fused moe: marlin / serial decode + prefill
    S, H4, I4 = 4, 256, 128
    gup = torch.randint(0, 256, (S, 2 * I4, H4 // 2), dtype=torch.uint8, device=dev)
    gus4 = (torch.rand(S, 2 * I4, H4 // 16, device=dev) + 0.1).to(FP8)
    gug = (torch.rand(S, 2 * I4, device=dev) * 0.05 + 0.01).to(torch.float16)
    dnp = torch.randint(0, 256, (S, H4, I4 // 2), dtype=torch.uint8, device=dev)
    dns4 = (torch.rand(S, H4, I4 // 16, device=dev) + 0.1).to(FP8)
    dng = (torch.rand(S, H4, device=dev) * 0.05 + 0.01).to(torch.float16)
    hid4 = torch.randn(2, H4, dtype=torch.bfloat16, device=dev)
    out["nv_moe_marlin"] = f32(fused_experts_decode_nvfp4_marlin(
        hid4, gup, gus4, gug, dnp, dns4, dng, tw, tids))
    out["nv_moe_serial"] = f32(fused_experts_decode_nvfp4_serial(
        hid4, gup, gus4, gug, dnp, dns4, dng, tw, tids))
    out["nv_moe_prefill"] = f32(fused_experts_nvfp4(
        hid4, gup, gus4, gug, dnp, dns4, dng, tw, tids, S))

    # fp8 KV cache: the fused quantize+scatter writer the pools call in store_kv, and the
    # QSA sparse reader that turns codes + row scales back into operands. Codes are plain
    # bytes on every arch (kv_quant.kv_codes_dtype), so comparing them as BYTES is what
    # pins a native run and a forced-EMU run to the very same encoding, and the two
    # QSA runs -- one over codes, one over the same real numbers pre-rounded into bf16 --
    # must produce identical bits: the reader casts back to the query dtype before the
    # dot, so any decode difference lands on the comparison instead of hiding in a
    # tolerance.
    from freetoken.kernel.triton.kv_quant import (
        alloc_codes, codes_to_f32, quantize_kv_to_cache,
    )
    from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention

    slots, kvh, hd, page = 128, 2, 64, 64
    kv_in = (torch.randn(slots, kvh * hd, device=dev, dtype=torch.float32) * 4.0).to(
        torch.bfloat16
    )
    k_codes = alloc_codes((slots, kvh, hd), dev)
    v_codes = alloc_codes((slots, kvh, hd), dev)
    k_sc = torch.zeros((slots, kvh), dtype=torch.float32, device=dev)
    v_sc = torch.zeros_like(k_sc)
    quantize_kv_to_cache(
        k=kv_in,
        v=kv_in.flip(-1).contiguous(),
        out_loc=torch.arange(slots, dtype=torch.int32, device=dev),
        k_cache=k_codes,
        v_cache=v_codes,
        k_scale=k_sc,
        v_scale=v_sc,
    )
    out["kvfp8_codes"] = k_codes.view(torch.uint8).to(torch.int16).cpu()
    out["kvfp8_scale"] = k_sc.cpu()

    pages = slots // page
    kc4, vc4 = k_codes.view(pages, page, kvh, hd), v_codes.view(pages, page, kvh, hd)
    q = torch.randn(2, 2 * kvh, hd, device=dev, dtype=torch.bfloat16)
    sel = (
        torch.arange(2 * page, dtype=torch.int32, device=dev)[None, :]
        .repeat(2, 1)
        .contiguous()
    )
    table = torch.arange(pages, dtype=torch.int32, device=dev)[None, :].contiguous()
    t2r = torch.zeros(2, dtype=torch.int32, device=dev)
    out["kvfp8_qsa"] = f32(qsa_sparse_paged_attention(
        q, kc4, vc4, sel, table, t2r, k_scale=k_sc, v_scale=v_sc))
    # The very same numbers, pre-rounded into a bf16 cache. The reader casts its
    # dequantized operands to the query dtype before tl.dot, so the two runs must end up
    # bit-identical (asserted where launches really execute: tests/kernels/test_qsa_fp8.py
    # -- the compile gate below runs warmup-only, where outputs are never written).
    out["kvfp8_qsa_bf16"] = f32(qsa_sparse_paged_attention(
        q,
        (codes_to_f32(kc4) * k_sc.view(pages, page, kvh, 1)).to(torch.bfloat16),
        (codes_to_f32(vc4) * v_sc.view(pages, page, kvh, 1)).to(torch.bfloat16),
        sel, table, t2r))
    torch.save(out, path)


def _child_env(tmp_path, **extra) -> dict:
    import freetoken

    env = os.environ.copy()
    pkg_root = os.path.dirname(os.path.dirname(freetoken.__file__))
    env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
    env["TRITON_CACHE_DIR"] = str(tmp_path / "triton-cache")
    env.update(extra)
    return env


# ======================================================================================
# 2. Forced-EMU vs native A/B across every wrapper.
# ======================================================================================
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available() or not _native_cc(),
                    reason="needs native fp8 (sm_89+) as reference")
def test_forced_emu_matches_native(tmp_path):
    native_pt = str(tmp_path / "native.pt")
    emu_pt = str(tmp_path / "emu.pt")
    _emit_all(native_pt)
    r = subprocess.run(
        [sys.executable, __file__, "emit", emu_pt],
        env=_child_env(tmp_path, FREETOKEN_FORCE_E4M3_EMU="1"),
        capture_output=True, text=True, timeout=1200,
    )
    assert r.returncode == 0, f"EMU emit failed:\n{r.stdout}\n{r.stderr}"
    a, b = torch.load(native_pt), torch.load(emu_pt)
    assert a["native"] and not b["native"]
    for k in [k for k in a if k != "native"]:
        if k in _MMA_TOL_KEYS:
            torch.testing.assert_close(a[k], b[k], **_MMA_TOL)
        else:
            assert torch.equal(a[k], b[k]), f"{k}: EMU output differs from native"


# ======================================================================================
# 3. Cross-arch compile gate (full wrapper->kernel paths, compile-only).
# ======================================================================================
@pytest.mark.slow
@pytest.mark.parametrize("arch", [80, 86, 89, 120])
def test_compile_gate_foreign_arch(arch, tmp_path):
    r = subprocess.run(
        [sys.executable, __file__, "gate", str(arch)],
        # pin the flag off so a developer's exported FORCE_EMU cannot dead-code
        # the native branch out of the sm_89/120 gates
        env=_child_env(tmp_path, FREETOKEN_FORCE_E4M3_EMU=""),
        capture_output=True, text=True, timeout=1200,
    )
    assert r.returncode == 0, f"sm_{arch} compile gate failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}"


def _gate_main(arch: int) -> None:
    """Patch driver target + host capability to ``arch`` and force every triton
    launch into warmup, then drive the full emit path (compile-only)."""
    from triton.backends.compiler import GPUTarget
    from triton.runtime import driver
    from triton.runtime.jit import JITFunction

    tgt = GPUTarget("cuda", arch, 32)
    driver.active.get_current_target = lambda: tgt
    torch.cuda.get_device_capability = lambda device=None: (arch // 10, arch % 10)

    # flashinfer JIT-compiles its ops for the (patched) capability and then really
    # launches them -- unlike triton there is no warmup hook, so force the triton
    # fallbacks instead. sgl_kernel ships multi-arch fatbins and loads by the real
    # device, so it needs no such treatment.
    import freetoken.kernel.backend as backend

    backend.is_flashinfer_installed = lambda: False

    orig_run = JITFunction.run

    def compile_only_run(self, *args, **kwargs):
        kwargs["warmup"] = True
        return orig_run(self, *args, **kwargs)

    JITFunction.run = compile_only_run
    _emit_all(os.devnull)


if __name__ == "__main__":
    if sys.argv[1] == "emit":
        _emit_all(sys.argv[2])
    elif sys.argv[1] == "gate":
        _gate_main(int(sys.argv[2]))
