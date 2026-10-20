"""Numerical tests for the NVFP4 MoE backends (Triton inline-dequant / Marlin / b12x).

Each verifies the full chain ``native banks -> (in-place repack) -> offload cache slot
gather -> fused forward`` against a pure-torch dequant reference, for both regimes (decode
routes slot ids into the full cache; full-layer prefill routes raw expert ids into the
materialized ``[:E]`` view or the overlap double-buffer views).

Coverage by hardware:
  - Triton (any CUDA GPU): prefill + the production fast decode GEMV, plus a fast-vs-
    baseline-kernel equality guard. This is the path used on sm_120 + CUDA 12.x.
  - Marlin (sm_80..sm_99, e.g. H100): prefill + decode + overlap.
  - b12x (sm_120 + CUDA>=13): pack + fused decode, skipped where the kernel cannot run.
  - Gluon (sm_120): the prefill GEMMs it takes over from the Triton kernel, against that kernel, and the W4A4 prefill
    against a torch emulation of its NVFP4 activation rounding.
Banks are built the way the loader builds them: the bound kernel object packs native pieces.
"""

from __future__ import annotations

import importlib.util
import math

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
# vllm (marlin W4A16 path) is intentionally not co-installable with the core transformers
# pin; it lives in a dedicated venv. Skip rather than fail.
marlin = pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="needs vllm (marlin path)",
)

L, E, S = 2, 8, 8  # layers, experts/layer, cache slots
H, I = 256, 128  # hidden, moe intermediate
TOPK = 2

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def _dequant_ref(packed: torch.Tensor, scale: torch.Tensor, row_global: torch.Tensor) -> torch.Tensor:
    """[N, K//2] u8 + [N, K//16] e4m3 + [N] global -> [N, K] fp32 (low nibble first)."""
    n, k2 = packed.shape
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).view(n, 2 * k2).long()
    w = _E2M1.to(packed.device)[codes]
    s = scale.float().repeat_interleave(16, dim=1)
    return w * s * row_global.float().unsqueeze(1)


def _make_native_sources(device: torch.device, seed: int = 0) -> dict[str, list[torch.Tensor]]:
    """Random ModelOpt-style banks, CPU pinned, with one expert whose w1/w3 globals differ.

    One flat ``[L*E, ...]`` RNG draw (so seeding is unaffected) split into L
    per-layer views.
    """
    g = torch.Generator().manual_seed(seed)
    total = L * E

    def rand_u8(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)

    def rand_scale(*shape):
        return (torch.rand(*shape, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn)

    gate_up_global = torch.full((total, 2 * I), 1.0, dtype=torch.float32)
    gate_up_global[:, I:] = 0.5  # w3 global != w1 global: exercises the alpha fold
    down_global = torch.full((total, H), 0.75, dtype=torch.float32)
    flat = {
        "gate_up_packed": rand_u8(total, 2 * I, H // 2),
        "gate_up_scale": rand_scale(total, 2 * I, H // 16),
        "gate_up_global": gate_up_global,
        "down_packed": rand_u8(total, H, I // 2),
        "down_scale": rand_scale(total, H, I // 16),
        "down_global": down_global,
    }
    return {name: list(t.pin_memory().split(E)) for name, t in flat.items()}


def _assert_close(out: torch.Tensor, ref: torch.Tensor) -> None:
    """bf16 grouped GEMMs round the (large) gate_up intermediates to bf16, so the
    achievable accuracy is relative to the output magnitude, not absolute."""
    tol = 0.03 * float(ref.abs().max())
    torch.testing.assert_close(out.float(), ref, rtol=3e-2, atol=tol)


def _swigluoai_ref(h: torch.Tensor, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    """MiniMax-M3 / gpt-oss clamped swiglu over UNINTERLEAVED [gate; up] halves."""
    gate = h[:I].clamp(max=limit)
    up = h[I:].clamp(-limit, limit)
    return gate * torch.sigmoid(gate * alpha) * (up + 1.0)


def _ref_moe(sources, layer_id, hidden, topk_weights, topk_ids, activation="silu") -> torch.Tensor:
    """Dequant + dense per-token reference for the gated MoE (silu or swigluoai)."""
    out = torch.zeros(hidden.shape, dtype=torch.float32, device=hidden.device)
    x = hidden.float()
    for t in range(hidden.size(0)):
        for j in range(topk_ids.size(1)):
            e = int(topk_ids[t, j])
            gu = _dequant_ref(
                sources["gate_up_packed"][layer_id][e].to(hidden.device),
                sources["gate_up_scale"][layer_id][e].to(hidden.device),
                sources["gate_up_global"][layer_id][e].to(hidden.device),
            )
            dn = _dequant_ref(
                sources["down_packed"][layer_id][e].to(hidden.device),
                sources["down_scale"][layer_id][e].to(hidden.device),
                sources["down_global"][layer_id][e].to(hidden.device),
            )
            h = gu @ x[t]
            if activation == "swigluoai":
                act = _swigluoai_ref(h)
            else:
                act = torch.nn.functional.silu(h[:I]) * h[I:]
            out[t] += float(topk_weights[t, j]) * (dn @ act)
    return out


def _bound_layer(kernel: str):
    """An NVFP4 offload layer with ``kernel`` bound, as the engine binds it."""
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.layers.quantization import QuantBackend, QuantConfig, set_quant_backend

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    set_quant_backend(QuantBackend.parse(f"moe.nvfp4={kernel}"))
    quant = QuantConfig.from_hf({"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ["lm_head"]}})
    return OffloadMoELayer(0, E, TOPK, H, I, quant_config=quant, prefix="model.layers.0.mlp.experts")


def _pieces(sources):
    for layer_id in range(L):
        yield layer_id, 0, E, {
            "gate_up": sources["gate_up_packed"][layer_id],
            "gate_up_scale": sources["gate_up_scale"][layer_id],
            "gate_up_global": sources["gate_up_global"][layer_id],
            "down": sources["down_packed"][layer_id],
            "down_scale": sources["down_scale"][layer_id],
            "down_global": sources["down_global"][layer_id],
        }


def _packed_cache(device, kernel: str, *, seed=0, cache_size=S, prefill_overlap=False):
    """Native sources packed by ``kernel`` into an offload cache; the sources stay the dequant reference."""
    from freetoken.moe.expert_banks import build_expert_banks
    from freetoken.moe.offload_cache import OffloadMoeCache

    sources = _make_native_sources(device, seed=seed)
    layer = _bound_layer(kernel)
    banks = build_expert_banks(layer.quant_method, L, _pieces(sources), device=device)
    cache = OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        cache_size=cache_size,
        device=device,
        quant_format=banks.quant_format,
        prefill_overlap=prefill_overlap,
        layout=banks.layout,
        max_slots=layer.quant_method.slot_limit(),
    )
    cache.set_bank_sources(banks.sources)
    cache.set_alphas(banks.gate_up_alpha, banks.down_alpha)
    cache.reset()
    return cache, sources


def _marlin_cache(device, *, cache_size=S, prefill_overlap=False):
    return _packed_cache(device, "marlin", cache_size=cache_size, prefill_overlap=prefill_overlap)


@cuda
@marlin
def test_marlin_prefill_matches_dequant_reference():
    from freetoken.layers.quantization.moe.nvfp4 import marlin_fused_experts

    device = torch.device("cuda")
    cache, ref_sources = _marlin_cache(device)
    torch.manual_seed(1)
    M = 16
    hidden = torch.randn(M, H, dtype=torch.bfloat16, device=device) / 4
    topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=device)
    topk_weights = torch.rand(M, TOPK, dtype=torch.float32, device=device)

    ref = _ref_moe(ref_sources, 0, hidden, topk_weights, topk_ids)

    # Synchronous full-layer prefill: slot == expert id, raw routing ids pass through.
    cache.materialize_layer(0)
    cache.copy_missing()
    g1, g2 = cache.alphas_for_layer(0)
    gu_p, gu_s, dn_p, dn_s = cache.stage_views(0, 0)
    out = marlin_fused_experts(
        hidden, gu_p, gu_s, g1, dn_p, dn_s, g2,
        topk_weights, topk_ids, "silu", False,
    )
    _assert_close(out, ref)


@cuda
@marlin
def test_marlin_decode_matches_dequant_reference_after_prefill_stomp():
    """Decode through the slot cache, including the request-B-after-request-A pattern
    that B1 guarded against: a layer-1 full-layer prefill between two layer-0 decodes."""
    from freetoken.layers.quantization.moe.nvfp4 import marlin_fused_experts

    device = torch.device("cuda")
    cache, ref_sources = _marlin_cache(device)
    torch.manual_seed(2)
    hidden = torch.randn(1, H, dtype=torch.bfloat16, device=device) / 4
    topk_weights = torch.rand(1, TOPK, dtype=torch.float32, device=device)

    def decode(layer_id, experts):
        ids = torch.tensor([experts], dtype=torch.int32, device=device)
        ref = _ref_moe(ref_sources, layer_id, hidden, topk_weights, ids)
        cache.ensure_experts(layer_id, ids)  # rewrites ids -> slots in place
        cache.copy_missing()
        g1, g2 = cache.alphas_for_slots(layer_id)
        gu_p, gu_s, dn_p, dn_s = cache.pool_bank_views(layer_id)
        out = marlin_fused_experts(
            hidden, gu_p, gu_s, g1, dn_p, dn_s, g2,
            topk_weights, ids, "silu", False,
        )
        _assert_close(out, ref)

    decode(0, [3, 5])
    cache.materialize_layer(1)  # full-layer prefill overwrites every slot (S == E)
    cache.copy_missing()
    decode(0, [3, 5])  # must miss + reload, not serve layer-1 bytes
    decode(1, [1, 2])  # pure hits on the prefilled layer


@cuda
@marlin
def test_marlin_overlap_prefill_matches_dequant_reference():
    """prefill_overlap=True over NVFP4 banks: every layer streams through the generic
    double buffer (full-layer views, routing ids unmapped), and a decode afterwards is
    still correct -- the prefetch invalidated the bookkeeping of the stomped slots.

    The decode-after check is armed by claiming layer-0 slots *before* the prefill
    (cache_size == 2E, so every slot is buffer-backed): if the prefetch failed to
    invalidate them, the post-prefill decode would "hit" stale mappings and read other
    experts' bytes; we assert it misses both experts instead."""
    from freetoken.layers.quantization.moe.nvfp4 import marlin_fused_experts

    device = torch.device("cuda")
    cache, ref_sources = _marlin_cache(device, cache_size=2 * E, prefill_overlap=True)
    torch.manual_seed(4)
    M = 16
    hidden = torch.randn(M, H, dtype=torch.bfloat16, device=device) / 4
    topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=device)
    topk_weights = torch.rand(M, TOPK, dtype=torch.float32, device=device)

    warm_ids = torch.tensor([[3, 5]], dtype=torch.int32, device=device)
    cache.ensure_experts(0, warm_ids)
    cache.copy_missing()

    cache.begin_prefill()
    for layer_id in range(L):
        cache.prefetch_prefill_layer(layer_id)
        cache.prefetch_prefill_layer(layer_id + 1)
        gu_p, gu_s, dn_p, dn_s = cache.wait_prefill_layer(layer_id)
        g1, g2 = cache.alphas_for_layer(layer_id)
        ref = _ref_moe(ref_sources, layer_id, hidden, topk_weights, topk_ids)
        out = marlin_fused_experts(
            hidden, gu_p, gu_s, g1, dn_p, dn_s, g2,
            topk_weights, topk_ids, "silu", False,
        )
        _assert_close(out, ref)
        cache.release_prefill_layer(layer_id)

    # Decode the pre-claimed experts after the buffers stomped the whole cache: their
    # old slot mappings must be gone (forced miss + reload), not "hit" stale entries
    # now holding other layers' prefill bytes.
    dec_hidden = torch.randn(1, H, dtype=torch.bfloat16, device=device) / 4
    dec_weights = torch.rand(1, TOPK, dtype=torch.float32, device=device)
    ids = torch.tensor([[3, 5]], dtype=torch.int32, device=device)
    ref = _ref_moe(ref_sources, 0, dec_hidden, dec_weights, ids)
    cache.ensure_experts(0, ids)
    assert int(cache.num_indices.item()) == 2, "stale slot mappings survived the prefetch"
    cache.copy_missing()
    g1, g2 = cache.alphas_for_slots(0)
    gu_p, gu_s, dn_p, dn_s = cache.pool_bank_views(0)
    out = marlin_fused_experts(
        dec_hidden, gu_p, gu_s, g1, dn_p, dn_s, g2,
        dec_weights, ids, "silu", False,
    )
    _assert_close(out, ref)


@cuda
def test_triton_overlap_prefill_matches_dequant_reference():
    """The 6-bank native layout through the same generic double buffer, consumed by
    the Triton inline-dequant grouped GEMM with unmapped routing ids (n = E)."""
    from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4
    from freetoken.moe.offload_cache import OffloadMoeCache

    device = torch.device("cuda")
    sources = _make_native_sources(device, seed=5)
    cache = OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        cache_size=2 * E,
        device=device,
        quant_format="nvfp4",
        prefill_overlap=True,
    )
    cache.set_bank_sources({name: sources[name] for name in cache.bank_schema})
    cache.reset()
    torch.manual_seed(6)
    M = 8
    hidden = torch.randn(M, H, dtype=torch.bfloat16, device=device) / 4
    topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=device)
    topk_weights = torch.rand(M, TOPK, dtype=torch.float32, device=device)

    cache.begin_prefill()
    for layer_id in range(L):
        cache.prefetch_prefill_layer(layer_id)
        cache.prefetch_prefill_layer(layer_id + 1)
        gu_p, gu_s, gu_g, dn_p, dn_s, dn_g = cache.wait_prefill_layer(layer_id)
        ref = _ref_moe(sources, layer_id, hidden, topk_weights, topk_ids)
        out = fused_experts_nvfp4(
            hidden, gu_p, gu_s, gu_g, dn_p, dn_s, dn_g,
            topk_weights, topk_ids, E, "silu", False,
        )
        _assert_close(out, ref)
        cache.release_prefill_layer(layer_id)


@cuda
def test_triton_swigluoai_matches_dequant_reference():
    """MiniMax-M3's swigluoai routed experts through the Triton prefill grouped GEMM
    and the marlin-style decode GEMV: same banks, the clamped (up+1) swiglu instead
    of silu, alpha/limit threaded through the fused entry points."""
    from freetoken.moe.fused_nvfp4 import (
        fused_experts_decode_nvfp4_marlin,
        fused_experts_nvfp4,
    )

    device = torch.device("cuda")
    sources = _make_native_sources(device, seed=11)
    torch.manual_seed(12)
    M = 8
    hidden = torch.randn(M, H, dtype=torch.bfloat16, device=device) / 4
    topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=device)
    topk_weights = torch.rand(M, TOPK, dtype=torch.float32, device=device)
    layer_id = 0
    banks = [
        sources[name][layer_id].to(device)
        for name in (
            "gate_up_packed", "gate_up_scale", "gate_up_global",
            "down_packed", "down_scale", "down_global",
        )
    ]
    ref = _ref_moe(sources, layer_id, hidden, topk_weights, topk_ids, activation="swigluoai")
    out = fused_experts_nvfp4(
        hidden, *banks, topk_weights, topk_ids, E, "swigluoai", False, 1.702, 7.0
    )
    _assert_close(out, ref)

    dec_hidden = hidden[:1]
    dec_ids = topk_ids[:1]
    dec_weights = topk_weights[:1]
    ref = _ref_moe(sources, layer_id, dec_hidden, dec_weights, dec_ids, activation="swigluoai")
    out = fused_experts_decode_nvfp4_marlin(
        dec_hidden, *banks, dec_weights, dec_ids, "swigluoai", False, 1.702, 7.0
    )
    _assert_close(out, ref)


@cuda
@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0), reason="needs sm_120")
@pytest.mark.parametrize("hidden_size,inter", [(256, 128), (384, 192)])
@pytest.mark.parametrize("block_m", [64, 128])
def test_triton_prefill_routes_to_gluon_on_sm120(monkeypatch, hidden_size, inter, block_m):
    """Both prefill GEMMs go to the Gluon kernel and match the Triton kernel: persistent (BLOCK_SIZE_M 64) and not (128), 256- and 128-row tiles, K % 256 == 0 and not."""
    import freetoken.moe.fused_nvfp4 as fused_nvfp4
    from freetoken.kernel import gluon, nvfp4_moe

    if not gluon.available():
        pytest.skip(f"Gluon is off or needs Triton {'.'.join(map(str, gluon.TRITON_VERSION))}")
    from freetoken.kernel.gluon.nvfp4.moe import sm120

    # the GPU class picks BLOCK_SIZE_M from tokens per expert; pin it so every sm_120 class runs both kernel modes
    config = nvfp4_moe.prefill_config
    monkeypatch.setattr(nvfp4_moe, "prefill_config", lambda *args: {**config(*args), "BLOCK_SIZE_M": block_m})

    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(0)

    def bank(n, k):
        packed = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=device, generator=g)
        scale = (torch.rand(E, n, k // 16, device=device, generator=g) + 0.25).to(torch.float8_e4m3fn)
        return packed, scale, torch.rand(E, n, device=device, generator=g) + 0.5

    banks = (*bank(2 * inter, hidden_size), *bank(hidden_size, inter))
    M = 1024
    hidden = torch.randn(M, hidden_size, dtype=torch.bfloat16, device=device, generator=g) / 4
    topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=device, generator=g)
    topk_weights = torch.rand(M, TOPK, device=device, generator=g)

    calls = []
    gluon_gemm = sm120.prefill_gemm

    def counted(*args):
        calls.append(args)
        return gluon_gemm(*args)

    monkeypatch.setattr(sm120, "prefill_gemm", counted)
    out = fused_nvfp4.fused_experts_nvfp4(hidden, *banks, topk_weights, topk_ids, E)
    assert len(calls) == 2
    monkeypatch.setattr(nvfp4_moe, "_gluon_prefill", lambda device_index: None)
    ref = fused_nvfp4.fused_experts_nvfp4(hidden, *banks, topk_weights, topk_ids, E)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2 * float(ref.abs().max()))


def _round_e2m1(v: torch.Tensor) -> torch.Tensor:
    """Round to the nearest e2m1 value, ties to the even code, saturating at 6 (cvt.rn.satfinite)."""
    grid = _E2M1[:8].to(v.device)
    a = v.abs().clamp(max=6.0)
    hi = torch.searchsorted(grid, a.contiguous()).clamp(max=7)
    lo = (hi - 1).clamp(min=0)
    d_lo, d_hi = a - grid[lo], grid[hi] - a
    code = torch.where(d_lo < d_hi, lo, torch.where(d_hi < d_lo, hi, torch.where(lo % 2 == 0, lo, hi)))
    return torch.copysign(grid[code], v)


def _nvfp4_qdq(x: torch.Tensor, g: float) -> torch.Tensor:
    """x [R, K] through NVFP4 under the static global g, back to fp32."""
    xb = x.float().view(x.shape[0], -1, 16)
    # fp32 scalar math like the kernels: a one-ulp difference in 6 * g can flip an e4m3 block scale
    g = torch.tensor(g, dtype=torch.float32, device=x.device)
    sf = ((xb.abs().amax(-1) / (6.0 * g)).to(torch.float8_e4m3fn).float() * g).unsqueeze(-1)
    inv = torch.where(sf > 0, 1.0 / sf, torch.zeros_like(sf))
    return (_round_e2m1(xb * inv) * sf).view_as(x)


def _gated_ref(h: torch.Tensor, activation: str, inter: int, alpha: float, limit: float) -> torch.Tensor:
    gate, up = h[:, :inter], h[:, inter:]
    if activation == "gelu_tanh":
        return torch.nn.functional.gelu(gate, approximate="tanh") * up
    if activation in ("swiglu_clamp", "swigluoai"):
        gate = gate.clamp(max=limit)
        return gate * torch.sigmoid(alpha * gate) * (up.clamp(-limit, limit) + (activation == "swigluoai"))
    return torch.nn.functional.silu(gate) * up


@cuda
@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0), reason="needs sm_120")
@pytest.mark.parametrize("hidden_size,inter,activation,tokens", [
    (256, 128, "silu", 256), (320, 192, "gelu_tanh", 1024), (512, 1024, "swiglu_clamp", 256), (256, 128, "swigluoai", 256),
])
def test_w4a4_prefill_matches_nvfp4_activation_reference(hidden_size, inter, activation, tokens):
    """The w4a4 kernel from checkpoint pieces to the prefill output, against a torch emulation of the same NVFP4 activation rounding.

    Covers 256- / 128-row gated tiles, K % 256 != 0 in both GEMMs, the persistent down GEMM (I < 1024) and the other one, BLOCK_SIZE_M 64 and 128.
    """
    from types import SimpleNamespace

    from freetoken.kernel import nvfp4_moe
    from freetoken.layers.quantization.moe.base import ExpertView, MoEConfig
    from freetoken.layers.quantization.moe.nvfp4 import W4a4Nvfp4MoEKernel

    device = torch.device("cuda")
    if not nvfp4_moe.w4a4_available(device.index or 0):
        pytest.skip("the W4A4 Gluon kernels need Triton's Gluon on sm_120")
    alpha, limit = (1.5, 2.0) if activation in ("swiglu_clamp", "swigluoai") else (1.0, None)
    g = torch.Generator(device=device).manual_seed(0)

    def bank(n, k):
        packed = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=device, generator=g)
        scale = (torch.rand(E, n, k // 16, device=device, generator=g) + 0.25).to(torch.float8_e4m3fn)
        return packed, scale, torch.rand(E, n, device=device, generator=g) * 0.02 + 0.01

    gu, gus, gug = bank(2 * inter, hidden_size)
    dn, dns, dng = bank(hidden_size, inter)
    hidden = torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device=device, generator=g)
    topk_ids = torch.randint(0, E, (tokens, TOPK), dtype=torch.int32, device=device, generator=g)
    topk_weights = torch.rand(tokens, TOPK, device=device, generator=g)

    # static scales near a calibration's amax / (6 * 448), off by an irrational factor: with amax / 2688 itself many values sit exactly on an
    # e2m1 midpoint, where the kernel's approximate fp32 reciprocal and torch's exact one round apart; per expert, below the max for some experts
    g1 = float(hidden.float().abs().max()) / 2688 * math.pi / 3
    xq = _nvfp4_qdq(hidden, g1)
    w_gu = torch.stack([_dequant_ref(gu[e], gus[e], gug[e]) for e in range(E)])
    w_dn = torch.stack([_dequant_ref(dn[e], dns[e], dng[e]) for e in range(E)])
    routes = topk_ids.reshape(-1).long()
    h = torch.bmm(w_gu[routes], xq.repeat_interleave(TOPK, 0).unsqueeze(-1)).squeeze(-1).bfloat16().float()
    act = _gated_ref(h, activation, inter, alpha, limit or 0.0).bfloat16().float()
    g2 = float(act.abs().max()) / 2688 * math.pi / 3
    ref = torch.bmm(w_dn[routes], _nvfp4_qdq(act, g2).unsqueeze(-1)).squeeze(-1)
    ref = (ref * topk_weights.reshape(-1, 1)).view(tokens, TOPK, hidden_size).sum(1)

    def input_scale(top):
        return torch.full((E, 1), top, device=device).index_fill_(0, torch.tensor([1, 2], device=device), top / 2)

    pieces = {"gate_up": gu, "gate_up_scale": gus, "gate_up_global": gug, "down": dn, "down_scale": dns, "down_global": dng,
              "gate_input": input_scale(g1), "up_input": input_scale(g1).roll(1, 0), "down_input": input_scale(g2)}
    cfg = MoEConfig(num_experts=E, hidden=hidden_size, intermediate=inter, top_k=TOPK, activation=activation, alpha=alpha, limit=limit)
    kernel = W4a4Nvfp4MoEKernel()
    layout = kernel.layout(cfg)
    banks = {role: torch.empty((E, *spec.shape), dtype=spec.dtype, device=device) for role, spec in layout.items() if not spec.resident}
    alphas = kernel.pack(pieces, cfg, banks)
    view = ExpertView(banks, n=E, alphas=(alphas["gate_up_alpha"], alphas["down_alpha"]))
    layer = SimpleNamespace(activation=activation, alpha=alpha, limit=limit, apply_router_weight_on_input=False)
    out = kernel.apply(layer, hidden, topk_weights, topk_ids, view, is_prefill=True)
    assert torch.equal(alphas["gate_up_alpha"], torch.maximum(pieces["gate_input"], pieces["up_input"]).squeeze(1))
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2 * float(ref.abs().max()))


def _triton_cache(device, *, cache_size=S, prefill_overlap=False):
    """Native 6-bank NVFP4 cache (no repack), consumed directly by the Triton kernels.
    The banks are not transformed, so ``sources`` doubles as the dequant reference."""
    from freetoken.moe.offload_cache import OffloadMoeCache

    sources = _make_native_sources(device, seed=7)
    cache = OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        cache_size=cache_size,
        device=device,
        quant_format="nvfp4",
        prefill_overlap=prefill_overlap,
    )
    cache.set_bank_sources({name: sources[name] for name in cache.bank_schema})
    cache.reset()
    return cache, sources


@cuda
def test_triton_decode_marlin_matches_dequant_reference_after_prefill_stomp():
    """The production marlin-style int32 decode GEMV through the slot cache, including the
    request-B-after-request-A pattern (a layer-1 full-layer prefill between two layer-0
    decodes) that must force a miss + reload rather than serve stale slot bytes."""
    from freetoken.moe.fused_nvfp4 import fused_experts_decode_nvfp4_marlin

    device = torch.device("cuda")
    cache, ref_sources = _triton_cache(device)
    torch.manual_seed(2)
    hidden = torch.randn(1, H, dtype=torch.bfloat16, device=device) / 4
    topk_weights = torch.rand(1, TOPK, dtype=torch.float32, device=device)

    def decode(layer_id, experts):
        ids = torch.tensor([experts], dtype=torch.int32, device=device)
        ref = _ref_moe(ref_sources, layer_id, hidden, topk_weights, ids)
        cache.ensure_experts(layer_id, ids)  # rewrites ids -> slots in place
        cache.copy_missing()
        gu_p, gu_s, gu_g, dn_p, dn_s, dn_g = cache.pool_bank_views(layer_id)
        out = fused_experts_decode_nvfp4_marlin(
            hidden, gu_p, gu_s, gu_g, dn_p, dn_s, dn_g, topk_weights, ids, "silu", False
        )
        _assert_close(out, ref)

    decode(0, [3, 5])
    cache.materialize_layer(1)  # full-layer prefill overwrites every slot (S == E)
    cache.copy_missing()
    decode(0, [3, 5])  # must miss + reload, not serve layer-1 bytes
    decode(1, [1, 2])  # pure hits on the prefilled layer


@cuda
def test_triton_decode_marlin_matches_baseline_kernel():
    """The production marlin-style decode GEMV must match the original LUT-gather decode
    within tolerance (it only reorders the dequant math: int32 wide load + deferred reduce)."""
    from freetoken.moe.fused_nvfp4 import (
        fused_experts_decode_nvfp4_marlin,
        fused_experts_decode_nvfp4_serial,
    )

    device = torch.device("cuda")
    cache, _ = _triton_cache(device)
    torch.manual_seed(11)
    hidden = torch.randn(1, H, dtype=torch.bfloat16, device=device) / 4
    topk_weights = torch.rand(1, TOPK, dtype=torch.float32, device=device)
    ids = torch.tensor([[1, 6]], dtype=torch.int32, device=device)
    cache.ensure_experts(0, ids)
    cache.copy_missing()
    banks = cache.pool_bank_views(0)
    marlin = fused_experts_decode_nvfp4_marlin(hidden, *banks, topk_weights, ids, "silu", False)
    base = fused_experts_decode_nvfp4_serial(hidden, *banks, topk_weights, ids, "silu", False)
    torch.testing.assert_close(marlin.float(), base.float(), rtol=2e-3, atol=2e-3)


@cuda
def test_b12x_decode_matches_dequant_reference():
    """sm_120 + CUDA>=13 only: the flashinfer b12x W4A16 fused MoE over the slot cache
    vs the dequant reference (skipped on hardware/toolkits where b12x cannot run)."""
    from freetoken.layers.quantization.moe.nvfp4 import _b12x_unusable_reason, b12x_fused_experts

    device = torch.device("cuda")
    reason = _b12x_unusable_reason(torch.cuda.get_device_capability(device))
    if reason is not None:
        pytest.skip(f"b12x not runnable here: {reason}")

    cache, ref_sources = _packed_cache(device, "b12x", seed=8)

    torch.manual_seed(2)
    hidden = torch.randn(1, H, dtype=torch.bfloat16, device=device) / 4
    topk_weights = torch.rand(1, TOPK, dtype=torch.float32, device=device)
    ids = torch.tensor([[3, 5]], dtype=torch.int32, device=device)
    ref = _ref_moe(ref_sources, 0, hidden, topk_weights, ids)

    cache.ensure_experts(0, ids)
    cache.copy_missing()
    g1, g2 = cache.alphas_for_slots(0)
    gu_p, gu_s, dn_p, dn_s = cache.pool_bank_views(0)
    out = b12x_fused_experts(
        hidden, gu_p, gu_s, g1, dn_p, dn_s, g2, topk_weights, ids, "silu", False
    )
    _assert_close(out, ref)


@cuda
def test_dummy_nvfp4_banks_match_the_triton_layout():
    """--use-dummy-weight banks follow the bound kernel's layout (shapes, dtypes, pinning)."""
    from freetoken.moe.expert_banks import build_expert_banks

    layer = _bound_layer("triton")
    banks = build_expert_banks(layer.quant_method, L, None, device=torch.device("cuda"), dummy=True)
    expected = {
        "gate_up": ((E, 2 * I, H // 2), torch.uint8),
        "gate_up_scale": ((E, 2 * I, H // 16), torch.float8_e4m3fn),
        "gate_up_global": ((E, 2 * I), torch.float32),
        "down": ((E, H, I // 2), torch.uint8),
        "down_scale": ((E, H, I // 16), torch.float8_e4m3fn),
        "down_global": ((E, H), torch.float32),
    }
    assert banks.sources.keys() == expected.keys()
    for name, (shape, dtype) in expected.items():
        layers = banks.sources[name]
        assert len(layers) == L, (name, len(layers))
        for t in layers:
            assert t.shape == shape and t.dtype == dtype and t.is_pinned(), name


@cuda
@marlin
def test_dummy_nvfp4_banks_marlin_pack_gathers_zero_copy():
    """Dummy banks packed by marlin drop into the offload gather path like the real loader's."""
    from freetoken.moe.expert_banks import build_expert_banks
    from freetoken.moe.offload_cache import OffloadMoeCache

    device = torch.device("cuda")
    layer = _bound_layer("marlin")
    banks = build_expert_banks(layer.quant_method, L, None, device=device, dummy=True)
    assert torch.isfinite(banks.gate_up_alpha.float()).all()
    assert torch.isfinite(banks.down_alpha.float()).all()

    cache = OffloadMoeCache(
        num_layers=L, num_experts=E, cache_size=S, device=device, quant_format=banks.quant_format,
        layout=banks.layout, max_slots=layer.quant_method.kernel.max_slots,
    )
    cache.set_bank_sources(banks.sources)
    cache.set_alphas(banks.gate_up_alpha, banks.down_alpha)
    cache.reset()
    cache.materialize_layer(0)
    cache.copy_missing()
    torch.cuda.synchronize()
    assert torch.equal(cache.bank_caches["gate_up"][:E].cpu(), banks.sources["gate_up"][0])


@cuda
@pytest.mark.slow
def test_b12x_pack_keeps_per_layer_banks_and_flat_alphas():
    """The b12x pack is pure torch: its banks stay per-layer lists with flat [L*E] alphas."""
    from freetoken.layers.quantization.moe.nvfp4 import _b12x_unusable_reason

    device = torch.device("cuda")
    reason = _b12x_unusable_reason(torch.cuda.get_device_capability(device))
    if reason is not None:
        pytest.skip(f"b12x not runnable here: {reason}")
    cache, _ = _packed_cache(device, "b12x", seed=3)
    total = L * E
    assert len(cache.bank_sources["gate_up"]) == L
    assert sum(t.shape[0] for t in cache.bank_sources["gate_up"]) == total
    assert cache.gate_up_alpha.shape == (total,)
