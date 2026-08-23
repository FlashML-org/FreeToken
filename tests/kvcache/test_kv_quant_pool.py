"""KV pools backed by 8-bit storage: allocation, store/read-back, cost, rebuild."""

from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.quant import BLOCK, FP8_E4M3, NONE, Q8_0

from .test_hybrid_swa_kv_cache import _kv_group_specs, _patch_tp

SPECS = [Q8_0, FP8_E4M3]
IDS = [spec.name for spec in SPECS]

# Measured relative L2 error of a round trip through each scheme on gaussian KV with a
# 32-element block: q8_0 ~0.005, fp8_e4m3 ~0.024. int8 wins by ~4.5x because a block that
# small keeps outliers from stretching the scale, which is the reason q8_0 is the default.
MAX_REL_ERR = {Q8_0.name: 0.01, FP8_E4M3.name: 0.03}

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _swa_pool(quant, device="cuda", num_full_pages=64, num_swa_tokens=32):
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache

    return HybridSWAKVCache(
        groups=_kv_group_specs(),
        num_layers=6,
        num_full_pages=num_full_pages,
        page_size=1,
        num_swa_tokens=num_swa_tokens,
        device=torch.device(device),
        dtype=torch.bfloat16,
        quant=quant,
    )


def _mha_pool(quant, device="cuda", num_pages=64):
    from freetoken.kvcache.mha_pool import MHAKVCache

    return MHAKVCache(
        num_kv_heads=2,
        num_layers=4,
        head_dim=256,
        num_pages=num_pages,
        page_size=1,
        dtype=torch.bfloat16,
        device=torch.device(device),
        quant=quant,
    )


@cuda_only
@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_swa_pool_allocates_8bit_slabs_and_matching_scales(monkeypatch, spec):
    _patch_tp(monkeypatch)
    pool = _swa_pool(spec)

    # Layer 0 is SWA (head_dim 256, 8 kv heads), layer 2 full (head_dim 512, 2 heads).
    for layer, head_dim, heads in ((0, 256, 8), (2, 512, 2)):
        k = pool.k_cache(layer)
        s = pool.k_scale(layer)
        assert k.dtype == spec.storage_dtype
        assert s.dtype == torch.float16
        assert k.shape[-2:] == (heads, head_dim)
        assert s.shape[-2:] == (heads, head_dim // BLOCK)
        assert s.shape[:-1] == k.shape[:-1]


@cuda_only
def test_unquantized_pool_keeps_bf16_and_has_no_scales(monkeypatch):
    _patch_tp(monkeypatch)
    pool = _swa_pool(NONE)
    assert pool.k_cache(0).dtype == torch.bfloat16
    assert pool.k_scale(0) is None
    assert pool.v_scale(0) is None
    assert not pool.quant.enabled


@cuda_only
@pytest.mark.parametrize("spec", SPECS, ids=IDS)
@pytest.mark.parametrize("layer", [0, 2], ids=["swa", "full"])
def test_store_kv_round_trips_through_the_quantized_pool(monkeypatch, spec, layer):
    _patch_tp(monkeypatch)
    pool = _swa_pool(spec)
    heads, head_dim = pool.k_cache(layer).shape[-2:]

    tokens = 8
    g = torch.Generator(device="cuda").manual_seed(0)
    k = torch.randn(tokens, heads, head_dim, generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(tokens, heads, head_dim, generator=g, device="cuda", dtype=torch.bfloat16)
    out_loc = torch.arange(1, tokens + 1, device="cuda", dtype=torch.int32)
    if pool.is_swa_layer(layer):
        pool.alloc_swa(out_loc)

    pool.store_kv(k, v, out_loc, layer)

    slots = (
        pool.translate_loc_from_full_to_swa(out_loc) if pool.is_swa_layer(layer) else out_loc
    ).to(torch.long)
    got = spec.dequantize(
        pool.k_cache(layer).view(-1, heads, head_dim)[slots].float(),
        pool.k_scale(layer).view(-1, heads, head_dim // BLOCK)[slots],
    )
    # Storing is lossy by construction; what must hold is that it round-trips to within
    # the scheme's envelope, not that it is exact.
    rel = ((got - k.float()).norm() / k.float().norm()).item()
    assert rel < MAX_REL_ERR[spec.name], (
        f"{spec.name} layer {layer}: relative round-trip error {rel:.4f}"
    )


@cuda_only
@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_unit_bytes_counts_the_scale_slab(monkeypatch, spec):
    """Budgeting has to see 1 + 2/32 bytes per element, not 1 -- otherwise a rebuild
    would size pools against memory the scales are quietly consuming."""
    _patch_tp(monkeypatch)
    quantized = _swa_pool(spec)
    plain = _swa_pool(NONE)

    q_full, q_swa = quantized.unit_bytes()
    p_full, p_swa = plain.unit_bytes()
    for q, p in ((q_full, p_full), (q_swa, p_swa)):
        # bf16 is 2 bytes/element, the quantized pool 1 + 2/32 = 1.0625.
        assert q == pytest.approx(p * (1.0625 / 2.0), rel=1e-3)


@cuda_only
@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_rebuild_reallocates_scales_and_keeps_identity(monkeypatch, spec):
    _patch_tp(monkeypatch)
    pool = _swa_pool(spec, num_full_pages=64, num_swa_tokens=32)
    before = id(pool)

    pool.rebuild(num_full_pages=128, num_swa_tokens=64)

    assert id(pool) == before, "rebuild must preserve object identity"
    for layer in (0, 2):
        k, s = pool.k_cache(layer), pool.k_scale(layer)
        assert k.dtype == spec.storage_dtype
        assert s is not None and s.dtype == torch.float16
        assert s.shape[:-1] == k.shape[:-1]
        assert s.shape[-1] == k.shape[-1] // BLOCK
    assert pool.k_cache(2).shape[0] == 128
    assert pool.k_cache(0).shape[0] == 64


@cuda_only
@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_mha_pool_quantizes_and_round_trips(monkeypatch, spec):
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.mha_pool.get_tp_info", lambda: DistributedInfo(rank=0, size=1)
    )
    pool = _mha_pool(spec)
    assert pool.k_cache(0).dtype == spec.storage_dtype
    assert pool.k_scale(0).shape[-1] == 256 // BLOCK

    g = torch.Generator(device="cuda").manual_seed(1)
    k = torch.randn(6, 2, 256, generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(6, 2, 256, generator=g, device="cuda", dtype=torch.bfloat16)
    out_loc = torch.arange(1, 7, device="cuda", dtype=torch.int32)
    pool.store_kv(k, v, out_loc, 0)

    idx = out_loc.to(torch.long)
    got = spec.dequantize(
        pool.k_cache(0).view(-1, 2, 256)[idx].float(),
        pool.k_scale(0).view(-1, 2, 256 // BLOCK)[idx],
    )
    rel = ((got - k.float()).norm() / k.float().norm()).item()
    assert rel < MAX_REL_ERR[spec.name]

    pool.rebuild(32)
    assert pool.k_cache(0).shape[0] == 32
    assert pool.k_scale(0).shape[0] == 32


def test_cost_model_prices_the_quantized_pool_below_bf16(monkeypatch):
    """The whole point of the flag, as arithmetic: same geometry, ~half the bytes."""
    from freetoken.kvcache.base import spec_kv_bytes_per_token
    from freetoken.distributed.info import DistributedInfo
    from types import SimpleNamespace

    spec = _kv_group_specs()[0]  # full group: 2 kv heads, head_dim 512, 2 layers
    tp = DistributedInfo(rank=0, size=1)

    def cfg(quant):
        return SimpleNamespace(tp_info=tp, dtype=torch.bfloat16, kv_quant=quant)

    plain = spec_kv_bytes_per_token(spec, cfg(NONE))
    quantized = spec_kv_bytes_per_token(spec, cfg(Q8_0))
    assert quantized == pytest.approx(plain * (1.0625 / 2.0), rel=1e-3)
    # A config with no kv_quant attribute at all must price as bf16 (back-compat with
    # every caller that predates the flag).
    legacy = spec_kv_bytes_per_token(spec, SimpleNamespace(tp_info=tp, dtype=torch.bfloat16))
    assert legacy == plain


@cuda_only
def test_q8_0_stores_kv_more_accurately_than_fp8(monkeypatch):
    """Why q8_0 is the default, measured rather than argued.

    Both schemes cost the same bytes and the same kernel work here, so the choice is
    purely numerical. On a 32-element block int8's uniform grid beats e4m3's 3-bit
    mantissa by several times -- fp8's advantage only shows up with a whole-head scale,
    where one outlier would crush everything else, or on kernels with native fp8 support
    (which the SWA path cannot use anyway).
    """
    _patch_tp(monkeypatch)
    g = torch.Generator(device="cuda").manual_seed(7)
    k = torch.randn(16, 8, 256, generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(16, 8, 256, generator=g, device="cuda", dtype=torch.bfloat16)
    out_loc = torch.arange(1, 17, device="cuda", dtype=torch.int32)

    errs = {}
    for spec in SPECS:
        pool = _swa_pool(spec, num_swa_tokens=64)
        pool.alloc_swa(out_loc)
        pool.store_kv(k, v, out_loc, 0)
        slots = pool.translate_loc_from_full_to_swa(out_loc).to(torch.long)
        got = spec.dequantize(
            pool.k_cache(0).view(-1, 8, 256)[slots].float(),
            pool.k_scale(0).view(-1, 8, 256 // BLOCK)[slots],
        )
        errs[spec.name] = ((got - k.float()).norm() / k.float().norm()).item()

    assert errs[Q8_0.name] < errs[FP8_E4M3.name] / 2, errs
