"""The QSA pool under ``--kv-cache-dtype fp8``: quantized K/V, 16-bit index tiers.

Only the paged K/V slabs change. The compressed index slab, the per-request ring and
their scratch rows must stay 16-bit whatever the KV store does -- block selection scores
against them and the score kernel asserts their dtype -- while the byte account the
startup budget, the cache sliders and the runtime rebuild all divide by has to keep
telling the two halves apart. A drift here does not crash: it silently buys the wrong
number of pages.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention import AttnType
from freetoken.kvcache.base import spec_kv_bytes_per_token
from freetoken.models.config import KVCacheGroupSpec

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from freetoken.kernel.triton.kv_quant import codes_to_f32, kv_codes_dtype
from freetoken.kvcache.qsa_pool import QSAKVCache

DEV = torch.device("cuda")
PAGE_SIZE = 64  # the page size the qsa_sparse backend registers
LAYER_IDS = (1, 3, 5, 7)
HEADS, DIM, INDEX_DIM, INDEX_LAYERS, RATIO = 2, 64, 32, 4, 4


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.mha_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def _pool(kv_quant="fp8", num_pages=4, num_req_slots=4):
    return QSAKVCache(
        num_kv_heads=HEADS,
        num_layers=8,
        head_dim=DIM,
        num_pages=num_pages,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        device=DEV,
        index_head_dim=INDEX_DIM,
        num_index_layers=INDEX_LAYERS,
        index_ratio=RATIO,
        num_req_slots=num_req_slots,
        layer_ids=LAYER_IDS,
        kv_quant=kv_quant,
    )


def _spec():
    return KVCacheGroupSpec(
        name="full",
        layer_ids=LAYER_IDS,
        num_kv_heads=HEADS,
        head_dim=DIM,
        sliding_window=None,
        index_head_dim=INDEX_DIM,
        num_index_layers=INDEX_LAYERS,
        index_ratio=RATIO,
        attn_type=AttnType.QSA,
    )


def _config(kv_quant, *, page_size=PAGE_SIZE, max_running_req=3):
    mc = SimpleNamespace(num_layers=8, has_swa_attention=False, has_linear_attention=True)
    mc.kv_cache_group_specs = lambda: (_spec(),)
    return SimpleNamespace(
        model_config=mc,
        page_size=page_size,
        dtype=torch.bfloat16,
        tp_info=SimpleNamespace(size=1),
        max_running_req=max_running_req,
        kv_quant=kv_quant,
    )


def test_fp8_replaces_the_kv_slab_and_adds_scale_views():
    pool = _pool()
    bf16 = _pool(kv_quant="none")
    assert pool.kv_quant == "fp8"
    # Same geometry as the 16-bit pool -- only the element type changed, because the
    # attend kernels index codes exactly the way they index values.
    assert pool._kv_buffer.shape == bf16._kv_buffer.shape
    assert pool._kv_buffer.dtype == kv_codes_dtype()
    # dtype = compute dtype (what the backend sizes its INDEXER scratch with),
    # store_dtype = what the buffer holds. Swapping them is the bug that compiled an
    # e4m3 operand into QSA's scoring dot and died at graph capture.
    assert pool.dtype is torch.bfloat16
    assert pool.store_dtype == kv_codes_dtype()
    assert bf16.dtype is torch.bfloat16 and bf16.store_dtype is torch.bfloat16
    assert pool.k_cache(3).shape == (4, PAGE_SIZE, HEADS, DIM)
    assert pool.k_cache(3).element_size() == 1
    slots = 4 * PAGE_SIZE
    assert pool.k_scale(3).shape == (slots, HEADS)
    assert pool.k_scale(3).dtype is torch.float32
    assert pool.v_scale(3).shape == (slots, HEADS)
    # Zero-filled: e4m3 has NaN bit patterns, and the dummy page / unwritten tail rows
    # must never read back as one.
    assert pool.k_scale(3).abs().sum().item() == 0.0
    assert codes_to_f32(pool.k_cache(3)).abs().sum().item() == 0.0
    # A 16-bit pool keeps answering None, so the backends' k_scale(...) pass-through is
    # the only branch that ever differs between the two.
    assert bf16.k_scale(3) is None and bf16.v_scale(3) is None


@pytest.mark.parametrize("kv_quant", ["none", "fp8"])
def test_index_tiers_stay_16_bit_whatever_the_kv_store_does(kv_quant):
    """Block selection is quantization-agnostic by construction: it reads the compressed
    index keys, not the KV rows, so fp8 must not touch these three buffers."""
    pool = _pool(kv_quant=kv_quant)
    assert pool.cmp_k_cache(0).dtype is torch.bfloat16
    assert pool.pending_ring(0).dtype is torch.bfloat16
    assert pool.cmp_k_cache(0).shape == (4 * PAGE_SIZE // RATIO + 4, INDEX_DIM)
    # ...and the byte account says so too: only the 1-byte KV codes got cheaper.
    spec = _spec()
    cost = spec_kv_bytes_per_token(spec, _config(kv_quant))
    plain = spec_kv_bytes_per_token(spec, _config("none"))
    kv_layers = len(LAYER_IDS)
    kv_16bit = 2 * DIM * HEADS * 2 * kv_layers          # two slabs, 16-bit codes
    scale_term = 2 * kv_layers * HEADS * 4              # one fp32 scale per (slot, head)
    index_term = INDEX_DIM * INDEX_LAYERS * 2 // RATIO  # the untouched 16-bit slab
    assert plain == kv_16bit + index_term
    if kv_quant == "fp8":
        assert cost == kv_16bit // 2 + scale_term + index_term
    else:
        assert cost == plain


def test_unit_bytes_and_kv_cost_still_agree_when_quantized():
    """The pool's own allocation and the budget model must divide the same way -- the
    scale sidecar is priced in base.spec_kv_bytes_per_token, not here."""
    spec, config = _spec(), _config("fp8")
    pool = _pool()
    kv_bytes, swa_bytes = pool.unit_bytes()
    assert swa_bytes == 0
    assert kv_bytes == spec_kv_bytes_per_token(spec, config)
    assert kv_bytes * PAGE_SIZE == QSAKVCache.kv_cost(config)[0]
    # The 16-bit pool's per-token figure is the reference: codes halve the KV term, the
    # scale sidecar and the untouched index slab keep the total above a clean half.
    plain_kv = spec_kv_bytes_per_token(spec, _config("none"))
    assert plain_kv // 2 < kv_bytes < plain_kv


def test_rebuild_resizes_codes_and_scales_together():
    pool = _pool(num_pages=4)
    ident = id(pool)
    pool.rebuild(16)
    assert id(pool) == ident
    assert pool.k_cache(1).shape == (16, PAGE_SIZE, HEADS, DIM)
    assert pool.k_scale(1).shape == (16 * PAGE_SIZE, HEADS)
    assert pool.k_scale(1).abs().sum().item() == 0.0
    assert pool.cmp_k_cache(0).shape == (16 * PAGE_SIZE // RATIO + 4, INDEX_DIM)


def test_store_kv_writes_the_slot_the_attend_kernel_will_read():
    """out_loc numbering (page * page_size + offset) is the contract between the fused
    writer and the attend kernel's scale slot arithmetic -- this is that round trip."""
    torch.manual_seed(0)
    pool = _pool(num_pages=4)
    slots = 4 * PAGE_SIZE
    rows = (0, 1, 63, 64, 255, 256)  # page boundaries included: 63/64 and 255/256
    k = torch.randn(len(rows), HEADS * DIM, device=DEV, dtype=torch.bfloat16) * 3.0
    v = torch.randn(len(rows), HEADS * DIM, device=DEV, dtype=torch.bfloat16) * 0.25
    out_loc = torch.tensor(rows, dtype=torch.int32, device=DEV)

    pool.store_kv(k, v, out_loc, layer_id=3)
    torch.cuda.synchronize()

    codes = codes_to_f32(pool.k_cache(3).view(slots, HEADS, DIM))
    decoded_k = codes[out_loc] * pool.k_scale(3)[out_loc].unsqueeze(-1)
    decoded_v = codes_to_f32(pool.v_cache(3).view(slots, HEADS, DIM))[out_loc] * pool.v_scale(3)[
        out_loc
    ].unsqueeze(-1)
    for want, got in ((k, decoded_k), (v, decoded_v)):
        w = want.view(len(rows), HEADS, DIM).to(torch.float32)
        amax = w.abs().amax(dim=-1, keepdim=True)
        assert (((got - w).abs() / amax).max().item()) < 0.07
    # Rows nobody wrote stay zero rather than NaN -- the dummy page depends on it.
    untouched = torch.tensor([r for r in range(64) if r not in rows], device=DEV)
    assert pool.k_scale(3)[untouched].abs().sum().item() == 0.0


def test_factory_threads_kv_quant_into_the_qsa_pool():
    from freetoken.kvcache import create_kvcache_pool

    mc = SimpleNamespace(
        num_layers=8, has_swa_attention=False, has_linear_attention=True,
        num_kv_heads=HEADS, head_dim=DIM, dsv4_args=None,
    )
    mc.kv_cache_group_specs = lambda: (_spec(),)
    pool = create_kvcache_pool(
        mc, num_pages=4, page_size=PAGE_SIZE, dtype=torch.bfloat16, device=DEV,
        num_req_slots=4, kv_quant="fp8",
    )
    assert isinstance(pool, QSAKVCache) and pool.kv_quant == "fp8"
    assert pool.k_cache(1).element_size() == 1 and pool.k_scale(1) is not None

