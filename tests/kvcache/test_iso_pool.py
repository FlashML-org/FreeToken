"""ISOKVCache pool tests (kvcache/iso_pool.py)."""

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kvcache.iso_pool import ISOKVCache
from freetoken.kvcache.mha_pool import MHAKVCache


def _init_tp() -> None:
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


class _TP:
    size = 1
    rank = 0


class _Spec:
    def __init__(self, num_kv_heads=2, head_dim=256, num_layers=10, is_swa=False):
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.is_swa = is_swa
        self.mla = False
        self.index_head_dim = 0
        self.num_index_layers = 0
        self.index_ratio = 1


class _ModelConfig:
    def __init__(self, spec):
        self._spec = spec

    def kv_cache_group_specs(self):
        return [self._spec]


class _Config:
    def __init__(self, spec, fmt, page_size=1):
        self.model_config = _ModelConfig(spec)
        self.tp_info = _TP()
        self.page_size = page_size
        self.kv_cache_iso = fmt
        import torch as _t

        self.dtype = _t.bfloat16


@pytest.mark.parametrize("fmt,rowb", [("iso3", 100), ("iso4", 136)])
def test_kv_cost_vs_mha(fmt, rowb):
    spec = _Spec()  # 2 heads x 256 dim x 10 layers
    cfg = _Config(spec, fmt)
    per_page, fixed, page_tokens, reserve = ISOKVCache.kv_cost(cfg)
    # per token: 2 (K+V) * rowb * 2 heads * 10 layers
    assert per_page == 2 * rowb * 2 * 10
    assert (fixed, page_tokens, reserve) == (0, 1, 0)
    mha_per_page, _, _, _ = MHAKVCache.kv_cost(cfg)
    # bf16: 2 * 256 * 2 * 2 * 10 = 20480 per token
    assert mha_per_page == 20480
    ratio = mha_per_page / per_page
    assert ratio > 3.7 if fmt == "iso4" else ratio > 5.0


def test_pool_alloc_and_accessors():
    _init_tp()
    pool = ISOKVCache(
        num_kv_heads=2, num_layers=3, head_dim=256, num_pages=8, page_size=1,
        dtype=torch.bfloat16, device=torch.device("cpu"), iso_fmt="iso3",
    )
    assert pool.k_cache(0).shape == (8, 1, 2, 100)
    assert pool.k_cache(0).dtype == torch.uint8
    ub, swa = pool.unit_bytes()
    assert ub == 2 * 3 * 8 * 1 * 2 * 100 // 8  # bytes per token-slot
    assert swa == 0
    assert pool.num_layers == 3
    with pytest.raises(KeyError):
        ISOKVCache(
            num_kv_heads=2, num_layers=3, head_dim=256, num_pages=8, page_size=1,
            dtype=torch.bfloat16, device=torch.device("cpu"), layer_ids=[0, 2],
        ).k_cache(1)


def test_layer_remap():
    _init_tp()
    pool = ISOKVCache(
        num_kv_heads=2, num_layers=40, head_dim=256, num_pages=4, page_size=1,
        dtype=torch.bfloat16, device=torch.device("cpu"),
        layer_ids=[3, 7, 11, 15, 19, 23, 27, 31, 35, 39], iso_fmt="iso4",
    )
    assert pool.k_cache(39).shape == (4, 1, 2, 136)
    assert pool._kv_buffer.shape[1] == 10  # only the 10 full-attention layers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("fmt", ["iso3", "iso4"])
def test_store_roundtrip(fmt):
    from freetoken.kernel import iso

    _init_tp()
    torch.manual_seed(13)
    pool = ISOKVCache(
        num_kv_heads=2, num_layers=2, head_dim=256, num_pages=64, page_size=1,
        dtype=torch.bfloat16, device=torch.device("cuda"), iso_fmt=fmt,
    )
    n = 24
    k = (torch.randn(n, 2, 256, device="cuda") * 2).to(torch.bfloat16)
    v = (torch.randn(n, 2, 256, device="cuda") * 2).to(torch.bfloat16)
    out_loc = torch.randperm(64, device="cuda", dtype=torch.int32)[:n]
    pool.store_kv(k, v, out_loc, layer_id=1)
    torch.cuda.synchronize()
    ref = iso.quantize_ref(k.float().reshape(n * 2, 256), fmt)
    got = pool.k_cache(1).view(64, -1)[out_loc.long()]
    want = ref.view(torch.uint8).cuda().reshape(n, 2 * iso.packed_row_bytes(256, fmt))
    assert torch.equal(got, want)
