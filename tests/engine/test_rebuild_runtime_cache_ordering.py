"""Engine.rebuild_runtime_cache resize ordering (#643): shrinking resizes must run
before growing ones. Each pool frees only its own old tensors before reallocating, so a
grow that runs while a still-large sibling pool is resident can OOM even though the
TARGET geometry passed validate_rebuild. Drives the real Engine method unbound on a
SimpleNamespace engine shell (the idiom of test_kv_cache_rebuild.py's page-table test),
with fake pools mirroring each real pool's free-own-then-alloc sequence under a shared
accounting ledger, and the real BaseKVCachePool.validate_rebuild as the fit check."""

from __future__ import annotations

import types

import pytest
import torch

from freetoken.engine.engine import Engine
from freetoken.kvcache.base import BaseKVCachePool


class _Ledger:
    """Shared GPU-memory stand-in: cap in units (1 unit = 1 MoE slot / KV page)."""

    def __init__(self, cap: int):
        self.cap = cap
        self.used = 0
        self.peak = 0
        self.events: list[tuple] = []

    def alloc(self, tag: str, n: int) -> None:
        if self.used + n > self.cap:
            self.events.append(("OOM", tag, n, self.used))
            raise torch.OutOfMemoryError(
                f"alloc {tag}={n} would peak at {self.used + n} > cap {self.cap}"
            )
        self.used += n
        self.peak = max(self.peak, self.used)
        self.events.append(("alloc", tag, n, self.used))

    def free(self, tag: str, n: int) -> None:
        self.used -= n
        self.events.append(("free", tag, n, self.used))


class _FakeMoeCache:
    """Mirrors OffloadMoeCache.rebuild: validate -> free own tensors -> cache_size
    mutated (offload_cache.py:491, before the alloc loop) -> reallocate."""

    def __init__(self, ledger: _Ledger, size: int):
        self.ledger = ledger
        self.num_experts = 8
        self.max_slots = None
        self.layout = None
        self.quant_format = "none"
        # one uint8 per expert row -> expert_bytes_per_slot == 1 unit
        self.bank_sources = {"w13": [torch.zeros((self.num_experts, 1), dtype=torch.uint8)]}
        self.cache_size = size
        self._held = size
        ledger.alloc("moe", size)

    def validate_rebuild(self, cache_size: int) -> None:
        if cache_size < self.num_experts:
            raise ValueError(f"cache_size {cache_size} < num_experts {self.num_experts}")

    def rebuild(self, cache_size: int) -> None:
        self.validate_rebuild(cache_size)
        self.ledger.free("moe", self._held)
        self._held = 0
        self.cache_size = cache_size
        self.ledger.alloc("moe", cache_size)
        self._held = cache_size


class _FakeKvPool(BaseKVCachePool):
    """Mirrors MHAKVCache.rebuild (free the one buffer, then realloc); reuses the REAL
    BaseKVCachePool.validate_rebuild with 1 unit per page and no fixed term."""

    needs_rebind_on_rebuild = False

    def __init__(self, ledger: _Ledger, num_pages: int):
        self.ledger = ledger
        self.pages = num_pages
        ledger.alloc("kv", num_pages)

    @classmethod
    def kv_cost(cls, config):
        return 1, 0, 16, 0

    def rebuild_from_config(self, config, num_pages, *, num_swa_pages=None):
        self.ledger.free("kv", self.pages)
        self.ledger.alloc("kv", num_pages)
        self.pages = num_pages

    def attach_page_table(self, page_table):
        pass

    def unit_bytes(self):
        return 1, 0

    def k_cache(self, index): ...
    def v_cache(self, index): ...
    def store_kv(self, k, v, out_loc, layer_id): ...

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return torch.float16

    @property
    def num_layers(self):
        return 1


class _FakeGraphRunner:
    def __init__(self, **kwargs):
        self.graph_bs_list = [1, 2, 4]
        self.destroy_cuda_graphs = lambda: None


def _build_engine(monkeypatch, ledger: _Ledger, *, moe: int, kv: int, budget: int):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr("freetoken.engine.engine.GraphRunner", _FakeGraphRunner)
    config = types.SimpleNamespace(
        page_size=16, max_seq_len=4096, max_running_req=4, memory_ratio=1.0,
        cuda_graph_max_bs=4, swa_num_pages_override=None,
        model_config=types.SimpleNamespace(vocab_size=128, model_is_mrope=False),
    )
    page_table = torch.zeros((config.max_running_req + 1, 256), dtype=torch.int32)
    eng = types.SimpleNamespace(
        config=config, moe_offload_cache=_FakeMoeCache(ledger, moe), kv_cache=_FakeKvPool(ledger, kv),
        linear_state_pool=None, num_pages=kv, device=torch.device("cpu"), stream=None, model=None,
        _baseline_free=budget, _weights_bytes=0, page_table=page_table, max_seq_len=4096,
        ctx=types.SimpleNamespace(page_table=page_table),
        dummy_req=types.SimpleNamespace(table_idx=0),
        attn_backend=types.SimpleNamespace(reset_capture=lambda: None),
        graph_runner=_FakeGraphRunner(),
        _sync_get_memory=lambda: (0, 0),
    )
    eng._target_moe_and_expert_bytes = Engine._target_moe_and_expert_bytes.__get__(eng)
    eng._resize_kv_pool = Engine._resize_kv_pool.__get__(eng)
    eng._refresh_seq_state = Engine._refresh_seq_state.__get__(eng)
    return eng


def _alloc_index(ledger: _Ledger, tag: str, n: int) -> int:
    return next(
        i for i, ev in enumerate(ledger.events) if ev[0] == "alloc" and ev[1] == tag and ev[2] == n
    )


def test_rebuild_runtime_cache_applies_shrinks_before_grows(monkeypatch):
    # moe 40->60 grows, kv 50->30 shrinks, target total 90 <= cap 100: the fit check
    # passes, so the rebuild must complete live instead of OOMing on the MoE grow.
    ledger = _Ledger(cap=100)
    eng = _build_engine(monkeypatch, ledger, moe=40, kv=50, budget=100)

    Engine.rebuild_runtime_cache(eng, moe_cache_size=60, num_pages=30)

    assert eng.moe_offload_cache.cache_size == 60
    assert eng.kv_cache.pages == 30 and eng.num_pages == 30
    assert ledger.used == 90  # target total resident
    assert ledger.peak == 90  # never above max(current, target) totals
    assert _alloc_index(ledger, "kv", 30) < _alloc_index(ledger, "moe", 60)  # shrink ran first


def test_rebuild_runtime_cache_reverse_direction_still_fits(monkeypatch):
    # The mirror request (moe shrinks, kv grows) worked before the fix because the old
    # fixed order happened to shrink MoE first; a naive static reorder would break it.
    ledger = _Ledger(cap=100)
    eng = _build_engine(monkeypatch, ledger, moe=60, kv=30, budget=100)

    Engine.rebuild_runtime_cache(eng, moe_cache_size=40, num_pages=50)

    assert eng.moe_offload_cache.cache_size == 40
    assert eng.kv_cache.pages == 50 and eng.num_pages == 50
    assert ledger.used == 90 and ledger.peak == 90
    assert _alloc_index(ledger, "moe", 40) < _alloc_index(ledger, "kv", 50)  # shrink ran first


def test_rebuild_runtime_cache_grow_failure_rolls_back(monkeypatch):
    # Budget 110 prices the 105-unit target as fitting but the real cap is 100 (fragmentation
    # / unaccounted allocations), so the grow phase OOMs after the shrink phase. The
    # scheduler's rollback (rebuild_cache(**prior) -> another rebuild_runtime_cache) must
    # restore the snapshot geometry. An OOM inside a KV/mamba realloc still wedges the
    # rollback itself; that is #526, out of scope here.
    ledger = _Ledger(cap=100)
    eng = _build_engine(monkeypatch, ledger, moe=40, kv=50, budget=110)

    with pytest.raises(torch.OutOfMemoryError):
        Engine.rebuild_runtime_cache(eng, moe_cache_size=70, num_pages=35)
    assert eng.rebuild_teardown_started  # the scheduler's mid-teardown signal

    Engine.rebuild_runtime_cache(eng, moe_cache_size=40, num_pages=50)  # rollback to prior

    assert eng.moe_offload_cache.cache_size == 40
    assert eng.kv_cache.pages == 50 and eng.num_pages == 50
    assert ledger.used == 90 and ledger.peak <= 100  # serving geometry restored under the cap
