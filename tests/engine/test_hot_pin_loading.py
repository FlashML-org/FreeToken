"""CPU startup coverage for bounded hot staging and final cache contents."""

from types import SimpleNamespace
import weakref

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.engine import engine as engine_module
from freetoken.layers.quantization import QuantKind
from freetoken.layers.quantization.moe.base import BankSpec
from freetoken.moe import expert_banks, expert_pieces


class _PackedMethod:
    """Byte-copy pack oracle, including the resident-alpha part of the contract."""

    def __init__(self, dtype=torch.bfloat16, resident=False):
        self.cfg = SimpleNamespace(num_experts=8)
        self.kind = QuantKind.NVFP4 if resident else QuantKind.NONE
        self.kernel = SimpleNamespace(name="marlin" if resident else "fused")
        self.specs = {
            "gate_up": BankSpec((4, 4), dtype),
            "down": BankSpec((4, 2), dtype),
        }
        if resident:
            self.specs.update(
                gate_up_alpha=BankSpec((), torch.float32, resident=True),
                down_alpha=BankSpec((), torch.float32, resident=True),
            )
        self.stages = {}
        self.peak_stage_bytes = 0

    def layout(self):
        return self.specs

    def slot_limit(self):
        return None

    def track(self, tensor):
        root = tensor
        while root._base is not None:
            root = root._base
        # HostBank roots are flat mmap views; packed hot scratch owns its shape.
        if root.ndim == 3:
            self.stages[id(root)] = weakref.ref(root)
        self.peak_stage_bytes = max(self.peak_stage_bytes, self.live_stage_bytes())

    def pack(self, piece, out):
        for role, tensor in out.items():
            self.track(tensor)
            tensor.copy_(piece[role])
        return {role: piece[role].clone() for role, spec in self.specs.items() if spec.resident}

    def live_stage_bytes(self):
        return sum(t.numel() * t.element_size() for ref in self.stages.values() if (t := ref()) is not None)


def _source(method, num_layers=3):
    refs = []
    for layer in range(num_layers):
        roles = {}
        for i, (role, spec) in enumerate(method.layout().items()):
            tensor = torch.empty((8, *spec.shape), dtype=spec.dtype)
            if spec.resident:
                tensor.copy_(torch.arange(8) + 10 * layer + i)
            else:
                # Compare bytes, including FP8 representations with no CPU arithmetic.
                data = tensor.view(torch.uint8).reshape(8, -1)
                data.copy_((torch.arange(data.numel()).reshape(data.shape) + 29 * layer + i).to(torch.uint8))
            roles[role] = tensor
        refs.append(roles)
    return refs


def _pieces(refs):
    # Interleave partial layers, finish in a different order, and reverse expert order.
    for lo, hi in ((4, 8), (0, 4)):
        for layer in (2, 0, 1):
            yield layer, lo, hi, {role: tensor[lo:hi] for role, tensor in refs[layer].items()}


def _engine(monkeypatch, method, *, dummy=False, fail_sizing=False):
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    pins = [[7, 2], [], [5, 0]]
    model_config = SimpleNamespace(num_moe_layers=3, num_experts=8, decode_target="gpu")
    config = SimpleNamespace(
        model_path="checkpoint", model_config=model_config, moe_strategy="offload",
        moe_cpu_layers=None, moe_prefill_overlap=False, expert_load="serial",
        use_dummy_weight=dummy, moe_cache_auto=True, num_page_override=None,
        moe_cache_size=0, moe_cache_policy="lru", moe_prefill_hit_d2d=False,
        moe_hybrid_max_fetch=1, hot_expert_slots=2, hot_expert_lru_floor=0,
        tune_file=None, moe_collect_stats=False, hot_expert_repin_interval_s=0,
        hot_stats_out=None,
    )
    engine = engine_module.Engine.__new__(engine_module.Engine)
    engine.model = object()
    engine.device = torch.device("cpu")
    engine.dtype = torch.bfloat16
    engine._host_tables_bytes = 0
    engine.ctx = SimpleNamespace()
    monkeypatch.setattr(engine_module, "shared_offload_method", lambda _model: method)
    monkeypatch.setattr(engine_module, "_check_pin_budget", lambda *a, **kw: None)
    monkeypatch.setattr(engine_module, "_resolve_hot_pin_plan", lambda *a: SimpleNamespace(pins=pins, catalog=None))
    monkeypatch.setattr(engine_module, "attach_offload_moe_cache", lambda *a: [None] * 3)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def size(*args):
        assert method.live_stage_bytes() == 0, "completed hot stages survived until cache sizing"
        if fail_sizing:
            raise RuntimeError("injected sizing failure")
        return 520, 37, False

    engine._resolve_auto_moe_cache_size = size
    return engine, config, pins


@pytest.mark.parametrize("dtype,resident", [
    (torch.bfloat16, False), (torch.float16, False), (torch.float32, False),
    (torch.float8_e4m3fn, False), (torch.float8_e8m0fnu, False),
    (torch.int32, True), (torch.uint8, True),
])
def test_engine_hot_staging_is_bounded_and_installs_exact_rows(monkeypatch, dtype, resident):
    method = _PackedMethod(dtype, resident)
    refs = _source(method)
    reads = []

    def reader(*args, **kwargs):
        reads.append(kwargs["parallel"])
        return _pieces(refs)

    monkeypatch.setattr(expert_pieces, "iter_expert_pieces", reader)
    engine, config, pins = _engine(monkeypatch, method)
    cache = engine._init_offload_moe_cache(config)

    assert reads == [False, False]
    assert (config.moe_cache_size, config.num_page_override) == (520, 37)
    row_bytes = sum(torch.empty(spec.shape, dtype=spec.dtype).numel() * torch.empty((), dtype=spec.dtype).element_size()
                    for spec in method.layout().values() if not spec.resident)
    assert method.peak_stage_bytes <= row_bytes
    assert method.live_stage_bytes() == 0
    for layer in range(3):
        for expert in range(8):
            for role, spec in method.layout().items():
                if spec.resident:
                    actual = getattr(cache, role)[layer * 8 + expert]
                elif expert in pins[layer]:
                    actual = cache.bank_caches[role][int(cache.slot_for_id[layer, expert])]
                else:
                    actual = cache.bank_sources[role][layer][int(cache.cold_row[layer, expert])]
                assert torch.equal(actual.reshape(-1).view(torch.uint8), refs[layer][role][expert].reshape(-1).view(torch.uint8))


def test_engine_dummy_hot_staging_does_not_read_checkpoint(monkeypatch):
    method = _PackedMethod(resident=True)
    original_fill = expert_banks._dummy_fill

    def fill(role, tensor):
        method.track(tensor)
        original_fill(role, tensor)

    monkeypatch.setattr(expert_banks, "_dummy_fill", fill)
    monkeypatch.setattr(expert_pieces, "iter_expert_pieces", lambda *a, **kw: pytest.fail("dummy load read checkpoint"))
    engine, config, pins = _engine(monkeypatch, method, dummy=True)
    cache = engine._init_offload_moe_cache(config)
    for layer, experts in enumerate(pins):
        for expert in experts:
            slot = int(cache.slot_for_id[layer, expert])
            assert torch.isfinite(cache.bank_caches["gate_up"][slot]).all()
            assert cache.bank_caches["gate_up"][slot].count_nonzero() > 0
    assert torch.equal(cache.gate_up_alpha, torch.ones(24))
    assert torch.equal(cache.down_alpha, torch.ones(24))
    assert method.peak_stage_bytes <= (4 * 4 + 4 * 2) * 2
    assert method.live_stage_bytes() == 0


def test_engine_failed_sizing_does_not_retain_hot_stages(monkeypatch):
    method = _PackedMethod()
    reads = []
    refs = _source(method)

    def reader(*args, **kwargs):
        reads.append(1)
        return _pieces(refs)

    monkeypatch.setattr(expert_pieces, "iter_expert_pieces", reader)
    engine, config, _ = _engine(monkeypatch, method, fail_sizing=True)
    with pytest.raises(RuntimeError, match="injected sizing failure"):
        engine._init_offload_moe_cache(config)
    assert reads == [1]
    assert method.live_stage_bytes() == 0


@pytest.mark.parametrize("corruption,match", [
    ("duplicate", "more than once"),
    ("missing", "rows missing"),
    ("layer", "out of range"),
    ("range", "out of range"),
    ("read", "injected read failure"),
])
def test_engine_hot_reload_rejects_bad_source_and_closes_reader(monkeypatch, corruption, match):
    method = _PackedMethod()
    refs = _source(method)
    reads = 0
    closed = []

    def reader(*args, **kwargs):
        nonlocal reads
        reads += 1
        if reads == 1:
            return _pieces(refs)

        def bad_pieces():
            try:
                rows = list(_pieces(refs))
                if corruption == "missing":
                    rows.pop(3)  # includes layer 2's pinned expert 0
                elif corruption == "duplicate":
                    rows.append(rows[0])
                elif corruption == "layer":
                    rows[0] = (3, *rows[0][1:])
                elif corruption == "range":
                    rows[0] = (2, 4, 9, rows[0][3])
                for row in rows:
                    yield row
                    if corruption == "read":
                        raise OSError("injected read failure")
            finally:
                closed.append(True)

        return bad_pieces()

    monkeypatch.setattr(expert_pieces, "iter_expert_pieces", reader)
    engine, config, _ = _engine(monkeypatch, method)
    with pytest.raises(engine_module.WeightLoadError, match=match):
        engine._init_offload_moe_cache(config)
    assert reads == 2
    assert closed == [True]
    assert not hasattr(engine, "moe_offload_cache")


def test_engine_hot_reload_updates_resident_alphas_on_cache_copy(monkeypatch):
    method = _PackedMethod(torch.int32, resident=True)
    refs = _source(method)
    reads = 0

    def reader(*args, **kwargs):
        nonlocal reads
        reads += 1
        if reads == 2:
            for layer in (0, 2):
                refs[layer]["gate_up_alpha"].add_(100)
                refs[layer]["down_alpha"].add_(200)
        return _pieces(refs)

    monkeypatch.setattr(expert_pieces, "iter_expert_pieces", reader)
    original = engine_module.OffloadMoeCache.set_alphas

    def copied_alphas(cache, gate_up, down):
        original(cache, gate_up.clone(), down.clone())

    monkeypatch.setattr(engine_module.OffloadMoeCache, "set_alphas", copied_alphas)
    engine, config, pins = _engine(monkeypatch, method)
    cache = engine._init_offload_moe_cache(config)
    for layer, experts in enumerate(pins):
        for expert in experts:
            assert cache.gate_up_alpha[layer * 8 + expert] == refs[layer]["gate_up_alpha"][expert]
            assert cache.down_alpha[layer * 8 + expert] == refs[layer]["down_alpha"][expert]


def test_engine_without_hot_pins_keeps_single_pass(monkeypatch):
    method = _PackedMethod()
    refs = _source(method)
    reads = []

    def reader(*args, **kwargs):
        reads.append(kwargs["parallel"])
        return _pieces(refs)

    monkeypatch.setattr(expert_pieces, "iter_expert_pieces", reader)
    engine, config, _ = _engine(monkeypatch, method)
    monkeypatch.setattr(engine_module, "_resolve_hot_pin_plan", lambda *a: None)
    cache = engine._init_offload_moe_cache(config)
    assert reads == [False]
    assert cache.pin_ids is None
    assert method.peak_stage_bytes == 0
    for layer in range(3):
        for role in ("gate_up", "down"):
            assert torch.equal(cache.bank_sources[role][layer].view(torch.uint8), refs[layer][role].view(torch.uint8))


def test_engine_parallel_cold_load_rereads_hot_rows_serially(monkeypatch):
    method = _PackedMethod()
    refs = _source(method)
    reads = []

    def reader(*args, **kwargs):
        reads.append(kwargs["parallel"])
        return _pieces(refs)

    monkeypatch.setattr(expert_pieces, "iter_expert_pieces", reader)
    engine, config, _ = _engine(monkeypatch, method)
    config.expert_load = "parallel"
    engine._init_offload_moe_cache(config)
    assert reads == [True, False]
