"""The shared expert of Qwen3.5 / 3.6 MoE as one more route of the routed experts' decode kernels.

Which steps take it (decode steps of 1 to 6 rows whose router runs in the top-k launch, on resident experts under the
Triton NVFP4 kernels, one rank), what it is built from, and on CUDA the MoE block's output against the same routes with
the shared expert as one more slot of the experts' bank, at the 35B-A3B shapes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import freetoken.layers.moe as moe_layers
import freetoken.models.qwen3_5_moe.moe as qmoe
from freetoken.layers.moe import MoELayer, OffloadMoELayer
from freetoken.layers.quantization.linear.nvfp4 import TritonNvfp4LinearKernel
from freetoken.layers.quantization.linear.unquantized import TorchLinearKernel
from freetoken.layers.quantization.moe.base import ExpertView
from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel

E, TOPK, H, MI = 256, 8, 2048, 512
ROLES = ("gate_up", "gate_up_scale", "gate_up_global", "down", "down_scale", "down_global")
FP8 = torch.float8_e4m3fn


def _ctx(prefill: bool):
    return SimpleNamespace(batch=SimpleNamespace(is_prefill=prefill))


def _row_major(n, k, gen, device="cpu", glob=1.0):
    """One NVFP4 projection [n, k] in the bank's row-major layout: codes, block scales, per-row globals."""
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=gen)
    scale = (torch.rand(n, k // 16, generator=gen) * 1.5 + 0.25).to(FP8)
    row_global = (torch.rand(n, generator=gen) + 0.5) * glob
    return packed.to(device), scale.to(device), row_global.to(device)


def _resident_linear(packed, scale, row_global, out_features, in_features):
    """The projection as TritonNvfp4LinearKernel keeps it: K-major words and scales (nvfp4_transpose_resident)."""
    return SimpleNamespace(weight=packed.view(torch.int32).t().contiguous(), weight_scale=scale.t().contiguous(),
                           weight_global=row_global, out_features=out_features, in_features=in_features, _tp_size=1,
                           quant_method=SimpleNamespace(kernel=object.__new__(TritonNvfp4LinearKernel)))


def _experts(banks, *, cls=MoELayer, kernel=None, tp_size=1):
    layer = object.__new__(cls)
    kernel = kernel or TritonNvfp4MoEKernel()
    layer.quant_method = SimpleNamespace(
        kernel=kernel,
        resident_view=lambda lyr: ExpertView(banks, n=E),
        apply=lambda x, w, i, view, *, layer, is_prefill: kernel.apply(layer, x, w, i, view, is_prefill=is_prefill),
    )
    layer._expert_banks = banks
    layer.num_experts, layer.top_k, layer.renormalize = E, TOPK, True
    layer.tp_size, layer.activation, layer.apply_router_weight_on_input = tp_size, "silu", False
    layer.alpha, layer.limit = 1.0, None
    return layer


def _moe(device="cpu", seed=0, glob=1.0):
    """A Qwen3_5MoE with resident NVFP4 experts and shared expert, a bf16 router and gate, built without a model."""
    gen = torch.Generator().manual_seed(seed)
    bank_gate_up = [_row_major(2 * MI, H, gen, device, glob) for _ in range(E)] if device != "cpu" else None
    if bank_gate_up is None:   # the CPU checks look only at shapes
        banks = {"gate_up": torch.zeros(E, 2 * MI, H // 2, dtype=torch.uint8), "gate_up_scale": torch.zeros(E, 2 * MI, H // 16, dtype=FP8),
                 "gate_up_global": torch.zeros(E, 2 * MI), "down": torch.zeros(E, H, MI // 2, dtype=torch.uint8),
                 "down_scale": torch.zeros(E, H, MI // 16, dtype=FP8), "down_global": torch.zeros(E, H)}
    else:
        bank_down = [_row_major(H, MI, gen, device, glob) for _ in range(E)]
        banks = {role: torch.stack([p[i] for p in parts]) for parts, names in ((bank_gate_up, ROLES[:3]), (bank_down, ROLES[3:]))
                 for i, role in enumerate(names)}
    moe = object.__new__(qmoe.Qwen3_5MoE)
    moe.experts = _experts(banks)
    router = (torch.randn(E, H, generator=gen) * 0.02).to(device, torch.bfloat16)
    gate_w = (torch.randn(1, H, generator=gen) * 0.05).to(device, torch.bfloat16)
    moe.gate = SimpleNamespace(weight=router, bias=None, quant_method=SimpleNamespace(kernel=TorchLinearKernel()))
    moe.shared_expert_gate = SimpleNamespace(weight=gate_w, bias=None, quant_method=SimpleNamespace(kernel=TorchLinearKernel()),
                                             forward=lambda x: torch.nn.functional.linear(x, gate_w))
    moe.shared_expert = SimpleNamespace(gate_up_proj=_resident_linear(*_row_major(2 * MI, H, gen, device, glob), 2 * MI, H),
                                        down_proj=_resident_linear(*_row_major(H, MI, gen, device, glob), H, MI))
    moe._router_weight = False
    moe._shared_route = False
    return moe


@pytest.fixture(autouse=True)
def switched_on(monkeypatch):
    """Both switches are opt-in; these tests check the path they turn on."""
    monkeypatch.setattr(qmoe, "_ROUTER_FOLD", True)
    monkeypatch.setattr(qmoe, "_SHARED_ROUTE", True)


@pytest.fixture
def decode(monkeypatch):
    state = {"prefill": False}
    monkeypatch.setattr(qmoe, "get_global_ctx", lambda: _ctx(state["prefill"]))
    monkeypatch.setattr(moe_layers, "get_global_ctx", lambda: _ctx(state["prefill"]))
    return state


# --------------------------------------------------------------------------- which steps take it (no GPU)


@pytest.mark.parametrize("rows,takes", [(0, False), (1, True), (2, True), (6, True), (7, False), (8, False), (16, False)])
def test_shared_route_takes_decode_steps_of_one_to_six_rows(decode, rows, takes):
    moe = _moe()
    moe._router_weight = moe.gate.weight   # the fold's weight, decided
    moe._shared_route = {"gate_up": torch.empty(0)}
    x = torch.zeros(rows, H, dtype=torch.bfloat16)
    assert (moe._shared_route_for(x) is moe._shared_route) is takes


@pytest.mark.parametrize("rows", [7, 8])
def test_shared_route_follows_the_router_folds_row_cap(decode, monkeypatch, rows):
    """The gate rides the fold's launch, so a step the fold does not take keeps today's kernels even where the
    route's own row limit would let it through."""
    moe = _moe()
    moe._router_weight = moe.gate.weight
    moe._shared_route = {"gate_up": torch.empty(0)}
    monkeypatch.setattr(qmoe, "_SHARED_ROUTE_MAX_ROWS", 8)
    assert moe._shared_route_for(torch.zeros(rows, H, dtype=torch.bfloat16)) is None
    assert moe._shared_route_for(torch.zeros(6, H, dtype=torch.bfloat16)) is moe._shared_route


def test_shared_route_leaves_prefill_other_dtypes_the_knob_and_the_unfolded_router_alone(decode, monkeypatch):
    moe = _moe()
    moe._router_weight = moe.gate.weight
    moe._shared_route = route = {"gate_up": torch.empty(0)}
    x = torch.zeros(6, H, dtype=torch.bfloat16)
    assert moe._shared_route_for(x) is route
    decode["prefill"] = True
    assert moe._shared_route_for(x) is None
    decode["prefill"] = False
    assert moe._shared_route_for(x.half()) is None
    assert moe._shared_route_for(torch.zeros(6, 2 * H, dtype=torch.bfloat16)[:, ::2]) is None
    monkeypatch.setattr(qmoe, "_SHARED_ROUTE", False)
    assert moe._shared_route_for(x) is None
    monkeypatch.setattr(qmoe, "_SHARED_ROUTE", True)
    moe._router_weight = None   # the router's linear stays its own launch: no row for the gate to join
    assert moe._shared_route_for(x) is None
    moe._router_weight, moe._shared_route = moe.gate.weight, None   # built and refused once: never again
    assert moe._shared_route_for(x) is None


def test_shared_route_is_built_for_resident_triton_nvfp4_experts_only(monkeypatch):
    moe = _moe()
    route = moe._build_shared_route()
    assert set(route) == set(ROLES)
    for role in ROLES:
        assert route[role].shape == moe.experts._expert_banks[role].shape[1:]
    loaded = _moe()
    loaded.prepare_decode_weights()   # the engine's load makes it, before the KV pool is sized
    assert set(loaded._shared_route) == set(ROLES)
    monkeypatch.setattr(qmoe, "_SHARED_ROUTE", False)
    off = _moe()
    off.prepare_decode_weights()   # the knob off: no copy
    assert off._shared_route is False
    monkeypatch.setattr(qmoe, "_SHARED_ROUTE", True)

    def refused(change):
        m = _moe()
        change(m)
        return m._build_shared_route() is None

    # the shared expert's linears are not on the triton NVFP4 kernel's resident layout, or not on one rank
    assert refused(lambda m: setattr(m.shared_expert.down_proj, "quant_method", SimpleNamespace(kernel=TorchLinearKernel())))
    assert refused(lambda m: setattr(m.shared_expert.gate_up_proj, "quant_method", SimpleNamespace(kernel=TorchLinearKernel())))
    assert refused(lambda m: setattr(m.shared_expert.down_proj, "_tp_size", 2))
    assert refused(lambda m: setattr(m, "experts", _experts(m.experts._expert_banks, cls=OffloadMoELayer)))
    assert refused(lambda m: setattr(m, "experts", _experts(m.experts._expert_banks, kernel=SimpleNamespace(takes_shared_route=False))))
    assert refused(lambda m: setattr(m, "experts", _experts(m.experts._expert_banks, tp_size=2)))
    assert refused(lambda m: setattr(m.experts, "activation", "gelu"))
    assert refused(lambda m: setattr(m.shared_expert_gate, "bias", torch.zeros(1, dtype=torch.bfloat16)))
    assert refused(lambda m: setattr(m.experts, "_expert_banks", None))
    # a bank of another width: the shared expert is not one of its slots
    assert refused(lambda m: m.experts._expert_banks.update(down=torch.zeros(E, H, MI // 4, dtype=torch.uint8)))


def test_shared_route_is_not_built_with_the_router_fold_off(decode, monkeypatch):
    """FREETOKEN_ROUTER_FOLD=0: the route's gate rides the fold's launch, so no step can take it; the engine's load
    makes no copy (none of the KV pool goes to it) and says so once, naming the switch."""
    said = []
    monkeypatch.setattr(qmoe, "_ROUTER_FOLD", False)
    monkeypatch.setattr(qmoe, "_ROUTER_FOLD_SET", "0")
    monkeypatch.setattr(qmoe, "_SHARED_ROUTE", True)
    monkeypatch.setattr(qmoe.logger, "info_once", lambda msg, *a, **k: said.append(msg), raising=False)
    moe = _moe()
    built = []
    monkeypatch.setattr(moe, "_build_shared_route", lambda: built.append(1) or {})
    moe.prepare_decode_weights()
    assert moe._shared_route is None and not built
    assert said == ["shared expert as a ninth route on decode: off (FREETOKEN_ROUTER_FOLD=0)"]
    moe._router_weight = False
    assert moe._shared_route_for(torch.zeros(6, H, dtype=torch.bfloat16)) is None
    assert not built


def test_shared_route_is_not_built_and_says_nothing_with_both_switches_unset(decode, monkeypatch):
    """Nothing set: the load makes no route copy, logs no line, and no decode step takes the route."""
    said = []
    for name, value in (("_ROUTER_FOLD", False), ("_ROUTER_FOLD_SET", ""), ("_SHARED_ROUTE", False), ("_SHARED_ROUTE_SET", "")):
        monkeypatch.setattr(qmoe, name, value)
    monkeypatch.setattr(qmoe.logger, "info_once", lambda msg, *a, **k: said.append(msg), raising=False)
    moe = _moe()
    built = []
    monkeypatch.setattr(moe, "_build_shared_route", lambda: built.append(1) or {})
    moe.prepare_decode_weights()
    assert moe._shared_route is None and not built and said == []
    moe._router_weight = False
    assert moe._shared_route_for(torch.zeros(1, H, dtype=torch.bfloat16)) is None
    assert not built


def test_qwen4_exp_moe_builds_no_route_and_says_nothing_with_both_switches_set(decode, monkeypatch):
    """Qwen4ExpMoE has its own forward, which takes neither the fold nor the route: with both switches on, the load
    still makes no route copy and logs no route line."""
    from freetoken.models.qwen4_exp.moe import Qwen4ExpMoE

    said = []
    monkeypatch.setattr(qmoe.logger, "info_once", lambda msg, *a, **k: said.append(msg), raising=False)
    moe = _moe()
    moe.__class__ = Qwen4ExpMoE
    moe.prepare_decode_weights()
    assert moe._shared_route is None and said == []


def test_shared_route_says_once_a_row_count_that_a_served_step_took_it(decode, monkeypatch):
    """One line a row count a process, from the blocks the engine prepared (the served model's, not a draft's)."""
    from freetoken.kernel.triton import moe_router

    said, seen = [], []
    monkeypatch.setattr(qmoe, "_SHARED_ROUTE_ROWS_SAID", set())
    monkeypatch.setattr(qmoe.logger, "info_rank0", lambda msg, *a, **k: said.append(msg), raising=False)
    monkeypatch.setattr(moe_router, "router_gemv_topk_softmax", lambda x, w, k, renorm, shared_gate=None: (None, None))

    def block(prepared):
        moe = _moe()
        moe._router_weight = moe.gate.weight
        if prepared:
            moe._decode_prepared = True
        moe.experts.routed_forward = lambda x, w, i, shared=None: seen.append(x.shape[0]) or x
        return moe

    served, draft = block(True), block(False)
    route = {"gate_up": torch.empty(0)}
    for moe, rows in ((draft, 6), (served, 6), (served, 6), (draft, 1), (served, 1), (served, 6)):
        moe._forward_shared_route(torch.zeros(rows, H, dtype=torch.bfloat16), route)
    assert seen == [6, 6, 6, 1, 1, 6]
    assert said == ["shared expert as a ninth route on decode: taken at 6 rows",
                    "shared expert as a ninth route on decode: taken at 1 row"]


def test_shared_expert_bank_slot_undoes_the_resident_transpose():
    gen = torch.Generator().manual_seed(3)
    gate_up, down = _row_major(2 * MI, H, gen), _row_major(H, MI, gen)
    route = qmoe.shared_expert_bank_slot(_resident_linear(*gate_up, 2 * MI, H), _resident_linear(*down, H, MI))
    for role, want in zip(ROLES, (*gate_up, *down)):
        assert route[role].dtype == want.dtype and route[role].is_contiguous()
        assert torch.equal(route[role].view(torch.uint8), want.contiguous().view(torch.uint8)), role


# --------------------------------------------------------------------------- the MoE block on CUDA


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
@pytest.mark.parametrize("rows", [1, 2, 6])
def test_shared_route_block_output_is_the_shared_expert_as_one_more_slot_of_the_bank(decode, rows):
    """The MoE block of a decode step through the ninth route against the same routes with the shared expert copied
    into one more slot of the experts' bank and the last ids naming it: the eight routed ids and weights unchanged, the
    output bit for bit equal, finite, and not the routed experts alone."""
    from freetoken.kernel.triton.moe_router import router_gemv_topk_softmax

    moe = _moe(torch.device("cuda"), seed=rows, glob=1.0 / 64)
    gen = torch.Generator(device="cuda").manual_seed(100 + rows)
    x = torch.randn(rows, H, generator=gen, device="cuda").to(torch.bfloat16)

    on = moe.forward(x.clone())
    assert isinstance(moe._shared_route, dict), "the step did not take the ninth route"

    w8, i8 = router_gemv_topk_softmax(x, moe.gate.weight, TOPK, True)
    w9, i9 = router_gemv_topk_softmax(x, moe.gate.weight, TOPK, True, shared_gate=moe.shared_expert_gate.weight)
    assert torch.equal(i9[:, :TOPK], i8) and torch.equal(w9[:, :TOPK].contiguous().view(torch.int32), w8.view(torch.int32))

    banks = moe.experts._expert_banks
    one_more = {role: torch.cat([banks[role], moe._shared_route[role][None]]).contiguous() for role in ROLES}
    ids = i9.clone()
    ids[:, TOPK] = E
    ref = _experts(one_more).routed_forward(x.clone(), w9.contiguous(), ids.contiguous())
    routed = moe.experts.routed_forward(x.clone(), w8, i8.clone())
    assert on.isfinite().all()
    assert torch.equal(on.view(torch.int16), ref.view(torch.int16))
    # the shared expert is in the output: the routed experts alone differ from it
    assert not torch.equal(on.view(torch.int16), routed.view(torch.int16))


@cuda
@pytest.mark.parametrize("rows", [1, 6])
def test_shared_route_block_replays_in_a_cuda_graph(decode, rows):
    """Built at the eager step before the capture, as the graph runner runs one; every replay routes its own input."""
    moe = _moe(torch.device("cuda"), seed=40 + rows, glob=1.0 / 64)
    gen = torch.Generator(device="cuda").manual_seed(41)
    x = torch.randn(rows, H, generator=gen, device="cuda").to(torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        moe.forward(x)
    torch.cuda.current_stream().wait_stream(stream)
    assert isinstance(moe._shared_route, dict)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = moe.forward(x)
    for _ in range(3):
        x.copy_(torch.randn(x.shape, generator=gen, device="cuda").to(torch.bfloat16))
        graph.replay()
        ref = moe.forward(x.clone())
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
