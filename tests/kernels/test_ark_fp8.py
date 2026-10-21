"""Native decode, fallback, bank and stream contracts against independent FT paths."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from benchmarks.ark.cases import rand, quant
from benchmarks.ark.timing import metrics, assert_close

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip,
    reason="requires ROCm gfx1201",
)


@pytest.fixture(autouse=True)
def native(monkeypatch):
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1201"):
        pytest.skip("requires gfx1201")
    from freetoken.kernel.ark.common import policy
    from freetoken.kernel.ark import initialize

    monkeypatch.setenv("FREETOKEN_USE_ARK", "1")
    monkeypatch.setattr(policy, "MODE", "native")
    initialize()
    torch.manual_seed(20261021)


def compare(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
    assert_close(metrics(a, b), "decode")


def moe_inputs(m=1):
    x = rand(m, 256)
    w1, s1 = quant(rand(4, 256, 256, scale=1 / 16))
    w2, s2 = quant(rand(4, 256, 128, scale=128**-0.5))
    s1 = torch.cat((s1, torch.full((4, 2, 3), 19.0, device="cuda", dtype=s1.dtype)), -1)
    s2 = torch.cat((s2, torch.full((4, 2, 2), 23.0, device="cuda", dtype=s2.dtype)), -1)
    weights = torch.tensor([[0.2, 0.3, 0.0, 0.5]], device="cuda").expand(m, 4).contiguous()
    ids = torch.tensor([[1, 1, 0, 3]], device="cuda", dtype=torch.int32).expand(m, 4).contiguous()
    return x, w1, s1, w2, s2, weights, ids


@pytest.mark.parametrize("value", [None, 0.0, 1e-12])
@pytest.mark.parametrize("bias", [False, True])
def test_dense_bias_padding_alias(value, bias, monkeypatch):
    from freetoken.kernel.ark.dense_fp8_decode import adapter
    from freetoken.kernel.triton.fp8_block_linear import block_fp8_linear as baseline

    x = rand(1, 1, 384) if value is None else torch.full((1, 1, 384), value, device="cuda", dtype=torch.bfloat16)
    initial = x.clone()
    w, scale = quant(rand(384, 384))
    scale = torch.cat((scale, torch.full((3, 2), 17.0, device="cuda", dtype=scale.dtype)), -1)
    b = rand(384, dtype=torch.float32) if bias else None
    hits = []
    monkeypatch.setattr(adapter, "record", lambda *a: hits.append(a))
    actual = adapter.block_fp8_linear(x, w, scale, b)
    compare(actual, baseline(x, w, scale, b))
    assert torch.equal(x, initial) and actual.data_ptr() != x.data_ptr()
    assert hits[-1][-1] == "ark"


@pytest.mark.parametrize("kind", ["batch", "strided", "disabled", "unknown_shape"])
def test_dense_fallback(kind, monkeypatch):
    from freetoken.kernel.ark.dense_fp8_decode import adapter
    from freetoken.kernel.ark.common import policy
    from freetoken.kernel.triton.fp8_block_linear import block_fp8_linear as baseline

    x = rand(2 if kind == "batch" else 1, 256)
    if kind == "strided":
        x = rand(1, 512)[:, ::2]
    w, scale = quant(rand(256, 256))
    if kind == "disabled":
        monkeypatch.setenv("FREETOKEN_USE_ARK", "0")
    if kind == "unknown_shape":
        monkeypatch.setattr(policy, "MODE", "validated")
        monkeypatch.setattr(policy, "environment_matches", lambda: True)
        monkeypatch.setattr(policy, "_profiles", {})
    hits = []
    monkeypatch.setattr(adapter, "record", lambda *a: hits.append(a))
    compare(adapter.block_fp8_linear(x, w, scale), baseline(x, w, scale))
    assert hits[-1][-1] == "freetoken"


def test_moe_routes_zero_inactive_and_input_unchanged():
    from freetoken.kernel.ark.moe_decode_fp8.adapter import fused_experts_decode_fp8_block as candidate
    from freetoken.moe.fused_fp8_block import fused_experts_decode_fp8_block as baseline

    args = moe_inputs()
    original = [v.clone() for v in args]
    expected = baseline(*args)
    actual = candidate(*args)
    compare(actual, expected)
    assert actual.data_ptr() != args[0].data_ptr()
    for a, b in zip(args, original):
        assert torch.equal(a, b)
    args[-1][0, 2] = -1
    compare(candidate(*args), expected)
    args[-2].zero_()
    assert candidate(*args).count_nonzero().item() == 0


@pytest.mark.parametrize("kind", ["batch", "gelu", "disabled", "graph_excluded"])
def test_moe_fallback(kind, monkeypatch):
    from freetoken.kernel.ark.moe_decode_fp8 import adapter
    from freetoken.kernel.ark.common import policy
    from freetoken.moe.fused_fp8_block import fused_experts_decode_fp8_block as baseline

    args = moe_inputs(2 if kind == "batch" else 1)
    kwargs = {"activation": "gelu"} if kind == "gelu" else {}
    if kind == "disabled":
        monkeypatch.setenv("FREETOKEN_USE_ARK", "0")
    if kind == "graph_excluded":
        monkeypatch.setattr(policy, "MODE", "validated")
        monkeypatch.setattr(policy, "environment_matches", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    hits = []
    monkeypatch.setattr(adapter, "record", lambda *a: hits.append(a))
    compare(adapter.fused_experts_decode_fp8_block(*args, **kwargs), baseline(*args, **kwargs))
    assert hits[-1][-1] == "freetoken"


@pytest.mark.parametrize("op", ["dense", "moe"])
def test_nondefault_stream_graph_and_updated_inputs(op):
    if op == "dense":
        from freetoken.kernel.ark.dense_fp8_decode.adapter import block_fp8_linear as candidate
        from freetoken.kernel.triton.fp8_block_linear import block_fp8_linear as baseline

        w, scale = quant(rand(384, 256))
        args = (rand(1, 256), w, scale)
    else:
        from freetoken.kernel.ark.moe_decode_fp8.adapter import fused_experts_decode_fp8_block as candidate
        from freetoken.moe.fused_fp8_block import fused_experts_decode_fp8_block as baseline

        args = moe_inputs()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        candidate(*args)
        baseline(*args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = candidate(*args)
        for step in range(3):
            args[0].copy_(rand(*args[0].shape))
            if op == "moe":
                args[-1].copy_(torch.tensor([[3, 1, 1, 0]], device="cuda", dtype=torch.int32))
                args[-2].copy_(torch.tensor([[0.0, 0.3, 0.3, 0.4]], device="cuda"))
            expected = baseline(*args).clone()
            graph.replay()
            compare(actual, expected)
    stream.synchronize()


@pytest.mark.parametrize("resident", [False, True])
def test_real_method_bank_pack_apply_prefill(resident, monkeypatch):
    from freetoken.kernel.ark.kernels import ArkFp8MoEKernel, ArkFp8LinearKernel
    from freetoken.layers.quantization.moe.base import MoEConfig, ExpertView
    from freetoken.layers.quantization.moe.fp8_block import Fp8BlockMoEMethod, TritonFp8BlockMoEKernel
    from freetoken.layers.quantization.linear.base import LinearConfig
    from freetoken.layers.quantization.linear.fp8_block import Fp8BlockLinearMethod, TritonFp8BlockLinearKernel
    from freetoken.layers.quantization.scheme import fp8_block_scheme

    scheme = fp8_block_scheme("bf16")
    cfg = MoEConfig(4, 256, 128, 4, scheme=scheme)
    candidate, baseline = ArkFp8MoEKernel(), TritonFp8BlockMoEKernel()
    assert candidate.layout(cfg) == baseline.layout(cfg)
    assert candidate.cpu_format == baseline.cpu_format and not candidate.supports_inactive_slots
    x, gu, gs, dn, ds, weights, ids = moe_inputs()
    pieces = dict(gate_up=gu, gate_up_scale=gs[:, :, :2], down=dn, down_scale=ds[:, :, :1])
    specs = candidate.layout(cfg)
    bank = {k: torch.zeros(4, *v.shape, device="cuda", dtype=v.dtype) for k, v in specs.items()}
    ref_bank = {k: torch.zeros_like(v) for k, v in bank.items()}
    assert candidate.pack(pieces, cfg, bank) == baseline.pack(pieces, cfg, ref_bank)
    for k in bank:
        assert torch.equal(bank[k], ref_bank[k])
    view = ExpertView(bank, n=4)
    layer = SimpleNamespace(
        activation="silu",
        alpha=1.0,
        limit=None,
        num_experts=4,
        apply_router_weight_on_input=False,
        quant_method=SimpleNamespace(cfg=SimpleNamespace(strategy="resident" if resident else "offload")),
    )
    for prefill in (False, True):
        m = 33 if prefill else 1
        x = rand(m, 256)
        initial = x.clone()
        tw = weights.expand(m, 4).contiguous()
        idx = ids.expand(m, 4).contiguous()
        compare(candidate.apply(layer, x, tw, idx, view, is_prefill=prefill), baseline.apply(layer, x, tw, idx, view, is_prefill=prefill))
        assert torch.equal(x, initial)
    assert isinstance(Fp8BlockMoEMethod(cfg).kernel, ArkFp8MoEKernel)
    linear_cfg = LinearConfig(256, 384, scheme=scheme)
    assert isinstance(Fp8BlockLinearMethod(linear_cfg).kernel, ArkFp8LinearKernel)
    assert type(Fp8BlockMoEMethod(cfg, requested="triton").kernel) is TritonFp8BlockMoEKernel
    monkeypatch.setenv("FREETOKEN_USE_ARK", "0")
    assert type(Fp8BlockMoEMethod(cfg).kernel) is TritonFp8BlockMoEKernel
    assert type(Fp8BlockLinearMethod(linear_cfg).kernel) is TritonFp8BlockLinearKernel


def test_ftw_identity_and_auto_selection(monkeypatch):
    from freetoken.engine.engine import _adjust_ftw_quant_backend
    from freetoken.layers.quantization import QuantBackend, LayerKind, QuantKind
    from freetoken.moe.legacy_format import legacy_format_for, kind_kernel_for, bank_kernel_compatible
    import freetoken.checkpoint.ftw as ftw

    monkeypatch.setattr(ftw, "ftw_quant_format", lambda _: "fp8_block")
    kind = QuantKind.FP8_BLOCK
    auto = QuantBackend()
    assert _adjust_ftw_quant_backend("/unused", auto).select(LayerKind.MOE, kind) == "auto"
    assert legacy_format_for(kind, "ark_r9700") == "fp8_block"
    assert kind_kernel_for("fp8_block") == (kind, "triton")
    assert bank_kernel_compatible(kind, "triton", "ark_r9700")
    assert not bank_kernel_compatible(QuantKind.NONE, "fused", "ark_r9700")
    assert not bank_kernel_compatible(kind, "unknown", "ark_r9700")
    monkeypatch.setenv("FREETOKEN_USE_ARK", "0")
    assert _adjust_ftw_quant_backend("/unused", auto).select(LayerKind.MOE, kind) == "triton"


def test_policy_mismatch_default_and_no_builder_import(monkeypatch):
    from freetoken.kernel.ark.common import policy

    monkeypatch.setattr(policy, "MODE", "validated")
    monkeypatch.setattr(policy, "_profiles", {"_environment": {"torch": "mismatch"}})
    policy.environment_matches.cache_clear()
    assert not policy.environment_matches()
    assert not policy.allowed("dense_fp8_decode", "1,5120,2048")
    policy.environment_matches.cache_clear()
    env = dict(os.environ, FREETOKEN_USE_ARK="0")
    script = "import sys; import freetoken.kernel.ark as a; assert not a.enabled(); assert 'freetoken.kernel.ark.common.native' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], env=env, check=True)


def test_finite_fp8_codes():
    from freetoken.kernel.ark.dense_fp8_decode.adapter import block_fp8_linear

    codes = torch.arange(256, device="cuda", dtype=torch.int32).to(torch.uint8)
    codes[127] = codes[255] = 0
    storage = torch.zeros(256, 128, device="cuda", dtype=torch.uint8)
    storage[:, 0] = codes
    x = torch.zeros(1, 128, device="cuda", dtype=torch.bfloat16)
    x[0, 0] = 1
    actual = block_fp8_linear(x, storage.view(torch.float8_e4m3fn), torch.ones(2, 1, device="cuda", dtype=torch.bfloat16))
    assert torch.equal(actual, codes.view(torch.float8_e4m3fn).bfloat16().reshape(1, 256))


def test_packaged_policy_keeps_graph_fallback(monkeypatch):
    from freetoken.kernel.ark.common import policy

    monkeypatch.setattr(policy, "MODE", "validated")
    monkeypatch.setattr(policy, "environment_matches", lambda: True)
    for op in ("dense_fp8_decode", "moe_decode_fp8"):
        assert not policy._profiles[op]["graph"]
        assert policy._profiles[op]["eager"]
        for shape in policy._profiles[op]["eager"]:
            assert policy.allowed(op, shape, capturing=False)
            assert not policy.allowed(op, shape, capturing=True)
