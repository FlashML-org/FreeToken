"""Load-time per-tensor FP8 for the qwen4_exp dense projections (FREETOKEN_FP8_DENSE=1)."""

from types import SimpleNamespace

import pytest
import torch
from freetoken.models.qwen4_exp.weight import _fp8_dense, _quantize_per_tensor

from .common import requires_cuda

E4M3_MAX = 448.0


def _cfg():
    g = SimpleNamespace(
        num_key_heads=4, key_head_dim=8, num_value_heads=12, value_head_dim=8
    )
    return SimpleNamespace(linear_attention_group=lambda: g)


def _assert_e4m3_close(
    deq: torch.Tensor, ref: torch.Tensor, scale: torch.Tensor
) -> None:
    # e4m3 keeps 3 mantissa bits (rel 2^-4); below scale * 2^-6 it is subnormal (abs 2^-9 steps)
    tol = ref.abs() * 2**-4 + float(scale) * 2**-9 + 1e-7
    assert ((deq - ref).abs() <= tol).all()


def test_quantize_per_tensor_round_trips_within_e4m3():
    w = torch.randn(64, 32) * 0.02
    w8, scale = _quantize_per_tensor(w)
    assert (
        w8.dtype == torch.float8_e4m3fn
        and scale.dtype == torch.float32
        and scale.shape == ()
    )
    assert torch.isclose(scale * E4M3_MAX, w.abs().max())
    _assert_e4m3_close(w8.float() * scale, w, scale)


def test_in_proj_splits_into_fp8_qkvz_and_bf16_ba_per_rank():
    cfg, world = (
        _cfg(),
        2,
    )  # local: 2 k heads, 6 v heads -> qkvz = 2*2*8 + 2*6*8 = 128, ba = 12
    t = torch.randn(128 + 12, 16, dtype=torch.bfloat16)
    out = dict(_fp8_dense("model.layers.3.linear_attn.in_proj.weight", t, cfg, world))
    assert sorted(out) == [
        "model.layers.3.linear_attn.in_proj_ba.weight",
        "model.layers.3.linear_attn.in_proj_qkvz.weight",
        "model.layers.3.linear_attn.in_proj_qkvz.weight_scale",
    ]
    w8 = out["model.layers.3.linear_attn.in_proj_qkvz.weight"]
    scale = out["model.layers.3.linear_attn.in_proj_qkvz.weight_scale"]
    assert w8.shape == (128, 16) and w8.dtype == torch.float8_e4m3fn
    _assert_e4m3_close(w8.float() * scale, t[:128].float(), scale)
    ba = out["model.layers.3.linear_attn.in_proj_ba.weight"]
    assert ba.dtype == torch.bfloat16 and torch.equal(ba, t[128:])
    # its own storage: a view would keep the whole bf16 in_proj alive next to the fp8 copy
    assert ba.untyped_storage().data_ptr() != t.untyped_storage().data_ptr()


def test_other_projections_gain_a_scale_and_the_rest_pass_through():
    cfg = _cfg()
    t = torch.randn(32, 16, dtype=torch.bfloat16)
    for name in (
        "x.self_attn.qkv_proj.weight",
        "x.self_attn.o_proj.weight",
        "x.linear_attn.out_proj.weight",
    ):
        out = dict(_fp8_dense(name, t, cfg, 1))
        assert sorted(out) == [name, name[: -len("weight")] + "weight_scale"]
        assert out[name].dtype == torch.float8_e4m3fn
    out = dict(_fp8_dense("x.mlp.shared_expert.gate_up_proj.weight", t, cfg, 1))
    assert (
        list(out) == ["x.mlp.shared_expert.gate_up_proj.weight"]
        and out.popitem()[1] is t
    )


def test_ops_declare_fp8_weight_and_scalar_scale():
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.fp8_dynamic import Fp8DynamicColMerged, Fp8DynamicRowParallel

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    col = Fp8DynamicColMerged(32, [64, 16, 16])
    row = Fp8DynamicRowParallel(64, 32)
    for op in (col, row):
        sd = op.state_dict()
        assert set(sd) == {"weight", "weight_scale"}
        assert (
            sd["weight"].dtype == torch.float8_e4m3fn and sd["weight_scale"].shape == ()
        )
    assert col.weight.shape == (96, 32) and row.weight.shape == (32, 64)


@requires_cuda
@pytest.mark.parametrize("rows", [1, 16, 300], ids=["decode-1", "decode-16", "prefill"])
def test_fp8_linear_matches_bf16_on_the_dequantized_weight(rows: int):
    from freetoken.layers.fp8_dynamic import fp8_dynamic_linear

    torch.manual_seed(0)
    w = (torch.randn(256, 128, device="cuda") * 0.02).to(torch.bfloat16)
    w8, scale = _quantize_per_tensor(w)
    x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x, (w8.float() * scale).to(torch.bfloat16))
    got = fp8_dynamic_linear(x, w8, scale.to("cuda"))
    assert got.shape == ref.shape and got.dtype == torch.bfloat16
    # the per-tensor activation cast is the only extra rounding: 2^-4 relative on the inputs
    torch.testing.assert_close(
        got.float(), ref.float(), rtol=0.1, atol=0.08 * ref.abs().max().item()
    )


@requires_cuda
def test_quant_per_tensor_zero_input_is_finite():
    from freetoken.layers.fp8_dynamic import quant_per_tensor

    x8, scale = quant_per_tensor(
        torch.zeros(16, 128, device="cuda", dtype=torch.bfloat16)
    )
    assert torch.isfinite(scale).item() and (x8.float() == 0).all()
