"""qwen3_5_moe dense-pass normalization for mixed-precision NVFP4 exports.

unsloth/Qwen3.8-27B-NVFP4 keeps some dense projections as weight-only FP8 instead of
bf16. The dense pass used to feed the raw e4m3 tensor straight into ``ct_bf16_fuse``,
whose ``torch.cat`` dies on fp8/bf16 promotion:

    RuntimeError: Promotion for Float8 Types is not supported, attempted to promote
    Float8_e4m3fn and BFloat16

The pass now keeps scaled fp8 linears native where the model built fp8 linears (W8A16:
fp8 weight + per-row fp32 scale, fused q/k/v -> qkv_proj, in_proj_qkv/z -> in_proj_qkvz,
gate/up -> gate_up_proj, o_proj / GDN out_proj / down_proj / lm_head singletons) and
dequantizes the remainder to bf16 first (W8A16 dequant with a per-tensor scale,
broadcast with a block scale, exact cast without a scale).
"""

import torch

from freetoken.models.loader import ct_bf16_fuse
from freetoken.models.qwen3_5_moe.weight import (
    _CT_BF16_FUSE,
    _CT_NVFP4_FUSE,
    _FP8_DTYPES,
    _fp8_weight_to_bf16,
)


def test_fp8_with_per_tensor_scale_dequants():
    w = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32).to(torch.float8_e4m3fn)
    scale = torch.tensor(0.25)
    out = _fp8_weight_to_bf16(w, scale)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, w.to(torch.bfloat16) * scale.to(torch.bfloat16))


def test_fp8_with_scalar_shape1_scale_dequants():
    w = torch.tensor([[1.0, 2.0]], dtype=torch.float32).to(torch.float8_e4m3fn)
    out = _fp8_weight_to_bf16(w, torch.tensor([2.0]))
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, w.to(torch.bfloat16) * torch.tensor([2.0]).to(torch.bfloat16))


def test_fp8_without_scale_is_exact_cast():
    w = torch.tensor([1.0, 2.0, 3.5], dtype=torch.float32).to(torch.float8_e4m3fn)
    out = _fp8_weight_to_bf16(w, None)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, w.to(torch.bfloat16))


def test_fp8_with_block_scale_broadcasts():
    # [O, IN] weight, block scale [O, IN//2] -> group size 2. fp8 values and the scale
    # are exact in bf16, so the bf16 multiply rounds once (same result as the fp32
    # product before its final cast) without a materialized fp32 copy.
    w = torch.arange(8, dtype=torch.float32).reshape(2, 4).to(torch.float8_e4m3fn)
    s = torch.tensor([[1.0, 2.0], [0.5, 4.0]], dtype=torch.bfloat16)
    out = _fp8_weight_to_bf16(w, s)
    assert out.shape == (2, 4)
    assert out.dtype == torch.bfloat16
    expected = w.to(torch.bfloat16) * s.to(torch.bfloat16).repeat_interleave(2, dim=1)
    assert torch.equal(out, expected)


def test_fp8_with_per_row_scale_broadcasts():
    # unsloth's mixed exports store a per-row [O, 1] scale: pure broadcast multiply
    # (no repeat materialized -- a per-row lm_head scale would otherwise transiently
    # double the memory of the dequant).
    w = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    s = torch.tensor([[0.5], [2.0]], dtype=torch.bfloat16)
    out = _fp8_weight_to_bf16(w, s)
    assert out.shape == (2, 4)
    assert out.dtype == torch.bfloat16
    expected = w.to(torch.bfloat16) * s.to(torch.bfloat16)
    assert torch.equal(out, expected)


def test_bf16_tensor_untouched_by_dtype_gate():
    """The call site only normalizes fp8 dtypes; bf16 parts flow through as-is."""
    w = torch.randn(4, 4, dtype=torch.bfloat16)
    assert w.dtype not in _FP8_DTYPES


def test_mixed_group_bf16_fusion_repro():
    """Regression: one fp8 part + bf16 parts in the GDN ``in_proj`` group. Without the
    normalization the group's torch.cat raises 'Promotion for Float8 Types is not
    supported'; with it, the fused ``in_proj.weight`` is bf16 and output-dim-concatenated
    in the canonical (qkv, z, b, a) order."""
    O, IN = 8, 4
    # All parts share the input dim (last dim) and differ only in output rows, like a
    # real in_proj group.
    parts = (
        (".linear_attn.in_proj_qkv", torch.randn(O, IN).to(torch.float8_e4m3fn)),
        (".linear_attn.in_proj_z", torch.randn(O, IN).to(torch.bfloat16)),
        (".linear_attn.in_proj_b", torch.randn(O, IN).to(torch.bfloat16)),
        (".linear_attn.in_proj_a", torch.randn(O, IN).to(torch.bfloat16)),
    )
    base = "model.layers.0"
    buf: dict = {}
    completed: list[tuple[str, torch.Tensor]] = []
    for part, t in parts:
        # The dense pass normalizes fp8 parts before fusing (the fix under test).
        if t.dtype in _FP8_DTYPES:
            t = _fp8_weight_to_bf16(t, None)
        emit = ct_bf16_fuse(base + part, t, buf, _CT_BF16_FUSE)
        if emit:
            completed.extend(emit)
    assert len(completed) == 1
    (key, fused), = completed
    assert key == base + ".linear_attn.in_proj.weight"
    assert fused.dtype == torch.bfloat16
    assert fused.shape == (O * 4, IN)
    assert not buf


def test_fp8_qkv_parts_fuse_to_qkv_proj():
    """unsloth's self_attn q/k/v are weight-only FP8: after normalization they must fuse
    into qkv_proj through the same bf16 fusion map the NVFP4 dequant branch uses.
    The model builds a bf16 qkv linear for mixed layouts (dense pass emits qkv_proj)."""
    O, IN = 4, 4
    base = "model.layers.0.self_attn"
    buf: dict = {}
    completed: list[tuple[str, torch.Tensor]] = []
    for part in (".q_proj", ".k_proj", ".v_proj"):
        t = _fp8_weight_to_bf16(torch.randn(O, IN).to(torch.float8_e4m3fn), None)
        emit = ct_bf16_fuse(base + part, t, buf, _CT_NVFP4_FUSE)
        if emit:
            completed.extend(emit)
    assert len(completed) == 1
    (key, fused), = completed
    assert key == base + ".qkv_proj.weight"
    assert fused.dtype == torch.bfloat16
    assert fused.shape == (O * 3, IN)
    assert not buf


def test_fp8_gate_up_parts_fuse_to_gate_up_proj():
    """unsloth stores some dense-MLP layers' gate/up as weight-only FP8: both must fuse
    into gate_up_proj (the model builds a bf16 gate_up linear for mixed layouts)."""
    O, IN = 6, 4
    base = "model.layers.0.mlp"
    buf: dict = {}
    completed: list[tuple[str, torch.Tensor]] = []
    for part in (".gate_proj", ".up_proj"):
        t = _fp8_weight_to_bf16(torch.randn(O, IN).to(torch.float8_e4m3fn), None)
        emit = ct_bf16_fuse(base + part, t, buf, _CT_NVFP4_FUSE)
        if emit:
            completed.extend(emit)
    assert len(completed) == 1
    (key, fused), = completed
    assert key == base + ".gate_up_proj.weight"
    assert fused.shape == (O * 2, IN)
    assert not buf


def test_fp8_o_proj_not_fused_stays_standalone():
    """o_proj / out_proj are singletons in every layout: the qkv/gate_up fusion map must
    not swallow them (they pass through as bare .weight)."""
    for part in (".o_proj", ".out_proj"):
        t = _fp8_weight_to_bf16(torch.randn(4, 4).to(torch.float8_e4m3fn), None)
        buf: dict = {}
        assert ct_bf16_fuse("model.layers.0" + part, t, buf, _CT_NVFP4_FUSE) is None
        assert not buf


def test_per_row_scale_used_verbatim():
    from freetoken.models.qwen3_5_moe.weight import _per_row_scale

    s = torch.tensor([[0.25], [0.5]], dtype=torch.bfloat16)
    out = _per_row_scale(s, 2)
    assert out.dtype == torch.float32
    assert out.shape == (2,)
    assert torch.equal(out, torch.tensor([0.25, 0.5], dtype=torch.float32))


def test_scalar_scale_still_broadcasts():
    from freetoken.models.qwen3_5_moe.weight import _per_row_scale

    out = _per_row_scale(torch.tensor([2.0]), 3)
    assert torch.equal(out, torch.full((3,), 2.0, dtype=torch.float32))


def test_fp8_mlp_gate_up_fusion_local_map():
    """The mixed layout extends the fp8 fusion map locally (dense-MLP gate/up stay native
    W8A16 instead of dequantizing to bf16): fp8 weights + concatenated per-row fp32
    scales, no input_scale (acts are None)."""
    from freetoken.models.qwen3_5_moe.weight import _PT_FP8_FUSE, _pt_fp8_fuse

    O, IN = 4, 8
    base = "model.layers.56.mlp"
    local_map = {**_PT_FP8_FUSE, ".mlp.gate_up_proj": (".mlp.gate_proj", ".mlp.up_proj")}
    w = {p: torch.randn(O, IN).to(torch.float8_e4m3fn) for p in (".gate_proj", ".up_proj")}
    s = {p: torch.full((O, 1), 0.5, dtype=torch.bfloat16) for p in w}
    buf: dict = {}
    completed: list[tuple[str, torch.Tensor]] = []
    for part in (".gate_proj", ".up_proj"):
        emit = _pt_fp8_fuse(base + part, w[part], s[part], None, buf, local_map)
        if emit:
            completed.extend(emit)
    assert len(completed) == 2  # weight + weight_scale (no input_scale: acts are None)
    (key_w, fused_w), (key_s, fused_s) = completed
    assert key_w == base + ".gate_up_proj.weight"
    assert fused_w.dtype == torch.float8_e4m3fn
    assert fused_w.shape == (O * 2, IN)
    assert key_s == base + ".gate_up_proj.weight_scale"
    assert fused_s.dtype == torch.float32
    assert fused_s.shape == (O * 2,)
    assert not buf


def test_fp8_native_qkv_fusion_per_row_scales():
    """unsloth's attention: fp8 q/k/v with per-row [O, 1] scales stay native FP8 (W8A16)
    -- the fusion concatenates the fp8 weights and the per-row fp32 scales, no dequant."""
    from freetoken.models.qwen3_5_moe.weight import _pt_fp8_fuse

    O, IN = 4, 8
    base = "model.layers.3.self_attn"
    wq = torch.randn(O, IN).to(torch.float8_e4m3fn)
    wk = torch.randn(O, IN).to(torch.float8_e4m3fn)
    wv = torch.randn(O, IN).to(torch.float8_e4m3fn)
    sq = torch.tensor([[0.25], [0.5], [1.0], [0.125]], dtype=torch.bfloat16)
    sk = torch.tensor([[0.5], [0.25], [2.0], [1.0]], dtype=torch.bfloat16)
    sv = torch.tensor([[1.0], [1.0], [1.0], [1.0]], dtype=torch.bfloat16)
    buf: dict = {}
    completed: list[tuple[str, torch.Tensor]] = []
    for part, w, s in (".q_proj", wq, sq), (".k_proj", wk, sk), (".v_proj", wv, sv):
        emit = _pt_fp8_fuse(base + part, w, s, None, buf)
        if emit:
            completed.extend(emit)
    assert len(completed) == 2  # weight + weight_scale (no input_scale: acts are None)
    (key_w, w), (key_s, s) = completed
    assert key_w == base + ".qkv_proj.weight"
    assert w.dtype == torch.float8_e4m3fn
    assert w.shape == (O * 3, IN)
    assert key_s == base + ".qkv_proj.weight_scale"
    assert s.dtype == torch.float32
    assert s.shape == (O * 3,)
    expected = torch.cat(
        [sq.reshape(-1).float(), sk.reshape(-1).float(), sv.reshape(-1).float()]
    )
    assert torch.equal(s, expected)
    assert not buf
