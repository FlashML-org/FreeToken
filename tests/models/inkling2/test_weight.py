"""The expert reader against the Hugging Face dequantization (``Inkling2LoopDequantizeExperts`` then
``Interleave(dim=1)``) on a synthetic checkpoint with one NVFP4 and one MXFP8 layer: gate/up rows
interleaved in the checkpoint, NVFP4 global ``amax / outer_scale_denominator``, MXFP8 ue8m0 scales."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from freetoken.kernel.triton.mxfp8_linear import mxfp8_dequant  # noqa: E402
from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4  # noqa: E402
from freetoken.layers.quantization import QuantKind  # noqa: E402
from freetoken.layers.quantization.moe.base import fused_global, fused_piece, global_rows  # noqa: E402
from freetoken.models.inkling2.config import parse_config  # noqa: E402
from freetoken.models.inkling2.weight import iter_expert_pieces  # noqa: E402

E, H, I = 20, 64, 32
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _hf_config():
    text = SimpleNamespace(
        num_hidden_layers=12, hidden_size=H, num_attention_heads=2, num_key_value_heads=1, head_dim=32, rms_norm_eps=1e-6,
        num_experts_per_tok=2, intermediate_size=I, vocab_size=256, n_routed_experts=E,
        loop_config={"core_start": 6, "core_end": 12, "num_passes": 2},
    )
    return SimpleNamespace(architectures=["Inkling2LoopForConditionalGeneration"], model_type="inkling2_loop", text_config=text)


def _hf_dequant(weight, scale, amax=None, denom=None):
    if amax is None:
        values = weight.float().unflatten(-1, (-1, 32)) * torch.exp2(scale.float() - 127)[..., None]
    else:
        codes = weight.long()
        pairs = torch.stack([E2M1[codes & 0xF], E2M1[codes >> 4]], dim=-1).flatten(-2)
        values = pairs.unflatten(-1, (-1, 16)) * scale.float()[..., None] * (amax / denom)[:, None, None, None]
    return values.flatten(-2).bfloat16()


def _hf_interleave(w):
    return w.unflatten(1, (-1, 2)).transpose(1, 2).reshape(w.shape)


def _checkpoint(tmp_path):
    import safetensors.torch

    g = torch.Generator().manual_seed(0)
    t = {}
    p = "model.llm.layers.0.mlp.experts."
    for proj, rows, cols in (("w13", 2 * I, H), ("w2", H, I)):
        t[p + f"{proj}_weight"] = torch.randint(0, 256, (E, rows, cols // 2), generator=g, dtype=torch.uint8)
        t[p + f"{proj}_weight_scale"] = (torch.rand(E, rows, cols // 16, generator=g) * 2 + 0.25).to(torch.float8_e4m3fn)
        t[p + f"{proj}_weight_amax"] = torch.rand(E, generator=g) * 3 + 0.5
        t[p + f"{proj}_weight_outer_scale_denominator"] = torch.tensor([2688.0])
    p = "model.llm.layers.1.mlp.experts."
    for proj, rows, cols in (("w13", 2 * I, H), ("w2", H, I)):
        t[p + f"{proj}_weight"] = torch.randn(E, rows, cols, generator=g).mul(40).to(torch.float8_e4m3fn)
        t[p + f"{proj}_weight_scale"] = torch.randint(118, 132, (E, rows, cols // 32), generator=g, dtype=torch.uint8)
    safetensors.torch.save_file(t, str(tmp_path / "model.safetensors"))
    return t


def test_expert_pieces_dequantize_like_the_reference(tmp_path):
    t = _checkpoint(tmp_path)
    config = parse_config(_hf_config())

    ((layer, e0, e1, piece),) = list(iter_expert_pieces(str(tmp_path), config, QuantKind.NVFP4, layers=[0]))
    assert (layer, e0, e1) == (0, 0, E)
    p = "model.llm.layers.0.mlp.experts."
    slots = torch.arange(E, dtype=torch.int32, device="cuda")
    for proj, role, globals_ in (("w13", "gate_up", fused_global(piece, I)), ("w2", "down", global_rows(piece["down_global"], H))):
        got = dequant_nvfp4(fused_piece(piece, role).cuda(), fused_piece(piece, f"{role}_scale").cuda(), globals_.contiguous().cuda(), slots)
        want = _hf_dequant(*(t[p + f"{proj}_weight{s}"] for s in ("", "_scale", "_amax", "_outer_scale_denominator")))
        if proj == "w13":
            want = _hf_interleave(want)
        torch.testing.assert_close(got.cpu().float(), want.float(), atol=0, rtol=1e-2)

    ((layer, e0, e1, piece),) = list(iter_expert_pieces(str(tmp_path), config, QuantKind.MXFP8, layers=[1]))
    assert (layer, e0, e1) == (1, 0, E)
    p = "model.llm.layers.1.mlp.experts."
    for proj, role in (("w13", "gate_up"), ("w2", "down")):
        got = mxfp8_dequant(piece[role].flatten(0, 1), piece[f"{role}_scale"].flatten(0, 1), torch.float32).view(piece[role].shape)
        want = _hf_dequant(t[p + f"{proj}_weight"], t[p + f"{proj}_weight_scale"])
        if proj == "w13":
            want = _hf_interleave(want)
        assert torch.equal(got, want.float())
