"""Per-tensor FP8 linear: the W8A16 kernels against the dequant reference, and the W8A8
(torch._scaled_mm) path against a W8A8 reference.

Which of the two runs is fixed by the deployment -- an ``input_scale`` in the checkpoint plus
sm_89+ -- and never by M, so each is checked against the reference matching its own contract:
W8A16 keeps the activation in bf16 and matches the exact dequant reference, W8A8 quantizes it
to fp8 and is held to a reference that applies the same quantization.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from freetoken.kernel.triton.e4m3_compat import e4m3_native

DEV = "cuda"
FP8 = torch.float8_e4m3fn


def _quant_parts(part_rows: list[int], K: int, seed: int = 0):
    """A fused per-tensor-FP8 weight: each part quantized under its own scalar, exactly how
    modelopt stores q/k/v (and how the loader concatenates them into a per-row vector)."""
    torch.manual_seed(seed)
    N = sum(part_rows)
    wf = torch.randn(N, K, device=DEV) * 0.05
    rows, qs, scales = 0, [], []
    for p in part_rows:
        block = wf[rows : rows + p]
        s = block.abs().max() / 448.0
        qs.append((block / s).clamp(-448, 448).to(FP8))
        scales.append(s.expand(p))
        rows += p
    return torch.cat(qs, 0), torch.cat(scales).contiguous().float()


def _dequant(w8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return w8.to(torch.float32) * scale[:, None]


# 1 = split-K GEMV; 2..16 = decode batch (the CUDA-graph ladder); 64/300 = prefill.
@pytest.mark.parametrize("M", [1, 2, 4, 8, 16, 64, 300])
# (5120, 14336) and (5120, 16384) are Qwen3.5-27B's fused qkv_proj / in_proj_qkvz;
# (1024, 6144) is a standalone o_proj shape; 6112 leaves a k-mask tail.
@pytest.mark.parametrize("K,part_rows", [
    (5120, [12288, 1024, 1024]),
    (1024, [6144]),
    (6112, [512, 128]),
])
def test_w8a16_matches_dequant_reference(M: int, K: int, part_rows: list[int]):
    """Without an ``input_scale`` every M stays on the W8A16 kernels, activation exact."""
    from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

    w8, scale = _quant_parts(part_rows, K, seed=M)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    y = fp8_pertensor_linear(x, w8, scale)
    y_ref = (x.float() @ _dequant(w8, scale).t()).to(torch.bfloat16)
    rel = (y.float() - y_ref.float()).abs().max() / y_ref.float().abs().max().clamp(min=1e-6)
    assert rel.item() < 2e-2, rel.item()


@pytest.mark.skipif(not e4m3_native(), reason="torch._scaled_mm needs sm_89+")
@pytest.mark.parametrize("M", [1, 2, 4, 16, 64])
@pytest.mark.parametrize("part_rows,uniform", [
    ([12288, 1024, 1024], False),  # fused -> piecewise-constant scale -> row-wise
    ([6144], True),                # standalone -> one scalar -> tensor-wise
])
def test_w8a8_matches_w8a8_reference(M: int, part_rows: list[int], uniform: bool):
    from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

    K = 2048
    w8, scale = _quant_parts(part_rows, K, seed=M)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    input_scale = (x.abs().max().float() / 448.0).reshape(())

    y = fp8_pertensor_linear(x, w8, scale, None, input_scale, uniform)

    xq = (x.float() / input_scale).clamp(-448, 448).to(FP8)
    y_ref = (xq.to(torch.float32) * input_scale) @ _dequant(w8, scale).t()
    rel = ((y.float() - y_ref).norm() / y_ref.norm()).item()
    assert rel < 1e-2, rel


@pytest.mark.skipif(not e4m3_native(), reason="torch._scaled_mm needs sm_89+")
def test_batch_size_does_not_change_the_numeric_scheme():
    """A deployment that can run W8A8 must run it at every M, so that a reply reproduces at
    bs=1 regardless of how many other requests shared its forward. Feeding the same row alone
    and as part of a batch must therefore agree bit-for-bit."""
    from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

    K, part_rows = 2048, [1024, 256]
    w8, scale = _quant_parts(part_rows, K)
    x = torch.randn(8, K, device=DEV, dtype=torch.bfloat16)
    input_scale = (x.abs().max().float() / 448.0).reshape(())

    batched = fp8_pertensor_linear(x, w8, scale, None, input_scale, False)
    alone = fp8_pertensor_linear(x[:1], w8, scale, None, input_scale, False)
    assert torch.equal(alone, batched[:1])


def test_layer_load_marks_uniform_scale_and_optional_input_scale():
    from freetoken.kernel.triton.fp8_pertensor_linear import (
        Fp8PerTensorColMerged,
        Fp8PerTensorLinear,
    )

    K = 512
    w8, scale = _quant_parts([256, 64, 64], K)
    merged = Fp8PerTensorColMerged(K, [256, 64, 64])
    merged.load_state_dict({
        "weight": w8, "weight_scale": scale, "input_scale": torch.tensor(0.01, device=DEV),
    })
    assert merged._uniform_scale is False
    assert merged.input_scale is not None

    single = Fp8PerTensorLinear(K, 384)
    flat = scale[:1].expand(384).contiguous()
    single.load_state_dict({"weight": w8, "weight_scale": flat})  # no input_scale
    assert single._uniform_scale is True
    assert single.input_scale is None

    # a reload must not trip over the input_scale it kept from the first load
    single.load_state_dict({"weight": w8, "weight_scale": flat})
    assert single.input_scale is None


@pytest.mark.skipif(not e4m3_native(), reason="torch._scaled_mm needs sm_89+")
@pytest.mark.parametrize("M", [1, 4, 64, 300])
def test_per_part_path_matches_rowwise(M: int, monkeypatch):
    """Where row-wise ``_scaled_mm`` is unsafe a fused projection runs one tensor-wise GEMM per
    part instead. Same scheme, so the two paths agree up to accumulation order (~7e-4)."""
    import freetoken.kernel.triton.fp8_pertensor_linear as mod

    K, part_rows = 2048, [1024, 256, 256]
    w8, scale = _quant_parts(part_rows, K, seed=M)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    input_scale = (x.abs().max().float() / 448.0).reshape(())

    monkeypatch.setattr(mod, "rowwise_scaled_mm_ok", lambda: True)
    y_row = mod.fp8_pertensor_linear(x, w8, scale, None, input_scale, False)
    monkeypatch.setattr(mod, "rowwise_scaled_mm_ok", lambda: False)
    y_part = mod.fp8_pertensor_linear(x, w8, scale, None, input_scale, False)
    rel = ((y_part.float() - y_row.float()).norm() / y_row.float().norm()).item()
    assert rel < 2e-3, rel


@pytest.mark.skipif(not e4m3_native(), reason="torch._scaled_mm needs sm_89+")
def test_fused_layer_forward_on_a_side_stream_completes():
    """Regression for #182 / #72 / #220: on sm_89 with torch < 2.12 a fused FP8 projection's
    row-wise ``_scaled_mm`` issued from a non-default stream stalls the GPU (PyTorch's
    CUTLASS row-wise kernel ignored the current stream; fixed upstream in 2.12). The layer
    must take a path that completes on every supported build. Runs in a subprocess so a
    stall fails the test instead of hanging the session."""
    script = textwrap.dedent("""
        import os, time, torch
        from freetoken.kernel.triton.fp8_pertensor_linear import FP8, Fp8PerTensorColMerged

        torch.manual_seed(0)
        K, parts = 2048, [8192, 512, 512]  # a prefill-sized fused qkv
        w8 = (torch.randn(sum(parts), K, device="cuda") * 8).clamp(-448, 448).to(FP8)
        scale = torch.cat([torch.full((p,), 0.01 * (i + 1), device="cuda")
                           for i, p in enumerate(parts)])
        layer = Fp8PerTensorColMerged(K, parts)
        layer.load_state_dict({"weight": w8, "weight_scale": scale,
                               "input_scale": torch.tensor(0.02, device="cuda")})
        x = torch.randn(2010, K, device="cuda", dtype=torch.bfloat16)  # #182 shape
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        events = []
        with torch.cuda.stream(stream):
            for _ in range(128):
                layer.forward(x)
                ev = torch.cuda.Event()
                ev.record(stream)
                events.append(ev)
        deadline = time.monotonic() + 30
        done = 0
        while time.monotonic() < deadline:
            while done < len(events) and events[done].query():
                done += 1
            if done == len(events):
                print("completed", done, flush=True)
                os._exit(0)
            time.sleep(0.05)
        print("stalled at", done, "of", len(events), flush=True)
        os._exit(124)  # a normal exit would wait on the stuck kernel
    """)
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"rc={proc.returncode}\n{proc.stdout}\n{proc.stderr[-2000:]}"
