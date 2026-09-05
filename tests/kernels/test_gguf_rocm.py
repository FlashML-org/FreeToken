"""ROCm GGUF native-path gates: policy, Q4_0 reference parity, and MoE shape."""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel import gguf as kernel
from freetoken.models.gguf.dequant import GGML_Q4_0, dequant_q4_0, row_bytes


def test_rocm_runtime_metadata_uses_visible_target(monkeypatch):
    monkeypatch.setattr(kernel, "_runtime_backend", lambda: "rocm")
    monkeypatch.setattr(kernel, "_runtime_arch", lambda: "gfx1201")
    metadata = kernel.gguf_runtime_metadata()
    assert metadata["backend"] == "rocm"
    assert metadata["arch"] == "gfx1201"
    assert "USE_HIP=1" in metadata["compile_flags"]
    assert "offload-arch=gfx1201" in metadata["compile_flags"]


def test_rocm_jit_arches_use_explicit_targets(monkeypatch):
    monkeypatch.setenv("FREETOKEN_ROCM_ARCH", "gfx1151;gfx1201")
    monkeypatch.delenv("FREETOKEN_KERNEL_CACHE_GFX", raising=False)
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    assert kernel._rocm_jit_arches() == ("gfx1151", "gfx1201")


def test_rocm_jit_arches_fail_without_target_or_device(monkeypatch):
    from freetoken.utils import arch

    monkeypatch.delenv("FREETOKEN_KERNEL_CACHE_GFX", raising=False)
    monkeypatch.delenv("FREETOKEN_ROCM_ARCH", raising=False)
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    monkeypatch.setattr(arch, "get_rocm_gfx_arch", lambda: None)
    with pytest.raises(RuntimeError, match="ROCm GGUF JIT needs"):
        kernel._rocm_jit_arches()


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1103", "gfx1200", "gfx1201"])
def test_rocm_q4_dispatch_accepts_declared_wave32_families(monkeypatch, arch):
    monkeypatch.setattr(kernel, "_runtime_backend", lambda: "rocm")
    report = kernel.gguf_dispatch("dense", GGML_Q4_0, 8, 256, 1, arch)
    assert report["implementation"] == "ggml_mul_mat_vec_a8"
    assert report["reason"] is None


def test_rocm_dispatch_rejects_cross_backend_architecture(monkeypatch):
    monkeypatch.setattr(kernel, "_runtime_backend", lambda: "rocm")
    report = kernel.gguf_dispatch("dense", GGML_Q4_0, 8, 256, 1, "sm_90")
    assert report["implementation"] == "unsupported"
    assert report["reason"] == "NVIDIA architecture requested on ROCm"


def test_rocm_dispatch_rejects_target_outside_matrix(monkeypatch):
    monkeypatch.setattr(kernel, "_runtime_backend", lambda: "rocm")
    with pytest.raises(ValueError, match="not in FreeToken target matrix"):
        kernel.gguf_dispatch("dense", GGML_Q4_0, 8, 256, 1, "gfx9999")


def test_rocm_auto_reports_generic_fallback_for_candidate_request(monkeypatch):
    monkeypatch.setattr(kernel, "_runtime_backend", lambda: "rocm")
    report = kernel.gguf_dispatch("moe", 12, 16, 256, 1, "gfx1103", impl="auto")
    assert report["route"] == "generic"
    assert report["fallback_route"] == "generic"
    assert report["implementation"] == "ggml_moe_a8_vec"


def test_rocm_forced_candidate_rejects_non_exact_target(monkeypatch):
    monkeypatch.setattr(kernel, "_runtime_backend", lambda: "rocm")
    with pytest.raises(RuntimeError, match="exact target gfx1100"):
        kernel.gguf_dispatch("moe", 12, 16, 256, 1, "gfx1103", impl="rdna3_mmid")


def test_cuda_forced_candidate_never_probes_rocm(monkeypatch):
    monkeypatch.setattr(kernel, "_runtime_backend", lambda: "cuda")
    monkeypatch.setattr(
        kernel,
        "ensure_gguf_moe_candidate_ready",
        lambda: pytest.fail("CUDA path probed ROCm candidate"),
    )
    with pytest.raises(RuntimeError, match="requires ROCm backend"):
        kernel.gguf_dispatch("moe", 12, 16, 256, 1, None, impl="rdna3_mmid")


def test_rocm_exact_candidate_route_requires_registered_abi(monkeypatch):
    monkeypatch.setattr(kernel, "_runtime_backend", lambda: "rocm")
    monkeypatch.setattr(kernel, "ensure_gguf_moe_candidate_ready", lambda: True)
    monkeypatch.setitem(kernel._GGUF_MOE_ABI, "ggml_moe_mmvq_id", kernel._GGUF_ABI_VERSION)
    report = kernel.gguf_dispatch("moe", 12, 16, 256, 1, "gfx1100", impl="rdna3_mmid")
    assert report["route"] == "candidate"
    assert report["capability_status"] == "correctness"


def test_q4_0_reference_zero_row_is_finite():
    raw = torch.zeros((2, row_bytes(256, GGML_Q4_0)), dtype=torch.uint8)
    output = dequant_q4_0(raw, torch.float32)
    assert output.shape == (512,)
    assert torch.isfinite(output).all()
    assert torch.equal(output, torch.zeros_like(output))


ROCM_DEVICE = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="needs ROCm device",
)


@ROCM_DEVICE
def test_q4_0_native_matvec_matches_dequant_reference():
    from freetoken.kernel.gguf import ggml_mul_mat_vec_a8

    rows, cols = 3, 256
    raw = torch.zeros((rows, row_bytes(cols, GGML_Q4_0)), dtype=torch.uint8, device="cuda")
    raw[..., :2] = torch.tensor([0, 56], dtype=torch.uint8, device="cuda")  # fp16 0.5
    raw[..., 2:] = 0x88  # centered q=0 in both half-nibbles
    x = torch.randn((1, cols), dtype=torch.bfloat16, device="cuda")
    output = ggml_mul_mat_vec_a8(raw, x, GGML_Q4_0, rows).float()
    reference_weight = dequant_q4_0(raw.cpu(), torch.float32).reshape(rows, cols).to("cuda")
    reference = x.float() @ reference_weight.T
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference, rtol=5e-2, atol=0.25)


@ROCM_DEVICE
def test_q4_0_native_moe_zero_route_is_finite():
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    experts, rows, cols = 4, 6, 256
    weights = torch.zeros(
        (experts, rows, row_bytes(cols, GGML_Q4_0)), dtype=torch.uint8, device="cuda"
    )
    hidden = torch.zeros((2, cols), dtype=torch.bfloat16, device="cuda")
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="cuda")
    output = ggml_moe_a8_vec(hidden, weights, ids, 2, GGML_Q4_0, rows, 2)
    torch.cuda.synchronize()
    assert output.shape == (4, rows)
    assert torch.isfinite(output).all()
    assert torch.equal(output, torch.zeros_like(output))
