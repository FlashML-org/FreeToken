"""Pure GGUF dispatch matrix; no kernel build or GPU required."""

from __future__ import annotations

import pytest

import freetoken.kernel.gguf as gguf


def _backend(monkeypatch, *, rocm: bool):
    monkeypatch.setattr(gguf.torch.version, "hip", "7.2" if rocm else None)
    monkeypatch.setattr(gguf.torch.version, "cuda", None if rocm else "13.0")


@pytest.mark.parametrize(
    ("tokens", "implementation"),
    [(1, "ggml_mul_mat_vec_a8"), (6, "ggml_mul_mat_vec_a8"), (8, "ggml_mul_mat_vec_a8")],
)
def test_dense_shape_policy(monkeypatch, tokens, implementation):
    _backend(monkeypatch, rocm=True)
    report = gguf.gguf_dispatch("dense", 8, 2048, 2048, tokens, "gfx1100")
    assert report["implementation"] == implementation
    assert report["quant_type"] == "Q8_0"
    assert report["rows"] == 2048
    assert report["cols"] == 2048


@pytest.mark.parametrize(
    ("op", "implementation", "reason"),
    [
        ("moe_decode", "ggml_moe_a8_vec", None),
        ("moe_prefill", "ggml_moe_a8_vec", None),
        ("grouped_prefill", "ggml_moe_a8", None),
    ],
)
def test_moe_phase_is_explicit(monkeypatch, op, implementation, reason):
    _backend(monkeypatch, rocm=True)
    report = gguf.gguf_dispatch(op, 12, 4096, 2048, 4, "gfx1100")
    assert report["implementation"] == implementation
    assert report["reason"] == reason


def test_rdna3_prefill_switches_to_grouped_mmq(monkeypatch):
    _backend(monkeypatch, rocm=True)
    monkeypatch.setenv("FREETOKEN_GGUF_GROUPED_PREFILL", "1")
    report = gguf.gguf_dispatch("moe_prefill", 12, 4096, 2048, 5, "gfx1100")
    assert report["implementation"] == "ggml_moe_a8"
    assert report["reason"] == "multi-token grouped path"


def test_rdna3_prefill_fails_closed_to_vector_by_default(monkeypatch):
    _backend(monkeypatch, rocm=True)
    monkeypatch.delenv("FREETOKEN_GGUF_GROUPED_PREFILL", raising=False)
    report = gguf.gguf_dispatch("moe_prefill", 12, 4096, 2048, 5, "gfx1100")
    assert report["implementation"] == "ggml_moe_a8_vec"
    assert report["reason"] == "grouped path disabled after gfx1100 launch failure"


def test_quant_k_alignment_is_fail_closed(monkeypatch):
    _backend(monkeypatch, rocm=True)
    report = gguf.gguf_dispatch("dense", 13, 4096, 255, 1, "gfx1100")
    assert report["implementation"] == "unsupported"
    assert report["reason"] == "K dimension 255 is not aligned to 256"


def test_q4_k_and_cuda_matrix(monkeypatch):
    _backend(monkeypatch, rocm=False)
    report = gguf.gguf_dispatch("dense", 12, 4096, 2048, 1, "sm90")
    assert report["backend"] == "cuda"
    assert report["implementation"] == "ggml_mul_mat_vec_a8"
    assert report["quant_type"] == "Q4_K"


def test_arch_mismatch_is_unsupported(monkeypatch):
    _backend(monkeypatch, rocm=True)
    report = gguf.gguf_dispatch("dense", 8, 32, 32, 1, "sm90")
    assert report["implementation"] == "unsupported"
    assert report["reason"] == "NVIDIA architecture requested on ROCm"


def test_nvidia_only_forced_implementation_fails_loudly(monkeypatch):
    _backend(monkeypatch, rocm=True)
    with pytest.raises(ValueError, match="unsupported GGUF implementation"):
        gguf.gguf_dispatch("dense", 8, 16, 16, 1, "gfx1100", impl="marlin")


def test_dispatch_trace_aggregates_calls(monkeypatch):
    _backend(monkeypatch, rocm=True)
    monkeypatch.setenv("FREETOKEN_GGUF_DISPATCH_TRACE", "1")
    gguf._DISPATCH_COUNTS.clear()
    gguf.gguf_dispatch("dense", 8, 32, 32, 1, "gfx1100")
    rows = gguf.gguf_dispatch_report()
    assert rows[-1]["implementation"] == "ggml_mul_mat_vec_a8"
    assert rows[-1]["calls"] == 1


def test_rocm_jit_source_stays_cache_local(monkeypatch, tmp_path):
    from torch.utils import cpp_extension

    build_root = tmp_path / "torch_extensions"
    source = tmp_path / "gguf_kernel.cu"
    source.write_text("// source\n")
    monkeypatch.setattr(
        cpp_extension,
        "_get_build_directory",
        lambda name, verbose: str(build_root / name),
    )

    target = gguf._rocm_jit_source("gguf_probe", source)

    assert target == build_root / "gguf_probe" / "gguf_kernel.cu"
    assert target.read_text() == source.read_text()


def test_b10434_bs1_workspace_is_aligned_and_fail_closed():
    from freetoken.kernel.gguf import mmvq_bs1_workspace_bytes, validate_mmvq_bs1_workspace

    required = mmvq_bs1_workspace_bytes(2048, 512, 8)
    assert required % 256 == 0
    validate_mmvq_bs1_workspace(2048, 512, 8, required)
    with pytest.raises(ValueError, match="too small"):
        validate_mmvq_bs1_workspace(2048, 512, 8, required - 1)
