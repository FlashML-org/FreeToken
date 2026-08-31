import os

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="a ROCm GPU is required",
)


def _make_q4_0_weight(rows: int, columns: int, *, phase: int = 0) -> torch.Tensor:
    assert columns % 32 == 0
    packed_rows = []
    for row in range(rows):
        blocks = []
        for block in range(columns // 32):
            scale = torch.tensor(
                [0.0625 * (1 + (row + block + phase) % 7)], dtype=torch.float16
            ).view(torch.uint8)
            low = (torch.arange(16, dtype=torch.uint8) + row + block + phase) % 16
            high = (15 - torch.arange(16, dtype=torch.uint8) + 2 * row + phase) % 16
            blocks.append(torch.cat((scale, low | (high << 4))))
        packed_rows.append(torch.cat(blocks))
    return torch.stack(packed_rows)


def _dequantize_q4_0(weight: torch.Tensor, columns: int) -> torch.Tensor:
    from freetoken.models.gguf.dequant import GGML_Q4_0, dequantize

    return dequantize(weight, GGML_Q4_0, torch.float32).reshape(weight.shape[0], columns)


@pytest.mark.parametrize(
    ("arches", "expected"),
    [
        ("gfx1201", ["gfx1201"]),
        ("gfx1100;gfx1201", ["gfx1100", "gfx1201"]),
    ],
)
def test_gguf_build_uses_freetoken_rocm_arch(monkeypatch, arches, expected):
    import torch.utils.cpp_extension as cpp_extension

    import freetoken.kernel.gguf as gguf
    import freetoken.kernel.utils as kernel_utils

    captured = {}
    sentinel = object()

    def fake_load(**kwargs):
        captured.update(kwargs)
        captured["pytorch_rocm_arch"] = os.environ.get("PYTORCH_ROCM_ARCH")
        return sentinel

    monkeypatch.setenv("FREETOKEN_ROCM_ARCH", arches)
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    monkeypatch.setattr(cpp_extension, "load", fake_load)
    monkeypatch.setattr(gguf, "_staged_rocm_sources", lambda: gguf._CSRC)
    monkeypatch.setattr(kernel_utils, "_rocm_link_flags", lambda: [])
    gguf._module.cache_clear()
    try:
        assert gguf._module() is sentinel
        flags = captured["extra_cuda_cflags"]
        assert not any(flag.startswith("--offload-arch=") for flag in flags)
        assert captured["pytorch_rocm_arch"] == ";".join(expected)
        assert "-mno-wavefrontsize64" in flags
    finally:
        gguf._module.cache_clear()

    assert "PYTORCH_ROCM_ARCH" not in os.environ


def test_gguf_build_rejects_non_rdna_wavefront_target(monkeypatch):
    from freetoken.kernel.gguf import _rocm_gguf_build_config

    monkeypatch.setenv("FREETOKEN_ROCM_ARCH", "gfx90a")
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)

    with pytest.raises(RuntimeError, match="wave32 RDNA3/RDNA4"):
        _rocm_gguf_build_config(["-O3"])


def test_q4_0_dequant_matches_torch_reference():
    from freetoken.kernel.gguf import ggml_dequantize
    from freetoken.models.gguf.dequant import GGML_Q4_0, dequantize

    scale = torch.tensor([0.5], dtype=torch.float16).view(torch.uint8)
    quants = torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2,
        dtype=torch.uint8,
    )
    packed_cpu = torch.cat((scale, quants)).reshape(1, 18)
    expected = dequantize(packed_cpu, GGML_Q4_0, torch.float32).reshape(1, 32)

    actual = ggml_dequantize(
        packed_cpu.to("cuda"), GGML_Q4_0, m=1, n=32, dtype=torch.float32
    )

    torch.testing.assert_close(actual.cpu(), expected)


def test_q4_0_matvec_matches_dequantized_torch_reference():
    from freetoken.kernel.gguf import ggml_mul_mat_vec_a8
    from freetoken.models.gguf.dequant import GGML_Q4_0

    rows, columns = 64, 512
    packed_cpu = _make_q4_0_weight(rows, columns)
    dense = _dequantize_q4_0(packed_cpu, columns)
    base = (torch.arange(columns, dtype=torch.float32) % 3) - 1
    x = torch.stack((base, base.roll(1)))

    actual = ggml_mul_mat_vec_a8(
        packed_cpu.to("cuda"), x.to("cuda"), GGML_Q4_0, rows
    ).cpu()
    expected = x @ dense.T

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=0.1)


def test_q4_0_moe_matvec_routes_experts_and_matches_reference():
    from freetoken.kernel.gguf import ggml_moe_a8_vec
    from freetoken.models.gguf.dequant import GGML_Q4_0

    experts, rows, columns, tokens, top_k = 2, 64, 512, 2, 2
    packed_cpu = torch.stack(
        (
            _make_q4_0_weight(rows, columns, phase=0),
            _make_q4_0_weight(rows, columns, phase=5),
        )
    )
    dense = torch.stack(
        tuple(_dequantize_q4_0(packed_cpu[expert], columns) for expert in range(experts))
    )
    base = (torch.arange(columns, dtype=torch.float32) % 3) - 1
    x = torch.stack((base, base.roll(1)))
    topk_ids = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)

    actual = ggml_moe_a8_vec(
        x.to("cuda"),
        packed_cpu.to("cuda"),
        topk_ids.to("cuda"),
        top_k,
        GGML_Q4_0,
        rows,
        tokens,
    ).cpu()
    expected = torch.stack(
        tuple(
            x[token] @ dense[int(topk_ids[token, route])].T
            for token in range(tokens)
            for route in range(top_k)
        )
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=0.1)
