import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="a ROCm GPU is required",
)


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
