import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="a ROCm GPU is required",
)


class _KernelRecorder:
    def __init__(self):
        self.kwargs = None

    def __getitem__(self, _grid):
        def launch(*_args, **kwargs):
            self.kwargs = kwargs

        return launch


def test_rocm_activation_launch_omits_cuda_pdl_attribute(monkeypatch):
    import freetoken.kernel.triton.activation as activation

    recorder = _KernelRecorder()
    monkeypatch.setattr(activation, "_act_and_mul_kernel", recorder)
    monkeypatch.setattr(activation, "_pdl_supported", lambda: True)
    monkeypatch.setattr(activation, "is_rocm", lambda: True)

    activation.silu_and_mul(torch.ones((2, 128), device="cuda"))

    assert recorder.kwargs is not None
    assert recorder.kwargs["ENABLE_PDL"] is True
    assert "launch_pdl" not in recorder.kwargs


def test_rocm_norm_launch_omits_cuda_pdl_attribute(monkeypatch):
    import freetoken.kernel.triton.norm as norm

    recorder = _KernelRecorder()
    monkeypatch.setattr(norm, "_rmsnorm_kernel", recorder)
    monkeypatch.setattr(norm, "is_sm90_supported", lambda: True)
    monkeypatch.setattr(norm, "is_rocm", lambda: True)

    x = torch.ones((2, 64), device="cuda")
    weight = torch.ones(64, device="cuda")
    norm.rmsnorm(x, weight)

    assert recorder.kwargs is not None
    assert recorder.kwargs["ENABLE_PDL"] is True
    assert "launch_pdl" not in recorder.kwargs
