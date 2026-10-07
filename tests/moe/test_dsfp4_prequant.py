"""ds_fp4 GPU input pre-quantization must be bit-identical to the CPU round-trip.

The ds_fp4 path FP8-round-trips the input activations to match DeepSeek-V4's W4A8
reference. Doing that round-trip on the GPU (``act_quant_fp8_roundtrip``, the same
kernel the GPU W4A8 path uses) before the D2H copy removes a single-threaded scalar
pass from the decode critical path (~0.3ms/layer at H=4096) -- but only counts as
faithful if the CPU sees the exact same bits. This test pins that equivalence by
running the same decode with the round-trip on either side.
"""

from types import SimpleNamespace

import pytest
import torch

H, I, E, TOPK = 1024, 512, 16, 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_dsfp4_gpu_prequant_bit_parity():
    from freetoken.kernel.pinned import alloc_pinned_tensor
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    torch.manual_seed(0)
    banks = {
        "gate_up_packed": alloc_pinned_tensor(E, 2 * I, H // 2, dtype=torch.uint8),
        "gate_up_scale": alloc_pinned_tensor(E, 2 * I, H // 32, dtype=torch.uint8),
        "down_packed": alloc_pinned_tensor(E, H, I // 2, dtype=torch.uint8),
        "down_scale": alloc_pinned_tensor(E, H, I // 32, dtype=torch.uint8),
    }
    banks["gate_up_packed"].random_(0, 256)
    banks["down_packed"].random_(0, 256)
    banks["gate_up_scale"].fill_(121)  # e8m0 2^-6: realistic weight magnitudes
    banks["down_scale"].fill_(121)
    cache = SimpleNamespace(
        quant_format="ds_fp4",
        bank_sources={k: [v] for k, v in banks.items()},
        num_layers=1,
        num_experts=E,
    )
    dev = torch.device("cuda", 0)
    x = torch.randn(2, H, dtype=torch.bfloat16, device=dev) * 0.3
    ids = torch.tensor([[0, 1, 2, 3], [4, 5, -1, -1]], dtype=torch.int32, device=dev)
    wts = torch.rand(2, TOPK, device=dev)

    def run(cpu_roundtrip: bool) -> torch.Tensor:
        ex = CpuMoeExecutor(
            cache, top_k=TOPK, activation="silu", apply_router_weight_on_input=False,
            num_threads=4, max_tokens=2, device=dev, swiglu_limit=10.0, act_block=128,
        )
        assert ex._gpu_prequant, "ds_fp4 on CUDA should enable GPU prequant"
        if cpu_roundtrip:  # control: the original CPU-side scalar round-trip
            ex._gpu_prequant = False
            ex._ext.set_input_prequant(False)
        out = ex.decode(0, x, wts, ids.clone()).clone()
        torch.cuda.synchronize()
        del ex
        return out

    y_gpu = run(False)
    y_cpu = run(True)
    assert torch.equal(y_gpu, y_cpu), "GPU prequant diverges from the CPU round-trip"
