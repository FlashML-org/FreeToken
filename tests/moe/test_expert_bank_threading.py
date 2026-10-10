from types import SimpleNamespace

import pytest
import torch
from freetoken.layers.quantization import QuantKind
from freetoken.moe.expert_banks import build_expert_banks


@pytest.mark.parametrize("fail", [False, True])
def test_nvfp4_host_bank_copy_threads_are_restored(monkeypatch, fail):
    threads = 24
    changes = []
    observed = []

    def set_threads(value):
        nonlocal threads
        threads = value
        changes.append(value)

    monkeypatch.setattr(torch, "get_num_threads", lambda: threads)
    monkeypatch.setattr(torch, "set_num_threads", set_threads)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class Method:
        kind = QuantKind.NVFP4
        kernel = SimpleNamespace(name="triton")
        cfg = SimpleNamespace(num_experts=2)

        @staticmethod
        def layout():
            spec = SimpleNamespace(shape=(2,), dtype=torch.uint8, resident=False)
            return {"gate_up": spec}

        @staticmethod
        def pack(piece, out):
            observed.append(threads)
            if fail and piece["gate_up"][0, 0] == 2:
                raise RuntimeError("failed during expert placement")
            out["gate_up"].copy_(piece["gate_up"])
            return {}

    pieces = (
        (0, expert, expert + 1, {
            "gate_up": torch.full((1, 2), expert + 1, dtype=torch.uint8),
        })
        for expert in range(2)
    )
    if fail:
        with pytest.raises(RuntimeError, match="failed during expert placement"):
            build_expert_banks(Method(), 1, pieces, device=torch.device("cpu"))
    else:
        banks = build_expert_banks(Method(), 1, pieces, device=torch.device("cpu"))
        assert banks.sources["gate_up"][0].tolist() == [[1, 1], [2, 2]]

    assert observed == [1, 1]
    assert changes == [1, 24]
    assert threads == 24
