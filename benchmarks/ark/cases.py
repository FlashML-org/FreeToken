"""Deterministic Qwen3 decode inputs, including the actual 20-GiB cache layout."""

from dataclasses import dataclass

import torch


@dataclass
class Case:
    name: str
    baseline: object
    candidate: object
    metadata: dict

    def reset(self):
        pass


def rand(*shape, scale=0.2, dtype=torch.bfloat16):
    return (torch.randn(*shape, device="cuda") * scale).to(dtype)


def quant(w):
    *lead, n, k = w.shape
    blocks = w.float().reshape(*lead, n // 128, 128, k // 128, 128)
    scales = blocks.abs().amax(dim=(-3, -1), keepdim=True).clamp_min(1e-10) / 448
    q = (blocks / scales).to(torch.float8_e4m3fn).reshape(*lead, n, k).contiguous()
    return q, scales.squeeze(-3).squeeze(-1).bfloat16().contiguous()


def dense_cases():
    from freetoken.kernel.ark.dense_fp8_decode.adapter import (
        block_fp8_linear as candidate,
    )
    from freetoken.kernel.triton.fp8_block_linear import block_fp8_linear as baseline

    for n, k in ((5120, 2048), (2048, 4096), (151936, 2048)):
        w, scale = quant(rand(n, k, scale=k**-0.5))
        x = rand(1, k, scale=0.5)
        args = (x, w, scale)
        yield Case(f"N{n}_K{k}", lambda a=args: baseline(*a), lambda a=args: candidate(*a), {"M": 1, "N": n, "K": k})


def moe_cases():
    from freetoken.kernel.ark.moe_decode_fp8.adapter import (
        fused_experts_decode_fp8_block as candidate,
    )
    from freetoken.layers.quantization.moe.base import MoEConfig
    from freetoken.layers.quantization.moe.fp8_block import TritonFp8BlockMoEKernel
    from freetoken.moe.fused_fp8_block import fused_experts_decode_fp8_block as baseline

    h, i, topk = 2048, 768, 8
    for rows in (32, 4550):
        specs = TritonFp8BlockMoEKernel().layout(MoEConfig(rows, h, i, topk))
        banks = {name: torch.empty(rows, *s.shape, device="cuda", dtype=s.dtype) for name, s in specs.items()}
        chosen = torch.linspace(1, rows - 1, 16, device="cuda").long()
        gu, gs = quant(rand(16, 2 * i, h, scale=h**-0.5))
        dn, ds = quant(rand(16, h, i, scale=i**-0.5))
        banks["gate_up"].view(torch.uint8)[chosen] = gu.view(torch.uint8)
        banks["down"].view(torch.uint8)[chosen] = dn.view(torch.uint8)
        for name, values in (("gate_up_scale", gs), ("down_scale", ds)):
            padded = torch.full((16, *banks[name].shape[1:]), 19.0, device="cuda", dtype=values.dtype)
            padded[:, :, : values.shape[-1]] = values
            banks[name][chosen] = padded
        x = rand(1, h, scale=0.5)
        ids = chosen[:topk].int().view(1, topk)
        weights = torch.softmax(torch.randn(1, topk, device="cuda"), -1)
        args = (x, banks["gate_up"], banks["gate_up_scale"], banks["down"], banks["down_scale"], weights, ids)
        yield Case(
            f"C{rows}_H{h}_I{i}", lambda a=args: baseline(*a), lambda a=args: candidate(*a), {"M": 1, "H": h, "I": i, "bank_rows": rows, "top_k": topk}
        )


def cases(op):
    return moe_cases() if op == "moe_decode_fp8" else dense_cases()
