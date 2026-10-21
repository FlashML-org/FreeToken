"""Untimed diagnostic server: real dispatch, GPU symbols and same-input shadows."""

import json
import os
from pathlib import Path

import torch

from freetoken.engine.engine import Engine
from freetoken.kernel.ark.common.trace import flush
from freetoken.kernel.ark.kernels import ArkFp8LinearKernel, ArkFp8MoEKernel
from freetoken.layers.quantization.linear.fp8_block import TritonFp8BlockLinearKernel
from freetoken.layers.quantization.moe.fp8_block import TritonFp8BlockMoEKernel
from .timing import metrics, assert_close

ROOT = Path(os.environ["ARK_MODEL_PROBE"])
SEEN = set()
ROWS = []
ACTIVE = False
PROFILED = False


def shadow(op, key, actual, expected):
    error = metrics(actual, expected)
    assert_close(error, op)
    ROWS.append(dict(operator=op, key=str(key), **error))
    (ROOT / "shadows.json").write_text(json.dumps(ROWS, indent=2) + "\n")


_linear = ArkFp8LinearKernel.apply


def linear(self, layer, x):
    result = _linear(self, layer, x)
    key = ("dense_fp8_decode", id(layer))
    if ACTIVE and x.numel() == x.shape[-1] and key not in SEEN:
        SEEN.add(key)
        shadow(key[0], (tuple(x.shape), tuple(layer.weight.shape)), result, TritonFp8BlockLinearKernel().apply(layer, x))
    return result


_moe = ArkFp8MoEKernel.apply


def moe(self, layer, x, weights, ids, view, *, is_prefill):
    result = _moe(self, layer, x, weights, ids, view, is_prefill=is_prefill)
    key = ("moe_decode_fp8", layer.layer_id)
    if ACTIVE and not is_prefill and x.shape[0] == 1 and key not in SEEN:
        SEEN.add(key)
        shadow(
            key[0],
            (layer.layer_id, tuple(x.shape), view.tensors["gate_up"].shape[0]),
            result,
            TritonFp8BlockMoEKernel().apply(layer, x, weights, ids, view, is_prefill=False),
        )
    return result


ArkFp8LinearKernel.apply = linear
ArkFp8MoEKernel.apply = moe
_forward = Engine.forward_batch


def forward(self, batch, args):
    global ACTIVE, PROFILED
    ACTIVE = True
    try:
        if not batch.is_prefill and not PROFILED:
            PROFILED = True
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as trace:
                result = _forward(self, batch, args)
                torch.cuda.synchronize()
            trace.export_chrome_trace(str(ROOT / "decode-trace.json"))
        else:
            result = _forward(self, batch, args)
        flush()
        return result
    finally:
        ACTIVE = False


Engine.forward_batch = forward

if __name__ == "__main__":
    from freetoken.cli import main

    main()
