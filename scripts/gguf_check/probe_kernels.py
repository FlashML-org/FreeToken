"""Record which ggml ops this build runs for each quantized ggml type, as the kernel table ``check.py --kernels`` reads.

    python scripts/gguf_check/probe_kernels.py <out.json>

Runs every op on random packed weights on the current GPU, at the token counts that pick different kernels (vector, small-batch and MMQ).
An op counts as covered when it runs; the values are not checked. The table holds for this build and GPU architecture only.
Each type runs in its own process, so a CUDA fault in one type leaves the others' results intact.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import sys

K, N, EXPERTS, TOP_K = 1024, 256, 8, 2
TOKENS = (1, 8, 16, 512)
GLU_TOKENS = (1, 8)
# not weight types: plain integers and doubles, and the q8 blocks ggml quantizes activations to
_NOT_WEIGHTS = ("F64", "I8", "I16", "I32", "I64", "Q8_1", "Q8_K")


def _run(fn) -> str:
    import torch

    try:
        fn()
        torch.cuda.synchronize()
        return "ok"
    except Exception as e:
        return f"{type(e).__name__}: {e}"[:200]


def probe(ggml_type: int) -> dict:
    import torch

    from freetoken.kernel import ggml
    from freetoken.layers.quantization.ggml_types import row_bytes

    gen = torch.Generator(device="cuda").manual_seed(0)
    row = row_bytes(ggml_type, K)
    w = torch.randint(0, 256, (N, row), dtype=torch.uint8, device="cuda", generator=gen)
    banks = torch.randint(0, 256, (EXPERTS, N, row), dtype=torch.uint8, device="cuda", generator=gen)
    out = {"mmvq_max_tokens": ggml.mmvq_max_tokens(ggml_type)}
    for t in TOKENS:
        x = torch.randn(t, K, dtype=torch.bfloat16, device="cuda", generator=gen)
        # distinct experts per token, as a router picks them: MMQ faults on a token that routes to one expert twice
        ids = torch.rand(t, EXPERTS, device="cuda", generator=gen).argsort(dim=1)[:, :TOP_K].to(torch.int32).contiguous()
        out[f"mm{t}"] = _run(lambda: ggml.mul_mat(w, ggml_type, x))
        out[f"mmid{t}"] = _run(lambda: ggml.mul_mat_id(banks, ggml_type, x, ids))
        if t in GLU_TOKENS:
            out[f"glu{t}"] = _run(lambda: ggml.mul_mat_id_glu(banks, ggml_type, x, ids, "swiglu"))
    ids = torch.arange(4, dtype=torch.int32, device="cuda")
    out["get_rows"] = _run(lambda: ggml.get_rows(w, ggml_type, K, ids))
    out["dequant"] = _run(lambda: ggml.dequantize(w, ggml_type, K))
    return out


def weight_types() -> list[str]:
    from freetoken.layers.quantization.ggml_types import is_quantized, row_bytes
    from freetoken.third_party.gguf import GGML_QUANT_SIZES, GGMLQuantizationType

    names = []
    for qtype in GGMLQuantizationType:
        if qtype not in GGML_QUANT_SIZES or not is_quantized(int(qtype)) or qtype.name in _NOT_WEIGHTS:
            continue
        try:
            row_bytes(int(qtype), K)
        except ValueError:
            continue
        names.append(qtype.name)
    return names


def _probe_by_name(name: str) -> dict:
    from freetoken.third_party.gguf import GGMLQuantizationType

    return probe(int(GGMLQuantizationType[name]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    args = ap.parse_args()

    import torch

    from freetoken.kernel import ggml

    # build the module once up front: a failed build would otherwise show up as every op missing
    ggml.load_module()
    ctx = mp.get_context("spawn")
    table = {}
    for name in weight_types():
        with cf.ProcessPoolExecutor(1, mp_context=ctx) as pool:
            try:
                table[name] = pool.submit(_probe_by_name, name).result()
            except cf.process.BrokenProcessPool:
                table[name] = {"error": "probe process died"}
        bad = sorted(op for op, r in table[name].items() if op != "mmvq_max_tokens" and r != "ok")
        print(f"{name:10s} {'ok' if not bad else 'missing ' + ' '.join(bad)}", flush=True)
    cc = torch.cuda.get_device_capability()
    with open(args.out, "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(), "sm": f"{cc[0]}{cc[1]}", "types": table}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
