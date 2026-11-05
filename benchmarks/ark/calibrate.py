"""Recalibrate the exact source/environment; retain raw paired repeats and fallbacks."""

import argparse
import json
import os
from pathlib import Path

import torch
from freetoken.kernel.ark import initialize
from freetoken.kernel.ark.common import policy

from .cases import cases
from .timing import assert_close, graph_pair, metrics, timing_pair


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--write-profile", type=Path)
    a = p.parse_args()
    torch.manual_seed(20261021)
    torch.set_num_threads(8)
    os.environ["FREETOKEN_USE_ARK"] = "1"
    initialize()
    policy.MODE = "native"
    profile = {
        "_environment": policy.current_environment(),
        "_rules": {"eager_min_speedup": 1.05, "each_repeat_faster": True},
        "_scope": "Eager decode only; Graph entries stay empty pending separate model qualification.",
    }
    rows = []
    for op in ("dense_fp8_decode", "moe_decode_fp8"):
        profile[op] = {"eager": {}, "graph": {}}
        for case in cases(op):
            expected = case.baseline().clone()
            actual = case.candidate()
            error = metrics(actual, expected)
            assert_close(error, op)
            row = {
                "operator": op,
                "case": case.name,
                "metadata": case.metadata,
                "correctness": error,
                "eager": timing_pair(case.baseline, case.candidate, case.reset, False, 50),
                "graph": graph_pair(case.baseline, case.candidate, case.reset, False, 100),
            }
            shape = policy.metadata_key(op, case.metadata)
            timing = row["eager"]
            paired = [x / y for x, y in zip(timing["samples_us"]["baseline"], timing["samples_us"]["candidate"])]
            timing["enabled"] = timing["speedup"] >= 1.05 and min(paired) > 1
            row["graph"]["enabled"] = False
            if timing["enabled"]:
                profile[op]["eager"][shape] = {"speedup": timing["speedup"], "min_paired_speedup": min(paired)}
            rows.append(row)
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps({"environment": profile["_environment"], "rows": rows}, indent=2) + "\n")
            print(op, case.name, row["eager"]["speedup"], row["graph"]["speedup"], flush=True)
    if a.write_profile:
        a.write_profile.write_text(json.dumps(profile, indent=2) + "\n")


if __name__ == "__main__":
    main()
