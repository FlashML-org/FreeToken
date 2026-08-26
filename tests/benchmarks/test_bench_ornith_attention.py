"""Lightweight, GPU-free tests for the Ornith attention bench's case-building.

The bench itself needs CUDA (it calls the production Triton kernels), so these tests
only cover the pure argparse/case-expansion logic -- kept as plain functions with no
torch import specifically so this file can catch sweep-construction bugs (wrong case
count, wrong op routing, geometry drift) on any machine, CI included.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import bench_ornith_attention as bench  # noqa: E402


def test_default_geometry_matches_the_tuned_decode_launch():
    # This is the exact shape kernel/triton/attention.py::decode_launch_config
    # special-cases for packed int4 (BLOCK_N=32/32 splits/4 warps). If either drifts,
    # the bench silently stops exercising the tuned path -- pin both here.
    assert (bench.Q_HEADS, bench.KV_HEADS, bench.HEAD_DIM) == (16, 2, 256)


def test_quant_alias_covers_every_cli_choice():
    for name in bench.QUANT_CHOICES:
        assert name in bench._QUANT_ALIAS
    assert bench._QUANT_ALIAS["int4"] == bench._QUANT_ALIAS["q4_0"] == "int4"
    assert bench._QUANT_ALIAS["bf16"] == "auto"  # the unquantized pool, no oracle check


def test_default_args_build_the_documented_case_counts():
    args = bench.parse_args([])
    cases = bench.build_cases(args)

    n_decode = len(args.kv_quant) * len(args.decode_lengths) * len(args.batch_sizes) * 1
    n_prefill = len(args.kv_quant) * len(args.prefill_chunk_sizes)
    n_extend = len(args.kv_quant) * len(args.decode_lengths) * len(args.extend_chunk_sizes)
    assert len(cases) == n_decode + n_prefill + n_extend

    assert sum(isinstance(c, bench.DecodeCase) for c in cases) == n_decode
    assert sum(isinstance(c, bench.PrefillCase) for c in cases) == n_prefill
    assert sum(isinstance(c, bench.ExtendCase) for c in cases) == n_extend
    assert {c.op for c in cases} == set(bench.OPS)


def test_ops_filter_restricts_to_the_requested_families():
    args = bench.parse_args(["--ops", "decode"])
    cases = bench.build_cases(args)
    assert cases and all(c.op == "decode" for c in cases)


def test_max_kv_splits_sweep_multiplies_only_decode_cases():
    base = bench.build_cases(bench.parse_args(["--ops", "decode", "--decode-lengths", "1024"]))
    swept = bench.build_cases(
        bench.parse_args(["--ops", "decode", "--decode-lengths", "1024", "--max-kv-splits", "8", "16", "32"])
    )
    assert len(swept) == len(base) * 3
    assert {c.max_kv_splits for c in swept} == {8, 16, 32}
    assert base[0].max_kv_splits is None  # default: let the kernel pick its own tuned preference


def test_decode_case_field_layout():
    case = bench.DecodeCase(quant="int4", ctx_len=131072, batch=4, max_kv_splits=32)
    assert case.op == "decode"
    assert (case.quant, case.ctx_len, case.batch, case.max_kv_splits) == ("int4", 131072, 4, 32)


def test_build_cases_is_deterministic():
    args = bench.parse_args(["--kv-quant", "q8_0", "int4"])
    first = bench.build_cases(args)
    second = bench.build_cases(args)
    assert first == second


def test_help_exits_cleanly():
    """--help / case-building must stay usable on a machine with no GPU: torch/triton are
    imported lazily inside the run_* functions, never at module scope or in parse_args."""
    with pytest.raises(SystemExit) as exc:
        bench.parse_args(["--help"])
    assert exc.value.code == 0
    assert "torch" not in vars(bench)  # confirms parse_args never pulled torch into scope
