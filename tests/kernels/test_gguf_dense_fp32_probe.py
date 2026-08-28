"""Guard the benchmark-only FP32 dense-output experiment's public boundary."""

from __future__ import annotations

import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_dense_probe_is_explicitly_opt_in_and_not_used_by_layers() -> None:
    """Prevent an experiment flag from changing normal GGUF model execution."""
    wrapper_path = REPOSITORY_ROOT / "python" / "freetoken" / "kernel" / "gguf.py"
    layer_path = REPOSITORY_ROOT / "python" / "freetoken" / "layers" / "gguf.py"
    wrapper_module = ast.parse(wrapper_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in wrapper_module.body
        if isinstance(node, ast.FunctionDef) and node.name == "ggml_mul_mat_vec_a8"
    )
    output_argument = next(arg for arg in function.args.kwonlyargs if arg.arg == "output_fp32")
    default = function.args.kw_defaults[function.args.kwonlyargs.index(output_argument)]

    assert isinstance(default, ast.Constant) and default.value is False
    assert "output_fp32" not in layer_path.read_text(encoding="utf-8")
