"""Every dSpark piece must be REACHED, not merely defined.

The costliest failures in this feature were not wrong code -- they were correct code
nobody called:

* ``logit_indices`` was added so a verify could score every drafted position, then never
  passed. The verify returned one row per request and acceptance died on an IndexError
  four frames away from the cause.
* Per-token compressor states must be selected after acceptance; restoring a single
  pre-block snapshot loses any accepted prefix in the same page.
* ``catch_up_context`` was written to keep the draft layers' window fresh, and never
  called, which reads as a weak drafter rather than a missing call.

None of those show up in a unit test of the piece itself, because the piece works. They
show up here: an import-graph check that each public entry point has a caller in the
package, and behavioural checks that the wiring carries data through.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

PKG = pathlib.Path(__file__).resolve().parents[2] / "python" / "freetoken"


def _calls_in_package() -> set[str]:
    """Every attribute and plain name that appears in a call position, package-wide."""
    called: set[str] = set()
    for path in PKG.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - a broken file is its own failure
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Name):
                called.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                called.add(fn.attr)
        # getattr(obj, "name") is how the engine reaches optional model hooks.
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and node.args
                and len(node.args) > 1
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                called.add(node.args[1].value)
    return called


@pytest.mark.parametrize(
    "symbol,why",
    [
        ("catch_up_context", "draft layers would attend over a stale window"),
        ("store_context_kv", "draft layers would have no context KV at all"),
        ("propose", "nothing would produce a draft"),
        ("rejection_accept_device", "sampled requests would fall back to argmax bias"),
        ("accepted_prefix", "greedy requests would have no acceptance rule"),
        ("sampling_probs", "q and p would not be shaped like the sampler's draws"),
        ("window_cols_for_block", "the block mask would silently stay causal"),
        ("_restore_speculative_carry", "acceptance would leave the rejected carry live"),
        ("_trim_dspark_target_features", "rejected target rows would become draft context"),
        ("draft_into_batch", "the block would keep its noise placeholders"),
        ("_finish_speculative", "nothing would apply acceptance"),
        ("_maybe_make_speculative", "no batch would ever become speculative"),
        ("catch_up_draft_context", "the engine's hook into the drafter's context"),
    ],
)
def test_the_piece_has_a_caller(symbol, why):
    assert symbol in _calls_in_package(), (
        f"{symbol}() is defined but never called anywhere in freetoken -- {why}"
    )


class TestSpeculativeStateIsCarried:
    """The batch fields the speculative path hands between its stages must exist.

    Each of these is written in one place and read in another; a rename on one side is
    invisible until a block runs.
    """

    @pytest.mark.parametrize(
        "field",
        ["speculative", "spec_block", "spec_emitted", "draft_probs",
         "draft_confidence", "draft_tokens", "spec_carry_states",
         "spec_verify_decode"],
    )
    def test_batch_declares_the_field(self, field):
        import torch

        from freetoken.core import Batch

        assert field in Batch.__dataclass_fields__, (
            f"Batch.{field} is read by the speculative path but not declared"
        )
        assert not Batch.__dataclass_fields__[field].init, (
            f"Batch.{field} must not be a constructor argument"
        )
        del torch

    def test_a_fresh_batch_is_not_speculative(self):
        from freetoken.core import Batch

        b = Batch(reqs=[], phase="decode")
        assert b.speculative is False
        assert b.spec_block == 0
        assert b.spec_emitted is None
        assert b.draft_tokens is None
        assert b.spec_carry_states is None
        assert b.spec_verify_decode is False


class TestFixedVerifyGraphWiring:
    def test_target_graph_is_selected_before_ordinary_decode_graph(self):
        engine = (PKG / "engine" / "engine.py").read_text()
        spec_at = engine.index("can_use_spec_cuda_graph")
        decode_at = engine.index("can_use_cuda_graph", spec_at)
        assert spec_at < decode_at

    def test_graph_replays_graph_owned_partial_states(self):
        graph = (PKG / "engine" / "graph.py").read_text()
        assert "journal = self._spec_carry_map[key]" in graph
        assert "batch.spec_carry_states = journal" in graph
        assert "prepare_for_spec_replay" in graph

    def test_verify_uses_device_positions_not_a_host_start_position(self):
        from freetoken.models.deepseek_v4.model import Transformer

        body = inspect.getsource(Transformer.verify_block)
        assert "pos: torch.Tensor" in body
        assert "start_pos" not in body


class TestVerifyReturnsEveryPosition:
    """The bug that cost a restart: a verify must score all 1+k positions."""

    def test_prefill_batched_accepts_logit_indices(self):
        import inspect

        from freetoken.models.deepseek_v4.model import Transformer

        sig = inspect.signature(Transformer.prefill_batched)
        assert "logit_indices" in sig.parameters, (
            "the verify pass needs to select every drafted position"
        )

    def test_the_forward_passes_them_for_a_speculative_batch(self):
        src = (PKG / "models" / "deepseek_v4" / "model.py").read_text()
        assert "logit_indices=logit_indices" in src, (
            "prefill_batched takes logit_indices but the forward never passes it, so a "
            "speculative verify scores only each request's last token"
        )
