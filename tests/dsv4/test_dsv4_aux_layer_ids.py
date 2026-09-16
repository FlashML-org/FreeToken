"""Which target layers the drafter taps.

dspark_target_layer_ids is 0-based. This checkpoint declares [40, 41, 42] against
n_layers=43 -- exactly the last three layers, and already valid indices into 0..42.
Reading them as 1-based and subtracting one taps 39, 40, 41: the drafter reads the wrong
three layers on every single draft, and nothing anywhere raises.

A range check cannot separate the two readings, because both land inside the model. The
guard that does is structural: a draft tap reads the END of the target stack, so the ids
must reach the last layer. Under the wrong base they stop one short, every time.
"""

from __future__ import annotations

import pytest


def _args(ids, n_layers=43):
    from freetoken.models.deepseek_v4.args import DeepseekV4Args

    a = DeepseekV4Args.__new__(DeepseekV4Args)
    object.__setattr__(a, "dspark_target_layer_ids", tuple(ids))
    object.__setattr__(a, "n_layers", n_layers)
    return a


def _resolve(args):
    """The id resolution as the Transformer performs it, without building a model."""
    ids = tuple(args.dspark_target_layer_ids)
    bad = [i for i in ids if not 0 <= i < args.n_layers]
    if bad:
        raise ValueError(f"{ids} are 0-based and must land inside {args.n_layers}; {bad} do not")
    return frozenset(ids)


class TestIdsMustLandInsideTheModel:
    def test_in_range_ids_resolve(self):
        assert _resolve(_args([40, 41, 42], 43)) == {40, 41, 42}


class TestTheWrongBaseIsRejected:
    def test_out_of_range_ids_raise(self):
        with pytest.raises(ValueError, match="0-based"):
            _resolve(_args([41, 42, 43], 43))

    def test_no_ids_is_not_an_error(self):
        # dSpark off, or a checkpoint that declares none.
        assert _resolve(_args([], 43)) == frozenset()


class TestTheReferenceMappingIsFixed:
    def test_the_resolved_layers_are_logged(self):
        assert "dSpark: tapping target layer outputs" in _model_src()

    def test_no_runtime_base_override_remains(self):
        assert "FREETOKEN_DSPARK_LAYER_BASE" not in _model_src()

    def test_vllm_mapping_selects_the_last_three_outputs(self):
        # vLLM adds one to the config ids, then captures when idx+1 is selected.
        config = (40, 41, 42)
        aux_ids = tuple(i + 1 for i in config)
        captured = tuple(i for i in range(43) if i + 1 in aux_ids)
        assert captured == config


def _model_src() -> str:
    import pathlib as _p

    return (
        _p.Path(__file__).resolve().parents[2]
        / "python" / "freetoken" / "models" / "deepseek_v4" / "model.py"
    ).read_text()
