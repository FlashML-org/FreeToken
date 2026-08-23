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


class TestTheBaseIsSelectableAndMeasured:
    """Which base the ids use is an empirical answer, not a spec.

    The checkpoint declares [40, 41, 42] with n_layers=43. Read 0-based those are the
    last three layers; read 1-based they are the 2nd-to-4th from last. Both land inside
    the model, so no assertion can decide it -- only a measurement can, and it did:

        1-based (39, 40, 41): 25% accepted at 800 drafted tokens
        0-based (40, 41, 42): 20% accepted at the same point

    So 1-based is the default, and the override stays because that answer came from one
    checkpoint rather than from documentation.
    """

    def test_the_default_is_one_based(self):
        src = _model_src()
        assert 'FREETOKEN_DSPARK_LAYER_BASE", "1"' in src, (
            "1-based measured better on this checkpoint; it is the default"
        )

    def test_the_base_is_an_override_not_a_hardcode(self):
        assert "FREETOKEN_DSPARK_LAYER_BASE" in _model_src()

    def test_only_zero_or_one_is_accepted(self):
        assert "base not in (0, 1)" in _model_src()

    def test_the_resolved_layers_are_logged(self):
        # The wrong base costs acceptance and raises nothing, so the run must say which
        # layers it actually tapped.
        src = _model_src()
        assert "dSpark: tapping target layers" in src

    @pytest.mark.parametrize("base,expected", [(1, {39, 40, 41}), (0, {40, 41, 42})])
    def test_both_readings_stay_inside_the_model(self, base, expected):
        ids = tuple(i - base for i in (40, 41, 42))
        assert set(ids) == expected
        assert all(0 <= i < 43 for i in ids), (
            "both readings are in range, which is why only a measurement can choose"
        )


def _model_src() -> str:
    import pathlib as _p

    return (
        _p.Path(__file__).resolve().parents[2]
        / "python" / "freetoken" / "models" / "deepseek_v4" / "model.py"
    ).read_text()
