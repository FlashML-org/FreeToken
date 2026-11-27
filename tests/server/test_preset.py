"""--preset: fills only the fields the command line left alone; with no preset named, a one-request server is single."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from freetoken.server.args import ServerArgs, parse_args


def _parse(*argv: str):
    hf = SimpleNamespace(to_dict=lambda: {"architectures": ["Qwen3ForCausalLM"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: hf):
        return parse_args(["--model", "/models/anon", *argv])[0]


_PRESET_SINGLE = {
    "max_running_req": 1, "embed_weights": "host", "memory_ratio": 0.95, "moe_prefill_hit_d2d": True,
    "speculative_algorithm": "mtp", "speculative_num_steps": 3, "kv_reserve_tokens": 32768 + 4096,
}


def test_preset_single_changes_its_fields_and_nothing_without_the_flag():
    # --preset none is the server without a preset (no flag at all now serves as single)
    plain, preset = _parse("--preset", "none"), _parse("--preset", "single")
    changed = {f.name: getattr(preset, f.name) for f in dataclasses.fields(plain) if getattr(plain, f.name) != getattr(preset, f.name)}
    assert changed == {**_PRESET_SINGLE, "preset": "single", "preset_fields": tuple(_PRESET_SINGLE)}
    # without the flag: the class defaults, and a startup dump of the arguments that does not name it
    assert all(getattr(plain, name) == getattr(ServerArgs, name) for name in _PRESET_SINGLE)
    assert "preset" not in repr(plain)
    with pytest.raises(SystemExit):
        _parse("--preset", "no-such-preset")


@pytest.mark.parametrize(
    "flag, field, value",
    [
        (("--max-running-requests", "4"), "max_running_req", 4),  # a default typed out is still the user's
        (("--embed-weights", "gpu"), "embed_weights", "gpu"),
        (("--memory-ratio", "0.9"), "memory_ratio", 0.9),
        (("--moe-prefill-hit-d2d",), "moe_prefill_hit_d2d", True),
        (("--speculative-algorithm", "off"), "speculative_algorithm", None),
        (("--speculative-algorithm", "dflash"), "speculative_algorithm", "dflash"),
        (("--speculative-num-steps", "5"), "speculative_num_steps", 5),
        (("--kv-reserve-tokens", "8192"), "kv_reserve_tokens", 8192),
    ],
)
def test_explicit_flag_wins_over_the_preset(flag, field, value):
    args = _parse("--preset", "single", *flag)
    assert getattr(args, field) == value and field not in args.preset_fields
    # a method the user chose keeps its own default k; every other field is still the preset's
    theirs = {field, "speculative_num_steps"} if field == "speculative_algorithm" else {field}
    assert set(args.preset_fields) == set(_PRESET_SINGLE) - theirs
    assert all(getattr(args, name) == _PRESET_SINGLE[name] for name in args.preset_fields)
    if field == "speculative_algorithm":
        assert args.speculative_num_steps == 0


def test_a_one_request_server_without_a_preset_serves_as_single():
    """No --preset and no --max-running-requests above 1: 'single''s fields, filled as --preset single fills them."""
    with patch("freetoken.server.args.logger") as log:
        alone = _parse("--max-running-requests", "1")
    single = _parse("--preset", "single", "--max-running-requests", "1")
    assert dataclasses.asdict(alone) == dataclasses.asdict(single)
    assert alone.preset == "single" and alone.preset_fields == tuple(f for f in _PRESET_SINGLE if f != "max_running_req")
    assert all(getattr(alone, name) == value for name, value in _PRESET_SINGLE.items())
    # one line says it serves as single and names the fields it filled
    said = [c.args[0] for c in log.info.call_args_list if "serving as --preset single" in c.args[0]]
    assert len(said) == 1 and all(name in said[0] for name in alone.preset_fields)
    # --preset single itself differs only in the field the command line typed here
    named = _parse("--preset", "single")
    differ = {f.name for f in dataclasses.fields(named) if getattr(named, f.name) != getattr(alone, f.name)}
    assert differ == {"preset_fields"} and named.preset_fields == tuple(_PRESET_SINGLE)
    # no flag at all resolves to the same config as --preset single
    assert dataclasses.asdict(_parse()) == dataclasses.asdict(named)
    # several running requests: no preset, as before
    assert _parse("--max-running-requests", "2").preset is None


def test_preset_none_keeps_a_one_request_server_without_a_preset():
    bare = _parse("--preset", "none", "--max-running-requests", "1")
    assert bare.preset is None and bare.preset_fields == ()
    # the config without a preset: the class defaults, the one running request it typed apart
    plain = _parse("--preset", "none")
    differ = {f.name: getattr(bare, f.name) for f in dataclasses.fields(bare) if getattr(bare, f.name) != getattr(plain, f.name)}
    assert differ == {"max_running_req": 1}
    assert all(getattr(plain, name) == getattr(ServerArgs, name) for name in _PRESET_SINGLE)
    assert plain.preset is None and plain.preset_fields == ()


@pytest.mark.parametrize(
    "flag, field, value",
    [
        (("--memory-ratio", "0.9"), "memory_ratio", 0.9),
        (("--embed-weights", "gpu"), "embed_weights", "gpu"),
        (("--speculative-algorithm", "off"), "speculative_algorithm", None),
        (("--kv-reserve-tokens", "8192"), "kv_reserve_tokens", 8192),
    ],
)
def test_explicit_flag_wins_over_the_one_request_default(flag, field, value):
    args = _parse("--max-running-requests", "1", *flag)
    assert args.preset == "single" and getattr(args, field) == value and field not in args.preset_fields
    assert dataclasses.asdict(args) == dataclasses.asdict(_parse("--preset", "single", "--max-running-requests", "1", *flag))


def test_preset_kv_reserve_follows_the_output_limit_and_leaves_a_sized_kv_pool_alone():
    assert _parse("--preset", "single", "--max-output-tokens", "8192").kv_reserve_tokens == 8192 + 4096
    sized = _parse("--preset", "single", "--num-tokens", "65536")
    assert sized.kv_reserve_tokens == ServerArgs.kv_reserve_tokens and "kv_reserve_tokens" not in sized.preset_fields


def _preset_spec_config(model_path: str, raises: Exception | None):
    from freetoken.engine import engine

    config = SimpleNamespace(
        model_path=model_path, speculative_draft_model_path=None, preset="single",
        speculative_algorithm="mtp", speculative_num_steps=3, spec_slot_states=("state",),
        preset_fields=("max_running_req", "speculative_algorithm", "speculative_num_steps"),
    )

    def checks(*_args):
        if raises is not None:
            raise raises

    with patch.object(engine, "_adjust_spec_config", checks):
        engine._adjust_preset_spec_config(config, lambda name, value: setattr(config, name, value), windowed_kv=False)
    return config


def test_preset_keeps_speculative_decoding_for_a_hub_id():
    config = _preset_spec_config("nvidia/Qwen3.6-35B-A3B-NVFP4", raises=None)
    assert (config.speculative_algorithm, config.speculative_num_steps, config.spec_slot_states) == ("mtp", 3, ("state",))
    assert config.preset_fields == ("max_running_req", "speculative_algorithm", "speculative_num_steps")


def test_preset_drops_the_speculative_decoding_its_checks_reject():
    config = _preset_spec_config("/models/anon", raises=ValueError("the checkpoint has no MTP layers"))
    assert (config.speculative_algorithm, config.speculative_num_steps, config.spec_slot_states) == (None, 0, ())
    assert config.preset_fields == ("max_running_req",)
