"""cached_load_hf_config with --hf-overrides, against the same config written into config.json."""

import copy
import json

import pytest

from freetoken.utils.hf import cached_load_hf_config


def _checkpoint(path, model_type: str, dtype: str, rope_parameters: dict) -> str:
    path.mkdir()
    text_config = {"hidden_size": 64, "max_position_embeddings": 4096, "rope_parameters": rope_parameters}
    (path / "config.json").write_text(json.dumps({"model_type": model_type, "dtype": dtype, "text_config": text_config}))
    return str(path)


@pytest.mark.parametrize("model_type", ["qwen3_5", "a_model_type_transformers_does_not_know"])
def test_hf_overrides_load_as_if_the_checkpoint_config_said_so(tmp_path, model_type):
    # vLLM's merge: a nested config section updates key by key, any other value is replaced whole
    yarn = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096}
    checkpoint = _checkpoint(tmp_path / "checkpoint", model_type, "bfloat16", {"rope_type": "default", "rope_theta": 1e6})
    edited = _checkpoint(tmp_path / "edited", model_type, "float16", yarn)
    overrides = {"dtype": "float16", "text_config": {"rope_parameters": yarn}}
    requested = copy.deepcopy(overrides)

    hf = cached_load_hf_config(checkpoint, overrides)
    expected = cached_load_hf_config(edited)
    assert hf.dtype == expected.dtype
    assert hf.text_config.to_dict() == expected.text_config.to_dict()
    # transformers fills rope defaults into the rope_parameters it is handed
    assert overrides == requested
    # a fresh copy each time: the override never reaches the cached checkpoint config
    assert cached_load_hf_config(checkpoint).text_config.rope_parameters["rope_type"] == "default"
