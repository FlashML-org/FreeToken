"""``--model org/name``: a hub repo id resolves to a local checkpoint directory.

``ft serve --model deepseek-ai/DeepSeek-V4-Flash-0731`` used to pass the repo id through
untouched as a filesystem path. Config parsing then opened it relative to cwd:

    FileNotFoundError: No DeepSeek-V4 ModelArgs JSON found under
    deepseek-ai/DeepSeek-V4-Flash-0731 (looked for inference/config.json)

Only the modelscope branch ever downloaded, and ``download_hf_weight`` (used later, by the
weight loaders) fetches ``*.safetensors`` alone -- so it would never have produced
``inference/config.json`` either.

These pin the resolve for both sources, and pin the two things that must keep reading the
*repo id* rather than the resolved snapshot path: the served model name and the parser
cascade. Resolving too early renames the served model to a snapshot hash.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

from freetoken.server.args import parse_args

REPO_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
SNAPSHOT = "/cache/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01"

# Explicit dtype and parsers keep parse_args off the network: all three otherwise read the
# checkpoint config, which is exactly what cannot be loaded until the path resolves.
BASE_ARGS = ["--dtype", "bfloat16", "--tool-call-parser", "llama3", "--reasoning-parser", "off"]


def _parse(model_path: str, *extra: str):
    return parse_args([*BASE_ARGS, "--model", model_path, *extra])[0]


@pytest.fixture
def hf_download():
    with patch("freetoken.utils.hf.snapshot_download", return_value=SNAPSHOT) as m:
        yield m


def test_repo_id_resolves_to_local_snapshot(hf_download):
    assert _parse(REPO_ID).model_path == SNAPSHOT
    assert hf_download.call_args.args[0] == REPO_ID


def test_served_model_name_keeps_the_repo_id(hf_download):
    """Not the snapshot hash -- this is the id clients send in the `model` field."""
    assert _parse(REPO_ID).served_model_name == "DeepSeek-V4-Flash-0731"


def test_local_dir_is_never_downloaded(tmp_path, hf_download):
    assert _parse(str(tmp_path)).model_path == str(tmp_path)
    hf_download.assert_not_called()


def test_formats_freetoken_cannot_load_are_skipped(hf_download):
    """Many repos ship a duplicate .bin copy; the loaders only read safetensors/gguf."""
    _parse(REPO_ID)
    ignored = hf_download.call_args.kwargs["ignore_patterns"]
    assert "*.bin" in ignored
    assert "*.safetensors" not in ignored


def test_dummy_weight_skips_the_tensors(hf_download):
    """--dummy-weight still needs config + tokenizer, but not 156 GiB of weights."""
    _parse(REPO_ID, "--dummy-weight")
    ignored = hf_download.call_args.kwargs["ignore_patterns"]
    assert "*.safetensors" in ignored
    assert "*.gguf" in ignored


def test_modelscope_source_still_uses_modelscope(monkeypatch):
    """The isdir check moved out of the modelscope branch; it must still short-circuit."""
    fake = types.ModuleType("modelscope")
    calls = []
    fake.snapshot_download = lambda repo, **kw: calls.append((repo, kw)) or SNAPSHOT
    monkeypatch.setitem(sys.modules, "modelscope", fake)

    with patch("freetoken.utils.hf.snapshot_download") as hf:
        assert _parse(REPO_ID, "--model-source", "modelscope").model_path == SNAPSHOT
        hf.assert_not_called()
    assert calls[0][0] == REPO_ID
