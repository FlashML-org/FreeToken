"""``load_weight`` must forward ``tp_shard`` ONLY to readers that declare it.

Regression: the engine asks for ``tp_shard`` on every TP>1 launch, and
``load_weight`` used to raise ``NotImplementedError`` for any reader without that
parameter.  Seven readers shard *internally* instead (they call ``shard_tensor``
with ``tp_info.rank``/``tp_info.size`` inside ``iter_weights``: llama, qwen2,
qwen3, qwen3_moe, mistral, gpt_oss, minimax_m2), so forwarding the flag turned a
working TP>1 launch into a startup failure.  FTW checkpoints store POST-shard
weights, so the flag is a no-op there and must not be rejected either.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.models import weight as weight_mod


def _reader(declares: bool, seen: list):
    if declares:

        def iter_weights(
            model_path, device, *, include_moe_experts, include_non_moe,
            tp_shard=False, config=None,
        ):
            seen.append({"tp_shard": tp_shard, "config": config})
            yield "w", torch.zeros(2, 2)

    else:

        def iter_weights(model_path, device, *, include_moe_experts, include_non_moe):
            seen.append({"tp_shard": "<absent>"})
            yield "w", torch.zeros(2, 2)

    return iter_weights


def _patch(monkeypatch, reader):
    monkeypatch.setattr(
        weight_mod,
        "_spec_for_model_path",
        lambda path: (
            None,
            SimpleNamespace(module="fake.mod", iter_weights="iter_weights"),
        ),
    )
    monkeypatch.setattr(weight_mod, "_load_attr", lambda module, name: reader)
    monkeypatch.setattr("freetoken.checkpoint.ftw.is_ftw_checkpoint", lambda path: False)


def test_tp_shard_is_forwarded_to_a_reader_that_declares_it(monkeypatch):
    seen: list = []
    _patch(monkeypatch, _reader(True, seen))
    list(weight_mod.load_weight("/m", torch.device("cpu"), tp_shard=True))
    assert seen == [{"tp_shard": True, "config": None}]


def test_tp_config_rides_along_only_when_the_reader_accepts_it(monkeypatch):
    seen: list = []
    _patch(monkeypatch, _reader(True, seen))
    sentinel = object()
    list(
        weight_mod.load_weight(
            "/m", torch.device("cpu"), tp_shard=True, tp_config=sentinel
        )
    )
    assert seen == [{"tp_shard": True, "config": sentinel}]


def test_tp_shard_is_not_forwarded_to_a_reader_that_shards_internally(monkeypatch):
    """The regression: this raised, breaking TP>1 for the internally-sharded readers."""
    seen: list = []
    _patch(monkeypatch, _reader(False, seen))
    out = list(weight_mod.load_weight("/m", torch.device("cpu"), tp_shard=True))
    assert [name for name, _ in out] == ["w"]
    assert seen == [{"tp_shard": "<absent>"}]


def test_tp1_does_not_forward_tp_shard_to_such_a_reader_either(monkeypatch):
    seen: list = []
    _patch(monkeypatch, _reader(False, seen))
    list(weight_mod.load_weight("/m", torch.device("cpu"), tp_shard=False))
    assert seen == [{"tp_shard": "<absent>"}]


def test_ftw_checkpoints_accept_tp_shard(monkeypatch):
    """FTW stores post-shard weights: tp_shard is a no-op there, not an error."""
    monkeypatch.setattr("freetoken.checkpoint.ftw.is_ftw_checkpoint", lambda path: True)
    monkeypatch.setattr(
        "freetoken.checkpoint.ftw.iter_ftw_weights",
        lambda path: iter([("w", torch.zeros(2, 2))]),
    )
    monkeypatch.setattr("freetoken.models.config.vision_load_enabled", lambda: False)
    out = list(weight_mod.load_weight("/m", torch.device("cpu"), tp_shard=True))
    assert [name for name, _ in out] == ["w"]


def test_a_reader_without_tp_shard_is_never_asked_for_it(monkeypatch):
    """Guards the mechanism, not just the outcome: the kwarg must not be built at all."""
    seen: list = []
    _patch(monkeypatch, _reader(False, seen))
    list(weight_mod.load_weight("/m", torch.device("cpu"), tp_shard=True))
    assert "<absent>" in seen[0]["tp_shard"], "tp_shard reached a reader that cannot take it"


def test_a_non_callable_reader_attribute_is_not_probed_as_a_signature(monkeypatch):
    """``_load_attr`` may return a non-function; only a real signature decides."""
    monkeypatch.setattr(
        weight_mod,
        "_spec_for_model_path",
        lambda path: (None, SimpleNamespace(module="fake.mod", iter_weights="iter_weights")),
    )
    monkeypatch.setattr(
        weight_mod, "_load_attr", lambda module, name: "not-callable"
    )
    monkeypatch.setattr("freetoken.checkpoint.ftw.is_ftw_checkpoint", lambda path: False)
    with pytest.raises(Exception):
        list(weight_mod.load_weight("/m", torch.device("cpu"), tp_shard=True))
