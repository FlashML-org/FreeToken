from types import SimpleNamespace

import pytest
import torch
from freetoken.engine.engine import _validate_ple_backend
from freetoken.engine.graph import GraphRunner


def _config(backend: str, *, graph_bs=None, graph_max=None):
    return SimpleNamespace(
        ple_backend=backend,
        cuda_graph_bs=graph_bs,
        cuda_graph_max_bs=graph_max,
    )


def _qwen_config(*, ple: bool = True):
    return SimpleNamespace(
        qwen4_args=SimpleNamespace(ple_layer_ids=(2,) if ple else ()),
    )


def _adjust(config, model_config):
    return _validate_ple_backend(config, model_config)


def test_pinned_ple_keeps_cuda_graph_configuration():
    config = _config("pinned", graph_bs=[1, 2], graph_max=2)
    _adjust(config, _qwen_config())
    assert config.cuda_graph_bs == [1, 2]
    assert config.cuda_graph_max_bs == 2


def test_mmap_ple_keeps_cuda_graph_configuration():
    config = _config("mmap", graph_bs=[1, 2], graph_max=2)
    _adjust(config, _qwen_config())
    assert config.cuda_graph_bs == [1, 2]
    assert config.cuda_graph_max_bs == 2


def test_mmap_ple_rejects_models_without_a_ple_table():
    config = _config("mmap")
    with pytest.raises(ValueError, match="Qwen3.8-Flash-Next"):
        _adjust(config, SimpleNamespace())
    with pytest.raises(ValueError, match="PLE table"):
        _adjust(config, _qwen_config(ple=False))


def test_unknown_ple_backend_is_rejected_programmatically():
    with pytest.raises(ValueError, match="ple_backend"):
        _adjust(_config("nvme"), _qwen_config())


def test_graph_replay_prepares_model_before_replay():
    events = []

    class Buffer:
        logits = torch.zeros(2, 3)

        def copy_from(self, batch):
            events.append("copy")

    class Model:
        def prepare_cuda_graph_replay(self, batch):
            events.append("model")

    class Attention:
        def prepare_for_replay(self, batch):
            events.append("attention")

    class Graph:
        def replay(self):
            events.append("replay")

    runner = object.__new__(GraphRunner)
    runner.max_graph_bs = 2
    runner.buffer = Buffer()
    runner.model = Model()
    runner.attn_backend = Attention()
    runner.graph_map = {2: Graph()}
    batch = SimpleNamespace(is_decode=True, size=1, padded_size=2)

    logits = runner.replay(batch)

    assert logits.shape == (1, 3)
    assert events == ["copy", "model", "attention", "replay"]
