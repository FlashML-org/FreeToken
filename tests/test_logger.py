from __future__ import annotations

import sys
from types import ModuleType
from types import SimpleNamespace


def test_logger_formatter_preserves_tensor_parallel_rank(monkeypatch, capsys):
    distributed = ModuleType("freetoken.distributed")
    distributed.try_get_tp_info = lambda: SimpleNamespace(
        rank=2, is_primary=lambda: False
    )
    monkeypatch.setitem(sys.modules, "freetoken.distributed", distributed)
    from freetoken.logging import init_logger
    logger = init_logger("tests.logger.rank", use_tp_rank=True)

    logger.info("ranked")

    assert "|core|rank=2]" in capsys.readouterr().out
