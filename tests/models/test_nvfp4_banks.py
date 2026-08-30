from __future__ import annotations

import pytest
from freetoken.models import nvfp4_banks


@pytest.mark.parametrize("fail", [False, True])
def test_single_threaded_torch_copies_restores_threads(monkeypatch, fail):
    current = 32
    calls: list[int] = []

    monkeypatch.setattr(nvfp4_banks.torch, "get_num_threads", lambda: current)
    monkeypatch.setattr(nvfp4_banks.torch, "set_num_threads", calls.append)

    if fail:
        with pytest.raises(RuntimeError):
            with nvfp4_banks._single_threaded_torch_copies():
                raise RuntimeError("loader failed")
    else:
        with nvfp4_banks._single_threaded_torch_copies():
            pass

    assert calls == [1, current]
