"""The scheduler worker's own process setup, before it builds the engine."""

from __future__ import annotations

import queue
import threading

import pytest
import torch
from tqdm import tqdm

import freetoken.gpu_select
import freetoken.scheduler
from freetoken.distributed import DistributedInfo
from freetoken.server import launch
from freetoken.server.args import ServerArgs
from freetoken.utils import progress


def test_the_scheduler_worker_draws_its_bars_under_a_lock_of_its_own(monkeypatch):
    """Only rank 0 draws a bar, so no bar lock spans processes. tqdm's default one is a multiprocessing
    semaphore, which a worker ended by a signal leaves to resource_tracker as leaked."""

    class EngineBuilt(Exception):
        pass

    def scheduler(args):
        raise EngineBuilt(tqdm.get_lock())

    monkeypatch.setattr(freetoken.scheduler, "Scheduler", scheduler)
    monkeypatch.setattr(freetoken.gpu_select, "set_assigned_gpu", lambda target: None)
    monkeypatch.setattr(progress, "_PROGRESS_SINK", None)
    monkeypatch.setattr(tqdm, "_lock", tqdm.get_lock())
    args = ServerArgs(model_path="unused", tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16)
    with pytest.raises(EngineBuilt) as built:
        launch._run_scheduler(args, queue.Queue())
    assert type(built.value.args[0]) is type(threading.RLock())
