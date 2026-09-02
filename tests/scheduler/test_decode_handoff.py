"""One-step overlap token staging/event-order gates."""

from __future__ import annotations

import pytest
import torch

from freetoken.engine.engine import TokenStaging
from freetoken.scheduler.status import SchedulerStatusReporter


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/ROCm device")
def test_token_staging_alternates_slots_and_preserves_cpu_values():
    device = torch.device("cuda")
    stream = torch.cuda.Stream(device=device)
    staging = TokenStaging(device, capacity=2)
    with torch.cuda.stream(stream):
        gpu0, cpu0, event0 = staging.stage(torch.tensor([11], device=device), stream)
        gpu1, cpu1, event1 = staging.stage(torch.tensor([22], device=device), stream)
        stream.synchronize()
    event0.synchronize()
    event1.synchronize()
    assert gpu0.data_ptr() != 0 and gpu1.data_ptr() != 0
    assert cpu0.data_ptr() != cpu1.data_ptr()
    assert event0 is not event1
    assert cpu0.item() == 11 and cpu1.item() == 22

    # First event is drained before slot zero is reused, matching scheduler overlap order.
    event0.synchronize()
    with torch.cuda.stream(stream):
        _, cpu2, event2 = staging.stage(torch.tensor([33], device=device), stream)
        stream.synchronize()
    assert event2 is event0
    event2.synchronize()
    assert cpu2.item() == 33


def test_decode_stats_only_read_on_configured_log_interval():
    reporter = SchedulerStatusReporter(log=lambda _: None, decode_log_interval=2)
    batch = type("DecodeBatch", (), {"is_decode": True})()
    assert not reporter.decode_stats_due(batch)
    reporter._decode_forward_count = 1
    assert reporter.decode_stats_due(batch)
