from __future__ import annotations

import subprocess

from freetoken.server.process_utils import (
    reap_backend_workers,
    terminate_backend_workers,
)


class FakePopen:
    def __init__(self, *, exits_on_terminate: bool) -> None:
        self.running = True
        self.exits_on_terminate = exits_on_terminate
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self.running else 0

    def terminate(self):
        self.terminated = True
        if self.exits_on_terminate:
            self.running = False

    def wait(self, timeout=None):
        if self.running:
            raise subprocess.TimeoutExpired("fake", timeout)
        return 0

    def kill(self):
        self.killed = True
        self.running = False


def test_popen_workers_are_terminated_and_reaped():
    graceful = FakePopen(exits_on_terminate=True)
    stubborn = FakePopen(exits_on_terminate=False)

    terminate_backend_workers([graceful, stubborn])
    reap_backend_workers([graceful, stubborn], timeout=0)

    assert graceful.terminated and not graceful.killed
    assert stubborn.terminated and stubborn.killed
