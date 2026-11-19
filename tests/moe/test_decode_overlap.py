import pytest

from freetoken.decode_overlap import ENV, PRIORITY_ENV, DecodeSharedOverlap, enabled


class _Log:
    def __init__(self):
        self.events = []

    def add(self, kind, *args):
        self.events.append((kind, *args))


class _Event:
    def __init__(self, log, name):
        self.log = log
        self.name = name

    def record(self, stream):
        self.log.add("record", self.name, stream.name)


class _Stream:
    def __init__(self, log, name, priority=0):
        self.log = log
        self.name = name
        self.priority = priority

    def wait_event(self, event):
        self.log.add("wait", self.name, event.name)

    def __enter__(self):
        self.log.add("enter", self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.log.add("exit", self.name)
        return False


class _Cuda:
    def __init__(self):
        self.log = _Log()
        self.producer = _Stream(self.log, "compute")
        self._n = 0

    def Stream(self, device=None, priority=0):
        self._n += 1
        return _Stream(self.log, f"side{self._n}", priority)

    def Event(self):
        self._n += 1
        return _Event(self.log, f"ev{self._n}")

    def current_stream(self, device):
        return self.producer

    def stream(self, stream):
        return stream


def test_overlap_disabled_by_default(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert enabled() is False


@pytest.mark.parametrize("value", ["2", "true", "on"])
def test_overlap_flag_rejects_other_values(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    with pytest.raises(ValueError, match="expected unset, 0 or 1"):
        enabled()


def test_run_orders_ready_copy_shared_and_done(monkeypatch):
    monkeypatch.setenv(PRIORITY_ENV, "0")
    cuda = _Cuda()
    overlap = DecodeSharedOverlap(cuda, device=0)
    seen = []

    def copy_missing():
        seen.append("copy")

    def shared():
        seen.append("shared")
        return "shared-out"

    assert overlap.run(copy_missing, shared) == "shared-out"
    assert seen == ["copy", "shared"]
    names = [item[0] for item in cuda.log.events]
    # Construction and the call each record ready, then the side stream waits.
    assert names.count("record") >= 2
    assert names.count("wait") >= 2
    copy_at = seen.index("copy")
    # The side stream must wait for ready before copy, and the producer must
    # wait for done after shared returns.
    wait_ready = [i for i, item in enumerate(cuda.log.events) if item[0] == "wait" and item[1].startswith("side")]
    wait_done = [i for i, item in enumerate(cuda.log.events) if item[0] == "wait" and item[1] == "compute"]
    assert wait_ready and wait_done
    assert cuda.log.events[wait_ready[-1]][2] != cuda.log.events[wait_done[-1]][2]
    assert copy_at == 0


def test_shared_error_still_joins(monkeypatch):
    monkeypatch.setenv(PRIORITY_ENV, "0")
    cuda = _Cuda()
    overlap = DecodeSharedOverlap(cuda, device=0)

    def copy_missing():
        return None

    def shared():
        raise RuntimeError("shared failed")

    with pytest.raises(RuntimeError, match="shared failed"):
        overlap.run(copy_missing, shared)
    waits = [item for item in cuda.log.events if item[0] == "wait"]
    assert ("wait", "compute", overlap.done.name) in waits


def test_copy_error_records_done_and_joins(monkeypatch):
    monkeypatch.setenv(PRIORITY_ENV, "0")
    cuda = _Cuda()
    overlap = DecodeSharedOverlap(cuda, device=0)

    def copy_missing():
        raise RuntimeError("copy failed")

    def shared():
        raise AssertionError("shared ran after a failed copy")

    with pytest.raises(RuntimeError, match="copy failed"):
        overlap.run(copy_missing, shared)
    assert any(item[0] == "record" and item[1] == overlap.done.name for item in cuda.log.events)
    waits = [item for item in cuda.log.events if item[0] == "wait"]
    assert ("wait", "compute", overlap.done.name) in waits
