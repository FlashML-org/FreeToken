"""A tokenizer/detokenizer worker that dies during load must report WHY.

Regression test for the asymmetry that made every tokenizer-side startup failure surface as
the generic "backend worker freetoken-detokenizer-0 exited during load": ``_run_scheduler``
pushed an ("error", reason) ack before dying, ``_run_tokenize_worker`` did not.
"""

from __future__ import annotations

import queue

import pytest

from freetoken.server.launch import _run_tokenize_worker


def _drain(q: "queue.Queue") -> list:
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def test_tokenize_worker_reports_the_real_reason_before_dying(monkeypatch):
    """A raising tokenize_worker pushes ("error", reason) so the supervisor can surface it."""
    import freetoken.tokenizer as tok_mod

    def boom(**_kwargs):
        raise FileNotFoundError("tokenizer.json is missing from the checkpoint")

    monkeypatch.setattr(tok_mod, "tokenize_worker", boom)
    q: "queue.Queue" = queue.Queue()

    with pytest.raises(FileNotFoundError):
        _run_tokenize_worker(detach=False, ack_queue=q, tokenizer_path="/nope")

    acks = _drain(q)
    assert acks, "the dying worker pushed no ack at all"
    kind, reason = acks[0]
    assert kind == "error"
    assert "FileNotFoundError" in reason
    assert "tokenizer.json is missing" in reason


def test_tokenize_worker_without_ack_queue_still_propagates(monkeypatch):
    """No ack_queue (direct call / test harness) must not turn the failure into an AttributeError."""
    import freetoken.tokenizer as tok_mod

    def boom(**_kwargs):
        raise ValueError("bad tokenizer config")

    monkeypatch.setattr(tok_mod, "tokenize_worker", boom)

    with pytest.raises(ValueError, match="bad tokenizer config"):
        _run_tokenize_worker(detach=False, tokenizer_path="/nope")


def test_tokenize_worker_success_pushes_no_error_ack(monkeypatch):
    """The happy path must stay clean — no spurious ("error", …) on the ack queue."""
    import freetoken.tokenizer as tok_mod

    monkeypatch.setattr(tok_mod, "tokenize_worker", lambda **_k: None)
    q: "queue.Queue" = queue.Queue()

    _run_tokenize_worker(detach=False, ack_queue=q, tokenizer_path="/nope")

    assert not [a for a in _drain(q) if isinstance(a, tuple) and a and a[0] == "error"]
