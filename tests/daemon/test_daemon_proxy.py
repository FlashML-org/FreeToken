from __future__ import annotations

import io
import json
import urllib.error

import pytest

from freetoken.daemon.proxy import ServeProbe

LOADING = {
    "status": "loading",
    "phase": "weights",
    "progress": {"done_bytes": 1, "total_bytes": 2},
    "model": "proxy-test",
    "instance_id": "proxy-test-instance",
}


def _answering(status_code: int, body: bytes):
    """An opener that answers like urllib.request.urlopen: the decoded body on 200, an HTTPError
    carrying the body otherwise."""

    def opener(url: str, timeout: float) -> dict:
        if status_code == 200:
            return json.loads(body)
        raise urllib.error.HTTPError(url, status_code, "error", {}, io.BytesIO(body))

    return opener


def test_a_health_document_answered_with_503_reads_as_the_document():
    """The serve's /health answers 503 until it is serving; /engine/health still shows the load."""
    body = json.dumps(LOADING).encode()
    answered_ok = ServeProbe(opener=_answering(200, body)).health(1919)
    answered_unavailable = ServeProbe(opener=_answering(503, body)).health(1919)
    assert answered_unavailable == answered_ok
    assert answered_unavailable["progress"] == {"doneBytes": 1, "totalBytes": 2}


@pytest.mark.parametrize("body", [b"Internal Server Error", b'{"detail": "Internal Server Error"}'])
def test_an_error_without_a_status_document_is_still_an_error(body: bytes):
    answered = ServeProbe(opener=_answering(500, body)).health(1919)
    assert answered == {"reachable": True, "status": "error", "httpStatus": 500}
