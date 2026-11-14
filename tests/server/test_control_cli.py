"""ft ctl against a server that refuses: a refusal carrying a status document is the answer to show."""

from __future__ import annotations

import io
import json
import urllib.error

from freetoken import control_cli

LOADING = {"status": "loading", "maintenance": "loading", "phase": "weights", "progress": 0.25}


def _server_answering(monkeypatch, status: int, body: dict) -> None:
    def urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, status, "refused", {}, io.BytesIO(json.dumps(body).encode()))

    monkeypatch.setattr(control_cli.urllib.request, "urlopen", urlopen)


def test_health_answered_with_503_prints_its_status_document(monkeypatch, capsys):
    _server_answering(monkeypatch, 503, LOADING)
    assert control_cli.main(["--json", "health"]) == 0
    assert json.loads(capsys.readouterr().out) == LOADING


def test_a_refusal_without_a_status_document_is_an_error(monkeypatch, capsys):
    _server_answering(monkeypatch, 401, {"error": "Unauthorized"})
    assert control_cli.main(["stats"]) == 1
    assert capsys.readouterr().err.startswith("HTTP 401: ")
