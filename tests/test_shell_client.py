"""Focused regressions for shell control-plane behavior."""

from __future__ import annotations

import asyncio

from freetoken.shell import client as shell_client


def test_load_model_reports_health_progress_while_post_is_pending(monkeypatch):
    client = object.__new__(shell_client.ShellClient)
    release_post: asyncio.Event
    health_docs = [
        {"status": "loading", "phase": "starting"},
        {
            "status": "loading",
            "phase": "weights",
            "progress": {"done_bytes": 5, "total_bytes": 10},
        },
    ]
    progress: list[dict] = []

    async def run():
        nonlocal release_post
        release_post = asyncio.Event()

        async def request_json(method, path, *, body=None, timeout=None):
            assert (method, path, body, timeout) == (
                "POST",
                "/v1/model/load",
                {"model": "new-model"},
                60.0,
            )
            await release_post.wait()
            return {"status": "ok", "model": "new-model"}

        async def health():
            doc = health_docs.pop(0)
            if not health_docs:
                release_post.set()
            return doc

        client._request_json = request_json
        client.health = health
        return await client.load_model(
            "new-model", wait=60.0, on_progress=progress.append
        )

    monkeypatch.setattr(shell_client, "READY_POLL_INTERVAL", 0)
    result = asyncio.run(run())

    assert result == {"status": "ok", "model": "new-model"}
    assert [doc["phase"] for doc in progress] == ["starting", "weights"]
