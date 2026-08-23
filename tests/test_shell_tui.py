"""Focused regressions for the interactive shell command loop."""

from __future__ import annotations

import asyncio

from freetoken.shell import tui


class _FakeClient:
    async def wait_until_ready(self, **_kwargs):
        return {"status": "ok"}

    async def model_id(self):
        return "google/gemma-4-26B-A4B-it"

    async def cache_status(self):
        return {
            "geometry": {
                "reasoning": {
                    "gears": ["low", "high"],
                    "kwargs": {},
                    "default": "low",
                }
            }
        }

    async def stats(self):
        return {}


class _FakePromptSession:
    def __init__(self, *_args, **_kwargs):
        self._commands = ["/think status"]

    async def prompt_async(self):
        if self._commands:
            return self._commands.pop(0)
        raise EOFError


def test_think_command_still_uses_gears_when_model_switch_can_refresh_them(monkeypatch):
    output: list[str] = []
    monkeypatch.setattr(tui, "PromptSession", _FakePromptSession)
    monkeypatch.setattr(tui.ShellConsoleRenderer, "_write_stdout", output.append)

    result = asyncio.run(
        tui._run_shell(_FakeClient(), "http://test", connect_grace=0.0)
    )

    assert result == 0
    assert any("Thinking: low (available: low, high)" in line for line in output)
