"""Failure/retry and concurrency contracts for the terminal accounting barrier."""
import asyncio
from types import SimpleNamespace

import pytest

from freetoken.message import UserReply
from freetoken.server.accounting import AccountingDrainError, prepare_stop_accounting
from freetoken.server.stats import StatsTracker


def state_with_request():
    stats = StatsTracker()
    stats.on_new_user(7)
    return SimpleNamespace(stats=stats, maintenance_state="serving", instance_id="instance",
                           ready_at=None, config=SimpleNamespace(served_model_name="test"))


def terminal(state, tokens=3):
    state.stats.observe(UserReply(uid=7, incremental_output="", finished=True,
                                 completion_tokens_delta=tokens, error="aborted"))


def test_drain_timeout_then_late_terminal_can_retry_without_losing_tokens():
    async def run():
        state = state_with_request()
        aborted = []

        async def abort(uid):
            aborted.append(uid)
            state.stats.on_abort(uid)  # enqueueing abort is NOT a terminal reply

        state.abort_user = abort
        with pytest.raises(AccountingDrainError, match="abort barrier timed out"):
            await prepare_stop_accounting(state, drain_timeout_s=0, abort_timeout_s=0)
        assert state.maintenance_state == "stopping"
        assert state.stats.active == 1 and not hasattr(state, "_sealed_accounting")
        terminal(state, tokens=11)  # last sampled-token delta arrives AFTER the timeout
        sealed = await prepare_stop_accounting(state, drain_timeout_s=0, abort_timeout_s=0)
        assert sealed["completion_tokens_total"] == 11
        assert sealed["drain_complete"] is True
        assert aborted == [7]
        sealed["completion_tokens_total"] = -1
        assert (await prepare_stop_accounting(state))["completion_tokens_total"] == 11
    asyncio.run(run())


def test_concurrent_prepare_stop_serializes_abort_and_returns_independent_snapshots():
    async def run():
        state = state_with_request()
        entered, release = asyncio.Event(), asyncio.Event()
        aborted = []

        async def abort(uid):
            assert state.maintenance_state == "stopping"
            aborted.append(uid)
            entered.set()
            await release.wait()
            terminal(state)

        state.abort_user = abort
        first = asyncio.create_task(prepare_stop_accounting(state, drain_timeout_s=0))
        await entered.wait()
        second = asyncio.create_task(prepare_stop_accounting(state, drain_timeout_s=0))
        await asyncio.sleep(0)  # second caller reaches the shared lock, no wall-clock race
        assert not second.done()
        release.set()
        a, b = await asyncio.gather(first, second)
        assert a == b and a is not b
        assert aborted == [7]
        assert a["completion_tokens_total"] == 3
    asyncio.run(run())


def test_abort_transport_failure_preserves_engine_and_closes_gate():
    state = state_with_request()

    async def abort(uid):
        raise ConnectionError("scheduler disconnected")

    state.abort_user = abort
    with pytest.raises(AccountingDrainError, match="failed to abort active requests") as error:
        asyncio.run(prepare_stop_accounting(state, drain_timeout_s=0, abort_timeout_s=0))
    assert isinstance(error.value.__cause__, ConnectionError)
    assert state.maintenance_state == "stopping" and state.stats.active == 1
    assert not hasattr(state, "_sealed_accounting")


def test_unidentified_active_request_cannot_be_sealed():
    state = state_with_request()
    state.stats = SimpleNamespace(active=1, inflight_uids=())
    with pytest.raises(AccountingDrainError, match="unidentified"):
        asyncio.run(prepare_stop_accounting(state, drain_timeout_s=0))
    assert state.maintenance_state == "stopping"
    assert not hasattr(state, "_sealed_accounting")


@pytest.mark.parametrize("maintenance", ["rebuilding", "unknown"])
def test_invalid_prepare_transition_preserves_previous_state(maintenance):
    state = state_with_request()
    state.maintenance_state = maintenance
    with pytest.raises(AccountingDrainError):
        asyncio.run(prepare_stop_accounting(state, drain_timeout_s=0))
    assert state.maintenance_state == maintenance
    assert not hasattr(state, "_sealed_accounting")


def test_cancelled_drain_releases_lock_without_reopening_admission():
    async def run():
        state = state_with_request()
        task = asyncio.create_task(prepare_stop_accounting(state, drain_timeout_s=10))
        await asyncio.sleep(0)  # admission closes before the first drain wait
        assert state.maintenance_state == "stopping"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not state._accounting_prepare_lock.locked()
        assert not hasattr(state, "_sealed_accounting")
        terminal(state)
        assert (await prepare_stop_accounting(state))["drain_complete"] is True
    asyncio.run(run())
