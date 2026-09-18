"""Queue telemetry carries real scheduler snapshots across the existing IPC codecs."""

from types import SimpleNamespace

from freetoken.message import BaseFrontendMsg, BaseTokenizerMsg, QueueStatsMsg, QueueStatsReply
from freetoken.scheduler.scheduler import Scheduler


def scheduler(offline=False):
    obj = Scheduler.__new__(Scheduler)
    obj.config = SimpleNamespace(offline_mode=offline)
    obj.decode_manager = SimpleNamespace(running_reqs={1, 2})
    obj.prefill_manager = SimpleNamespace(pending_list=[3, 4, 5])
    obj.sent = []
    obj.send_result = obj.sent.extend
    return obj


def test_queue_snapshot_changes_and_idle_zero_are_published():
    obj = scheduler()
    obj._report_queue_stats()
    obj._report_queue_stats()
    assert obj.sent == [QueueStatsMsg(running=2, waiting=3)]
    obj.prefill_manager.pending_list.clear()
    obj.decode_manager.running_reqs.clear()
    obj._report_queue_stats()
    assert obj.sent[-1] == QueueStatsMsg(running=0, waiting=0)


def test_offline_mode_does_not_receive_online_control_messages():
    obj = scheduler(offline=True)
    obj._report_queue_stats()
    assert obj.sent == []


def test_queue_messages_round_trip_both_ipc_hops():
    msg = QueueStatsMsg(2, 3)
    assert BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg)) == msg
    reply = QueueStatsReply(msg.running, msg.waiting)
    assert BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(reply)) == reply
