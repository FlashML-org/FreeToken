"""--moe-collect-stats: the flag, and the report it produces.

The counters themselves are accumulated device-side inside ``ensure_experts`` and were
already covered; what was missing until this flag existed was any way to turn them on from
the command line or read them back. These tests cover that wiring -- the flag reaching
``ServerArgs``, and the emit formatting the numbers and resetting the window afterwards.
"""

import contextlib
import io
from types import SimpleNamespace

from freetoken.engine.engine import MOE_STATS_INTERVAL, Engine
from freetoken.server.args import ServerArgs, parse_args


def test_flag_is_registered_and_defaults_off():
    """``--help`` short-circuits before the model is resolved, so this needs no checkpoint."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
        parse_args(["--help"])
    assert "--moe-collect-stats" in buf.getvalue()
    # Off unless asked for: the counters ride in the decode CUDA graph and cost throughput.
    assert ServerArgs.moe_collect_stats is False


class _StubCache:
    """Just enough cache to exercise the emit: the four readers plus the window reset."""

    def __init__(self, decode_target="gpu", layer_calls=512):
        self.collect_stats = True
        self.decode_target = decode_target
        self._layer_calls = layer_calls
        self.reset_calls = 0

    def decode_miss_stats(self):
        return {
            "layer_calls": self._layer_calls,
            "active_per_layer": 8.0,
            "missing_per_layer": 2.0,
            "miss_rate": 0.25,
            "fetched_per_layer": 1.5,
            "cpu_per_layer": 0.5,
            "fetch_rate": 0.75,
            "prefill_hit_rows": 0,
            "prefill_rows": 0,
        }

    def decode_miss_stats_per_layer(self):
        return {
            "per_layer": [
                {"layer": 0, "steps": 4, "miss_rate": 0.5},
                {"layer": 1, "steps": 4, "miss_rate": 0.1},
                # steps == 0 means the layer never ran in this window; it must not be
                # ranked as a 0.0-miss-rate "best" layer.
                {"layer": 2, "steps": 0, "miss_rate": 0.0},
            ]
        }

    def decode_routing_stats(self):
        return {
            "slots_per_layer": 56.7,
            "working_set_mean": 173.1,
            "working_set_max": 243,
            "experts_for_90pct": 92.3,
            "oracle_hit_at_slots": 0.764,
            "norm_entropy": 0.813,
        }

    def reset_stats(self):
        self.reset_calls += 1


def _emit(cache, caplog):
    engine = SimpleNamespace(moe_offload_cache=cache, _emit_moe_stats=None)
    with caplog.at_level("INFO"):
        Engine._emit_moe_stats(engine)
    return "\n".join(r.getMessage() for r in caplog.records)


def test_emit_reports_and_resets_the_window(caplog):
    cache = _StubCache()
    out = _emit(cache, caplog)
    assert "miss_rate=0.250" in out
    # The oracle bound is the whole point of the report: it says how much room a different
    # eviction policy could possibly have.
    assert "oracle_hit=0.764" in out
    assert "(realized 0.750)" in out
    # Ranked worst-first, and the layer that never ran is left out entirely.
    assert "L0=0.500, L1=0.100" in out
    assert "L2" not in out
    assert cache.reset_calls == 1


def test_hybrid_split_only_reported_for_hybrid(caplog):
    assert "fetch_rate" not in _emit(_StubCache(decode_target="gpu"), caplog)
    caplog.clear()
    assert "fetch_rate=0.750" in _emit(_StubCache(decode_target="hybrid"), caplog)


def test_idle_window_emits_nothing_and_keeps_counters(caplog):
    """No decode ran, so there is nothing to report -- and nothing to reset either."""
    cache = _StubCache(layer_calls=0)
    assert _emit(cache, caplog) == ""
    assert cache.reset_calls == 0


def test_interval_is_a_sane_window():
    assert MOE_STATS_INTERVAL >= 1
