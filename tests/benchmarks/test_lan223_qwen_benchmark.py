"""Unit tests for the LAN-223 Qwen API benchmark safety primitives."""

from __future__ import annotations

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from benchmarks.lan223_qwen.run_api_benchmark import (
    nearest_rank_percentile,
    numeric_summary,
    parse_args,
    require_expected_host,
)
from benchmarks.lan223_qwen.run_quality_suite import evaluate_check
from benchmarks.lan223_qwen.run_multiturn_state_suite import nearest_rank
from benchmarks.lan223_qwen.run_long_context_control import build_prompt
from benchmarks.lan223_qwen.run_concurrent_api_control import parse_args as parse_concurrent_args
from benchmarks.lan223_qwen.summarize_qwen_gguf_endurance import summarize


class RequireExpectedHostTests(unittest.TestCase):
    """Exercise the host guard without requiring any third-party test package."""

    def test_accepts_lan223_short_name(self) -> None:
        """The harness accepts the exact LAN-223 host name used by the test policy."""

        with patch("socket.gethostname", return_value="lan-223"):
            self.assertEqual(require_expected_host("lan-223"), "lan-223")

    def test_rejects_other_hosts(self) -> None:
        """The harness prevents accidental benchmark traffic to any other LAN machine."""

        with patch("socket.gethostname", return_value="lan-199"):
            with self.assertRaisesRegex(RuntimeError, "refusing benchmark"):
                require_expected_host("lan-223")

    def test_throughput_mode_requires_two_requested_tokens(self) -> None:
        """The TPS mode rejects a one-token interval before it can produce nonsense."""

        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--model", "qwen",
                    "--tokenizer", "tokenizer",
                    "--artifact-dir", "artifacts",
                    "--mode", "throughput",
                    "--max-tokens", "1",
                ]
            )

    def test_quality_mode_defaults_to_no_reasoning(self) -> None:
        """The canary requests final-answer text instead of an unbounded thought stream."""

        args = parse_args(
            [
                "--model", "qwen",
                "--tokenizer", "tokenizer",
                "--artifact-dir", "artifacts",
            ]
        )
        self.assertEqual(args.reasoning_effort, "none")


class TailMetricTests(unittest.TestCase):
    """Keep percentile output stable and auditable for later tail studies."""

    def test_nearest_rank_percentiles_select_observed_values(self) -> None:
        """A four-event stream has no fictional interpolated p95 or p99 value."""

        values = [0.01, 0.02, 0.03, 0.04]
        self.assertEqual(nearest_rank_percentile(values, 0.50), 0.02)
        self.assertEqual(nearest_rank_percentile(values, 0.95), 0.04)
        self.assertEqual(nearest_rank_percentile(values, 0.99), 0.04)

    def test_empty_metric_summary_has_explicit_nulls(self) -> None:
        """A one-token answer must not fabricate token-gap tail statistics."""

        self.assertTrue(all(value is None for value in numeric_summary([]).values()))


class QualitySuiteCheckTests(unittest.TestCase):
    """Verify fixture scoring without needing a server or model weights."""

    def test_exact_check_accepts_only_visible_exact_text(self) -> None:
        """Whitespace around an otherwise exact completion is acceptable."""

        self.assertEqual(evaluate_check(" LAN223\n", {"kind": "exact", "value": "LAN223"}), (True, None))
        self.assertFalse(evaluate_check("LAN223!", {"kind": "exact", "value": "LAN223"})[0])

    def test_json_fields_check_rejects_nonvisible_or_wrong_structure(self) -> None:
        """The gate requires a valid visible JSON object with the requested fields."""

        self.assertEqual(
            evaluate_check('{"status":"ok","value":7}', {"kind": "json_fields", "fields": {"status": "ok", "value": 7}}),
            (True, None),
        )
        self.assertFalse(evaluate_check("not json", {"kind": "json_fields", "fields": {"status": "ok"}})[0])


class MultiTurnTailMetricTests(unittest.TestCase):
    """Keep short-suite tail aggregation tied to recorded rather than invented values."""

    def test_nearest_rank_uses_the_observed_worst_value_for_p99(self) -> None:
        """Three turn values make p99 the actual worst measured turn."""

        self.assertEqual(nearest_rank([0.1, 0.2, 0.3], 0.99), 0.3)


class LongContextControlTests(unittest.TestCase):
    """Keep the controlled long prompt deterministic and retrieval-focused."""

    def test_prompt_starts_with_marker_and_ends_with_exact_instruction(self) -> None:
        """The retrieval answer appears only in the protected prefix."""

        prompt = build_prompt(2)
        self.assertTrue(prompt.startswith("Protected marker: azure-17"))
        self.assertEqual(prompt.count("azure-17"), 1)
        self.assertTrue(prompt.endswith("Reply with only the protected marker and no other text."))

    def test_prompt_rejects_zero_filler(self) -> None:
        """A zero-context request cannot accidentally masquerade as a long test."""

        with self.assertRaises(ValueError):
            build_prompt(0)

    def test_prefix_nonce_precedes_the_long_filler(self) -> None:
        """A changing early nonce prevents reuse of the long filler prefix."""

        prompt = build_prompt(2, prefix_nonce="sample-1")
        self.assertIn("Per-sample prefix nonce: sample-1", prompt)
        self.assertLess(prompt.index("sample-1"), prompt.index("This is deterministic filler"))


class DpmPolicyWrapperTests(unittest.TestCase):
    """Protect the policy wrapper's separate telemetry and harness paths."""

    def test_dpm_wrapper_reserves_a_new_harness_child_directory(self) -> None:
        """Policy logs use a parent while the immutable harness receives `benchmark`."""

        repository_root = Path(__file__).resolve().parents[2]
        wrapper = repository_root / "scripts" / "lan223" / "run_qwen_dpm_policy_benchmark.sh"
        contents = wrapper.read_text(encoding="utf-8")

        self.assertIn('readonly BENCHMARK_DIR="${ARTIFACT_ROOT}/benchmark"', contents)
        self.assertIn('bash "${HARNESS}" "${BENCHMARK_DIR}"', contents)
        self.assertNotIn('mkdir -p "${BENCHMARK_DIR}"', contents)


class QwenRecoveryContextTests(unittest.TestCase):
    """Protect the recovery server's validated long-context cache allocation."""

    def test_recovery_reserves_the_advertised_8192_token_context(self) -> None:
        """A restart must not silently shrink the usable cache back to 2,068 tokens."""

        repository_root = Path(__file__).resolve().parents[2]
        recovery = repository_root / "scripts" / "lan223" / "start_qwen_recovery_server.sh"
        contents = recovery.read_text(encoding="utf-8")

        self.assertIn('readonly KV_RESERVE_TOKENS="${FREETOKEN_KV_RESERVE_TOKENS:-8192}"', contents)
        self.assertIn('--kv-reserve-tokens "${KV_RESERVE_TOKENS}"', contents)

    def test_multiturn_battery_requires_swap_free_preflight(self) -> None:
        """Repeated state tests must not begin from a swapped memory condition."""

        repository_root = Path(__file__).resolve().parents[2]
        wrapper = repository_root / "scripts" / "lan223" / "run_qwen_multiturn_battery.sh"
        contents = wrapper.read_text(encoding="utf-8")

        self.assertIn('readonly MAX_SWAP_KIB="${LAN223_BATTERY_MAX_SWAP_KIB:-64}"', contents)
        self.assertIn('refusing multi-turn battery with swap in use: ${used} KiB exceeds ${MAX_SWAP_KIB} KiB', contents)
        self.assertIn('if (( used > MAX_SWAP_KIB )); then', contents)
        self.assertIn('assert_clean_swap\ncurl -fsS', contents)
        self.assertIn('"requested_sessions": expected', contents)


class ConcurrentControlArgumentTests(unittest.TestCase):
    """Reject nonsensical concurrent workloads before they can reach LAN-223."""

    def test_concurrency_must_be_positive(self) -> None:
        """Zero clients has no latency or throughput meaning."""

        with self.assertRaises(SystemExit):
            parse_concurrent_args(["--model", "qwen", "--tokenizer", "tokenizer", "--artifact", "artifact", "--concurrency", "0"])


class QwenEnduranceSummaryTests(unittest.TestCase):
    """Ensure retained endurance evidence cannot hide missing or swapped sessions."""

    def _write_session(self, root: Path, number: int, runner_swap_kib: int = 0) -> None:
        """Write the smallest valid passed session plus its explicit telemetry."""

        sessions = root / "sessions"
        sessions.mkdir(exist_ok=True)
        payload = {
            "status": "passed",
            "results": [{"id": "remember", "status": "passed"}],
            "tail_metrics": {"max_ttft_seconds": 0.4, "max_token_gap_seconds": 0.02},
        }
        (sessions / f"session-{number:02d}.json").write_text(json.dumps(payload))
        (sessions / f"session-{number:02d}-telemetry.txt").write_text(
            f"runner_swap_kib={runner_swap_kib}\nwhole_host_swap_kib=39088\n"
        )

    def test_summary_passes_only_complete_zero_runner_swap_evidence(self) -> None:
        """A complete artifact can include background swap without failing the runner gate."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_session(root, 1)
            summary = summarize(root, expected_sessions=1)

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["runner_swap_kib"]["max"], 0)
        self.assertEqual(summary["whole_host_swap_kib"]["max"], 39088)

    def test_summary_rejects_swapped_runner_or_missing_session(self) -> None:
        """Neither runner paging nor an incomplete series may be reported as endurance-qualified."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_session(root, 1, runner_swap_kib=4)
            summary = summarize(root, expected_sessions=2)

        self.assertFalse(summary["passed"])
        self.assertTrue(any("expected 2 sessions" in failure for failure in summary["failures"]))
        self.assertTrue(any("runner swap=4" in failure for failure in summary["failures"]))


class LlamaCppControlScriptTests(unittest.TestCase):
    """Protect the isolated ROCm llama.cpp control lifecycle and workload reuse."""

    def test_control_uses_a_loopback_child_and_existing_fixed_harness(self) -> None:
        """The control must terminate its own port-1921 child and reuse Qwen inputs."""

        repository_root = Path(__file__).resolve().parents[2]
        wrapper = repository_root / "scripts" / "lan223" / "run_qwen_llamacpp_rocm_control.sh"
        contents = wrapper.read_text(encoding="utf-8")

        self.assertIn('readonly BASE_URL="http://127.0.0.1:1921/v1"', contents)
        self.assertIn('trap cleanup_server EXIT', contents)
        self.assertIn('LAN223_QWEN_BASE_URL="${BASE_URL}"', contents)
        self.assertIn('run_qwen_scheduler_baseline.sh', contents)
        self.assertIn('--port 1921', contents)

    def test_timeshare_control_requires_serving_state_before_returning(self) -> None:
        """A port-1919 HTTP response is insufficient while FreeToken is loading."""

        repository_root = Path(__file__).resolve().parents[2]
        wrapper = repository_root / "scripts" / "lan223" / "run_qwen_llamacpp_rocm_timeshare_control.sh"
        contents = wrapper.read_text(encoding="utf-8")

        self.assertIn('"status":"ok"', contents)
        self.assertIn('find_freetoken_pid', contents)
        self.assertIn('sudo swapoff -a', contents)
        self.assertIn('bash "${RECOVERY_SCRIPT}"', contents)

    def test_gemma_control_releases_stale_swap_only_after_qwen_stops(self) -> None:
        """Gemma must start from a clean state without changing host swap policy."""

        repository_root = Path(__file__).resolve().parents[2]
        wrapper = repository_root / "scripts" / "lan223" / "run_gemma4_gguf_text_control.sh"
        contents = wrapper.read_text(encoding="utf-8")

        self.assertIn('sudo swapoff -a', contents)
        self.assertIn('sudo swapon -a', contents)
        self.assertIn('swap-after-qwen-release.txt', contents)
        self.assertLess(contents.index('production_pid="$(port_pid'), contents.index('sudo swapoff -a'))
