"""Unit tests for the LAN-223 Qwen API benchmark safety primitives."""

from __future__ import annotations

import unittest
from pathlib import Path
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
