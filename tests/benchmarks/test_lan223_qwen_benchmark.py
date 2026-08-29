"""Unit tests for the LAN-223 Qwen API benchmark safety primitives."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.lan223_qwen.run_api_benchmark import parse_args, require_expected_host


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
