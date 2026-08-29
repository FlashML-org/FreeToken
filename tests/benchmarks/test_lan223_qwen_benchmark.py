"""Unit tests for the LAN-223 Qwen API benchmark safety primitives."""

from __future__ import annotations

import unittest
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
