"""Unit tests for the LAN-223 Qwen API benchmark safety primitives."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from benchmarks.lan223_qwen.run_api_benchmark import require_expected_host


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
