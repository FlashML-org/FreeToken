"""Safety tests for the portable loopback-only API benchmark client."""

from __future__ import annotations

import contextlib
import io
import unittest

from benchmarks.reproduce.run_local_api_benchmark import parse_args, require_loopback_url


class LoopbackUrlTests(unittest.TestCase):
    def test_accepts_localhost_variants(self) -> None:
        self.assertEqual(require_loopback_url("http://127.0.0.1:8000/v1"), "http://127.0.0.1:8000/v1")
        self.assertEqual(require_loopback_url("https://localhost/v1/"), "https://localhost/v1")

    def test_rejects_remote_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            require_loopback_url("http://192.168." + "1.223:1919/v1")

    def test_quality_mode_requires_visible_text_gate(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--model", "model", "--tokenizer", "tokenizer", "--artifact-dir", "artifact", "--prompt", "hello"])

    def test_throughput_requires_at_least_two_tokens(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args([
                "--model", "model", "--tokenizer", "tokenizer", "--artifact-dir", "artifact",
                "--prompt", "hello", "--mode", "throughput", "--max-tokens", "1",
            ])


if __name__ == "__main__":
    unittest.main()
