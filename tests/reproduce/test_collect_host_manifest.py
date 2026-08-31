"""Static safety checks for the portable public manifest collector."""

from __future__ import annotations

import unittest
from pathlib import Path


class CollectHostManifestTests(unittest.TestCase):
    """Keep the public collector portable, privacy-conscious, and HIP-only."""

    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[2]
        cls.script = (root / "scripts" / "reproduce" / "collect_host_manifest.sh").read_text(
            encoding="utf-8"
        )

    def test_requires_a_native_hip_pytorch_device(self) -> None:
        self.assertIn("PyTorch does not report a HIP runtime", self.script)
        self.assertIn("PyTorch HIP device is unavailable", self.script)

    def test_default_manifest_redacts_hostname_and_omits_sensitive_inventory(self) -> None:
        self.assertIn('PUBLIC_HOSTNAME="redacted"', self.script)
        self.assertNotIn("ps -eo", self.script)
        self.assertNotIn("lsblk -o NAME,MODEL,SERIAL", self.script)

    def test_public_collector_has_no_host_identifier_or_personal_path_dependency(self) -> None:
        forbidden_host = "lan" + "-" + "223"
        self.assertNotIn(forbidden_host, self.script.lower())
        self.assertNotIn("/home/" + "david", self.script)

    def test_artifact_directory_must_be_new(self) -> None:
        self.assertIn('if [[ -e "${ARTIFACT_DIR}" ]]', self.script)
        self.assertIn("artifact directory already exists", self.script)

    def test_checksums_cover_the_raw_reports_and_manifest(self) -> None:
        self.assertIn("sha256sum system.txt source-state.txt python-hip.json rocminfo.txt rocm-smi.txt", self.script)
        self.assertIn('"manifest.json"', self.script)
        self.assertIn('"SHA256SUMS"', self.script)


if __name__ == "__main__":
    unittest.main()
