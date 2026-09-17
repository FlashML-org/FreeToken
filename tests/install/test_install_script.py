from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
INSTALL_SH = ROOT / "install.sh"


@unittest.skipUnless(os.name == "posix", "AppImage installer is Linux-only")
class AppImageInstallerEnvironmentTests(unittest.TestCase):
    def _capture_curl_library_path(
        self,
        *,
        ld_library_path: str,
        appdir: str | None,
    ) -> tuple[subprocess.CompletedProcess[str], str]:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fakebin = tmp / "fakebin"
            fakebin.mkdir()
            capture = tmp / "curl-env.txt"

            curl = fakebin / "curl"
            curl.write_text(
                "#!/bin/sh\n"
                "printf '%s' \"${LD_LIBRARY_PATH-<unset>}\" > \"$FREETOKEN_TEST_CAPTURE\"\n"
                "exit 42\n",
                encoding="utf-8",
            )
            curl.chmod(curl.stat().st_mode | stat.S_IXUSR)
            (fakebin / "sh").symlink_to("/bin/sh")
            (fakebin / "mkdir").symlink_to("/bin/mkdir")

            env = {
                "HOME": str(tmp / "home"),
                "PATH": str(fakebin),
                "LD_LIBRARY_PATH": ld_library_path,
                "FREETOKEN_TEST_CAPTURE": str(capture),
                "FREETOKEN_BIN_DIR": str(tmp / "bin"),
                "FREETOKEN_HOME": str(tmp / "freetoken-home"),
                "FREETOKEN_ENV_DIR": str(tmp / "env"),
            }
            if appdir is not None:
                env["APPDIR"] = appdir

            result = subprocess.run(
                ["/bin/bash", str(INSTALL_SH), "--yes"],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(
                result.returncode,
                42,
                msg=f"installer did not reach the fake curl:\nstdout={result.stdout}\nstderr={result.stderr}",
            )
            self.assertTrue(capture.exists(), "fake curl did not record its environment")
            return result, capture.read_text(encoding="utf-8")

    def test_appimage_entries_are_removed_before_system_curl(self) -> None:
        appdir = "/tmp/.mount_freetoken-test"
        near_prefix = f"{appdir}-other/usr/lib"
        original = ":".join(
            [
                f"{appdir}/usr/lib",
                "/opt/user/lib",
                f"{appdir}/usr/lib/x86_64-linux-gnu",
                near_prefix,
                "/usr/local/lib",
            ]
        )

        _, seen = self._capture_curl_library_path(
            ld_library_path=original,
            appdir=appdir,
        )

        self.assertEqual(seen, f"/opt/user/lib:{near_prefix}:/usr/local/lib")

    def test_non_appimage_invocation_preserves_library_path(self) -> None:
        original = "/opt/user/lib:/usr/local/lib"

        _, seen = self._capture_curl_library_path(
            ld_library_path=original,
            appdir=None,
        )

        self.assertEqual(seen, original)

    def test_all_appimage_entries_unset_library_path(self) -> None:
        appdir = "/tmp/.mount_freetoken-test"

        _, seen = self._capture_curl_library_path(
            ld_library_path=f"{appdir}/usr/lib:{appdir}/usr/lib/x86_64-linux-gnu",
            appdir=appdir,
        )

        self.assertEqual(seen, "<unset>")


if __name__ == "__main__":
    unittest.main(verbosity=2)
