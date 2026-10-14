"""Small CLI-facing regressions for ``ft bench bw``."""

from freetoken.moe.benchbw import custom_profile_hint


def test_custom_profile_hint_is_shell_safe():
    assert custom_profile_hint("/tmp/my profile.json") == (
        "export FREETOKEN_BENCHBW_PATH='/tmp/my profile.json'"
    )
