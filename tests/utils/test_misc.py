"""Small helpers of freetoken.utils.misc."""

import pytest


@pytest.mark.parametrize("value,text,on", [(None, "", False), ("", "", False), (" 1 ", "1", True), ("On", "on", True),
                                           ("true", "true", True), ("yes", "yes", True), ("0", "0", False), ("2", "2", False)])
def test_opt_in_env_is_on_only_when_asked(value, text, on):
    from freetoken.utils.misc import opt_in_env

    assert opt_in_env(value) == (text, on)
