import pytest

from freetoken.decode_gather import DECODE_GATHER_ENV, TUNED_DECODE_GATHER, decode_gather_config


def test_decode_gather_defaults_off(monkeypatch):
    monkeypatch.delenv(DECODE_GATHER_ENV, raising=False)
    assert decode_gather_config() is None


@pytest.mark.parametrize("value", ["", "0", "off", "default", " tuned "])
def test_decode_gather_off_aliases_and_tuned(monkeypatch, value):
    monkeypatch.setenv(DECODE_GATHER_ENV, value)
    if value.strip().lower() == "tuned":
        assert decode_gather_config() == TUNED_DECODE_GATHER
    else:
        assert decode_gather_config() is None


def test_decode_gather_typo_raises(monkeypatch):
    monkeypatch.setenv(DECODE_GATHER_ENV, "tma")
    with pytest.raises(ValueError, match="expected 'tuned'"):
        decode_gather_config()
