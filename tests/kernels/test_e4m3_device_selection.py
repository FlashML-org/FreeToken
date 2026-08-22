import pytest

from freetoken.kernel.triton import e4m3_compat


PRE_FP8_CAPABILITY = (8, 0)
NATIVE_FP8_CAPABILITY = (8, 9)


@pytest.mark.parametrize(
    ("current_capability", "expected"),
    [(PRE_FP8_CAPABILITY, False), (NATIVE_FP8_CAPABILITY, True)],
)
def test_e4m3_native_uses_current_device(monkeypatch, current_capability, expected):
    capabilities = [PRE_FP8_CAPABILITY, NATIVE_FP8_CAPABILITY]

    def get_device_capability(device=None):
        return current_capability if device is None else capabilities[device]

    monkeypatch.setattr(e4m3_compat, "_native", None)
    monkeypatch.setattr(e4m3_compat.torch.cuda, "device_count", lambda: len(capabilities))
    monkeypatch.setattr(e4m3_compat.torch.cuda, "get_device_capability", get_device_capability)

    assert e4m3_compat.e4m3_native() is expected
