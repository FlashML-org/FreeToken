"""HIP ctypes host_register / device_ptr fallback (#122) — mocked, no GPU."""

from __future__ import annotations

import ctypes

import pytest

from freetoken.kernel import pinned


class _FakeHip:
    def __init__(self, identity: bool = True, register_status: int = 0):
        self.identity = identity
        self.register_status = register_status
        self.registered = []
        self.last_dev = None

    def hipHostRegister(self, addr, nbytes, flags):
        self.registered.append((int(addr), int(nbytes), int(flags)))
        return self.register_status

    def hipHostUnregister(self, addr):
        return 0

    def hipHostGetDevicePointer(self, out, host, flags):
        host_i = host.value if isinstance(host, ctypes.c_void_p) else int(host)
        val = host_i if self.identity else host_i + 0x2000
        ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p)).contents.value = val
        self.last_dev = val
        return 0


@pytest.fixture(autouse=True)
def _clear_caches():
    pinned._load_pinned_extension.cache_clear()
    pinned._hip_runtime.cache_clear()
    pinned._host_ptr_identity.cache_clear()
    yield
    pinned._load_pinned_extension.cache_clear()
    pinned._hip_runtime.cache_clear()
    pinned._host_ptr_identity.cache_clear()


def test_host_register_uses_hip_when_extension_missing(monkeypatch):
    hip = _FakeHip()
    monkeypatch.setattr(pinned, "_load_pinned_extension", lambda: None)
    monkeypatch.setattr(pinned, "_hip_runtime", lambda: hip)
    pinned.host_register(0x1000, 4096)
    assert hip.registered == [(0x1000, 4096, 3)]  # Portable | Mapped


def test_host_register_no_silent_noop(monkeypatch):
    monkeypatch.setattr(pinned, "_load_pinned_extension", lambda: None)
    monkeypatch.setattr(pinned, "_hip_runtime", lambda: None)
    with pytest.raises(RuntimeError, match="host_register requires"):
        pinned.host_register(0x1000, 64)


def test_device_ptr_translates_when_not_identity(monkeypatch):
    hip = _FakeHip(identity=False)
    monkeypatch.setattr(pinned, "_load_pinned_extension", lambda: None)
    monkeypatch.setattr(pinned, "_hip_runtime", lambda: hip)
    monkeypatch.setattr(pinned, "_host_ptr_identity", lambda: False)

    class T:
        is_cuda = False

        def data_ptr(self):
            return 0xABC0

    assert pinned.device_ptr(T()) == 0xABC0 + 0x2000


def test_device_ptr_identity_uses_host_va(monkeypatch):
    monkeypatch.setattr(pinned, "_load_pinned_extension", lambda: None)
    monkeypatch.setattr(pinned, "_host_ptr_identity", lambda: True)

    class T:
        is_cuda = False

        def data_ptr(self):
            return 0xFEED

    assert pinned.device_ptr(T()) == 0xFEED


def test_create_pinned_requires_ext_on_cuda_torch(monkeypatch):
    monkeypatch.setattr(pinned, "_load_pinned_extension", lambda: None)
    monkeypatch.setattr(pinned, "_hip_pin_memory_fallback", lambda: False)
    with pytest.raises(ImportError, match="_pinned_tensor"):
        pinned.create_pinned_tensor_like(None)
