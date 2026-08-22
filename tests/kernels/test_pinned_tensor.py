import importlib

import pytest
import torch


def test_pinned_extension_uses_packaged_module_not_runtime_jit(monkeypatch):
    import torch.utils.cpp_extension as cpp_extension

    import freetoken.kernel.pinned as pinned

    class FakeExtension:
        pass

    fake_extension = FakeExtension()
    real_import_module = importlib.import_module

    def fake_import_module(name):
        if name == "freetoken.kernel._pinned_tensor":
            return fake_extension
        return real_import_module(name)

    def fail_runtime_jit(*args, **kwargs):
        raise AssertionError("pinned tensor extension must not use runtime JIT")

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(cpp_extension, "load", fail_runtime_jit)
    pinned._load_pinned_extension.cache_clear()

    try:
        assert pinned._load_pinned_extension() is fake_extension
    finally:
        pinned._load_pinned_extension.cache_clear()


def test_copy_to_pinned_tensor_preserves_strided_cpu_tensor_values():
    if not torch.cuda.is_available():
        pytest.skip("cudaMallocHost requires CUDA")

    from freetoken.kernel import copy_to_pinned_tensor

    source = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()

    output = copy_to_pinned_tensor(source)

    assert output.device.type == "cpu"
    assert output.shape == source.shape
    assert output.stride() == source.stride()
    assert output.dtype == source.dtype
    torch.testing.assert_close(output, source)


def test_fast_index_copy_accepts_exact_pinned_cpu_source():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fast index copy")

    from freetoken.kernel import copy_to_pinned_tensor, fast_index_copy_jit

    source = copy_to_pinned_tensor(torch.arange(6 * 32, dtype=torch.float32).reshape(6, 32))
    output = torch.empty((3, 32), dtype=torch.float32, device="cuda")
    dst_indices = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    src_indices = torch.tensor([5, 2, 0], dtype=torch.int32, device="cuda")

    fast_index_copy_jit(output, dst_indices, source, src_indices)
    torch.cuda.synchronize()

    torch.testing.assert_close(output.cpu(), source[[5, 2, 0]])


def test_fast_index_copy_skip_env_noops_without_jit(monkeypatch):
    import freetoken.kernel.fast_index_copy as fast_index_copy

    source = torch.arange(6 * 32, dtype=torch.float32).reshape(6, 32)
    output = torch.full((3, 32), -1.0, dtype=torch.float32)
    dst_indices = torch.tensor([0, 1, 2], dtype=torch.int32)
    src_indices = torch.tensor([5, 2, 0], dtype=torch.int32)

    def fail_jit_load(*args, **kwargs):
        raise AssertionError("skip env should bypass fast_index_copy JIT")

    monkeypatch.setenv(fast_index_copy.SKIP_FAST_INDEX_COPY_ENV, "1")
    monkeypatch.setattr(fast_index_copy, "_jit_fast_index_copy_module", fail_jit_load)

    fast_index_copy.fast_index_copy_jit(output, dst_indices, source, src_indices)

    torch.testing.assert_close(output, torch.full_like(output, -1.0))


def test_device_ptr_pinned_bank_resolves():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from freetoken.kernel.pinned import _host_ptr_identity, alloc_pinned_tensor, device_ptr

    t = alloc_pinned_tensor(8, 16, dtype=torch.bfloat16)
    if _host_ptr_identity():
        # Linux/UVA: the GPU dereferences pinned memory at its host VA.
        assert device_ptr(t) == t.data_ptr()
    else:
        # Windows/WDDM: a real alias; may legitimately differ from data_ptr().
        assert device_ptr(t) != 0


def test_device_ptr_cuda_tensor_passthrough():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from freetoken.kernel.pinned import device_ptr

    t = torch.empty(4, device="cuda")
    assert device_ptr(t) == t.data_ptr()


def test_host_device_ptr_is_identity_under_uva():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from freetoken.kernel.pinned import _host_ptr_identity, _load_pinned_extension

    torch.cuda.init()
    if not _host_ptr_identity():
        pytest.skip("non-UVA platform: host_device_ptr rejects unregistered memory instead")
    # cudaHostGetDevicePointer on *unregistered* memory is unspecified: recent arches let
    # UVA degenerate it to identity, while Pascal validates registration and returns
    # cudaErrorInvalidValue. Either is fine -- what must never happen is a different
    # nonzero alias, which would silently misaddress the zero-copy gather. The in-contract
    # case (registered memory -> identity) is test_host_bank_pin_registers_and_translates.
    pageable = torch.empty(64, dtype=torch.uint8)
    ext = _load_pinned_extension()
    try:
        assert ext.host_device_ptr(pageable.data_ptr()) == pageable.data_ptr()
    except RuntimeError as exc:
        assert "cudaHostGetDevicePointer failed" in str(exc)


def test_host_bank_pin_registers_and_translates():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from freetoken.kernel.pinned import _host_ptr_identity, _load_pinned_extension
    from freetoken.moe.host_banks import HostBank

    bank = HostBank((4, 32), torch.bfloat16)
    bank.tensor.fill_(1.0)
    bank.pin()
    bank.pin()  # idempotent
    # registered+mapped memory must have a device alias
    dev = _load_pinned_extension().host_device_ptr(bank.addr)
    if _host_ptr_identity():
        assert dev == bank.addr
    else:
        assert dev != 0
